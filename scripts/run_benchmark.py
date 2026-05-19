#!/usr/bin/env python3
"""Benchmark local LLM inference on Apple Silicon via MLX."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate

from optimizations import (
    WEIGHT_BITS_ORDER,
    check_hf_auth_for_repo,
    collect_gated_repos_in_sweep,
    config_fits_memory,
    estimate_peak_gb,
    get_memory_budget_gb,
    get_model_repos,
    get_system_memory_gb,
    should_skip_preset,
    OptimizationConfig,
    RunParams,
    iter_sweep_configs,
    resolve_run_params,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


@dataclass
class BenchmarkResult:
    timestamp: str
    hardware: str
    model_preset: str
    configuration: str
    weight_bits: int
    optimizations: dict
    model_repo: str | None
    kv_bits: int | None
    prefill_step_size: int
    prompt_tokens: int
    generation_tokens: int
    num_trials: int
    memory_gb: float
    ttft_ms: float
    throughput_tps: float
    prompt_tps: float
    mlx_version: str
    mlx_lm_version: str
    platform: str
    status: str = "ok"
    error: str | None = None

    @property
    def runtime_combo_size(self) -> int:
        return int(self.optimizations.get("kv_cache", False)) + int(
            self.optimizations.get("prefill", False)
        )


def _package_version(module: str) -> str:
    try:
        return subprocess.check_output(
            [sys.executable, "-m", module, "--version"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _detect_hardware(label: str | None) -> str:
    if label:
        return label
    chip = platform.processor() or "Apple Silicon"
    return f"{chip} ({platform.machine()})"


def _run_trial(
    model,
    tokenizer,
    prompt: list[int],
    generation_tokens: int,
    params: RunParams,
) -> tuple[float, float, float, float]:
    tokenizer._eos_token_ids = {}
    mx.reset_peak_memory()

    ttft_ms = 0.0
    prompt_tps = 0.0
    generation_tps = 0.0
    peak_memory_gb = 0.0

    gen_kwargs: dict = {
        "max_tokens": generation_tokens,
        "prefill_step_size": params.prefill_step_size,
    }
    if params.kv_bits is not None:
        gen_kwargs["kv_bits"] = params.kv_bits

    for response in stream_generate(model, tokenizer, prompt, **gen_kwargs):
        if response.generation_tokens == 1:
            ttft_ms = (response.prompt_tokens / response.prompt_tps) * 1000
            prompt_tps = response.prompt_tps
        generation_tps = response.generation_tps
        peak_memory_gb = response.peak_memory

    return ttft_ms, prompt_tps, generation_tps, peak_memory_gb


def _avg(values: list[float]) -> float:
    return sum(values) / len(values)


def benchmark_once(
    *,
    model_preset: str,
    config: OptimizationConfig,
    hardware: str,
    prompt_tokens: int,
    generation_tokens: int,
    num_trials: int,
    delay: int,
    seed: int,
    model_override: str | None = None,
) -> BenchmarkResult:
    params = resolve_run_params(model_preset, config)
    if model_override:
        params = RunParams(
            model_repo=model_override,
            kv_bits=params.kv_bits,
            prefill_step_size=params.prefill_step_size,
            config=config,
        )

    if params.model_repo is None:
        raise ValueError(
            f"No model repo for {model_preset} at {config.weight_bits}-bit. "
            f"Add one in models.json under overrides.{model_preset}.{config.weight_bits}"
        )

    auth_err = check_hf_auth_for_repo(params.model_repo)
    if auth_err:
        raise RuntimeError(auth_err)

    print(f"\n{'=' * 60}")
    print(f"Model: {model_preset}  |  Config: {config.label}")
    print(f"Repo:  {params.model_repo}")
    print(f"Opts:  {params.to_dict()}")
    print(f"{'=' * 60}")

    model, tokenizer = load(
        params.model_repo,
        tokenizer_config={"trust_remote_code": True},
    )

    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer.get_vocab())
    mx.random.seed(seed)
    prompt = mx.random.randint(0, vocab_size, (prompt_tokens,)).tolist()

    print("Warmup ...")
    _run_trial(model, tokenizer, prompt, generation_tokens, params)

    ttft_samples: list[float] = []
    throughput_samples: list[float] = []
    prompt_tps_samples: list[float] = []
    memory_samples: list[float] = []

    for trial in range(1, num_trials + 1):
        if delay > 0:
            time.sleep(delay)
        ttft_ms, prompt_tps, generation_tps, peak_memory_gb = _run_trial(
            model, tokenizer, prompt, generation_tokens, params
        )
        ttft_samples.append(ttft_ms)
        throughput_samples.append(generation_tps)
        prompt_tps_samples.append(prompt_tps)
        memory_samples.append(peak_memory_gb)
        print(
            f"  trial {trial}/{num_trials}: "
            f"ttft={ttft_ms:.1f} ms, throughput={generation_tps:.1f} t/s, "
            f"memory={peak_memory_gb:.2f} GB"
        )

    del model

    return BenchmarkResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        hardware=hardware,
        model_preset=model_preset,
        configuration=config.label,
        weight_bits=config.weight_bits,
        optimizations=config.to_dict(),
        model_repo=params.model_repo,
        kv_bits=params.kv_bits,
        prefill_step_size=params.prefill_step_size,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        num_trials=num_trials,
        memory_gb=_avg(memory_samples),
        ttft_ms=_avg(ttft_samples),
        throughput_tps=_avg(throughput_samples),
        prompt_tps=_avg(prompt_tps_samples),
        mlx_version=_package_version("mlx"),
        mlx_lm_version=_package_version("mlx_lm"),
        platform=platform.platform(),
    )


def _failed_result(
    *,
    model_preset: str,
    config: OptimizationConfig,
    hardware: str,
    error: str,
    prompt_tokens: int,
    generation_tokens: int,
    num_trials: int,
) -> BenchmarkResult:
    params = resolve_run_params(model_preset, config)
    return BenchmarkResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        hardware=hardware,
        model_preset=model_preset,
        configuration=config.label,
        weight_bits=config.weight_bits,
        optimizations=config.to_dict(),
        model_repo=params.model_repo,
        kv_bits=params.kv_bits,
        prefill_step_size=params.prefill_step_size,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        num_trials=num_trials,
        memory_gb=0.0,
        ttft_ms=0.0,
        throughput_tps=0.0,
        prompt_tps=0.0,
        mlx_version=_package_version("mlx"),
        mlx_lm_version=_package_version("mlx_lm"),
        platform=platform.platform(),
        status="error",
        error=error,
    )


def _save_result(result: BenchmarkResult, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(result), indent=2) + "\n")
    return output


def _result_path(hardware: str, model_preset: str, config_label: str) -> Path:
    safe_hw = hardware.replace(" ", "_").replace("/", "-")
    return RESULTS_DIR / safe_hw / model_preset / f"{config_label}.json"


def run_single(args: argparse.Namespace) -> list[BenchmarkResult]:
    if args.weight_bits is not None:
        config = OptimizationConfig(
            weight_bits=args.weight_bits,
            kv_cache=args.kv_cache,
            prefill=args.prefill,
        )
    elif args.config:
        config = OptimizationConfig.from_label(args.config)
    else:
        config = OptimizationConfig(weight_bits=16)

    hardware = _detect_hardware(args.hardware)
    try:
        result = benchmark_once(
            model_preset=args.preset,
            config=config,
            hardware=hardware,
            prompt_tokens=args.prompt_tokens,
            generation_tokens=args.generation_tokens,
            num_trials=args.num_trials,
            delay=args.delay,
            seed=args.seed,
            model_override=args.model,
        )
    except Exception as exc:
        result = _failed_result(
            model_preset=args.preset,
            config=config,
            hardware=hardware,
            error=str(exc),
            prompt_tokens=args.prompt_tokens,
            generation_tokens=args.generation_tokens,
            num_trials=args.num_trials,
        )
        print(f"FAILED: {exc}")

    out = args.output or _result_path(hardware, args.preset, config.label)
    _save_result(result, out)
    print(f"\nSaved: {out}")
    return [result]


def _run_single_subprocess(
    args: argparse.Namespace,
    *,
    model_preset: str,
    config: OptimizationConfig,
    hardware: str,
    output: Path,
) -> int:
    """Run one benchmark in a child process so OOM cannot kill the sweep."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_benchmark.py"),
        "--preset",
        model_preset,
        "--config",
        config.label,
        "--hardware",
        hardware,
        "-n",
        str(args.num_trials),
        "-p",
        str(args.prompt_tokens),
        "-g",
        str(args.generation_tokens),
        "-o",
        str(output),
    ]
    if args.delay:
        cmd.extend(["--delay", str(args.delay)])
    return subprocess.run(cmd, cwd=ROOT).returncode


def _load_result_from_file(path: Path) -> BenchmarkResult | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return BenchmarkResult(**{k: v for k, v in data.items() if k in BenchmarkResult.__dataclass_fields__})


def run_sweep(args: argparse.Namespace) -> list[BenchmarkResult]:
    hardware = _detect_hardware(args.hardware)
    model_repos = get_model_repos()
    models = sorted(model_repos) if args.all_models else [args.preset]
    if args.skip_presets:
        skip = set(args.skip_presets.split(","))
        models = [m for m in models if m not in skip]

    system_ram = get_system_memory_gb()
    memory_budget = get_memory_budget_gb(args.memory_fraction)

    if not args.include_qwen:
        models = [m for m in models if m != "qwen-35b"]

    configs = list(
        iter_sweep_configs(
            weights_only=args.weights_only,
            max_runtime_combo=args.max_combo_size,
        )
    )

    gated = collect_gated_repos_in_sweep(models, configs)
    for repo in gated:
        auth_err = check_hf_auth_for_repo(repo)
        if auth_err:
            print(f"\nERROR: {auth_err}\n")
            raise SystemExit(1)

    print(f"System RAM: ~{system_ram:.0f} GB | memory budget: ~{memory_budget:.1f} GB")
    if system_ram < 36 and not args.include_qwen:
        print("Note: qwen-35b skipped on ≤32GB machines (use --include-qwen to force).")

    total = len(models) * len(configs)
    print(f"Sweep: {len(models)} model(s) × {len(configs)} config(s) = {total} runs")
    print("Order: fp16 → w8 → w4 → w2; per weight: runtime off → kv → prefill → both")
    for i, c in enumerate(configs, 1):
        print(f"  {i}. {c.label}")

    results: list[BenchmarkResult] = []
    run_idx = 0

    for model_preset in models:
        preset_skip = should_skip_preset(model_preset, system_ram)
        if preset_skip:
            print(f"\n--- Skipping all {model_preset} runs: {preset_skip}")
            for config in configs:
                run_idx += 1
                result = _failed_result(
                    model_preset=model_preset,
                    config=config,
                    hardware=hardware,
                    error=preset_skip,
                    prompt_tokens=args.prompt_tokens,
                    generation_tokens=args.generation_tokens,
                    num_trials=args.num_trials,
                )
                result.status = "skipped"
                _save_result(result, _result_path(hardware, model_preset, config.label))
                results.append(result)
            continue

        for config in configs:
            run_idx += 1
            print(f"\n>>> Run {run_idx}/{total}")
            params = resolve_run_params(model_preset, config)
            if params.model_repo is None:
                print(f"SKIP: no repo for {model_preset} {config.weight_label}")
                result = _failed_result(
                    model_preset=model_preset,
                    config=config,
                    hardware=hardware,
                    error=f"No repo configured for {config.weight_bits}-bit",
                    prompt_tokens=args.prompt_tokens,
                    generation_tokens=args.generation_tokens,
                    num_trials=args.num_trials,
                )
                result.status = "skipped"
                out = _result_path(hardware, model_preset, config.label)
                _save_result(result, out)
                results.append(result)
                continue

            if not config_fits_memory(model_preset, config, memory_budget):
                est = estimate_peak_gb(model_preset, config)
                msg = (
                    f"Estimated peak {est:.1f} GB exceeds budget {memory_budget:.1f} GB"
                )
                print(f"SKIP (memory): {msg}")
                result = _failed_result(
                    model_preset=model_preset,
                    config=config,
                    hardware=hardware,
                    error=msg,
                    prompt_tokens=args.prompt_tokens,
                    generation_tokens=args.generation_tokens,
                    num_trials=args.num_trials,
                )
                result.status = "skipped"
                out = _result_path(hardware, model_preset, config.label)
                _save_result(result, out)
                results.append(result)
                continue

            out = _result_path(hardware, model_preset, config.label)
            exit_code = _run_single_subprocess(
                args,
                model_preset=model_preset,
                config=config,
                hardware=hardware,
                output=out,
            )
            result = _load_result_from_file(out)
            if result is None:
                err = "subprocess failed"
                if exit_code in (-6, 134, 137):
                    err = "Out of memory (Metal OOM)"
                result = _failed_result(
                    model_preset=model_preset,
                    config=config,
                    hardware=hardware,
                    error=err,
                    prompt_tokens=args.prompt_tokens,
                    generation_tokens=args.generation_tokens,
                    num_trials=args.num_trials,
                )
                if exit_code in (-6, 134, 137):
                    result.status = "oom"
                _save_result(result, out)
            elif exit_code != 0 and result.status == "ok":
                result.status = "error"
                result.error = f"exit code {exit_code}"

            results.append(result)

            if args.delay_between_configs > 0:
                time.sleep(args.delay_between_configs)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_hw = hardware.replace(" ", "_").replace("/", "-")
    summary_path = RESULTS_DIR / f"sweep_{safe_hw}_{stamp}.json"
    summary = {
        "hardware": hardware,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "config_order": [c.label for c in configs],
        "results": [asdict(r) for r in results],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{'=' * 60}")
    print("SWEEP COMPLETE")
    print(f"Summary: {summary_path}")
    _print_summary_table(results)
    return results


def _print_summary_table(results: list[BenchmarkResult]) -> None:
    print(f"\n{'Model':<14} {'Config':<28} {'Memory':>8} {'TTFT':>8} {'t/s':>8} {'Status'}")
    print("-" * 80)
    for r in results:
        if r.status == "ok":
            print(
                f"{r.model_preset:<14} {r.configuration:<28} "
                f"{r.memory_gb:>7.2f}G {r.ttft_ms:>7.0f}ms {r.throughput_tps:>7.1f}  ok"
            )
        else:
            status = {"skipped": "SKIP", "oom": "OOM"}.get(r.status, "FAIL")
            print(f"{r.model_preset:<14} {r.configuration:<28} {'—':>8} {'—':>8} {'—':>8}  {status}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark MLX LLM inference with optimization sweeps."
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Full matrix: fp16/w8/w4/w2 × runtime combos (kv_cache, prefill).",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="With --sweep, only fp16 + w8 + w4 + w2 (no kv_cache/prefill).",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="With --sweep, run every model preset (llama3-8b, mistral-7b, qwen-35b).",
    )
    parser.add_argument(
        "--include-qwen",
        action="store_true",
        help="Include qwen-35b even on low-memory Macs (may OOM on 24GB).",
    )
    parser.add_argument(
        "--skip-presets",
        help="Comma-separated presets to skip, e.g. qwen-35b",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.75,
        help="Fraction of RAM used as MLX memory budget (default 0.75).",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(get_model_repos()),
        default="llama3-8b",
        help="Model preset (single run, or one model in sweep without --all-models).",
    )
    parser.add_argument(
        "--config",
        help="Single run label, e.g. 'fp16', 'w4', 'w8+kv_cache', 'w4+kv_cache+prefill'.",
    )
    parser.add_argument(
        "--weight-bits",
        type=int,
        choices=WEIGHT_BITS_ORDER,
        help="Single run: 16=fp16, 8, 4, or 2.",
    )
    parser.add_argument("--kv-cache", action="store_true", help="Enable KV cache quant.")
    parser.add_argument("--prefill", action="store_true", help="Enable large prefill chunks.")
    parser.add_argument(
        "--max-combo-size",
        type=int,
        choices=[0, 1, 2],
        help="Limit runtime combo depth per weight (0=weight only, 1=+one, 2=all).",
    )
    parser.add_argument("--model", help="Override Hugging Face repo for single run.")
    parser.add_argument("--hardware", help="Machine label (e.g. 'Mac M3').")
    parser.add_argument("-p", "--prompt-tokens", type=int, default=512)
    parser.add_argument("-g", "--generation-tokens", type=int, default=128)
    parser.add_argument("-n", "--num-trials", type=int, default=3)
    parser.add_argument("--delay", type=int, default=0, help="Seconds between trials.")
    parser.add_argument(
        "--delay-between-configs",
        type=int,
        default=5,
        help="Seconds between sweep configs (cool-down). Default: 5.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--output", type=Path, help="Output JSON (single run only).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sweep:
        run_sweep(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
