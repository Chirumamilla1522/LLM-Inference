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

from benchmark_schema import (
    SCHEMA_VERSION,
    WARMUP_POLICY,
    build_stats_block,
    primary_metric,
)
from workloads import WorkloadProfile, get_workload, iter_workloads, build_prompt_ids
from optimizations import (
    DEFAULT_NUM_DRAFT_TOKENS,
    LARGE_MODEL_PRESETS,
    MODEL_PARAM_LABELS,
    WEIGHT_BITS_ORDER,
    check_hf_auth_for_repo,
    collect_repos_in_sweep,
    config_fits_memory,
    estimate_peak_gb,
    get_memory_budget_gb,
    get_model_repos,
    get_system_memory_gb,
    hf_token_present,
    resolve_draft_preset,
    resolve_draft_repo,
    should_skip_preset,
    sort_presets,
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
    article_id: int | None = None
    run_label: str | None = None
    benchmark_mode: str = "standard"
    draft_model_repo: str | None = None
    draft_preset: str | None = None
    num_draft_tokens: int | None = None
    draft_accept_rate: float | None = None
    prefix_cache_cold_ttft_ms: float | None = None
    prefix_cache_warm_ttft_ms: float | None = None
    prefix_system_tokens: int | None = None
    workload: dict | None = None
    schema_version: int = SCHEMA_VERSION
    trials: dict | None = None
    stats: dict | None = None
    warmup_policy: dict | None = None

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
    *,
    draft_model=None,
    num_draft_tokens: int = DEFAULT_NUM_DRAFT_TOKENS,
    prompt_cache=None,
) -> tuple[float, float, float, float, float | None]:
    tokenizer._eos_token_ids = {}
    mx.reset_peak_memory()

    ttft_ms = 0.0
    prompt_tps = 0.0
    generation_tps = 0.0
    peak_memory_gb = 0.0
    draft_accept_rate: float | None = None

    gen_kwargs: dict = {
        "max_tokens": generation_tokens,
        "prefill_step_size": params.prefill_step_size,
    }
    if params.kv_bits is not None:
        gen_kwargs["kv_bits"] = params.kv_bits
    if prompt_cache is not None:
        gen_kwargs["prompt_cache"] = prompt_cache
    if draft_model is not None:
        gen_kwargs["draft_model"] = draft_model
        gen_kwargs["num_draft_tokens"] = num_draft_tokens

    gen_count = 0
    from_draft_count = 0

    for response in stream_generate(model, tokenizer, prompt, **gen_kwargs):
        if response.generation_tokens == 1:
            ttft_ms = (response.prompt_tokens / response.prompt_tps) * 1000
            prompt_tps = response.prompt_tps
        generation_tps = response.generation_tps
        peak_memory_gb = response.peak_memory
        if draft_model is not None:
            gen_count += 1
            if response.from_draft:
                from_draft_count += 1

    if draft_model is not None and gen_count > 0:
        draft_accept_rate = from_draft_count / gen_count

    return ttft_ms, prompt_tps, generation_tps, peak_memory_gb, draft_accept_rate


def _avg(values: list[float]) -> float:
    return sum(values) / len(values)


def benchmark_prefix_cache(
    *,
    model_preset: str,
    config: OptimizationConfig,
    hardware: str,
    prompt_tokens: int,
    generation_tokens: int,
    num_trials: int,
    delay: int,
    seed: int,
    article_id: int | None = None,
    run_label: str | None = None,
) -> BenchmarkResult:
    """Compare full-prompt TTFT vs TTFT after reusing a saved prefix KV cache."""
    import tempfile

    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache

    params = resolve_run_params(model_preset, config)
    if params.model_repo is None:
        raise ValueError(f"No model repo for {model_preset} at {config.weight_bits}-bit")

    auth_err = check_hf_auth_for_repo(params.model_repo)
    if auth_err:
        raise RuntimeError(auth_err)

    system_tokens = max(64, prompt_tokens // 2)
    user_tokens = max(32, prompt_tokens - system_tokens)

    print(f"\n{'=' * 60}")
    print(f"PREFIX CACHE  |  {model_preset}  |  {config.label}")
    print(f"system={system_tokens} user={user_tokens} tokens")
    print(f"{'=' * 60}")

    model, tokenizer = load(
        params.model_repo,
        tokenizer_config={"trust_remote_code": True},
    )
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer.get_vocab())
    mx.random.seed(seed)

    cold_ttft: list[float] = []
    warm_ttft: list[float] = []
    memory_samples: list[float] = []
    throughput_samples: list[float] = []

    gen_kwargs = {
        "prefill_step_size": params.prefill_step_size,
        "max_tokens": generation_tokens,
    }
    if params.kv_bits is not None:
        gen_kwargs["kv_bits"] = params.kv_bits

    for trial in range(1, num_trials + 1):
        if delay > 0:
            time.sleep(delay)

        mx.random.seed(seed + trial)
        system_ids = mx.random.randint(0, vocab_size, (system_tokens,)).tolist()
        user_ids = mx.random.randint(0, vocab_size, (user_tokens,)).tolist()
        full_ids = system_ids + user_ids

        mx.reset_peak_memory()
        cold_ms = 0.0
        for response in stream_generate(model, tokenizer, full_ids, **gen_kwargs):
            if response.generation_tokens == 1:
                cold_ms = (response.prompt_tokens / response.prompt_tps) * 1000
                break
        cold_ttft.append(cold_ms)

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            cache_path = tmp.name

        cache = make_prompt_cache(model)
        system_mx = mx.array(system_ids)
        for _ in generate_step(
            system_mx,
            model,
            prompt_cache=cache,
            max_tokens=1,
            prefill_step_size=params.prefill_step_size,
            kv_bits=params.kv_bits,
        ):
            pass
        save_prompt_cache(
            cache_path,
            cache,
            metadata={
                "model": params.model_repo,
                "tokenizer_config": json.dumps({"trust_remote_code": True}),
            },
        )

        loaded_cache, _meta = load_prompt_cache(cache_path, return_metadata=True)
        user_mx = mx.array(user_ids)
        mx.reset_peak_memory()
        warm_ms = 0.0
        gen_tps = 0.0
        peak_gb = 0.0
        for response in stream_generate(
            model,
            tokenizer,
            user_mx,
            prompt_cache=loaded_cache,
            **gen_kwargs,
        ):
            if response.generation_tokens == 1:
                warm_ms = (response.prompt_tokens / response.prompt_tps) * 1000
            gen_tps = response.generation_tps
            peak_gb = response.peak_memory

        warm_ttft.append(warm_ms)
        memory_samples.append(peak_gb)
        throughput_samples.append(gen_tps)
        Path(cache_path).unlink(missing_ok=True)

        print(
            f"  trial {trial}/{num_trials}: cold_ttft={cold_ttft[-1]:.1f} ms, "
            f"warm_ttft={warm_ms:.1f} ms"
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
        ttft_ms=_avg(warm_ttft),
        throughput_tps=_avg(throughput_samples),
        prompt_tps=0.0,
        mlx_version=_package_version("mlx"),
        mlx_lm_version=_package_version("mlx_lm"),
        platform=platform.platform(),
        benchmark_mode="prefix_cache",
        article_id=article_id,
        run_label=run_label,
        prefix_cache_cold_ttft_ms=_avg(cold_ttft),
        prefix_cache_warm_ttft_ms=_avg(warm_ttft),
        prefix_system_tokens=system_tokens,
    )


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
    draft_preset: str | None = None,
    num_draft_tokens: int = DEFAULT_NUM_DRAFT_TOKENS,
    article_id: int | None = None,
    run_label: str | None = None,
    benchmark_mode: str = "standard",
    workload: WorkloadProfile | None = None,
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

    draft_model = None
    draft_repo: str | None = None
    effective_draft_preset = draft_preset or resolve_draft_preset(model_preset)
    if benchmark_mode == "speculative":
        draft_repo = resolve_draft_repo(
            model_preset, config.weight_bits
        )
        if draft_repo is None and effective_draft_preset:
            draft_repo = get_model_repos().get(effective_draft_preset, {}).get(
                config.weight_bits
            )
        if draft_repo is None:
            raise ValueError(
                f"No draft repo for {model_preset} at {config.weight_bits}-bit. "
                f"Set DRAFT_PRESET_BY_TARGET in optimizations.py."
            )
        auth_err = check_hf_auth_for_repo(draft_repo)
        if auth_err:
            raise RuntimeError(auth_err)
        print(f"Draft: {draft_repo} (num_draft_tokens={num_draft_tokens})")

    model, tokenizer = load(
        params.model_repo,
        tokenizer_config={"trust_remote_code": True},
    )
    if draft_repo:
        draft_model, draft_tokenizer = load(
            draft_repo,
            tokenizer_config={"trust_remote_code": True},
        )
        d_vocab = getattr(draft_tokenizer, "vocab_size", None) or len(
            draft_tokenizer.get_vocab()
        )
        t_vocab = getattr(tokenizer, "vocab_size", None) or len(tokenizer.get_vocab())
        if d_vocab != t_vocab:
            raise ValueError(
                f"Draft vocab ({d_vocab}) != target vocab ({t_vocab}). "
                "Pick a draft preset with a matching tokenizer."
            )

    if workload is not None and not workload.runnable:
        raise ValueError(
            f"Workload '{workload.id}' ({workload.modality.value}) is not runnable "
            "in the text mlx-lm harness yet."
        )

    effective_prompt_tokens = (
        workload.prompt_tokens if workload is not None else prompt_tokens
    )
    effective_generation_tokens = (
        workload.generation_tokens if workload is not None else generation_tokens
    )

    if workload is not None:
        print(
            f"Workload: {workload.id} (pressure={workload.pressure}, "
            f"task={workload.task.value}, data={workload.data_type.value}, "
            f"stress={workload.primary_stress.value})"
        )

    print("Warmup ...")
    warmup_prompt = build_prompt_ids(
        tokenizer,
        workload or get_workload("random_baseline"),
        seed=seed,
        prompt_tokens=effective_prompt_tokens,
    )
    _run_trial(
        model,
        tokenizer,
        warmup_prompt,
        effective_generation_tokens,
        params,
        draft_model=draft_model,
        num_draft_tokens=num_draft_tokens,
    )

    ttft_samples: list[float] = []
    throughput_samples: list[float] = []
    prompt_tps_samples: list[float] = []
    memory_samples: list[float] = []
    accept_samples: list[float] = []

    for trial in range(1, num_trials + 1):
        if delay > 0:
            time.sleep(delay)
        trial_prompt = build_prompt_ids(
            tokenizer,
            workload or get_workload("random_baseline"),
            seed=seed + trial,
            prompt_tokens=effective_prompt_tokens,
        )
        ttft_ms, prompt_tps, generation_tps, peak_memory_gb, accept = _run_trial(
            model,
            tokenizer,
            trial_prompt,
            effective_generation_tokens,
            params,
            draft_model=draft_model,
            num_draft_tokens=num_draft_tokens,
        )
        ttft_samples.append(ttft_ms)
        throughput_samples.append(generation_tps)
        prompt_tps_samples.append(prompt_tps)
        memory_samples.append(peak_memory_gb)
        if accept is not None:
            accept_samples.append(accept)
        extra = ""
        if accept is not None:
            extra = f", draft_accept={accept:.2%}"
        print(
            f"  trial {trial}/{num_trials}: "
            f"ttft={ttft_ms:.1f} ms, throughput={generation_tps:.1f} t/s, "
            f"memory={peak_memory_gb:.2f} GB{extra}"
        )

    del model
    if draft_model is not None:
        del draft_model

    config_label = config.label
    if benchmark_mode == "speculative":
        config_label = f"{config.label}+speculative"

    trials_block = {
        "ttft_ms": ttft_samples,
        "throughput_tps": throughput_samples,
        "memory_gb": memory_samples,
        "prompt_tps": prompt_tps_samples,
    }
    if accept_samples:
        trials_block["draft_accept_rate"] = accept_samples

    stats_block = build_stats_block(
        ttft_ms=ttft_samples,
        throughput_tps=throughput_samples,
        memory_gb=memory_samples,
        prompt_tps=prompt_tps_samples,
        draft_accept_rate=accept_samples or None,
    )
    mem_stats = stats_block.get("memory_gb", {})
    memory_primary = float(mem_stats.get("max", mem_stats.get("median", 0.0)))

    return BenchmarkResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        hardware=hardware,
        model_preset=model_preset,
        configuration=config_label,
        weight_bits=config.weight_bits,
        optimizations=config.to_dict(),
        model_repo=params.model_repo,
        kv_bits=params.kv_bits,
        prefill_step_size=params.prefill_step_size,
        prompt_tokens=effective_prompt_tokens,
        generation_tokens=effective_generation_tokens,
        num_trials=num_trials,
        memory_gb=memory_primary,
        ttft_ms=primary_metric(stats_block.get("ttft_ms")),
        throughput_tps=primary_metric(stats_block.get("throughput_tps")),
        prompt_tps=primary_metric(stats_block.get("prompt_tps")),
        mlx_version=_package_version("mlx"),
        mlx_lm_version=_package_version("mlx_lm"),
        platform=platform.platform(),
        article_id=article_id,
        run_label=run_label,
        benchmark_mode=benchmark_mode,
        draft_model_repo=draft_repo,
        draft_preset=effective_draft_preset,
        num_draft_tokens=num_draft_tokens if draft_repo else None,
        draft_accept_rate=primary_metric(stats_block.get("draft_accept_rate"))
        if accept_samples
        else None,
        workload=workload.to_metadata() if workload else None,
        schema_version=SCHEMA_VERSION,
        trials=trials_block,
        stats=stats_block,
        warmup_policy=WARMUP_POLICY,
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
    workload: WorkloadProfile | None = None,
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
        workload=workload.to_metadata() if workload else None,
        schema_version=SCHEMA_VERSION,
        warmup_policy=WARMUP_POLICY,
    )


def _save_result(result: BenchmarkResult, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(result), indent=2) + "\n")
    return output


def _result_path(
    hardware: str,
    model_preset: str,
    config_label: str,
    *,
    output_root: Path | None = None,
    run_label: str | None = None,
) -> Path:
    safe_hw = hardware.replace(" ", "_").replace("/", "-")
    base = output_root if output_root is not None else RESULTS_DIR / safe_hw
    if output_root is not None:
        filename = f"{run_label}.json" if run_label else f"{config_label}.json"
        return base / model_preset / filename
    return base / model_preset / f"{config_label}.json"


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
    output_root = Path(args.output_root) if args.output_root else None
    workload = get_workload(args.workload) if args.workload else None

    if args.dry_run:
        p_tok = workload.prompt_tokens if workload else args.prompt_tokens
        g_tok = workload.generation_tokens if workload else args.generation_tokens
        print("DRY-RUN — would benchmark:")
        print(f"  preset={args.preset} config={config.label} hardware={hardware}")
        print(f"  prompt_tokens={p_tok} generation_tokens={g_tok} trials={args.num_trials}")
        if workload:
            print(f"  workload={workload.id} pressure={workload.pressure}")
        print(f"  schema_version={SCHEMA_VERSION} warmup={WARMUP_POLICY['warmup_trials']} trial")
        return []

    try:
        if args.prefix_cache:
            result = benchmark_prefix_cache(
                model_preset=args.preset,
                config=config,
                hardware=hardware,
                prompt_tokens=args.prompt_tokens,
                generation_tokens=args.generation_tokens,
                num_trials=args.num_trials,
                delay=args.delay,
                seed=args.seed,
                article_id=args.article_id,
                run_label=args.run_label,
            )
        else:
            mode = "speculative" if args.speculative else "standard"
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
                draft_preset=args.draft_preset,
                num_draft_tokens=args.num_draft_tokens,
                article_id=args.article_id,
                run_label=args.run_label,
                benchmark_mode=mode,
                workload=workload,
            )
    except Exception as exc:
        p_tok = workload.prompt_tokens if workload else args.prompt_tokens
        g_tok = workload.generation_tokens if workload else args.generation_tokens
        result = _failed_result(
            model_preset=args.preset,
            config=config,
            hardware=hardware,
            error=str(exc),
            prompt_tokens=p_tok,
            generation_tokens=g_tok,
            num_trials=args.num_trials,
            workload=workload,
        )
        print(f"FAILED: {exc}")
        traceback.print_exc()

    label = args.run_label or (args.workload if args.workload else config.label)
    if args.speculative and not args.prefix_cache:
        label = f"{config.label}+speculative"
    out = args.output or _result_path(
        hardware,
        args.preset,
        config.label,
        output_root=output_root,
        run_label=label if output_root else None,
    )
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
    run_label: str | None = None,
    speculative: bool = False,
    prefix_cache: bool = False,
    prompt_tokens: int | None = None,
    generation_tokens: int | None = None,
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
        str(prompt_tokens if prompt_tokens is not None else args.prompt_tokens),
        "-g",
        str(
            generation_tokens
            if generation_tokens is not None
            else args.generation_tokens
        ),
        "-o",
        str(output),
    ]
    if args.delay:
        cmd.extend(["--delay", str(args.delay)])
    if args.output_root:
        cmd.extend(["--output-root", str(args.output_root)])
    if args.article_id is not None:
        cmd.extend(["--article-id", str(args.article_id)])
    if run_label:
        cmd.extend(["--run-label", run_label])
    if speculative:
        cmd.append("--speculative")
    if prefix_cache:
        cmd.append("--prefix-cache")
    if getattr(args, "workload", None):
        cmd.extend(["--workload", args.workload])
    return subprocess.run(cmd, cwd=ROOT).returncode


def _load_result_from_file(path: Path) -> BenchmarkResult | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return BenchmarkResult(**{k: v for k, v in data.items() if k in BenchmarkResult.__dataclass_fields__})


def run_sweep(args: argparse.Namespace) -> list[BenchmarkResult]:
    hardware = _detect_hardware(args.hardware)
    safe_hw = hardware.replace(" ", "_").replace("/", "-")
    sweep_output_root = (
        Path(args.output_root)
        if args.output_root
        else RESULTS_DIR / safe_hw
    )
    model_repos = get_model_repos()
    include_large = args.include_large or args.include_qwen
    models = sort_presets(list(model_repos)) if args.all_models else [args.preset]
    if args.skip_presets:
        skip = set(args.skip_presets.split(","))
        models = [m for m in models if m not in skip]

    system_ram = get_system_memory_gb()
    memory_budget = get_memory_budget_gb(args.memory_fraction)

    if not include_large:
        models = [m for m in models if m not in LARGE_MODEL_PRESETS]

    configs = list(
        iter_sweep_configs(
            weights_only=args.weights_only,
            max_runtime_combo=args.max_combo_size,
        )
    )

    sweep_repos = collect_repos_in_sweep(models, configs)
    for repo in sweep_repos:
        auth_err = check_hf_auth_for_repo(repo)
        if auth_err:
            print(f"\nERROR: {auth_err}\n")
            raise SystemExit(1)

    print(f"System RAM: ~{system_ram:.0f} GB | memory budget: ~{memory_budget:.1f} GB")
    if not include_large and LARGE_MODEL_PRESETS:
        skipped = ", ".join(sorted(LARGE_MODEL_PRESETS))
        print(f"Note: large presets skipped ({skipped}). Use --include-large to force.")

    total = len(models) * len(configs)
    print(f"Sweep: {len(models)} model(s) × {len(configs)} config(s) = {total} runs")
    for m in models:
        print(f"  - {m} ({MODEL_PARAM_LABELS.get(m, '?')})")
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
                _save_result(
                    result,
                    _result_path(
                        hardware,
                        model_preset,
                        config.label,
                        output_root=sweep_output_root
                        if args.output_root
                        else None,
                    ),
                )
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
                out = _result_path(
                    hardware,
                    model_preset,
                    config.label,
                    output_root=sweep_output_root if args.output_root else None,
                )
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
                out = _result_path(
                    hardware,
                    model_preset,
                    config.label,
                    output_root=sweep_output_root if args.output_root else None,
                )
                _save_result(result, out)
                results.append(result)
                continue

            out = _result_path(
                hardware,
                model_preset,
                config.label,
                output_root=sweep_output_root if args.output_root else None,
            )
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
    summary_path = sweep_output_root.parent / f"sweep_{safe_hw}_{stamp}.json"
    if args.output_root:
        summary_path = sweep_output_root / f"sweep_summary_{stamp}.json"
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
        help="With --sweep, all presets from qwen-0.5b through qwen-35b (smallest first).",
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Include 12B+ presets (nemo-12b, qwen-14b, qwen-35b) on low-RAM Macs.",
    )
    parser.add_argument(
        "--include-qwen",
        action="store_true",
        help="Alias for --include-large (backwards compatible).",
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
        choices=sort_presets(list(get_model_repos())),
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
    parser.add_argument(
        "--hf-check",
        action="store_true",
        help="Verify Hugging Face token and gated model access, then exit.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Base directory for results (e.g. results/Mac_M3/article_01_weight-quant).",
    )
    parser.add_argument(
        "--article-id",
        type=int,
        help="Article number (0-11) for metadata in JSON.",
    )
    parser.add_argument(
        "--run-label",
        help="Output filename stem when using --output-root.",
    )
    parser.add_argument(
        "--speculative",
        action="store_true",
        help="Enable speculative decoding with draft model from optimizations.py.",
    )
    parser.add_argument(
        "--draft-preset",
        help="Override draft preset for speculative decoding.",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=DEFAULT_NUM_DRAFT_TOKENS,
        help="Draft tokens per speculative step (default: 3).",
    )
    parser.add_argument(
        "--prefix-cache",
        action="store_true",
        help="Benchmark cold vs warm TTFT with saved prefix KV cache.",
    )
    parser.add_argument(
        "--workload",
        metavar="ID",
        help="Stress profile id (task × data × pressure). See: python scripts/workloads.py",
    )
    parser.add_argument(
        "--workload-sweep",
        action="store_true",
        help="Run all runnable workload profiles for --preset and --config (or w4 default).",
    )
    parser.add_argument(
        "--list-workloads",
        action="store_true",
        help="Print workload matrix and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned single run and exit (no MLX).",
    )
    return parser


def run_hf_check() -> None:
    print(f"Token present: {hf_token_present()}")
    if not hf_token_present():
        print("\nNot logged in. Run: ./scripts/hf_login.sh")
        raise SystemExit(1)
    try:
        from huggingface_hub import HfApi

        who = HfApi().whoami()
        print(f"Logged in as: {who.get('name', who)}")
    except Exception as exc:
        print(f"whoami failed: {exc}")
        raise SystemExit(1) from exc

    repos = sorted(
        {r for bits in get_model_repos().values() for r in bits.values() if r}
    )
    print(f"\nModel repo access ({len(repos)} repos):")
    ok = True
    for repo in repos:
        err = check_hf_auth_for_repo(repo)
        if err:
            print(f"  FAIL  {repo}")
            print(f"        {err}")
            ok = False
        else:
            print(f"  OK    {repo}")
    raise SystemExit(0 if ok else 1)


def run_workload_sweep(args: argparse.Namespace) -> list[BenchmarkResult]:
    """Run low→high pressure workloads for one model + config."""
    if args.weight_bits is not None:
        config = OptimizationConfig(
            weight_bits=args.weight_bits,
            kv_cache=args.kv_cache,
            prefill=args.prefill,
        )
    elif args.config:
        config = OptimizationConfig.from_label(args.config)
    else:
        config = OptimizationConfig.from_label("w4+kv_cache+prefill")

    hardware = _detect_hardware(args.hardware)
    safe_hw = hardware.replace(" ", "_").replace("/", "-")
    sweep_root = (
        Path(args.output_root)
        if args.output_root
        else RESULTS_DIR / safe_hw / args.preset / "workloads"
    )

    profiles = list(iter_workloads(sweep=True))
    print(
        f"Workload sweep: {args.preset} @ {config.label} — "
        f"{len(profiles)} profiles (pressure 1→5)"
    )
    results: list[BenchmarkResult] = []
    for i, profile in enumerate(profiles, 1):
        print(f"\n>>> Workload {i}/{len(profiles)}: {profile.id}")
        out = sweep_root / f"{profile.id}.json"
        args.workload = profile.id
        exit_code = _run_single_subprocess(
            args,
            model_preset=args.preset,
            config=config,
            hardware=hardware,
            output=out,
            run_label=profile.id,
        )
        result = _load_result_from_file(out)
        if result is None:
            result = _failed_result(
                model_preset=args.preset,
                config=config,
                hardware=hardware,
                error="subprocess failed" if exit_code else "no output",
                prompt_tokens=profile.prompt_tokens,
                generation_tokens=profile.generation_tokens,
                num_trials=args.num_trials,
                workload=profile,
            )
        results.append(result)
        if args.delay_between_configs > 0 and i < len(profiles):
            time.sleep(args.delay_between_configs)

    summary_path = sweep_root / "workload_sweep_summary.json"
    sweep_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "hardware": hardware,
                "model_preset": args.preset,
                "configuration": config.label,
                "workloads": [p.id for p in profiles],
                "results": [asdict(r) for r in results],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nSaved summary: {summary_path}")
    return results


def main() -> None:
    args = build_parser().parse_args()
    if args.list_workloads:
        from workloads import print_workload_table

        print_workload_table()
        return
    if args.hf_check:
        run_hf_check()
    elif args.workload_sweep:
        run_workload_sweep(args)
    elif args.sweep:
        run_sweep(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
