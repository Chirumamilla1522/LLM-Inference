"""Optimization matrix sweeps with subprocess isolation and checkpointing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from benchmark.paths import RESULTS_DIR, ROOT, RUN_BENCHMARK_SCRIPT
from benchmark.result import (
    BenchmarkResult,
    detect_hardware,
    failed_result,
    load_result_from_file,
    result_path,
    save_result,
)
from benchmark.sweep_state import (
    init_state,
    load_state,
    mark_completed,
    save_state,
    should_skip,
    state_path,
    summary_counts,
)
from optimizations import (
    LARGE_MODEL_PRESETS,
    OptimizationConfig,
    check_hf_auth_for_repo,
    collect_repos_in_sweep,
    config_fits_memory,
    estimate_peak_gb,
    get_memory_budget_gb,
    get_model_repos,
    get_system_memory_gb,
    iter_sweep_configs,
    resolve_run_params,
    should_skip_preset,
    sort_presets,
)


def run_single_subprocess(
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
    cmd = [
        sys.executable,
        str(RUN_BENCHMARK_SCRIPT),
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


def print_summary_table(results: list[BenchmarkResult]) -> None:
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


def run_sweep(args: argparse.Namespace) -> list[BenchmarkResult]:
    hardware = detect_hardware(args.hardware)
    safe_hw = hardware.replace(" ", "_").replace("/", "-")
    sweep_output_root = (
        Path(args.output_root) if args.output_root else RESULTS_DIR / safe_hw
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

    ckpt_path = state_path(sweep_output_root)
    if getattr(args, "reset_checkpoint", False) and ckpt_path.exists():
        ckpt_path.unlink()
        print(f"Reset checkpoint: {ckpt_path}")

    if ckpt_path.exists():
        state = load_state(ckpt_path)
    else:
        state = init_state(
            hardware=hardware,
            models=models,
            configs=[c.label for c in configs],
            sweep_output_root=str(sweep_output_root),
        )

    if getattr(args, "from_checkpoint", False) or getattr(args, "retry_failed", False):
        counts = summary_counts(state)
        print(f"Checkpoint: {ckpt_path} ({counts})")

    results: list[BenchmarkResult] = []
    run_idx = 0
    skipped_ckpt = 0

    for model_preset in models:
        preset_skip = should_skip_preset(model_preset, system_ram)
        if preset_skip:
            print(f"\n--- Skipping all {model_preset} runs: {preset_skip}")
            for config in configs:
                run_idx += 1
                result = failed_result(
                    model_preset=model_preset,
                    config=config,
                    hardware=hardware,
                    error=preset_skip,
                    prompt_tokens=args.prompt_tokens,
                    generation_tokens=args.generation_tokens,
                    num_trials=args.num_trials,
                )
                result.status = "skipped"
                out = result_path(
                    hardware,
                    model_preset,
                    config.label,
                    output_root=sweep_output_root if args.output_root else None,
                )
                save_result(result, out)
                mark_completed(
                    state,
                    preset=model_preset,
                    config_label=config.label,
                    status="skipped",
                    output_path=str(out),
                    error=preset_skip,
                )
                save_state(ckpt_path, state)
                results.append(result)
            continue

        for config in configs:
            run_idx += 1
            out = result_path(
                hardware,
                model_preset,
                config.label,
                output_root=sweep_output_root if args.output_root else None,
            )

            if should_skip(
                state,
                model_preset,
                config.label,
                from_checkpoint=getattr(args, "from_checkpoint", False),
                retry_failed=getattr(args, "retry_failed", False),
            ):
                skipped_ckpt += 1
                existing = load_result_from_file(out)
                if existing:
                    results.append(existing)
                    print(f"\n>>> Run {run_idx}/{total} SKIP (checkpoint): {model_preset} {config.label}")
                    continue

            print(f"\n>>> Run {run_idx}/{total}")
            params = resolve_run_params(model_preset, config)
            if params.model_repo is None:
                print(f"SKIP: no repo for {model_preset} {config.weight_label}")
                result = failed_result(
                    model_preset=model_preset,
                    config=config,
                    hardware=hardware,
                    error=f"No repo configured for {config.weight_bits}-bit",
                    prompt_tokens=args.prompt_tokens,
                    generation_tokens=args.generation_tokens,
                    num_trials=args.num_trials,
                )
                result.status = "skipped"
                save_result(result, out)
                mark_completed(
                    state,
                    preset=model_preset,
                    config_label=config.label,
                    status="skipped",
                    output_path=str(out),
                )
                save_state(ckpt_path, state)
                results.append(result)
                continue

            if not config_fits_memory(model_preset, config, memory_budget):
                est = estimate_peak_gb(model_preset, config)
                msg = (
                    f"Estimated peak {est:.1f} GB exceeds budget {memory_budget:.1f} GB"
                )
                print(f"SKIP (memory): {msg}")
                result = failed_result(
                    model_preset=model_preset,
                    config=config,
                    hardware=hardware,
                    error=msg,
                    prompt_tokens=args.prompt_tokens,
                    generation_tokens=args.generation_tokens,
                    num_trials=args.num_trials,
                )
                result.status = "skipped"
                save_result(result, out)
                mark_completed(
                    state,
                    preset=model_preset,
                    config_label=config.label,
                    status="skipped",
                    output_path=str(out),
                    error=msg,
                )
                save_state(ckpt_path, state)
                results.append(result)
                continue

            exit_code = run_single_subprocess(
                args,
                model_preset=model_preset,
                config=config,
                hardware=hardware,
                output=out,
            )
            result = load_result_from_file(out)
            if result is None:
                err = "subprocess failed"
                if exit_code in (-6, 134, 137):
                    err = "Out of memory (Metal OOM)"
                result = failed_result(
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
                save_result(result, out)
            elif exit_code != 0 and result.status == "ok":
                result.status = "error"
                result.error = f"exit code {exit_code}"
                save_result(result, out)

            mark_completed(
                state,
                preset=model_preset,
                config_label=config.label,
                status=result.status,
                output_path=str(out),
                error=result.error,
            )
            save_state(ckpt_path, state)
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
        "checkpoint": str(ckpt_path),
        "skipped_from_checkpoint": skipped_ckpt,
        "models": models,
        "config_order": [c.label for c in configs],
        "results": [asdict(r) for r in results],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{'=' * 60}")
    print("SWEEP COMPLETE")
    print(f"Checkpoint: {ckpt_path} ({summary_counts(state)})")
    print(f"Summary: {summary_path}")
    print_summary_table(results)
    return results
