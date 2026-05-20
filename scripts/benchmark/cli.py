#!/usr/bin/env python3
"""CLI entry for MLX benchmark harness."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from benchmark.paths import RESULTS_DIR
from benchmark.result import (
    BenchmarkResult,
    detect_hardware,
    failed_result,
    load_result_from_file,
    result_path,
    save_result,
)
from benchmark.runner import benchmark_once, benchmark_prefix_cache
from benchmark.schema import SCHEMA_VERSION, WARMUP_POLICY
from benchmark.sweep import run_single_subprocess, run_sweep
from optimizations import (
    DEFAULT_NUM_DRAFT_TOKENS,
    WEIGHT_BITS_ORDER,
    OptimizationConfig,
    check_hf_auth_for_repo,
    get_model_repos,
    hf_token_present,
    sort_presets,
)
from workloads import get_workload, iter_workloads


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

    hardware = detect_hardware(args.hardware)
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
        result = failed_result(
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
    out = args.output or result_path(
        hardware,
        args.preset,
        config.label,
        output_root=output_root,
        run_label=label if output_root else None,
    )
    save_result(result, out)
    print(f"\nSaved: {out}")
    return [result]


def run_workload_sweep(args: argparse.Namespace) -> list[BenchmarkResult]:
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

    hardware = detect_hardware(args.hardware)
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
        exit_code = run_single_subprocess(
            args,
            model_preset=args.preset,
            config=config,
            hardware=hardware,
            output=out,
            run_label=profile.id,
        )
        result = load_result_from_file(out)
        if result is None:
            result = failed_result(
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark MLX LLM inference with optimization sweeps."
    )
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--weights-only", action="store_true")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--include-large", action="store_true")
    parser.add_argument("--include-qwen", action="store_true")
    parser.add_argument("--skip-presets", help="Comma-separated presets to skip")
    parser.add_argument("--memory-fraction", type=float, default=0.75)
    parser.add_argument(
        "--preset",
        choices=sort_presets(list(get_model_repos())),
        default="llama3-8b",
    )
    parser.add_argument("--config", help="e.g. fp16, w4+kv_cache+prefill")
    parser.add_argument("--weight-bits", type=int, choices=WEIGHT_BITS_ORDER)
    parser.add_argument("--kv-cache", action="store_true")
    parser.add_argument("--prefill", action="store_true")
    parser.add_argument("--max-combo-size", type=int, choices=[0, 1, 2])
    parser.add_argument("--model", help="Override HF repo")
    parser.add_argument("--hardware", help="e.g. Mac M3")
    parser.add_argument("-p", "--prompt-tokens", type=int, default=512)
    parser.add_argument("-g", "--generation-tokens", type=int, default=128)
    parser.add_argument("-n", "--num-trials", type=int, default=3)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--delay-between-configs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--hf-check", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--article-id", type=int)
    parser.add_argument("--run-label")
    parser.add_argument("--speculative", action="store_true")
    parser.add_argument("--draft-preset")
    parser.add_argument("--num-draft-tokens", type=int, default=DEFAULT_NUM_DRAFT_TOKENS)
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--workload", metavar="ID")
    parser.add_argument("--workload-sweep", action="store_true")
    parser.add_argument("--list-workloads", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--from-checkpoint",
        action="store_true",
        help="Resume sweep: skip (preset, config) already in sweep_state.json.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Resume sweep: only run entries not status=ok in checkpoint.",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Delete sweep_state.json before starting.",
    )
    return parser


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
