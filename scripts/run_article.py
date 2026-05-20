#!/usr/bin/env python3
"""Run benchmarks for one article (0-11) in the 12-post series."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from articles import (
    Article,
    RunKind,
    count_planned_runs,
    get_article,
    iter_sweep_configs_for_article,
    list_articles,
    presets_for_article_sweep,
)
from optimizations import OptimizationConfig, get_model_repos

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RUN_BENCHMARK = ROOT / "scripts" / "run_benchmark.py"


def _safe_hw(hardware: str) -> str:
    return hardware.replace(" ", "_").replace("/", "-")


def article_output_dir(hardware: str, article: Article) -> Path:
    return RESULTS_DIR / _safe_hw(hardware) / article.dir_name


def _write_concept_manifest(
    hardware: str,
    article: Article,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "article_id": article.id,
        "slug": article.slug,
        "title": article.title,
        "hardware": hardware,
        "benchmark_mode": "concept",
        "status": "concept_only",
        "topics": list(article.concept_topics),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "No MLX benchmark for this article. See docs/articles/ and write concept content.",
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def _run_subprocess(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode


def run_article_sweep(
    article: Article,
    hardware: str,
    args: argparse.Namespace,
) -> int:
    if article.sweep is None:
        return 0

    out_dir = article_output_dir(hardware, article)
    cmd = [
        sys.executable,
        str(RUN_BENCHMARK),
        "--sweep",
        "--hardware",
        hardware,
        "--output-root",
        str(out_dir),
        "--article-id",
        str(article.id),
        "-n",
        str(args.num_trials),
        "-p",
        str(args.prompt_tokens),
        "-g",
        str(args.generation_tokens),
        "--delay-between-configs",
        str(args.delay_between_configs),
    ]
    if article.sweep.all_models:
        cmd.append("--all-models")
    if article.sweep.weights_only:
        cmd.append("--weights-only")
    if args.include_large or article.sweep.include_large:
        cmd.append("--include-large")
    if article.sweep.max_combo_size is not None:
        cmd.extend(["--max-combo-size", str(article.sweep.max_combo_size)])

    return _run_subprocess(cmd)


def run_article_runs(
    article: Article,
    hardware: str,
    args: argparse.Namespace,
) -> int:
    out_dir = article_output_dir(hardware, article)
    failures = 0

    for run in article.runs:
        config = OptimizationConfig.from_label(run.config)
        output = out_dir / run.preset / f"{run.label}.json"
        cmd = [
            sys.executable,
            str(RUN_BENCHMARK),
            "--preset",
            run.preset,
            "--config",
            run.config,
            "--hardware",
            hardware,
            "--output-root",
            str(out_dir),
            "--run-label",
            run.label,
            "--article-id",
            str(article.id),
            "-n",
            str(args.num_trials),
            "-p",
            str(run.prompt_tokens or args.prompt_tokens),
            "-g",
            str(run.generation_tokens or args.generation_tokens),
            "-o",
            str(output),
        ]
        if run.kind == RunKind.SPECULATIVE:
            cmd.append("--speculative")
            if run.num_draft_tokens is not None:
                cmd.extend(["--num-draft-tokens", str(run.num_draft_tokens)])
            if run.draft_preset:
                cmd.extend(["--draft-preset", run.draft_preset])
        elif run.kind == RunKind.PREFIX_CACHE:
            cmd.append("--prefix-cache")

        code = _run_subprocess(cmd)
        if code != 0:
            failures += 1
        if args.delay_between_configs > 0:
            import time

            time.sleep(args.delay_between_configs)

    return failures


def run_article(article_id: int, args: argparse.Namespace) -> int:
    article = get_article(article_id)
    hardware = args.hardware
    out_dir = article_output_dir(hardware, article)

    print(f"\n{'=' * 60}")
    print(f"Article {article.id}: {article.title}")
    print(f"Output: {out_dir}")
    planned = count_planned_runs(article, args.include_large)
    if planned:
        print(f"Planned MLX runs: ~{planned}")
    print(f"{'=' * 60}\n")

    if not article.benchmarked:
        if args.dry_run:
            print("Concept article — would write manifest.json (no MLX runs).")
            return 0
        path = _write_concept_manifest(hardware, article, out_dir)
        if article.id == 11:
            checklist = {
                "decision_tree": [
                    "1. Enough RAM for fp16 weights? → if no, use w8/w4/w2 (Article 1)",
                    "2. Long generations eating RAM? → enable kv_cache (Article 2)",
                    "3. High TTFT on long prompts? → enable prefill (Article 3)",
                    "4. Need faster decode? → speculative decoding (Article 6)",
                    "5. Repeated system prompt? → prefix KV cache (Article 7)",
                    "6. Multi-user production? → serving stack (Article 8), not single-stream MLX",
                ],
            }
            (out_dir / "optimization_checklist.json").write_text(
                json.dumps(checklist, indent=2) + "\n"
            )
        print(f"Concept article — wrote {path}")
        return 0

    if args.dry_run:
        if article.sweep:
            configs = [c.label for c in iter_sweep_configs_for_article(article)]
            models = presets_for_article_sweep(article, args.include_large)
            print(f"Sweep: {len(models)} models × {len(configs)} configs")
            for c in configs:
                print(f"  config: {c}")
            for m in models:
                print(f"  model: {m}")
        for run in article.runs:
            print(
                f"  run: {run.label} preset={run.preset} config={run.config} "
                f"kind={run.kind.value}"
            )
        return 0

    if args.hf_check:
        code = _run_subprocess([sys.executable, str(RUN_BENCHMARK), "--hf-check"])
        if code != 0:
            return code

    failures = 0
    if article.sweep and not args.runs_only:
        failures += run_article_sweep(article, hardware, args)
    if article.runs and not args.sweep_only:
        failures += run_article_runs(article, hardware, args)

    summary = {
        "article_id": article.id,
        "slug": article.slug,
        "title": article.title,
        "hardware": hardware,
        "output_dir": str(out_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "failures": failures,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "article_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"\nArticle {article.id} done. failures={failures}")
    print(f"Summary: {out_dir / 'article_summary.json'}")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run benchmarks for one article in the 12-post series."
    )
    parser.add_argument(
        "--article",
        type=int,
        help="Article id 0-11 (use --list to see all).",
    )
    parser.add_argument(
        "--all-benchmarked",
        action="store_true",
        help="Run all benchmarked articles (0-7) in order.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List articles and exit.",
    )
    parser.add_argument("--hardware", default="Mac M3", help="Hardware label.")
    parser.add_argument("-n", "--num-trials", type=int, default=3)
    parser.add_argument("-p", "--prompt-tokens", type=int, default=512)
    parser.add_argument("-g", "--generation-tokens", type=int, default=128)
    parser.add_argument(
        "--delay-between-configs",
        type=int,
        default=5,
        help="Seconds between runs.",
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Include 12B+ presets in sweeps.",
    )
    parser.add_argument("--hf-check", action="store_true", help="Run HF check first.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing.",
    )
    parser.add_argument(
        "--runs-only",
        action="store_true",
        help="Skip article sweep; only explicit runs.",
    )
    parser.add_argument(
        "--sweep-only",
        action="store_true",
        help="Skip explicit runs; only sweep.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.list:
        for article in list_articles():
            flag = "bench" if article.benchmarked else "concept"
            n = count_planned_runs(article, include_large=False)
            print(f"  {article.id:2d}  [{flag:7s}]  {article.title}  (~{n} runs)")
        return

    if args.all_benchmarked:
        code = 0
        for article in list_articles():
            if not article.benchmarked:
                continue
            if run_article(article.id, args) != 0:
                code = 1
        raise SystemExit(code)

    if args.article is None:
        print("Specify --article N or --all-benchmarked or --list")
        raise SystemExit(2)

    raise SystemExit(run_article(args.article, args))


if __name__ == "__main__":
    main()
