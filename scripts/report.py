#!/usr/bin/env python3
"""Validate results, print tables, and report data freshness."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
GENERATED_DIR = ROOT / "docs" / "articles" / "_generated"


def _run_script(script: str, *args: str) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    print("$", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode


def collect_freshness(results_dir: Path) -> dict:
    latest: datetime | None = None
    ok = err = stale = 0
    for path in results_dir.rglob("*.json"):
        if path.name.startswith("sweep_") or path.parent == results_dir:
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        ts = data.get("timestamp")
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if latest is None or dt > latest:
                    latest = dt
            except ValueError:
                pass
        st = data.get("status")
        if st == "ok":
            ok += 1
        elif st == "error":
            err += 1
        cfg = data.get("configuration", "")
        if cfg in ("baseline", "quantization", "kv_cache"):
            stale += 1
    return {
        "latest_run": latest.isoformat() if latest else None,
        "ok_runs": ok,
        "error_runs": err,
        "stale_config_labels": stale,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark report pipeline")
    parser.add_argument("--hardware", default="Mac M3")
    parser.add_argument("--article", type=int, help="Generate tables for one article")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--migrate-dry-run", action="store_true")
    parser.add_argument("-o", "--output", type=Path, help="Write combined markdown report")
    args = parser.parse_args()

    safe = args.hardware.replace(" ", "_").replace("/", "-")
    hw_dir = RESULTS_DIR / safe

    lines = [f"# Benchmark report — {args.hardware}\n"]
    exit_code = 0

    if not args.skip_validate:
        code = _run_script(
            "validate_results.py",
            "--hardware",
            args.hardware,
            "--failures-only",
        )
        if code:
            exit_code = code
        lines.append("\n## Validation\n\nSee stdout (failures only).\n")

    if args.migrate_dry_run:
        _run_script("migrate_results.py", "--hardware", args.hardware, "--dry-run")

    fresh = collect_freshness(hw_dir)
    lines.append("## Data freshness\n\n")
    lines.append(f"- Latest run: `{fresh['latest_run'] or 'none'}`\n")
    lines.append(f"- OK: {fresh['ok_runs']} | Errors: {fresh['error_runs']}\n")
    lines.append(f"- Stale config labels: {fresh['stale_config_labels']}\n")
    if fresh["stale_config_labels"]:
        lines.append(
            "\n> Run `python scripts/migrate_results.py --apply --hardware \""
            + args.hardware
            + '"`\n'
        )

    GEN_DIR = GENERATED_DIR
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    table_args = ["--hardware", args.hardware]
    if args.article is not None:
        table_args.extend(["--article", str(args.article)])
    out_tables = GEN_DIR / f"tables_{safe}.md"
    if args.article is not None:
        out_tables = GEN_DIR / f"tables_article{args.article:02d}_{safe}.md"
    table_args.extend(["-o", str(out_tables)])

    if _run_script("generate_article_tables.py", *table_args) == 0:
        lines.append(f"\n## Tables\n\nWritten to `{out_tables.relative_to(ROOT)}`.\n")
        if out_tables.exists():
            lines.append(out_tables.read_text())

    # Runtime compare summary
    a10 = hw_dir / "article_10_runtimes"
    compare_files = list(a10.rglob("*_compare.json")) if a10.exists() else []
    if compare_files:
        lines.append("\n## Article 10 — MLX vs llama.cpp\n\n")
        lines.append("| Preset | Config | MLX tok/s | llama.cpp tg | Ratio |\n")
        lines.append("|--------|--------|-----------|--------------|-------|\n")
        for path in sorted(compare_files):
            d = json.loads(path.read_text())
            mlx_t = (d.get("mlx") or {}).get("throughput_tps")
            lc_t = (d.get("llamacpp") or {}).get("tg_tps")
            ratio = (d.get("comparison") or {}).get(
                "throughput_ratio_mlx_over_llamacpp"
            )
            lines.append(
                f"| {d.get('model_preset')} | {d.get('configuration')} | "
                f"{mlx_t or '—'} | {lc_t or '—'} | {ratio or '—'} |\n"
            )

    m5 = RESULTS_DIR / "Mac_M5_Max"
    if not m5.exists():
        lines.append(
            "\n## M5 Max\n\nNo `results/Mac_M5_Max/` yet. "
            "Run `./scripts/run_m5_ladder.sh` on M5 hardware.\n"
        )

    report_text = "".join(lines)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_text)
        print(f"Wrote {args.output}")
    else:
        print(report_text)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
