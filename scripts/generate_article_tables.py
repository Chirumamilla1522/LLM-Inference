#!/usr/bin/env python3
"""Print markdown tables from article benchmark JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from articles import get_article, list_articles

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def _safe_hw(hardware: str) -> str:
    return hardware.replace(" ", "_").replace("/", "-")


def load_json_results(article_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not article_dir.exists():
        return rows
    for path in sorted(article_dir.rglob("*.json")):
        if path.name in ("manifest.json", "article_summary.json", "sweep_summary.json"):
            continue
        if path.name.startswith("sweep_"):
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "throughput_tps" in data or "status" in data:
            data["_path"] = str(path.relative_to(ROOT))
            rows.append(data)
    return rows


def print_table(rows: list[dict], title: str) -> None:
    if not rows:
        print(f"\n### {title}\n\n_No results found._\n")
        return

    print(f"\n### {title}\n")
    has_workload = any(r.get("workload") for r in rows)
    if has_workload:
        print("| Model | Run | Workload | P | Memory (GB) | TTFT (ms) | tok/s | Status |")
        print("|-------|-----|----------|---|-------------|-----------|-------|--------|")
    else:
        print("| Model | Run | Memory (GB) | TTFT (ms) | tok/s | Status |")
        print("|-------|-----|-------------|-----------|-------|--------|")
    for r in sorted(rows, key=lambda x: (x.get("model_preset", ""), x.get("run_label", x.get("configuration", "")))):
        preset = r.get("model_preset", "—")
        label = r.get("run_label") or r.get("configuration", "—")
        wl = r.get("workload") or {}
        wl_id = wl.get("workload_id", "—") if isinstance(wl, dict) else "—"
        pressure = wl.get("workload_pressure", "—") if isinstance(wl, dict) else "—"
        if r.get("status") != "ok":
            if has_workload:
                print(
                    f"| {preset} | {label} | {wl_id} | {pressure} | — | — | — | "
                    f"{r.get('status', 'fail')} |"
                )
            else:
                print(f"| {preset} | {label} | — | — | — | {r.get('status', 'fail')} |")
            continue
        mem = r.get("memory_gb", 0)
        ttft = r.get("ttft_ms", 0)
        tps = r.get("throughput_tps", 0)
        extra = ""
        if r.get("prefix_cache_warm_ttft_ms") is not None:
            extra = (
                f" cold={r.get('prefix_cache_cold_ttft_ms', 0):.0f}ms "
                f"warm={r.get('prefix_cache_warm_ttft_ms', 0):.0f}ms"
            )
        if r.get("draft_accept_rate") is not None:
            extra += f" accept={r['draft_accept_rate']:.1%}"
        if has_workload:
            print(
                f"| {preset} | {label} | {wl_id} | {pressure} | {mem:.2f} | "
                f"{ttft:.0f}{extra} | {tps:.1f} | ok |"
            )
        else:
            print(f"| {preset} | {label} | {mem:.2f} | {ttft:.0f}{extra} | {tps:.1f} | ok |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", default="Mac M3")
    parser.add_argument("--article", type=int, help="Article id; default all with data.")
    parser.add_argument("-o", "--output", type=Path, help="Write markdown to file.")
    args = parser.parse_args()

    lines: list[str] = ["# Benchmark tables\n"]
    articles = [get_article(args.article)] if args.article is not None else list_articles()
    hw_dir = RESULTS_DIR / _safe_hw(args.hardware)

    for article in articles:
        article_dir = hw_dir / article.dir_name
        rows = load_json_results(article_dir)
        section = []
        print_table(rows, f"Article {article.id}: {article.title}")
        # capture via re-print to file is messy; build inline
        if rows:
            section.append(f"\n## Article {article.id}: {article.title}\n")
            section.append("| Model | Run | Memory | TTFT | tok/s | Status |\n")
            section.append("|-------|-----|--------|------|-------|--------|\n")
            for r in rows:
                preset = r.get("model_preset", "—")
                label = r.get("run_label") or r.get("configuration", "—")
                if r.get("status") != "ok":
                    section.append(
                        f"| {preset} | {label} | — | — | — | {r.get('status')} |\n"
                    )
                    continue
                section.append(
                    f"| {preset} | {label} | {r.get('memory_gb', 0):.2f} | "
                    f"{r.get('ttft_ms', 0):.0f} | {r.get('throughput_tps', 0):.1f} | ok |\n"
                )
        lines.extend(section)

    if args.output:
        args.output.write_text("".join(lines))
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
