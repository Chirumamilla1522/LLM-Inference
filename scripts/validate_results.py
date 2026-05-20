#!/usr/bin/env python3
"""Validate benchmark JSON under results/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark_schema import SCHEMA_VERSION, validate_result

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

SKIP_NAMES = frozenset(
    {
        "manifest.json",
        "article_summary.json",
        "sweep_summary.json",
        "workload_sweep_summary.json",
    }
)


def iter_result_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    if not root.exists():
        return paths
    for path in sorted(root.rglob("*.json")):
        if path.name in SKIP_NAMES:
            continue
        if path.name.startswith("sweep_"):
            continue
        if path.parent == RESULTS_DIR and path.suffix == ".json":
            continue  # top-level sweep dumps
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate results JSON schema")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Root results directory",
    )
    parser.add_argument("--hardware", help="Only Mac_M3 / Mac_M5_Max subdir")
    parser.add_argument("--failures-only", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    args = parser.parse_args()

    root = args.results_dir
    if args.hardware:
        safe = args.hardware.replace(" ", "_").replace("/", "-")
        root = root / safe

    files = iter_result_files(root)
    report: dict = {
        "schema_version": SCHEMA_VERSION,
        "files_checked": len(files),
        "ok": 0,
        "invalid_json": 0,
        "validation_errors": 0,
        "run_errors": 0,
        "stale_config": 0,
        "entries": [],
    }

    exit_code = 0
    for path in files:
        rel = str(path.relative_to(ROOT))
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            report["invalid_json"] += 1
            report["entries"].append({"path": rel, "error": f"invalid JSON: {exc}"})
            exit_code = 1
            continue

        errs = validate_result(data, path=rel + ": ")
        status = data.get("status", "unknown")
        cfg = data.get("configuration", "")

        entry = {
            "path": rel,
            "status": status,
            "configuration": cfg,
            "errors": errs,
        }
        if errs:
            report["validation_errors"] += 1
            exit_code = 1
        elif status == "error":
            report["run_errors"] += 1
            if args.failures_only or not args.json:
                entry["run_error"] = data.get("error")
        elif status == "ok":
            report["ok"] += 1
        else:
            report["ok"] += 1

        if cfg in ("baseline", "quantization", "kv_cache", "prefill") or "+" in cfg and any(
            p in cfg for p in ("quantization", "baseline")
        ):
            report["stale_config"] += 1
            entry["stale_config"] = True

        if args.failures_only:
            if not (errs or status == "error" or entry.get("stale_config")):
                continue
        report["entries"].append(entry)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Schema v{SCHEMA_VERSION} | checked {report['files_checked']} files")
        print(
            f"  ok={report['ok']}  run_errors={report['run_errors']}  "
            f"validation_errors={report['validation_errors']}  "
            f"stale_config_labels={report['stale_config']}"
        )
        for entry in report["entries"]:
            if entry.get("errors"):
                print(f"  INVALID {entry['path']}: {entry['errors']}")
            elif entry.get("stale_config"):
                print(f"  STALE   {entry['path']}: configuration={entry['configuration']}")
            elif entry.get("run_error") and args.failures_only:
                print(f"  FAIL    {entry['path']}: {entry['run_error']}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
