#!/usr/bin/env python3
"""Migrate legacy result JSON to schema v1 labels and modern repos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark_schema import (
    SCHEMA_VERSION,
    WARMUP_POLICY,
    fix_stale_repo,
    infer_config_from_legacy,
    legacy_optimizations_to_modern,
    normalize_config_label,
)

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


def migrate_data(data: dict) -> tuple[dict, list[str]]:
    changes: list[str] = []
    out = dict(data)

    old_cfg = out.get("configuration", "")
    new_cfg = infer_config_from_legacy(out)
    if new_cfg != old_cfg:
        out["configuration"] = new_cfg
        changes.append(f"configuration: {old_cfg} → {new_cfg}")

    opt = out.get("optimizations")
    if isinstance(opt, dict) and "weight_bits" not in opt:
        modern = legacy_optimizations_to_modern(opt)
        out["optimizations"] = modern
        if "weight_bits" not in data.get("optimizations", {}):
            changes.append(f"optimizations: legacy → {modern}")

    if out.get("weight_bits") is None and isinstance(out.get("optimizations"), dict):
        out["weight_bits"] = out["optimizations"].get("weight_bits", 16)

    repo = out.get("model_repo")
    fixed_repo, repo_changed = fix_stale_repo(repo)
    if repo_changed and fixed_repo:
        out["model_repo"] = fixed_repo
        changes.append(f"model_repo: {repo} → {fixed_repo}")

    if out.get("schema_version") != SCHEMA_VERSION:
        out["schema_version"] = SCHEMA_VERSION
        changes.append(f"schema_version: → {SCHEMA_VERSION}")

    if "warmup_policy" not in out:
        out["warmup_policy"] = WARMUP_POLICY
        changes.append("warmup_policy: added")

    return out, changes


def target_path(path: Path, new_cfg: str) -> Path:
    """Rename file stem if it matches legacy configuration name."""
    stem = path.stem
    normalized_stem = normalize_config_label(stem)
    if stem != normalized_stem:
        return path.with_name(f"{normalized_stem}.json")
    if stem == path.stem and stem != new_cfg and stem in (
        "baseline",
        "quantization",
        "kv_cache",
        "prefill",
    ):
        return path.with_name(f"{new_cfg}.json")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy results JSON")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--hardware", help="Only one hardware subdir")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview only (default). Use --apply to write.",
    )
    parser.add_argument("--apply", action="store_true", help="Write migrations")
    parser.add_argument("--delete-legacy-files", action="store_true")
    args = parser.parse_args()

    apply = args.apply
    root = args.results_dir
    if args.hardware:
        safe = args.hardware.replace(" ", "_").replace("/", "-")
        root = root / safe

    if not root.exists():
        print(f"No directory: {root}")
        return 0

    migrated = 0
    skipped = 0
    for path in sorted(root.rglob("*.json")):
        if path.name in SKIP_NAMES or path.name.startswith("sweep_"):
            continue
        if path.parent == RESULTS_DIR:
            continue

        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"SKIP (bad JSON): {path}")
            skipped += 1
            continue

        new_data, changes = migrate_data(data)
        if not changes:
            continue

        new_cfg = new_data["configuration"]
        dest = target_path(path, new_cfg)

        print(f"{path.relative_to(ROOT)}")
        for c in changes:
            print(f"  {c}")
        if dest != path:
            print(f"  rename: {path.name} → {dest.name}")

        if apply:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(new_data, indent=2) + "\n")
            if dest != path:
                if args.delete_legacy_files and path.exists():
                    path.unlink()
                elif path.exists() and dest.resolve() != path.resolve():
                    print(f"  (left {path.name}; remove manually or use --delete-legacy-files)")
            migrated += 1
        else:
            migrated += 1

    mode = "APPLY" if apply else "DRY-RUN"
    print(f"\n{mode}: {migrated} file(s) would change, {skipped} skipped")
    if not apply and migrated:
        print("Re-run with: python scripts/migrate_results.py --apply --hardware 'Mac M3'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
