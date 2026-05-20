#!/usr/bin/env python3
"""Generate matplotlib charts from benchmark JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"


def _safe_hw(hardware: str) -> str:
    return hardware.replace(" ", "_").replace("/", "-")


def load_ok_results(hw_dir: Path, preset: str) -> list[dict]:
    rows: list[dict] = []
    model_dir = hw_dir / preset
    if not model_dir.exists():
        return rows
    for path in sorted(model_dir.glob("*.json")):
        if path.name.startswith("sweep_"):
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("status") == "ok" and data.get("throughput_tps"):
            data["_file"] = path.name
            rows.append(data)
    return rows


def plot_config_bars(rows: list[dict], *, title: str, output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [r.get("configuration", r.get("_file", "")) for r in rows]
    tps = [r["throughput_tps"] for r in rows]
    ttft = [r.get("ttft_ms", 0) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.barh(labels, tps, color="steelblue")
    ax1.set_xlabel("Decode throughput (tok/s)")
    ax1.set_title("Throughput")
    ax1.invert_yaxis()

    ax2.barh(labels, ttft, color="coral")
    ax2.set_xlabel("TTFT (ms)")
    ax2.set_title("Time to first token")
    ax2.invert_yaxis()

    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")


def plot_hardware_compare(
    preset: str,
    config: str,
    hardware_dirs: dict[str, Path],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    names: list[str] = []
    tps_vals: list[float] = []
    mem_vals: list[float] = []

    for hw_label, hw_dir in hardware_dirs.items():
        path = hw_dir / preset / f"{config}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if data.get("status") != "ok":
            continue
        names.append(hw_label)
        tps_vals.append(data["throughput_tps"])
        mem_vals.append(data.get("memory_gb", 0))

    if not names:
        print(f"No comparable OK results for {preset} / {config}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.bar(names, tps_vals, color=["#4C72B0", "#DD8452"][: len(names)])
    ax1.set_ylabel("tok/s")
    ax1.set_title(f"{preset} {config} — throughput")

    ax2.bar(names, mem_vals, color=["#55A868", "#C44E52"][: len(names)])
    ax2.set_ylabel("GB")
    ax2.set_title("Peak memory")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot benchmark JSON")
    parser.add_argument("--hardware", default="Mac M3")
    parser.add_argument("--preset", default="llama3-8b")
    parser.add_argument("--compare-hardware", help="Second hardware label, e.g. 'Mac M5 Max'")
    parser.add_argument("--config", default="w4+kv_cache+prefill")
    parser.add_argument("-o", "--output-dir", type=Path, default=PLOTS_DIR)
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("Install plots extra: pip install -r requirements-dev.txt")
        return 1

    safe = _safe_hw(args.hardware)
    hw_dir = RESULTS_DIR / safe
    rows = load_ok_results(hw_dir, args.preset)
    if rows:
        plot_config_bars(
            rows,
            title=f"{args.hardware} — {args.preset}",
            output=args.output_dir / safe / f"{args.preset}_configs.png",
        )

    if args.compare_hardware:
        other_safe = _safe_hw(args.compare_hardware)
        dirs = {
            args.hardware: hw_dir,
            args.compare_hardware: RESULTS_DIR / other_safe,
        }
        plot_hardware_compare(
            args.preset,
            args.config,
            dirs,
            args.output_dir / f"{args.preset}_{args.config}_hardware.png",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
