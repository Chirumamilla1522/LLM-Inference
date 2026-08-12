#!/usr/bin/env python3
"""Generate Medium-ready PNG charts from article benchmark JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "docs" / "medium" / "images"


def _safe_hw(hardware: str) -> str:
    return hardware.replace(" ", "_").replace("/", "-")


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _ok(data: dict | None) -> bool:
    return bool(data and data.get("status") == "ok" and data.get("throughput_tps"))


def _bar_chart(
    labels: list[str],
    values: list[float],
    *,
    title: str,
    xlabel: str,
    output: Path,
    color: str = "steelblue",
    horizontal: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.45 + 1.5)))
    if horizontal:
        ax.barh(labels, values, color=color)
        ax.set_xlabel(xlabel)
        ax.invert_yaxis()
    else:
        ax.bar(labels, values, color=color)
        ax.set_ylabel(xlabel)
        plt.xticks(rotation=25, ha="right")
    ax.set_title(title, fontsize=11, pad=12)
    for i, v in enumerate(values):
        if horizontal:
            ax.text(v + max(values) * 0.01, i, f"{v:.1f}", va="center", fontsize=9)
        else:
            ax.text(i, v + max(values) * 0.02, f"{v:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def _grouped_bars(
    groups: list[str],
    series: dict[str, list[float]],
    *,
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(groups))
    width = 0.8 / max(len(series), 1)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, vals) in enumerate(series.items()):
        offset = (i - (len(series) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=name, color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def plot_article_0(hw: str, hw_dir: Path, out: Path) -> None:
    m3 = _load(hw_dir / "article_00_introduction" / "llama3-8b" / "demo_fp16.json")
    m5_dir = RESULTS_DIR / "Mac_M5_Max" / "article_01_weight-quantization" / "llama3-8b"
    m5_fp16 = _load(m5_dir / "fp16.json")
    m5_w4 = _load(m5_dir / "w4.json")
    m3_w4 = _load(hw_dir / "article_01_weight-quantization" / "llama3-8b" / "w4.json")
    if not _ok(m3):
        return
    labels = ["M3 fp16", "M3 w4"]
    tps = [m3["throughput_tps"]]
    mem = [m3["memory_gb"]]
    if _ok(m3_w4):
        tps.append(m3_w4["throughput_tps"])
        mem.append(m3_w4["memory_gb"])
    if _ok(m5_fp16) and _ok(m5_w4):
        labels.extend(["M5 Max fp16", "M5 Max w4"])
        tps.extend([m5_fp16["throughput_tps"], m5_w4["throughput_tps"]])
        mem.extend([m5_fp16["memory_gb"], m5_w4["memory_gb"]])
    _grouped_bars(
        labels,
        {"tok/s": tps, "Peak GB": mem},
        title=f"Llama 3.1 8B — {hw} vs M5 Max (when available)",
        ylabel="Value",
        output=out / "00_intro_hardware_compare.png",
    )


def plot_article_1(hw_dir: Path, out: Path) -> None:
    preset = "llama3-8b"
    labels, tps, mem = [], [], []
    for cfg in ("fp16", "w8", "w4", "w2"):
        data = _load(hw_dir / "article_01_weight-quantization" / preset / f"{cfg}.json")
        if _ok(data):
            labels.append(cfg)
            tps.append(data["throughput_tps"])
            mem.append(data["memory_gb"])
    if labels:
        _grouped_bars(
            labels,
            {"Decode tok/s": tps, "Peak memory (GB)": mem},
            title="Weight quantization — Llama 3.1 8B on Mac M3",
            ylabel="Value",
            output=out / "01_weight_quant_llama3-8b.png",
        )


def plot_article_2(hw_dir: Path, out: Path) -> None:
    presets = ["llama3-8b", "mistral-7b", "qwen-7b"]
    w4, w4kv = [], []
    ok_presets = []
    for p in presets:
        base = _load(hw_dir / "article_02_kv-cache-quantization" / p / f"{p}_w4.json")
        kv = _load(hw_dir / "article_02_kv-cache-quantization" / p / f"{p}_w4_kv.json")
        if _ok(base) and _ok(kv):
            ok_presets.append(p)
            w4.append(base["throughput_tps"])
            w4kv.append(kv["throughput_tps"])
    if ok_presets:
        _grouped_bars(
            ok_presets,
            {"w4 baseline": w4, "w4 + KV quant": w4kv},
            title="KV cache quantization — decode throughput (Mac M3)",
            ylabel="tok/s",
            output=out / "02_kv_cache_compare.png",
        )


def plot_article_3(hw_dir: Path, out: Path) -> None:
    runs = [
        ("512 tok baseline", "w4_baseline.json"),
        ("prefill ON", "w4_prefill.json"),
        ("p=1024", "w4_prefill_p1024.json"),
        ("p=256", "w4_prefill_p256.json"),
    ]
    labels, ttft = [], []
    base = hw_dir / "article_03_prefill-ttft" / "llama3-8b"
    for label, fname in runs:
        data = _load(base / fname)
        if _ok(data):
            labels.append(label)
            ttft.append(data["ttft_ms"])
    if labels:
        _bar_chart(
            labels,
            ttft,
            title="Prefill tuning — TTFT vs prompt shape (Llama 3.1 8B, w4)",
            xlabel="TTFT (ms)",
            output=out / "03_prefill_ttft.png",
            color="coral",
            horizontal=True,
        )


def plot_article_4(hw_dir: Path, out: Path) -> None:
    presets = [
        ("0.5B", "qwen-0.5b"),
        ("1.5B", "qwen-1.5b"),
        ("3B", "qwen-3b"),
        ("7B", "qwen-7b"),
        ("8B", "llama3-8b"),
        ("9B", "gemma-9b"),
    ]
    labels, tps, mem = [], [], []
    for short, preset in presets:
        data = _load(hw_dir / "article_04_model-size-ladder" / preset / "w4.json")
        if _ok(data):
            labels.append(short)
            tps.append(data["throughput_tps"])
            mem.append(data["memory_gb"])
    if labels:
        _grouped_bars(
            labels,
            {"tok/s @ w4": tps, "Peak GB": mem},
            title="Model size ladder — w4 on Mac M3 (24 GB)",
            ylabel="Value",
            output=out / "04_model_size_ladder.png",
        )


def plot_article_5(hw_dir: Path, out: Path) -> None:
    base = hw_dir / "article_05_full-stack" / "llama3-8b"
    fp16 = _load(base / "fp16.json")
    opt = _load(base / "optimized.json")
    if _ok(fp16) and _ok(opt):
        _grouped_bars(
            ["fp16", "w4+kv+prefill"],
            {
                "tok/s": [fp16["throughput_tps"], opt["throughput_tps"]],
                "Peak GB": [fp16["memory_gb"], opt["memory_gb"]],
            },
            title="Full stack — fp16 vs optimized (Llama 3.1 8B, Mac M3)",
            ylabel="Value",
            output=out / "05_full_stack.png",
        )


def plot_article_6(hw_dir: Path, out: Path) -> None:
    preset = "qwen-7b"
    base = _load(
        hw_dir / "article_06_speculative-decoding" / preset / f"{preset}_w4_baseline.json"
    )
    spec = _load(
        hw_dir / "article_06_speculative-decoding" / preset / f"{preset}_w4_speculative.json"
    )
    if _ok(base) and _ok(spec):
        accept = spec.get("draft_accept_rate", 0) * 100
        _grouped_bars(
            ["baseline w4", f"speculative ({accept:.0f}% accept)"],
            {"tok/s": [base["throughput_tps"], spec["throughput_tps"]]},
            title=f"Speculative decoding — {preset} on Mac M3",
            ylabel="Decode tok/s",
            output=out / "06_speculative_qwen-7b.png",
        )


def plot_article_7(hw_dir: Path, out: Path) -> None:
    base = hw_dir / "article_07_context-and-cache" / "llama3-8b"
    prompt_runs = [("p=256", "ctx_p256.json"), ("p=512", "ctx_p512.json"),
                   ("p=1024", "ctx_p1024.json"), ("p=2048", "ctx_p2048.json")]
    labels, ttft = [], []
    for label, fname in prompt_runs:
        data = _load(base / fname)
        if _ok(data):
            labels.append(label)
            ttft.append(data["ttft_ms"])
    if labels:
        _bar_chart(
            labels,
            ttft,
            title="Context length vs TTFT (Llama 3.1 8B, w4+prefill)",
            xlabel="TTFT (ms)",
            output=out / "07_context_ttft.png",
            color="#6A994E",
            horizontal=True,
        )
    prefix = _load(base / "prefix_cache.json")
    if prefix and prefix.get("prefix_cache_cold_ttft_ms"):
        cold = prefix["prefix_cache_cold_ttft_ms"]
        warm = prefix["prefix_cache_warm_ttft_ms"]
        _grouped_bars(
            ["cold prefill", "warm (cached prefix)"],
            {"TTFT (ms)": [cold, warm]},
            title="Prefix KV cache — cold vs warm TTFT",
            ylabel="ms",
            output=out / "07_prefix_cache.png",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", default="Mac M3")
    parser.add_argument("-o", "--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("Install: pip install -r requirements-dev.txt")
        return 1

    hw_dir = RESULTS_DIR / _safe_hw(args.hardware)
    out = args.output_dir
    plot_article_0(args.hardware, hw_dir, out)
    plot_article_1(hw_dir, out)
    plot_article_2(hw_dir, out)
    plot_article_3(hw_dir, out)
    plot_article_4(hw_dir, out)
    plot_article_5(hw_dir, out)
    plot_article_6(hw_dir, out)
    plot_article_7(hw_dir, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
