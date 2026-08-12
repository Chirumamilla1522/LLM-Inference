#!/usr/bin/env python3
"""Generate Medium-ready PNG charts from article benchmark JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from medium_image_layout import SOURCE, add_article_arg, emit_file, resolve_article, source_keys_for_article  # noqa: E402

RESULTS_DIR = ROOT / "results"
OUT_DIR = SOURCE
_ARTICLE: str | None = None


def _emit(output: Path) -> None:
    src_key = output.name  # flat chart assets
    if _ARTICLE and src_key not in source_keys_for_article(_ARTICLE):
        print(f"skip {src_key} (not used by {_ARTICLE})")
        return
    emit_file(src_key, output, article=_ARTICLE)


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
    _emit(output)


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
    _emit(output)


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


def plot_extra_results(hw_dir: Path, out: Path) -> None:
    """Additional result figures for denser Medium posts."""
    import matplotlib.pyplot as plt

    # Multi-model weight quant throughput
    presets = [
        ("qwen-0.5b", "0.5B"),
        ("llama-3.2-1b", "1B"),
        ("qwen-3b", "3B"),
        ("mistral-7b", "7B"),
        ("llama3-8b", "8B"),
        ("gemma-9b", "9B"),
    ]
    series: dict[str, list[float]] = {"fp16": [], "w8": [], "w4": []}
    groups: list[str] = []
    for preset, short in presets:
        row = {}
        for cfg in ("fp16", "w8", "w4"):
            data = _load(hw_dir / "article_01_weight-quantization" / preset / f"{cfg}.json")
            if _ok(data):
                row[cfg] = data["throughput_tps"]
        if len(row) >= 2:
            groups.append(short)
            for cfg in series:
                series[cfg].append(row.get(cfg, 0.0))
    if groups:
        _grouped_bars(
            groups,
            {k: v for k, v in series.items() if any(x > 0 for x in v)},
            title="Weight quant across model sizes — decode tok/s (Mac M3)",
            ylabel="tok/s",
            output=out / "01_multi_model_quant_tps.png",
        )

    # Memory vs throughput scatter for llama3-8b configs
    preset = "llama3-8b"
    xs, ys, labels = [], [], []
    for cfg in ("fp16", "w8", "w4", "w2"):
        data = _load(hw_dir / "article_01_weight-quantization" / preset / f"{cfg}.json")
        if _ok(data):
            xs.append(data["memory_gb"])
            ys.append(data["throughput_tps"])
            labels.append(cfg)
    if xs:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(xs, ys, s=120, c=["#A94442", "#C26A1A", "#2E7D4F", "#2C5F8A"][: len(xs)], zorder=3)
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(8, 4), fontsize=10)
        ax.set_xlabel("Peak memory (GB)")
        ax.set_ylabel("Decode tok/s")
        ax.set_title("Llama 3.1 8B — memory vs speed Pareto (Mac M3)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "01_pareto_memory_speed.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        _emit(out / "01_pareto_memory_speed.png")

    # Speedup vs fp16
    base = _load(hw_dir / "article_01_weight-quantization" / "llama3-8b" / "fp16.json")
    if _ok(base):
        labs, speedups = [], []
        for cfg in ("w8", "w4", "w2"):
            data = _load(hw_dir / "article_01_weight-quantization" / "llama3-8b" / f"{cfg}.json")
            if _ok(data):
                labs.append(cfg)
                speedups.append(data["throughput_tps"] / base["throughput_tps"])
        if labs:
            _bar_chart(
                labs,
                speedups,
                title="Decode speedup vs fp16 — Llama 3.1 8B (Mac M3)",
                xlabel="Speedup (×)",
                output=out / "01_speedup_vs_fp16.png",
                color="#2E7D4F",
            )

    # Context: TTFT + tok/s dual axis
    base7 = hw_dir / "article_07_context-and-cache" / "llama3-8b"
    ps, ttfts, tpss = [], [], []
    for p, fname in [(256, "ctx_p256.json"), (512, "ctx_p512.json"),
                     (1024, "ctx_p1024.json"), (2048, "ctx_p2048.json")]:
        data = _load(base7 / fname)
        if _ok(data):
            ps.append(p)
            ttfts.append(data["ttft_ms"] / 1000)
            tpss.append(data["throughput_tps"])
    if ps:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(ps, ttfts, "o-", color="#C26A1A", lw=2, label="TTFT (s)")
        ax1.set_xlabel("Prompt tokens")
        ax1.set_ylabel("TTFT (seconds)", color="#C26A1A")
        ax2 = ax1.twinx()
        ax2.plot(ps, tpss, "s--", color="#2C5F8A", lw=2, label="tok/s")
        ax2.set_ylabel("Decode tok/s", color="#2C5F8A")
        ax1.set_title("Context length: TTFT explodes, decode slows (Mac M3)")
        ax1.grid(True, alpha=0.3)
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")
        fig.tight_layout()
        fig.savefig(out / "07_context_dual_axis.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        _emit(out / "07_context_dual_axis.png")

    # Workload bar chart
    workloads = [
        ("chat_light", "wl_chat_light.json"),
        ("chat_std", "wl_chat_standard.json"),
        ("code", "wl_complete_code.json"),
        ("summarize", "wl_summarize_long.json"),
        ("rag_agent", "wl_rag_agent.json"),
    ]
    wlabels, wttft = [], []
    for name, fname in workloads:
        data = _load(base7 / fname)
        if _ok(data):
            wlabels.append(name)
            wttft.append(data["ttft_ms"] / 1000)
    if wlabels:
        _bar_chart(
            wlabels,
            wttft,
            title="Workload stress — TTFT in seconds (Llama 8B, Mac M3)",
            xlabel="TTFT (s)",
            output=out / "07_workload_ttft.png",
            color="#A94442",
            horizontal=True,
        )

    # Generation length
    gens = [("g=64", "gen_g64.json"), ("g=256", "gen_g256.json"), ("g=512", "gen_g512.json")]
    glabels, gtps = [], []
    for name, fname in gens:
        data = _load(base7 / fname)
        if _ok(data):
            glabels.append(name)
            gtps.append(data["throughput_tps"])
    if glabels:
        _bar_chart(
            glabels,
            gtps,
            title="Generation length vs decode tok/s (w4+kv_cache)",
            xlabel="tok/s",
            output=out / "07_generation_length.png",
            color="#6B3FA0",
        )

    # Full stack mistral + llama
    rows = []
    for preset, fp, opt in [
        ("llama3-8b", "fp16.json", "optimized.json"),
        ("mistral-7b", "fp16_mistral.json", "optimized_mistral.json"),
    ]:
        a = _load(hw_dir / "article_05_full-stack" / preset / fp)
        b = _load(hw_dir / "article_05_full-stack" / preset / opt)
        if _ok(a) and _ok(b):
            rows.append((preset, a, b))
    if rows:
        groups = [r[0] for r in rows]
        _grouped_bars(
            groups,
            {
                "fp16 tok/s": [r[1]["throughput_tps"] for r in rows],
                "optimized tok/s": [r[2]["throughput_tps"] for r in rows],
            },
            title="Full stack speedup — Llama 8B & Mistral 7B (Mac M3)",
            ylabel="tok/s",
            output=out / "05_full_stack_two_models.png",
        )
        _grouped_bars(
            groups,
            {
                "fp16 GB": [r[1]["memory_gb"] for r in rows],
                "optimized GB": [r[2]["memory_gb"] for r in rows],
            },
            title="Full stack memory — Llama 8B & Mistral 7B (Mac M3)",
            ylabel="Peak GB",
            output=out / "05_full_stack_memory.png",
        )

    # Model ladder scatter mem vs tps
    ladder = [
        ("qwen-0.5b", "0.5B"),
        ("llama-3.2-1b", "1B"),
        ("qwen-1.5b", "1.5B"),
        ("qwen-3b", "3B"),
        ("phi-3-mini", "Phi-3"),
        ("mistral-7b", "7B"),
        ("llama3-8b", "8B"),
        ("gemma-9b", "9B"),
    ]
    mx, my, ml = [], [], []
    for preset, short in ladder:
        data = _load(hw_dir / "article_04_model-size-ladder" / preset / "w4.json")
        if _ok(data):
            mx.append(data["memory_gb"])
            my.append(data["throughput_tps"])
            ml.append(short)
    if mx:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(mx, my, s=100, c="#2C5F8A", zorder=3)
        for x, y, lab in zip(mx, my, ml):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("Peak memory (GB) @ w4")
        ax.set_ylabel("Decode tok/s")
        ax.set_title("Model size ladder — memory vs speed (Mac M3, w4)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "04_ladder_scatter.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        _emit(out / "04_ladder_scatter.png")

    # Prefill: TTFT vs prompt as line + quadratic reference
    base3 = hw_dir / "article_03_prefill-ttft" / "llama3-8b"
    pts = []
    for p, fname in [(256, "w4_prefill_p256.json"), (512, "w4_prefill.json"),
                     (1024, "w4_prefill_p1024.json")]:
        data = _load(base3 / fname)
        if _ok(data):
            pts.append((p, data["ttft_ms"]))
    # also article 7 longer
    data2048 = _load(base7 / "ctx_p2048.json")
    if _ok(data2048):
        pts.append((2048, data2048["ttft_ms"]))
    if len(pts) >= 3:
        pts.sort()
        px = [p for p, _ in pts]
        py = [t / 1000 for _, t in pts]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(px, py, "o-", color="#C26A1A", lw=2, label="Measured TTFT (s)")
        # rough quadratic fit relative to first point
        p0, t0 = px[0], py[0]
        quad = [t0 * ((p / p0) ** 2) for p in px]
        ax.plot(px, quad, "--", color="#888", label="∝ T² reference (scaled)")
        ax.set_xlabel("Prompt tokens")
        ax.set_ylabel("TTFT (seconds)")
        ax.set_title("TTFT vs prompt length — quadratic pressure (Mac M3)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "03_ttft_vs_prompt_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        _emit(out / "03_ttft_vs_prompt_curve.png")

    # Speculative: memory + speed for qwen
    preset = "qwen-7b"
    base = _load(hw_dir / "article_06_speculative-decoding" / preset / f"{preset}_w4_baseline.json")
    spec = _load(hw_dir / "article_06_speculative-decoding" / preset / f"{preset}_w4_speculative.json")
    if _ok(base) and _ok(spec):
        _grouped_bars(
            ["baseline", "speculative"],
            {
                "tok/s": [base["throughput_tps"], spec["throughput_tps"]],
                "Peak GB": [base["memory_gb"], spec["memory_gb"]],
            },
            title="Speculative decode — speed & memory (Qwen-7B, Mac M3)",
            ylabel="Value",
            output=out / "06_speculative_speed_memory.png",
        )


def main() -> int:
    global _ARTICLE
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", default="Mac M3")
    parser.add_argument("-o", "--output-dir", type=Path, default=OUT_DIR)
    add_article_arg(parser)
    args = parser.parse_args()
    _ARTICLE = resolve_article(args.article)

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("Install: pip install -r requirements-dev.txt")
        return 1

    hw_dir = RESULTS_DIR / _safe_hw(args.hardware)
    out = args.output_dir
    jobs = [
        ("00-introduction", plot_article_0, True),  # needs hw string
        ("01-weight-quantization", plot_article_1, False),
        ("02-kv-cache-quantization", plot_article_2, False),
        ("03-prefill-and-ttft", plot_article_3, False),
        ("04-model-size-ladder", plot_article_4, False),
        ("05-full-optimization-stack", plot_article_5, False),
        ("06-speculative-decoding", plot_article_6, False),
        ("07-context-and-cache", plot_article_7, False),
    ]
    for slug, fn, needs_hw in jobs:
        if _ARTICLE and slug != _ARTICLE:
            continue
        if needs_hw:
            fn(args.hardware, hw_dir, out)
        else:
            fn(hw_dir, out)
    # extras may feed multiple articles — always run unless filtered emit skips
    if not _ARTICLE or _ARTICLE in {
        "00-introduction",
        "01-weight-quantization",
        "02-kv-cache-quantization",
        "03-prefill-and-ttft",
        "04-model-size-ladder",
        "05-full-optimization-stack",
        "06-speculative-decoding",
        "07-context-and-cache",
    }:
        plot_extra_results(hw_dir, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
