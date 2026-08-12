#!/usr/bin/env python3
"""Paper-style workflow / how-it-works diagrams for Medium articles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "medium" / "images" / "workflows"


def _save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"Wrote {output}")


def _box(ax, xy, w, h, text, *, fc="#E8F1FA", ec="#2C5F8A", fontsize=9, lw=1.5):
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def _arrow(ax, start, end, color="#333333"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.6),
    )


def diagram_unified_memory(out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Unified Memory on Apple Silicon — Why Quantization Matters", fontsize=13, pad=14)

    _box(ax, (0.4, 3.8), 2.2, 1.4, "CPU cores\n(scheduling, I/O)", fc="#DCEFE4", ec="#2E7D4F")
    _box(ax, (3.9, 3.8), 2.2, 1.4, "GPU / Metal\n(matmul, attention)", fc="#FCE8D5", ec="#C26A1A")
    _box(ax, (7.4, 3.8), 2.2, 1.4, "Neural Engine\n(optional accel)", fc="#EDE4F7", ec="#6B3FA0")

    _box(ax, (1.5, 1.4), 7, 1.6, "ONE unified DRAM pool\nweights + KV cache + OS + apps share the same bytes", fc="#FFF3CD", ec="#B8860B", fontsize=10)

    _arrow(ax, (1.5, 3.8), (3.5, 3.0))
    _arrow(ax, (5.0, 3.8), (5.0, 3.0))
    _arrow(ax, (8.5, 3.8), (6.5, 3.0))

    ax.text(
        5,
        0.55,
        "Discrete GPU PC: separate VRAM  ·  Mac: total RAM is the hard ceiling",
        ha="center",
        fontsize=9,
        style="italic",
        color="#444",
    )
    _save(fig, out / "00_unified_memory.png")


def diagram_inference_pipeline(out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Autoregressive Inference Pipeline (What We Measure)", fontsize=13, pad=12)

    steps = [
        (0.3, "Load weights\n(fp16 / w4…)", "#E8F1FA", "#2C5F8A"),
        (2.5, "PREFILL\nprocess all\nprompt tokens", "#FCE8D5", "#C26A1A"),
        (4.7, "First token\n→ TTFT", "#FFF3CD", "#B8860B"),
        (6.9, "DECODE loop\n1 token / step\n(+ KV append)", "#DCEFE4", "#2E7D4F"),
        (9.1, "Stream output\ntok/s", "#EDE4F7", "#6B3FA0"),
    ]
    for x, text, fc, ec in steps:
        _box(ax, (x, 1.3), 1.9, 1.8, text, fc=fc, ec=ec, fontsize=8.5)
    for i in range(len(steps) - 1):
        x0 = steps[i][0] + 1.9
        x1 = steps[i + 1][0]
        _arrow(ax, (x0 + 0.05, 2.2), (x1 - 0.05, 2.2))

    ax.text(3.45, 0.55, "Optimize: prefill chunking, Flash-style kernels", ha="center", fontsize=8, color="#C26A1A")
    ax.text(7.85, 0.55, "Optimize: weight quant, speculative decode, KV quant", ha="center", fontsize=8, color="#2E7D4F")
    _save(fig, out / "00_inference_pipeline.png")


def diagram_affine_quant(out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.35, wspace=0.25)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_xlim(0, 10)
    ax0.set_ylim(0, 3.2)
    ax0.axis("off")
    ax0.set_title("Affine Weight Quantization (Jacob et al. / GPTQ / AWQ family)", fontsize=12, pad=8)
    _box(ax0, (0.3, 0.9), 2.2, 1.6, "FP16 weight\nmatrix W\n~16 GB (8B)", fc="#F8D7DA", ec="#A94442")
    _box(ax0, (3.2, 0.9), 2.4, 1.6, "Per-group scale s\n& zero-point z\nq = round(w/s + z)", fc="#FFF3CD", ec="#B8860B")
    _box(ax0, (6.4, 0.9), 3.2, 1.6, "Packed INT4 codes\n+ tiny metadata\n~5 GB on disk/RAM", fc="#DCEFE4", ec="#2E7D4F")
    _arrow(ax0, (2.5, 1.7), (3.15, 1.7))
    _arrow(ax0, (5.6, 1.7), (6.35, 1.7))
    ax0.text(5, 0.25, "Offline (checkpoint) · at inference: dequantize on-the-fly during matmul", ha="center", fontsize=8.5, style="italic")

    ax1 = fig.add_subplot(gs[1, 0])
    bits = [16, 8, 4, 2]
    mem = [16, 8, 4, 2]
    colors = ["#A94442", "#C26A1A", "#2E7D4F", "#2C5F8A"]
    bars = ax1.bar([str(b) for b in bits], mem, color=colors)
    ax1.set_xlabel("Bits per weight (ideal)")
    ax1.set_ylabel("Weight memory (GB, 8B model)")
    ax1.set_title("Ideal memory scaling")
    for b, m in zip(bars, mem):
        ax1.text(b.get_x() + b.get_width() / 2, m + 0.3, f"{m} GB", ha="center", fontsize=8)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 5)
    ax2.axis("off")
    ax2.set_title("Group-wise packing (intuition)", fontsize=11)
    # one byte with two nibbles
    from matplotlib.patches import Rectangle

    ax2.add_patch(Rectangle((0.5, 2.2), 3.2, 1.6, facecolor="#E8F1FA", edgecolor="#2C5F8A", lw=1.5))
    ax2.plot([2.1, 2.1], [2.2, 3.8], color="#2C5F8A", lw=1.2)
    ax2.text(1.3, 3.0, "q₀\n4-bit", ha="center", va="center", fontsize=9)
    ax2.text(2.9, 3.0, "q₁\n4-bit", ha="center", va="center", fontsize=9)
    ax2.text(2.1, 1.6, "1 byte holds 2 × INT4 codes", ha="center", fontsize=8)
    ax2.text(5.8, 3.2, "Each group of\n64–128 weights\nshares (s, z)", ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="#FFF3CD", edgecolor="#B8860B"))
    ax2.text(5.8, 1.4, "GPTQ: Hessian-aware\nAWQ: protect salient w", ha="center", fontsize=8, color="#555")

    _save(fig, out / "01_affine_quantization.png")


def diagram_decode_bandwidth(out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Why Fewer Bits → Faster Decode (Bandwidth-Bound Intuition)", fontsize=12, pad=10)

    _box(ax, (0.4, 2.8), 2.8, 1.5, "Each decode step\nreads nearly ALL weights\nfrom DRAM → GPU", fc="#F8D7DA", ec="#A94442")
    _box(ax, (3.8, 2.8), 2.4, 1.5, "Memory bandwidth\nis often the ceiling\n(Roofline model)", fc="#FFF3CD", ec="#B8860B")
    _box(ax, (6.8, 2.8), 2.8, 1.5, "w4 moves ~4× fewer\nbytes per token\n→ higher tok/s", fc="#DCEFE4", ec="#2E7D4F")
    _arrow(ax, (3.2, 3.55), (3.75, 3.55))
    _arrow(ax, (6.2, 3.55), (6.75, 3.55))

    ax.text(5, 1.6, "fp16: ~16 GB / step   ·   w4: ~5 GB / step   ·   same matmul shape", ha="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#E8F1FA", edgecolor="#2C5F8A"))
    ax.text(5, 0.6, "Williams et al. Roofline (2009): when arithmetic intensity is low, bandwidth wins", ha="center", fontsize=8, style="italic", color="#555")
    _save(fig, out / "01_bandwidth_bound.png")


def diagram_kv_growth(out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(11, 5.5))
    gs = fig.add_gridspec(1, 2, wspace=0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_xlim(0, 10)
    ax0.set_ylim(0, 8)
    ax0.axis("off")
    ax0.set_title("KV Cache During Decode", fontsize=11)
    _box(ax0, (0.5, 5.5), 4, 1.8, "Prefill: build K,V for\nentire prompt (T_prompt)", fc="#FCE8D5", ec="#C26A1A")
    _box(ax0, (0.5, 3.0), 4, 1.8, "Decode step t:\nappend 1 row to K and V\nattend to all past tokens", fc="#E8F1FA", ec="#2C5F8A")
    _box(ax0, (0.5, 0.5), 4, 1.8, "Without cache:\nrecompute K,V every step\n(prohibitively expensive)", fc="#F8D7DA", ec="#A94442")
    _arrow(ax0, (2.5, 5.5), (2.5, 4.85))
    _arrow(ax0, (2.5, 3.0), (2.5, 2.35))

    ax1 = fig.add_subplot(gs[0, 1])
    T = np.array([256, 512, 1024, 2048, 4096, 8192])
    # rough MB for Llama-8B style: 2*32*8*T*128*2 / 1e6 ≈ T * 0.131
    fp16 = T * 0.131
    kv4 = fp16 / 4
    ax1.plot(T, fp16, "o-", color="#A94442", label="FP16 KV (~MB)")
    ax1.plot(T, kv4, "s-", color="#2E7D4F", label="4-bit KV (~MB)")
    ax1.set_xlabel("Total tokens T (prompt + gen)")
    ax1.set_ylabel("KV memory (MB, rough)")
    ax1.set_title("Linear growth · 4-bit ≈ ¼ size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(2048, color="#B8860B", ls="--", alpha=0.7)
    ax1.text(2100, max(fp16) * 0.55, "RAG zone", color="#B8860B", fontsize=8)

    fig.suptitle("KV Cache: Why Long Context Has a Second Memory Bill", fontsize=13, y=1.02)
    _save(fig, out / "02_kv_cache_workflow.png")


def diagram_attention_kv(out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Scaled Dot-Product Attention + Cached K/V (Decode Step)", fontsize=12, pad=10)

    _box(ax, (0.3, 2.8), 1.8, 1.5, "New token\nhidden state", fc="#EDE4F7", ec="#6B3FA0")
    _box(ax, (2.6, 3.5), 1.5, 1.0, "Q", fc="#E8F1FA", ec="#2C5F8A")
    _box(ax, (2.6, 2.0), 1.5, 1.0, "K_new", fc="#E8F1FA", ec="#2C5F8A")
    _box(ax, (2.6, 0.5), 1.5, 1.0, "V_new", fc="#E8F1FA", ec="#2C5F8A")
    _arrow(ax, (2.1, 3.6), (2.55, 4.0))
    _arrow(ax, (2.1, 3.3), (2.55, 2.5))
    _arrow(ax, (2.1, 3.0), (2.55, 1.0))

    _box(ax, (4.6, 1.6), 2.2, 2.6, "KV CACHE\n[K_1…K_t]\n[V_1…V_t]\n\nappend K_new,V_new", fc="#FFF3CD", ec="#B8860B", fontsize=8.5)
    _arrow(ax, (4.1, 2.5), (4.55, 2.5))
    _arrow(ax, (4.1, 1.0), (4.55, 2.0))

    _box(ax, (7.4, 2.0), 2.3, 1.8, "softmax(QKᵀ/√d) V\n→ next layer\n→ logits", fc="#DCEFE4", ec="#2E7D4F")
    _arrow(ax, (6.85, 2.9), (7.35, 2.9))
    ax.text(5, 0.25, "Quantize cache entries (b_kv=4) independently of weight bits", ha="center", fontsize=9, style="italic", color="#555")
    _save(fig, out / "02_attention_with_cache.png")


def diagram_gqa(out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, title, n_q, n_kv, note in [
        (axes[0], "Multi-Head Attention", 8, 8, "Every Q head has its own K/V\n→ large KV cache"),
        (axes[1], "Grouped-Query Attention (GQA)", 8, 2, "Many Q heads share K/V\n→ smaller cache (Llama 3)"),
    ]:
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 6)
        ax.axis("off")
        ax.set_title(title, fontsize=11)
        for i in range(n_q):
            y = 5 - i * 0.55
            _box(ax, (0.5, y - 0.35), 1.4, 0.45, f"Q{i}", fc="#E8F1FA", ec="#2C5F8A", fontsize=7)
        for i in range(n_kv):
            y = 4.5 - i * (3.5 / max(n_kv, 1))
            _box(ax, (4.5, y - 0.4), 1.6, 0.7, f"KV{i}", fc="#FCE8D5", ec="#C26A1A", fontsize=8)
            # connect some Qs
            step = n_q // n_kv
            for j in range(step):
                qi = i * step + j
                if qi < n_q:
                    yq = 5 - qi * 0.55
                    ax.plot([1.9, 4.5], [yq, y], color="#888", lw=0.8, alpha=0.7)
        ax.text(4, 0.4, note, ha="center", fontsize=8, color="#444")
    fig.suptitle("GQA Shrinks KV Heads Before You Even Quantize (Ainslie et al., 2023)", fontsize=12)
    fig.tight_layout()
    _save(fig, out / "02_gqa_vs_mha.png")


def diagram_prefill_vs_decode(out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Prefill vs Decode — Two Different Bottlenecks", fontsize=13, pad=10)

    # timeline bar
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((0.5, 2.2), 5.5, 1.4, boxstyle="round,pad=0.02", facecolor="#FCE8D5", edgecolor="#C26A1A", lw=1.5))
    ax.text(3.25, 2.9, "PREFILL  (all prompt tokens at once)\nCost ~ O(T²) attention  ·  Metric: TTFT", ha="center", va="center", fontsize=9)

    ax.add_patch(FancyBboxPatch((6.3, 2.2), 5.2, 1.4, boxstyle="round,pad=0.02", facecolor="#DCEFE4", edgecolor="#2E7D4F", lw=1.5))
    ax.text(8.9, 2.9, "DECODE  (one token at a time)\nCost ~ weight bandwidth  ·  Metric: tok/s", ha="center", va="center", fontsize=9)

    _arrow(ax, (6.0, 2.9), (6.25, 2.9))
    ax.text(6.15, 3.85, "first token", ha="center", fontsize=8, color="#B8860B")

    _box(ax, (0.8, 0.4), 4.8, 1.2, "Optimize: FlashAttention-style tiling,\nprefill_step_size chunking, shorter prompts", fc="#FFF3CD", ec="#B8860B", fontsize=8)
    _box(ax, (6.6, 0.4), 4.8, 1.2, "Optimize: w4 weights, speculative decoding,\nKV quant for long generations", fc="#E8F1FA", ec="#2C5F8A", fontsize=8)
    _save(fig, out / "03_prefill_vs_decode.png")


def diagram_flash_attention(out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    ax = axes[0]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Naive attention (materialize N×N)", fontsize=11)
    ax.add_patch(Rectangle((1, 1), 4, 4, facecolor="#F8D7DA", edgecolor="#A94442", lw=1.5))
    ax.text(3, 3, "Full attention\nmatrix in HBM\nO(N²) memory", ha="center", va="center", fontsize=10)
    ax.text(3, 0.4, "Slow when N is large (long prompt)", ha="center", fontsize=8, color="#555")

    ax = axes[1]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("FlashAttention (tiled, exact)", fontsize=11)
    colors = ["#E8F1FA", "#DCEFE4", "#FFF3CD", "#FCE8D5"]
    for i, (x, y) in enumerate([(1, 3.5), (2.2, 3.5), (1, 2.2), (2.2, 2.2)]):
        ax.add_patch(Rectangle((x, y), 1.0, 1.0, facecolor=colors[i], edgecolor="#2C5F8A", lw=1.2))
    ax.text(4.5, 3.8, "Tiles fit in\nfast SRAM", ha="center", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#DCEFE4", edgecolor="#2E7D4F"))
    ax.text(4.5, 2.4, "Online softmax\nstreams results\n(exact, not approx)", ha="center", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="#FFF3CD", edgecolor="#B8860B"))
    ax.text(3, 0.5, "Dao et al. 2022/23 — IO-aware algorithm, same math", ha="center", fontsize=8, style="italic", color="#555")

    fig.suptitle("FlashAttention Idea: Exact Attention with Better Memory Traffic", fontsize=12)
    fig.tight_layout()
    _save(fig, out / "03_flash_attention.png")


def diagram_model_ladder(out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("What Fits on 24 GB Unified Memory? (Decision Ladder)", fontsize=13, pad=10)

    tiers = [
        (5.2, "#DCEFE4", "#2E7D4F", "Tier A — Instant\n0.5B–1.5B @ w4\n~90–240 tok/s  ·  <2 GB", "Routers, classifiers, draft models"),
        (3.5, "#E8F1FA", "#2C5F8A", "Tier B — Daily driver\n3B–8B @ w4\n~17–48 tok/s  ·  3–6 GB", "Chat, coding, summarization"),
        (1.8, "#FFF3CD", "#B8860B", "Tier C — Pushing it\n9B+ @ w4 / 8B @ w8\n~15–20 tok/s  ·  6–10 GB", "Max quality that still fits"),
        (0.2, "#F8D7DA", "#A94442", "Usually skip on 24 GB\n8B+ @ fp16 as daily driver", "Fits but leaves no headroom + slow"),
    ]
    for y, fc, ec, title, note in tiers:
        ax.add_patch(FancyBboxPatch((0.5, y), 6.5, 1.4, boxstyle="round,pad=0.02", facecolor=fc, edgecolor=ec, lw=1.5))
        ax.text(3.75, y + 0.7, title, ha="center", va="center", fontsize=9)
        ax.text(8.5, y + 0.7, note, ha="center", va="center", fontsize=8, color="#444",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="#ccc"))
    _save(fig, out / "04_fit_ladder.png")


def diagram_opt_stack(out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Stacking Optimizations — Funnel from Slow/Heavy → Fast/Lean", fontsize=12, pad=10)

    layers = [
        (6.2, 7.5, "#F8D7DA", "#A94442", "fp16 baseline\n~16 GB · ~5–6 tok/s"),
        (5.0, 6.2, "#FCE8D5", "#C26A1A", "+ weight quant (w4)\n~5 GB · ~20 tok/s"),
        (3.8, 4.9, "#FFF3CD", "#B8860B", "+ KV cache quant\nlong-context insurance"),
        (2.6, 3.6, "#E8F1FA", "#2C5F8A", "+ prefill tuning\nbetter TTFT at long p"),
        (1.4, 2.3, "#DCEFE4", "#2E7D4F", "Recipe: w4+kv_cache+prefill\n~5 GB · ~18–21 tok/s"),
    ]
    for y, w, fc, ec, text in layers:
        x = (9 - w) / 2
        ax.add_patch(FancyBboxPatch((x, y), w, 1.1, boxstyle="round,pad=0.02", facecolor=fc, edgecolor=ec, lw=1.5))
        ax.text(4.5, y + 0.55, text, ha="center", va="center", fontsize=9)
    ax.text(4.5, 0.6, "Optional next lever: speculative decoding (draft model) → Part 7", ha="center", fontsize=9, style="italic", color="#555")
    _save(fig, out / "05_optimization_funnel.png")


def diagram_decision_tree(out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("When to Enable Which Optimization", fontsize=13, pad=10)

    _box(ax, (4.0, 5.5), 3.0, 1.0, "Goal?", fc="#EDE4F7", ec="#6B3FA0", fontsize=10)
    _box(ax, (0.4, 3.5), 2.6, 1.2, "Fit in RAM /\nfaster decode", fc="#E8F1FA", ec="#2C5F8A")
    _box(ax, (4.0, 3.5), 3.0, 1.2, "Lower TTFT /\nlong prompts", fc="#FCE8D5", ec="#C26A1A")
    _box(ax, (8.0, 3.5), 2.6, 1.2, "Long generation\nspeed", fc="#DCEFE4", ec="#2E7D4F")
    _arrow(ax, (4.5, 5.5), (1.7, 4.75))
    _arrow(ax, (5.5, 5.5), (5.5, 4.75))
    _arrow(ax, (6.5, 5.5), (9.3, 4.75))

    _box(ax, (0.3, 1.5), 2.8, 1.3, "→ w4 weights\n(+ KV if ctx long)", fc="#FFF3CD", ec="#B8860B", fontsize=8.5)
    _box(ax, (4.0, 1.5), 3.0, 1.3, "→ prefill ON\nshorter prompts\nprefix cache", fc="#FFF3CD", ec="#B8860B", fontsize=8.5)
    _box(ax, (8.0, 1.5), 2.6, 1.3, "→ speculative\ndecode", fc="#FFF3CD", ec="#B8860B", fontsize=8.5)
    _arrow(ax, (1.7, 3.5), (1.7, 2.85))
    _arrow(ax, (5.5, 3.5), (5.5, 2.85))
    _arrow(ax, (9.3, 3.5), (9.3, 2.85))
    ax.text(5.5, 0.5, "Default daily driver on 24 GB Mac: w4+kv_cache+prefill", ha="center", fontsize=9, style="italic")
    _save(fig, out / "05_decision_tree.png")


def diagram_speculative(out: Path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    ax0.set_xlim(0, 12)
    ax0.set_ylim(0, 4)
    ax0.axis("off")
    ax0.set_title("Baseline Decode: 1 Target Forward Pass per Token", fontsize=11)
    for i in range(6):
        x = 0.5 + i * 1.9
        _box(ax0, (x, 1.2), 1.5, 1.5, f"target\n→ tok {i+1}", fc="#F8D7DA", ec="#A94442", fontsize=8)
        if i < 5:
            _arrow(ax0, (x + 1.5, 1.95), (x + 1.85, 1.95))
    ax0.text(6, 0.4, "Throughput ≈ 1 / τ_target", ha="center", fontsize=9, style="italic")

    ax1 = fig.add_subplot(gs[1])
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 5)
    ax1.axis("off")
    ax1.set_title("Speculative Decode: Draft Proposes k Tokens → Target Verifies Once (Leviathan / Chen)", fontsize=11)
    _box(ax1, (0.4, 2.5), 2.4, 1.8, "Draft model\n(0.5B–1B)\npropose k=3", fc="#E8F1FA", ec="#2C5F8A")
    _box(ax1, (3.5, 2.5), 2.8, 1.8, "Target verify\n1 parallel pass\nover k tokens", fc="#FCE8D5", ec="#C26A1A")
    _box(ax1, (7.0, 2.5), 2.4, 1.8, "Accept m ≤ k\nreject + resample\nat first mismatch", fc="#DCEFE4", ec="#2E7D4F")
    _box(ax1, (9.8, 2.5), 1.8, 1.8, "α accept\nrate", fc="#FFF3CD", ec="#B8860B")
    _arrow(ax1, (2.85, 3.4), (3.45, 3.4))
    _arrow(ax1, (6.35, 3.4), (6.95, 3.4))
    _arrow(ax1, (9.45, 3.4), (9.75, 3.4))
    ax1.text(6, 1.2, "High α (same family draft/target) → 1.5–2× tok/s without retraining", ha="center", fontsize=9)
    ax1.text(6, 0.4, "Quality unchanged: rejected drafts never appear in the output", ha="center", fontsize=8, style="italic", color="#555")
    _save(fig, out / "06_speculative_workflow.png")


def diagram_accept_timeline(out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("One Speculative Round (k=3) — Accept / Reject", fontsize=12, pad=8)

    # draft tokens
    for i, (label, ok) in enumerate([("d1 ✓", True), ("d2 ✓", True), ("d3 ✗", False)]):
        fc = "#DCEFE4" if ok else "#F8D7DA"
        ec = "#2E7D4F" if ok else "#A94442"
        _box(ax, (0.5 + i * 1.6, 2.0), 1.4, 1.2, label, fc=fc, ec=ec, fontsize=10)
    ax.text(2.9, 1.3, "draft proposals", ha="center", fontsize=8, color="#555")

    _box(ax, (5.8, 1.8), 3.5, 1.6, "Emit accepted tokens\nthen sample from target\nat rejection point", fc="#FFF3CD", ec="#B8860B", fontsize=9)
    _arrow(ax, (5.1, 2.6), (5.75, 2.6))
    ax.text(5, 0.5, "Acceptance rate α = accepted_draft_tokens / total_generated_tokens", ha="center", fontsize=9, style="italic")
    _save(fig, out / "06_accept_reject.png")


def diagram_prefix_cache(out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Cold turn (no reuse)", fontsize=11)
    _box(ax, (0.5, 4.0), 5, 1.2, "System prefix (256 tok) + user", fc="#F8D7DA", ec="#A94442")
    _box(ax, (0.5, 2.2), 5, 1.2, "Full prefill every request", fc="#FCE8D5", ec="#C26A1A")
    _box(ax, (0.5, 0.5), 5, 1.2, "TTFT = cost(prefix + user)", fc="#FFF3CD", ec="#B8860B")
    _arrow(ax, (3, 4.0), (3, 3.45))
    _arrow(ax, (3, 2.2), (3, 1.75))

    ax = axes[1]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Warm turn (prefix KV cached)", fontsize=11)
    _box(ax, (0.5, 4.0), 5, 1.2, "Load cached KV for system prefix", fc="#DCEFE4", ec="#2E7D4F")
    _box(ax, (0.5, 2.2), 5, 1.2, "Prefill only user suffix", fc="#E8F1FA", ec="#2C5F8A")
    _box(ax, (0.5, 0.5), 5, 1.2, "TTFT ≈ cost(user) ≪ cold", fc="#FFF3CD", ec="#B8860B")
    _arrow(ax, (3, 4.0), (3, 3.45))
    _arrow(ax, (3, 2.2), (3, 1.75))

    fig.suptitle("Prefix Prompt Cache — Skip Re-Prefilling Stable Instructions", fontsize=12)
    fig.tight_layout()
    _save(fig, out / "07_prefix_cache_workflow.png")


def diagram_rag_wall(out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("The RAG Wall — Why Pasting a PDF Feels Broken Locally", fontsize=12, pad=10)

    _box(ax, (0.3, 2.8), 2.2, 1.6, "Retrieve\nchunks", fc="#EDE4F7", ec="#6B3FA0")
    _box(ax, (3.0, 2.8), 2.4, 1.6, "Stuff into\nprompt (2K+)", fc="#FCE8D5", ec="#C26A1A")
    _box(ax, (5.9, 2.8), 1.8, 1.6, "Prefill\nO(T²)", fc="#F8D7DA", ec="#A94442")
    _box(ax, (8.2, 2.8), 1.5, 1.6, "15–30s\nTTFT", fc="#FFF3CD", ec="#B8860B")
    for a, b in [(2.5, 3.0), (5.4, 5.9), (7.7, 8.2)]:
        _arrow(ax, (a, 3.6), (b, 3.6))

    ax.text(5, 1.5, "Mitigations: fewer chunks · prefix cache · w4+kv · smaller router model · speculative decode", ha="center", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#DCEFE4", edgecolor="#2E7D4F"))
    ax.text(5, 0.5, "Measured: rag_agent workload ≈ 31s TTFT on Llama 8B @ M3", ha="center", fontsize=8, style="italic", color="#555")
    _save(fig, out / "07_rag_wall.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("Install: pip install -r requirements-dev.txt")
        return 1

    out = args.output_dir
    diagram_unified_memory(out)
    diagram_inference_pipeline(out)
    diagram_affine_quant(out)
    diagram_decode_bandwidth(out)
    diagram_kv_growth(out)
    diagram_attention_kv(out)
    diagram_gqa(out)
    diagram_prefill_vs_decode(out)
    diagram_flash_attention(out)
    diagram_model_ladder(out)
    diagram_opt_stack(out)
    diagram_decision_tree(out)
    diagram_speculative(out)
    diagram_accept_timeline(out)
    diagram_prefix_cache(out)
    diagram_rag_wall(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
