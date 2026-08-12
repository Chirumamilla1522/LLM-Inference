#!/usr/bin/env python3
"""
Original redraws of paper *ideas* for Medium articles.

These are NOT copies of published figures. Each diagram is drawn from scratch
to teach the concept, with attribution in the title/caption:

  "Original redraw — idea from Author et al. (YEAR)"

Safe practice: explain algorithms + cite papers; use our own visuals.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from medium_image_layout import SOURCE, add_article_arg, emit_file, resolve_article, source_keys_for_article  # noqa: E402

OUT_DIR = SOURCE / "papers"
_ARTICLE: str | None = None


def _save(fig, output: Path) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    src_key = f"papers/{output.name}"
    if _ARTICLE and src_key not in source_keys_for_article(_ARTICLE):
        print(f"skip {src_key} (not used by {_ARTICLE})")
        return
    emit_file(src_key, output, article=_ARTICLE)


def _box(ax, xy, w, h, text, *, fc="#E8F1FA", ec="#2C5F8A", fontsize=8.5, lw=1.4):
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, start, end, color="#333"):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=color, lw=1.5))


def _credit(ax, text: str, y: float = 0.08) -> None:
    ax.text(
        0.5,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        fontsize=7.5,
        style="italic",
        color="#666",
    )


# --- Vaswani et al. 2017: Attention Is All You Need ---
def redraw_transformer_attention(out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Scaled Dot-Product Attention — original redraw", fontsize=12, pad=10)

    _box(ax, (0.4, 3.5), 1.6, 1.4, "Input\nhidden\nstates", fc="#EDE4F7", ec="#6B3FA0")
    _box(ax, (2.5, 4.5), 1.4, 0.9, "Q", fc="#E8F1FA", ec="#2C5F8A")
    _box(ax, (2.5, 3.2), 1.4, 0.9, "K", fc="#E8F1FA", ec="#2C5F8A")
    _box(ax, (2.5, 1.9), 1.4, 0.9, "V", fc="#E8F1FA", ec="#2C5F8A")
    _arrow(ax, (2.0, 4.4), (2.45, 4.95))
    _arrow(ax, (2.0, 4.0), (2.45, 3.65))
    _arrow(ax, (2.0, 3.7), (2.45, 2.35))

    _box(ax, (4.5, 3.5), 2.2, 1.4, "scores =\nQ Kᵀ / √d", fc="#FFF3CD", ec="#B8860B")
    _box(ax, (7.2, 3.5), 2.3, 1.4, "softmax(scores) V\n→ context vector", fc="#DCEFE4", ec="#2E7D4F")
    _arrow(ax, (3.95, 4.0), (4.45, 4.2))
    _arrow(ax, (3.95, 2.35), (4.45, 3.8))
    _arrow(ax, (6.75, 4.2), (7.15, 4.2))

    ax.text(5, 1.2, "Autoregressive decode: cache K,V so each new token only builds Q_new", ha="center", fontsize=9)
    _credit(ax, "Original redraw of the attention idea from Vaswani et al., Attention Is All You Need (2017)")
    _save(fig, out / "vaswani_attention_redraw.png")


# --- Williams et al. 2009: Roofline ---
def redraw_roofline(out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 5.5))
    # conceptual roofline
    intensity = np.linspace(0.05, 20, 200)
    peak_flops = 10.0
    bandwidth = 1.5  # slope
    roof = np.minimum(peak_flops, bandwidth * intensity)
    ax.plot(intensity, roof, color="#2C5F8A", lw=2.5)
    ax.fill_between(intensity, 0, roof, alpha=0.08, color="#2C5F8A")
    ax.axhline(peak_flops, color="#A94442", ls="--", lw=1, alpha=0.7)
    ax.text(12, peak_flops + 0.3, "compute roof (peak FLOPS)", color="#A94442", fontsize=8)
    ax.text(0.8, 3.5, "bandwidth\nslope", color="#C26A1A", fontsize=8, rotation=55)

    # mark LLM decode as bandwidth-bound
    ax.scatter([0.4], [0.4 * bandwidth], s=80, color="#2E7D4F", zorder=5)
    ax.annotate(
        "LLM decode\n(low arithmetic intensity)\n→ memory-bound",
        xy=(0.4, 0.4 * bandwidth),
        xytext=(3.5, 2.2),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#2E7D4F"),
        color="#2E7D4F",
    )
    ax.scatter([8], [peak_flops], s=70, color="#6B3FA0", zorder=5)
    ax.annotate(
        "dense GEMM\n(compute-bound)",
        xy=(8, peak_flops),
        xytext=(11, 7),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#6B3FA0"),
        color="#6B3FA0",
    )

    ax.set_xlabel("Arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("Attainable performance (relative)")
    ax.set_title("Roofline intuition — original redraw", fontsize=12)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.grid(True, alpha=0.25)
    fig.text(
        0.5,
        0.01,
        "Original redraw of the Roofline idea from Williams et al. (2009). Why w4 can raise tok/s: fewer bytes moved.",
        ha="center",
        fontsize=7.5,
        style="italic",
        color="#666",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, out / "williams_roofline_redraw.png")


# --- Jacob et al. 2018: affine quantization ---
def redraw_affine_quant_detail(out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(11, 5.5))
    gs = fig.add_gridspec(1, 2, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    x = np.linspace(-3, 3, 400)
    # fake weight-ish distribution
    dens = np.exp(-0.5 * (x / 1.1) ** 2)
    ax0.fill_between(x, dens, alpha=0.35, color="#4C72B0")
    ax0.plot(x, dens, color="#2C5F8A", lw=2)
    # quantization bins
    edges = np.linspace(-2.5, 2.5, 9)
    for e in edges:
        ax0.axvline(e, color="#C26A1A", alpha=0.35, lw=1)
    ax0.set_title("Real weights → discrete levels")
    ax0.set_xlabel("weight value")
    ax0.set_ylabel("density (schematic)")
    ax0.text(0, max(dens) * 0.55, "bins defined by\nscale s & zero-point z", ha="center", fontsize=8,
             bbox=dict(boxstyle="round", facecolor="#FFF3CD", edgecolor="#B8860B"))

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.axis("off")
    ax1.set_title("Affine map (per group)")
    _box(ax1, (0.5, 5.5), 4, 1.5, "q = clip(round(w/s + z),\n0 … 2ᵇ−1)", fc="#E8F1FA", ec="#2C5F8A", fontsize=9)
    _box(ax1, (0.5, 3.2), 4, 1.5, "ŵ = s · (q − z)\ndequantize for matmul", fc="#DCEFE4", ec="#2E7D4F", fontsize=9)
    _box(ax1, (5.2, 3.8), 4.3, 2.5, "Group of 64–128 weights\nshare one (s, z)\n→ tiny metadata\n→ good accuracy/size", fc="#FFF3CD", ec="#B8860B", fontsize=9)
    _arrow(ax1, (2.5, 5.5), (2.5, 4.75))
    ax1.text(5, 1.2, "Used by INT8/INT4 LLM checkpoints (GPTQ/AWQ family)", ha="center", fontsize=8)

    fig.suptitle("Affine quantization — original redraw", fontsize=13, y=1.02)
    fig.text(
        0.5,
        0.01,
        "Original redraw of the affine quant idea from Jacob et al. (2018); LLM PTQ recipes in GPTQ/AWQ.",
        ha="center",
        fontsize=7.5,
        style="italic",
        color="#666",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, out / "jacob_affine_quant_redraw.png")


# --- Frantar GPTQ: column-wise error compensation (conceptual) ---
def redraw_gptq_idea(out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("GPTQ idea — quantize a column, compensate the rest (original redraw)", fontsize=12, pad=8)

    # weight matrix grid
    for i in range(5):
        for j in range(8):
            color = "#F8D7DA" if j == 2 else "#E8F1FA"
            ax.add_patch(Rectangle((1 + j * 0.7, 2.5 + i * 0.45), 0.65, 0.4, facecolor=color, edgecolor="#2C5F8A", lw=0.8))
    ax.text(3.7, 5.2, "Weight matrix W", ha="center", fontsize=9)

    ax.add_patch(Rectangle((1 + 2 * 0.7, 2.5), 0.65, 5 * 0.45, fill=False, edgecolor="#A94442", lw=2.5))
    ax.text(2.55, 1.9, "quantize\nthis column", ha="center", fontsize=8, color="#A94442")

    _box(ax, (7.5, 3.2), 4, 2.2, "Update remaining\nweights to cancel\nquantization error\n(Hessian-aware)", fc="#FFF3CD", ec="#B8860B", fontsize=9)
    _arrow(ax, (6.5, 3.8), (7.45, 4.0))
    ax.text(6, 0.7, "Post-training · no fine-tune required · accurate 3–4 bit weights", ha="center", fontsize=9)
    _credit(ax, "Original redraw of the GPTQ column-compensation idea from Frantar et al. (2022) — not a copy of their figure")
    _save(fig, out / "frantar_gptq_redraw.png")


# --- Lin AWQ: protect salient weights ---
def redraw_awq_idea(out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("AWQ idea — protect activation-salient channels (original redraw)", fontsize=12, pad=8)

    # channels as bars; a few are "salient"
    rng = np.random.default_rng(0)
    heights = rng.uniform(0.4, 1.0, 12)
    heights[[2, 7, 9]] = [1.8, 1.9, 1.7]
    for i, h in enumerate(heights):
        salient = i in (2, 7, 9)
        ax.bar(1.2 + i * 0.55, h, width=0.45, bottom=1.5, color="#C26A1A" if salient else "#4C72B0", edgecolor="white")
    ax.text(4.3, 3.7, "activation magnitude per channel", ha="center", fontsize=8, color="#555")
    _box(ax, (0.8, 0.35), 4.2, 0.9, "Salient channels (orange):\nkeep higher precision / scale care", fc="#FCE8D5", ec="#C26A1A", fontsize=8)
    _box(ax, (5.5, 0.35), 3.8, 0.9, "Other channels:\naggressive 4-bit OK", fc="#E8F1FA", ec="#2C5F8A", fontsize=8)
    _box(ax, (5.5, 2.0), 3.8, 1.6, "Result: better accuracy\nat same bit-width\nwithout training", fc="#DCEFE4", ec="#2E7D4F", fontsize=9)
    _credit(ax, "Original redraw of the AWQ salience idea from Lin et al. (2023) — illustrative, not their figure")
    _save(fig, out / "lin_awq_redraw.png")


# --- Dao FlashAttention: HBM vs SRAM tiling ---
def redraw_flashattention(out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyBboxPatch

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Naive attention IO", fontsize=11)
    ax.add_patch(FancyBboxPatch((0.5, 5.2), 5, 1.2, boxstyle="round", facecolor="#F8D7DA", edgecolor="#A94442"))
    ax.text(3, 5.8, "HBM (slow, large)\nmaterialize full N×N scores", ha="center", va="center", fontsize=9)
    ax.add_patch(Rectangle((1.5, 1.5), 3, 3, facecolor="#FCE8D5", edgecolor="#C26A1A", lw=1.5))
    ax.text(3, 3, "Attention\nmatrix\nO(N²)", ha="center", va="center", fontsize=10)
    ax.text(3, 0.6, "Many HBM reads/writes", ha="center", fontsize=8, color="#A94442")

    ax = axes[1]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("FlashAttention-style tiling", fontsize=11)
    ax.add_patch(FancyBboxPatch((0.5, 5.2), 5, 1.2, boxstyle="round", facecolor="#E8F1FA", edgecolor="#2C5F8A"))
    ax.text(3, 5.8, "HBM holds Q,K,V + output\n(not the full score matrix)", ha="center", va="center", fontsize=9)
    ax.add_patch(FancyBboxPatch((0.8, 2.8), 4.4, 1.8, boxstyle="round", facecolor="#DCEFE4", edgecolor="#2E7D4F"))
    ax.text(3, 3.7, "SRAM (fast, small)\nprocess Q/K/V tiles", ha="center", va="center", fontsize=9)
    # tiles
    for i, (x, y) in enumerate([(1.2, 1.2), (2.4, 1.2), (3.6, 1.2)]):
        ax.add_patch(Rectangle((x, y), 0.9, 0.9, facecolor=["#FFF3CD", "#FCE8D5", "#EDE4F7"][i], edgecolor="#333", lw=1))
        ax.text(x + 0.45, y + 0.45, f"T{i+1}", ha="center", va="center", fontsize=8)
    ax.text(3, 0.5, "Exact softmax via online stats — same math, better IO", ha="center", fontsize=8, color="#2E7D4F")

    fig.suptitle("FlashAttention IO pattern — original redraw", fontsize=13)
    fig.text(
        0.5,
        0.01,
        "Original redraw of the tiled-attention IO idea from Dao et al., FlashAttention (2022) / FlashAttention-2 (2023)",
        ha="center",
        fontsize=7.5,
        style="italic",
        color="#666",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, out / "dao_flashattention_redraw.png")


# --- Milakov online softmax ---
def redraw_online_softmax(out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    ax.set_title("Online softmax — stream tiles without storing all scores (original redraw)", fontsize=11, pad=8)

    for i, lab in enumerate(["tile 1", "tile 2", "tile 3", "…"]):
        _box(ax, (0.4 + i * 1.7, 2.6), 1.4, 1.0, lab, fc="#E8F1FA", ec="#2C5F8A", fontsize=8)
        if i < 3:
            _arrow(ax, (1.8 + i * 1.7, 3.1), (2.05 + i * 1.7, 3.1))
    _box(ax, (7.2, 2.4), 2.4, 1.4, "running\nmax + sum", fc="#FFF3CD", ec="#B8860B", fontsize=9)
    _arrow(ax, (6.9, 3.1), (7.15, 3.1))
    _box(ax, (3.5, 0.5), 3, 1.2, "stable softmax output\nwithout full vector in SRAM", fc="#DCEFE4", ec="#2E7D4F", fontsize=9)
    _arrow(ax, (8.4, 2.4), (5.5, 1.75))
    _credit(ax, "Original redraw of the streaming-softmax idea used by FlashAttention; see Milakov & Gimelshein (2018)")
    _save(fig, out / "milakov_online_softmax_redraw.png")


# --- Leviathan speculative decoding ---
def redraw_leviathan_speculative(out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.35})

    ax = axes[0]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    ax.set_title("Baseline: one expensive target forward per token", fontsize=11)
    for i in range(5):
        _box(ax, (0.5 + i * 2.2, 1.0), 1.8, 1.4, f"target\n→ token {i+1}", fc="#F8D7DA", ec="#A94442", fontsize=8)
        if i < 4:
            _arrow(ax, (2.3 + i * 2.2, 1.7), (2.65 + i * 2.2, 1.7))

    ax = axes[1]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    ax.set_title("Speculative: cheap draft proposes k tokens · target verifies in parallel", fontsize=11)
    _box(ax, (0.4, 2.0), 2.6, 1.8, "Draft model\npropose\nt1 t2 t3", fc="#E8F1FA", ec="#2C5F8A", fontsize=9)
    _box(ax, (3.6, 2.0), 3.0, 1.8, "Target model\n1 forward over\nthe draft prefix", fc="#FCE8D5", ec="#C26A1A", fontsize=9)
    _box(ax, (7.2, 2.0), 2.4, 1.8, "Accept longest\nvalid prefix\n(rej. sample)", fc="#DCEFE4", ec="#2E7D4F", fontsize=9)
    _box(ax, (10.0, 2.0), 1.6, 1.8, "α\naccept\nrate", fc="#FFF3CD", ec="#B8860B", fontsize=9)
    _arrow(ax, (3.05, 2.9), (3.55, 2.9))
    _arrow(ax, (6.65, 2.9), (7.15, 2.9))
    _arrow(ax, (9.65, 2.9), (9.95, 2.9))
    ax.text(6, 0.6, "Same output distribution as target-alone sampling (under the paper’s acceptance rule)", ha="center", fontsize=8, style="italic")

    fig.suptitle("Speculative decoding — original redraw", fontsize=13, y=0.98)
    fig.text(
        0.5,
        0.01,
        "Original redraw of the draft/verify idea from Leviathan et al. (2023) and speculative sampling (Chen et al., 2023)",
        ha="center",
        fontsize=7.5,
        style="italic",
        color="#666",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    _save(fig, out / "leviathan_speculative_redraw.png")


# --- Ainslie GQA ---
def redraw_gqa(out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, title, n_q, n_kv, note, fc in [
        (axes[0], "Multi-Head (MHA)", 6, 6, "1 KV head per Q head\n→ large cache", "#F8D7DA"),
        (axes[1], "Grouped-Query (GQA)", 6, 2, "Many Q heads share KV\n→ smaller cache", "#DCEFE4"),
    ]:
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 7)
        ax.axis("off")
        ax.set_title(title, fontsize=11)
        for i in range(n_q):
            y = 5.8 - i * 0.7
            _box(ax, (0.4, y - 0.25), 1.5, 0.5, f"Q{i}", fc="#E8F1FA", ec="#2C5F8A", fontsize=7)
        for i in range(n_kv):
            y = 5.2 - i * (3.8 / max(n_kv, 1))
            _box(ax, (4.2, y - 0.35), 1.8, 0.7, f"KV{i}", fc=fc, ec="#C26A1A", fontsize=8)
            step = n_q // n_kv
            for j in range(step):
                qi = i * step + j
                if qi < n_q:
                    yq = 5.8 - qi * 0.7
                    ax.plot([1.9, 4.2], [yq, y], color="#888", lw=0.9, alpha=0.75)
        ax.text(3.5, 0.5, note, ha="center", fontsize=8, color="#444")
    fig.suptitle("GQA shrinks KV heads — original redraw", fontsize=12)
    fig.text(
        0.5,
        0.01,
        "Original redraw of GQA vs MHA from the idea in Ainslie et al. (2023). Used by Llama 3, Mistral, Qwen, …",
        ha="center",
        fontsize=7.5,
        style="italic",
        color="#666",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, out / "ainslie_gqa_redraw.png")


# --- Kwon PagedAttention (conceptual) ---
def redraw_paged_attention(out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("PagedAttention idea — KV in non-contiguous blocks (original redraw)", fontsize=12, pad=8)

    _box(ax, (0.4, 3.5), 2.8, 1.8, "Logical KV\nper request\n(contiguous view)", fc="#E8F1FA", ec="#2C5F8A", fontsize=9)
    _box(ax, (4.0, 3.5), 2.4, 1.8, "Block table\nmaps pages", fc="#FFF3CD", ec="#B8860B", fontsize=9)
    # physical blocks scattered
    positions = [(7.0, 4.6), (8.2, 4.6), (7.0, 3.5), (8.2, 3.5), (7.6, 2.4)]
    for i, (x, y) in enumerate(positions):
        ax.add_patch(Rectangle((x, y), 1.0, 0.8, facecolor="#DCEFE4", edgecolor="#2E7D4F", lw=1.2))
        ax.text(x + 0.5, y + 0.4, f"p{i}", ha="center", va="center", fontsize=8)
    ax.text(8.0, 1.8, "Physical GPU/unified memory", ha="center", fontsize=8, color="#555")
    _arrow(ax, (3.25, 4.4), (3.95, 4.4))
    _arrow(ax, (6.45, 4.4), (6.95, 4.9))
    ax.text(5, 0.7, "Cuts fragmentation for multi-request serving (vLLM). Local single-user MLX is simpler — same KV math.", ha="center", fontsize=8)
    _credit(ax, "Original redraw of the paged-KV idea from Kwon et al., Efficient Memory Management… / PagedAttention (2023)")
    _save(fig, out / "kwon_paged_attention_redraw.png")


# --- Pope KV scaling ---
def redraw_pope_kv_scaling(out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 5))
    T = np.array([512, 1024, 2048, 4096, 8192, 16384])
    # schematic: weights fixed, KV grows
    weights = np.full_like(T, 5.0, dtype=float)  # ~5GB w4 8B
    kv = T / 512 * 0.08  # schematic GB
    ax.plot(T, weights, "o-", color="#4C72B0", lw=2, label="Weights (fixed @ w4)")
    ax.plot(T, kv, "s-", color="#C26A1A", lw=2, label="KV cache (grows with T)")
    ax.plot(T, weights + kv, "^-", color="#2E7D4F", lw=2, label="Weights + KV (schematic)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(T)
    ax.set_xticklabels([str(t) for t in T])
    ax.set_xlabel("Sequence length T")
    ax.set_ylabel("Memory (GB, schematic)")
    ax.set_title("When KV rivals weights — original redraw", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(4096, color="#888", ls="--", alpha=0.6)
    ax.text(4200, 4.2, "long-context /\nRAG zone", fontsize=8, color="#555")
    fig.text(
        0.5,
        0.01,
        "Original redraw inspired by KV-scaling discussion in Pope et al., Efficiently Scaling Transformer Inference (2022)",
        ha="center",
        fontsize=7.5,
        style="italic",
        color="#666",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, out / "pope_kv_scaling_redraw.png")


# --- Medusa multi-head draft idea ---
def redraw_medusa_idea(out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Medusa-style idea — extra heads draft multiple tokens (original redraw)", fontsize=11, pad=8)

    _box(ax, (0.5, 2.0), 2.5, 2.0, "Shared\nbackbone", fc="#E8F1FA", ec="#2C5F8A", fontsize=9)
    for i, lab in enumerate(["head+1", "head+2", "head+3"]):
        _box(ax, (3.8, 3.5 - i * 1.1), 2.2, 0.9, lab, fc="#FCE8D5", ec="#C26A1A", fontsize=8)
        _arrow(ax, (3.05, 3.0), (3.75, 3.9 - i * 1.1))
    _box(ax, (6.8, 2.0), 2.7, 2.0, "Verify with\nbase model\n(tree / parallel)", fc="#DCEFE4", ec="#2E7D4F", fontsize=9)
    _arrow(ax, (6.05, 3.0), (6.75, 3.0))
    ax.text(5, 0.6, "Variant of speculative decoding — still draft + verify, different draft mechanism", ha="center", fontsize=8)
    _credit(ax, "Original redraw of the multi-head drafting idea from Cai et al., Medusa (2024) — conceptual only")
    _save(fig, out / "cai_medusa_redraw.png")


def main() -> int:
    global _ARTICLE
    parser = argparse.ArgumentParser(description="Original paper-idea redraws for Medium")
    parser.add_argument("-o", "--output-dir", type=Path, default=OUT_DIR)
    add_article_arg(parser)
    args = parser.parse_args()
    _ARTICLE = resolve_article(args.article)

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("Install: pip install matplotlib")
        return 1

    out = args.output_dir
    for job in (
        redraw_transformer_attention,
        redraw_roofline,
        redraw_affine_quant_detail,
        redraw_gptq_idea,
        redraw_awq_idea,
        redraw_flashattention,
        redraw_online_softmax,
        redraw_leviathan_speculative,
        redraw_gqa,
        redraw_paged_attention,
        redraw_pope_kv_scaling,
        redraw_medusa_idea,
    ):
        job(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
