#!/usr/bin/env python3
"""Deep Medium result plots — mine all Mac M3 / M5 Max article JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from medium_image_layout import SOURCE, add_article_arg, emit_file, resolve_article, source_keys_for_article  # noqa: E402

RESULTS = ROOT / "results"
OUT = SOURCE
_ARTICLE: str | None = None


def _safe(hw: str) -> str:
    return hw.replace(" ", "_").replace("/", "-")


def _load(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _ok(d: dict | None) -> bool:
    return bool(d and d.get("status") == "ok" and d.get("throughput_tps"))


def _save(fig, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    src_key = path.name
    if _ARTICLE and src_key not in source_keys_for_article(_ARTICLE):
        print(f"skip {src_key} (not used by {_ARTICLE})")
        return
    emit_file(src_key, path, article=_ARTICLE)


def _style():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
        }
    )


def iter_ok(article_dir: Path):
    if not article_dir.exists():
        return
    for path in sorted(article_dir.rglob("*.json")):
        if path.name in ("manifest.json", "article_summary.json", "sweep_summary.json"):
            continue
        if path.name.startswith("sweep_"):
            continue
        data = _load(path)
        if _ok(data):
            data["_path"] = path
            yield data


def plot_heatmap_quant(m3: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    models = []
    art = m3 / "article_01_weight-quantization"
    for d in sorted(art.iterdir()):
        if d.is_dir():
            models.append(d.name)
    configs = ["fp16", "w8", "w4", "w2"]
    mat = np.full((len(models), len(configs)), np.nan)
    for i, m in enumerate(models):
        for j, c in enumerate(configs):
            data = _load(art / m / f"{c}.json")
            if _ok(data):
                mat[i, j] = data["throughput_tps"]
    fig, ax = plt.subplots(figsize=(8, max(5, len(models) * 0.38)))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_title("Decode tok/s heatmap — all models × weight configs (Mac M3)")
    for i in range(len(models)):
        for j in range(len(configs)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > np.nanmax(mat) * 0.55 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="tok/s")
    fig.tight_layout()
    _save(fig, out / "01_heatmap_tps.png")

    # memory heatmap
    matm = np.full((len(models), len(configs)), np.nan)
    for i, m in enumerate(models):
        for j, c in enumerate(configs):
            data = _load(art / m / f"{c}.json")
            if _ok(data):
                matm[i, j] = data.get("memory_gb", 0)
    fig, ax = plt.subplots(figsize=(8, max(5, len(models) * 0.38)))
    im = ax.imshow(matm, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_title("Peak memory (GB) heatmap — all models × weight configs (Mac M3)")
    for i in range(len(models)):
        for j in range(len(configs)):
            v = matm[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="GB")
    fig.tight_layout()
    _save(fig, out / "01_heatmap_memory.png")


def plot_speedup_family(m3: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    art = m3 / "article_01_weight-quantization"
    rows = []
    for d in sorted(art.iterdir()):
        if not d.is_dir():
            continue
        fp = _load(d / "fp16.json")
        w4 = _load(d / "w4.json")
        if _ok(fp) and _ok(w4):
            rows.append((d.name, w4["throughput_tps"] / fp["throughput_tps"],
                         fp["memory_gb"] / w4["memory_gb"]))
    if not rows:
        return
    rows.sort(key=lambda x: x[1])
    names = [r[0] for r in rows]
    sp = [r[1] for r in rows]
    mem = [r[2] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(4, len(names) * 0.35)))
    ax1.barh(names, sp, color="#2E7D4F")
    ax1.set_xlabel("tok/s speedup (w4 / fp16)")
    ax1.set_title("Decode speedup from fp16 → w4")
    ax1.axvline(1, color="#888", ls="--")
    ax2.barh(names, mem, color="#C26A1A")
    ax2.set_xlabel("Memory reduction (fp16 GB / w4 GB)")
    ax2.set_title("Memory shrink factor")
    fig.suptitle("Weight quantization payoff across every model (Mac M3)", fontsize=12)
    fig.tight_layout()
    _save(fig, out / "01_speedup_all_models.png")


def plot_m3_vs_m5_quant(m3: Path, m5: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    presets = ["qwen-0.5b", "llama-3.2-1b", "qwen-3b", "mistral-7b", "llama3-8b", "qwen-7b", "gemma-9b"]
    labels, m3v, m5v = [], [], []
    for p in presets:
        a = _load(m3 / "article_01_weight-quantization" / p / "w4.json")
        b = _load(m5 / "article_01_weight-quantization" / p / "w4.json")
        if _ok(a) and _ok(b):
            labels.append(p)
            m3v.append(a["throughput_tps"])
            m5v.append(b["throughput_tps"])
    if not labels:
        return
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, m3v, w, label="Mac M3", color="#4C72B0")
    ax.bar(x + w / 2, m5v, w, label="Mac M5 Max", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Decode tok/s @ w4")
    ax.set_title("Same w4 checkpoints — M3 vs M5 Max throughput")
    ax.legend()
    for i, (a, b) in enumerate(zip(m3v, m5v)):
        ax.text(i, max(a, b) + 3, f"{b/a:.1f}×", ha="center", fontsize=8, color="#333")
    fig.tight_layout()
    _save(fig, out / "01_m3_vs_m5_w4.png")

    # llama3-8b all configs M3 vs M5
    labs, a, b = [], [], []
    for cfg in ("fp16", "w8", "w4", "w2"):
        d3 = _load(m3 / "article_01_weight-quantization" / "llama3-8b" / f"{cfg}.json")
        d5 = _load(m5 / "article_01_weight-quantization" / "llama3-8b" / f"{cfg}.json")
        if _ok(d3) and _ok(d5):
            labs.append(cfg)
            a.append(d3["throughput_tps"])
            b.append(d5["throughput_tps"])
    if labs:
        x = np.arange(len(labs))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - 0.2, a, 0.4, label="M3", color="#4C72B0")
        ax.bar(x + 0.2, b, 0.4, label="M5 Max", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(labs)
        ax.set_ylabel("tok/s")
        ax.set_title("Llama 3.1 8B — every weight config on M3 vs M5 Max")
        ax.legend()
        fig.tight_layout()
        _save(fig, out / "01_llama_m3_m5_all_bits.png")


def plot_ttft_heatmap_models(m3: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    art = m3 / "article_01_weight-quantization"
    models, fp, w4 = [], [], []
    for d in sorted(art.iterdir()):
        if not d.is_dir():
            continue
        a = _load(d / "fp16.json")
        b = _load(d / "w4.json")
        if _ok(a) and _ok(b):
            models.append(d.name)
            fp.append(a.get("ttft_ms", 0) / 1000)
            w4.append(b.get("ttft_ms", 0) / 1000)
    if not models:
        return
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, fp, 0.4, label="fp16 TTFT (s)", color="#A94442")
    ax.bar(x + 0.2, w4, 0.4, label="w4 TTFT (s)", color="#2E7D4F")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("TTFT (seconds)")
    ax.set_title("Time-to-first-token across models — fp16 vs w4 (Mac M3)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out / "01_ttft_all_models.png")


def plot_full_stack_m5_matrix(m5: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    base = m5 / "article_05_full-stack" / "llama3-8b"
    if not base.exists():
        return
    bits = ["fp16", "w8", "w4", "w2"]
    extras = ["", "+prefill", "+kv_cache", "+kv_cache+prefill"]
    mat = np.full((len(bits), len(extras)), np.nan)
    for i, b in enumerate(bits):
        for j, e in enumerate(extras):
            label = b + e
            # optimized alias
            path = base / f"{label}.json"
            if label == "w4+kv_cache+prefill":
                opt = _load(base / "optimized.json")
                if _ok(opt):
                    mat[i, j] = opt["throughput_tps"]
                    continue
            data = _load(path)
            if _ok(data):
                mat[i, j] = data["throughput_tps"]
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(extras)))
    ax.set_xticklabels(["plain", "+prefill", "+kv", "+kv+prefill"], fontsize=9)
    ax.set_yticks(range(len(bits)))
    ax.set_yticklabels(bits)
    ax.set_title("Llama 3.1 8B full config matrix — tok/s (Mac M5 Max)")
    for i in range(len(bits)):
        for j in range(len(extras)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v > np.nanmean(mat) else "black")
    fig.colorbar(im, ax=ax, label="tok/s")
    fig.tight_layout()
    _save(fig, out / "05_m5_config_matrix.png")


def plot_m3_m5_full_stack(m3: Path, m5: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    pairs = []
    for preset, fp_name, opt_name in [
        ("llama3-8b", "fp16.json", "optimized.json"),
        ("mistral-7b", "fp16_mistral.json", "optimized_mistral.json"),
    ]:
        a = _load(m3 / "article_05_full-stack" / preset / fp_name)
        b = _load(m3 / "article_05_full-stack" / preset / opt_name)
        c = _load(m5 / "article_05_full-stack" / preset / "fp16.json")
        d = _load(m5 / "article_05_full-stack" / preset / "optimized.json")
        if not _ok(c):
            c = _load(m5 / "article_01_weight-quantization" / preset / "fp16.json")
        if not _ok(d):
            d = _load(m5 / "article_05_full-stack" / preset / "w4+kv_cache+prefill.json")
        if _ok(a) and _ok(b):
            pairs.append((preset, a, b, c if _ok(c) else None, d if _ok(d) else None))
    if not pairs:
        return
    # grouped: M3 fp16, M3 opt, M5 fp16, M5 opt
    labels = [p[0] for p in pairs]
    series = {
        "M3 fp16": [p[1]["throughput_tps"] for p in pairs],
        "M3 optimized": [p[2]["throughput_tps"] for p in pairs],
    }
    if all(p[3] for p in pairs):
        series["M5 fp16"] = [p[3]["throughput_tps"] for p in pairs]
    if all(p[4] for p in pairs):
        series["M5 optimized"] = [p[4]["throughput_tps"] for p in pairs]
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    colors = ["#A94442", "#2E7D4F", "#4C72B0", "#DD8452"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - (len(series) - 1) / 2) * width, vals, width, label=name, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("tok/s")
    ax.set_title("Full stack: fp16 vs optimized — M3 and M5 Max")
    ax.legend()
    fig.tight_layout()
    _save(fig, out / "05_m3_m5_full_stack.png")


def plot_speculative_m3_m5(m3: Path, m5: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    for hw_name, hw in [("M3", m3), ("M5 Max", m5)]:
        art = hw / "article_06_speculative-decoding"
        for preset in ("qwen-7b", "llama3-8b", "mistral-7b"):
            base = _load(art / preset / f"{preset}_w4_baseline.json")
            spec = _load(art / preset / f"{preset}_w4_speculative.json")
            if _ok(base):
                rows.append({
                    "hw": hw_name,
                    "preset": preset,
                    "base": base["throughput_tps"],
                    "spec": spec["throughput_tps"] if _ok(spec) else None,
                    "alpha": spec.get("draft_accept_rate") if _ok(spec) else None,
                })
    # Qwen comparison chart
    q = [r for r in rows if r["preset"] == "qwen-7b"]
    if q:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels, base, spec = [], [], []
        for r in q:
            labels.append(r["hw"])
            base.append(r["base"])
            spec.append(r["spec"] or 0)
        x = np.arange(len(labels))
        ax.bar(x - 0.2, base, 0.4, label="baseline w4", color="#4C72B0")
        ax.bar(x + 0.2, spec, 0.4, label="speculative", color="#2E7D4F")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("tok/s")
        ax.set_title("Speculative decoding — Qwen-7B on M3 vs M5 Max")
        for i, r in enumerate(q):
            if r["alpha"]:
                ax.text(i + 0.2, (r["spec"] or 0) + 2, f"α={r['alpha']:.0%}", ha="center", fontsize=8)
        ax.legend()
        fig.tight_layout()
        _save(fig, out / "06_spec_m3_m5_qwen.png")

    # acceptance + speedup table figure for all ok speculative
    ok_rows = [r for r in rows if r["spec"]]
    if ok_rows:
        fig, ax = plt.subplots(figsize=(9, 4))
        names = [f"{r['preset']}\n{r['hw']}" for r in ok_rows]
        speedups = [r["spec"] / r["base"] for r in ok_rows]
        alphas = [r["alpha"] * 100 for r in ok_rows]
        x = np.arange(len(names))
        ax.bar(x - 0.2, speedups, 0.4, label="speedup (×)", color="#6B3FA0")
        ax2 = ax.twinx()
        ax2.bar(x + 0.2, alphas, 0.4, label="accept %", color="#B8860B", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("Speedup vs baseline")
        ax2.set_ylabel("Acceptance rate %")
        ax.set_title("Speculative runs that succeeded — speedup vs acceptance")
        ax.axhline(1, color="#888", ls="--")
        fig.tight_layout()
        _save(fig, out / "06_spec_speedup_vs_accept.png")


def plot_model_ladder_m3_m5(m3: Path, m5: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    # large models on M5
    large = [
        ("qwen-0.5b", "0.5B"),
        ("llama-3.2-1b", "1B"),
        ("qwen-3b", "3B"),
        ("mistral-7b", "7B"),
        ("llama3-8b", "8B"),
        ("gemma-9b", "9B"),
        ("mistral-nemo-12b", "12B"),
        ("qwen-14b", "14B"),
        ("mistral-small-22b", "22B"),
        ("gemma-27b", "27B"),
    ]
    labs, tps, mem = [], [], []
    for preset, short in large:
        d = _load(m5 / "article_04_model-size-ladder" / preset / "w4.json")
        if not _ok(d):
            d = _load(m5 / "article_01_weight-quantization" / preset / "w4.json")
        if _ok(d):
            labs.append(short)
            tps.append(d["throughput_tps"])
            mem.append(d["memory_gb"])
    if labs:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(labs, tps, color="#4C72B0", label="tok/s")
        ax1.set_ylabel("Decode tok/s @ w4")
        ax2 = ax1.twinx()
        ax2.plot(labs, mem, "o-", color="#C26A1A", lw=2, label="Peak GB")
        ax2.set_ylabel("Peak memory (GB)")
        ax1.set_title("Extended model ladder on Mac M5 Max (w4)")
        ax1.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        _save(fig, out / "04_m5_extended_ladder.png")


def plot_context_deep(m3: Path, m5: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    def collect(hw: Path):
        base = hw / "article_07_context-and-cache" / "llama3-8b"
        pts = []
        for p, f in [(256, "ctx_p256.json"), (512, "ctx_p512.json"),
                     (1024, "ctx_p1024.json"), (2048, "ctx_p2048.json")]:
            d = _load(base / f)
            if _ok(d):
                pts.append((p, d["ttft_ms"] / 1000, d["throughput_tps"], d.get("memory_gb", 0)))
        return pts

    m3p, m5p = collect(m3), collect(m5)
    if m3p:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        px = [p[0] for p in m3p]
        axes[0].plot(px, [p[1] for p in m3p], "o-", color="#C26A1A", label="M3")
        if m5p:
            axes[0].plot([p[0] for p in m5p], [p[1] for p in m5p], "s-", color="#4C72B0", label="M5 Max")
        axes[0].set_title("TTFT (s)")
        axes[0].set_xlabel("prompt tokens")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(px, [p[2] for p in m3p], "o-", color="#C26A1A", label="M3")
        if m5p:
            axes[1].plot([p[0] for p in m5p], [p[2] for p in m5p], "s-", color="#4C72B0", label="M5 Max")
        axes[1].set_title("Decode tok/s")
        axes[1].set_xlabel("prompt tokens")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(px, [p[3] for p in m3p], "o-", color="#C26A1A", label="M3")
        if m5p:
            axes[2].plot([p[0] for p in m5p], [p[3] for p in m5p], "s-", color="#4C72B0", label="M5 Max")
        axes[2].set_title("Peak GB")
        axes[2].set_xlabel("prompt tokens")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
        fig.suptitle("Context length sweep — Llama 3.1 8B", fontsize=12)
        fig.tight_layout()
        _save(fig, out / "07_context_m3_m5_panels.png")


def plot_workloads_deep(m3: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    base = m3 / "article_07_context-and-cache" / "llama3-8b"
    rows = []
    for j in sorted(base.glob("wl_*.json")):
        d = _load(j)
        if _ok(d):
            rows.append((j.stem.replace("wl_", ""), d["ttft_ms"] / 1000, d["throughput_tps"], d.get("memory_gb", 0)))
    if not rows:
        return
    names = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].barh(names, [r[1] for r in rows], color="#A94442")
    axes[0].set_xlabel("TTFT (s)")
    axes[0].set_title("Latency")
    axes[0].invert_yaxis()
    axes[1].barh(names, [r[2] for r in rows], color="#2E7D4F")
    axes[1].set_xlabel("tok/s")
    axes[1].set_title("Throughput")
    axes[1].invert_yaxis()
    axes[2].barh(names, [r[3] for r in rows], color="#4C72B0")
    axes[2].set_xlabel("Peak GB")
    axes[2].set_title("Memory")
    axes[2].invert_yaxis()
    fig.suptitle("Workload stress matrix — Llama 8B (Mac M3)", fontsize=12)
    fig.tight_layout()
    _save(fig, out / "07_workload_panels.png")


def plot_kv_long_gen(m3: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    base = m3 / "article_02_kv-cache-quantization" / "llama3-8b"
    rows = []
    for label, f in [
        ("w4 short", "llama3-8b_w4.json"),
        ("w4+kv short", "llama3-8b_w4_kv.json"),
        ("w4+kv long-g", "llama3-8b_w4_kv_long_g.json"),
    ]:
        d = _load(base / f)
        if _ok(d):
            rows.append((label, d["throughput_tps"], d.get("ttft_ms", 0) / 1000, d.get("generation_tokens", 128)))
    if rows:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar([r[0] for r in rows], [r[1] for r in rows], color=["#4C72B0", "#55A868", "#C44E52"])
        ax.set_ylabel("tok/s")
        ax.set_title("KV quant: short vs long generation (Llama 8B, Mac M3)")
        for i, r in enumerate(rows):
            ax.text(i, r[1] + 0.3, f"g={r[3]}\nTTFT={r[2]:.1f}s", ha="center", fontsize=8)
        fig.tight_layout()
        _save(fig, out / "02_kv_long_generation.png")


def plot_runtime_compare(m3: Path, m5: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    for hw_name, hw in [("M3", m3), ("M5 Max", m5)]:
        for preset, cfg in [("llama3-8b", "w4"), ("llama3-8b", "fp16"), ("mistral-7b", "w4")]:
            path = hw / "article_10_runtimes" / preset / f"{cfg}_compare.json"
            d = _load(path)
            if not d:
                continue
            mlx = d.get("mlx") or {}
            lcpp = d.get("llamacpp") or {}
            mlx_tps = mlx.get("throughput_tps")
            lcpp_tps = lcpp.get("tg_tps") or lcpp.get("throughput_tps")
            if mlx_tps and lcpp_tps:
                rows.append((f"{preset}\n{cfg}\n{hw_name}", mlx_tps, lcpp_tps))
    if not rows:
        return
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.4), 5))
    ax.bar(x - 0.2, [r[1] for r in rows], 0.4, label="MLX", color="#4C72B0")
    ax.bar(x + 0.2, [r[2] for r in rows], 0.4, label="llama.cpp", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("Decode tok/s")
    ax.set_title("Runtime compare — MLX vs llama.cpp")
    ax.legend()
    fig.tight_layout()
    _save(fig, out / "10_mlx_vs_llamacpp.png")


def plot_family_groups(m3: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    families = {
        "Qwen": ["qwen-0.5b", "qwen-1.5b", "qwen-3b", "qwen-7b"],
        "Llama": ["llama-3.2-1b", "llama-3.2-3b", "llama3-8b"],
        "Phi": ["phi-3-mini", "phi-3.5-mini"],
        "Gemma": ["gemma-2-2b", "gemma-9b"],
        "Mistral/DeepSeek": ["mistral-7b", "deepseek-r1-qwen-7b", "deepseek-r1-llama-8b"],
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.ravel()
    art = m3 / "article_01_weight-quantization"
    for ax, (fam, presets) in zip(axes, families.items()):
        labs, fp, w4 = [], [], []
        for p in presets:
            a = _load(art / p / "fp16.json")
            b = _load(art / p / "w4.json")
            if _ok(a) and _ok(b):
                labs.append(p.replace("deepseek-r1-", "ds-"))
                fp.append(a["throughput_tps"])
                w4.append(b["throughput_tps"])
        if not labs:
            ax.axis("off")
            continue
        x = np.arange(len(labs))
        ax.bar(x - 0.2, fp, 0.4, label="fp16", color="#A94442")
        ax.bar(x + 0.2, w4, 0.4, label="w4", color="#2E7D4F")
        ax.set_xticks(x)
        ax.set_xticklabels(labs, rotation=20, ha="right", fontsize=7)
        ax.set_title(fam)
        ax.legend(fontsize=7)
    axes[-1].axis("off")
    fig.suptitle("Family zoom-ins — fp16 vs w4 decode tok/s (Mac M3)", fontsize=13)
    fig.tight_layout()
    _save(fig, out / "01_family_panels.png")


def plot_efficiency(m3: Path, out: Path) -> None:
    """tok/s per GB — efficiency metric."""
    import matplotlib.pyplot as plt

    art = m3 / "article_01_weight-quantization"
    rows = []
    for d in sorted(art.iterdir()):
        if not d.is_dir():
            continue
        data = _load(d / "w4.json")
        if _ok(data) and data.get("memory_gb"):
            eff = data["throughput_tps"] / data["memory_gb"]
            rows.append((d.name, eff, data["throughput_tps"], data["memory_gb"]))
    rows.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(9, max(4, len(rows) * 0.35)))
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color="#6B3FA0")
    ax.set_xlabel("Efficiency = tok/s per GB @ w4")
    ax.set_title("Which model gives the most speed per gigabyte? (Mac M3)")
    fig.tight_layout()
    _save(fig, out / "01_efficiency_tps_per_gb.png")


def main() -> int:
    global _ARTICLE
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=Path, default=OUT)
    add_article_arg(parser)
    args = parser.parse_args()
    _ARTICLE = resolve_article(args.article)
    try:
        import matplotlib  # noqa
    except ImportError:
        print("pip install matplotlib")
        return 1
    _style()
    m3 = RESULTS / "Mac_M3"
    m5 = RESULTS / "Mac_M5_Max"
    out = args.output_dir
    for job in (
        plot_heatmap_quant,
        plot_speedup_family,
        plot_m3_vs_m5_quant,
        plot_ttft_heatmap_models,
        plot_full_stack_m5_matrix,
        plot_m3_m5_full_stack,
        plot_speculative_m3_m5,
        plot_model_ladder_m3_m5,
        plot_context_deep,
        plot_workloads_deep,
        plot_kv_long_gen,
        plot_runtime_compare,
        plot_family_groups,
        plot_efficiency,
    ):
        # Jobs that need (m3, m5, out) vs (m3, out) / (m5, out)
        name = job.__name__
        if name in {"plot_m3_vs_m5_quant", "plot_m3_m5_full_stack", "plot_speculative_m3_m5", "plot_model_ladder_m3_m5", "plot_context_deep", "plot_runtime_compare"}:
            job(m3, m5, out)
        elif name == "plot_full_stack_m5_matrix":
            job(m5, out)
        else:
            job(m3, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
