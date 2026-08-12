#!/usr/bin/env python3
"""Per-article Medium image layout.

Canonical publish paths:
  docs/medium/images/<article-slug>/<dest_name>.png

Generators still identify assets by a stable *source key* (old relative path).
This module maps each source key onto one or more article destinations and
can write/copy a produced PNG into those folders.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "medium" / "images"
SOURCE = IMG / "_source"

# article slug -> list of (source key relative to _source or legacy IMG root, dest filename)
MANIFEST: dict[str, list[tuple[str, str]]] = {
    "00-introduction": [
        ("thumbnails/thumb_00_introduction.png", "thumb.png"),
        ("workflows/00_unified_memory.png", "fig1.png"),
        ("workflows/00_inference_pipeline.png", "fig2.png"),
        ("papers/williams_roofline_redraw.png", "fig3.png"),
        ("papers/vaswani_attention_redraw.png", "fig4.png"),
        ("00_intro_hardware_compare.png", "fig5.png"),
        ("01_heatmap_tps.png", "fig6.png"),
        ("01_heatmap_memory.png", "fig7.png"),
        ("01_speedup_all_models.png", "fig8.png"),
        ("01_efficiency_tps_per_gb.png", "fig9.png"),
        ("01_m3_vs_m5_w4.png", "fig10.png"),
        ("01_llama_m3_m5_all_bits.png", "fig11.png"),
        ("04_model_size_ladder.png", "fig12.png"),
        ("04_m5_extended_ladder.png", "fig13.png"),
        ("07_context_m3_m5_panels.png", "fig14.png"),
        ("05_m3_m5_full_stack.png", "fig15.png"),
        ("06_spec_m3_m5_qwen.png", "fig16.png"),
    ],
    "01-weight-quantization": [
        ("thumbnails/thumb_01_weight_quantization.png", "thumb.png"),
        ("papers/jacob_affine_quant_redraw.png", "fig1.png"),
        ("papers/frantar_gptq_redraw.png", "fig2.png"),
        ("papers/lin_awq_redraw.png", "fig3.png"),
        ("papers/williams_roofline_redraw.png", "fig4.png"),
        ("01_weight_quant_llama3-8b.png", "fig5.png"),
        ("01_pareto_memory_speed.png", "fig6.png"),
        ("01_speedup_vs_fp16.png", "fig7.png"),
        ("01_heatmap_tps.png", "fig8.png"),
        ("01_heatmap_memory.png", "fig9.png"),
        ("01_speedup_all_models.png", "fig10.png"),
        ("01_family_panels.png", "fig11.png"),
        ("01_m3_vs_m5_w4.png", "fig12.png"),
        ("01_llama_m3_m5_all_bits.png", "fig13.png"),
    ],
    "02-kv-cache-quantization": [
        ("thumbnails/thumb_02_kv_cache.png", "thumb.png"),
        ("workflows/02_kv_cache_workflow.png", "fig1.png"),
        ("papers/vaswani_attention_redraw.png", "fig2.png"),
        ("papers/pope_kv_scaling_redraw.png", "fig3.png"),
        ("papers/ainslie_gqa_redraw.png", "fig4.png"),
        ("02_kv_cache_compare.png", "fig5.png"),
        ("02_kv_long_generation.png", "fig6.png"),
        ("07_context_dual_axis.png", "fig7.png"),
        ("papers/kwon_paged_attention_redraw.png", "fig8.png"),
    ],
    "03-prefill-and-ttft": [
        ("thumbnails/thumb_03_prefill_ttft.png", "thumb.png"),
        ("workflows/03_prefill_vs_decode.png", "fig1.png"),
        ("papers/dao_flashattention_redraw.png", "fig2.png"),
        ("papers/milakov_online_softmax_redraw.png", "fig3.png"),
        ("03_prefill_ttft.png", "fig4.png"),
        ("03_ttft_vs_prompt_curve.png", "fig5.png"),
        ("07_workload_ttft.png", "fig6.png"),
    ],
    "04-model-size-ladder": [
        ("thumbnails/thumb_04_model_ladder.png", "thumb.png"),
        ("workflows/04_fit_ladder.png", "fig1.png"),
        ("04_model_size_ladder.png", "fig2.png"),
        ("04_ladder_scatter.png", "fig3.png"),
        ("01_efficiency_tps_per_gb.png", "fig4.png"),
        ("04_m5_extended_ladder.png", "fig5.png"),
        ("01_m3_vs_m5_w4.png", "fig6.png"),
    ],
    "05-full-optimization-stack": [
        ("thumbnails/thumb_05_full_stack.png", "thumb.png"),
        ("workflows/05_optimization_funnel.png", "fig1.png"),
        ("workflows/05_decision_tree.png", "fig2.png"),
        ("05_full_stack.png", "fig3.png"),
        ("05_full_stack_two_models.png", "fig4.png"),
        ("05_full_stack_memory.png", "fig5.png"),
        ("05_m5_config_matrix.png", "fig6.png"),
        ("05_m3_m5_full_stack.png", "fig7.png"),
        ("papers/williams_roofline_redraw.png", "fig8.png"),
    ],
    "06-speculative-decoding": [
        ("thumbnails/thumb_06_speculative.png", "thumb.png"),
        ("papers/leviathan_speculative_redraw.png", "fig1.png"),
        ("workflows/06_accept_reject.png", "fig2.png"),
        ("papers/cai_medusa_redraw.png", "fig3.png"),
        ("06_speculative_qwen-7b.png", "fig4.png"),
        ("06_speculative_speed_memory.png", "fig5.png"),
        ("06_spec_m3_m5_qwen.png", "fig6.png"),
        ("06_spec_speedup_vs_accept.png", "fig7.png"),
    ],
    "07-context-and-cache": [
        ("thumbnails/thumb_07_rag_context.png", "thumb.png"),
        ("workflows/07_rag_wall.png", "fig1.png"),
        ("papers/pope_kv_scaling_redraw.png", "fig2.png"),
        ("07_context_ttft.png", "fig3.png"),
        ("07_context_dual_axis.png", "fig4.png"),
        ("07_context_m3_m5_panels.png", "fig5.png"),
        ("workflows/07_prefix_cache_workflow.png", "fig6.png"),
        ("07_prefix_cache.png", "fig7.png"),
        ("07_workload_panels.png", "fig8.png"),
        ("07_workload_ttft.png", "fig9.png"),
    ],
}

ARTICLES = list(MANIFEST.keys())


def reverse_index() -> dict[str, list[tuple[str, str]]]:
    """source_key -> [(article_slug, dest_name), ...]"""
    out: dict[str, list[tuple[str, str]]] = {}
    for slug, files in MANIFEST.items():
        for src, dest in files:
            out.setdefault(src, []).append((slug, dest))
    return out


REVERSE = reverse_index()


def resolve_article(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().rstrip("/")
    if v in MANIFEST:
        return v
    # allow "00" or "00-introduction" prefix / short forms
    for slug in MANIFEST:
        if slug == v or slug.startswith(v) or slug.split("-", 1)[0] == v.zfill(2):
            return slug
    raise SystemExit(f"Unknown article {value!r}. Choose from:\n  " + "\n  ".join(ARTICLES))


def destinations(src_key: str, article: str | None = None) -> list[tuple[str, str]]:
    dests = REVERSE.get(src_key, [])
    if article:
        dests = [(s, d) for s, d in dests if s == article]
    return dests


def article_dir(slug: str) -> Path:
    path = IMG / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def publish_path(slug: str, dest_name: str) -> str:
    return f"docs/medium/images/{slug}/{dest_name}"


def emit_file(src_key: str, produced: Path, *, article: str | None = None) -> list[Path]:
    """Copy a produced PNG into every matching article folder."""
    dests = destinations(src_key, article)
    if not dests:
        # Asset not in any article manifest — keep under _source only.
        keep = SOURCE / src_key
        keep.parent.mkdir(parents=True, exist_ok=True)
        if produced.resolve() != keep.resolve():
            shutil.copy2(produced, keep)
        print(f"_source only: {src_key}")
        return [keep]

    written: list[Path] = []
    # Also stash under _source for regenerating / reuse.
    stash = SOURCE / src_key
    stash.parent.mkdir(parents=True, exist_ok=True)
    if produced.resolve() != stash.resolve():
        shutil.copy2(produced, stash)

    for slug, dest_name in dests:
        dest = article_dir(slug) / dest_name
        shutil.copy2(produced, dest)
        written.append(dest)
        print(f"{slug}/{dest_name}")
    return written


def emit_figure(src_key: str, fig, *, article: str | None = None, dpi: int = 150) -> list[Path]:
    """Save a matplotlib figure into matching article folders."""
    import matplotlib.pyplot as plt

    dests = destinations(src_key, article)
    if not dests:
        keep = SOURCE / src_key
        keep.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(keep, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"_source only: {src_key}")
        return [keep]

    first_slug, first_name = dests[0]
    first = article_dir(first_slug) / first_name
    first.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(first, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stash = SOURCE / src_key
    stash.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(first, stash)

    written = [first]
    print(f"{first_slug}/{first_name}")
    for slug, dest_name in dests[1:]:
        dest = article_dir(slug) / dest_name
        shutil.copy2(first, dest)
        written.append(dest)
        print(f"{slug}/{dest_name}")
    return written


def source_keys_for_article(article: str) -> set[str]:
    return {src for src, _ in MANIFEST[article]}


def migrate_legacy_into_source() -> None:
    """Move flat / workflows / papers / thumbnails into _source/ if present."""
    SOURCE.mkdir(parents=True, exist_ok=True)
    for name in ("workflows", "papers", "thumbnails"):
        legacy = IMG / name
        if legacy.is_dir() and legacy.resolve() != (SOURCE / name).resolve():
            dest = SOURCE / name
            if dest.exists():
                for p in legacy.rglob("*.png"):
                    target = dest / p.relative_to(legacy)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, target)
                shutil.rmtree(legacy)
            else:
                shutil.move(str(legacy), str(dest))
            print(f"migrated {name}/ -> _source/{name}/")

    for png in sorted(IMG.glob("*.png")):
        dest = SOURCE / png.name
        shutil.copy2(png, dest)
        png.unlink()
        print(f"migrated {png.name} -> _source/")


def write_article_readmes(article: str | None = None) -> None:
    slugs = [article] if article else ARTICLES
    for slug in slugs:
        lines = [
            f"# Images for `{slug}`",
            "",
            "Featured cover: `thumb.png` (not numbered).",
            "",
            "## Figures",
            "",
        ]
        for src, dest in MANIFEST[slug]:
            if dest == "thumb.png":
                continue
            m = re.match(r"fig(\d+)\.png$", dest)
            label = f"Fig {m.group(1)}" if m else dest
            hint = Path(src).stem.replace("_", " ").replace(" redraw", "").strip()
            lines.append(f"- **{label}** — `{dest}` ({hint})")
        lines.append("")
        (article_dir(slug) / "README.md").write_text("\n".join(lines) + "\n")

    index = [
        "# Medium images by article",
        "",
        "Each article has its own folder. Shared concepts are copied into every",
        "article that uses them (so publishing never depends on a sibling folder).",
        "",
    ]
    for slug in ARTICLES:
        index.append(f"- [`{slug}/`]({slug}/)")
    index.append("")
    index.append("Internal regenerator cache: [`_source/`](_source/) (not for publishing).")
    index.append("")
    (IMG / "README.md").write_text("\n".join(index))


def add_article_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--article",
        "-a",
        default=None,
        help="Only write images for one article slug (e.g. 00-introduction or 00)",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migrate-legacy", action="store_true", help="Move flat shared folders into _source/")
    add_article_arg(parser)
    args = parser.parse_args()
    article = resolve_article(args.article)
    if args.migrate_legacy:
        migrate_legacy_into_source()
    write_article_readmes(article)
    print("layout ok")
