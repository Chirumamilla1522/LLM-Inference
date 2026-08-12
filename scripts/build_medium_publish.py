#!/usr/bin/env python3
"""
Emit Medium *editor* paste kits — not HTML.

Each .medium.txt maps 1:1 to Medium's block editor:
  Big T      → Title
  Little T   → Subtitle
  Featured   → Story image (wide horizontal)
  Subhead    → Section header (H2-style in Medium)
  Body       → Short paragraphs
  Pull quote → Medium quote block
  Image      → + menu → image (upload the file)
  Code       → Medium code block
  List       → bullets / numbers
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "medium" / "publish"
IMG = "docs/medium/images"


def kit(
    *,
    slug: str,
    title: str,
    subtitle: str,
    featured: str,
    featured_caption: str,
    tags: list[str],
    series: str,
    blocks: list[str],
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 72,
        "MEDIUM EDITOR PASTE KIT",
        f"File: {slug}.medium.txt",
        "=" * 72,
        "",
        "HOW TO USE IN MEDIUM",
        "1. New story",
        "2. Click the title area → apply Big T → paste TITLE",
        "3. Click under title → apply Little T → paste SUBTITLE",
        "4. + menu → Image → upload FEATURED IMAGE (wide 16:9)",
        "5. Add caption under featured image",
        "6. For each block below, use the matching Medium control",
        "7. Tags → Publish → share (see DISTRIBUTION.md)",
        "",
        "-" * 72,
        "SERIES: " + series,
        "TAGS: " + " · ".join(tags),
        "-" * 72,
        "",
        "┌─────────────────────────────────────────────────────────────",
        "│ TITLE  (Medium: Big T)",
        "└─────────────────────────────────────────────────────────────",
        title,
        "",
        "┌─────────────────────────────────────────────────────────────",
        "│ SUBTITLE  (Medium: Little T)",
        "└─────────────────────────────────────────────────────────────",
        subtitle,
        "",
        "┌─────────────────────────────────────────────────────────────",
        "│ FEATURED IMAGE  (wide horizontal cover under subtitle)",
        "└─────────────────────────────────────────────────────────────",
        f"UPLOAD: {featured}",
        f"CAPTION: {featured_caption}",
        "",
    ]
    lines.extend(blocks)
    lines += [
        "",
        "┌─────────────────────────────────────────────────────────────",
        "│ END CTA",
        "└─────────────────────────────────────────────────────────────",
        "",
        "BODY:",
        "If this was useful, clone the harness and run the same benches on your Mac:",
        "",
        "BODY:",
        "https://github.com/Chirumamilla1522/LLM-Inference",
        "",
        "┌─────────────────────────────────────────────────────────────",
        "│ TAGS (add in Medium’s tag picker — max ~5 strong ones)",
        "└─────────────────────────────────────────────────────────────",
        ", ".join(tags),
        "",
    ]
    path = OUT / f"{slug}.medium.txt"
    path.write_text("\n".join(lines))
    # companion meta for quick copy
    (OUT / f"{slug}-meta.txt").write_text(
        f"TITLE (Big T): {title}\n"
        f"SUBTITLE (Little T): {subtitle}\n"
        f"FEATURED IMAGE: {featured}\n"
        f"FEATURED CAPTION: {featured_caption}\n"
        f"SERIES: {series}\n"
        f"TAGS: {', '.join(tags)}\n"
    )
    print(f"Wrote {path.name}")


def subhead(text: str) -> list[str]:
    return [
        "┌─────────────────────────────────────────────────────────────",
        "│ SUBHEAD  (Medium: smaller section header / Subtitle style for sections)",
        "└─────────────────────────────────────────────────────────────",
        text,
        "",
    ]


def body(*paragraphs: str) -> list[str]:
    out: list[str] = []
    for p in paragraphs:
        out += ["BODY:", p, ""]
    return out


def quote(text: str) -> list[str]:
    return [
        "┌─────────────────────────────────────────────────────────────",
        "│ PULL QUOTE  (select text → Medium quote button)",
        "└─────────────────────────────────────────────────────────────",
        text,
        "",
    ]


def image(path: str, caption: str) -> list[str]:
    return [
        "┌─────────────────────────────────────────────────────────────",
        "│ INLINE IMAGE  (+ menu → Image → upload file → add caption)",
        "└─────────────────────────────────────────────────────────────",
        f"UPLOAD: {path}",
        f"CAPTION: {caption}",
        "",
    ]


def bullets(items: list[str]) -> list[str]:
    return [
        "┌─────────────────────────────────────────────────────────────",
        "│ BULLET LIST  (type - or * then space in Medium)",
        "└─────────────────────────────────────────────────────────────",
        *[f"• {i}" for i in items],
        "",
    ]


def numbers(items: list[str]) -> list[str]:
    return [
        "┌─────────────────────────────────────────────────────────────",
        "│ NUMBERED LIST",
        "└─────────────────────────────────────────────────────────────",
        *[f"{i+1}. {t}" for i, t in enumerate(items)],
        "",
    ]


def code(text: str) -> list[str]:
    return [
        "┌─────────────────────────────────────────────────────────────",
        "│ CODE BLOCK  (+ menu → Code block)",
        "└─────────────────────────────────────────────────────────────",
        text,
        "",
    ]


def divider_note(text: str) -> list[str]:
    return [
        f"— {text} —",
        "",
    ]


def art00() -> None:
    blocks: list[str] = []
    blocks += body(
        "I loaded Meta’s Llama 3.1 8B on a MacBook Pro and watched Activity Monitor go red.",
        "No cloud bill. No NVIDIA card. The model ran.",
        "It also felt like dial-up: about 5 tokens per second, with a multi-second freeze before the first word.",
        "That gap — between “it runs” and “I’d use this every day” — is this series.",
    )
    blocks += subhead("Why Apple Silicon changes the rules")
    blocks += body(
        "On a gaming PC, GPU VRAM is a separate pool. On Apple Silicon, CPU and GPU share one unified memory pool.",
        "Your browser tabs and your 8B weights fight for the same bytes.",
    )
    blocks += image(
        f"{IMG}/workflows/00_unified_memory.png",
        "Unified memory — weights, KV cache, OS, and apps share one DRAM pool",
    )
    blocks += quote(
        "Fun fact: Local LLMs on Mac only became practical once consumer unified memory crossed roughly 16–24 GB."
    )
    blocks += subhead("The only three metrics that matter")
    blocks += image(
        f"{IMG}/workflows/00_inference_pipeline.png",
        "Load → prefill → first token (TTFT) → decode (tok/s)",
    )
    blocks += bullets(
        [
            "Peak memory (GB) — will it fit without swap?",
            "TTFT (ms) — how long you stare at a blank cursor",
            "Decode tok/s — how fast the answer streams",
        ]
    )
    blocks += body("Optimize the wrong one and your “faster model” still feels broken.")
    blocks += subhead("The brutal FP16 baseline")
    blocks += body("Llama 3.1 8B, FP16, Mac M3 (24 GB), 512-token prompt, 128-token generation:")
    blocks += bullets(
        [
            "16.33 GB peak memory",
            "2,651 ms to first token",
            "5.3 tok/s decode",
        ]
    )
    blocks += image(
        f"{IMG}/00_intro_hardware_compare.png",
        "Same model family — precision and silicon change everything",
    )
    blocks += body(
        "On Mac M5 Max, the same FP16 demo jumps to roughly 34 tok/s with far lower TTFT. Silicon matters. Software still matters more for fitting."
    )
    blocks += image(
        f"{IMG}/papers/williams_roofline_redraw.png",
        "Original redraw — Roofline idea (Williams et al., 2009). Decode is often bandwidth-bound.",
    )
    blocks += subhead("What this series will do")
    blocks += numbers(
        [
            "Weight quantization",
            "KV cache quantization",
            "Prefill & TTFT",
            "Model size ladder",
            "Full optimization stack",
            "Speculative decoding",
            "Bonus: context, RAG, prefix cache",
        ]
    )
    blocks += body(
        "Every number comes from reproducible JSON in an open harness on MLX."
    )
    blocks += image(
        f"{IMG}/01_heatmap_tps.png",
        "Sneak peek — tok/s heatmap across models and bit-widths on Mac M3",
    )
    blocks += subhead("A 10-minute sanity check")
    blocks += numbers(
        [
            "Pick one model you care about.",
            "Run FP16 and 4-bit only.",
            "Confirm memory drops ~2–3× and decode rises ~3× on an M3-class chip.",
            "Measure TTFT with your prompt length.",
        ]
    )
    blocks += code('./scripts/run_article.sh 0 "Mac M3"')
    blocks += body("Next → Part 2: 4-Bit Weights Changed Everything")
    kit(
        slug="00-introduction",
        title="Running 8B LLMs on a MacBook: What Actually Matters",
        subtitle="Unified memory, the metrics that matter, and a brutal FP16 baseline on Apple Silicon",
        featured=f"{IMG}/thumbnails/thumb_00_introduction.png",
        featured_caption="Local LLMs on Apple Silicon — Part 1",
        tags=["Machine Learning", "Apple", "LLM", "Artificial Intelligence", "Programming"],
        series="Local LLMs on Apple Silicon — Part 1 of 7",
        blocks=blocks,
    )


def art01() -> None:
    b: list[str] = []
    b += body(
        "An 8B model in FP16 needs ~16 GB just for weights.",
        "On a 24 GB MacBook, that leaves almost nothing for the OS, your editor, and the KV cache.",
        "Weight quantization is the highest-leverage change for local Mac inference.",
    )
    b += subhead("How it works (redrawn from the papers — not copied)")
    b += body("We store each weight with fewer bits — usually 8, 4, or 2 — plus a tiny scale.")
    b += image(f"{IMG}/papers/jacob_affine_quant_redraw.png", "Original redraw — affine quantization (Jacob et al., 2018)")
    b += image(f"{IMG}/papers/frantar_gptq_redraw.png", "Original redraw — GPTQ idea (Frantar et al., 2022)")
    b += image(f"{IMG}/papers/lin_awq_redraw.png", "Original redraw — AWQ idea (Lin et al., 2023)")
    b += quote("Fun fact: GPTQ was built for 175B-class models that couldn’t fit on one GPU at FP16. The same math now makes 8B models comfortable on a laptop.")
    b += subhead("Why fewer bits also make decode faster")
    b += body("Each decode step often reads nearly all weights from memory. Fewer bytes per weight → higher tok/s on a bandwidth-bound chip.")
    b += image(f"{IMG}/papers/williams_roofline_redraw.png", "Original redraw — Roofline: LLM decode sits on the bandwidth slope")
    b += subhead("Llama 3.1 8B on Mac M3")
    b += bullets(
        [
            "fp16 — 16.3 GB · 5.8 tok/s",
            "w8 — 9.0 GB · 11.3 tok/s (~1.9×)",
            "w4 — 5.1 GB · 20.5 tok/s (~3.5×)",
            "w2 — 3.1 GB · 35.8 tok/s (~6×)",
        ]
    )
    b += image(f"{IMG}/01_weight_quant_llama3-8b.png", "Memory vs speed as bit-width drops")
    b += image(f"{IMG}/01_pareto_memory_speed.png", "Pareto frontier — w4 is the practical sweet spot on 24 GB")
    b += image(f"{IMG}/01_speedup_vs_fp16.png", "Explicit speedup vs FP16")
    b += subhead("All 14 models (the heatmap)")
    b += image(f"{IMG}/01_heatmap_tps.png", "Decode tok/s — every model × bit-width on Mac M3")
    b += image(f"{IMG}/01_heatmap_memory.png", "Peak memory — FP16 is the red zone on 24 GB")
    b += image(f"{IMG}/01_speedup_all_models.png", "fp16→w4 speedup and memory shrink across the board")
    b += subhead("M3 vs M5 Max")
    b += body("Same w4 checkpoints. Different silicon.")
    b += bullets(
        [
            "Llama 8B w4: 20.5 → 112 tok/s",
            "Qwen 0.5B w4: 215 → 581 tok/s",
        ]
    )
    b += image(f"{IMG}/01_m3_vs_m5_w4.png", "M3 vs M5 Max at w4")
    b += image(f"{IMG}/01_llama_m3_m5_all_bits.png", "Llama 8B across every bit-width on both chips")
    b += subhead("What you should actually run")
    b += bullets(
        [
            "16 GB Mac — 3B–7B @ w4",
            "24 GB Mac — 8B @ w4 as daily driver",
            "Skip FP16 8B as your everyday chat config",
        ]
    )
    b += code('./scripts/run_article.sh 1 "Mac M3"')
    b += body("← Part 1  ·  Next → Part 3: KV Cache")
    kit(
        slug="01-weight-quantization",
        title="4-Bit Weights Changed Everything on My M3 Mac",
        subtitle="Affine quantization, GPTQ/AWQ ideas redrawn, and 14-model heatmaps on Apple Silicon",
        featured=f"{IMG}/thumbnails/thumb_01_weight_quantization.png",
        featured_caption="Weight quantization — Part 2",
        tags=["Machine Learning", "Quantization", "LLM", "Apple", "Artificial Intelligence"],
        series="Local LLMs on Apple Silicon — Part 2 of 7",
        blocks=b,
    )


def art02() -> None:
    b: list[str] = []
    b += body(
        "Weight quantization gets the spotlight.",
        "Once generation starts, something else grows: the KV cache — keys and values for every token in context.",
        "For short chats it barely shows up in tok/s. For RAG, it’s the second memory bill.",
    )
    b += subhead("How the cache works")
    b += image(f"{IMG}/workflows/02_kv_cache_workflow.png", "KV grows linearly with sequence length; 4-bit KV ≈ ¼ the footprint")
    b += image(f"{IMG}/papers/vaswani_attention_redraw.png", "Original redraw — attention (Vaswani et al., 2017)")
    b += image(f"{IMG}/papers/pope_kv_scaling_redraw.png", "Original redraw — inspired by Pope et al. (2022)")
    b += subhead("GQA: shrink heads before you quantize")
    b += image(f"{IMG}/papers/ainslie_gqa_redraw.png", "Original redraw — GQA vs MHA (Ainslie et al., 2023)")
    b += quote("Llama 3, Mistral, and Qwen already cut KV heads with GQA. 4-bit KV stacks on top of that.")
    b += subhead("Why our short-context bench “does nothing”")
    b += body("At 512 prompt + 128 gen on Mac M3:")
    b += bullets(
        [
            "Llama 8B: 20.7 → 20.4 tok/s",
            "Mistral 7B: 21.6 → 21.2",
            "Qwen 7B: 21.8 → 21.4",
        ]
    )
    b += image(f"{IMG}/02_kv_cache_compare.png", "Short context: throughput almost unchanged")
    b += image(f"{IMG}/02_kv_long_generation.png", "Longer generation: still weight-bound at laptop batch size 1")
    b += body("The win appears at long context, multi-session serving, or tight RAM — not in a 640-token microbench.")
    b += image(f"{IMG}/07_context_dual_axis.png", "Where KV pressure shows up — TTFT explodes as prompts grow")
    b += image(f"{IMG}/papers/kwon_paged_attention_redraw.png", "Original redraw — paged KV idea (Kwon et al., 2023)")
    b += subhead("When to enable it")
    b += bullets(
        [
            "Always quantize weights first (w4)",
            "Turn on KV quant for >2K context or RAG",
            "Prefer GQA models",
        ]
    )
    b += code('./scripts/run_article.sh 2 "Mac M3"')
    b += body("← Part 2  ·  Next → Part 4: Prefill & TTFT")
    kit(
        slug="02-kv-cache-quantization",
        title="The Hidden Memory Hog: KV Cache Quantization",
        subtitle="Why short benches look boring — and when 4-bit KV actually saves you",
        featured=f"{IMG}/thumbnails/thumb_02_kv_cache.png",
        featured_caption="KV cache quantization — Part 3",
        tags=["Machine Learning", "LLM", "Artificial Intelligence", "Apple", "Programming"],
        series="Local LLMs on Apple Silicon — Part 3 of 7",
        blocks=b,
    )


def art03() -> None:
    b: list[str] = []
    b += body(
        "Users blame “slow AI” on streaming speed.",
        "Often the real pain is earlier: time-to-first-token — the pause before the first character.",
    )
    b += subhead("Prefill vs decode")
    b += image(f"{IMG}/workflows/03_prefill_vs_decode.png", "Two phases, two bottlenecks")
    b += bullets(
        [
            "Prefill → TTFT (attention over the whole prompt)",
            "Decode → tok/s (weight bandwidth)",
        ]
    )
    b += subhead("FlashAttention — exact, not approximate")
    b += image(f"{IMG}/papers/dao_flashattention_redraw.png", "Original redraw — FlashAttention IO (Dao et al., 2022/23)")
    b += image(f"{IMG}/papers/milakov_online_softmax_redraw.png", "Original redraw — online softmax (Milakov & Gimelshein, 2018)")
    b += quote("Fun fact: FlashAttention computes the same math as naive attention. It just refuses to materialize the giant score matrix in slow memory.")
    b += subhead("The quadratic wall (real numbers)")
    b += body("Llama 3.1 8B, w4, Mac M3:")
    b += bullets(
        [
            "p=256 → ~2.4 s TTFT",
            "p=512 → ~3.1 s",
            "p=1024 → ~5.8 s",
            "p=2048 → ~15.4 s",
        ]
    )
    b += image(f"{IMG}/03_prefill_ttft.png", "TTFT vs prompt shape")
    b += image(f"{IMG}/03_ttft_vs_prompt_curve.png", "Measured TTFT vs a ∝ T² reference")
    b += image(f"{IMG}/07_workload_ttft.png", "rag_agent workload ≈ 31 s TTFT on M3")
    b += subhead("What to do in product")
    b += bullets(
        [
            "Chat → shorten system prompts, enable prefill chunking",
            "RAG → fewer chunks, prefix cache, don’t paste the whole PDF",
            "Long writing → optimize tok/s (w4) after TTFT is acceptable",
        ]
    )
    b += code('./scripts/run_article.sh 3 "Mac M3"')
    b += body("← Part 3  ·  Next → Part 5: Model Size Ladder")
    kit(
        slug="03-prefill-and-ttft",
        title="Why Your Chatbot Feels Slow Before the First Word",
        subtitle="Prefill, FlashAttention intuition, and TTFT curves that go quadratic",
        featured=f"{IMG}/thumbnails/thumb_03_prefill_ttft.png",
        featured_caption="Prefill & TTFT — Part 4",
        tags=["Machine Learning", "LLM", "Artificial Intelligence", "Apple", "UX"],
        series="Local LLMs on Apple Silicon — Part 4 of 7",
        blocks=b,
    )


def art04() -> None:
    b: list[str] = []
    b += body(
        "“Which model should I run locally?” is two questions:",
        "Will it fit? Will it be fast enough?",
    )
    b += image(f"{IMG}/workflows/04_fit_ladder.png", "Decision ladder for 24 GB unified memory")
    b += subhead("The w4 ladder on Mac M3")
    b += bullets(
        [
            "Qwen 0.5B — 238 tok/s · 0.64 GB",
            "Llama 3.2 1B — 112 tok/s · 1.2 GB",
            "Qwen 3B — 48 tok/s · 2.2 GB",
            "Llama 8B — 21 tok/s · 5.1 GB",
            "Gemma 9B — 15 tok/s · 5.9 GB",
        ]
    )
    b += image(f"{IMG}/04_model_size_ladder.png", "tok/s and memory across sizes @ w4")
    b += image(f"{IMG}/04_ladder_scatter.png", "Memory vs speed scatter")
    b += image(f"{IMG}/01_efficiency_tps_per_gb.png", "Efficiency = tok/s per GB @ w4")
    b += quote("Fun fact: Qwen 0.5B @ w4 exceeds 238 tok/s on M3 — faster than most people type.")
    b += subhead("M5 Max extends the ladder")
    b += image(f"{IMG}/04_m5_extended_ladder.png", "M5 Max w4 ladder through larger models")
    b += image(f"{IMG}/01_m3_vs_m5_w4.png", "Same checkpoints, different silicon")
    b += subhead("Cheat sheet (24 GB)")
    b += bullets(
        [
            "IDE copilot → 7B w4",
            "Offline chat → 8B w4",
            "Router / draft model → 0.5B–1.5B w4",
            "Max quality that still fits → 9B w4 or 8B w8",
        ]
    )
    b += code('./scripts/run_article.sh 4 "Mac M3"')
    b += body("← Part 4  ·  Next → Part 6: Full Stack")
    kit(
        slug="04-model-size-ladder",
        title="From 0.5B to 70B: What Fits on Apple Silicon",
        subtitle="A practical size ladder with M3 and M5 Max numbers",
        featured=f"{IMG}/thumbnails/thumb_04_model_ladder.png",
        featured_caption="Model size ladder — Part 5",
        tags=["Machine Learning", "LLM", "Apple", "Artificial Intelligence", "Data Science"],
        series="Local LLMs on Apple Silicon — Part 5 of 7",
        blocks=b,
    )


def art05() -> None:
    b: list[str] = []
    b += body(
        "Blog posts love clean A/B tests.",
        "Real local inference turns several knobs at once.",
    )
    b += image(f"{IMG}/workflows/05_optimization_funnel.png", "Stacking funnel — fp16 → w4 → +KV → +prefill")
    b += image(f"{IMG}/workflows/05_decision_tree.png", "Pick the lever that matches your pain")
    b += subhead("The headline result (Mac M3, Llama 8B)")
    b += bullets(
        [
            "fp16 — 16.3 GB · 5.6 tok/s",
            "w4+kv+prefill — 5.1 GB · 19.9 tok/s (~3.5×)",
        ]
    )
    b += image(f"{IMG}/05_full_stack.png", "fp16 vs optimized — speed and memory")
    b += image(f"{IMG}/05_full_stack_two_models.png", "Llama and Mistral both jump when stacked")
    b += image(f"{IMG}/05_full_stack_memory.png", "Both models drop to ~5 GB peak")
    b += subhead("M5 Max: the 16-config matrix")
    b += image(f"{IMG}/05_m5_config_matrix.png", "Llama 8B full config matrix on M5 Max")
    b += image(f"{IMG}/05_m3_m5_full_stack.png", "Same stack on M3 vs M5 Max")
    b += quote("Fun fact: A full article sweep can take hours. Isolate each config so one Metal OOM doesn’t kill the batch.")
    b += subhead("Daily driver recipe (24 GB)")
    b += body("Use w4+kv_cache+prefill on llama3-8b / mistral-7b / qwen-7b.")
    b += body("Expect ~5 GB peak and ~18–21 tok/s on M3.")
    b += code('python scripts/run_benchmark.py --preset llama3-8b --config w4+kv_cache+prefill --hardware "Mac M3"')
    b += body("← Part 5  ·  Next → Part 7: Speculative Decoding")
    kit(
        slug="05-full-optimization-stack",
        title="Stacking Optimizations: 3.5× Faster Than FP16",
        subtitle="The daily-driver recipe on a 24 GB Mac — and the full M5 Max matrix",
        featured=f"{IMG}/thumbnails/thumb_05_full_stack.png",
        featured_caption="Full optimization stack — Part 6",
        tags=["Machine Learning", "Optimization", "LLM", "Apple", "Artificial Intelligence"],
        series="Local LLMs on Apple Silicon — Part 6 of 7",
        blocks=b,
    )


def art06() -> None:
    b: list[str] = []
    b += body(
        "A small draft model proposes tokens. The large target verifies them in one parallel pass.",
        "When the draft is right, you emit multiple tokens per expensive step — without retraining.",
    )
    b += image(f"{IMG}/papers/leviathan_speculative_redraw.png", "Original redraw — draft/verify (Leviathan / Chen, 2023)")
    b += image(f"{IMG}/workflows/06_accept_reject.png", "Accept matching prefix; reject and resample at first mismatch")
    b += image(f"{IMG}/papers/cai_medusa_redraw.png", "Original redraw — Medusa-style drafting (Cai et al., 2024)")
    b += subhead("The clean win: Qwen-7B on Mac M3")
    b += bullets(
        [
            "Baseline w4 — 15.9 tok/s",
            "Speculative (Qwen 0.5B draft) — 28.3 tok/s",
            "Acceptance α — 74.2%",
        ]
    )
    b += image(f"{IMG}/06_speculative_qwen-7b.png", "1.78× throughput at 74% acceptance")
    b += image(f"{IMG}/06_speculative_speed_memory.png", "Big speed gain for ~0.3 GB extra RAM")
    b += subhead("Honest failures")
    b += body("On M3, Llama and Mistral speculative runs errored (draft/tokenizer pairing / memory).")
    b += body("On M5 Max, Qwen still wins (122 → 170 tok/s). Llama speculative was slightly slower (113 → 110) at 59% acceptance.")
    b += image(f"{IMG}/06_spec_m3_m5_qwen.png", "Qwen speculative on M3 vs M5 Max")
    b += image(f"{IMG}/06_spec_speedup_vs_accept.png", "Speedup vs acceptance — low α can erase the win")
    b += quote("Fun fact: Speculative decoding can make you slower if the draft is wrong too often. Measure α. Don’t assume.")
    b += subhead("Do this")
    b += bullets(
        [
            "Same family + same tokenizer",
            "Tiny draft (0.5B–1B)",
            "Long generations",
            "Budget RAM for two models",
        ]
    )
    b += code('./scripts/run_article.sh 6 "Mac M3"')
    b += body("← Part 6  ·  Bonus → Context, RAG & Prefix Cache")
    kit(
        slug="06-speculative-decoding",
        title="Draft Models: Free Speed Without Retraining",
        subtitle="74% acceptance and 1.8× on Qwen — plus the case where speculation got slower",
        featured=f"{IMG}/thumbnails/thumb_06_speculative.png",
        featured_caption="Speculative decoding — Part 7",
        tags=["Machine Learning", "LLM", "Optimization", "Apple", "Artificial Intelligence"],
        series="Local LLMs on Apple Silicon — Part 7 of 7",
        blocks=b,
    )


def art07() -> None:
    b: list[str] = []
    b += body(
        "Short prompts hide sins.",
        "Paste a PDF into a local RAG app and three forces collide: quadratic prefill, growing KV, and falling tok/s.",
    )
    b += image(f"{IMG}/workflows/07_rag_wall.png", "Retrieve → stuff context → O(T²) prefill → multi-second TTFT")
    b += image(f"{IMG}/papers/pope_kv_scaling_redraw.png", "Original redraw — KV grows until it rivals weights")
    b += subhead("Context length vs TTFT")
    b += bullets(
        [
            "256 tok → 1.4 s",
            "512 → 2.8 s",
            "1024 → 6.5 s",
            "2048 → 15.4 s",
        ]
    )
    b += image(f"{IMG}/07_context_ttft.png", "TTFT crosses 15 seconds at 2048 tokens on M3")
    b += image(f"{IMG}/07_context_dual_axis.png", "TTFT explodes while decode tok/s decays")
    b += image(f"{IMG}/07_context_m3_m5_panels.png", "M5 Max lowers the wall — it doesn’t remove the shape")
    b += subhead("Prefix cache: cold vs warm")
    b += image(f"{IMG}/workflows/07_prefix_cache_workflow.png", "Skip re-prefilling a stable system prompt")
    b += image(f"{IMG}/07_prefix_cache.png", "Cold 3180 ms → warm 1547 ms (~51% faster)")
    b += subhead("Workload stress")
    b += image(f"{IMG}/07_workload_panels.png", "Latency, throughput, and memory across workloads")
    b += image(f"{IMG}/07_workload_ttft.png", "rag_agent ≈ 31 s TTFT on M3")
    b += quote("If your local RAG demo feels broken, it’s probably prefill — not “tok/s.”")
    b += subhead("Mitigations that actually work")
    b += bullets(
        [
            "Retrieve less (top-3, not top-20)",
            "Prefix-cache system + tools",
            "w4 + KV quant",
            "Small router for easy queries",
        ]
    )
    b += code('./scripts/run_article.sh 7 "Mac M3"')
    b += body("← Back to Part 1  ·  Full series on GitHub")
    kit(
        slug="07-context-and-cache",
        title="The RAG Wall: Context, Cache, and Why Your Demo Freezes",
        subtitle="Quadratic TTFT, prefix caching, and workload stress on Apple Silicon",
        featured=f"{IMG}/thumbnails/thumb_07_rag_context.png",
        featured_caption="Context & prefix cache — Bonus",
        tags=["Machine Learning", "RAG", "LLM", "Apple", "Artificial Intelligence"],
        series="Local LLMs on Apple Silicon — Bonus",
        blocks=b,
    )


def write_howto() -> None:
    (OUT / "HOW_TO_PUBLISH.md").write_text(
        """# How to publish (Medium block editor — not HTML)

Medium is a **block editor**. These kits map to Medium’s controls:

| In the `.medium.txt` file | In Medium |
|---------------------------|-----------|
| **TITLE (Big T)** | Title field → large **Big T** |
| **SUBTITLE (Little T)** | Line under title → **Little T** |
| **FEATURED IMAGE** | Wide horizontal image under subtitle (story cover) |
| **SUBHEAD** | Section header style |
| **BODY** | Normal paragraph (keep short) |
| **PULL QUOTE** | Select text → quote button |
| **INLINE IMAGE** | `+` → Image → upload → caption |
| **BULLET / NUMBERED LIST** | `-` or `1.` shortcuts |
| **CODE BLOCK** | `+` → Code block |
| **TAGS** | Tag picker at the bottom |

## Steps for each article

1. Open `NN-*.medium.txt` and the matching `NN-*-meta.txt`
2. Medium → **New story**
3. Paste **TITLE** with **Big T**
4. Paste **SUBTITLE** with **Little T**
5. `+` → Image → upload the **FEATURED IMAGE** (from `images/thumbnails/`, 16:9)
6. Walk the file top to bottom:
   - SUBHEAD → apply section header
   - BODY → paste as normal text (one paragraph per BODY block)
   - PULL QUOTE → quote style
   - INLINE IMAGE → upload from the `UPLOAD:` path, paste CAPTION under it
7. Add **TAGS**
8. Publish at 95%, then fix typos
9. Spend ~1 hour on [`DISTRIBUTION.md`](DISTRIBUTION.md)

## Why not HTML?

Medium strips most HTML/CSS. Pasting HTML fights the editor.
These kits match how Medium actually wants you to write.

## Files

- `00-introduction.medium.txt` … `07-context-and-cache.medium.txt`
- matching `*-meta.txt` for quick copy of title/subtitle/tags/cover

Regenerate:

```bash
python scripts/build_medium_publish.py
```
"""
    )


def cleanup_html() -> None:
    for p in OUT.glob("*.html"):
        p.unlink()
        print(f"Removed {p.name}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cleanup_html()
    art00()
    art01()
    art02()
    art03()
    art04()
    art05()
    art06()
    art07()
    write_howto()
    leftovers = OUT / "leftovers"
    leftovers.mkdir(exist_ok=True)
    (leftovers / "README.md").write_text(
        "Long research drafts live in docs/medium/0*.md.\n"
        "Paste only the .medium.txt kits into Medium.\n"
    )
    print(f"Done → {OUT}")


if __name__ == "__main__":
    main()
