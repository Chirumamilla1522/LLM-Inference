#!/usr/bin/env python3
"""Emit Medium-ready HTML publish files (tight, image-broken, pull quotes)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "medium" / "publish"
IMG = "../images"  # relative from publish/ when documenting paths


def page(title: str, subtitle: str, series: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <meta name="description" content="{subtitle}" />
  <style>
    /* Preview only — Medium strips CSS on paste. Keep structure semantic. */
    body {{ font-family: Georgia, 'Times New Roman', serif; max-width: 740px;
           margin: 2rem auto; padding: 0 1.25rem; line-height: 1.7; color: #242424; }}
    h1 {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         font-size: 2.4rem; line-height: 1.15; margin-bottom: 0.4rem; }}
    .subtitle {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                font-size: 1.25rem; color: #6b6b6b; margin-bottom: 1.5rem; }}
    .series {{ font-style: italic; color: #6b6b6b; margin-bottom: 2rem; }}
    h2 {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         font-size: 1.55rem; margin-top: 2.4rem; line-height: 1.25; }}
    p {{ font-size: 1.2rem; margin: 1.1rem 0; }}
    blockquote {{ border-left: 3px solid #242424; margin: 1.8rem 0; padding: 0.2rem 0 0.2rem 1.2rem;
                 font-style: italic; font-size: 1.35rem; }}
    figure {{ margin: 2rem 0; }}
    figcaption {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                 font-size: 0.95rem; color: #6b6b6b; margin-top: 0.6rem; text-align: center; }}
    img {{ max-width: 100%; height: auto; }}
    ul, ol {{ font-size: 1.15rem; }}
    li {{ margin: 0.4rem 0; }}
    .img-slot {{ background: #f2f2f2; border: 1px dashed #bbb; padding: 1.2rem; text-align: center;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 0.9rem; color: #666; }}
    .cta {{ background: #fafafa; padding: 1.2rem 1.4rem; margin: 2.5rem 0; border-radius: 4px; }}
    .tags {{ color: #6b6b6b; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 0.95rem; }}
  </style>
</head>
<body>
<article>
  <p class="series">{series}</p>
  <h1>{title}</h1>
  <p class="subtitle">{subtitle}</p>
{body}
</article>
</body>
</html>
"""


def fig(rel: str, caption: str) -> str:
    return f"""  <figure>
    <div class="img-slot">UPLOAD THIS IMAGE IN MEDIUM<br/><code>{rel}</code></div>
    <img src="{rel}" alt="{caption[:80]}" />
    <figcaption>{caption}</figcaption>
  </figure>"""


def meta(path: Path, title: str, subtitle: str, tags: list[str], cover: str, series: str) -> None:
    path.write_text(
        f"""TITLE: {title}
SUBTITLE: {subtitle}
SERIES: {series}
COVER: {cover}
TAGS: {', '.join(tags)}
READ_TIME_TARGET: 8–12 min
NOTE: Paste the matching .html body into Medium. Upload COVER as story image (wide 16:9).
"""
    )


def write(name: str, title: str, subtitle: str, series: str, tags: list[str], cover: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.html").write_text(page(title, subtitle, series, body))
    meta(OUT / f"{name}-meta.txt", title, subtitle, tags, cover, series)
    print(f"Wrote {name}.html")


# ---------------------------------------------------------------------------
# Articles (tight Medium format)
# ---------------------------------------------------------------------------

def art00() -> None:
    body = f"""
{fig(f'{IMG}/thumbnails/thumb_00_introduction.png', 'Wide cover — use as Medium story header')}

  <p>I loaded Meta’s Llama 3.1 8B on a MacBook Pro and watched Activity Monitor go red.</p>
  <p>No cloud bill. No NVIDIA card. The model ran.</p>
  <p>It also felt like dial-up: about <strong>5 tokens per second</strong>, with a multi-second freeze before the first word.</p>
  <p>That gap — between “it runs” and “I’d use this every day” — is this series.</p>

  <h2>Why Apple Silicon changes the rules</h2>
  <p>On a gaming PC, GPU VRAM is a separate pool. On Apple Silicon, CPU and GPU share <strong>one unified memory pool</strong>.</p>
  <p>Your browser tabs and your 8B weights fight for the same bytes.</p>

{fig(f'{IMG}/workflows/00_unified_memory.png', 'Unified memory — weights, KV cache, OS, and apps share one DRAM pool')}

  <blockquote>Fun fact: Local LLMs on Mac only became practical once consumer unified memory crossed roughly 16–24 GB.</blockquote>

  <h2>The only three metrics that matter</h2>
{fig(f'{IMG}/workflows/00_inference_pipeline.png', 'Load → prefill → first token (TTFT) → decode (tok/s)')}

  <ul>
    <li><strong>Peak memory (GB)</strong> — will it fit without swap?</li>
    <li><strong>TTFT (ms)</strong> — how long you stare at a blank cursor</li>
    <li><strong>Decode tok/s</strong> — how fast the answer streams</li>
  </ul>
  <p>Optimize the wrong one and your “faster model” still feels broken.</p>

  <h2>The brutal FP16 baseline</h2>
  <p>Llama 3.1 8B, FP16, Mac M3 (24 GB), 512-token prompt, 128-token generation:</p>
  <ul>
    <li><strong>16.33 GB</strong> peak memory</li>
    <li><strong>2,651 ms</strong> to first token</li>
    <li><strong>5.3 tok/s</strong> decode</li>
  </ul>

{fig(f'{IMG}/00_intro_hardware_compare.png', 'Same model family — precision and silicon change everything')}

  <p>On Mac M5 Max, the same FP16 demo jumps to roughly <strong>34 tok/s</strong> with far lower TTFT. Silicon matters. Software still matters more for fitting.</p>

{fig(f'{IMG}/papers/williams_roofline_redraw.png', 'Original redraw — Roofline idea (Williams et al., 2009). Decode is often bandwidth-bound.')}

  <h2>What this series will do</h2>
  <ol>
    <li>Weight quantization</li>
    <li>KV cache quantization</li>
    <li>Prefill &amp; TTFT</li>
    <li>Model size ladder</li>
    <li>Full optimization stack</li>
    <li>Speculative decoding</li>
    <li>Bonus: context, RAG, prefix cache</li>
  </ol>
  <p>Every number comes from reproducible JSON in an open harness on <a href="https://github.com/ml-explore/mlx">MLX</a>.</p>

{fig(f'{IMG}/01_heatmap_tps.png', 'Sneak peek — tok/s heatmap across models and bit-widths on Mac M3')}

  <h2>A 10-minute sanity check</h2>
  <p>Before you trust any blog number — including mine:</p>
  <ol>
    <li>Pick one model you care about.</li>
    <li>Run FP16 and 4-bit only.</li>
    <li>Confirm memory drops ~2–3× and decode rises ~3× on an M3-class chip.</li>
    <li>Measure TTFT with <em>your</em> prompt length.</li>
  </ol>

  <div class="cta">
    <p><strong>Reproduce:</strong> <a href="https://github.com/Chirumamilla1522/LLM-Inference">github.com/Chirumamilla1522/LLM-Inference</a></p>
    <p><code>./scripts/run_article.sh 0 "Mac M3"</code></p>
  </div>

  <p><em>Next → Part 2: 4-Bit Weights Changed Everything</em></p>
  <p class="tags">Tags: Machine Learning · Apple · LLM · MLX · Local AI · Apple Silicon</p>
"""
    write(
        "00-introduction",
        "Running 8B LLMs on a MacBook: What Actually Matters",
        "Unified memory, the metrics that matter, and a brutal FP16 baseline on Apple Silicon",
        "Local LLMs on Apple Silicon — Part 1 of 7",
        ["Machine Learning", "Apple", "LLM", "Artificial Intelligence", "Programming"],
        f"{IMG}/thumbnails/thumb_00_introduction.png",
        body,
    )


def art01() -> None:
    body = f"""
{fig(f'{IMG}/thumbnails/thumb_01_weight_quantization.png', 'Wide cover — Medium story header')}

  <p>An 8B model in FP16 needs ~16 GB just for weights.</p>
  <p>On a 24 GB MacBook, that leaves almost nothing for the OS, your editor, and the KV cache.</p>
  <p><strong>Weight quantization</strong> is the highest-leverage change for local Mac inference.</p>

  <h2>How it works (without copying paper figures)</h2>
  <p>We store each weight with fewer bits — usually 8, 4, or 2 — plus a tiny scale.</p>

{fig(f'{IMG}/papers/jacob_affine_quant_redraw.png', 'Original redraw — affine quantization idea (Jacob et al., 2018)')}

{fig(f'{IMG}/papers/frantar_gptq_redraw.png', 'Original redraw — GPTQ column-compensation idea (Frantar et al., 2022)')}

{fig(f'{IMG}/papers/lin_awq_redraw.png', 'Original redraw — AWQ salient-channel idea (Lin et al., 2023)')}

  <blockquote>Fun fact: GPTQ was built for 175B-class models that couldn’t fit on one GPU at FP16. The same math now makes 8B models comfortable on a laptop.</blockquote>

  <h2>Why fewer bits also make decode faster</h2>
  <p>Each decode step often reads nearly all weights from memory. Fewer bytes per weight → higher tok/s on a bandwidth-bound chip.</p>

{fig(f'{IMG}/papers/williams_roofline_redraw.png', 'Original redraw — Roofline: LLM decode sits on the bandwidth slope')}

  <h2>Llama 3.1 8B on Mac M3</h2>
  <ul>
    <li><strong>fp16</strong> — 16.3 GB · 5.8 tok/s</li>
    <li><strong>w8</strong> — 9.0 GB · 11.3 tok/s (~1.9×)</li>
    <li><strong>w4</strong> — 5.1 GB · <strong>20.5 tok/s (~3.5×)</strong></li>
    <li><strong>w2</strong> — 3.1 GB · 35.8 tok/s (~6×)</li>
  </ul>

{fig(f'{IMG}/01_weight_quant_llama3-8b.png', 'Memory vs speed as bit-width drops')}
{fig(f'{IMG}/01_pareto_memory_speed.png', 'Pareto frontier — w4 is the practical sweet spot on 24 GB')}
{fig(f'{IMG}/01_speedup_vs_fp16.png', 'Explicit speedup vs FP16')}

  <h2>All 14 models (the heatmap)</h2>
{fig(f'{IMG}/01_heatmap_tps.png', 'Decode tok/s — every model × bit-width on Mac M3')}
{fig(f'{IMG}/01_heatmap_memory.png', 'Peak memory — FP16 is the red zone on 24 GB')}
{fig(f'{IMG}/01_speedup_all_models.png', 'fp16→w4 speedup and memory shrink across the board')}

  <h2>M3 vs M5 Max</h2>
  <p>Same w4 checkpoints. Different silicon.</p>
  <ul>
    <li>Llama 8B w4: <strong>20.5 → 112 tok/s</strong></li>
    <li>Qwen 0.5B w4: <strong>215 → 581 tok/s</strong></li>
  </ul>

{fig(f'{IMG}/01_m3_vs_m5_w4.png', 'M3 vs M5 Max at w4 — annotations show the speedup factor')}
{fig(f'{IMG}/01_llama_m3_m5_all_bits.png', 'Llama 8B across every bit-width on both chips')}

  <h2>What you should actually run</h2>
  <ul>
    <li><strong>16 GB Mac</strong> — 3B–7B @ w4</li>
    <li><strong>24 GB Mac</strong> — 8B @ w4 as daily driver</li>
    <li><strong>Skip</strong> FP16 8B as your everyday chat config</li>
  </ul>

  <div class="cta">
    <p><strong>Reproduce:</strong> <code>./scripts/run_article.sh 1 "Mac M3"</code></p>
    <p>Repo: <a href="https://github.com/Chirumamilla1522/LLM-Inference">LLM-Inference</a></p>
  </div>

  <p><em>← Part 1 · Next → Part 3: KV Cache</em></p>
  <p class="tags">Tags: Machine Learning · Quantization · LLM · Apple · Artificial Intelligence</p>
"""
    write(
        "01-weight-quantization",
        "4-Bit Weights Changed Everything on My M3 Mac",
        "Affine quantization, GPTQ/AWQ ideas redrawn, and 14-model heatmaps on Apple Silicon",
        "Local LLMs on Apple Silicon — Part 2 of 7",
        ["Machine Learning", "Quantization", "LLM", "Apple", "Artificial Intelligence"],
        f"{IMG}/thumbnails/thumb_01_weight_quantization.png",
        body,
    )


def art02() -> None:
    body = f"""
{fig(f'{IMG}/thumbnails/thumb_02_kv_cache.png', 'Wide cover — Medium story header')}

  <p>Weight quantization gets the spotlight.</p>
  <p>Once generation starts, something else grows: the <strong>KV cache</strong> — keys and values for every token in context.</p>
  <p>For short chats it barely shows up in tok/s. For RAG, it’s the second memory bill.</p>

  <h2>How the cache works</h2>
{fig(f'{IMG}/workflows/02_kv_cache_workflow.png', 'KV grows linearly with sequence length; 4-bit KV ≈ ¼ the footprint')}
{fig(f'{IMG}/papers/vaswani_attention_redraw.png', 'Original redraw — attention (Vaswani et al., 2017); decode caches K/V')}
{fig(f'{IMG}/papers/pope_kv_scaling_redraw.png', 'Original redraw — inspired by Pope et al. (2022): weights flat, KV grows')}

  <h2>GQA: shrink heads before you quantize</h2>
{fig(f'{IMG}/papers/ainslie_gqa_redraw.png', 'Original redraw — GQA vs MHA (Ainslie et al., 2023)')}

  <blockquote>Llama 3, Mistral, and Qwen already cut KV heads with GQA. 4-bit KV stacks on top of that.</blockquote>

  <h2>Why our short-context bench “does nothing”</h2>
  <p>At 512 prompt + 128 gen on Mac M3:</p>
  <ul>
    <li>Llama 8B: 20.7 → 20.4 tok/s</li>
    <li>Mistral 7B: 21.6 → 21.2</li>
    <li>Qwen 7B: 21.8 → 21.4</li>
  </ul>

{fig(f'{IMG}/02_kv_cache_compare.png', 'Short context: throughput almost unchanged')}
{fig(f'{IMG}/02_kv_long_generation.png', 'Longer generation: still weight-bound at laptop batch size 1')}

  <p>The win appears at long context, multi-session serving, or tight RAM — not in a 640-token microbench.</p>

{fig(f'{IMG}/07_context_dual_axis.png', 'Where KV pressure shows up — TTFT explodes as prompts grow')}
{fig(f'{IMG}/papers/kwon_paged_attention_redraw.png', 'Original redraw — paged KV idea (Kwon et al., 2023) for multi-request serving')}

  <h2>When to enable it</h2>
  <ul>
    <li>Always quantize <strong>weights</strong> first (w4)</li>
    <li>Turn on KV quant for &gt;2K context or RAG</li>
    <li>Prefer GQA models</li>
  </ul>

  <div class="cta">
    <p><code>./scripts/run_article.sh 2 "Mac M3"</code></p>
    <p><a href="https://github.com/Chirumamilla1522/LLM-Inference">Reproduce in the repo</a></p>
  </div>

  <p><em>← Part 2 · Next → Part 4: Prefill &amp; TTFT</em></p>
  <p class="tags">Tags: Machine Learning · LLM · Transformers · Apple · Artificial Intelligence</p>
"""
    write(
        "02-kv-cache-quantization",
        "The Hidden Memory Hog: KV Cache Quantization",
        "Why short benches look boring — and when 4-bit KV actually saves you",
        "Local LLMs on Apple Silicon — Part 3 of 7",
        ["Machine Learning", "LLM", "Artificial Intelligence", "Apple", "Programming"],
        f"{IMG}/thumbnails/thumb_02_kv_cache.png",
        body,
    )


def art03() -> None:
    body = f"""
{fig(f'{IMG}/thumbnails/thumb_03_prefill_ttft.png', 'Wide cover — Medium story header')}

  <p>Users blame “slow AI” on streaming speed.</p>
  <p>Often the real pain is earlier: <strong>time-to-first-token</strong> — the pause before the first character.</p>

  <h2>Prefill vs decode</h2>
{fig(f'{IMG}/workflows/03_prefill_vs_decode.png', 'Two phases, two bottlenecks — optimize the one users feel')}

  <ul>
    <li><strong>Prefill</strong> → TTFT (attention over the whole prompt)</li>
    <li><strong>Decode</strong> → tok/s (weight bandwidth)</li>
  </ul>

  <h2>FlashAttention — exact, not approximate</h2>
{fig(f'{IMG}/papers/dao_flashattention_redraw.png', 'Original redraw — FlashAttention IO pattern (Dao et al., 2022/23)')}
{fig(f'{IMG}/papers/milakov_online_softmax_redraw.png', 'Original redraw — online softmax (Milakov & Gimelshein, 2018)')}

  <blockquote>Fun fact: FlashAttention computes the same math as naive attention. It just refuses to materialize the giant score matrix in slow memory.</blockquote>

  <h2>The quadratic wall (real numbers)</h2>
  <p>Llama 3.1 8B, w4, Mac M3:</p>
  <ul>
    <li>p=256 → ~2.4 s TTFT</li>
    <li>p=512 → ~3.1 s</li>
    <li>p=1024 → ~5.8 s</li>
    <li>p=2048 → <strong>~15.4 s</strong></li>
  </ul>

{fig(f'{IMG}/03_prefill_ttft.png', 'TTFT vs prompt shape')}
{fig(f'{IMG}/03_ttft_vs_prompt_curve.png', 'Measured TTFT vs a ∝ T² reference')}
{fig(f'{IMG}/07_workload_ttft.png', 'rag_agent workload ≈ 31 s TTFT on M3 — unusable for interactive UX')}

  <h2>What to do in product</h2>
  <ul>
    <li>Chat → shorten system prompts, enable prefill chunking</li>
    <li>RAG → fewer chunks, prefix cache, don’t paste the whole PDF</li>
    <li>Long writing → optimize tok/s (w4) after TTFT is acceptable</li>
  </ul>

  <div class="cta">
    <p><code>./scripts/run_article.sh 3 "Mac M3"</code></p>
    <p><a href="https://github.com/Chirumamilla1522/LLM-Inference">Repo</a></p>
  </div>

  <p><em>← Part 3 · Next → Part 5: Model Size Ladder</em></p>
  <p class="tags">Tags: Machine Learning · LLM · UX · Apple · Artificial Intelligence</p>
"""
    write(
        "03-prefill-and-ttft",
        "Why Your Chatbot Feels Slow Before the First Word",
        "Prefill, FlashAttention intuition, and TTFT curves that go quadratic",
        "Local LLMs on Apple Silicon — Part 4 of 7",
        ["Machine Learning", "LLM", "Artificial Intelligence", "Apple", "UX"],
        f"{IMG}/thumbnails/thumb_03_prefill_ttft.png",
        body,
    )


def art04() -> None:
    body = f"""
{fig(f'{IMG}/thumbnails/thumb_04_model_ladder.png', 'Wide cover — Medium story header')}

  <p>“Which model should I run locally?” is two questions:</p>
  <ol>
    <li>Will it fit?</li>
    <li>Will it be fast enough?</li>
  </ol>

{fig(f'{IMG}/workflows/04_fit_ladder.png', 'Decision ladder for 24 GB unified memory')}

  <h2>The w4 ladder on Mac M3</h2>
  <ul>
    <li>Qwen 0.5B — <strong>238 tok/s</strong> · 0.64 GB</li>
    <li>Llama 3.2 1B — 112 tok/s · 1.2 GB</li>
    <li>Qwen 3B — 48 tok/s · 2.2 GB</li>
    <li>Llama 8B — 21 tok/s · 5.1 GB</li>
    <li>Gemma 9B — 15 tok/s · 5.9 GB</li>
  </ul>

{fig(f'{IMG}/04_model_size_ladder.png', 'tok/s and memory across sizes @ w4')}
{fig(f'{IMG}/04_ladder_scatter.png', 'Memory vs speed scatter — pick your point on the frontier')}
{fig(f'{IMG}/01_efficiency_tps_per_gb.png', 'Efficiency = tok/s per GB @ w4')}

  <blockquote>Fun fact: Qwen 0.5B @ w4 exceeds 238 tok/s on M3 — faster than most people type.</blockquote>

  <h2>M5 Max extends the ladder</h2>
{fig(f'{IMG}/04_m5_extended_ladder.png', 'M5 Max w4 ladder through larger models')}
{fig(f'{IMG}/01_m3_vs_m5_w4.png', 'Same checkpoints, different silicon')}

  <h2>Cheat sheet (24 GB)</h2>
  <ul>
    <li>IDE copilot → 7B w4</li>
    <li>Offline chat → 8B w4</li>
    <li>Router / draft model → 0.5B–1.5B w4</li>
    <li>Max quality that still fits → 9B w4 or 8B w8</li>
  </ul>

  <div class="cta">
    <p><code>./scripts/run_article.sh 4 "Mac M3"</code></p>
    <p><a href="https://github.com/Chirumamilla1522/LLM-Inference">Repo</a></p>
  </div>

  <p><em>← Part 4 · Next → Part 6: Full Stack</em></p>
  <p class="tags">Tags: Machine Learning · LLM · Apple · Artificial Intelligence · Benchmark</p>
"""
    write(
        "04-model-size-ladder",
        "From 0.5B to 70B: What Fits on Apple Silicon",
        "A practical size ladder with M3 and M5 Max numbers",
        "Local LLMs on Apple Silicon — Part 5 of 7",
        ["Machine Learning", "LLM", "Apple", "Artificial Intelligence", "Data Science"],
        f"{IMG}/thumbnails/thumb_04_model_ladder.png",
        body,
    )


def art05() -> None:
    body = f"""
{fig(f'{IMG}/thumbnails/thumb_05_full_stack.png', 'Wide cover — Medium story header')}

  <p>Blog posts love clean A/B tests.</p>
  <p>Real local inference turns several knobs at once.</p>

{fig(f'{IMG}/workflows/05_optimization_funnel.png', 'Stacking funnel — fp16 → w4 → +KV → +prefill')}
{fig(f'{IMG}/workflows/05_decision_tree.png', 'Pick the lever that matches your pain')}

  <h2>The headline result (Mac M3, Llama 8B)</h2>
  <ul>
    <li><strong>fp16</strong> — 16.3 GB · 5.6 tok/s</li>
    <li><strong>w4+kv+prefill</strong> — 5.1 GB · <strong>19.9 tok/s (~3.5×)</strong></li>
  </ul>

{fig(f'{IMG}/05_full_stack.png', 'fp16 vs optimized — speed and memory')}
{fig(f'{IMG}/05_full_stack_two_models.png', 'Llama and Mistral both jump when stacked')}
{fig(f'{IMG}/05_full_stack_memory.png', 'Both models drop to ~5 GB peak')}

  <h2>M5 Max: the 16-config matrix</h2>
{fig(f'{IMG}/05_m5_config_matrix.png', 'Llama 8B full config matrix on M5 Max — tok/s')}
{fig(f'{IMG}/05_m3_m5_full_stack.png', 'Same stack on M3 vs M5 Max')}

  <blockquote>Fun fact: A full article sweep can take hours. Isolate each config in a subprocess so one Metal OOM doesn’t kill the batch.</blockquote>

  <h2>Daily driver recipe (24 GB)</h2>
  <p><code>w4+kv_cache+prefill</code> on llama3-8b / mistral-7b / qwen-7b.</p>
  <p>Expect ~5 GB peak and ~18–21 tok/s on M3.</p>

  <div class="cta">
    <p><code>python scripts/run_benchmark.py --preset llama3-8b --config w4+kv_cache+prefill --hardware "Mac M3"</code></p>
    <p><a href="https://github.com/Chirumamilla1522/LLM-Inference">Repo</a></p>
  </div>

  <p><em>← Part 5 · Next → Part 7: Speculative Decoding</em></p>
  <p class="tags">Tags: Machine Learning · Optimization · LLM · Apple · Artificial Intelligence</p>
"""
    write(
        "05-full-optimization-stack",
        "Stacking Optimizations: 3.5× Faster Than FP16",
        "The daily-driver recipe on a 24 GB Mac — and the full M5 Max matrix",
        "Local LLMs on Apple Silicon — Part 6 of 7",
        ["Machine Learning", "Optimization", "LLM", "Apple", "Artificial Intelligence"],
        f"{IMG}/thumbnails/thumb_05_full_stack.png",
        body,
    )


def art06() -> None:
    body = f"""
{fig(f'{IMG}/thumbnails/thumb_06_speculative.png', 'Wide cover — Medium story header')}

  <p>A small <strong>draft</strong> model proposes tokens. The large <strong>target</strong> verifies them in one parallel pass.</p>
  <p>When the draft is right, you emit multiple tokens per expensive step — without retraining.</p>

{fig(f'{IMG}/papers/leviathan_speculative_redraw.png', 'Original redraw — draft/verify (Leviathan / Chen, 2023)')}
{fig(f'{IMG}/workflows/06_accept_reject.png', 'Accept the matching prefix; reject and resample at the first mismatch')}
{fig(f'{IMG}/papers/cai_medusa_redraw.png', 'Original redraw — Medusa-style multi-head drafting (Cai et al., 2024)')}

  <h2>The clean win: Qwen-7B on Mac M3</h2>
  <ul>
    <li>Baseline w4 — 15.9 tok/s</li>
    <li>Speculative (Qwen 0.5B draft) — <strong>28.3 tok/s</strong></li>
    <li>Acceptance α — <strong>74.2%</strong></li>
  </ul>

{fig(f'{IMG}/06_speculative_qwen-7b.png', '1.78× throughput at 74% acceptance')}
{fig(f'{IMG}/06_speculative_speed_memory.png', 'Big speed gain for ~0.3 GB extra RAM')}

  <h2>Honest failures</h2>
  <p>On M3, Llama and Mistral speculative runs errored (draft/tokenizer pairing / memory).</p>
  <p>On M5 Max, Qwen still wins (122 → 170 tok/s). Llama speculative was <strong>slightly slower</strong> (113 → 110) at 59% acceptance.</p>

{fig(f'{IMG}/06_spec_m3_m5_qwen.png', 'Qwen speculative on M3 vs M5 Max')}
{fig(f'{IMG}/06_spec_speedup_vs_accept.png', 'Speedup vs acceptance — low α can erase the win')}

  <blockquote>Fun fact: Speculative decoding can make you slower if the draft is wrong too often. Measure α. Don’t assume.</blockquote>

  <h2>Do this</h2>
  <ul>
    <li>Same family + same tokenizer</li>
    <li>Tiny draft (0.5B–1B)</li>
    <li>Long generations</li>
    <li>Budget RAM for two models</li>
  </ul>

  <div class="cta">
    <p><code>./scripts/run_article.sh 6 "Mac M3"</code></p>
    <p><a href="https://github.com/Chirumamilla1522/LLM-Inference">Repo</a></p>
  </div>

  <p><em>← Part 6 · Bonus → Context, RAG &amp; Prefix Cache</em></p>
  <p class="tags">Tags: Machine Learning · LLM · Optimization · Apple · Artificial Intelligence</p>
"""
    write(
        "06-speculative-decoding",
        "Draft Models: Free Speed Without Retraining",
        "74% acceptance and 1.8× on Qwen — plus the case where speculation got slower",
        "Local LLMs on Apple Silicon — Part 7 of 7",
        ["Machine Learning", "LLM", "Optimization", "Apple", "Artificial Intelligence"],
        f"{IMG}/thumbnails/thumb_06_speculative.png",
        body,
    )


def art07() -> None:
    body = f"""
{fig(f'{IMG}/thumbnails/thumb_07_rag_context.png', 'Wide cover — Medium story header')}

  <p>Short prompts hide sins.</p>
  <p>Paste a PDF into a local RAG app and three forces collide: quadratic prefill, growing KV, and falling tok/s.</p>

{fig(f'{IMG}/workflows/07_rag_wall.png', 'Retrieve → stuff context → O(T²) prefill → multi-second TTFT')}
{fig(f'{IMG}/papers/pope_kv_scaling_redraw.png', 'Original redraw — KV grows until it rivals weights')}

  <h2>Context length vs TTFT</h2>
  <ul>
    <li>256 tok → 1.4 s</li>
    <li>512 → 2.8 s</li>
    <li>1024 → 6.5 s</li>
    <li>2048 → <strong>15.4 s</strong></li>
  </ul>

{fig(f'{IMG}/07_context_ttft.png', 'TTFT crosses 15 seconds at 2048 tokens on M3')}
{fig(f'{IMG}/07_context_dual_axis.png', 'TTFT explodes while decode tok/s decays')}
{fig(f'{IMG}/07_context_m3_m5_panels.png', 'M5 Max lowers the wall — it doesn’t remove the shape')}

  <h2>Prefix cache: cold vs warm</h2>
{fig(f'{IMG}/workflows/07_prefix_cache_workflow.png', 'Skip re-prefilling a stable system prompt')}
{fig(f'{IMG}/07_prefix_cache.png', 'Cold 3180 ms → warm 1547 ms (~51% faster)')}

  <h2>Workload stress</h2>
{fig(f'{IMG}/07_workload_panels.png', 'Latency, throughput, and memory across workloads')}
{fig(f'{IMG}/07_workload_ttft.png', 'rag_agent ≈ 31 s TTFT on M3')}

  <blockquote>If your local RAG demo feels broken, it’s probably prefill — not “tok/s.”</blockquote>

  <h2>Mitigations that actually work</h2>
  <ul>
    <li>Retrieve less (top-3, not top-20)</li>
    <li>Prefix-cache system + tools</li>
    <li>w4 + KV quant</li>
    <li>Small router for easy queries</li>
  </ul>

  <div class="cta">
    <p><code>./scripts/run_article.sh 7 "Mac M3"</code></p>
    <p><a href="https://github.com/Chirumamilla1522/LLM-Inference">Repo</a></p>
  </div>

  <p><em>← Back to Part 1 · Full series on GitHub</em></p>
  <p class="tags">Tags: Machine Learning · RAG · LLM · Apple · Artificial Intelligence</p>
"""
    write(
        "07-context-and-cache",
        "The RAG Wall: Context, Cache, and Why Your Demo Freezes",
        "Quadratic TTFT, prefix caching, and workload stress on Apple Silicon",
        "Local LLMs on Apple Silicon — Bonus",
        ["Machine Learning", "RAG", "LLM", "Apple", "Artificial Intelligence"],
        f"{IMG}/thumbnails/thumb_07_rag_context.png",
        body,
    )


def leftovers_note() -> None:
    d = OUT / "leftovers"
    d.mkdir(parents=True, exist_ok=True)
    (d / "README.md").write_text(
        """# Leftovers

The long research drafts in `docs/medium/0*.md` are the full notebooks.

The HTML files in `docs/medium/publish/` are the **cut** Medium versions (tip: kill your darlings).

Do not paste leftovers into Medium. Mine them later for:

- appendices
- Twitter/LinkedIn carousels
- a “deep dive addendum” post
- talks / slide diagrams
"""
    )


def main() -> None:
    art00()
    art01()
    art02()
    art03()
    art04()
    art05()
    art06()
    art07()
    leftovers_note()
    print(f"Done → {OUT}")


if __name__ == "__main__":
    main()
