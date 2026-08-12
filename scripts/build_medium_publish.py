#!/usr/bin/env python3
"""Emit clean Medium-ready article text (finished story format, not HTML / not paste-kit chrome)."""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "medium" / "publish"
IMG_ROOT = "docs/medium/images"


def I(article: str, name: str) -> str:
    """Per-article image path."""
    return f"{IMG_ROOT}/{article}/{name}"



def write(slug: str, meta: dict[str, str], body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = body.strip() + "\n"
    (OUT / f"{slug}.medium.txt").write_text(text)
    (OUT / f"{slug}-meta.txt").write_text(
        "\n".join(
            [
                f"TITLE: {meta['title']}",
                f"SUBTITLE: {meta['subtitle']}",
                f"FEATURED IMAGE: {meta['featured']}",
                f"FEATURED CAPTION: {meta.get('featured_caption', '')}",
                f"SERIES: {meta['series']}",
                f"TAGS: {meta['tags']}",
                "",
            ]
        )
    )
    print(f"Wrote {slug}.medium.txt")


def art00() -> None:
    body = f"""
---

Running 8B LLMs on a MacBook: What Actually Matters

Unified memory, the inference pipeline, and reproducible benchmarks on Apple Silicon — with M3 vs. M5 Max numbers

Part 1 of 7 — Local LLMs on Apple Silicon

FEATURED IMAGE: {I("00-introduction", "thumb.png")}
CAPTION: Local LLMs on Apple Silicon — Part 1

---

I still remember the first time I loaded Meta's Llama 3.1 8B on a stock MacBook Pro and watched Activity Monitor paint the memory bar red.

No cloud bill. No discrete NVIDIA GPU. Just mlx_lm.generate, a Hugging Face checkpoint, and a quiet fan that wasn't quiet for long.

The model ran.

It also felt like dial-up.

I was getting roughly 5 tokens per second, with a multi-second pause before the first word appeared.

That gap — between "it runs on my laptop" and "I would actually use this every day" — is the entire point of this series.

Marketing slides say Apple Silicon is great for on-device AI. Blog posts say, "just use 4-bit."

But very few discussions show the complete measurement loop:

• The same prompt length
• The same generation length
• Multiple trials
• Median measurements
• Reproducible JSON results
• The same benchmark across different Apple Silicon generations

So I built an open benchmark harness using MLX, pointed it at MLX Community checkpoints, and ran it on a Mac M3 with 24 GB of unified memory and a Mac M5 Max.

This opening post is the map.

We'll look at:

• Why unified memory matters
• How LLM inference actually works
• The three metrics that matter most
• A brutally inefficient FP16 baseline
• What M3 vs. M5 Max tells us
• How to reproduce the measurements
• Where the rest of this series goes next

---

Why This Matters

Local LLMs stopped being a curiosity once 7–9B instruction-tuned models became good enough for coding assistants, summarization, and private RAG.

The question is no longer:

Can my Mac run an LLM?

The better questions are:

• Will it fit? Memory pressure and swapping can make the system feel frozen
• How long until the first token? Prefill latency directly affects chat UX
• How fast does it stream? Decode tokens/sec is what you actually feel while reading
• What should I optimize first? Blindly enabling optimizations wastes time

Cloud APIs hide these questions behind a price tag.

On a laptop, you are the ops team.

Apple Silicon makes local inference attractive because the CPU and GPU share memory and Metal provides accelerated computation. But that shared-memory architecture comes with a catch:

Your browser tabs, IDE, operating system, model weights, and KV cache all compete for the same memory pool.

Quantization, KV-cache optimizations, prefill improvements, and speculative decoding aren't just nice-to-have optimizations.

They're what make local inference practical.

This series is designed around reproducible numbers rather than vibes.

Every figure is generated from JSON results, and the benchmark commands are available in the accompanying repository. If your Mac is faster or slower than mine, you should be able to reproduce the experiment and find out why.

---

How Apple Silicon Changes the Game

On a conventional PC with a discrete GPU, system RAM and GPU VRAM are separate memory pools.

The CPU operates primarily from system RAM while the GPU uses VRAM. Data frequently has to move between them.

Apple Silicon takes a different approach.

CPU and GPU share a unified memory pool.

That means model weights, KV cache, macOS, Safari, your IDE, and other applications all consume the same physical memory.

IMAGE: {I("00-introduction", "fig1.png")}
Figure 1 — CPU and GPU share unified memory. Model weights, KV cache, macOS, and applications all compete for the same memory ceiling.

This is why a 24 GB Mac can be both surprisingly capable and surprisingly easy to overwhelm.

An 8B parameter model stored in FP16 requires approximately:

8 billion × 16 bits ÷ 8 ≈ 16 GB

And that's just the weights.

The KV cache grows with context length, while the operating system and your applications still need memory.

So on a 24 GB machine, an FP16 8B model leaves surprisingly little headroom.

A simple memory equation

For a model with N parameters stored using b bits per parameter:

Memory_weights ≈ (N × b) / (8 × 10⁹) GB

For an 8B model:

• FP16: ≈ 16 GB
• 8-bit: ≈ 8 GB
• 4-bit: ≈ 4 GB
• 2-bit: ≈ 2 GB

Real memory usage is higher because of runtime overhead, embeddings, activations, and the KV cache.

But the equation explains the basic tradeoff:

Quantization isn't just about making an LLM faster. On a laptop, it is often what makes the model usable in the first place.

---

The LLM Inference Pipeline

Every chat response has two major phases, and they have different bottlenecks.

Confusing them is one of the easiest ways to optimize the wrong thing.

IMAGE: {I("00-introduction", "fig2.png")}
Figure 2 — Load weights → prefill the prompt → generate the first token → autoregressive decode.

The three metrics I care about most are:

• Peak memory — Loading + KV growth — Unified memory capacity — Whether the model fits
• TTFT — Prefill — Compute + memory — How long the cursor freezes
• Decode tok/s — Autoregressive generation — Memory bandwidth — How quickly the answer streams

Prefill

During prefill, the model processes the existing prompt.

This phase is relatively parallel and generally benefits from greater compute capability.

Longer prompts generally mean more work and therefore greater time-to-first-token (TTFT).

Decode

Decode is different.

The model generates one token at a time.

Each new token requires the model to repeatedly access a large portion of its weights.

That makes memory bandwidth extremely important.

This is why reducing the number of bytes required per parameter can improve throughput even when the hardware still has plenty of theoretical FLOPS available.

The Roofline model provides a useful mental model here: when arithmetic intensity is low, memory bandwidth rather than peak compute becomes the limiting factor.

IMAGE: {I("00-introduction", "fig3.png")}
Figure 3 — Redraw of the Roofline model. LLM decode often operates in a bandwidth-limited regime.

And attention introduces another important concept: the KV cache.

IMAGE: {I("00-introduction", "fig4.png")}
Figure 4 — Scaled dot-product attention and the role of cached keys and values during autoregressive decoding.

The rest of this series therefore attacks the system one bottleneck at a time:

1. Weight quantization → memory + decode speed
2. KV-cache quantization → long-context memory
3. Prefill optimization → TTFT
4. Model-size scaling → capacity planning
5. Full-stack optimization → practical daily-driver configurations
6. Speculative decoding → generating multiple tokens more efficiently

---

Baseline: Llama 3.1 8B in FP16

Before optimizing anything, we need a baseline.

The baseline experiment is intentionally boring:

• Llama 3.1 8B
• FP16
• 512-token prompt
• 128 generated tokens
• 1 warmup trial
• 3 measured trials
• Median values reported

Mac M3 — 24 GB

• Configuration: FP16
• Peak Memory: 16.33 GB
• TTFT: 2,651 ms
• Decode: 5.3 tok/s

Sixteen gigabytes for a single model.

Almost three seconds before the first token.

About five tokens per second afterward.

This is the "it runs" baseline.

But it isn't what most people would consider a good local AI experience.

On a 24 GB Mac, there may only be around 6–8 GB left for macOS, your IDE, browser, applications, and the growing KV cache.

That's fragile.

---

Then We Tried the M5 Max

The same FP16 demo on the M5 Max produced:

• M3 — 16.33 GB · 2,651 ms · 5.3 tok/s
• M5 Max — 16.46 GB · 193 ms · 34.4 tok/s

The interesting part is memory.

It barely changed.

The model is still the model.

The weights still require roughly the same amount of memory.

What changed dramatically was how quickly the hardware could process them.

The M5 Max delivered approximately:

• 14× lower TTFT
• 6.5× higher decode throughput

This gives us an important distinction:

Newer hardware dramatically increases the performance ceiling, but it doesn't magically make an FP16 model smaller.

Capacity is still a memory problem.

Performance is a silicon problem.

Quantization attacks both.

IMAGE: {I("00-introduction", "fig5.png")}
Figure 5 — Hardware and precision comparison. Quantization dramatically reduces memory requirements while newer Apple Silicon raises the absolute performance ceiling.

---

What 4-Bit Does to the Same Model

Here's the first major preview of the next article.

On the M3, Llama 3.1 8B moves approximately from:

FP16 — 5.8 tok/s @ 16.3 GB

to:

W4 — 20.5 tok/s @ 5.1 GB

and eventually:

W2 — 35.8 tok/s @ 3.1 GB

That is the fundamental reason quantization deserves its own article.

The surprising result isn't merely that the model becomes smaller.

It's that the model can become smaller and faster at the same time.

Why?

Because decode frequently involves repeatedly moving model weights through memory.

Fewer bytes per parameter means less memory traffic.

---

M3 vs. M5 Max: What We Learn

• FP16 8B memory — M3 ~16.3 GB · M5 Max ~16.5 GB — Capacity is model-bound
• FP16 8B decode — M3 ~5.3–5.8 tok/s · M5 Max ~34–35 tok/s — Silicon generation matters
• FP16 TTFT (512 tokens) — M3 ~2.6–2.7 s · M5 Max ~0.19 s — Prefill benefits from more capable silicon
• FP16 8B headroom — M3 tight · M5 Max comfortable — 24 GB systems benefit strongly from quantization

If you only benchmark on an M5 Max, it is easy to underestimate why quantization matters.

If you only benchmark on an M3, it's easy to underestimate how much modern Max-class silicon can do.

That's why this series benchmarks across both.

The goal isn't to crown one machine a winner.

It's to understand which limitation comes from the software, which comes from the model, and which comes from the hardware.

---

Benchmark Methodology

Numbers are only useful if they're reproducible.

Every benchmark result is stored as JSON with a consistent schema.

Each result includes:

• schema_version — Current result format
• warmup_policy — Number of discarded warmup trials
• num_trials — Number of measured trials
• trials — Raw per-trial measurements
• stats — Median, p50, p95, standard deviation, min/max
• ttft_ms — Median time to first token
• throughput_tps — Median decode throughput
• memory_gb — Median/recorded peak memory

The standard benchmark configuration uses:

• Runtime: MLX + mlx-lm
• Models: MLX Community checkpoints
• Prompt: 512 tokens
• Generation: 128 tokens
• Warmup: 1 trial
• Measured trials: 3
• Isolation: Each configuration runs in its own subprocess

The repository is:

https://github.com/Chirumamilla1522/LLM-Inference

A representative result file looks like:

results/Mac_M3/article_00_introduction/llama3-8b/demo_fp16.json

The raw JSON is important.

Suppose one benchmark suddenly reports an unusually high token rate.

Rather than trusting a plot, we can inspect the individual trials.

Was it a warmup artifact?

Thermal variation?

A configuration mismatch?

A different checkpoint?

The raw data makes those questions answerable.

---

Why Medians Matter

A single benchmark run is not enough.

Metal initialization, thermal state, background applications, and other system activity can move results around.

That's why this benchmark uses:

1 warmup + 3 measured trials

and reports the median.

The goal isn't to manufacture perfect numbers.

It's to make comparisons more robust.

A benchmark should behave like a lab notebook, not a screenshot.

---

What "8B" Actually Means

One thing became obvious while building the benchmark:

"8B" is a model-size category, not a performance specification.

Different model families behave very differently even when they have similar parameter counts.

• Qwen 2.5 — Excellent small-model performance and useful draft candidates
• Llama 3 / 3.2 — Strong reference point for local chat and coding
• Mistral — Strong instruct baseline
• Gemma 2 — Interesting architectural and packaging differences
• Phi-3 / 3.5 — Small models with surprisingly strong capability
• DeepSeek-R1 Distill — Reasoning-oriented models useful for quality comparisons

Part 2 expands this into a broader model × precision benchmark.

---

The Dataset Behind the Series

This isn't a one-model experiment.

The benchmark harness has already generated hundreds of JSON runs across the M3 and M5 Max.

The next articles will turn those runs into:

• Throughput heatmaps
• Memory heatmaps
• Quantization speedups
• Memory-reduction comparisons
• Efficiency plots
• M3 vs. M5 comparisons
• Model-family comparisons
• Model-size ladders
• Long-context experiments
• Full-stack optimization results
• Speculative decoding experiments

IMAGE: {I("00-introduction", "fig6.png")}
Figure 6 — Decode throughput across models and bit-widths.

IMAGE: {I("00-introduction", "fig7.png")}
Figure 7 — Peak memory across the same model/precision matrix.

IMAGE: {I("00-introduction", "fig8.png")}
Figure 8 — FP16 → W4 decode speedup and memory reduction.

IMAGE: {I("00-introduction", "fig9.png")}
Figure 9 — Decode efficiency measured as tokens/sec per GB.

---

The Silicon Gap

The same quantized model can behave very differently across Apple Silicon generations.

IMAGE: {I("00-introduction", "fig10.png")}
Figure 10 — W4 performance comparison between M3 and M5 Max.

For Llama 3.1 8B, the difference becomes particularly interesting when looking across multiple precisions.

IMAGE: {I("00-introduction", "fig11.png")}
Figure 11 — Llama 3.1 8B across FP16, W8, W4, and W2 on M3 and M5 Max.

One result from the broader sweep is particularly striking:

M5 Max W4 reaches roughly 112 tok/s, exceeding M3 W2 at roughly 36 tok/s.

In other words, hardware generation and quantization are not competing explanations.

They stack.

---

Model Size Changes the Game

Smaller models are dramatically easier to run.

IMAGE: {I("00-introduction", "fig12.png")}
Figure 12 — W4 model-size ladder on Mac M3.

The M3 benchmark spans models from roughly 0.5B to 9B, with performance dropping as model size increases.

On the M5 Max, the usable model range extends considerably further.

IMAGE: {I("00-introduction", "fig13.png")}
Figure 13 — M5 Max extends the practical model-size range into larger models.

This leads to another important principle:

The best local model isn't necessarily the largest model that fits.

It's the model that provides the best balance between:

• Quality
• Memory usage
• TTFT
• Decode speed
• Context length
• Available system headroom

---

What Happens With Long Context?

The short benchmark is only the beginning.

Real applications increasingly involve long prompts:

• RAG
• Coding repositories
• Document analysis
• Agent workflows
• Conversation history

As context grows, the KV cache becomes increasingly important.

IMAGE: {I("00-introduction", "fig14.png")}
Figure 14 — Context-length experiments showing the impact of longer prompts on TTFT and throughput.

This is why KV-cache optimization is Part 3 of the series.

A model that feels great with a 500-token prompt can behave very differently with a 20K-token context.

---

The Full Optimization Stack

Eventually, these optimizations need to work together.

IMAGE: {I("00-introduction", "fig15.png")}
Figure 15 — Preview of the optimized inference stack across M3 and M5 Max.

The goal isn't to maximize one benchmark metric.

The goal is to create a configuration that you would actually want to use every day.

---

Speculative Decoding

The final part of the series explores speculative decoding.

Instead of relying entirely on a large model to generate every token sequentially, a smaller draft model proposes multiple tokens that the larger model can verify.

IMAGE: {I("00-introduction", "fig16.png")}
Figure 16 — Speculative decoding preview using Qwen-7B.

The preliminary measurements show approximately:

• M3: ~16 → 28 tok/s
• M5 Max: ~122 → 170 tok/s
• Acceptance rate: ~74%

This is particularly interesting because it attacks the autoregressive nature of decoding itself rather than simply reducing model size.

---

Practical Decision Guide

So what should you actually do?

• 16 GB unified memory — Start with 3B–7B W4 — Don't start with FP16 8B
• 24 GB unified memory — Start with 7B–8B W4 — Don't use FP16 8B as a daily driver
• 32–64 GB+ / Max — FP16 for quality tests, W4 for speed — Don't assume hardware alone solves latency
• You care about first-token latency — Measure TTFT — Don't optimize only tok/s
• Long RAG contexts — Plan KV memory early — Don't ignore cache growth
• Maximum streaming speed — Quantize weights first — Don't buy a new Mac first

My practical rule is simple:

1. Fit the model. Start with W4.
2. Measure your actual prompt. Especially TTFT.
3. Then optimize further. Only after those two steps should you worry about speculative decoding, runtime tweaks, or buying new hardware.

Skipping the first step is how people conclude:

"Local LLMs are unusable."

after running an 8B model in FP16 on a 24 GB Mac.

---

A 10-Minute Sanity Check

Before trusting any benchmark — including mine — run a small experiment on your own machine.

Step 1

Pick one model you actually care about.

For example: llama3-8b or mistral-7b

Step 2

Run only:

• FP16
• W4

Don't start with a 14-model sweep.

Step 3

Check whether:

• Memory falls by roughly 2–3×
• Decode throughput increases substantially

Step 4

Measure TTFT using a prompt length representative of your actual workload.

A 512-token benchmark is not necessarily representative of a 40-token chat message.

If your results look dramatically different from the expected trend, investigate:

• Incorrect checkpoint precision
• Thermal throttling
• Runtime configuration
• UI-wrapper overhead
• Background applications

Only after the basic experiment behaves correctly should you move to more advanced optimizations.

---

Four Things I Learned

1. Unified memory is both a feature and a tax.

There is no isolated VRAM pool protecting your desktop.

When the model consumes the memory, your applications feel it too.

2. Decode is often bandwidth-bound.

That's why a lower-bit model can simultaneously be smaller and faster — rather than simply smaller.

3. Medians beat single runs.

Warmup behavior, thermal state, and background processes matter.

A benchmark needs repeated measurements.

4. The M5 Max doesn't make FP16 weights smaller.

It makes the same memory footprint much more usable by dramatically improving compute and memory performance.

For multi-application workflows, quantization remains important.

---

Limitations

This series focuses on systems performance, not model quality.

There are several things these benchmarks don't measure yet:

• Perplexity and standardized quality evaluations
• MMLU-style benchmarks
• Battery life
• Sustained thermal behavior
• Non-MLX runtimes
• Ollama defaults
• llama.cpp
• PyTorch MPS
• Interactive UI overhead
• Network latency
• Tool-calling agent overhead

The goal of this series is narrower:

Understand what actually determines local LLM performance on Apple Silicon.

---

Reproduce the Benchmark

The benchmark repository contains scripts for running the experiments and regenerating the figures.

For the baseline:

./scripts/run_article.sh 0 "Mac M3"

# Optional — M5 Max
./scripts/run_article.sh 0 "Mac M5 Max"

# Regenerate figures
python scripts/plot_medium_diagrams.py
python scripts/plot_medium_charts.py --hardware "Mac M3"
python scripts/plot_medium_deep.py

The baseline result is stored under:

results/Mac_M3/article_00_introduction/llama3-8b/demo_fp16.json

The important part is that the measurements aren't just screenshots.

They're data.

You can inspect the raw JSON, regenerate the plots, and compare them against your own hardware.

Repo: https://github.com/Chirumamilla1522/LLM-Inference

---

What Comes Next?

This is only Part 1.

• Part 1 — Introduction — Metrics + unified memory
• Part 2 — Weight quantization — FP16 → W8 → W4 → W2
• Part 3 — KV-cache quantization — Long-context memory
• Part 4 — Prefill & TTFT — First-token latency
• Part 5 — Model-size ladder — 0.5B → 70B
• Part 6 — Full stack — Combine everything
• Part 7 — Speculative decoding — Draft models

The key takeaway from this first experiment is simple:

The FP16 demo is the floor, not the product.

An 8B model running at 5 tok/s on a 24 GB Mac isn't the end of the story.

It's the baseline.

Once you understand memory capacity, memory bandwidth, prefill, decode, and quantization, the performance numbers start to make a lot more sense.

And that's where this series goes next.

---

References

1. Vaswani et al., Attention Is All You Need (2017)
2. Dubey et al., The Llama 3 Herd of Models (2024)
3. Williams et al., Roofline: An Insightful Performance Model (2009)
4. Apple Machine Learning Research — MLX
5. Apple — mlx-lm
6. Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (2018)
7. Frantar et al., GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (2022)
8. Dao et al., FlashAttention: Fast and Memory-Efficient Exact Attention (2022)
9. MLX Community model hub
10. LLM-Inference benchmark repository
11. Apple — Apple Silicon unified memory architecture
12. Kaplan et al., Scaling Laws for Neural Language Models (2020)

---

Series Navigation

Next → Part 2: 4-Bit Weights Changed Everything

Local LLMs on Apple Silicon — Part 1 of 7

Tags: Machine Learning · Apple · LLM · MLX · Local AI · Apple Silicon
"""
    write(
        "00-introduction",
        {
            "title": "Running 8B LLMs on a MacBook: What Actually Matters",
            "subtitle": "Unified memory, the inference pipeline, and reproducible benchmarks on Apple Silicon — with M3 vs. M5 Max numbers",
            "featured": I("00-introduction", "thumb.png"),
            "featured_caption": "Local LLMs on Apple Silicon — Part 1",
            "series": "Local LLMs on Apple Silicon — Part 1 of 7",
            "tags": "Machine Learning, Apple, LLM, MLX, Local AI, Apple Silicon",
        },
        body,
    )


# Remaining articles: same clean Medium prose style
# Kept substantial but scannable like the user's Part 1 sample.

def art01() -> None:
    body = f"""
---

4-Bit Weights Changed Everything on My M3 Mac

Affine quantization from the papers — then Pareto charts, heatmaps, and M3 vs. M5 Max numbers across 14 models

Part 2 of 7 — Local LLMs on Apple Silicon

FEATURED IMAGE: {I("01-weight-quantization", "thumb.png")}
CAPTION: Weight quantization — Part 2

---

In Part 1, I loaded Llama 3.1 8B in FP16 on a 24 GB Mac M3 and watched the machine flinch.

16.33 GB peak.

2.6+ seconds to first token.

Roughly 5–6 tokens per second.

The model worked.

The laptop did not feel like a product anymore.

Weight quantization is the highest-leverage fix.

---

The Problem in One Sentence

An 8B model in FP16 needs about 16 GB just for weights.

On a 24 GB Mac, that leaves almost nothing for macOS, your editor, and the KV cache.

---

How Quantization Works

We map each high-precision weight to a smaller integer code, plus a scale.

IMAGE: {I("01-weight-quantization", "fig1.png")}
Figure 1 — Original redraw of affine quantization (Jacob et al., 2018).

In practice, LLM checkpoints use recipes like GPTQ and AWQ.

IMAGE: {I("01-weight-quantization", "fig2.png")}
Figure 2 — Original redraw of the GPTQ idea (Frantar et al., 2022).

IMAGE: {I("01-weight-quantization", "fig3.png")}
Figure 3 — Original redraw of the AWQ idea (Lin et al., 2023).

Fun fact: GPTQ was built for 175B-class models that couldn't fit on one GPU at FP16. The same math now makes 8B models comfortable on a laptop.

---

Why Fewer Bits Also Make Decode Faster

Each decode step often reads nearly all weights from memory.

Fewer bytes per weight means less memory traffic.

IMAGE: {I("01-weight-quantization", "fig4.png")}
Figure 4 — Original redraw of the Roofline idea. LLM decode often sits on the bandwidth slope.

---

Llama 3.1 8B on Mac M3

• FP16 — 16.3 GB · 5.8 tok/s
• W8 — 9.0 GB · 11.3 tok/s (~1.9×)
• W4 — 5.1 GB · 20.5 tok/s (~3.5×)
• W2 — 3.1 GB · 35.8 tok/s (~6×)

IMAGE: {I("01-weight-quantization", "fig5.png")}
Figure 5 — Memory and throughput as bit-width drops.

IMAGE: {I("01-weight-quantization", "fig6.png")}
Figure 6 — Pareto frontier. W4 is the practical sweet spot on 24 GB.

IMAGE: {I("01-weight-quantization", "fig7.png")}
Figure 7 — Explicit speedup versus FP16.

---

All 14 Models

The surprising part isn't one lucky Llama run.

It's that the pattern holds across the board.

IMAGE: {I("01-weight-quantization", "fig8.png")}
Figure 8 — Decode tok/s across models and bit-widths on Mac M3.

IMAGE: {I("01-weight-quantization", "fig9.png")}
Figure 9 — Peak memory for the same matrix. FP16 is the danger zone on 24 GB.

IMAGE: {I("01-weight-quantization", "fig10.png")}
Figure 10 — FP16 → W4 speedup and memory reduction across models.

IMAGE: {I("01-weight-quantization", "fig11.png")}
Figure 11 — Family zoom-ins: Qwen, Llama, Phi, Gemma, Mistral/DeepSeek.

---

M3 vs. M5 Max

Same W4 checkpoints. Different silicon.

• Llama 8B W4 — 20.5 → 112 tok/s
• Qwen 0.5B W4 — 215 → 581 tok/s

IMAGE: {I("01-weight-quantization", "fig12.png")}
Figure 12 — M3 vs. M5 Max at W4.

IMAGE: {I("01-weight-quantization", "fig13.png")}
Figure 13 — Llama 8B across every bit-width on both chips.

Hardware generation and quantization stack. They are not competing explanations.

---

What You Should Actually Run

• 16 GB Mac — 3B–7B at W4
• 24 GB Mac — 8B at W4 as the daily driver
• Skip FP16 8B as your everyday chat config

Reproduce:

./scripts/run_article.sh 1 "Mac M3"

Repo: https://github.com/Chirumamilla1522/LLM-Inference

---

Series Navigation

← Part 1: Introduction
Next → Part 3: The Hidden Memory Hog — KV Cache Quantization

Local LLMs on Apple Silicon — Part 2 of 7

Tags: Machine Learning · Quantization · LLM · Apple · Artificial Intelligence
"""
    write(
        "01-weight-quantization",
        {
            "title": "4-Bit Weights Changed Everything on My M3 Mac",
            "subtitle": "Affine quantization from the papers — then Pareto charts, heatmaps, and M3 vs. M5 Max numbers across 14 models",
            "featured": I("01-weight-quantization", "thumb.png"),
            "featured_caption": "Weight quantization — Part 2",
            "series": "Local LLMs on Apple Silicon — Part 2 of 7",
            "tags": "Machine Learning, Quantization, LLM, Apple, Artificial Intelligence",
        },
        body,
    )


def art02() -> None:
    body = f"""
---

The Hidden Memory Hog: KV Cache Quantization

How attention caching works, why GQA helps, and when 4-bit KV pays off on Apple Silicon

Part 3 of 7 — Local LLMs on Apple Silicon

FEATURED IMAGE: {I("02-kv-cache-quantization", "thumb.png")}
CAPTION: KV cache quantization — Part 3

---

Weight quantization gets the spotlight.

Once generation starts, something else grows: the KV cache.

For short chats, it barely shows up in tokens/sec.

For RAG, it's the second memory bill.

---

How the Cache Works

During decode, each new token attends to all previous tokens.

Recomputing keys and values every step would be wasteful, so transformers cache them.

IMAGE: {I("02-kv-cache-quantization", "fig1.png")}
Figure 1 — KV grows linearly with sequence length. 4-bit KV is roughly one-fourth the footprint.

IMAGE: {I("02-kv-cache-quantization", "fig2.png")}
Figure 2 — Original redraw of attention (Vaswani et al., 2017).

IMAGE: {I("02-kv-cache-quantization", "fig3.png")}
Figure 3 — Original redraw inspired by Pope et al. (2022): weights stay flat while KV grows with T.

---

GQA: Shrink Heads Before You Quantize

Llama 3, Mistral, and Qwen use Grouped-Query Attention.

Many query heads share fewer KV heads.

IMAGE: {I("02-kv-cache-quantization", "fig4.png")}
Figure 4 — Original redraw of GQA vs. multi-head attention (Ainslie et al., 2023).

That shrinks the cache before you even touch bit width.

---

Why Short-Context Benches Look Boring

At 512 prompt + 128 generation tokens on Mac M3:

• Llama 8B — 20.7 → 20.4 tok/s
• Mistral 7B — 21.6 → 21.2
• Qwen 7B — 21.8 → 21.4

IMAGE: {I("02-kv-cache-quantization", "fig5.png")}
Figure 5 — Short context: throughput almost unchanged.

IMAGE: {I("02-kv-cache-quantization", "fig6.png")}
Figure 6 — Longer generation: still mostly weight-bound at laptop batch size 1.

The win appears at long context, multi-session serving, or tight RAM — not in a 640-token microbench.

IMAGE: {I("02-kv-cache-quantization", "fig7.png")}
Figure 7 — Where pressure shows up: TTFT explodes as prompts grow.

IMAGE: {I("02-kv-cache-quantization", "fig8.png")}
Figure 8 — Original redraw of paged KV (Kwon et al., 2023). Serving systems page the cache; local MLX is the single-user cousin of the same memory problem.

---

When to Enable It

1. Always quantize weights first (W4).
2. Enable KV quant for >2K context or RAG.
3. Prefer GQA models.

Reproduce:

./scripts/run_article.sh 2 "Mac M3"

---

Series Navigation

← Part 2: Weight Quantization
Next → Part 4: Why Your Chatbot Feels Slow Before the First Word

Local LLMs on Apple Silicon — Part 3 of 7

Tags: Machine Learning · LLM · Transformers · Apple · Artificial Intelligence
"""
    write(
        "02-kv-cache-quantization",
        {
            "title": "The Hidden Memory Hog: KV Cache Quantization",
            "subtitle": "How attention caching works, why GQA helps, and when 4-bit KV pays off on Apple Silicon",
            "featured": I("02-kv-cache-quantization", "thumb.png"),
            "featured_caption": "KV cache quantization — Part 3",
            "series": "Local LLMs on Apple Silicon — Part 3 of 7",
            "tags": "Machine Learning, LLM, Transformers, Apple, Artificial Intelligence",
        },
        body,
    )


def art03() -> None:
    body = f"""
---

Why Your Chatbot Feels Slow Before the First Word

Prefill vs. decode, FlashAttention intuition, and TTFT curves that go quadratic on Apple Silicon

Part 4 of 7 — Local LLMs on Apple Silicon

FEATURED IMAGE: {I("03-prefill-and-ttft", "thumb.png")}
CAPTION: Prefill & TTFT — Part 4

---

Users blame "slow AI" on streaming speed.

Often the real pain is earlier: time-to-first-token — the pause before the first character.

---

Prefill vs. Decode

IMAGE: {I("03-prefill-and-ttft", "fig1.png")}
Figure 1 — Two phases, two bottlenecks.

• Prefill → TTFT
• Decode → tok/s

Optimize the wrong one and your "faster model" still feels broken.

---

FlashAttention — Exact, Not Approximate

IMAGE: {I("03-prefill-and-ttft", "fig2.png")}
Figure 2 — Original redraw of the FlashAttention IO pattern (Dao et al., 2022/23).

IMAGE: {I("03-prefill-and-ttft", "fig3.png")}
Figure 3 — Original redraw of online softmax (Milakov & Gimelshein, 2018).

Fun fact: FlashAttention computes the same math as naive attention. It just refuses to materialize the giant score matrix in slow memory.

---

The Quadratic Wall

Llama 3.1 8B, W4, Mac M3:

• p=256 → ~2.4 s TTFT
• p=512 → ~3.1 s
• p=1024 → ~5.8 s
• p=2048 → ~15.4 s

IMAGE: {I("03-prefill-and-ttft", "fig4.png")}
Figure 4 — TTFT versus prompt shape.

IMAGE: {I("03-prefill-and-ttft", "fig5.png")}
Figure 5 — Measured TTFT versus a roughly quadratic reference.

IMAGE: {I("03-prefill-and-ttft", "fig6.png")}
Figure 6 — The rag_agent workload hits about 31 seconds TTFT on M3.

That is why pasting a PDF into a local RAG demo often feels broken.

---

What to Do in Product

• Chat — shorten system prompts; enable prefill chunking
• RAG — fewer chunks; prefix cache; don't paste the whole document
• Long writing — optimize tok/s after TTFT is acceptable

Reproduce:

./scripts/run_article.sh 3 "Mac M3"

---

Series Navigation

← Part 3: KV Cache
Next → Part 5: From 0.5B to 70B

Local LLMs on Apple Silicon — Part 4 of 7

Tags: Machine Learning · LLM · UX · Apple · Artificial Intelligence
"""
    write(
        "03-prefill-and-ttft",
        {
            "title": "Why Your Chatbot Feels Slow Before the First Word",
            "subtitle": "Prefill vs. decode, FlashAttention intuition, and TTFT curves that go quadratic on Apple Silicon",
            "featured": I("03-prefill-and-ttft", "thumb.png"),
            "featured_caption": "Prefill & TTFT — Part 4",
            "series": "Local LLMs on Apple Silicon — Part 4 of 7",
            "tags": "Machine Learning, LLM, UX, Apple, Artificial Intelligence",
        },
        body,
    )


def art04() -> None:
    body = f"""
---

From 0.5B to 70B: What Fits on Apple Silicon

A practical size ladder with M3 and M5 Max numbers

Part 5 of 7 — Local LLMs on Apple Silicon

FEATURED IMAGE: {I("04-model-size-ladder", "thumb.png")}
CAPTION: Model size ladder — Part 5

---

"Which model should I run locally?" is really two questions:

1. Will it fit?
2. Will it be fast enough?

IMAGE: {I("04-model-size-ladder", "fig1.png")}
Figure 1 — Decision ladder for 24 GB unified memory.

---

The W4 Ladder on Mac M3

• Qwen 0.5B — 238 tok/s · 0.64 GB
• Llama 3.2 1B — 112 tok/s · 1.2 GB
• Qwen 3B — 48 tok/s · 2.2 GB
• Llama 8B — 21 tok/s · 5.1 GB
• Gemma 9B — 15 tok/s · 5.9 GB

IMAGE: {I("04-model-size-ladder", "fig2.png")}
Figure 2 — Tokens/sec and memory across sizes at W4.

IMAGE: {I("04-model-size-ladder", "fig3.png")}
Figure 3 — Memory versus speed scatter.

IMAGE: {I("04-model-size-ladder", "fig4.png")}
Figure 4 — Efficiency = tok/s per GB at W4.

Fun fact: Qwen 0.5B at W4 exceeds 238 tok/s on M3 — faster than most people type.

---

M5 Max Extends the Ladder

IMAGE: {I("04-model-size-ladder", "fig5.png")}
Figure 5 — M5 Max W4 ladder through larger models.

IMAGE: {I("04-model-size-ladder", "fig6.png")}
Figure 6 — Same checkpoints, different silicon.

---

Cheat Sheet for 24 GB

• IDE copilot — 7B W4
• Offline chat — 8B W4
• Router / draft model — 0.5B–1.5B W4
• Max quality that still fits — 9B W4 or 8B W8

Reproduce:

./scripts/run_article.sh 4 "Mac M3"

---

Series Navigation

← Part 4: Prefill & TTFT
Next → Part 6: Stacking Optimizations

Local LLMs on Apple Silicon — Part 5 of 7

Tags: Machine Learning · LLM · Apple · Artificial Intelligence · Benchmark
"""
    write(
        "04-model-size-ladder",
        {
            "title": "From 0.5B to 70B: What Fits on Apple Silicon",
            "subtitle": "A practical size ladder with M3 and M5 Max numbers",
            "featured": I("04-model-size-ladder", "thumb.png"),
            "featured_caption": "Model size ladder — Part 5",
            "series": "Local LLMs on Apple Silicon — Part 5 of 7",
            "tags": "Machine Learning, LLM, Apple, Artificial Intelligence, Benchmark",
        },
        body,
    )


def art05() -> None:
    body = f"""
---

Stacking Optimizations: 3.5× Faster Than FP16

The daily-driver recipe on a 24 GB Mac — and the full M5 Max matrix

Part 6 of 7 — Local LLMs on Apple Silicon

FEATURED IMAGE: {I("05-full-optimization-stack", "thumb.png")}
CAPTION: Full optimization stack — Part 6

---

Blog posts love clean A/B tests.

Real local inference turns several knobs at once.

IMAGE: {I("05-full-optimization-stack", "fig1.png")}
Figure 1 — Stacking funnel: FP16 → W4 → +KV → +prefill.

IMAGE: {I("05-full-optimization-stack", "fig2.png")}
Figure 2 — Pick the lever that matches your pain.

---

Headline Result — Mac M3, Llama 8B

• FP16 — 16.3 GB · 5.6 tok/s
• W4+KV+prefill — 5.1 GB · 19.9 tok/s (~3.5×)

IMAGE: {I("05-full-optimization-stack", "fig3.png")}
Figure 3 — FP16 versus optimized.

IMAGE: {I("05-full-optimization-stack", "fig4.png")}
Figure 4 — Llama and Mistral both jump when stacked.

IMAGE: {I("05-full-optimization-stack", "fig5.png")}
Figure 5 — Both models drop to about 5 GB peak.

---

M5 Max: The 16-Config Matrix

IMAGE: {I("05-full-optimization-stack", "fig6.png")}
Figure 6 — Llama 8B full config matrix on M5 Max.

IMAGE: {I("05-full-optimization-stack", "fig7.png")}
Figure 7 — Same stack on M3 versus M5 Max.

IMAGE: {I("05-full-optimization-stack", "fig8.png")}
Figure 8 — Original Roofline redraw: stacking works because decode is bandwidth-bound.

---

Daily Driver Recipe

On a 24 GB Mac, start here:

w4+kv_cache+prefill

on llama3-8b, mistral-7b, or qwen-7b.

Expect roughly 5 GB peak and 18–21 tok/s on M3.

python scripts/run_benchmark.py --preset llama3-8b --config w4+kv_cache+prefill --hardware "Mac M3"

---

Series Navigation

← Part 5: Model Size Ladder
Next → Part 7: Draft Models / Speculative Decoding

Local LLMs on Apple Silicon — Part 6 of 7

Tags: Machine Learning · Optimization · LLM · Apple · Artificial Intelligence
"""
    write(
        "05-full-optimization-stack",
        {
            "title": "Stacking Optimizations: 3.5× Faster Than FP16",
            "subtitle": "The daily-driver recipe on a 24 GB Mac — and the full M5 Max matrix",
            "featured": I("05-full-optimization-stack", "thumb.png"),
            "featured_caption": "Full optimization stack — Part 6",
            "series": "Local LLMs on Apple Silicon — Part 6 of 7",
            "tags": "Machine Learning, Optimization, LLM, Apple, Artificial Intelligence",
        },
        body,
    )


def art06() -> None:
    body = f"""
---

Draft Models: Free Speed Without Retraining

74% acceptance and 1.8× on Qwen — plus the case where speculation got slower

Part 7 of 7 — Local LLMs on Apple Silicon

FEATURED IMAGE: {I("06-speculative-decoding", "thumb.png")}
CAPTION: Speculative decoding — Part 7

---

A small draft model proposes tokens.

The large target verifies them in one parallel pass.

When the draft is right, you emit multiple tokens per expensive step — without retraining.

IMAGE: {I("06-speculative-decoding", "fig1.png")}
Figure 1 — Original redraw of draft/verify speculative decoding (Leviathan / Chen, 2023).

IMAGE: {I("06-speculative-decoding", "fig2.png")}
Figure 2 — Accept the matching prefix; reject and resample at the first mismatch.

IMAGE: {I("06-speculative-decoding", "fig3.png")}
Figure 3 — Original redraw of a Medusa-style variant (Cai et al., 2024).

---

The Clean Win: Qwen-7B on Mac M3

• Baseline W4 — 15.9 tok/s
• Speculative (Qwen 0.5B draft) — 28.3 tok/s
• Acceptance rate — 74.2%

IMAGE: {I("06-speculative-decoding", "fig4.png")}
Figure 4 — About 1.78× throughput at 74% acceptance.

IMAGE: {I("06-speculative-decoding", "fig5.png")}
Figure 5 — Big speed gain for roughly 0.3 GB extra RAM.

---

Honest Failures

On M3, Llama and Mistral speculative runs errored — draft pairing / tokenizer / memory.

On M5 Max, Qwen still wins: 122 → 170 tok/s.

Llama speculative was slightly slower: 113 → 110 tok/s at 59% acceptance.

IMAGE: {I("06-speculative-decoding", "fig6.png")}
Figure 6 — Qwen speculative on M3 versus M5 Max.

IMAGE: {I("06-speculative-decoding", "fig7.png")}
Figure 7 — Speedup versus acceptance. Low acceptance can erase the win.

Fun fact: Speculative decoding can make you slower if the draft is wrong too often. Measure acceptance rate. Don't assume.

---

Do This

• Same family and same tokenizer
• Tiny draft (0.5B–1B)
• Long generations
• Budget RAM for two models

Reproduce:

./scripts/run_article.sh 6 "Mac M3"

---

Series Navigation

← Part 6: Full Stack
Bonus → The RAG Wall: Context, Cache, and Latency

Local LLMs on Apple Silicon — Part 7 of 7

Tags: Machine Learning · LLM · Optimization · Apple · Artificial Intelligence
"""
    write(
        "06-speculative-decoding",
        {
            "title": "Draft Models: Free Speed Without Retraining",
            "subtitle": "74% acceptance and 1.8× on Qwen — plus the case where speculation got slower",
            "featured": I("06-speculative-decoding", "thumb.png"),
            "featured_caption": "Speculative decoding — Part 7",
            "series": "Local LLMs on Apple Silicon — Part 7 of 7",
            "tags": "Machine Learning, LLM, Optimization, Apple, Artificial Intelligence",
        },
        body,
    )


def art07() -> None:
    body = f"""
---

The RAG Wall: Context, Cache, and Why Your Demo Freezes

Quadratic TTFT, prefix caching, and workload stress on Apple Silicon

Bonus — Local LLMs on Apple Silicon

FEATURED IMAGE: {I("07-context-and-cache", "thumb.png")}
CAPTION: Context & prefix cache — Bonus

---

Short prompts hide sins.

Paste a PDF into a local RAG app and three forces collide:

1. Prefill cost grows roughly with the square of context length
2. KV memory grows linearly
3. Decode tokens/sec falls as attention spans more tokens

IMAGE: {I("07-context-and-cache", "fig1.png")}
Figure 1 — Retrieve → stuff context → expensive prefill → multi-second TTFT.

IMAGE: {I("07-context-and-cache", "fig2.png")}
Figure 2 — Original redraw: KV grows until it rivals weights.

---

Context Length vs. TTFT

Llama 3.1 8B on Mac M3:

• 256 tokens — 1.4 s
• 512 — 2.8 s
• 1024 — 6.5 s
• 2048 — 15.4 s

IMAGE: {I("07-context-and-cache", "fig3.png")}
Figure 3 — TTFT crosses 15 seconds at 2048 tokens on M3.

IMAGE: {I("07-context-and-cache", "fig4.png")}
Figure 4 — TTFT explodes while decode tok/s decays.

IMAGE: {I("07-context-and-cache", "fig5.png")}
Figure 5 — M5 Max lowers the wall. It does not remove the shape of the curve.

---

Prefix Cache: Cold vs. Warm

IMAGE: {I("07-context-and-cache", "fig6.png")}
Figure 6 — Skip re-prefilling a stable system prompt.

IMAGE: {I("07-context-and-cache", "fig7.png")}
Figure 7 — Cold 3,180 ms → warm 1,547 ms — about 51% faster.

---

Workload Stress

IMAGE: {I("07-context-and-cache", "fig8.png")}
Figure 8 — Latency, throughput, and memory across workloads.

IMAGE: {I("07-context-and-cache", "fig9.png")}
Figure 9 — rag_agent is about 31 seconds TTFT on M3.

If your local RAG demo feels broken, it's probably prefill — not "tok/s."

---

Mitigations That Actually Work

• Retrieve less — top-3, not top-20
• Prefix-cache system prompts and tools
• Use W4 + KV quant
• Route easy queries to a smaller model

Reproduce:

./scripts/run_article.sh 7 "Mac M3"

Repo: https://github.com/Chirumamilla1522/LLM-Inference

---

Series Navigation

← Part 7: Speculative Decoding
← Back to Part 1: Introduction

Local LLMs on Apple Silicon — Bonus

Tags: Machine Learning · RAG · LLM · Apple · Artificial Intelligence
"""
    write(
        "07-context-and-cache",
        {
            "title": "The RAG Wall: Context, Cache, and Why Your Demo Freezes",
            "subtitle": "Quadratic TTFT, prefix caching, and workload stress on Apple Silicon",
            "featured": I("07-context-and-cache", "thumb.png"),
            "featured_caption": "Context & prefix cache — Bonus",
            "series": "Local LLMs on Apple Silicon — Bonus",
            "tags": "Machine Learning, RAG, LLM, Apple, Artificial Intelligence",
        },
        body,
    )


def howto() -> None:
    (OUT / "HOW_TO_PUBLISH.md").write_text(
        """# How to publish these on Medium

Each `*.medium.txt` is already written like a finished Medium story:

```
---
Title
Subtitle
Part N of 7
FEATURED IMAGE + CAPTION
---
Short paragraphs
Section headers
IMAGE: path
Figure 10 — caption
Bullets
---
```

## In the Medium editor

1. **Big T** → paste the title line
2. **Little T** → paste the subtitle line
3. Add the series line as normal italic text under the subtitle
4. `+` → Image → upload the **FEATURED IMAGE** (wide thumbnail)
5. Paste body top to bottom
6. For each `IMAGE:` line → upload that PNG and keep the `Figure — …` line as the caption
7. Turn short emphasized lines into **pull quotes** where it helps
8. Add tags from the bottom
9. Publish → share via `DISTRIBUTION.md`

Do **not** paste the `IMAGE:` path text into the story — replace it with the actual upload.
"""
    )


def main() -> None:
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from medium_image_layout import resolve_article

    parser = argparse.ArgumentParser()
    parser.add_argument("--article", "-a", default=None, help="Only rebuild one article publish file")
    args = parser.parse_args()
    article = resolve_article(args.article)

    OUT.mkdir(parents=True, exist_ok=True)
    # remove old paste-kit chrome if any leftover html
    for p in OUT.glob("*.html"):
        p.unlink()

    jobs = [
        ("00-introduction", art00),
        ("01-weight-quantization", art01),
        ("02-kv-cache-quantization", art02),
        ("03-prefill-and-ttft", art03),
        ("04-model-size-ladder", art04),
        ("05-full-optimization-stack", art05),
        ("06-speculative-decoding", art06),
        ("07-context-and-cache", art07),
    ]
    for slug, fn in jobs:
        if article and slug != article:
            continue
        fn()
    if not article:
        howto()
    print(f"Done → {OUT}")


if __name__ == "__main__":
    main()
