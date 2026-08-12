---
title: "From 0.5B to 70B: What Fits on Apple Silicon"
subtitle: "A model-size ladder measured in gigabytes and tokens per second"
tags: LLM, Model Size, Apple Silicon, Benchmark, Qwen, Llama
series: 5 of 7
read_time: 9 min
---

# From 0.5B to 70B: What Fits on Apple Silicon

*Part 5 of 7 — Local LLMs on Apple Silicon*

“Which model should I run locally?” is really two questions:

1. **Will it fit in RAM?**  
2. **Will it be fast enough to use?**

We benchmarked **14 presets from 0.5B to 9B** at w4 on a **24 GB Mac M3**, plus larger models on **Mac M5 Max (64–128 GB)**. Here is the ladder.

---

## The w4 ladder on Mac M3 (24 GB)

| Model | Params | Peak GB | TTFT | tok/s |
|-------|--------|---------|------|-------|
| Qwen 0.5B | 0.5B | 0.64 | 145 ms | **238.3** |
| Llama 3.2 1B | 1B | 1.24 | 347 ms | 112.0 |
| Qwen 1.5B | 1.5B | 1.43 | 488 ms | 90.8 |
| Qwen 3B | 3B | 2.22 | 1,017 ms | 48.3 |
| Phi-3 Mini | 3.8B | 2.93 | 1,505 ms | 35.6 |
| Mistral 7B | 7B | 4.62 | 3,456 ms | 17.4 |
| Llama 3.1 8B | 8B | 5.06 | 2,817 ms | 20.6 |
| Gemma 9B | 9B | 5.88 | 3,852 ms | 15.4 |

![Model size ladder — w4 on Mac M3](images/04_model_size_ladder.png)

*Figure 1: Smaller models dominate throughput; 7–9B models cluster around 15–21 tok/s at w4.*

> **Fun fact:** Qwen 0.5B at w4 on M3 exceeds **238 tok/s** — faster than most people type. At that speed, the bottleneck becomes your UI rendering tokens, not the model.

---

## Three tiers for 24 GB Macs

### Tier 1 — “Instant” (≤1.5B, w4)

- **Use case:** Classification, routing, simple Q&A, draft models for speculative decode  
- **Speed:** 90–240 tok/s  
- **RAM:** <2 GB model footprint  

### Tier 2 — “Daily driver” (3B–8B, w4)

- **Use case:** Coding assistants, chat, summarization  
- **Speed:** 17–48 tok/s  
- **RAM:** 3–6 GB model footprint  

### Tier 3 — “Pushing it” (9B+, w4 or w8)

- **Use case:** Best open-model quality that still fits  
- **Speed:** 15–20 tok/s  
- **RAM:** 6–10 GB — watch KV growth on long contexts  

**fp16 8B on 24 GB?** Possible, but you sacrifice headroom (16.3 GB peak) and decode drops to ~5 tok/s. Not recommended as a daily config.

---

## Family flavor (same size class)

At ~7–8B w4, families cluster:

| Family | tok/s | Notes |
|--------|-------|-------|
| Mistral 7B | 17.4 | Strong generalist; slightly slower TTFT in our run |
| Llama 3.1 8B | 20.6 | Meta ecosystem; good MLX support |
| Qwen 2.5 7B | 21.6 | Competitive speed; strong multilingual |
| DeepSeek R1 (Qwen 7B) | 18.6 | Reasoning-tuned variant |

Differences are **10–20%**, not 2×. Pick by license, language, and fine-tune — not micro-benchmarks.

---

## What about 32B, 70B, 72B? (M5 Max)

On **Mac M5 Max** with 64–128 GB unified memory, the ladder extends:

- **32B at w4** — fits comfortably, ~8–12 tok/s (varies by family)  
- **70B at w4** — fits at 64 GB+ with KV headroom  
- **fp16 70B** — needs ~140 GB; not on consumer Macs today  

The M5 Max sweep in our repo includes llama-70b, qwen-72b, gemma-27b, and mistral-small-22b — see `results/Mac_M5_Max/article_04_model-size-ladder/`.

> **Fun fact:** Apple’s M5 Max memory bandwidth (~500+ GB/s class) is the main reason larger models become usable at w4 — you still move fewer bytes per token, but the wider pipe helps more than on M1/M2.

---

## Choosing by task

| Task | Sweet spot on 24 GB |
|------|---------------------|
| Local copilot (IDE) | 7B w4 (Mistral/Qwen/Llama) |
| Offline chat | 8B w4 |
| Router / classifier | 0.5B–1.5B w4 |
| Reasoning-heavy | 7B reasoning-tuned w4 |
| Max quality local | 9B w4 or 8B w8 |

---

## Reproduce

```bash
./scripts/run_article.sh 4 "Mac M3"
python scripts/generate_article_tables.py --hardware "Mac M3" --article 4
```

---

## References

1. Dubey et al., *The Llama 3 Herd of Models* (2024) — [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)  
2. Touvron et al., *Llama 2* (2023) — [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)  
3. Qwen2.5 technical report — [Hugging Face](https://huggingface.co/Qwen)  
4. Williams et al., *Roofline model* (2009) — [CACM](https://people.csail.mit.edu/stajich/publications/cacm09.pdf)  

---

**← Previous:** [Part 4: Prefill & TTFT](03-prefill-and-ttft.md)  
**Next →** [Part 6: Full Optimization Stack](05-full-optimization-stack.md)

**Tags:** `LLM` `Model Size` `Apple Silicon` `Benchmark` `Qwen` `Llama`
