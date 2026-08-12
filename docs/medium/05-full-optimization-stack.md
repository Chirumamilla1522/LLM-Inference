---
title: "Stacking Optimizations: 3.5× Faster Than FP16"
subtitle: "Combining weight quant, KV cache, and prefill on Llama 3.1 8B"
tags: LLM, Optimization, Performance, MLX, Apple Silicon, Engineering
series: 6 of 7
read_time: 8 min
---

# Stacking Optimizations: 3.5× Faster Than FP16

*Part 6 of 7 — Local LLMs on Apple Silicon*

Individual optimizations are easy to understand in isolation. Production local inference uses **all of them together**:

```
fp16 baseline  →  w4 weights  →  + KV quant  →  + prefill tuning
     5 tok/s         20 tok/s        ~same           better TTFT at long ctx
```

This post is the **capstone** of the benchmark series: one table that shows the combined effect on Llama 3.1 8B and Mistral 7B.

---

## Llama 3.1 8B: fp16 vs fully optimized

| Config | Label | Peak GB | TTFT | tok/s |
|--------|-------|---------|------|-------|
| fp16 | baseline | **16.33** | 2,689 ms | **5.6** |
| w4 + kv_cache + prefill | optimized | **5.06** | 2,746 ms | **19.9** |

![Full stack comparison](images/05_full_stack.png)

*Figure 1: Optimized stack uses ~31% of fp16 memory and delivers 3.5× decode throughput.*

**Memory saved:** 11.3 GB — enough to run a second small model or a large KV cache for RAG.

**Speed gained:** 19.9 / 5.6 = **3.55×** decode throughput.

TTFT is similar at 512-token prompts because prefill and decode trade off differently at this length — the win is sustained generation speed and RAM headroom.

---

## Mistral 7B: same story

| Config | Peak GB | TTFT | tok/s |
|--------|---------|------|-------|
| fp16 | 14.77 | 4,350 ms | 3.6 |
| w4 + kv + prefill | 4.62 | 3,954 ms | **16.0** |

Mistral’s fp16 decode was slower in our run (3.6 tok/s) — possibly thermal or scheduling variance — but the **optimized stack still delivers 4.4×** improvement.

---

## The 16-config matrix

Our repo sweeps **all combinations** of:

| Axis | Options |
|------|---------|
| Weight bits | fp16, w8, w4, w2 |
| KV cache quant | on / off |
| Prefill tuning | on / off |

That is \(4 \times 2 \times 2 = 16\) configs per model. Article 5 runs this full matrix across 14+ presets on M3 and M5 Max — hundreds of JSON files you can diff in Git.

> **Fun fact:** The full M3 sweep (articles 0–7) can take **2–8 hours** depending on thermals and HF cache state. Each config runs in an isolated subprocess so one Metal OOM does not kill the batch — learned the hard way.

---

## Recommended “optimized” recipe for 24 GB Mac

```text
Config label: w4+kv_cache+prefill
Model: llama3-8b (or mistral-7b, qwen-7b)
Expected: ~5 GB peak, ~18–21 tok/s decode
```

```bash
python scripts/run_benchmark.py \
  --preset llama3-8b \
  --config w4+kv_cache+prefill \
  --hardware "Mac M3"
```

---

## When *not* to stack everything

| Situation | Skip | Why |
|-----------|------|-----|
| Short prompts only | KV quant | Negligible cache size |
| Max quality eval | fp16 or w8 | Quant affects benchmarks |
| Tiny models (<3B) | prefill | Already fast TTFT |
| 16 GB RAM Mac | fp16 anything 8B+ | Will swap |

---

## What comes next

Individual levers are covered. Part 7 explores **speculative decoding** — a different axis that can boost tok/s **without** changing weight precision, by drafting tokens with a small model and verifying with the large one.

---

## References

1. Frantar et al., *GPTQ* (2022) — [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)  
2. Kwon et al., *PagedAttention* (2023) — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)  
3. Dao et al., *FlashAttention* (2022) — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)  
4. LLM-Inference repo — [github.com/Chirumamilla1522/LLM-Inference](https://github.com/Chirumamilla1522/LLM-Inference)  

---

**← Previous:** [Part 5: Model Size Ladder](04-model-size-ladder.md)  
**Next →** [Part 7: Speculative Decoding](06-speculative-decoding.md)

**Tags:** `LLM` `Optimization` `Performance` `MLX` `Apple Silicon` `Engineering`
