---
title: "Stacking Optimizations: 3.5× Faster Than FP16"
subtitle: "The funnel from fp16 → w4+kv+prefill, plus a decision tree for when to enable what"
tags: LLM, Optimization, Performance, MLX, Apple Silicon, Engineering
series: 6 of 7
read_time: 11 min
figures: 6
---

# Stacking Optimizations: 3.5× Faster Than FP16

*Part 6 of 7 — Local LLMs on Apple Silicon*

Individual optimizations are easy in isolation. Production local inference uses **all of them together**.

---

## The funnel (how stacking works)

![Optimization funnel](images/workflows/05_optimization_funnel.png)

*Figure 1 — Workflow: each layer adds a lever — weights → KV → prefill → daily recipe.*

![Decision tree](images/workflows/05_decision_tree.png)

*Figure 2 — Workflow: pick the lever that matches your pain (RAM / TTFT / long generation).*

---

## Results: Llama 3.1 8B — fp16 vs fully optimized

| Config | Peak GB | TTFT | tok/s |
|--------|---------|------|-------|
| fp16 | **16.33** | 2,689 ms | **5.6** |
| w4 + kv_cache + prefill | **5.06** | 2,746 ms | **19.9** |

![Full stack llama](images/05_full_stack.png)

*Figure 3 — Results: ~31% of fp16 memory, **3.55×** decode throughput.*

![Two models speed](images/05_full_stack_two_models.png)

*Figure 4 — Results: Llama 8B and Mistral 7B both jump hard when optimized.*

![Two models memory](images/05_full_stack_memory.png)

*Figure 5 — Results: both models drop from ~15–16 GB → ~5 GB peak.*

**Memory saved:** ~11 GB — enough for a second small model or a fat RAG KV cache.  
**Speed gained:** 19.9 / 5.6 ≈ **3.55×**.

---

## The 16-config matrix

We sweep:

| Axis | Options |
|------|---------|
| Weight bits | fp16, w8, w4, w2 |
| KV quant | on / off |
| Prefill | on / off |

→ \(4 × 2 × 2 = 16\) configs per model across 14+ presets on M3 and M5 Max.

> **Fun fact:** A full M3 article sweep can take **2–8 hours**. Each config runs in an isolated subprocess so one Metal OOM does not kill the batch.

---

## Recommended daily recipe (24 GB Mac)

```text
Config: w4+kv_cache+prefill
Model:  llama3-8b / mistral-7b / qwen-7b
Expect: ~5 GB peak, ~18–21 tok/s
```

```bash
python scripts/run_benchmark.py \
  --preset llama3-8b \
  --config w4+kv_cache+prefill \
  --hardware "Mac M3"
```

| Situation | Skip | Why |
|-----------|------|-----|
| Short prompts only | KV quant | Tiny cache |
| Max quality eval | fp16 / w8 | Quant affects scores |
| Tiny models (<3B) | prefill | Already fast TTFT |
| 16 GB Mac | fp16 8B+ | Will swap |

---

## Next lever

Part 7 adds **speculative decoding** — a draft model that boosts tok/s **without** changing weight precision.

---

## References

1. Frantar et al., *GPTQ* (2022) — [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)  
2. Kwon et al., *PagedAttention* (2023) — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)  
3. Dao et al., *FlashAttention* (2022) — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)  
4. LLM-Inference — [github.com/Chirumamilla1522/LLM-Inference](https://github.com/Chirumamilla1522/LLM-Inference)  

---

**← Previous:** [Part 5](04-model-size-ladder.md) · **Next →** [Part 7: Speculative Decoding](06-speculative-decoding.md)

**Tags:** `LLM` `Optimization` `Performance` `MLX` `Apple Silicon`
