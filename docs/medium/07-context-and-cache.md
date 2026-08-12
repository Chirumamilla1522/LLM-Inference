---
title: "The RAG Wall: Context Length, KV Growth, and Prefix Caching"
subtitle: "Quadratic TTFT, workload stress tests, and cold vs warm system-prompt caching"
tags: LLM, RAG, Context Window, KV Cache, Caching, Apple Silicon, Latency
series: Bonus (Week 3)
read_time: 13 min
figures: 7
---

# The RAG Wall: Context Length, KV Growth, and Prefix Caching

*Bonus — Local LLMs on Apple Silicon*

Short prompts hide a lot of sins. Paste a PDF into a local RAG app and three forces collide:

1. **Prefill** grows ~quadratically with context  
2. **KV memory** grows linearly  
3. **Decode tok/s** falls as attention spans more tokens  

---

## The RAG wall (workflow)

![RAG wall](images/workflows/07_rag_wall.png)

*Figure 1 — Workflow: retrieve → stuff 2K+ tokens → O(T²) prefill → 15–30s TTFT.*

---

## Results: prompt length vs TTFT

| Prompt | TTFT | tok/s | Peak GB |
|--------|------|-------|---------|
| 256 | 1,406 ms | 20.3 | 4.92 |
| 512 | 2,839 ms | 20.5 | 5.06 |
| 1024 | 6,503 ms | 14.9 | 5.24 |
| 2048 | **15,355 ms** | 11.9 | 5.35 |

![Context TTFT bars](images/07_context_ttft.png)

*Figure 2 — Results: TTFT crosses 15 seconds at 2048 tokens.*

![Dual axis](images/07_context_dual_axis.png)

*Figure 3 — Results: TTFT explodes while decode tok/s slowly decays.*

> **Fun fact:** API pricing for long context partly reflects **super-linear prefill cost** — you pay for attention over every retrieved chunk unless something is cached.

---

## Prefix KV cache: cold vs warm

Split the prompt into a **fixed system prefix** and a **variable user suffix**.

![Prefix cache workflow](images/workflows/07_prefix_cache_workflow.png)

*Figure 4 — Workflow: cold = full prefill every turn; warm = load cached KV for system prefix.*

| Mode | TTFT |
|------|------|
| Cold | **3,180 ms** |
| Warm | **1,547 ms** (~51% faster) |

![Prefix cache bars](images/07_prefix_cache.png)

*Figure 5 — Results: ~2× TTFT win by reusing system-prompt KV across turns.*

MLX APIs: `save_prompt_cache` / `load_prompt_cache` (mlx-lm). Harness flag: `--prefix-cache`.

---

## Workload stress matrix

| Workload | Pressure | TTFT | tok/s |
|----------|----------|------|-------|
| chat_light | decode | 1.5 s | 13.5 |
| chat_standard | balanced | 4.0 s | 13.4 |
| complete_code | balanced | 2.1 s | 17.3 |
| summarize_long | prefill | 15.9 s | 10.2 |
| **rag_agent** | memory | **31.1 s** | 11.3 |

![Workload TTFT](images/07_workload_ttft.png)

*Figure 6 — Results: rag_agent is unusable for interactive UX without caching/chunking.*

![Generation length](images/07_generation_length.png)

*Figure 7 — Results: longer generations slightly slow decode as KV grows (w4+kv_cache).*

---

## Mitigation checklist

| Lever | How | Fixes |
|-------|-----|-------|
| Retrieve less | top-3 not top-20 | TTFT |
| Prefix cache | `--prefix-cache` | Repeat system prompt |
| KV quant | `w4+kv_cache` | Long-T memory |
| Smaller router | 0.5B–1.5B | Easy queries |
| Speculative decode | draft model | Long answers |

```bash
./scripts/run_article.sh 7 "Mac M3"
python scripts/run_benchmark.py --preset llama3-8b --config w4 \
  --prefix-cache --hardware "Mac M3"
```

---

## References

1. Dao et al., *FlashAttention* (2022) — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)  
2. Ainslie et al., *GQA* (2023) — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)  
3. Kwon et al., *PagedAttention* (2023) — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)  
4. Pope et al., *Efficiently Scaling Transformer Inference* (2022) — [arXiv:2211.05102](https://arxiv.org/abs/2211.05102)  
5. mlx-lm cache API — [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)  

---

**← Series start:** [Part 1](00-introduction.md) · **← Previous:** [Part 7](06-speculative-decoding.md)

**Tags:** `LLM` `RAG` `Context Window` `KV Cache` `Caching` `Apple Silicon`
