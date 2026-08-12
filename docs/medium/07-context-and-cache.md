---
title: "The RAG Wall: Context Length, KV Growth, and Prefix Caching"
subtitle: "Why long prompts hurt — and how to skip re-processing system instructions"
tags: LLM, RAG, Context Window, KV Cache, Caching, Apple Silicon
series: Bonus (Week 3)
read_time: 10 min
---

# The RAG Wall: Context Length, KV Growth, and Prefix Caching

*Bonus — Local LLMs on Apple Silicon*

Short prompts hide a lot of sins. The moment you paste a PDF into a local RAG app, three forces collide:

1. **Prefill cost** grows ~quadratically with context  
2. **KV cache memory** grows linearly — and adds up  
3. **Decode throughput** drops as attention spans more tokens  

This post covers Article 7 benchmarks: prompt-length sweeps, generation-length stress tests, **prefix KV caching**, and realistic workloads from light chat to heavy RAG.

---

## Prompt length vs TTFT (Llama 3.1 8B, w4+prefill)

| Prompt tokens | TTFT | Decode tok/s | Peak GB |
|---------------|------|--------------|---------|
| 256 | 1,406 ms | 20.3 | 4.92 |
| 512 | 2,839 ms | 20.5 | 5.06 |
| 1024 | 6,503 ms | 14.9 | 5.24 |
| 2048 | **15,355 ms** | 11.9 | 5.35 |

![Context length vs TTFT](images/07_context_ttft.png)

*Figure 1: TTFT crosses 15 seconds at 2048 tokens — the “RAG wall” on local 8B models.*

> **Fun fact:** GPT-4’s API charges more for long contexts partly because **prefill FLOPs scale super-linearly** — you are paying for attention over every token in your retrieved chunks, every request, unless you cache.

---

## Prefix KV cache: cold vs warm

Split your prompt into a **fixed system prefix** (instructions, tool schemas) and a **variable user suffix**.

| Mode | TTFT | What happened |
|------|------|---------------|
| Cold (full prefill) | **3,180 ms** | Process 256 system + user tokens |
| Warm (cached prefix) | **1,547 ms** | Load cached KV for 256 system tokens; prefill user only |

![Prefix cache cold vs warm](images/07_prefix_cache.png)

*Figure 2: ~51% TTFT reduction by reusing system-prompt KV across turns.*

MLX exposes this via `save_prompt_cache` / `load_prompt_cache` in mlx-lm — our harness runs it with `--prefix-cache`.

**When it helps:** Multi-turn chat with a fat system prompt, agent tool definitions, or repeated RAG templates where only the user query changes.

---

## Workload stress matrix

Beyond synthetic `-p` / `-g` flags, we run **realistic workload profiles**:

| Workload | Pressure | TTFT | tok/s | Notes |
|----------|----------|------|-------|-------|
| chat_light | decode | 1,518 ms | 13.5 | Short turns |
| chat_standard | balanced | 4,031 ms | 13.4 | Typical chat |
| complete_code | balanced | 2,071 ms | 17.3 | Code completion shape |
| summarize_long | prefill | 15,897 ms | 10.2 | Long input doc |
| **rag_agent** | memory | **31,118 ms** | 11.3 | Heavy retrieval context |

The **rag_agent** workload is the stress test: 31-second TTFT before the first token. That is unusable for interactive UX without caching, chunking, or a smaller model.

---

## Three levers (same math, different code)

| Lever | Flag / API | Fixes |
|-------|------------|-------|
| Shorter context | `-p`, better chunking | TTFT |
| KV quant | `w4+kv_cache` | Memory at long T |
| Prefix cache | `--prefix-cache` | Repeat system prompt |
| Smaller model | `--preset qwen-1.5b` | Everything (quality tradeoff) |

---

## Practical RAG on 24 GB Mac

1. **Retrieve less** — top-3 chunks, not top-20  
2. **Cache system + tool prefix** — prefix KV cache  
3. **Use w4 + kv_cache** — headroom for long KV  
4. **Consider 3B–7B router + 8B generator** — route easy queries to fast model  

---

## Reproduce

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

**← Series start:** [Part 1: Introduction](00-introduction.md)  
**← Previous:** [Part 7: Speculative Decoding](06-speculative-decoding.md)

**Tags:** `LLM` `RAG` `Context Window` `KV Cache` `Caching` `Apple Silicon`
