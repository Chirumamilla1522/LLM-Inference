---
title: "The Hidden Memory Hog: KV Cache Quantization"
subtitle: "How attention caching works, why GQA helps, and when 4-bit KV pays off"
tags: LLM, KV Cache, Memory, Apple Silicon, GQA, Inference, Transformers
series: 3 of 7
read_time: 12 min
figures: 5
---

# The Hidden Memory Hog: KV Cache Quantization

*Part 3 of 7 — Local LLMs on Apple Silicon*

Weight quantization gets the spotlight. Once generation starts, something else grows: the **KV cache** — stored key and value tensors for every token in context.

For long chat or RAG, that cache can rival weight memory. **KV cache quantization** stores those tensors in fewer bits during decode.

---

## How the cache works

During autoregressive generation, each new token attends to all previous tokens. Recomputing keys/values for the full history every step would be wasteful — so transformers **cache** per-layer **K** and **V**.

![KV cache workflow](images/workflows/02_kv_cache_workflow.png)

*Figure 1 — Workflow: cache grows linearly with T; 4-bit KV ≈ ¼ the footprint of FP16 KV.*

Memory formula:

\[
M_{\text{KV}} = 2 \cdot L \cdot H_{\text{kv}} \cdot T \cdot D \cdot \frac{b_{\text{kv}}}{8}
\]

| Symbol | Typical Llama 8B |
|--------|------------------|
| \(L\) | 32 layers |
| \(H_{\text{kv}}\) | 8 (fewer with GQA) |
| \(T\) | prompt + generated tokens |
| \(D\) | 128 head dim |
| \(b_{\text{kv}}\) | 16 (off) or 4 (on) |

---

## Attention with a cache (one decode step)

![Attention + KV](images/workflows/02_attention_with_cache.png)

*Figure 2 — Workflow: new token → Q/K/V; append to cache; softmax attention over history.*

> **Fun fact:** Pope et al. (2022) noted that for long sequences, **KV memory can exceed weight memory** — often somewhere between 2K and 8K tokens for 7B-class models.

---

## GQA: shrink heads before you quantize

Llama 3 uses **Grouped Query Attention** — many query heads share fewer KV heads (Ainslie et al., 2023). That shrinks \(H_{\text{kv}}\) before you even touch bit width.

![GQA vs MHA](images/workflows/02_gqa_vs_mha.png)

*Figure 3 — Workflow: MHA (1:1 Q/KV) vs GQA (many Q → few KV) — smaller cache by design.*

---

## Results: w4 vs w4+KV on Mac M3

At our default **512 prompt + 128 gen**, KV is still small vs weights — so throughput barely moves:

| Model | w4 tok/s | w4 + KV tok/s |
|-------|----------|---------------|
| Llama 3.1 8B | 20.7 | 20.4 |
| Mistral 7B | 21.6 | 21.2 |
| Qwen 7B | 21.8 | 21.4 |

![KV compare bars](images/02_kv_cache_compare.png)

*Figure 4 — Results: short-context throughput nearly identical — the win shows up at long T / multi-session.*

| Scenario | Weight quant | KV quant |
|----------|--------------|----------|
| Short chat (≤512) | **High** | Low |
| RAG / 4K+ context | Medium | **High** |
| Multi-user local server | Medium | **High** |
| Long code files | Medium | **High** |

Production systems go further: **PagedAttention** (Kwon et al., 2023 / vLLM) manages KV fragmentation across requests. Local MLX uses a simpler in-process cache — same math, different scheduler.

---

## Practical advice

1. Always quantize **weights** first (w4)  
2. Enable **KV quant** when context > ~2K or you run RAG  
3. Prefer GQA models (Llama 3, Mistral, Qwen 2.5)  

```bash
./scripts/run_article.sh 2 "Mac M3"
```

---

## References

1. Vaswani et al., *Attention Is All You Need* (2017) — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)  
2. Ainslie et al., *GQA* (2023) — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)  
3. Kwon et al., *PagedAttention* (2023) — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)  
4. Pope et al., *Efficiently Scaling Transformer Inference* (2022) — [arXiv:2211.05102](https://arxiv.org/abs/2211.05102)  

---

**← Previous:** [Part 2](01-weight-quantization.md) · **Next →** [Part 4: Prefill & TTFT](03-prefill-and-ttft.md)

**Tags:** `LLM` `KV Cache` `Memory` `Apple Silicon` `GQA` `Inference`
