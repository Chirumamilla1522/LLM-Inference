---
title: "Why Your Chatbot Feels Slow Before the First Word"
subtitle: "Prefill, Flash Attention, and time-to-first-token on Apple Silicon"
tags: LLM, TTFT, Flash Attention, Latency, Apple Silicon, UX
series: 4 of 7
read_time: 8 min
---

# Why Your Chatbot Feels Slow Before the First Word

*Part 4 of 7 — Local LLMs on Apple Silicon*

Users blame “slow AI” on token streaming speed. Often the real pain is **time-to-first-token (TTFT)** — the pause between hitting Enter and seeing the first character.

That pause is **prefill**: processing your entire prompt through every transformer layer before decode begins. Prefill is attention-heavy and scales roughly with \(O(T^2)\) in sequence length \(T\).

---

## Prefill vs decode: two different games

| Phase | What happens | Dominant cost | User feels |
|-------|--------------|---------------|------------|
| **Prefill** | Process all prompt tokens at once | Attention FLOPs, memory bandwidth | “Why is it thinking?” |
| **Decode** | Generate one token at a time | Weight bandwidth | “Why is it typing slowly?” |

Optimizations target different phases:

- **Weight quant (w4)** → mostly helps **decode**  
- **Prefill chunking** → helps **TTFT** on long prompts  
- **Flash Attention** → reduces memory traffic during **prefill**  

> **Fun fact:** FlashAttention (Dao et al., 2022) does not approximate attention — it computes the **exact** same result as naive attention, but tiles computation to stay in fast SRAM and avoid materializing the full \(T \times T\) attention matrix in HBM. It is an IO algorithm, not a shortcut.

---

## Our benchmark: prefill tuning on Llama 3.1 8B (w4)

| Run | Prompt tokens | TTFT | Decode tok/s |
|-----|---------------|------|--------------|
| w4 baseline | 512 | 3,103 ms | 20.6 |
| w4 + prefill | 512 | 3,185 ms | 18.7 |
| w4 + prefill, p=256 | 256 | **2,357 ms** | 13.7 |
| w4 + prefill, p=1024 | 1024 | **5,782 ms** | 20.1 |

![Prefill tuning — TTFT vs prompt shape](images/03_prefill_ttft.png)

*Figure 1: TTFT scales with prompt length; prefill optimization changes the curve shape at large p.*

At the default 512-token prompt, prefill tuning barely moves TTFT — the prompt is not long enough to benefit. But at **p=1024**, baseline attention work quadruples (roughly \(T^2\)), and prefill chunking becomes meaningful.

At **p=256**, TTFT drops 24% — shorter prompts are simply cheaper.

---

## The quadratic trap (with numbers)

| Prompt length | Relative attention work (rough) |
|---------------|--------------------------------|
| 256 | 1× |
| 512 | 4× |
| 1024 | 16× |
| 2048 | 64× |

This is why RAG apps feel sluggish: you are not sending 512 tokens — you are sending **your entire document**.

Article 7 covers context-length sweeps in depth; the headline from our M3 data at p=2048: **TTFT hits 15.4 seconds** on Llama 8B w4+prefill.

---

## Flash Attention in MLX

MLX implements IO-aware attention kernels on Metal — you do not toggle “Flash Attention” as a flag. The `prefill` optimization in our harness sets **`prefill_step_size`** (default 512), chunking long prefills so peak memory stays bounded.

Think of it as: *process the prompt in bites instead of one giant swallow.*

References: FlashAttention [1], FlashAttention-2 [2], online softmax [3].

---

## UX implications

| Product goal | Optimize for | Config priority |
|--------------|--------------|-----------------|
| Chat assistant | TTFT | prefill + shorter system prompts |
| Long-form writer | tok/s | w4 weights |
| RAG / search | TTFT + memory | prefill + KV quant |
| Code completion | tok/s | w4 + speculative decode |

> **Fun fact:** ChatGPT’s “instant” feel on short messages is partly **speculative decoding** and partly **prefix caching** of system prompts — not raw FLOPS. Local setups can replicate both (Parts 6 and 8).

---

## Try it yourself

```bash
./scripts/run_article.sh 3 "Mac M3"

# Long prompt stress test
python scripts/run_benchmark.py --preset llama3-8b --config w4+prefill \
  --hardware "Mac M3" -p 2048 -g 64
```

---

## References

1. Dao et al., *FlashAttention* (2022) — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)  
2. Dao, *FlashAttention-2* (2023) — [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)  
3. Milakov & Gimelshein, *Online normalizer calculation for softmax* (2018) — [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)  
4. Pope et al., *Efficiently Scaling Transformer Inference* (2022) — [arXiv:2211.05102](https://arxiv.org/abs/2211.05102)  

---

**← Previous:** [Part 3: KV Cache](02-kv-cache-quantization.md)  
**Next →** [Part 5: Model Size Ladder](04-model-size-ladder.md)

**Tags:** `LLM` `TTFT` `Flash Attention` `Latency` `Apple Silicon` `UX`
