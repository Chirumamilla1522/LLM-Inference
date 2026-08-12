---
title: "Why Your Chatbot Feels Slow Before the First Word"
subtitle: "Prefill vs decode, FlashAttention intuition, and TTFT curves that go quadratic"
tags: LLM, TTFT, Flash Attention, Latency, Apple Silicon, UX, Prefill
series: 4 of 7
read_time: 12 min
figures: 5
---

# Why Your Chatbot Feels Slow Before the First Word

*Part 4 of 7 — Local LLMs on Apple Silicon*

Users blame “slow AI” on streaming speed. Often the real pain is **time-to-first-token (TTFT)** — the pause between Enter and the first character.

That pause is **prefill**: processing your entire prompt through every layer before decode begins. Prefill is attention-heavy and scales roughly with \(O(T^2)\) in sequence length \(T\).

---

## Two phases, two bottlenecks

![Prefill vs decode](images/workflows/03_prefill_vs_decode.png)

*Figure 1 — Workflow: prefill owns TTFT; decode owns tok/s. Different optimizations apply.*

| Phase | Dominant cost | Metric | Optimize with |
|-------|---------------|--------|---------------|
| **Prefill** | Attention FLOPs / IO | TTFT | Flash-style kernels, chunking, shorter prompts |
| **Decode** | Weight bandwidth | tok/s | w4, speculative decode |

> **Fun fact:** ChatGPT’s “instant” feel on short messages is partly **prefix caching** of system prompts and partly **speculative decoding** — not raw FLOPS. Local setups can copy both ideas.

---

## FlashAttention (exact, not approximate)

Naive attention materializes an \(N \times N\) matrix in slow memory. **FlashAttention** (Dao et al., 2022/23) tiles the work so tiles fit in fast SRAM, streams softmax with online normalizers, and returns the **same mathematical result**.

![FlashAttention tiling](images/workflows/03_flash_attention.png)

*Figure 2 — Workflow: naive O(N²) materialization vs tiled FlashAttention (exact output).*

In MLX, IO-aware Metal kernels handle this — you do not flip a “Flash” flag. Our `prefill` config sets **`prefill_step_size`** so long prompts are chunked and peak memory stays bounded.

---

## Results: TTFT vs prompt shape (Llama 8B, w4)

| Run | Prompt | TTFT | tok/s |
|-----|--------|------|-------|
| baseline | 512 | 3,103 ms | 20.6 |
| prefill ON | 512 | 3,185 ms | 18.7 |
| prefill, p=256 | 256 | **2,357 ms** | 13.7 |
| prefill, p=1024 | 1024 | **5,782 ms** | 20.1 |

![Prefill TTFT bars](images/03_prefill_ttft.png)

*Figure 3 — Results: shorter prompts win TTFT; at 512 tokens prefill tuning is subtle; at 1024 pressure shows.*

![TTFT vs prompt curve](images/03_ttft_vs_prompt_curve.png)

*Figure 4 — Results: measured TTFT vs a ∝ T² reference curve — the RAG danger zone.*

At **p=2048** (Article 7 data): **~15.4 seconds** TTFT. That is why pasting a PDF into a local RAG app feels broken.

| Prompt length | Relative attention work (rough) |
|---------------|--------------------------------|
| 256 | 1× |
| 512 | 4× |
| 1024 | 16× |
| 2048 | 64× |

---

## UX mapping

| Product goal | Optimize for | Priority |
|--------------|--------------|----------|
| Chat assistant | TTFT | prefill + short system prompts |
| Long-form writer | tok/s | w4 weights |
| RAG / search | TTFT + memory | prefill + KV + fewer chunks |
| Code completion | tok/s | w4 + speculative |

```bash
./scripts/run_article.sh 3 "Mac M3"
python scripts/run_benchmark.py --preset llama3-8b --config w4+prefill \
  --hardware "Mac M3" -p 2048 -g 64
```

---

## References

1. Dao et al., *FlashAttention* (2022) — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)  
2. Dao, *FlashAttention-2* (2023) — [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)  
3. Milakov & Gimelshein, *Online softmax* (2018) — [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)  
4. Pope et al., *Efficiently Scaling Transformer Inference* (2022) — [arXiv:2211.05102](https://arxiv.org/abs/2211.05102)  

---

**← Previous:** [Part 3](02-kv-cache-quantization.md) · **Next →** [Part 5: Model Ladder](04-model-size-ladder.md)

**Tags:** `LLM` `TTFT` `Flash Attention` `Latency` `Apple Silicon` `UX`
