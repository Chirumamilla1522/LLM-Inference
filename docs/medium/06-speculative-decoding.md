---
title: "Draft Models: Free Speed Without Retraining"
subtitle: "Speculative decoding on Apple Silicon — 74% acceptance, 1.8× throughput"
tags: LLM, Speculative Decoding, Inference, Apple Silicon, Performance, MLX
series: 7 of 7
read_time: 9 min
---

# Draft Models: Free Speed Without Retraining

*Part 7 of 7 — Local LLMs on Apple Silicon*

Quantization shrinks weights. Prefill tuning helps TTFT. **Speculative decoding** is different: it runs **two models** — a small *draft* model proposes tokens, and the large *target* model verifies them in parallel.

When the draft guesses right, you generate multiple tokens per target forward pass. **No retraining required.**

---

## How it works (30 seconds)

1. Draft model generates \(k\) candidate tokens cheaply  
2. Target model evaluates all \(k\) in **one parallel forward pass**  
3. Accept tokens that match the target’s distribution; reject and resample from the first mismatch  

Speedup depends on:

- **Acceptance rate** \(\alpha\) — how often draft tokens survive verification  
- **Draft speed** — how fast the small model proposes  
- **\(k\)** — tokens proposed per round (we use \(k=3\))

Leviathan et al. (2023) and Chen et al. (2023) formalized this; Medusa (2024) extends the idea with multiple decoding heads.

> **Fun fact:** Google’s speculative decoding paper reported up to **2–3× speedup** on T5-XXL without quality loss — the draft model can be **15× smaller** and still achieve >60% acceptance on similar domains.

---

## Our results: Qwen 7B on Mac M3

| Mode | Draft model | Peak GB | TTFT | tok/s | Accept rate |
|------|-------------|---------|------|-------|-------------|
| Baseline w4 | — | 4.72 | 3,613 ms | 15.9 | — |
| Speculative w4 | Qwen 0.5B | 5.00 | 2,856 ms | **28.3** | **74.2%** |

![Speculative decoding — Qwen 7B](images/06_speculative_qwen-7b.png)

*Figure 1: 1.78× throughput with ~0.3 GB extra memory for the draft model.*

**74% acceptance** means roughly 3 out of 4 drafted tokens survive verification — excellent for a 0.5B → 7B draft/target pair in the same model family.

---

## Cross-model summary (M3)

| Target | Baseline tok/s | Speculative tok/s | Speedup | Status |
|--------|----------------|-------------------|---------|--------|
| Qwen 7B | 15.9 | **28.3** | 1.78× | ✅ |
| Llama 3.1 8B | 18.8 | — | — | ❌ draft OOM/error |
| Mistral 7B | 19.1 | — | — | ❌ draft OOM/error |

Llama and Mistral speculative runs failed on M3 — likely draft model pairing or memory pressure with two loaded models. On **M5 Max**, speculative runs succeeded for Llama 8B (check `results/Mac_M5_Max/article_06_speculative-decoding/`).

**Lesson:** Speculative decode needs **headroom for two models**. Budget ~5–6 GB for 7–8B target + 0.5–1B draft at w4.

---

## The acceptance rate equation (intuition)

Expected speedup grows with \(\alpha^k\) — high acceptance and larger \(k\) both help, but diminishing returns kick in when verification cost dominates.

Practical tips:

- **Match families** — Qwen draft → Qwen target worked; cross-family drafts often lower \(\alpha\)  
- **Same tokenizer** — required for token-level verification  
- **Keep draft tiny** — 0.5B–1B is the sweet spot on laptop RAM  

---

## When to use speculative decoding

| ✅ Good fit | ❌ Skip |
|------------|--------|
| Long generations (essays, code) | Short replies (<50 tokens) |
| 32 GB+ RAM or optimized w4 stack | Already at 200 tok/s (tiny models) |
| Same-domain draft/target pairs | Cross-architecture drafts |
| Chat UX where tok/s matters | TTFT-critical only workloads |

---

## Run it

```bash
./scripts/run_article.sh 6 "Mac M3"

python scripts/run_benchmark.py \
  --preset qwen-7b \
  --config w4 \
  --speculative \
  --hardware "Mac M3"
```

JSON fields to inspect: `draft_accept_rate`, `draft_model_repo`, `benchmark_mode: speculative`.

---

## Series wrap-up

Over seven posts we measured:

1. **Unified memory** constraints on Mac  
2. **Weight quant** — 3.5× speed at w4  
3. **KV cache quant** — long-context insurance  
4. **Prefill tuning** — TTFT at scale  
5. **Model ladder** — 0.5B to 70B  
6. **Full stack** — combined recipe  
7. **Speculative decode** — draft-model speedup  

All reproducible from the [LLM-Inference](https://github.com/Chirumamilla1522/LLM-Inference) repo.

**Bonus post:** [Context length & prefix cache](07-context-and-cache.md) — cold vs warm TTFT and the RAG wall.

---

## References

1. Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023) — [arXiv:2211.08920](https://arxiv.org/abs/2211.08920)  
2. Chen et al., *Accelerating LLM Decoding with Speculative Sampling* (2023) — [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)  
3. Cai et al., *Medusa* (2024) — [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)  
4. mlx-lm speculative API — [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)  

---

**← Previous:** [Part 6: Full Stack](05-full-optimization-stack.md)  
**Bonus →** [Context & Prefix Cache](07-context-and-cache.md)

**Tags:** `LLM` `Speculative Decoding` `Inference` `Apple Silicon` `Performance` `MLX`
