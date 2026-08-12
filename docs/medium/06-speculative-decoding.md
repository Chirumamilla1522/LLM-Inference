---
title: "Draft Models: Free Speed Without Retraining"
subtitle: "Speculative decoding workflow from the papers — then 74% acceptance / 1.8× tok/s on Qwen-7B"
tags: LLM, Speculative Decoding, Inference, Apple Silicon, Performance, MLX
series: 7 of 7
read_time: 12 min
figures: 5
---

# Draft Models: Free Speed Without Retraining

*Part 7 of 7 — Local LLMs on Apple Silicon*

Quantization shrinks weights. Prefill helps TTFT. **Speculative decoding** is different: a small **draft** model proposes tokens; the large **target** verifies them in one parallel pass.

When the draft guesses right, you emit multiple tokens per target forward. **No retraining. Same quality.**

---

## How it works (Leviathan / Chen)

![Speculative workflow](images/workflows/06_speculative_workflow.png)

*Figure 1 — Workflow: baseline = 1 target forward per token; speculative = draft proposes k, target verifies once.*

![Accept / reject](images/workflows/06_accept_reject.png)

*Figure 2 — Workflow: accept matching prefixes; reject at first mismatch and resample from the target.*

Speedup depends on:

- **Acceptance rate** \(\alpha\) — fraction of draft tokens that survive  
- **Draft speed** — how cheap proposals are  
- **\(k\)** — tokens proposed per round (we use \(k=3\))

Papers: Leviathan et al. (2023), Chen et al. (2023); Medusa (2024) extends with multi-head drafts.

> **Fun fact:** Google reported up to **2–3×** speedup on T5-XXL with a draft **~15× smaller** and >60% acceptance — without changing output distribution.

---

## Results: Qwen 7B on Mac M3

| Mode | Draft | Peak GB | TTFT | tok/s | α |
|------|-------|---------|------|-------|---|
| Baseline w4 | — | 4.72 | 3,613 ms | 15.9 | — |
| Speculative w4 | Qwen 0.5B | 5.00 | 2,856 ms | **28.3** | **74.2%** |

![Speculative tok/s](images/06_speculative_qwen-7b.png)

*Figure 3 — Results: **1.78×** throughput at 74% acceptance.*

![Speed + memory](images/06_speculative_speed_memory.png)

*Figure 4 — Results: big speed gain for ~0.3 GB extra (draft in memory).*

| Target | Baseline | Speculative | Notes |
|--------|----------|-------------|-------|
| Qwen 7B | 15.9 | **28.3** | ✅ same-family draft |
| Llama 3.1 8B | 18.8 | — | ❌ failed on M3 (memory / pairing) |
| Mistral 7B | 19.1 | — | ❌ failed on M3 |

**Lesson:** you need headroom for **two models**. Budget ~5–6 GB for 7–8B target + 0.5–1B draft at w4. M5 Max has more success headroom.

---

## Practical tips

| ✅ Do | ❌ Don’t |
|------|---------|
| Same family draft/target | Cross-architecture drafts |
| Same tokenizer | Expect wins on 50-token replies |
| Tiny draft (0.5B–1B) | Run speculative when already at 200 tok/s |
| Long generations | Ignore memory for two models |

```bash
./scripts/run_article.sh 6 "Mac M3"
python scripts/run_benchmark.py --preset qwen-7b --config w4 \
  --speculative --hardware "Mac M3"
```

Inspect JSON: `draft_accept_rate`, `draft_model_repo`, `benchmark_mode: speculative`.

---

## Series wrap-up

1. Unified memory constraints  
2. Weight quant → 3.5× at w4  
3. KV quant → long-context insurance  
4. Prefill → TTFT at scale  
5. Model ladder → 0.5B–70B  
6. Full stack → combined recipe  
7. Speculative decode → draft speedup  

**Bonus:** [Context length & prefix cache](07-context-and-cache.md)

Repo: [LLM-Inference](https://github.com/Chirumamilla1522/LLM-Inference)

---

## References

1. Leviathan et al., *Speculative Decoding* (2023) — [arXiv:2211.08920](https://arxiv.org/abs/2211.08920)  
2. Chen et al., *Speculative Sampling* (2023) — [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)  
3. Cai et al., *Medusa* (2024) — [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)  
4. mlx-lm — [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)  

---

**← Previous:** [Part 6](05-full-optimization-stack.md) · **Bonus →** [Context & Cache](07-context-and-cache.md)

**Tags:** `LLM` `Speculative Decoding` `Inference` `Apple Silicon` `Performance`
