# 2-week Medium posting schedule

**Cadence:** Mon / Wed / Fri (7 posts in 14 days)  
**Best time:** 8–10 AM local, or Tue 10 AM if your audience is US/EU mixed  
**Series name:** *Local LLMs on Apple Silicon*

---

## Week 1 — Foundations

| Day | Date (example) | Post | File | Hook for social |
|-----|----------------|------|------|-----------------|
| **Mon** | Aug 12 | 1 — Introduction | `00-introduction.md` | “I benchmarked Llama 8B on an M3 Mac. FP16 uses 16 GB and runs at 5 tok/s.” |
| **Wed** | Aug 14 | 2 — Weight quantization | `01-weight-quantization.md` | “4-bit weights: same model, 3× less RAM, 3.5× faster decode.” |
| **Fri** | Aug 16 | 3 — KV cache | `02-kv-cache-quantization.md` | “Your context window has a second memory bill. Here’s how to shrink it.” |

---

## Week 2 — Tuning & speed

| Day | Date (example) | Post | File | Hook for social |
|-----|----------------|------|------|-----------------|
| **Mon** | Aug 19 | 4 — Prefill & TTFT | `03-prefill-and-ttft.md` | “Time-to-first-token is a different problem than tokens/sec.” |
| **Wed** | Aug 21 | 5 — Model size ladder | `04-model-size-ladder.md` | “Qwen 0.5B hits 238 tok/s on a laptop. Llama 8B hits 21.” |
| **Fri** | Aug 23 | 6 — Full stack | `05-full-optimization-stack.md` | “FP16 vs fully optimized: 16 GB → 5 GB, 5.6 → 20 tok/s.” |
| **Mon** | Aug 26 | 7 — Speculative decoding | `06-speculative-decoding.md` | “A tiny draft model boosted Qwen-7B from 16 to 28 tok/s.” |

---

## Week 3+ (optional backlog)

| Post | File | When |
|------|------|------|
| 8 — Context & prefix cache | `07-context-and-cache.md` | Wed Aug 28 |
| 9 — MLX vs llama.cpp | (from `docs/articles/10-runtimes.md`) | Fri Aug 30 |
| 10 — Serving at scale | `docs/articles/08-serving.md` | concept |
| 11 — Tradeoffs checklist | `docs/articles/11-tradeoffs.md` | capstone |

---

## Pre-publish checklist (each post)

- [ ] Upload hero chart from `docs/medium/images/`
- [ ] Add 5 tags from article footer
- [ ] Link to GitHub repo in first or last section
- [ ] Cross-link previous/next series part
- [ ] Pin Post 1 to profile after publish
- [ ] Share hook on X/LinkedIn same day

---

## Engagement ideas

- **Poll (Post 2):** “What quant do you run locally? FP16 / 8-bit / 4-bit / 2-bit”
- **Comment bait (Post 5):** “What’s the largest model you’ve run on Apple Silicon?”
- **CTA (Post 7):** “Clone the repo and reply with your M-series numbers”
