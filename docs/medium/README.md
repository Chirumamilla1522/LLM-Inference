# Medium publishing guide

Ready-to-publish drafts for the **Local LLMs on Apple Silicon** series.

## Contents

| File | Medium title | Post # |
|------|--------------|--------|
| [00-introduction.md](00-introduction.md) | Running 8B LLMs on a MacBook: What Actually Matters | 1 |
| [01-weight-quantization.md](01-weight-quantization.md) | 4-Bit Weights Changed Everything on My M3 Mac | 2 |
| [02-kv-cache-quantization.md](02-kv-cache-quantization.md) | The Hidden Memory Hog: KV Cache Quantization | 3 |
| [03-prefill-and-ttft.md](03-prefill-and-ttft.md) | Why Your Chatbot Feels Slow Before the First Word | 4 |
| [04-model-size-ladder.md](04-model-size-ladder.md) | From 0.5B to 70B: What Fits on Apple Silicon | 5 |
| [05-full-optimization-stack.md](05-full-optimization-stack.md) | Stacking Optimizations: 3.5× Faster Than FP16 | 6 |
| [06-speculative-decoding.md](06-speculative-decoding.md) | Draft Models: Free Speed Without Retraining | 7 |

**Bonus (week 3):** [07-context-and-cache.md](07-context-and-cache.md)

Charts live in [`images/`](images/). Regenerate after new benchmark runs:

```bash
python scripts/plot_medium_charts.py --hardware "Mac M3"
```

## How to publish on Medium

1. Open [medium.com/new-story](https://medium.com/new-story)
2. Copy the markdown body from each file (skip the YAML-style header block)
3. Upload images from `docs/medium/images/` — drag into the story where `![caption](images/...)` appears
4. Add suggested **tags** from each article footer
5. Link **Series** parts: at the bottom, add “← Previous | Next →” links to your published URLs
6. Optional: import via [Medium import tool](https://medium.com/p/import) if you host markdown on GitHub Pages

## Reproducibility footnote (use on every post)

> All numbers from the open-source [LLM-Inference](https://github.com/Chirumamilla1522/LLM-Inference) benchmark harness on MLX. Hardware: Mac M3, 24 GB unified memory unless noted.

## License

Benchmark data and charts: same as repo. Medium story text: yours to publish.
