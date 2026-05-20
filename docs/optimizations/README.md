# LLM inference optimizations

Each **benchmarked** technique has its own guide. The combined guide explains how they stack in this repo.

**Articles (12 posts):** [ARTICLES_INDEX.md](../ARTICLES_INDEX.md) — one per optimization or set.

**All techniques (reference):** [INFERENCE_OPTIMIZATIONS_CATALOG.md](../INFERENCE_OPTIMIZATIONS_CATALOG.md)

**Sweep commands:** [ARTICLE_SERIES.md](../ARTICLE_SERIES.md)

## Benchmarked in this repo

| Document | Topic |
|----------|--------|
| [Weight quantization](weight-quantization.md) | fp16, 8-bit, 4-bit, 2-bit model weights |
| [KV cache quantization](kv-cache-quantization.md) | Compressing the growing key/value cache during decode |
| [Prefill & Flash Attention](prefill-and-flash-attention.md) | Prompt processing, tiling, and TTFT |

## Combined view

| Document | Topic |
|----------|--------|
| [**All optimizations together**](all-optimizations.md) | How the three layers interact, sweep design, metrics, hardware, code map |

## Related

- [**Articles index**](../ARTICLES_INDEX.md) — 12 posts
- [**Technique catalog**](../INFERENCE_OPTIMIZATIONS_CATALOG.md) — reference taxonomy
- [**Article series / sweeps**](../ARTICLE_SERIES.md)
- [Benchmark workflow](../BENCHMARK_WORKFLOW.md) — how to run experiments
- [Article draft / capstone table](../../notes.md) — M3 vs M5 Max results
