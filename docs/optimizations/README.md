# LLM inference optimizations

Each **benchmarked** technique has its own guide. The combined guide explains how they stack in this repo.

**Articles (12 posts):** [ARTICLES_INDEX.md](../ARTICLES_INDEX.md) — one per optimization or set.

**All techniques (reference):** [INFERENCE_OPTIMIZATIONS_CATALOG.md](../INFERENCE_OPTIMIZATIONS_CATALOG.md)

**Sweep commands:** [ARTICLE_SERIES.md](../ARTICLE_SERIES.md)

## Math and programming (read first for articles)

| Document | Topic |
|----------|--------|
| [**Math vs programming**](math-and-implementation.md) | Equations, examples, bitwise vs algorithmic map |

## Benchmarked in this repo

| Document | Topic |
|----------|--------|
| [Weight quantization](weight-quantization.md) | Affine quant math + packed weights |
| [KV cache quantization](kv-cache-quantization.md) | \(M_{\text{KV}}\) formula + `kv_bits` |
| [Prefill & Flash Attention](prefill-and-flash-attention.md) | Attention \(O(n^2)\) + `prefill_step_size` |
| [Speculative decoding](speculative-decoding.md) | Acceptance rate + `draft_model` |
| [**llama.cpp vs MLX**](llama-cpp-vs-mlx.md) | Article 10 — GGUF vs mlx-community, `llama-bench` |
| [**Workload stress matrix**](workload-stress-matrix.md) | Task × data × pressure × modality |

## Combined view

| Document | Topic |
|----------|--------|
| [**All optimizations together**](all-optimizations.md) | How the three layers interact, sweep design, metrics, hardware, code map |

## Diagrams & references

Each guide includes **mermaid figures** (pipelines, memory charts, sequence diagrams) and a **References** section with numbered citations.

**Master bibliography:** [REFERENCES.md](../REFERENCES.md) — papers [1]–[27], MLX + llama.cpp docs, citation examples for articles.

## Related

- [**References**](../REFERENCES.md) — full bibliography  
- [**Articles index**](../ARTICLES_INDEX.md) — 12 posts  
- [**Technique catalog**](../INFERENCE_OPTIMIZATIONS_CATALOG.md) — reference taxonomy  
- [**Article series / sweeps**](../ARTICLE_SERIES.md)  
- [Benchmark workflow](../BENCHMARK_WORKFLOW.md) — how to run experiments  
- [Article draft / capstone table](../../notes.md) — M3 vs M5 Max results
