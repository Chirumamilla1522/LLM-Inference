# Articles index

**Rule:** one article per **optimization** (or one per **small related set**). No micro-posts per model family, bit width, or sweep axis.

| # | Article | What it covers | Benchmarked in repo |
|---|---------|----------------|---------------------|
| 0 | [Introduction: local LLMs on Apple Silicon](articles/00-introduction.md) | M3 vs M5, unified memory, metrics, repo workflow | demo run |
| 1 | [Weight quantization](optimizations/weight-quantization.md) | fp16, 8 / 4 / 2-bit weights | yes |
| 2 | [KV cache quantization](optimizations/kv-cache-quantization.md) | `kv_bits` during decode | yes |
| 3 | [Prefill & Flash Attention](optimizations/prefill-and-flash-attention.md) | `prefill_step_size`, TTFT | yes |
| 4 | [Model size & memory ladder](articles/04-model-size.md) | 0.5B→72B presets, what fits 24 GB | yes (presets) |
| 5 | [The full optimization stack](optimizations/all-optimizations.md) | Combining weights + KV + prefill | yes |
| 6 | [Speculative decoding](articles/06-speculative-decoding.md) | Draft + target models | yes (`--speculative`) |
| 7 | [Context, generation length & prompt cache](articles/07-context-and-cache.md) | `-p` / `-g` sweeps, prefix KV reuse | yes |
| 8 | [Production serving at scale](articles/08-serving.md) | Continuous batching, PagedAttention, scheduling | concept |
| 9 | [Parallelism for huge models](articles/09-parallelism.md) | Tensor, pipeline, expert parallel | concept |
| 10 | [Local runtimes compared](articles/10-runtimes.md) | MLX vs llama.cpp vs Ollama | concept |
| 11 | [Quality, cost & when to optimize what](articles/11-tradeoffs.md) | Quant quality, local vs API, decision tree | concept |

**Total: 12 articles** (8 with MLX benchmarks via `run_article.py`, 4 concept manifests).

```bash
./scripts/run_article.sh all "Mac M3"    # articles 0–7
./scripts/run_article.sh 8 "Mac M3"      # concept manifest only
```

---

## Sets (why some articles bundle topics)

| Set article | Techniques grouped | Reason |
|-------------|-------------------|--------|
| **4** | Many model presets | One scaling story, not one post per model |
| **7** | Long prompt, long output, prefix cache | Same bottleneck: KV / context memory |
| **8** | Batching + paged KV + scheduling | Same layer: multi-user serving |
| **9** | TP + PP + EP | Same layer: multi-device |
| **10** | MLX + llama.cpp + Ollama | One “what should I run?” post |
| **11** | Quality + cost + checklist | One “what do I pick?” post |

---

## Not separate articles

These stay **sections** inside the articles above—not their own posts:

- w2 vs w4 vs w8 → § in **Article 1**
- KV + prefill interaction → § in **Articles 2–3** or **5**
- Gemma vs Llama vs Qwen → § in **Article 4**
- M3 vs M5 numbers → **Article 0** + tables in **5**
- Reproducibility / trials → § in **Article 0**
- LoRA, MoE, compile, offload → § in **6**, **9**, or **11** until benchmarked

Full technique list (reference only): [INFERENCE_OPTIMIZATIONS_CATALOG.md](INFERENCE_OPTIMIZATIONS_CATALOG.md).

---

## Publish order

```text
0 → 1 → 2 → 3 → 4 → 5   (MLX benchmarks, capstone last)
6 → 7                     (when benches exist)
8 → 9 → 10 → 11           (concept / ecosystem, any order)
```

Sweep commands: [ARTICLE_SERIES.md](ARTICLE_SERIES.md).
