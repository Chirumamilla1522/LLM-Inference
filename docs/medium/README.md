# Medium publishing package

Ready-to-publish drafts for **Local LLMs on Apple Silicon** — with **paper-style workflow diagrams** and **benchmark result plots**.

## Image inventory (36 PNGs)

### Workflow / how-it-works (`images/workflows/`)

| File | Topic |
|------|--------|
| `00_unified_memory.png` | Apple Silicon unified DRAM |
| `00_inference_pipeline.png` | Prefill → TTFT → decode |
| `01_affine_quantization.png` | Affine quant (Jacob / GPTQ / AWQ) |
| `01_bandwidth_bound.png` | Why w4 speeds decode |
| `02_kv_cache_workflow.png` | KV growth + 4-bit curve |
| `02_attention_with_cache.png` | Decode-step attention |
| `02_gqa_vs_mha.png` | GQA vs multi-head |
| `03_prefill_vs_decode.png` | Two bottlenecks |
| `03_flash_attention.png` | FlashAttention tiling |
| `04_fit_ladder.png` | What fits on 24 GB |
| `05_optimization_funnel.png` | Stacking funnel |
| `05_decision_tree.png` | When to enable what |
| `06_speculative_workflow.png` | Draft + verify |
| `06_accept_reject.png` | Accept / reject round |
| `07_prefix_cache_workflow.png` | Cold vs warm prefix |
| `07_rag_wall.png` | RAG latency wall |

### Result plots (`images/`)

Regenerated from `results/Mac_M3/` JSON — multi-model bars, Pareto scatters, TTFT curves, workload stress, full-stack comparisons, speculative speed/memory, etc.

## Regenerate everything

```bash
python scripts/plot_medium_diagrams.py
python scripts/plot_medium_charts.py --hardware "Mac M3"
```

## Articles

| File | Figures (approx) | Post |
|------|------------------|------|
| [00-introduction.md](00-introduction.md) | 3 | Series opener |
| [01-weight-quantization.md](01-weight-quantization.md) | 7 | Weight quant |
| [02-kv-cache-quantization.md](02-kv-cache-quantization.md) | 5 | KV cache |
| [03-prefill-and-ttft.md](03-prefill-and-ttft.md) | 5 | Prefill / TTFT |
| [04-model-size-ladder.md](04-model-size-ladder.md) | 4 | Model ladder |
| [05-full-optimization-stack.md](05-full-optimization-stack.md) | 6 | Full stack |
| [06-speculative-decoding.md](06-speculative-decoding.md) | 5 | Speculative |
| [07-context-and-cache.md](07-context-and-cache.md) | 7 | Bonus RAG / cache |

Schedule: [SCHEDULE.md](SCHEDULE.md)

## How to publish on Medium

1. Open [medium.com/new-story](https://medium.com/new-story)
2. Copy the markdown body (skip the YAML header)
3. For each `![...](images/...)` / `images/workflows/...`, **upload the matching PNG** and replace the placeholder
4. Add tags from the article footer
5. Link posts as a Series + Previous/Next URLs after publish

### Suggested upload order per post

Each article is structured as:

1. **Workflow figure(s)** first (how the optimization works)  
2. **Result plot(s)** next (what we measured)  
3. Tables + fun facts + references  

## Reproducibility footnote

> Numbers from [LLM-Inference](https://github.com/Chirumamilla1522/LLM-Inference) on MLX. Hardware: Mac M3, 24 GB unless noted.
