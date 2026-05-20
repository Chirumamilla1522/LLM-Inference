# LLM inference optimizations — full catalog

A **taxonomy of techniques** used in production and research LLM serving. Each technique maps to one of the [**12 articles**](ARTICLES_INDEX.md) (main topic or a section inside a set)—not its own post.

**Legend**

| Status | Meaning |
|--------|---------|
| **Benchmarked** | Implemented in `scripts/optimizations.py` + sweep |
| **Planned** | Good fit for this MLX/Mac repo; not implemented yet |
| **Reference** | Important for the series; measure elsewhere or concept-only on Mac |
| **Server-only** | Needs multi-request serving stack (vLLM, TGI, etc.) |

**Series:** [ARTICLES_INDEX.md](ARTICLES_INDEX.md) (**12 articles**, one per optimization or set) · [ARTICLE_SERIES.md](ARTICLE_SERIES.md) (sweep CLI)

---

## 1. Memory & weight compression

| Technique | What it does | Article angle | Status |
|-----------|--------------|---------------|--------|
| **FP16 / BF16 weights** | Baseline precision | “Native” memory footprint on unified memory | **Benchmarked** (`fp16`) |
| **INT8 / 4-bit / 2-bit weights** | Smaller checkpoints, less bandwidth | Sweet spot for M3 vs M5; quality tradeoffs | **Benchmarked** (`w8`, `w4`, `w2`) |
| **GPTQ / AWQ / similar** | Post-training weight formats | Same story as 4-bit; MLX uses pre-quantized HF repos | **Benchmarked** (via mlx-community repos) |
| **Activation quantization** | Quantize activations during matmul | When weights alone are not enough | **Planned** |
| **Mixed precision (AMP)** | FP16/BF16 compute, selective FP32 | Default in many frameworks; compare to full fp16 load | **Reference** |
| **Pruning (structured/unstructured)** | Remove weights or channels | Smaller dense models vs quant | **Reference** |
| **Knowledge distillation** | Train smaller student model | Different model, not a runtime flag | **Reference** |

---

## 2. KV cache & context memory

| Technique | What it does | Article angle | Status |
|-----------|--------------|---------------|--------|
| **KV cache quantization** | Lower precision for stored K/V | Long generations on fixed RAM | **Benchmarked** (`kv_cache`) |
| **Grouped-query / multi-query attention (GQA/MQA)** | Fewer KV heads | Why 7B models differ in memory slope | **Reference** (architecture) |
| **PagedAttention** | Non-contiguous KV blocks | How vLLM serves many users; contrast with local single-stream | **Server-only** |
| **Prefix / prompt KV caching** | Reuse KV for repeated system prompt | RAG and chat apps with static instructions | **Planned** |
| **KV eviction (H2O, StreamingLLM, …)** | Drop old tokens’ KV | “Infinite” context on bounded RAM | **Planned** |
| **Sliding-window attention** | Bounded KV by design | Mistral-style long context without full KV | **Reference** |

---

## 3. Attention & prefill (compute)

| Technique | What it does | Article angle | Status |
|-----------|--------------|---------------|--------|
| **Flash Attention / tiled attention** | IO-aware attention blocks | TTFT and long prompts; MLX uses internally | **Benchmarked** (proxy: `prefill`) |
| **Prefill chunking (`prefill_step_size`)** | Process prompt in chunks | Tunable TTFT vs peak memory | **Benchmarked** (`prefill`) |
| **Sparse / local attention** | Skip full quadratic cost | Long-context models | **Reference** |
| **Kernel fusion** | Fewer GPU round-trips | Framework-level; hard to toggle in user code | **Reference** |
| **RoPE scaling / YaRN** | Extrapolate position encodings | Long context without retraining | **Reference** |

---

## 4. Decode speed (token generation)

| Technique | What it does | Article angle | Status |
|-----------|--------------|---------------|--------|
| **Speculative decoding** | Draft model proposes tokens; target verifies | 2×+ decode on Apple Silicon? | **Planned** |
| **Medusa / multi-head speculative** | Extra heads predict multiple tokens | Variant of speculative path | **Planned** |
| **Parallel decoding / Jacobi** | Multiple candidates in one step | Research-serving topic | **Reference** |
| **CUDA graphs / Metal capture** | Replay fixed op sequence | Low-latency steady-state decode | **Planned** (MLX-dependent) |

---

## 5. Batching & scheduling (throughput at scale)

| Technique | What it does | Article angle | Status |
|-----------|--------------|---------------|--------|
| **Continuous batching** | Mix prefill + decode in one batch | Why local `mlx-lm` ≠ datacenter serving | **Server-only** |
| **Static batching** | Fixed batch size | Baseline for throughput benchmarks | **Reference** |
| **Request scheduling / prioritization** | SLOs, preemption | Production serving design | **Server-only** |
| **Disaggregated prefill/decode** | Separate pools for each phase | Large-scale architecture | **Server-only** |

---

## 6. Parallelism (multi-GPU / multi-node)

| Technique | What it does | Article angle | Status |
|-----------|--------------|---------------|--------|
| **Tensor parallelism** | Shard layers across devices | When one GPU is not enough | **Reference** (multi-GPU Mac rare) |
| **Pipeline parallelism** | Stage layers on devices | Very large models | **Reference** |
| **Expert parallelism (MoE)** | Route tokens to expert shards | Mixtral-class models | **Reference** |
| **Data parallelism** | Duplicate weights, split batch | Training; less common for single-user inference | **Reference** |

---

## 7. Model & architecture choices

| Technique | What it does | Article angle | Status |
|-----------|--------------|---------------|--------|
| **Smaller parameter count** | 0.5B–9B vs 70B | Your `list_models.py` ladder | **Benchmarked** (presets) |
| **MoE (e.g. Mixtral)** | Active params &lt; total params | Memory vs quality per active token | **Planned** (add preset) |
| **Long-context variants** | 128k vs 8k | Memory grows with context | **Planned** (sweep `-p`) |
| **Instruction-tuned vs base** | Same size, different use | Hold constant for fair benches | **Reference** |

---

## 8. Adapters & dynamic weights

| Technique | What it does | Article angle | Status |
|-----------|--------------|---------------|--------|
| **LoRA / QLoRA at inference** | Low-rank adapters on frozen base | Extra memory and matmul cost | **Planned** |
| **Multi-LoRA serving** | Swap adapters per request | Server feature | **Server-only** |

---

## 9. Compilation & runtime

| Technique | What it does | Article angle | Status |
|-----------|--------------|---------------|--------|
| **Graph / op compilation** | Fuse and specialize graphs | `mlx.compile` and friends | **Planned** |
| **Quantized matmul kernels** | Metal kernels for 4-bit | Why same bits ≠ same speed across frameworks | **Reference** |
| **CPU offload / disk offload** | Weights not all on GPU | Run huge models slowly vs OOM | **Planned** |
| **Memory mapping weights** | Lazy load from SSD | Cold start vs RAM pressure | **Reference** |

---

## 10. Application-level (still “inference”)

| Technique | What it does | Article angle | Status |
|-----------|--------------|---------------|--------|
| **Prompt caching (semantic)** | Reuse prior completions | Cost/latency for agents | **Reference** |
| **RAG: retrieve + short context** | Smaller effective prompt | Latency stack: embed + LLM | **Reference** |
| **Tool / JSON constrained decoding** | Logits masking | Extra CPU per token | **Reference** |
| **Early exit / confidence stopping** | Stop when “good enough” | Tokens saved | **Reference** |

---

## Map techniques → articles

Each technique maps to **one** of the [12 articles](ARTICLES_INDEX.md)—as the main topic or a **section** inside a set article.

| Art. | Post | Techniques in this catalog |
|------|------|---------------------------|
| 0 | Introduction | Methodology |
| 1 | Weight quant | §1 fp16, w8/w4/w2, GPTQ/AWQ |
| 2 | KV cache quant | §2 KV quant; long `-g` as § |
| 3 | Prefill | §3 Flash, prefill chunking; `-p` sweep as § |
| 4 | Model ladder | §7 model size, families as § |
| 5 | Full stack | Combines §1–3 |
| 6 | Speculative | §4 speculative, Medusa |
| 7 | Context set | §2 prefix cache, §3 length; `-p`/`-g` sweeps |
| 8 | Serving set | §5 batching, PagedAttention, scheduling, disaggregated |
| 9 | Parallelism set | §6 TP, PP, EP/MoE |
| 10 | Runtimes set | Framework comparison (not per-row) |
| 11 | Tradeoffs set | §1 activation quant, §8 LoRA, §10 app-level, quality/cost |

Pruning, distillation, kernel fusion, etc. → mention in **Article 11** unless benchmarked later.

---

## Metrics per optimization type

| Optimization family | Primary metrics | Secondary |
|--------------------|-----------------|-----------|
| Weight quant | Peak GB, tok/s | Quality (optional perplexity / spot checks) |
| KV quant | Peak GB (long `-g`), tok/s | TTFT (usually unchanged) |
| Prefill / Flash | TTFT | Peak GB during prefill |
| Speculative decode | tok/s, accept rate | Effective memory (two models) |
| Batching (server) | Throughput per GPU, latency p99 | — |
| Context sweep | TTFT vs `-p`, OOM threshold | tok/s at fixed `-g` |

Always record in JSON: `hardware`, `model_preset`, `configuration`, `prompt_tokens`, `generation_tokens`.

---

## Adding a new optimization to the repo

1. Add a field to `OptimizationConfig` in `scripts/optimizations.py`.
2. Wire `resolve_run_params()` → `mlx_lm` / `stream_generate` kwargs.
3. Add `docs/optimizations/<name>.md` and a row in this catalog (**Benchmarked**).
4. Add or extend the matching article in [ARTICLES_INDEX.md](ARTICLES_INDEX.md) (do not create a new post unless it is a new optimization family).
5. Update sweep order in `iter_sweep_configs()` only if it should join the default 16-config matrix (optional).

---

## Related reading (external)

- [MLX](https://github.com/ml-explore/mlx) / [mlx-lm](https://github.com/ml-explore/mlx-lm)
- FlashAttention papers (Dao et al.)
- vLLM PagedAttention, continuous batching
- Speculative decoding (Leviathan et al.; Chen et al.)
