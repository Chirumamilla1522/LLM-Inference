# Math vs programming optimizations

Every technique in this benchmark suite has two sides:

| Side | What it is | Typical artifacts |
|------|------------|-------------------|
| **Math / algorithmic** | Change the *computation* or *numerical representation* | Equations, complexity, error bounds |
| **Programming / systems** | Change *how* the math runs on hardware | Bit packing, kernels, memory layout, batching |

Articles in this repo should explain **both**: the formula (why it works) and the code path (what MLX actually does).

---

## Map: articles → math + code

| Article | Math focus | Programming focus |
|---------|------------|-------------------|
| [1 Weight quant](weight-quantization.md) | Affine quantization, memory scaling | Packed INT weights, dequant matmul |
| [2 KV cache quant](kv-cache-quantization.md) | Cache growth \(O(T)\), GQA head count | `kv_bits`, `to_quantized()` on cache |
| [3 Prefill / Flash](prefill-and-flash-attention.md) | Attention \(O(n^2)\), online softmax | `prefill_step_size`, Metal tiled kernels |
| [6 Speculative](speculative-decoding.md) | Acceptance probability, expected speedup | `draft_model`, `num_draft_tokens` |
| [7 Context / cache](articles/07-context-and-cache.md) | \(T\) in KV and TTFT | `-p`, `-g`, `save_prompt_cache` |

---

## Unified memory budget (all layers)

Peak RAM is dominated by three terms:

$$
M_{\text{peak}} \approx M_{\text{weights}} + M_{\text{KV}} + M_{\text{activations}}
$$

**Weights (static):**

$$
M_{\text{weights}} \approx \frac{N_{\text{params}} \cdot b_w}{8}
$$

where \(N_{\text{params}}\) is parameter count and \(b_w\) is bits per weight (16, 8, 4, or 2).

**KV cache (grows with sequence length \(T\)):**

$$
M_{\text{KV}} \approx 2 \cdot L \cdot H_{\text{kv}} \cdot T \cdot D \cdot \frac{b_{\text{kv}}}{8}
$$

- \(L\) = number of layers  
- \(H_{\text{kv}}\) = KV heads (equals query heads, or fewer with GQA)  
- \(D\) = head dimension  
- Factor \(2\) = separate K and V tensors  
- \(b_{\text{kv}}\) = 16 (full) or 4 (quantized in our benchmarks)

**Example (Llama-class 8B, \(L{=}32\), \(H_{\text{kv}}{=}8\), \(D{=}128\), \(T{=}640\) tokens, FP16 KV):**

$$
M_{\text{KV}} \approx 2 \times 32 \times 8 \times 640 \times 128 \times 2 \text{ bytes} \approx 84 \text{ MB}
$$

Same setup with **4-bit KV** (\(b_{\text{kv}}{=}4\)):

$$
M_{\text{KV}} \approx 2 \times 32 \times 8 \times 640 \times 128 \times 0.5 \text{ bytes} \approx 21 \text{ MB}
$$

Weights at 4-bit for 8B:

$$
M_{\text{weights}} \approx \frac{8 \times 10^9 \times 4}{8} \approx 4 \text{ GB}
$$

---

## Decode throughput (roofline intuition)

When decode is **memory-bandwidth bound**, tokens per second scales with bytes read per token:

$$
\text{tok/s} \approx \frac{B_{\text{mem}}}{B_{\text{token}}}
$$

where \(B_{\text{mem}}\) is sustained unified-memory bandwidth (GB/s) and \(B_{\text{token}}\) is bytes moved from RAM per generated token (mostly weights + KV updates).

**Example:** If each decode step reads ~5 GB of weights+activations at effective 100 GB/s:

$$
\text{tok/s} \approx \frac{100}{5} = 20 \text{ t/s}
$$

Halving weight bytes (\(b_w: 16 \to 4\)) roughly halves \(B_{\text{token}}\) for the weight-dominated part → **~2× tok/s** if bandwidth-bound (matches many M3 8B observations).

---

## Programming patterns used in MLX / this repo

| Pattern | Level | Example in repo |
|---------|-------|-----------------|
| **Separate checkpoints per bit width** | Load time | `fp16` vs `w4` HF repos |
| **Group-wise scales** | Storage | GPTQ/AWQ-style blocks in `mlx-community` |
| **Bit-packed integers** | Storage | 2× 4-bit values per byte |
| **Fused dequant matmul** | Kernel | Metal kernel in MLX, not Python loops |
| **Cache object quantization** | Runtime | `kv_bits=4` → `to_quantized()` |
| **Tiled attention** | Kernel | Flash-style inside MLX |
| **Chunked prefill loop** | Control flow | `prefill_step_size` 512 vs 2048 |
| **Subprocess isolation** | Reliability | `run_benchmark.py` sweep |

---

## Bitwise vs floating-point (quick reference)

| Operation | Math view | Programming view |
|-----------|-----------|------------------|
| Store weight \(x\) in 4 bits | \(q = \text{quantize}(x, s, z)\) | `uint8` array + scale tensor per group |
| Matrix multiply | \(\hat{W} \approx s \cdot Q\) | Dequant fused into GEMM |
| Store KV vector | Round FP16 → INT4 groups | In-place cache quant after step 5000 (MLX default) |
| Attention scores | \(\mathrm{softmax}(QK^\top/\sqrt{d})\) | Tiled softmax without full \(N \times N\) matrix |

---

## See also

- [Weight quantization](weight-quantization.md) — affine quant equations + packing example  
- [KV cache quantization](kv-cache-quantization.md) — cache size formula + `kv_bits`  
- [Prefill & Flash Attention](prefill-and-flash-attention.md) — attention math + tiling  
- [Speculative decoding](speculative-decoding.md) — acceptance rate math  
- [All optimizations together](all-optimizations.md) — stacked configs and combined budget
