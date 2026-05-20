# Weight quantization

**What it optimizes:** Static model weights (the billions of parameters loaded before generation).

**Benchmark labels:** `fp16`, `w8`, `w4`, `w2`

[← All optimizations](all-optimizations.md) · [KV cache →](kv-cache-quantization.md)

---

## The problem

A transformer’s **weights** are the largest fixed cost in memory. For an 8B-parameter model:

| Precision | Bits per weight (approx.) | Weight memory (approx.) |
|-----------|---------------------------|-------------------------|
| FP16 / BF16 | 16 | ~16 GB |
| 8-bit | 8 | ~8–10 GB |
| 4-bit | 4 | ~5–6 GB |
| 2-bit | 2 | ~3–4 GB |

On a **24 GB Mac**, fp16 for an 8B model can consume most of unified memory before the KV cache or activations exist. Larger models (32B+) may not load at all at full precision.

Quantization answers: *“Can we store and compute with fewer bits per weight while keeping output quality acceptable?”*

---

## How it works

High-precision weights are floating-point numbers. **Quantization** maps them to a small set of integer levels plus a **scale** (and sometimes **zero-point**) per group of weights.

```mermaid
flowchart LR
  FP["FP16 weight matrix<br/>high memory"] --> Q["Quantize offline"]
  Q --> INT["INT4 / INT8 stored weights<br/>low memory"]
  INT --> DQ["Dequantize on the fly<br/>during matmul"]
  DQ --> OUT["Same shape output<br/>approximate values"]
```

Common schemes in MLX Hugging Face repos:

- **Per-group affine quantization** — A block of weights shares one scale; good balance of size and accuracy.
- **Pre-quantized checkpoints** — `mlx-community` publishes separate repos per bit width; no runtime quant in our benchmarks.

---

## Math: affine quantization

For one weight (or one element in a tensor), **affine quantization** maps a float \(x\) to an integer code \(q\) with bit width \(b\):

$$
q = \mathrm{clip}\left( \mathrm{round}\left( \frac{x}{s} + z \right),\ 0,\ 2^b - 1 \right)
$$

**Dequantization** (reconstruct approximate float):

$$
\hat{x} = s \cdot (q - z)
$$

| Symbol | Meaning |
|--------|---------|
| \(s\) | Scale (positive float, per group) |
| \(z\) | Zero-point (integer offset, per group) |
| \(b\) | Bits per stored value (8, 4, 2 in our sweeps) |

**Group-wise:** Weights are split into blocks of size \(G\) (e.g. 64 or 128). One \((s, z)\) pair per block → much lower metadata overhead than per-tensor scales.

### Memory scaling (parameters only)

$$
M_{\text{weights}} \approx \frac{N_{\text{params}} \cdot b_w}{8} + M_{\text{scales}}
$$

\(M_{\text{scales}}\) is small (often &lt; 1% of weight bytes). Dominated by \(N_{\text{params}} \cdot b_w\).

| Model | \(N_{\text{params}}\) | \(b_w=16\) | \(b_w=4\) |
|-------|----------------------|------------|-----------|
| 8B | \(8 \times 10^9\) | ~16 GB | ~4 GB |
| 7B | \(7 \times 10^9\) | ~14 GB | ~3.5 GB |
| 32B | \(32 \times 10^9\) | ~64 GB | ~16 GB |

### Numeric example (one weight, 4-bit)

Suppose a group uses \(s = 0.05\), \(z = 8\), and \(x = 0.42\):

$$
q = \mathrm{round}(0.42 / 0.05 + 8) = \mathrm{round}(16.4) = 16
$$

$$
\hat{x} = 0.05 \times (16 - 8) = 0.40 \quad (\text{error } 0.02)
$$

Matmul uses \(\hat{W}\) built from many such \(\hat{x}\) values—not the original FP16 \(x\).

### Quantization error (intuition)

Reconstruction error per element is bounded by half a quantization step:

$$
|x - \hat{x}| \lesssim \frac{s}{2}
$$

Lower \(b\) → larger typical \(s\) for the same dynamic range → larger error. That is the **quality vs size** tradeoff Articles 1 and 11 discuss.

---

## Programming: bits, packing, and matmul

### Bit width is a storage/layout choice

| Label | \(b_w\) | Bytes per weight (ideal) | What MLX loads |
|-------|---------|--------------------------|----------------|
| `fp16` | 16 | 2 | BF16/FP16 tensors |
| `w8` | 8 | 1 | INT8 + scales |
| `w4` | 4 | 0.5 | Packed nibbles + scales |
| `w2` | 2 | 0.25 | Packed 2-bit fields + scales |

**4-bit packing (conceptual):** Two quantized weights per byte:

```text
byte = (q_0 & 0x0F) | ((q_1 & 0x0F) << 4)
```

Kernels unpack → dequant → multiply in registers; Python never loops per weight.

### What our benchmark does (no runtime quant)

```python
# scripts/optimizations.py — resolution only
model_repo = MODEL_REPOS["llama3-8b"][4]  # w4
# → "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
```

```python
# scripts/run_benchmark.py — load pre-quantized checkpoint
model, tokenizer = load(params.model_repo)
# MLX Metal kernel: fused dequant + matmul inside forward()
```

Changing `fp16` → `w4` in an article means **loading a different file from Hugging Face**, not calling a quantize function at runtime.

### Pseudocode: decode step with 4-bit weights

```python
# Conceptual — actual code is MLX/C++
for layer in model.layers:
    W_q, scale, zp = layer.weight_packed, layer.scale, layer.zp
    W_hat = dequant(W_q, scale, zp)      # often fused into matmul
    x = matmul(x, W_hat)                 # Metal GEMM
```

---

## Why we need it (local inference on Apple Silicon)

### 1. Capacity — fit the model

Unified memory is shared by CPU, GPU, and everything else. Smaller weights leave room for:

- KV cache (grows with context)
- Framework buffers
- macOS and other apps

Without 4-bit weights, many laptops cannot run 8B models comfortably.

### 2. Bandwidth — faster decode

During **decode**, the GPU reads weights for every new token. If the machine is **memory-bandwidth bound**, halving weight size can nearly double effective throughput—fewer bytes from RAM per token.

### 3. Quality tradeoff

Lower bit width can reduce reasoning quality on hard tasks. The benchmark sweep lets you compare **speed and memory vs. precision** on your hardware, not assume one setting for all use cases.

```mermaid
quadrantChart
  title Weight precision tradeoff (conceptual)
  x Low memory use
  x High memory use
  y Lower quality risk
  y Higher quality risk
  w2: [0.2, 0.25]
  w4: [0.35, 0.55]
  w8: [0.55, 0.75]
  fp16: [0.9, 0.9]
```

---

## What changes in the inference pipeline

Weight quantization affects **both** phases, but differently:

| Phase | Effect of lower bit weights |
|-------|-----------------------------|
| **Prefill** | Less weight data to load per layer; can improve TTFT when bandwidth-bound |
| **Decode** | Smaller reads each step; often improves tokens/sec on Mac |

It does **not** reduce KV cache size—that is a separate optimization.

---

## How this repository implements it

We benchmark **four weight levels** as separate Hugging Face models (not runtime conversion):

| Label | `weight_bits` | Llama 3.1 8B example |
|-------|---------------|----------------------|
| `fp16` | 16 | `mlx-community/Meta-Llama-3.1-8B-Instruct-bf16` |
| `w8` | 8 | `mlx-community/Meta-Llama-3.1-8B-Instruct-8bit` |
| `w4` | 4 | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` |
| `w2` | 2 | `mlx-community/Llama-3-8B-Instruct-262k-2bit` |

Mistral fp16 uses `mlx-community/Mistral-7B-Instruct-v0.3` (full-precision MLX build; no separate `*-bf16` repo).

Config resolution (`scripts/optimizations.py`):

```text
weight_bits=4  →  load MODEL_REPOS[preset][4]
```

---

## Expected impact (article targets)

From [notes.md](../../notes.md) on **Mac M3 + Llama 3 8B**:

| Config | Memory | TTFT | Throughput |
|--------|--------|------|------------|
| Native FP16 | ~16.2 GB | 145 ms | 22 t/s |
| Optimized 4-bit | ~5.8 GB | 85 ms | 48 t/s |

Roughly **64% less memory** and about **2× throughput** for this class of hardware when moving from fp16-class to 4-bit weights.

---

## When to use which level

| Situation | Suggested starting point |
|-----------|--------------------------|
| 24 GB Mac, 7B–8B chat | `w4` or `w8` |
| Maximum quality, enough RAM | `fp16` |
| Smallest footprint, experimentation | `w2` (if repo exists) |
| 32B model on laptop | `w4` only; fp16 needs workstation RAM |

---

## Limitations

- **Not all bit widths exist** for every model on `mlx-community` (e.g. Mistral 2-bit may be missing).
- **2-bit** can noticeably hurt quality on some tasks.
- Quantization is **offline** in our setup—changing bits requires a different download, not a CLI flag.

---

## Code references

| Item | Location |
|------|----------|
| Repo map | `scripts/optimizations.py` → `DEFAULT_MODEL_REPOS` |
| Sweep order | `fp16` → `w8` → `w4` → `w2` in `iter_sweep_configs()` |
| Overrides | `models.json` |

---

## See also

- [Math vs programming overview](math-and-implementation.md) — unified memory formulas  
- [KV cache quantization](kv-cache-quantization.md) — shrinks *dynamic* memory during generation  
- [All optimizations together](all-optimizations.md) — combining weight bits with runtime flags
