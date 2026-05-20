# Speculative decoding

**What it optimizes:** **Decode throughput** — tokens per second after the first token.

**Benchmark:** Article 6 — `--speculative` with draft + target models.

[← Prefill](prefill-and-flash-attention.md) · [Math overview](math-and-implementation.md) · [All optimizations](all-optimizations.md)

---

## The problem (math view)

Autoregressive decoding generates **one token per target-model forward pass** (simplified):

$$
t_{\text{decode}} \approx T_{\text{out}} \cdot \tau_{\text{target}}
$$

where \(\tau_{\text{target}}\) is seconds per target forward step. Throughput:

$$
\text{tok/s} = \frac{1}{\tau_{\text{target}}}
$$

Speculative decoding introduces a **small draft model** that proposes \(k\) tokens cheaply; the **target** verifies them in fewer parallel steps.

---

## Math: acceptance and expected speedup

Let:

- \(k\) = `num_draft_tokens` (draft proposals per round)  
- \(\alpha\) = probability a draft token is **accepted** (matches target distribution in verification)  
- \(\tau_d\) = time per draft step (small model)  
- \(\tau_t\) = time per target verification step (large model)

**Idealized speedup** when verification batches well:

$$
\text{speedup} \approx \frac{k \cdot \tau_t}{\tau_d + \tau_{\text{verify}}}
$$

In practice \(\tau_{\text{verify}}\) depends on how many draft tokens survive.

**Acceptance rate** (what we log as `draft_accept_rate`):

$$
\alpha = \frac{\text{tokens accepted from draft}}{\text{total generated tokens}}
$$

Higher \(\alpha\) → draft matches target well (similar family / distilled model).

**Example:** If \(\alpha = 0.7\) and average accepted bundle is 2 tokens per target call vs 1 without speculation:

$$
\frac{\text{tok/s}_{\text{spec}}}{\text{tok/s}_{\text{base}}} \approx 1.4\text{–}2.0\times
$$

(Depends on \(\tau_d\) and memory—two models loaded.)

### Memory budget with two models

$$
M_{\text{peak}} \approx M_{\text{target}} + M_{\text{draft}} + M_{\text{KV,target}} + M_{\text{KV,draft}}
$$

Draft is small (e.g. 0.5B) but not free:

$$
M_{\text{draft}} \approx \frac{0.5 \times 10^9 \times 4}{8} \approx 0.25 \text{ GB at 4-bit}
$$

---

## Programming: MLX `draft_model`

### Repo mapping

```python
# scripts/optimizations.py
DRAFT_PRESET_BY_TARGET = {
    "llama3-8b": "qwen-0.5b",
    ...
}
resolve_draft_repo("llama3-8b", weight_bits=4)
# → mlx-community/Qwen2.5-0.5B-Instruct-4bit (example)
```

Draft and target must share **tokenizer vocabulary size** (checked in `run_benchmark.py`).

### API

```python
stream_generate(
    model, tokenizer, prompt,
    draft_model=draft_model,
    num_draft_tokens=3,
    max_tokens=128,
)
```

`GenerationResponse.from_draft` indicates whether a token came from the draft path—used to compute `draft_accept_rate` in JSON.

### Article 6 runs

```bash
./scripts/run_article.sh 6 "Mac M3"
# Compares llama3-8b_w4_baseline vs llama3-8b_w4_speculative per preset
```

### Pseudocode: speculative loop

```python
while not done:
    draft_tokens = []
    for _ in range(k):
        draft_tokens.append(draft_model.sample_one(cache_draft))
    # Target verifies draft_tokens in parallel (fused in MLX)
    n_accept = target.verify_and_accept(draft_tokens, cache_target)
    emit(draft_tokens[:n_accept])
    if n_accept < k:
        rewind_cache(draft, n_accept)  # reject suffix
```

Bitwise/matmul work is identical to normal decode—**extra control flow** and **second model weights** are the programming cost.

---

## When it helps

| Good | Poor |
|------|------|
| Draft is much smaller / faster | Draft vocab ≠ target |
| High \(\alpha\) (distilled pair) | RAM too tight for two models |
| Long generations (`-g` large) | TTFT-bound short replies |

---

## See also

- [Math vs programming overview](math-and-implementation.md)  
- [Weight quantization](weight-quantization.md) — draft often 4-bit  
- [Article 6 outline](../articles/06-speculative-decoding.md)
