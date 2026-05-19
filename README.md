# LLM-Inference

Benchmark open-source LLM inference on Apple Silicon with [MLX](https://github.com/ml-explore/mlx). Compares **fp16 (bf16)**, **8-bit**, **4-bit**, and **2-bit** weights, combined with optional **KV cache** and **prefill** optimizations.

Article draft: [notes.md](notes.md).

## Weight precisions

| Label | Meaning | Typical repo suffix |
|-------|---------|---------------------|
| `fp16` | Native bf16 baseline | `*-bf16` |
| `w8` | 8-bit weights | `*-8bit` |
| `w4` | 4-bit weights | `*-4bit` |
| `w2` | 2-bit weights | `*-2bit` (where available) |

## Runtime optimizations (combine with any weight level)

| Flag | Off | On |
|------|-----|-----|
| `kv_cache` | full-precision KV | 4-bit KV (`kv_bits=4`) |
| `prefill` | 512-token steps | 2048-token tiled prefill |

## Full sweep order (16 configs × model)

Per weight level (`fp16` → `w8` → `w4` → `w2`):

1. weight only (e.g. `fp16`, `w8`)
2. `+kv_cache`
3. `+prefill`
4. `+kv_cache+prefill`

```bash
./scripts/setup_env.sh
source .venv/bin/activate
huggingface-cli login   # required for Mistral fp16 (gated on Hugging Face)

./scripts/run_full_sweep.sh "Mac M3"
```

On a **24GB M3**, the sweep automatically skips `qwen-35b` (32B model OOMs at 8-bit+). Llama uses public **Llama 3.1** fp16 repos (not gated 3.0).

**32 runs** on M3 (2 models × 16 configs). On **64GB+** Macs, pass `--include-qwen` for all three models (48 runs).

### Weight precision only (4 runs per model)

```bash
python scripts/run_benchmark.py --sweep --weights-only --all-models --hardware "Mac M3"
```

### Single run examples

```bash
python scripts/run_benchmark.py --preset llama3-8b --config fp16 --hardware "Mac M3"
python scripts/run_benchmark.py --preset llama3-8b --weight-bits 4 --hardware "Mac M3"
python scripts/run_benchmark.py --preset llama3-8b --config w4+kv_cache+prefill --hardware "Mac M3"
```

## Model repos

| Preset | fp16 | 8-bit | 4-bit | 2-bit |
|--------|------|-------|-------|-------|
| llama3-8b | Meta-Llama-3.1-8B-bf16 | …-8bit | …-4bit | Llama-3-8B-262k-2bit |
| mistral-7b | Mistral-7B-v0.3-bf16 (gated) | …-8bit | …-4bit | — (add in `models.json`) |
| qwen-35b | Qwen2.5-32B-bf16 | …-8bit | …-4bit | — (add in `models.json`) |

Override any path in `models.json`:

```json
{
  "overrides": {
    "mistral-7b": { "2": "mlx-community/your-mistral-2bit-repo" }
  }
}
```

## Results

```text
results/Mac_M3/llama3-8b/fp16.json
results/Mac_M3/llama3-8b/w4+kv_cache.json
results/sweep_Mac_M3_<timestamp>.json
```

## Troubleshooting your sweep log

| What happened | Why | What to do |
|---------------|-----|------------|
| `401` on `Meta-Llama-3-8B-Instruct-bf16` | Old gated Llama 3.0 repo | Pull latest code (uses **Llama 3.1** public bf16) |
| `401` on Mistral `*-bf16` | Gated model | `huggingface-cli login` + accept license |
| `Insufficient Memory` on Qwen 32B | ~35GB model on 24GB M3 | Skip qwen on M3; use M5 Max with `--include-qwen` |
| Sweep aborted at run 17 | OOM killed whole process | Updated runner isolates each config in a subprocess |

Re-run failed fp16 / w8 configs after login:

```bash
./scripts/retry_failed.sh "Mac M3"
```

## Auto-push after Cursor agent edits

This repo includes a [Cursor hook](https://cursor.com/docs/agent/hooks) that runs when an agent turn finishes (`stop` event). It commits any workspace changes and pushes to `origin` on the current branch.

| Control | How |
|---------|-----|
| Disable for a session | `export CURSOR_AUTO_PUSH=0` before using the agent |
| View push log | `.cursor/hooks/logs/push.log` |
| Reload hooks | Save `.cursor/hooks.json` or restart Cursor |

**Requirements:** Git credentials for `https://github.com/Chirumamilla1522/LLM-Inference.git` (SSH remote or cached HTTPS token). Sensitive paths (`.env`, keys) are never committed.

## Requirements

- macOS, Apple Silicon, Python 3.10+
- Hugging Face login for gated fp16 models (Mistral)
- 24GB M3: llama + mistral only; 64GB+ for full Qwen 32B sweep
