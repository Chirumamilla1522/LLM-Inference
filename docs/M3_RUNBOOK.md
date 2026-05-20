# Mac M3 — complete runbook

One machine label everywhere: **`"Mac M3"`** → results under `results/Mac_M3/`.

## What you already have

| Location | What |
|----------|------|
| `results/Mac_M3/llama3-8b/*.json` | 16-config matrix (partial) |
| `results/Mac_M3/mistral-7b/*.json` | 16-config matrix (partial) |
| 8× `fp16` files | `error` — need HF login + re-run |

**Not done yet:** article folders (`article_01_*`, …), Article 10 compare, most article-specific sweeps.

---

## One command (recommended)

```bash
./scripts/setup_env.sh
source .venv/bin/activate
./scripts/hf_login.sh
python scripts/run_benchmark.py --hf-check

# ~2–8 hours depending on downloads (14 presets × weight sweep + article runs)
./scripts/run_m3_all.sh
```

Resume after interrupt:

```bash
./scripts/run_m3_all.sh standard --from-checkpoint
```

---

## What `standard` mode runs

| Article | MLX runs | Output folder |
|---------|----------|---------------|
| 0 | 1 demo | `article_00_introduction/` |
| 1 | ~56 (weights only, all presets) | `article_01_weight-quantization/` |
| 2 | 7 | `article_02_kv-cache-quantization/` |
| 3 | 4 | `article_03_prefill-ttft/` |
| 4 | 14 (all presets @ w4) | `article_04_model-size-ladder/` |
| 5 | **4 hero** (no full 224 sweep) | `article_05_full-stack/` |
| 6 | 6 (3 models × baseline + speculative) | `article_06_speculative-decoding/` |
| 7 | 14 (context + workloads) | `article_07_context-and-cache/` |
| 10 | 3 MLX + 3 llama.cpp compares | `article_10_runtimes/` |
| 8, 9, 11 | manifest only | `article_*_*/manifest.json` |

**`full` mode** adds Article 5’s full 16-config × all-models sweep (~228 runs). Use only if you want every cell on M3 before M5.

---

## After the run

```bash
python scripts/validate_results.py --hardware "Mac M3"
make report HW="Mac M3"
make plot HW="Mac M3"
git add results/Mac_M3 docs/articles/_generated
```

---

## M5 (later)

M3 runbook does **not** replace M5. On the Max machine:

```bash
./scripts/run_m5_ladder.sh "Mac M5 Max"
python scripts/plot_results.py --hardware "Mac M5 Max" \
  --compare-hardware "Mac M3" --config w4+kv_cache+prefill
```

---

## Modes

| Command | Use when |
|---------|----------|
| `./scripts/run_m3_all.sh` | Normal M3 dataset for the series |
| `./scripts/run_m3_all.sh full` | Maximum M3 coverage (long) |
| `./scripts/run_m3_all.sh quick` | Smoke test (~30 min) |
| `./scripts/retry_failed.sh "Mac M3"` | Only fix old fp16 auth errors |
