# notes.md

# Deploying Open-Source LLMs Locally: Mac M3 vs. M5 Max (article workspace)

Benchmark repo + draft copy. **12 articles** — one per optimization (or one set). Index: **[docs/ARTICLES_INDEX.md](docs/ARTICLES_INDEX.md)** · run: **`./scripts/run_article.sh <id> "Mac M3"`**

| # | Article | Status |
|---|---------|--------|
| 0 | Introduction (M3 vs M5, methodology) | benchmarked |
| 1 | Weight quantization | benchmarked |
| 2 | KV cache quantization | benchmarked |
| 3 | Prefill & Flash Attention | benchmarked |
| 4 | Model size ladder | benchmarked — **run on M5**: `make m5` |
| 5 | Full stack (capstone) | benchmarked — **run on M5**: `make m5` |
| 6 | Speculative decoding | benchmarked |
| 7 | Context, length, prompt cache + workloads | benchmarked |
| 8 | Production serving | concept manifest |
| 9 | Parallelism | concept manifest |
| 10 | MLX vs llama.cpp | benchmarked (`compare_runtimes.py`) |
| 11 | Tradeoffs | concept manifest |

Technique reference: [INFERENCE_OPTIMIZATIONS_CATALOG.md](docs/INFERENCE_OPTIMIZATIONS_CATALOG.md)

**Tables from JSON (not hand-edited):**

```bash
make report HW="Mac M3"
make tables ARTICLE=5 HW="Mac M3"
```

---

## Capstone draft (Article 5)

See generated tables under `docs/articles/_generated/` after `./scripts/run_article.sh 5` on M3 and M5.

**M5 data:** run on M5 Max hardware:

```bash
./scripts/run_m5_ladder.sh "Mac M5 Max"
```

---

## Mac M3 — run everything (this machine)

```bash
./scripts/hf_login.sh
./scripts/run_m3_all.sh
```

Details: [docs/M3_RUNBOOK.md](docs/M3_RUNBOOK.md)

## Repo hygiene

```bash
python scripts/validate_results.py --hardware "Mac M3"
./scripts/retry_failed.sh "Mac M3"
make test
```
