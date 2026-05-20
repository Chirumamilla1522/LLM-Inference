# Article series

**12 articles** — one per optimization or set. Index: [ARTICLES_INDEX.md](ARTICLES_INDEX.md).

## Run benchmarks (code)

```bash
source .venv/bin/activate

# List articles
python scripts/run_article.py --list

# One article (results → results/<hardware>/article_XX_<slug>/)
./scripts/run_article.sh 1 "Mac M3"          # weight quantization sweep
./scripts/run_article.sh 2 "Mac M3"          # KV cache pairs
./scripts/run_article.sh 6 "Mac M3"          # speculative decoding
./scripts/run_article.sh 7 "Mac M3"          # context / generation / prefix cache
./scripts/run_article.sh 8 "Mac M3"          # concept manifest only

# All benchmarked articles (0–7)
./scripts/run_article.sh all "Mac M3"

# Dry-run plan
python scripts/run_article.py --article 5 --dry-run --hardware "Mac M3"

# Markdown tables from JSON
python scripts/generate_article_tables.py --hardware "Mac M3" --article 2
```

### Options

| Flag | Purpose |
|------|---------|
| `--dry-run` | Print planned runs |
| `--runs-only` | Skip sweep, only explicit runs |
| `--sweep-only` | Skip explicit runs |
| `--include-large` | 12B+ presets |
| `-n 1` | Quick smoke test |

---

## Article → what runs

| # | Slug | Command | Output |
|---|------|---------|--------|
| 0 | introduction | 1× `fp16` demo | `article_00_introduction/` |
| 1 | weight-quantization | `--weights-only --all-models` sweep | per preset: `fp16,w8,w4,w2` |
| 2 | kv-cache-quantization | `w4` vs `w4+kv_cache` × 3 models + long `-g` | labeled JSON |
| 3 | prefill-ttft | `w4` vs `w4+prefill`, `-p` 256/1024 | labeled JSON |
| 4 | model-size-ladder | sweep `w4` only, all models | one config per preset |
| 5 | full-stack | full 16-config sweep + hero runs | full matrix |
| 6 | speculative-decoding | `w4` vs `w4+speculative` × 3 models | draft metrics in JSON |
| 7 | context-and-cache | `-p`/`-g` sweeps + `--prefix-cache` | labeled JSON |
| 8–11 | concept | `manifest.json` only | no MLX |

---

## Low-level CLI (still supported)

```bash
python scripts/run_benchmark.py --preset llama3-8b --config w4+kv_cache --hardware "Mac M3"
python scripts/run_benchmark.py --preset llama3-8b --config w4 --speculative --hardware "Mac M3"
python scripts/run_benchmark.py --preset llama3-8b --config w4 --prefix-cache --hardware "Mac M3"
```
