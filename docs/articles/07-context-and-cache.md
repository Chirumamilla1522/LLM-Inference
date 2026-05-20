# Article 7: Context, generation length & prompt cache (set)

**One article** for:

1. **Prompt length** — TTFT & memory vs `-p` (subsection; was “A6”).
2. **Generation length** — KV growth vs `-g` (subsection; was “A7”).
3. **Prefix / prompt KV cache** — reuse static system prompt (when benchmark exists).

Example sweeps (subsections 1–2, today):

```bash
# Prompt length
for P in 256 512 1024 2048; do
  python scripts/run_benchmark.py --preset llama3-8b --config w4+prefill \
    --hardware "Mac M3" -p "$P" -g 128
done

# Generation length
for G in 64 256 512; do
  python scripts/run_benchmark.py --preset llama3-8b --config w4+kv_cache \
    --hardware "Mac M3" -p 512 -g "$G"
done
```

See [ARTICLE_SERIES.md](../ARTICLE_SERIES.md) § Article 7.
