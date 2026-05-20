# Article 6: Speculative decoding

**Covers:** Draft + target models, acceptance rate math, MLX programming path.

**Run:**

```bash
./scripts/run_article.sh 6 "Mac M3"
```

**Deep dive:** [speculative-decoding.md](../optimizations/speculative-decoding.md)

---

## Math summary

- Baseline decode: \(\text{tok/s} = 1 / \tau_{\text{target}}\)  
- Speculative: propose \(k\) tokens with draft, verify with target  
- Logged metric: \(\alpha\) = `draft_accept_rate`  
- Speedup grows with \(\alpha\) and \(k\), shrinks if two models OOM

---

## Programming summary

- `--speculative` in `run_benchmark.py`  
- `draft_model` passed to `stream_generate`  
- `DRAFT_PRESET_BY_TARGET` in `optimizations.py`  
- Compare `*_w4_baseline` vs `*_w4_speculative` JSON under `article_06_speculative-decoding/`
