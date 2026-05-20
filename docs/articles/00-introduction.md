# Article 0: Introduction — local LLMs on Apple Silicon

**Covers:** M3 vs M5 Max, unified memory, metrics, repo workflow.

**References:** [19], [20], [21], [24] — [REFERENCES.md](../REFERENCES.md).

---

## Figure — Unified memory on Apple Silicon

```mermaid
flowchart TB
  subgraph chip["Apple Silicon SoC"]
    CPU[CPU cores]
    GPU[GPU / Neural Engine]
    CPU <-->|same pool| UM[(Unified memory)]
    GPU <-->|same pool| UM
  end
  W["Model weights"] --> UM
  KV["KV cache"] --> UM
  OS["macOS + apps"] --> UM
```

Unlike discrete GPU PCs, there is no separate VRAM bar—**total RAM is shared** [20].

---

## Figure — What this repo measures

```mermaid
mindmap
  root((LLM-Inference))
    TTFT
      prefill
      prompt length
    Throughput
      decode
      weight bandwidth
    Memory
      weights
      KV cache
    Optimizations
      quant
      kv_bits
      prefill chunks
```

---

## Run

```bash
./scripts/run_article.sh 0 "Mac M3"
```

See [ARTICLE_SERIES.md](../ARTICLE_SERIES.md) § Article 0.
