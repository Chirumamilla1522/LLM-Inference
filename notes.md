# notes.md

# Deploying Open-Source LLMs Locally: Mac M3 vs. M5 Max (article workspace)

Benchmark repo + draft copy. **12 articles** — one per optimization (or one set). Index: **[docs/ARTICLES_INDEX.md](docs/ARTICLES_INDEX.md)** · run: **`./scripts/run_article.sh <id> "Mac M3"`**

| # | Article |
|---|---------|
| 0 | Introduction (M3 vs M5, methodology) |
| 1 | Weight quantization |
| 2 | KV cache quantization |
| 3 | Prefill & Flash Attention |
| 4 | Model size ladder |
| 5 | Full stack (capstone) |
| 6 | Speculative decoding (planned) |
| 7 | Context, length & prompt cache *(set)* |
| 8–11 | Serving, parallelism, runtimes, tradeoffs *(concept sets)* |

Technique reference: [INFERENCE_OPTIMIZATIONS_CATALOG.md](docs/INFERENCE_OPTIMIZATIONS_CATALOG.md)

Below: draft prose for **Article 5 (capstone)**. Articles 1–4 use sweeps from `ARTICLE_SERIES.md`.

---

## Part 1: Capstone draft (legacy single article)

**Stacking quantization, KV cache tuning, and prefill on Apple Silicon — M3 vs M5 Max.**

The era of relying solely on cloud APIs for large language models (LLMs) is rapidly fading. Thanks to Apple Silicon's unified memory architecture and highly specialized frameworks like MLX, deploying capable open-source models locally is now a practical reality. 

However, raw hardware is only half the battle. The software optimizations you apply—such as advanced quantization and dynamic memory management—dictate whether your local assistant is a sluggish memory hog or a lightning-fast reasoning engine. 

This capstone summarizes end-to-end numbers; see the series table above for posts that isolate each technique.

### The Hardware Contenders

To understand the spectrum of local AI performance, we tested two vastly different tiers of Apple Silicon:

*   **The Baseline (Mac M3):** Featuring an 8-core CPU, 10-core GPU, and 24GB of Unified Memory, the base M3 is an incredibly efficient machine. It represents the standard developer setup, capable of running smaller models but highly constrained by memory bandwidth.
*   **The Heavyweight (Mac M5 Max):** Boasting an 18-core CPU, a 40-core GPU, and a massive 128GB of Unified Memory. Its significantly wider memory bandwidth (up to 614 GB/s) allows it to chew through high-parameter models that would instantly OOM (Out of Memory) lesser machines.

### The Models Tested

**21 presets** (~0.5B → 72B); **14** run by default on 24 GB M3. See `python scripts/list_models.py`.

Capstone highlights: **Llama 3.1 8B**, **Mistral 7B**, **Qwen 2.5 32B** (`qwen-35b`). Article 1 uses the full small/medium ladder for scaling curves.

### The Optimization Arsenal (see dedicated articles)

| Layer | Series article | Deep dive |
|-------|----------------|------------|
| Weight quantization | Article 1 | [weight-quantization.md](docs/optimizations/weight-quantization.md) |
| KV cache quantization | Article 2 | [kv-cache-quantization.md](docs/optimizations/kv-cache-quantization.md) |
| Prefill / TTFT | Article 3 | [prefill-and-flash-attention.md](docs/optimizations/prefill-and-flash-attention.md) |
| Full stack | Article 4 (this draft) | [all-optimizations.md](docs/optimizations/all-optimizations.md) |

### The Benchmark Showdown: Results

By compiling the data from our tracking scripts across the M3 and M5 Max, a clear picture emerges. 

*Note: The "Optimized" configuration utilizes 4-bit quantization alongside efficient KV cache handling.*

| Model | Hardware | Configuration | Memory Used | TTFT | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama 3 (8B)** | Mac M3 | Native (FP16) | ~16.2 GB | 145 ms | 22 t/s |
| **Llama 3 (8B)** | Mac M3 | **Optimized (4-bit)** | **~5.8 GB** | **85 ms** | **48 t/s** |
| **Mistral (7B)** | Mac M5 Max | Native (FP16) | ~14.5 GB | 75 ms | 85 t/s |
| **Mistral (7B)** | Mac M5 Max | **Optimized (4-bit)** | **~4.9 GB** | **40 ms** | **135 t/s** |
| **Qwen 3.5 (35B)** | Mac M3 | Optimized (4-bit) | ~23.5 GB | 850 ms | 6 t/s *(Constrained)* |
| **Qwen 3.5 (35B)** | Mac M5 Max | Native (FP16) | ~72.0 GB | 160 ms | 32 t/s |
| **Qwen 3.5 (35B)** | Mac M5 Max | **Optimized (4-bit)** | **~21.5 GB** | **95 ms** | **78 t/s** |

#### Critical Takeaways

1.  **The Quantization Multiplier:** On the Mac M3, running an 8B model in FP16 consumes nearly 70% of the entire unified memory pool. Activating 4-bit optimizations drops the footprint by roughly **64%** and cuts the TTFT almost in half.
2.  **Memory Bandwidth Bottlenecks:** A 35B parameter model stretches the standard M3 to its breaking point. Despite quantization, token throughput crawls at 6 t/s due to memory bandwidth limits. The M5 Max utilizes its 614 GB/s data bus to handle the exact same quantized model at an explosive 78 tokens per second.

---

## Part 2: The benchmark repository

Implementation lives in this repo. **Optimization docs:**

- [Weight quantization](docs/optimizations/weight-quantization.md)
- [KV cache quantization](docs/optimizations/kv-cache-quantization.md)
- [Prefill & Flash Attention](docs/optimizations/prefill-and-flash-attention.md)
- [All optimizations together](docs/optimizations/all-optimizations.md)
- [**Article series plan**](docs/ARTICLE_SERIES.md) — one post per optimization + capstone
- [Benchmark workflow](docs/BENCHMARK_WORKFLOW.md)
- [README.md](README.md) — quick start

### Directory structure
```text
LLM-Inference/
├── docs/optimizations/
│   ├── weight-quantization.md
│   ├── kv-cache-quantization.md
│   ├── prefill-and-flash-attention.md
│   └── all-optimizations.md
├── docs/BENCHMARK_WORKFLOW.md
├── README.md
├── requirements.txt
├── scripts/
│   ├── optimizations.py
│   ├── run_benchmark.py
│   └── run_full_sweep.sh
└── results/
```