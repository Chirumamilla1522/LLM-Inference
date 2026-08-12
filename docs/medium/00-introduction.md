---
title: "Running 8B LLMs on a MacBook: What Actually Matters"
subtitle: "Unified memory, the metrics that matter, and reproducible benchmarks on Apple Silicon"
tags: Machine Learning, Apple, LLM, MLX, Local AI, Apple Silicon
series: 1 of 7
read_time: 8 min
---

# Running 8B LLMs on a MacBook: What Actually Matters

*Part 1 of 7 — Local LLMs on Apple Silicon*

You can run Meta’s Llama 3.1 8B on a stock MacBook Pro today. No cloud bill. No NVIDIA rig. But the first time you load it in full precision, two things happen: **Activity Monitor turns red**, and generation crawls at roughly **5 tokens per second**.

That gap — between “it runs” and “it runs well” — is what this series is about. I built an open benchmark harness, ran it on a **Mac M3 (24 GB)** and a **Mac M5 Max**, and measured every optimization I could enable in [MLX](https://github.com/ml-explore/mlx).

---

## Why Apple Silicon is different

On a gaming PC, GPU VRAM is a separate pool from system RAM. On Apple Silicon, **CPU and GPU share one unified memory pool**. There is no PCIe copy between “host” and “device.” That simplifies loading models — but it also means **your browser tabs compete with your 8B-parameter weights**.

> **Fun fact:** Apple’s M1 (2020) was the first Mac chip where the GPU could access the *same* physical DRAM as the CPU without copying. LLM inference on Mac only became practical at scale after unified memory crossed ~16–24 GB in consumer laptops.

This architecture is why quantization matters so much on Mac: you are not fighting a VRAM ceiling, you are fighting **total system RAM** plus **memory bandwidth** during decode.

---

## The three numbers that matter

Before diving into bit widths and cache tricks, fix these metrics in your head:

| Metric | What it measures | Why you care |
|--------|------------------|--------------|
| **Peak memory (GB)** | Weights + KV cache + runtime | Will the model load without swap? |
| **TTFT (ms)** | Time to first token | Chat feels “snappy” or “frozen” |
| **Decode throughput (tok/s)** | Tokens generated per second | How fast the answer streams |

These map directly to user experience:

- **TTFT** = how long you stare at a blinking cursor after hitting Enter  
- **tok/s** = how fast the answer appears once generation starts  
- **Memory** = whether you can run the model *at all* alongside your IDE

Our benchmarks use fixed prompts (512 tokens) and generation length (128 tokens) so comparisons are apples-to-apples across configs and hardware.

---

## Baseline: Llama 3.1 8B in FP16 on Mac M3

| Config | Peak memory | TTFT | Decode tok/s |
|--------|-------------|------|--------------|
| **fp16** | 16.33 GB | 2,651 ms | **5.3** |

At fp16, an 8B model consumes most of a 24 GB machine before the KV cache grows. Decode is bandwidth-bound — you are shuffling ~16 GB of weights every token step. The [Roofline model](https://people.csail.mit.edu/stajich/publications/cacm09.pdf) explains this well: when arithmetic intensity is low, **memory bandwidth caps throughput**, not FLOPS.

![Llama 3.1 8B — fp16 vs w4 on M3 and M5 Max](images/00_intro_hardware_compare.png)

*Figure 1: Same model, different precision — memory drops ~3× and throughput rises ~3.5× at 4-bit weights (M3). M5 Max numbers included when available.*

On **Mac M5 Max**, the same fp16 run fits with headroom, and w4 decode is faster still — but the *shape* of the tradeoff is identical: quantize weights first, then tune runtime.

---

## What this series covers

Over the next six posts we walk through each optimization layer:

1. **Weight quantization** — fp16 → w8 → w4 → w2  
2. **KV cache quantization** — compressing the growing attention cache  
3. **Prefill tuning** — shrinking time-to-first-token  
4. **Model size ladder** — what fits from 0.5B to 70B  
5. **The full stack** — combining all levers  
6. **Speculative decoding** — draft models for free speed  

Every number comes from reproducible JSON in the [LLM-Inference](https://github.com/Chirumamilla1522/LLM-Inference) repo. Clone it, run `./scripts/run_article.sh 0 "Mac M3"`, and you will get the same schema.

---

## Methodology in 30 seconds

- **Runtime:** [MLX](https://github.com/ml-explore/mlx) + [mlx-lm](https://github.com/ml-explore/mlx-lm) on macOS  
- **Models:** [mlx-community](https://huggingface.co/mlx-community) checkpoints on Hugging Face  
- **Trials:** 1 warmup + 3 measured runs; we report **medians**  
- **Isolation:** Each config runs in a subprocess — a Metal OOM does not kill the sweep  

---

## References

1. Vaswani et al., *Attention Is All You Need* (2017) — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)  
2. Dubey et al., *The Llama 3 Herd of Models* (2024) — [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)  
3. Williams et al., *Roofline: An Insightful Visual Performance Model* (2009) — [CACM PDF](https://people.csail.mit.edu/stajich/publications/cacm09.pdf)  
4. Apple, *MLX* — [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)  
5. Apple, *mlx-lm* — [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)  
6. LLM-Inference benchmark repo — [github.com/Chirumamilla1522/LLM-Inference](https://github.com/Chirumamilla1522/LLM-Inference)  

---

**Next →** [Part 2: 4-Bit Weights Changed Everything on My M3 Mac](01-weight-quantization.md)

**Tags:** `Machine Learning` `Apple` `LLM` `MLX` `Local AI` `Apple Silicon`
