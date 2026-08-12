---
title: "Running 8B LLMs on a MacBook: What Actually Matters"
subtitle: "Unified memory, the inference pipeline, and reproducible benchmarks on Apple Silicon"
tags: Machine Learning, Apple, LLM, MLX, Local AI, Apple Silicon
series: 1 of 7
read_time: 12 min
figures: 5
---

# Running 8B LLMs on a MacBook: What Actually Matters

*Part 1 of 7 — Local LLMs on Apple Silicon*

You can run Meta’s Llama 3.1 8B on a stock MacBook Pro today. No cloud bill. No NVIDIA rig. But the first time you load it in full precision, two things happen: **Activity Monitor turns red**, and generation crawls at roughly **5 tokens per second**.

That gap — between “it runs” and “it runs well” — is what this series is about. I built an open benchmark harness on [MLX](https://github.com/ml-explore/mlx), ran it on a **Mac M3 (24 GB)** and a **Mac M5 Max**, and measured every optimization I could enable.

---

## How Apple Silicon changes the game

On a gaming PC, GPU VRAM is a separate pool from system RAM. On Apple Silicon, **CPU and GPU share one unified memory pool**. There is no PCIe copy between “host” and “device.”

![Unified memory architecture](images/workflows/00_unified_memory.png)

*Figure 1 — Workflow: unified DRAM is the hard ceiling for weights + KV cache + OS + apps.*

That simplifies loading models — but it also means **your browser tabs compete with your 8B-parameter weights**. Quantization is not optional; it is how you leave room to breathe.

> **Fun fact:** Apple’s M1 (2020) was the first Mac chip where the GPU could access the *same* physical DRAM as the CPU without copying. Local LLM inference on Mac only became practical after unified memory crossed ~16–24 GB in consumer laptops.

---

## The inference pipeline (what we actually measure)

Every chat reply has two phases with different bottlenecks:

![Inference pipeline](images/workflows/00_inference_pipeline.png)

*Figure 2 — Workflow: load → prefill → first token (TTFT) → decode loop (tok/s).*

| Metric | Phase | User feels |
|--------|-------|------------|
| **Peak memory (GB)** | Load + KV growth | Will it fit without swap? |
| **TTFT (ms)** | Prefill | Cursor freeze after Enter |
| **Decode tok/s** | Autoregressive loop | How fast the answer streams |

This series optimizes each phase separately — then stacks them.

---

## Baseline: Llama 3.1 8B in FP16 on Mac M3

| Config | Peak memory | TTFT | Decode tok/s |
|--------|-------------|------|--------------|
| **fp16** | 16.33 GB | 2,651 ms | **5.3** |

At fp16, an 8B model consumes most of a 24 GB machine before the KV cache grows. Decode is bandwidth-bound — you shuffle ~16 GB of weights every token step. The [Roofline model](https://people.csail.mit.edu/stajich/publications/cacm09.pdf) explains why: when arithmetic intensity is low, **memory bandwidth caps throughput**, not FLOPS.

![Hardware compare M3 vs M5](images/00_intro_hardware_compare.png)

*Figure 3 — Results: same model, different precision — memory drops ~3× and throughput rises ~3.5× at 4-bit (M3).*

---

## Methodology (so you can reproduce)

- **Runtime:** [MLX](https://github.com/ml-explore/mlx) + [mlx-lm](https://github.com/ml-explore/mlx-lm)  
- **Models:** [mlx-community](https://huggingface.co/mlx-community) checkpoints  
- **Trials:** 1 warmup + 3 measured runs; we report **medians**  
- **Isolation:** each config in a subprocess (Metal OOM does not kill the sweep)  
- **Repo:** [LLM-Inference](https://github.com/Chirumamilla1522/LLM-Inference)

```bash
./scripts/run_article.sh 0 "Mac M3"
```

---

## What comes next

| Part | Topic | Main lever |
|------|-------|------------|
| 2 | Weight quantization | fp16 → w4 |
| 3 | KV cache quantization | Long-context memory |
| 4 | Prefill & TTFT | First-token latency |
| 5 | Model size ladder | 0.5B → 70B |
| 6 | Full stack | Combine everything |
| 7 | Speculative decoding | Draft models |

Every number ships as JSON you can plot yourself.

---

## References

1. Vaswani et al., *Attention Is All You Need* (2017) — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)  
2. Dubey et al., *The Llama 3 Herd of Models* (2024) — [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)  
3. Williams et al., *Roofline* (2009) — [CACM PDF](https://people.csail.mit.edu/stajich/publications/cacm09.pdf)  
4. Apple, *MLX* — [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)  
5. Apple, *mlx-lm* — [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)  
6. LLM-Inference — [github.com/Chirumamilla1522/LLM-Inference](https://github.com/Chirumamilla1522/LLM-Inference)  

---

**Next →** [Part 2: 4-Bit Weights Changed Everything](01-weight-quantization.md)

**Tags:** `Machine Learning` `Apple` `LLM` `MLX` `Local AI` `Apple Silicon`
