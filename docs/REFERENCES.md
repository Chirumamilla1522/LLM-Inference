# References

Curated bibliography for the article series and optimization guides. Links are stable public sources (papers, docs, repos).

---

## Foundational LLMs & attention

| ID | Reference | Why we cite it |
|----|-----------|----------------|
| [1] | Vaswani, A., et al. (2017). *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | Transformer, Q/K/V, autoregressive decoding |
| [2] | Touvron, H., et al. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models.* [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) | Llama family architecture context |
| [3] | Dubey, A., et al. (2024). *The Llama 3 Herd of Models.* [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) | Llama 3 / 3.1 presets in benchmarks |

---

## Attention efficiency (prefill / Flash)

| ID | Reference | Why we cite it |
|----|-----------|----------------|
| [4] | Dao, T., et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) | Tiled attention, \(O(n)\) memory |
| [5] | Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) | Improved GPU utilization |
| [6] | Milakov, M., & Gimelshein, N. (2018). *Online normalizer calculation for softmax.* [arXiv:1805.02867](https://arxiv.org/abs/1805.02867) | Stable streaming softmax (Flash building block) |

---

## Weight quantization

| ID | Reference | Why we cite it |
|----|-----------|----------------|
| [7] | Jacob, B., et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* [CVPR](https://arxiv.org/abs/1712.05877) | Affine quant \(q = \mathrm{round}(x/s + z)\) |
| [8] | Frantar, E., et al. (2022). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) | Post-training weight quant |
| [9] | Lin, J., et al. (2023). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) | Activation-aware 4-bit weights |
| [10] | Dettmers, M., et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) | Mixed precision / outliers |

---

## KV cache & memory-efficient attention

| ID | Reference | Why we cite it |
|----|-----------|----------------|
| [11] | Ainslie, J., et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) | Fewer KV heads → smaller \(H_{\text{kv}}\) |
| [12] | Kwon, W., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention.* [SOSP / arXiv:2309.06180](https://arxiv.org/abs/2309.06180) | Paged KV (Article 8 concept) |
| [13] | Pope, R., et al. (2022). *Efficiently Scaling Transformer Inference.* [MLSys](https://arxiv.org/abs/2211.05102) | KV cache analysis, batching |

---

## Speculative & parallel decoding

| ID | Reference | Why we cite it |
|----|-----------|----------------|
| [14] | Leviathan, Y., et al. (2023). *Fast Inference from Transformers via Speculative Decoding.* [arXiv:2211.08920](https://arxiv.org/abs/2211.08920) | Draft + verify decoding |
| [15] | Chen, C., et al. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling.* [arXiv:2302.01318](https://arxiv.org/abs/2302.01318) | Speculative sampling theory |
| [16] | Cai, T., et al. (2024). *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) | Multi-token prediction variant |

---

## Serving & batching (concept articles)

| ID | Reference | Why we cite it |
|----|-----------|----------------|
| [17] | Yu, G., et al. (2022). *Orca: A Distributed Serving System for Transformer-Based Generative Models.* [OSDI](https://www.usenix.org/conference/osdi22/presentation/yu) | Continuous batching ideas |
| [18] | Agrawal, A., et al. (2024). *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* [arXiv:2403.02310](https://arxiv.org/abs/2403.02310) | Disaggregated prefill/decode |

---

## Hardware & performance models

| ID | Reference | Why we cite it |
|----|-----------|----------------|
| [19] | Williams, S., et al. (2009). *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* [CACM](https://people.csail.mit.edu/stajich/publications/cacm09.pdf) | Bandwidth vs compute bounds |
| [20] | Apple Inc. *Apple Silicon* unified memory architecture. [Apple Platform Security / chip guides](https://www.apple.com/newsroom/2020/11/one-more-thing-a-homecoming-for-apple-silicon/) | M-series unified memory context |

---

## MLX & this repository stack

| ID | Reference | Why we cite it |
|----|-----------|----------------|
| [21] | Apple. *MLX* — array framework for Apple Silicon. [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx) | Runtime for all benchmarks |
| [22] | Apple. *mlx-lm* — LLM utilities for MLX. [github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) | `load`, `stream_generate`, cache APIs |
| [23] | Hugging Face. *mlx-community* model hub. [huggingface.co/mlx-community](https://huggingface.co/mlx-community) | Weight checkpoints per bit width |
| [24] | This repository. *LLM-Inference* benchmark scripts. [scripts/run_benchmark.py](../scripts/run_benchmark.py) | Reproducible measurement harness |

---

## How to cite in articles

**Inline (example):**

> Prefill attention is implemented with IO-aware tiling in the spirit of FlashAttention [4, 5], exposed in MLX via Metal kernels [21] rather than a user-facing flag.

**Figure caption (example):**

> **Figure 3:** KV cache memory vs sequence length \(T\) (Eq. in [KV cache guide](optimizations/kv-cache-quantization.md)); GQA reduces \(H_{\text{kv}}\) [11].

---

## See also

- [Math vs programming](optimizations/math-and-implementation.md) — equations tied to these sources  
- [Optimizations index](optimizations/README.md)
