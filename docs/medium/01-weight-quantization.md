---
title: "4-Bit Weights Changed Everything on My M3 Mac"
subtitle: "Weight quantization from fp16 to w2 — memory, speed, and when quality breaks"
tags: Quantization, LLM, Apple Silicon, MLX, GPTQ, Performance
series: 2 of 7
read_time: 10 min
---

# 4-Bit Weights Changed Everything on My M3 Mac

*Part 2 of 7 — Local LLMs on Apple Silicon*

An 8-billion-parameter model stored in FP16 needs roughly **16 GB** just for weights. On a 24 GB MacBook, that leaves almost nothing for the OS, your editor, and the KV cache that grows with every token you generate.

**Weight quantization** is the first optimization everyone should enable. It stores weights in fewer bits per parameter — typically 8, 4, or even 2 — with modest quality loss if done well.

We benchmarked **14 model families** at fp16, w8, w4, and w2 on Mac M3. Here is what the data says.

---

## The math in one paragraph

Each weight \(w\) is mapped to an integer code \(q\) with a per-group scale \(s\) and zero-point \(z\):

\[
q = \mathrm{clip}\left(\mathrm{round}\left(\frac{w}{s} + z\right),\ 0,\ 2^b - 1\right), \quad \hat{w} = s \cdot (q - z)
\]

This **affine quantization** scheme dates to Jacob et al. (2018) and was adapted for LLMs by GPTQ and AWQ. MLX loads **pre-quantized checkpoints** from Hugging Face — no runtime quant in our tests.

> **Fun fact:** GPTQ (2022) quantizes weights one column at a time while compensating for error accumulation — it was originally designed to shrink 175B-class models that literally could not fit on a single GPU at fp16.

---

## Llama 3.1 8B: the headline numbers

| Config | Peak memory | TTFT | Decode tok/s | vs fp16 speed |
|--------|-------------|------|--------------|---------------|
| **fp16** | 16.33 GB | 2,637 ms | 5.8 | 1.0× |
| **w8** | 8.96 GB | 2,775 ms | 11.3 | 1.9× |
| **w4** | 5.06 GB | 2,738 ms | **20.5** | **3.5×** |
| **w2** | 3.11 GB | 2,826 ms | 35.8 | 6.2× |

![Weight quantization — Llama 3.1 8B](images/01_weight_quant_llama3-8b.png)

*Figure 1: Memory halves roughly with each halving of bit width; decode throughput rises because less data moves per matmul.*

**Takeaway:** w4 is the sweet spot for 8B models on 24 GB Macs — **3.5× faster decode**, **~5 GB peak**, and widely available checkpoints. w2 is faster still but can degrade reasoning; we skipped w2 on several models where MLX repos are unavailable or unstable.

---

## It is not just Llama — family comparison at w4

| Model | Params | w4 memory | w4 tok/s |
|-------|--------|-----------|----------|
| Qwen 0.5B | 0.5B | 0.64 GB | **215.2** |
| Llama 3.2 1B | 1B | 1.24 GB | 102.9 |
| Phi-3 Mini | 3.8B | 2.93 GB | 37.1 |
| Mistral 7B | 7B | 4.62 GB | 21.7 |
| Llama 3.1 8B | 8B | 5.06 GB | 20.5 |
| Gemma 9B | 9B | 5.88 GB | 15.9 |

Smaller models are not just lighter — they are **dramatically faster** because decode is often memory-bandwidth bound. A 0.5B model at w4 can exceed **200 tok/s** on the same chip where 8B fp16 struggles to hit 6.

> **Fun fact:** Phi-3 Mini (3.8B) punches above its weight class — at w4 it hits 37 tok/s while using under 3 GB. Microsoft trained it on “textbook-quality” synthetic data, squeezing benchmark scores out of a smaller footprint.

---

## Why w4 is faster (not just smaller)

Quantization helps in two ways:

1. **Memory:** Fewer bytes to store → model fits → no swap thrashing  
2. **Bandwidth:** Each decode step reads fewer bytes of weights → higher effective tok/s  

On Apple Silicon, #2 is often the bigger win even when the model already fits. You are moving 4× less weight data per token at w4 vs fp16. That aligns with the Roofline model: LLM decode is frequently **bandwidth-bound**, not compute-bound.

---

## Quality: what the papers say

| Method | Paper | Key idea |
|--------|-------|----------|
| GPTQ | Frantar et al. 2022 | Post-training, Hessian-aware column quant |
| AWQ | Lin et al. 2023 | Protect “salient” weights using activation stats |
| LLM.int8() | Dettmers et al. 2022 | Mixed precision for outlier dimensions |

Community quants (mlx-community, bartowski GGUFs) inherit these ideas. For coding and chat, **w4 is usually indistinguishable from fp16** in blind tests; w2 is where you start noticing hallucination on math.

---

## Practical recommendation

| Your RAM | Start here | Avoid |
|----------|------------|-------|
| 16 GB | 3B–7B at w4 | fp16 8B |
| 24 GB | 8B at w4 | fp16 8B as daily driver |
| 32 GB+ | 8B fp16 for quality checks | w2 for production |
| 64 GB+ | 32B–70B at w4/w8 | — |

```bash
# Reproduce Article 1 sweep
./scripts/run_article.sh 1 "Mac M3"
```

---

## References

1. Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* (2018) — [arXiv:1712.05877](https://arxiv.org/abs/1712.05877)  
2. Frantar et al., *GPTQ* (2022) — [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)  
3. Lin et al., *AWQ* (2023) — [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)  
4. Dettmers et al., *LLM.int8()* (2022) — [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)  
5. mlx-community models — [huggingface.co/mlx-community](https://huggingface.co/mlx-community)  

---

**← Previous:** [Part 1: Introduction](00-introduction.md)  
**Next →** [Part 3: KV Cache Quantization](02-kv-cache-quantization.md)

**Tags:** `Quantization` `LLM` `Apple Silicon` `MLX` `GPTQ` `Performance`
