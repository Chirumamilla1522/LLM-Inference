# Medium publishing package (long-form)

Data-heavy Medium drafts for **Local LLMs on Apple Silicon**.

## Cover thumbnails (Medium story images)

All 16:9 covers live in [`images/thumbnails/`](images/thumbnails/):

| Article | Thumbnail |
|---------|-----------|
| 00 Introduction | `thumb_00_introduction.png` |
| 01 Weight quantization | `thumb_01_weight_quantization.png` |
| 02 KV cache | `thumb_02_kv_cache.png` |
| 03 Prefill / TTFT | `thumb_03_prefill_ttft.png` |
| 04 Model ladder | `thumb_04_model_ladder.png` |
| 05 Full stack | `thumb_05_full_stack.png` |
| 06 Speculative | `thumb_06_speculative.png` |
| 07 Context / RAG | `thumb_07_rag_context.png` |

On Medium: set each file as the **story cover / preview image** when publishing.

## Paper-idea redraws (preferred over copying paper figures)

Original diagrams in [`images/papers/`](images/papers/) — **not** copied from PDFs. Each image credits the paper whose *idea* it teaches:

| File | Inspired by |
|------|-------------|
| `vaswani_attention_redraw.png` | Vaswani et al. 2017 |
| `williams_roofline_redraw.png` | Williams et al. 2009 |
| `jacob_affine_quant_redraw.png` | Jacob et al. 2018 |
| `frantar_gptq_redraw.png` | Frantar et al. 2022 (GPTQ) |
| `lin_awq_redraw.png` | Lin et al. 2023 (AWQ) |
| `dao_flashattention_redraw.png` | Dao et al. 2022/23 |
| `milakov_online_softmax_redraw.png` | Milakov & Gimelshein 2018 |
| `leviathan_speculative_redraw.png` | Leviathan / Chen 2023 |
| `cai_medusa_redraw.png` | Cai et al. 2024 (Medusa) |
| `ainslie_gqa_redraw.png` | Ainslie et al. 2023 |
| `kwon_paged_attention_redraw.png` | Kwon et al. 2023 |
| `pope_kv_scaling_redraw.png` | Pope et al. 2022 |

Regenerate: `python scripts/plot_paper_redraws.py`

## Stats (current)

| Article | ~Words | Figures |
|---------|-------:|--------:|
| 00 Introduction | ~3,500+ | 15 |
| 01 Weight quantization | ~3,300 | 14 |
| 02 KV cache | ~3,200 | 8+ |
| 03 Prefill / TTFT | ~3,000 | 8+ |
| 04 Model ladder | ~3,000 | 6 |
| 05 Full stack | ~3,100 | 7 |
| 06 Speculative | ~2,800 | 6 |
| 07 Context / RAG | ~2,600 | 9 |

**Images:** 50+ PNGs under `images/` (result plots) and `images/workflows/` (paper-style diagrams).

## Regenerate all figures

```bash
./scripts/regenerate_medium_images.sh "Mac M3"
# or:
python scripts/plot_medium_diagrams.py
python scripts/plot_medium_charts.py --hardware "Mac M3"
python scripts/plot_medium_deep.py
```

| Script | What it makes |
|--------|----------------|
| `plot_medium_diagrams.py` | How-it-works workflows (quant, KV, FlashAttention, speculative, …) |
| `plot_medium_charts.py` | Per-article result bars/scatters |
| `plot_medium_deep.py` | Heatmaps, M3 vs M5, family panels, config matrices, workloads |

## Article structure (every post)

1. Hook  
2. Why it matters  
3. How it works (workflow figures + math + papers)  
4. Deep results (tables + many plots)  
5. M3 vs M5 Max where data exists  
6. Recipes / decision guide  
7. Fun facts  
8. Limitations  
9. Reproduce commands  
10. Long references + series nav  

## Schedule

See [SCHEDULE.md](SCHEDULE.md) — Mon/Wed/Fri for 2 weeks (+ bonus).

## Publish on Medium

1. Copy markdown body (skip YAML header)  
2. Upload each referenced PNG from `images/` and `images/workflows/`  
3. Add tags from the footer  
4. Link as a Series  

> Numbers from [LLM-Inference](https://github.com/Chirumamilla1522/LLM-Inference) on MLX.
