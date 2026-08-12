# Medium publishing package

## Publish files (finished Medium story format)

Use **`docs/medium/publish/*.medium.txt`**

These read like a finished Medium article:

- Title  
- Subtitle  
- Series line  
- Featured image path + caption  
- Short paragraphs  
- Section headers  
- `IMAGE:` + figure captions  
- Lists, references, series nav  

**Not** HTML. **Not** Big-T instruction chrome.

Part 1 matches the long-form Medium tone you drafted. Parts 2–7 follow the same layout.

```bash
python scripts/build_medium_publish.py
open docs/medium/publish/00-introduction.medium.txt
```

See [`publish/HOW_TO_PUBLISH.md`](publish/HOW_TO_PUBLISH.md).

## Images (one folder per article)

Publish figures live under [`images/<article-slug>/`](images/):

| Article | Folder | Cover |
|---------|--------|-------|
| 00 Introduction | [`images/00-introduction/`](images/00-introduction/) | `thumb.png` |
| 01 Weight quantization | [`images/01-weight-quantization/`](images/01-weight-quantization/) | `thumb.png` |
| 02 KV cache | [`images/02-kv-cache-quantization/`](images/02-kv-cache-quantization/) | `thumb.png` |
| 03 Prefill / TTFT | [`images/03-prefill-and-ttft/`](images/03-prefill-and-ttft/) | `thumb.png` |
| 04 Model ladder | [`images/04-model-size-ladder/`](images/04-model-size-ladder/) | `thumb.png` |
| 05 Full stack | [`images/05-full-optimization-stack/`](images/05-full-optimization-stack/) | `thumb.png` |
| 06 Speculative | [`images/06-speculative-decoding/`](images/06-speculative-decoding/) | `thumb.png` |
| 07 Context / RAG | [`images/07-context-and-cache/`](images/07-context-and-cache/) | `thumb.png` |

In-article figures are named `fig1.png`, `fig2.png`, … in story order (cover stays `thumb.png`). Each folder’s `README.md` lists **Fig N** with a short hint.

Shared concepts (roofline, heatmaps, …) are **copied into each article folder that uses them**, so publishing never depends on a sibling directory. Regenerator cache: [`images/_source/`](images/_source/).

On Medium: upload from that article’s folder; set `thumb.png` as the story cover.

## Regenerate figures

```bash
# all articles
./scripts/regenerate_medium_images.sh "Mac M3"

# one article only (e.g. Part 1)
./scripts/regenerate_medium_images.sh "Mac M3" 00
./scripts/regenerate_medium_images.sh "Mac M3" 01-weight-quantization
```

| Script | What it makes |
|--------|----------------|
| `plot_medium_diagrams.py` | How-it-works workflows |
| `plot_medium_charts.py` | Per-article result bars/scatters |
| `plot_medium_deep.py` | Heatmaps, M3 vs M5, matrices, workloads |
| `plot_paper_redraws.py` | Original paper-idea redraws (not PDF copies) |
| `organize_medium_images.py` | Sync `_source` → per-article folders |
| `medium_image_layout.py` | Manifest of which image belongs to which article |

All plot scripts accept `--article 00` (or a full slug).

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

1. Open `publish/<slug>.medium.txt`  
2. Upload each referenced PNG from `images/<slug>/`  
3. Add tags from the footer  
4. Link as a Series  

> Numbers from [LLM-Inference](https://github.com/Chirumamilla1522/LLM-Inference) on MLX.
