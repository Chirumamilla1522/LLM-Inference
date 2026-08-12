# How to publish these on Medium

Based on proven Medium formatting tips (wide covers, short paragraphs, image breaks, cut fat, publish at 95%).

## What’s in this folder

| File | Purpose |
|------|---------|
| `00`–`07` `.html` | **Paste-ready Medium body** (open → Select All → Copy → Paste into Medium) |
| `*-meta.txt` | Title, subtitle, tags, cover image path |
| `leftovers/` | Cut material (kept for later posts / appendices) |
| `DISTRIBUTION.md` | Where to share after publish |

## Medium editor checklist (do this every post)

### 1. Formatting (looks like a real Medium post)

1. New story → paste the **HTML body** (or paste from browser preview)
2. Set **Title** from `*-meta.txt` (Medium’s big title field — not an H1 in the body)
3. Set **Subtitle** in Medium’s subtitle field
4. Use Medium’s **H2** for section heads (already in the HTML as `<h2>`)
5. Turn fun facts into Medium **pull quotes** (select text → `"` quote button)
6. Prefer short paragraphs (already written that way)
7. Avoid giant markdown tables in the editor — the HTML uses compact lists / small tables

### 2. Wide header image

Upload the thumbnail from `../images/thumbnails/` as the **story cover**:

| Post | Cover file |
|------|------------|
| 00 | `thumb_00_introduction.png` |
| 01 | `thumb_01_weight_quantization.png` |
| 02 | `thumb_02_kv_cache.png` |
| 03 | `thumb_03_prefill_ttft.png` |
| 04 | `thumb_04_model_ladder.png` |
| 05 | `thumb_05_full_stack.png` |
| 06 | `thumb_06_speculative.png` |
| 07 | `thumb_07_rag_context.png` |

These are **16:9** so they look wide in the Medium feed — not a tiny square crop.

### 3. Break up the text with images

After paste, **re-upload** each figure from `../images/` where you see:

```text
[IMAGE: path/to/file.png]
Caption text
```

In Medium: drag the PNG in, then add the caption under it (italic).

Rule of thumb: a visual every **2–4 screens** of scrolling.

### 4. Cut mercilessly

These publish files are already tightened (~8–12 min reads).

Research-length drafts stay in `docs/medium/0*.md`.  
Cut paragraphs live in `leftovers/` — do **not** paste leftovers into Medium.

### 5. Publish at 95%

Ship, then fix typos in the first hour. Don’t polish the draft forever.

### 6. Create value + CTA

Every post ends with:

- Link to the GitHub repo (actionable)
- “Part N of series” prev/next links (update URLs after publish)
- Optional: your newsletter / X / LinkedIn

### 7. Tags (5 max that Medium suggests well)

Use the tags from `*-meta.txt`. Prefer tags with active readers: `Machine Learning`, `Artificial Intelligence`, `Apple`, `LLM`, `Programming`.

### 8. After publish — distribute (1 hour)

See `DISTRIBUTION.md`. This is often the difference between 1K and 10K reads.

### 9. Series

Create a Medium **Series** named e.g. `Local LLMs on Apple Silicon` and add posts 00→07 in order.

---

## Fast paste workflow

```bash
# Preview in browser (optional)
open docs/medium/publish/00-introduction.html
```

1. Open the `.html` in Chrome  
2. Cmd+A → Cmd+C  
3. Medium → New story → Cmd+V  
4. Fix title/subtitle fields  
5. Upload cover + inline images  
6. Add tags → Publish → share per `DISTRIBUTION.md`
