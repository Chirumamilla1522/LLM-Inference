# How to publish (Medium block editor — not HTML)

Medium is a **block editor**. These kits map to Medium’s controls:

| In the `.medium.txt` file | In Medium |
|---------------------------|-----------|
| **TITLE (Big T)** | Title field → large **Big T** |
| **SUBTITLE (Little T)** | Line under title → **Little T** |
| **FEATURED IMAGE** | Wide horizontal image under subtitle (story cover) |
| **SUBHEAD** | Section header style |
| **BODY** | Normal paragraph (keep short) |
| **PULL QUOTE** | Select text → quote button |
| **INLINE IMAGE** | `+` → Image → upload → caption |
| **BULLET / NUMBERED LIST** | `-` or `1.` shortcuts |
| **CODE BLOCK** | `+` → Code block |
| **TAGS** | Tag picker at the bottom |

## Steps for each article

1. Open `NN-*.medium.txt` and the matching `NN-*-meta.txt`
2. Medium → **New story**
3. Paste **TITLE** with **Big T**
4. Paste **SUBTITLE** with **Little T**
5. `+` → Image → upload the **FEATURED IMAGE** (from `images/thumbnails/`, 16:9)
6. Walk the file top to bottom:
   - SUBHEAD → apply section header
   - BODY → paste as normal text (one paragraph per BODY block)
   - PULL QUOTE → quote style
   - INLINE IMAGE → upload from the `UPLOAD:` path, paste CAPTION under it
7. Add **TAGS**
8. Publish at 95%, then fix typos
9. Spend ~1 hour on [`DISTRIBUTION.md`](DISTRIBUTION.md)

## Why not HTML?

Medium strips most HTML/CSS. Pasting HTML fights the editor.
These kits match how Medium actually wants you to write.

## Files

- `00-introduction.medium.txt` … `07-context-and-cache.medium.txt`
- matching `*-meta.txt` for quick copy of title/subtitle/tags/cover

Regenerate:

```bash
python scripts/build_medium_publish.py
```
