# Survey paper (NeurIPS 2025 style, arXiv preprint)

NeurIPS-formatted survey of LLM inference optimizations for **everyday devices** (laptops, phones, shared DRAM), plus the empirical Apple Silicon slice from this repo.

**Title:** *Running Large Language Models on Everyday Devices: A Survey of Inference Optimizations*

## Before you compile or upload

1. In `main.tex`, replace `REPLACE_WITH_EMAIL` and the affiliation footnote.
2. Fill `\begin{ack}...\end{ack}` if you have funding or hardware donors.
3. Keep `\usepackage[preprint]{neurips_2025}` for arXiv (authors visible, no line numbers).

## Build the PDF

```bash
cd paper
make          # uses tectonic
# or: tectonic -X compile main.tex
```

Or: `make -C paper`.

The thesis is on-device / daily-driver inference (laptops, phones, shared DRAM), not datacenter serving as the default.

## Upload to arXiv

1. Build locally and read the PDF end-to-end (author name, email, numbers, citations).
2. Create a zip that contains only what arXiv needs:

```bash
cd paper
zip -r llm-inference-survey-arxiv.zip \
  main.tex sec_*.tex references.bib neurips_2025.sty figures/
```

3. At [arxiv.org/submit](https://arxiv.org/submit):
   - **Primary category:** `cs.LG` (also consider `cs.CL`, `cs.DC`).
   - **License:** pick one you are willing to keep (e.g. CC BY 4.0).
   - Upload the zip; set the main file to `main.tex`.
4. Abstract: paste the abstract from `main.tex` (plain text, no LaTeX except simple math).
5. After announcement, add the arXiv ID to this README and to the repo root README if you want.

arXiv compiles with TeX Live; if the cloud build fails, upload a PDF-only submission as a fallback (less preferred).

## NeurIPS vs arXiv

| Goal | Package line | Notes |
|------|----------------|-------|
| **arXiv preprint (this default)** | `\usepackage[preprint]{neurips_2025}` | No page limit; authors visible |
| Main / D&B anonymous submit | `\usepackage{neurips_2025}` or `[dandb]` | Hide authors; 9-page main-text limit; add the NeurIPS checklist |
| Workshop, single-blind | `\usepackage[sglblindworkshop]{neurips_2025}` | Also set `\workshoptitle{...}` |

A broad survey is a natural **arXiv** paper. The NeurIPS *main* track rarely accepts surveys. Stronger conference fits are a **workshop** or **Datasets & Benchmarks** if you foreground the harness and JSON protocol and cut to the page limit.

## What the paper covers

Related work (datacenter serving surveys, on-device/SLM surveys, compression, KV, kernels, speculative decoding, local runtimes) plus ten optimization families ranked for daily-driver machines, recipes for 16/24/64+ GB, and M3 / M5 Max measurements.
