"""Article definitions and run plans for the 12-post inference series."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

from optimizations import (
    OptimizationConfig,
    get_model_repos,
    iter_sweep_configs,
    large_presets,
    sort_presets,
)
from workloads import ARTICLE_07_WORKLOADS

# Default presets for multi-model article sections (24 GB friendly).
DEFAULT_ARTICLE_PRESETS: tuple[str, ...] = (
    "llama3-8b",
    "mistral-7b",
    "qwen-7b",
)


class RunKind(str, Enum):
    STANDARD = "standard"
    SPECULATIVE = "speculative"
    PREFIX_CACHE = "prefix_cache"
    CONCEPT = "concept"


@dataclass(frozen=True)
class ArticleRun:
    """One benchmark invocation."""

    label: str
    preset: str = "llama3-8b"
    config: str = "w4"
    prompt_tokens: int | None = None
    generation_tokens: int | None = None
    draft_preset: str | None = None
    num_draft_tokens: int | None = None
    workload: str | None = None  # scripts/workloads.py profile id
    kind: RunKind = RunKind.STANDARD


@dataclass(frozen=True)
class ArticleSweep:
    """Delegate to run_benchmark.py --sweep with these flags."""

    weights_only: bool = False
    all_models: bool = False
    include_large: bool = False
    max_combo_size: int | None = None
    configs: tuple[str, ...] | None = None  # if set, only these config labels
    presets: tuple[str, ...] | None = None  # if set, only these presets


@dataclass(frozen=True)
class Article:
    id: int
    slug: str
    title: str
    description: str
    benchmarked: bool
    runs: tuple[ArticleRun, ...] = ()
    sweep: ArticleSweep | None = None
    concept_topics: tuple[str, ...] = ()

    @property
    def dir_name(self) -> str:
        return f"article_{self.id:02d}_{self.slug}"


def _configs_from_labels(labels: tuple[str, ...]) -> list[OptimizationConfig]:
    return [OptimizationConfig.from_label(l) for l in labels]


ARTICLES: dict[int, Article] = {
    0: Article(
        id=0,
        slug="introduction",
        title="Local LLMs on Apple Silicon",
        description="Hardware, methodology, and demo benchmark.",
        benchmarked=True,
        runs=(
            ArticleRun(label="demo_fp16", preset="llama3-8b", config="fp16"),
        ),
    ),
    1: Article(
        id=1,
        slug="weight-quantization",
        title="Weight quantization",
        description="fp16, w8, w4, w2 across all presets.",
        benchmarked=True,
        sweep=ArticleSweep(weights_only=True, all_models=True),
    ),
    2: Article(
        id=2,
        slug="kv-cache-quantization",
        title="KV cache quantization",
        description="w4 vs w4+kv_cache; long generation subsection.",
        benchmarked=True,
        runs=tuple(
            ArticleRun(label=f"{p}_w4", preset=p, config="w4")
            for p in DEFAULT_ARTICLE_PRESETS
        )
        + tuple(
            ArticleRun(label=f"{p}_w4_kv", preset=p, config="w4+kv_cache")
            for p in DEFAULT_ARTICLE_PRESETS
        )
        + (
            ArticleRun(
                label="llama3-8b_w4_kv_long_g",
                preset="llama3-8b",
                config="w4+kv_cache",
                generation_tokens=512,
            ),
        ),
    ),
    3: Article(
        id=3,
        slug="prefill-ttft",
        title="Prefill & TTFT",
        description="w4 vs w4+prefill; prompt length subsection.",
        benchmarked=True,
        runs=(
            ArticleRun(label="w4_baseline", preset="llama3-8b", config="w4"),
            ArticleRun(label="w4_prefill", preset="llama3-8b", config="w4+prefill"),
            ArticleRun(
                label="w4_prefill_p1024",
                preset="llama3-8b",
                config="w4+prefill",
                prompt_tokens=1024,
                generation_tokens=64,
            ),
            ArticleRun(
                label="w4_prefill_p256",
                preset="llama3-8b",
                config="w4+prefill",
                prompt_tokens=256,
                generation_tokens=64,
            ),
        ),
    ),
    4: Article(
        id=4,
        slug="model-size-ladder",
        title="Model size & memory ladder",
        description="All presets at w4 (weights-only slice).",
        benchmarked=True,
        sweep=ArticleSweep(
            weights_only=True,
            all_models=True,
            configs=("w4",),
        ),
    ),
    5: Article(
        id=5,
        slug="full-stack",
        title="The full optimization stack",
        description="fp16 vs w4+kv_cache+prefill; full 16-config sweep.",
        benchmarked=True,
        runs=(
            ArticleRun(label="fp16", preset="llama3-8b", config="fp16"),
            ArticleRun(label="optimized", preset="llama3-8b", config="w4+kv_cache+prefill"),
            ArticleRun(label="fp16_mistral", preset="mistral-7b", config="fp16"),
            ArticleRun(
                label="optimized_mistral",
                preset="mistral-7b",
                config="w4+kv_cache+prefill",
            ),
        ),
        sweep=ArticleSweep(all_models=True),
    ),
    6: Article(
        id=6,
        slug="speculative-decoding",
        title="Speculative decoding",
        description="Draft model + target at w4.",
        benchmarked=True,
        runs=tuple(
            ArticleRun(
                label=f"{p}_w4_baseline",
                preset=p,
                config="w4",
                kind=RunKind.STANDARD,
            )
            for p in DEFAULT_ARTICLE_PRESETS
        )
        + tuple(
            ArticleRun(
                label=f"{p}_w4_speculative",
                preset=p,
                config="w4",
                kind=RunKind.SPECULATIVE,
                num_draft_tokens=3,
            )
            for p in DEFAULT_ARTICLE_PRESETS
        ),
    ),
    7: Article(
        id=7,
        slug="context-and-cache",
        title="Context, generation length & prompt cache",
        description="Prompt/generation sweeps and prefix KV reuse.",
        benchmarked=True,
        runs=(
            ArticleRun(
                label="ctx_p256",
                preset="llama3-8b",
                config="w4+prefill",
                prompt_tokens=256,
            ),
            ArticleRun(
                label="ctx_p512",
                preset="llama3-8b",
                config="w4+prefill",
                prompt_tokens=512,
            ),
            ArticleRun(
                label="ctx_p1024",
                preset="llama3-8b",
                config="w4+prefill",
                prompt_tokens=1024,
            ),
            ArticleRun(
                label="ctx_p2048",
                preset="llama3-8b",
                config="w4+prefill",
                prompt_tokens=2048,
                generation_tokens=64,
            ),
            ArticleRun(
                label="gen_g64",
                preset="llama3-8b",
                config="w4+kv_cache",
                generation_tokens=64,
            ),
            ArticleRun(
                label="gen_g256",
                preset="llama3-8b",
                config="w4+kv_cache",
                generation_tokens=256,
            ),
            ArticleRun(
                label="gen_g512",
                preset="llama3-8b",
                config="w4+kv_cache",
                generation_tokens=512,
            ),
            ArticleRun(
                label="prefix_cache",
                preset="llama3-8b",
                config="w4",
                kind=RunKind.PREFIX_CACHE,
            ),
        )
        + tuple(
            ArticleRun(
                label=f"wl_{wid}",
                preset="llama3-8b",
                config="w4+kv_cache+prefill",
                workload=wid,
            )
            for wid in ARTICLE_07_WORKLOADS
        ),
    ),
    8: Article(
        id=8,
        slug="serving",
        title="Production serving at scale",
        description="Continuous batching, PagedAttention, scheduling.",
        benchmarked=False,
        concept_topics=(
            "continuous_batching",
            "paged_attention",
            "disaggregated_prefill_decode",
            "request_scheduling",
        ),
    ),
    9: Article(
        id=9,
        slug="parallelism",
        title="Parallelism for huge models",
        description="Tensor, pipeline, and expert parallelism.",
        benchmarked=False,
        concept_topics=(
            "tensor_parallelism",
            "pipeline_parallelism",
            "expert_parallelism",
        ),
    ),
    10: Article(
        id=10,
        slug="runtimes",
        title="Local runtimes compared",
        description="MLX vs llama.cpp (benchmarked); Ollama notes.",
        benchmarked=True,
        concept_topics=("mlx", "llama_cpp", "ollama"),
    ),
    11: Article(
        id=11,
        slug="tradeoffs",
        title="Quality, cost & when to optimize what",
        description="Decision tree and optimization checklist.",
        benchmarked=False,
        concept_topics=(
            "quantization_quality",
            "local_vs_cloud_cost",
            "optimization_decision_tree",
        ),
    ),
}


def get_article(article_id: int) -> Article:
    if article_id not in ARTICLES:
        known = ", ".join(str(i) for i in sorted(ARTICLES))
        raise ValueError(f"Unknown article id {article_id}. Choose: {known}")
    return ARTICLES[article_id]


def list_articles() -> list[Article]:
    return [ARTICLES[i] for i in sorted(ARTICLES)]


def iter_sweep_configs_for_article(article: Article) -> Iterator[OptimizationConfig]:
    if article.sweep is None:
        return
    if article.sweep.configs:
        for label in article.sweep.configs:
            yield OptimizationConfig.from_label(label)
        return
    yield from iter_sweep_configs(
        weights_only=article.sweep.weights_only,
        max_runtime_combo=article.sweep.max_combo_size,
    )


def presets_for_article_sweep(article: Article, include_large: bool) -> list[str]:
    repos = get_model_repos()
    if article.sweep and article.sweep.presets:
        presets = list(article.sweep.presets)
    elif article.sweep and article.sweep.all_models:
        presets = sort_presets(list(repos))
    else:
        presets = list(DEFAULT_ARTICLE_PRESETS)
    if not include_large:
        large = set(large_presets())
        presets = [p for p in presets if p not in large]
    return presets


def count_planned_runs(article: Article, include_large: bool = False) -> int:
    if not article.benchmarked:
        return 0
    total = len(article.runs)
    if article.sweep:
        configs = list(iter_sweep_configs_for_article(article))
        models = presets_for_article_sweep(article, include_large)
        total += len(configs) * len(models)
    return total
