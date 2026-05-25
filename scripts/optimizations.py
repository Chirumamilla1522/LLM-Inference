"""Weight precision levels and runtime optimization sweep ordering."""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

# Weight precisions: fp16 baseline, then 8 / 4 / 2-bit weights.
WEIGHT_BITS_ORDER: tuple[int, ...] = (16, 8, 4, 2)
RUNTIME_OPTIMIZATION_NAMES: tuple[str, ...] = ("kv_cache", "prefill")

ROOT = Path(__file__).resolve().parents[1]
MODELS_JSON = ROOT / "models.json"
ENV_FILE = ROOT / ".env"


def load_env_file() -> None:
    """Load HF_TOKEN from project .env (does not override existing env)."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()


@dataclass(frozen=True)
class OptimizationConfig:
    """One benchmark configuration."""

    weight_bits: int = 16
    kv_cache: bool = False
    prefill: bool = False

    def __post_init__(self) -> None:
        if self.weight_bits not in WEIGHT_BITS_ORDER:
            raise ValueError(f"weight_bits must be one of {WEIGHT_BITS_ORDER}")

    @classmethod
    def from_label(cls, label: str) -> OptimizationConfig:
        if label in ("baseline", "fp16"):
            return cls(weight_bits=16)

        weight_bits = 16
        kv_cache = False
        prefill = False

        for part in label.split("+"):
            if part in ("fp16", "f16"):
                weight_bits = 16
            elif part.startswith("w") and part[1:].isdigit():
                weight_bits = int(part[1:])
            elif part == "kv_cache":
                kv_cache = True
            elif part == "prefill":
                prefill = True
            else:
                raise ValueError(
                    f"Unknown config part '{part}'. "
                    "Use fp16, w8, w4, w2, kv_cache, prefill."
                )

        return cls(weight_bits=weight_bits, kv_cache=kv_cache, prefill=prefill)

    @property
    def weight_label(self) -> str:
        return "fp16" if self.weight_bits == 16 else f"w{self.weight_bits}"

    @property
    def runtime_names(self) -> list[str]:
        return [n for n in RUNTIME_OPTIMIZATION_NAMES if getattr(self, n)]

    @property
    def label(self) -> str:
        parts = [self.weight_label, *self.runtime_names]
        return "+".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight_bits": self.weight_bits,
            "kv_cache": self.kv_cache,
            "prefill": self.prefill,
        }


@dataclass
class RunParams:
    model_repo: str | None
    kv_bits: int | None
    prefill_step_size: int
    config: OptimizationConfig = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_repo": self.model_repo,
            "kv_bits": self.kv_bits,
            "prefill_step_size": self.prefill_step_size,
            "optimizations": self.config.to_dict(),
        }


# Hugging Face repos per preset and weight bits (None = no public repo; run is skipped).
# fp16 uses bf16 / full-precision MLX checkpoints where available.
DEFAULT_MODEL_REPOS: dict[str, dict[int, str | None]] = {
    # --- Tiny (sub-1B) ---
    "qwen-0.5b": {
        16: "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
        8: "mlx-community/Qwen2.5-0.5B-Instruct-8bit",
        4: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        2: None,
    },
    # --- Very small (1–2B) ---
    "llama-3.2-1b": {
        16: "mlx-community/Llama-3.2-1B-Instruct-bf16",
        8: "mlx-community/Llama-3.2-1B-Instruct-8bit",
        4: "mlx-community/Llama-3.2-1B-Instruct-4bit",
        2: None,
    },
    "qwen-1.5b": {
        16: "mlx-community/Qwen2.5-1.5B-Instruct-bf16",
        8: "mlx-community/Qwen2.5-1.5B-Instruct-8bit",
        4: "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        2: None,
    },
    "gemma-2-2b": {
        16: "mlx-community/gemma-2-2b-it-8bit",
        8: "mlx-community/gemma-2-2b-it-8bit",
        4: "mlx-community/gemma-2-2b-it-4bit",
        2: None,
    },
    # --- Small (3–4B) ---
    "llama-3.2-3b": {
        16: "mlx-community/Llama-3.2-3B-Instruct-bf16",
        8: "mlx-community/Llama-3.2-3B-Instruct-8bit",
        4: "mlx-community/Llama-3.2-3B-Instruct-4bit",
        2: None,
    },
    "qwen-3b": {
        16: "mlx-community/Qwen2.5-3B-Instruct-bf16",
        8: "mlx-community/Qwen2.5-3B-Instruct-8bit",
        4: "mlx-community/Qwen2.5-3B-Instruct-4bit",
        2: None,
    },
    "phi-3-mini": {
        16: "mlx-community/Phi-3-mini-4k-instruct-8bit",
        8: "mlx-community/Phi-3-mini-4k-instruct-8bit",
        4: "mlx-community/Phi-3-mini-4k-instruct-4bit",
        2: None,
    },
    "phi-3.5-mini": {
        16: "mlx-community/Phi-3.5-mini-instruct-bf16",
        8: "mlx-community/Phi-3.5-mini-instruct-8bit",
        4: "mlx-community/Phi-3.5-mini-instruct-4bit",
        2: None,
    },
    # --- Medium (7–9B) ---
    "qwen-7b": {
        16: "mlx-community/Qwen2.5-7B-Instruct-bf16",
        8: "mlx-community/Qwen2.5-7B-Instruct-8bit",
        4: "mlx-community/Qwen2.5-7B-Instruct-4bit",
        2: None,
    },
    "mistral-7b": {
        16: "mlx-community/Mistral-7B-Instruct-v0.3",
        8: "mlx-community/Mistral-7B-Instruct-v0.3-8bit",
        4: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        2: None,
    },
    "deepseek-r1-qwen-7b": {
        16: "mlx-community/DeepSeek-R1-Distill-Qwen-7B-bf16",
        8: "mlx-community/DeepSeek-R1-Distill-Qwen-7B-8bit",
        4: "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        2: None,
    },
    "llama3-8b": {
        16: "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16",
        8: "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
        4: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        2: "mlx-community/Llama-3-8B-Instruct-262k-2bit",
    },
    "deepseek-r1-llama-8b": {
        16: "mlx-community/DeepSeek-R1-Distill-Llama-8B-bf16",
        8: "mlx-community/DeepSeek-R1-Distill-Llama-8B-8bit",
        4: "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
        2: None,
    },
    "gemma-9b": {
        16: "mlx-community/gemma-2-9b-it-8bit",
        8: "mlx-community/gemma-2-9b-it-8bit",
        4: "mlx-community/gemma-2-9b-it-4bit",
        2: None,
    },
    # --- Large (12–22B) ---
    "mistral-nemo-12b": {
        16: "mlx-community/Mistral-Nemo-Instruct-2407-bf16",
        8: "mlx-community/Mistral-Nemo-Instruct-2407-8bit",
        4: "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
        2: None,
    },
    "qwen-14b": {
        16: "mlx-community/Qwen2.5-14B-Instruct-bf16",
        8: "mlx-community/Qwen2.5-14B-Instruct-8bit",
        4: "mlx-community/Qwen2.5-14B-Instruct-4bit",
        2: None,
    },
    "mistral-small-22b": {
        16: "mlx-community/Mistral-Small-Instruct-2409-bf16",
        8: "mlx-community/Mistral-Small-Instruct-2409-8bit",
        4: "mlx-community/Mistral-Small-Instruct-2409-4bit",
        2: None,
    },
    "gemma-27b": {
        16: "mlx-community/gemma-2-27b-it-bf16",
        8: "mlx-community/gemma-2-27b-it-8bit",
        4: "mlx-community/gemma-2-27b-it-4bit",
        2: None,
    },
    # --- XL (32B) ---
    "qwen-35b": {
        16: "mlx-community/Qwen2.5-32B-Instruct-bf16",
        8: "mlx-community/Qwen2.5-32B-Instruct-8bit",
        4: "mlx-community/Qwen2.5-32B-Instruct-4bit",
        2: None,
    },
    # --- XXL (70B+) ---
    "llama-70b": {
        16: "mlx-community/Meta-Llama-3.1-70B-Instruct-bf16",
        8: "mlx-community/Meta-Llama-3.1-70B-Instruct-8bit",
        4: "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
        2: None,
    },
    "qwen-72b": {
        16: "mlx-community/Qwen2.5-72B-Instruct-bf16",
        8: "mlx-community/Qwen2.5-72B-Instruct-8bit",
        4: "mlx-community/Qwen2.5-72B-Instruct-4bit",
        2: None,
    },
}

# Sweep order: smallest → largest (params approximate).
MODEL_PRESET_ORDER: tuple[str, ...] = (
    "qwen-0.5b",
    "llama-3.2-1b",
    "qwen-1.5b",
    "gemma-2-2b",
    "llama-3.2-3b",
    "qwen-3b",
    "phi-3-mini",
    "phi-3.5-mini",
    "qwen-7b",
    "mistral-7b",
    "deepseek-r1-qwen-7b",
    "llama3-8b",
    "deepseek-r1-llama-8b",
    "gemma-9b",
    "mistral-nemo-12b",
    "qwen-14b",
    "mistral-small-22b",
    "gemma-27b",
    "qwen-35b",
    "llama-70b",
    "qwen-72b",
)

# Human-readable size labels for docs / tables.
MODEL_PARAM_LABELS: dict[str, str] = {
    "qwen-0.5b": "~0.5B",
    "llama-3.2-1b": "~1B",
    "qwen-1.5b": "~1.5B",
    "gemma-2-2b": "~2B",
    "llama-3.2-3b": "~3B",
    "qwen-3b": "~3B",
    "phi-3-mini": "~3.8B",
    "phi-3.5-mini": "~3.8B",
    "qwen-7b": "~7B",
    "mistral-7b": "~7B",
    "deepseek-r1-qwen-7b": "~7B R1",
    "llama3-8b": "~8B",
    "deepseek-r1-llama-8b": "~8B R1",
    "gemma-9b": "~9B",
    "mistral-nemo-12b": "~12B",
    "qwen-14b": "~14B",
    "mistral-small-22b": "~22B",
    "gemma-27b": "~27B",
    "qwen-35b": "~32B",
    "llama-70b": "~70B",
    "qwen-72b": "~72B",
}

KV_BITS = 4
PREFILL_OPTIMIZED = 2048
PREFILL_BASELINE = 512

# Conservative peak memory (GB) for 512 prompt + 128 generation tokens.
ESTIMATED_PEAK_GB: dict[str, dict[int, float]] = {
    "qwen-0.5b": {16: 1.2, 8: 0.8, 4: 0.5, 2: 0.4},
    "llama-3.2-1b": {16: 2.0, 8: 1.4, 4: 0.9, 2: 0.6},
    "qwen-1.5b": {16: 3.0, 8: 2.0, 4: 1.2, 2: 0.8},
    "gemma-2-2b": {16: 4.0, 8: 4.0, 4: 2.0, 2: 1.2},
    "llama-3.2-3b": {16: 6.0, 8: 4.0, 4: 2.5, 2: 1.8},
    "qwen-3b": {16: 6.0, 8: 4.0, 4: 2.5, 2: 1.8},
    "phi-3-mini": {16: 7.5, 8: 7.5, 4: 3.0, 2: 2.0},
    "phi-3.5-mini": {16: 7.5, 8: 5.0, 4: 3.0, 2: 2.0},
    "qwen-7b": {16: 14.0, 8: 8.5, 4: 5.0, 2: 4.0},
    "mistral-7b": {16: 15.0, 8: 9.0, 4: 5.5, 2: 4.0},
    "deepseek-r1-qwen-7b": {16: 15.0, 8: 9.0, 4: 5.5, 2: 4.0},
    "llama3-8b": {16: 16.0, 8: 9.5, 4: 6.5, 2: 4.5},
    "deepseek-r1-llama-8b": {16: 16.0, 8: 9.5, 4: 6.5, 2: 4.5},
    "gemma-9b": {16: 10.0, 8: 10.0, 4: 6.0, 2: 4.5},
    "mistral-nemo-12b": {16: 24.0, 8: 14.0, 4: 8.5, 2: 6.0},
    "qwen-14b": {16: 28.0, 8: 16.0, 4: 10.0, 2: 7.0},
    "mistral-small-22b": {16: 44.0, 8: 26.0, 4: 14.0, 2: 10.0},
    "gemma-27b": {16: 54.0, 8: 30.0, 4: 18.0, 2: 12.0},
    "qwen-35b": {16: 72.0, 8: 34.0, 4: 22.0, 2: 14.0},
    "llama-70b": {16: 140.0, 8: 72.0, 4: 42.0, 2: 28.0},
    "qwen-72b": {16: 145.0, 8: 76.0, 4: 44.0, 2: 30.0},
}

# Skip entire preset if system RAM (GB) is below this threshold.
MIN_RAM_GB_BY_PRESET: dict[str, float] = {
    "qwen-0.5b": 0,
    "llama-3.2-1b": 0,
    "qwen-1.5b": 0,
    "gemma-2-2b": 0,
    "llama-3.2-3b": 0,
    "qwen-3b": 0,
    "phi-3-mini": 0,
    "phi-3.5-mini": 0,
    "qwen-7b": 0,
    "mistral-7b": 0,
    "deepseek-r1-qwen-7b": 0,
    "llama3-8b": 0,
    "deepseek-r1-llama-8b": 0,
    "gemma-9b": 0,
    "mistral-nemo-12b": 20,
    "qwen-14b": 22,
    "mistral-small-22b": 28,
    "gemma-27b": 32,
    "qwen-35b": 36,
    "llama-70b": 48,
    "qwen-72b": 64,
}


def requires_large_machine(preset: str) -> bool:
    """True if preset needs --include-large on typical 24GB Macs."""
    return MIN_RAM_GB_BY_PRESET.get(preset, 0) > 0


def large_presets() -> list[str]:
    return [p for p in MODEL_PRESET_ORDER if requires_large_machine(p)]


# Backwards-compatible name used in run_benchmark.py
LARGE_MODEL_PRESETS: frozenset[str] = frozenset(large_presets())


def _load_repo_overrides() -> dict[str, dict[str, str]]:
    if not MODELS_JSON.exists():
        return {}
    data = json.loads(MODELS_JSON.read_text())
    return data.get("overrides", {})


def get_model_repos() -> dict[str, dict[int, str | None]]:
    repos = {k: dict(v) for k, v in DEFAULT_MODEL_REPOS.items()}
    overrides = _load_repo_overrides()
    for preset, bits_map in overrides.items():
        if preset not in repos:
            repos[preset] = {16: None, 8: None, 4: None, 2: None}
        for bit_str, repo in bits_map.items():
            repos[preset][int(bit_str)] = repo
    return repos


MODEL_REPOS = get_model_repos()

# Draft presets for speculative decoding (Article 6). Must share tokenizer vocab.
DRAFT_PRESET_BY_TARGET: dict[str, str] = {
    "qwen-0.5b": "qwen-0.5b",
    "llama-3.2-1b": "llama-3.2-1b",
    "qwen-1.5b": "qwen-0.5b",
    "gemma-2-2b": "qwen-0.5b",
    "llama-3.2-3b": "llama-3.2-1b",
    "qwen-3b": "qwen-0.5b",
    "phi-3-mini": "qwen-0.5b",
    "phi-3.5-mini": "qwen-0.5b",
    "qwen-7b": "qwen-0.5b",
    # mistral-7b: no smaller mlx-community model shares v0.3 vocab (32768)
    "deepseek-r1-qwen-7b": "qwen-0.5b",
    "llama3-8b": "llama-3.2-1b",
    "deepseek-r1-llama-8b": "llama-3.2-1b",
    "gemma-9b": "qwen-0.5b",
    "mistral-nemo-12b": "qwen-0.5b",
    "qwen-14b": "qwen-1.5b",
    "mistral-small-22b": "qwen-1.5b",
    "gemma-27b": "qwen-1.5b",
    "qwen-35b": "qwen-1.5b",
    "llama-70b": "qwen-1.5b",
    "qwen-72b": "qwen-1.5b",
}

DEFAULT_NUM_DRAFT_TOKENS = 3


def resolve_draft_preset(target_preset: str) -> str | None:
    return DRAFT_PRESET_BY_TARGET.get(target_preset)


def resolve_draft_repo(target_preset: str, weight_bits: int) -> str | None:
    draft_preset = resolve_draft_preset(target_preset)
    if draft_preset is None:
        return None
    return get_model_repos().get(draft_preset, {}).get(weight_bits)


def iter_sweep_configs(
    *,
    weights_only: bool = False,
    max_runtime_combo: int | None = None,
    weight_bits_filter: tuple[int, ...] | None = None,
) -> Iterator[OptimizationConfig]:
    """
    Sweep order:
      fp16 → w8 → w4 → w2  (weight only)
      then per weight: no runtime opts → kv → prefill → kv+prefill
    """
    weights = weight_bits_filter or WEIGHT_BITS_ORDER

    if weights_only:
        for wb in weights:
            yield OptimizationConfig(weight_bits=wb)
        return

    for wb in weights:
        for size in range(len(RUNTIME_OPTIMIZATION_NAMES) + 1):
            if max_runtime_combo is not None and size > max_runtime_combo:
                continue
            for combo in itertools.combinations(RUNTIME_OPTIMIZATION_NAMES, size):
                yield OptimizationConfig(
                    weight_bits=wb,
                    kv_cache="kv_cache" in combo,
                    prefill="prefill" in combo,
                )


def resolve_run_params(model_preset: str, config: OptimizationConfig) -> RunParams:
    repos = get_model_repos()
    if model_preset not in repos:
        known = ", ".join(sorted(repos))
        raise ValueError(f"Unknown model preset '{model_preset}'. Choose: {known}")

    model_repo = repos[model_preset].get(config.weight_bits)
    kv_bits = KV_BITS if config.kv_cache else None
    prefill_step_size = PREFILL_OPTIMIZED if config.prefill else PREFILL_BASELINE

    return RunParams(
        model_repo=model_repo,
        kv_bits=kv_bits,
        prefill_step_size=prefill_step_size,
        config=config,
    )


def get_system_memory_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        pass
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return (pages * page_size) / (1024**3)
    except (AttributeError, ValueError, OSError):
        return 24.0


def get_memory_budget_gb(fraction: float = 0.75) -> float:
    """MLX warns near ~75% of unified memory on Apple Silicon."""
    return get_system_memory_gb() * fraction


def estimate_peak_gb(model_preset: str, config: OptimizationConfig) -> float:
    base = ESTIMATED_PEAK_GB.get(model_preset, {}).get(config.weight_bits, 0.0)
    kv_overhead = 1.5 if config.kv_cache else 0.0
    return base + kv_overhead


def config_fits_memory(
    model_preset: str, config: OptimizationConfig, budget_gb: float
) -> bool:
    estimate = estimate_peak_gb(model_preset, config)
    if estimate <= 0:
        return True
    return estimate <= budget_gb


def sort_presets(presets: list[str]) -> list[str]:
    order = {name: i for i, name in enumerate(MODEL_PRESET_ORDER)}
    return sorted(presets, key=lambda p: order.get(p, 999))


def should_skip_preset(model_preset: str, system_ram_gb: float) -> str | None:
    min_ram = MIN_RAM_GB_BY_PRESET.get(model_preset, 0)
    if min_ram and system_ram_gb < min_ram:
        label = MODEL_PARAM_LABELS.get(model_preset, model_preset)
        return (
            f"{model_preset} ({label}) needs ~{min_ram:.0f}GB+ unified memory "
            f"(this machine has ~{system_ram_gb:.0f}GB). "
            "Use --include-large on a bigger Mac, or run a single config with "
            f"`--preset {model_preset} --weight-bits 4`."
        )
    return None


def hf_token_present() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token

        return get_token() is not None
    except Exception:
        return False


def check_hf_auth_for_repo(repo: str | None) -> str | None:
    """Return an error message if a Hugging Face repo cannot be accessed."""
    if repo is None:
        return None
    try:
        from huggingface_hub import HfApi

        HfApi().model_info(repo)
        return None
    except Exception as exc:
        err = str(exc)
        if "401" in err or "403" in err:
            return (
                f"Cannot access {repo} (auth/license). Run ./scripts/hf_login.sh "
                "and accept the model license on huggingface.co."
            )
        if "404" in err:
            return (
                f"Repo not found: {repo}. Update models.json overrides or "
                "DEFAULT_MODEL_REPOS in scripts/optimizations.py."
            )
        return f"Cannot access {repo}: {exc}"


def collect_repos_in_sweep(
    models: list[str], configs: list[OptimizationConfig]
) -> list[str]:
    repos_map = get_model_repos()
    needed: list[str] = []
    for preset in models:
        for config in configs:
            repo = repos_map.get(preset, {}).get(config.weight_bits)
            if repo and repo not in needed:
                needed.append(repo)
    return needed
