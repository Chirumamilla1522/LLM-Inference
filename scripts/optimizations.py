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
# fp16 uses bf16 MLX checkpoints. Gated repos need: huggingface-cli login
DEFAULT_MODEL_REPOS: dict[str, dict[int, str | None]] = {
    "llama3-8b": {
        16: "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16",
        8: "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
        4: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        2: "mlx-community/Llama-3-8B-Instruct-262k-2bit",
    },
    "mistral-7b": {
        16: "mlx-community/Mistral-7B-Instruct-v0.3-bf16",
        8: "mlx-community/Mistral-7B-Instruct-v0.3-8bit",
        4: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        2: None,
    },
    "qwen-35b": {
        16: "mlx-community/Qwen2.5-32B-Instruct-bf16",
        8: "mlx-community/Qwen2.5-32B-Instruct-8bit",
        4: "mlx-community/Qwen2.5-32B-Instruct-4bit",
        2: None,
    },
}

KV_BITS = 4
PREFILL_OPTIMIZED = 2048
PREFILL_BASELINE = 512

# Repos that require Hugging Face authentication (gated licenses).
GATED_REPOS: frozenset[str] = frozenset(
    {
        "mlx-community/Mistral-7B-Instruct-v0.3-bf16",
        "mlx-community/Meta-Llama-3-8B-Instruct-bf16",
    }
)

# Conservative peak memory (GB) for 512 prompt + 128 generation tokens.
ESTIMATED_PEAK_GB: dict[str, dict[int, float]] = {
    "llama3-8b": {16: 16.0, 8: 9.5, 4: 6.5, 2: 4.5},
    "mistral-7b": {16: 15.0, 8: 9.0, 4: 5.5, 2: 4.0},
    "qwen-35b": {16: 72.0, 8: 34.0, 4: 22.0, 2: 14.0},
}

# Skip large-model presets below this unified memory (GB).
MIN_RAM_GB_FOR_QWEN_32B = 36


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


def should_skip_preset(model_preset: str, system_ram_gb: float) -> str | None:
    if model_preset == "qwen-35b" and system_ram_gb < MIN_RAM_GB_FOR_QWEN_32B:
        return (
            f"{model_preset} needs ~{MIN_RAM_GB_FOR_QWEN_32B}GB+ unified memory "
            f"(this machine has ~{system_ram_gb:.0f}GB). "
            "Use --include-qwen on a large-memory Mac, or run qwen alone with "
            "`--preset qwen-35b --weight-bits 4`."
        )
    return None


def hf_token_present() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def check_hf_auth_for_repo(repo: str | None) -> str | None:
    """Return an error message if a gated repo cannot be accessed."""
    if repo is None or repo not in GATED_REPOS:
        return None
    if hf_token_present():
        try:
            from huggingface_hub import HfApi

            HfApi().model_info(repo)
            return None
        except Exception as exc:
            return (
                f"HF token set but cannot access {repo}: {exc}. "
                "Accept the model license at huggingface.co and retry."
            )
    return (
        f"Repo {repo} requires Hugging Face login. Run:\n"
        "  huggingface-cli login\n"
        "  # accept the model license on huggingface.co first"
    )


def collect_gated_repos_in_sweep(
    models: list[str], configs: list[OptimizationConfig]
) -> list[str]:
    repos = get_model_repos()
    needed: list[str] = []
    for preset in models:
        for config in configs:
            repo = repos.get(preset, {}).get(config.weight_bits)
            if repo and repo in GATED_REPOS and repo not in needed:
                needed.append(repo)
    return needed
