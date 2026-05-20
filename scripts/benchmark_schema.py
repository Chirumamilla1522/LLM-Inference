"""Result JSON schema, validation, and trial statistics."""

from __future__ import annotations

import math
import re
from typing import Any

SCHEMA_VERSION = 1

REQUIRED_FIELDS = (
    "timestamp",
    "hardware",
    "model_preset",
    "configuration",
    "status",
    "prompt_tokens",
    "generation_tokens",
    "num_trials",
)

# Legacy filename / configuration labels → current labels.
LEGACY_CONFIG_MAP: dict[str, str] = {
    "baseline": "fp16",
    "quantization": "w4",
    "kv_cache": "fp16+kv_cache",
    "prefill": "fp16+prefill",
    "quantization+kv_cache": "w4+kv_cache",
    "quantization+prefill": "w4+prefill",
    "quantization+kv_cache+prefill": "w4+kv_cache+prefill",
    "kv_cache+prefill": "fp16+kv_cache+prefill",
}

# Stale HF repo id fragments → replacement (Llama 3 → 3.1).
STALE_REPO_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("Meta-Llama-3-8B-Instruct", "Meta-Llama-3.1-8B-Instruct"),
    ("Meta-Llama-3-8B", "Meta-Llama-3.1-8B"),
)

WARMUP_POLICY = {
    "warmup_trials": 1,
    "warmup_discarded": True,
    "measured_trials_field": "num_trials",
    "description": "One full generate before timed trials; trials use seed+trial.",
}

VALID_STATUSES = frozenset({"ok", "error", "skipped"})


def normalize_config_label(label: str) -> str:
    return LEGACY_CONFIG_MAP.get(label, label)


def is_valid_config_label(label: str) -> bool:
    """Accept current labels and known legacy (pre-migration)."""
    if label in LEGACY_CONFIG_MAP:
        return True
    try:
        from optimizations import OptimizationConfig

        OptimizationConfig.from_label(label)
        return True
    except ValueError:
        return False


def legacy_optimizations_to_modern(opt: dict[str, Any]) -> dict[str, Any]:
    """Convert old {quantization, kv_cache, prefill} to {weight_bits, kv_cache, prefill}."""
    if "weight_bits" in opt:
        return {
            "weight_bits": int(opt["weight_bits"]),
            "kv_cache": bool(opt.get("kv_cache", False)),
            "prefill": bool(opt.get("prefill", False)),
        }
    weight_bits = 4 if opt.get("quantization") else 16
    return {
        "weight_bits": weight_bits,
        "kv_cache": bool(opt.get("kv_cache", False)),
        "prefill": bool(opt.get("prefill", False)),
    }


def infer_config_from_legacy(data: dict[str, Any]) -> str:
    label = data.get("configuration", "")
    if label in LEGACY_CONFIG_MAP:
        return LEGACY_CONFIG_MAP[label]
    opt = data.get("optimizations") or {}
    if isinstance(opt, dict) and "weight_bits" not in opt:
        modern = legacy_optimizations_to_modern(opt)
        parts = ["fp16" if modern["weight_bits"] == 16 else f"w{modern['weight_bits']}"]
        if modern["kv_cache"]:
            parts.append("kv_cache")
        if modern["prefill"]:
            parts.append("prefill")
        return "+".join(parts)
    return label


def fix_stale_repo(repo: str | None) -> tuple[str | None, bool]:
    if not repo:
        return repo, False
    updated = repo
    changed = False
    for old, new in STALE_REPO_REPLACEMENTS:
        if old in updated and new not in updated:
            updated = updated.replace(old, new)
            changed = True
    return updated, changed


def trial_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {}
    n = len(values)
    sorted_v = sorted(values)
    mean = sum(values) / n

    def _percentile(p: float) -> float:
        if n == 1:
            return sorted_v[0]
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_v[int(k)]
        return sorted_v[f] + (sorted_v[c] - sorted_v[f]) * (k - f)

    median = _percentile(0.5)
    variance = sum((x - mean) ** 2 for x in values) / n
    return {
        "n": n,
        "mean": round(mean, 4),
        "median": round(median, 4),
        "p50": round(_percentile(0.5), 4),
        "p95": round(_percentile(0.95), 4),
        "std": round(math.sqrt(variance), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def build_stats_block(
    *,
    ttft_ms: list[float],
    throughput_tps: list[float],
    memory_gb: list[float],
    prompt_tps: list[float] | None = None,
    draft_accept_rate: list[float] | None = None,
) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    if ttft_ms:
        out["ttft_ms"] = trial_stats(ttft_ms)
    if throughput_tps:
        out["throughput_tps"] = trial_stats(throughput_tps)
    if memory_gb:
        out["memory_gb"] = trial_stats(memory_gb)
    if prompt_tps:
        out["prompt_tps"] = trial_stats(prompt_tps)
    if draft_accept_rate:
        out["draft_accept_rate"] = trial_stats(draft_accept_rate)
    return out


def primary_metric(stats: dict[str, float | int] | None, *, fallback: float = 0.0) -> float:
    """Prefer median for reporting."""
    if not stats:
        return fallback
    return float(stats.get("median", stats.get("mean", fallback)))


def parse_llama_bench_output(text: str) -> tuple[float | None, float | None]:
    """Parse llama-bench stdout for pp/tg t/s (shared with compare_runtimes)."""
    pp_vals: list[float] = []
    tg_vals: list[float] = []
    for line in text.splitlines():
        lower = line.lower()
        if "pp" in lower and "t/s" in lower:
            nums = re.findall(r"[\d.]+", line)
            if nums:
                pp_vals.append(float(nums[-1]))
        if "tg" in lower and "t/s" in lower:
            nums = re.findall(r"[\d.]+", line)
            if nums:
                tg_vals.append(float(nums[-1]))
    if not tg_vals:
        for line in text.splitlines():
            if "t/s" in line.lower():
                nums = re.findall(r"[\d.]+", line)
                if len(nums) >= 2:
                    tg_vals.append(float(nums[-1]))
    pp = sum(pp_vals) / len(pp_vals) if pp_vals else None
    tg = sum(tg_vals) / len(tg_vals) if tg_vals else None
    return pp, tg


def validate_result(data: dict[str, Any], *, path: str = "") -> list[str]:
    """Return list of validation errors (empty = ok)."""
    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"{path}missing field: {field}")

    status = data.get("status")
    if status and status not in VALID_STATUSES:
        errors.append(f"{path}invalid status: {status}")

    cfg = data.get("configuration", "")
    if cfg and not is_valid_config_label(str(cfg)):
        errors.append(f"{path}unknown configuration label: {cfg}")

    sv = data.get("schema_version")
    if sv is not None and sv != SCHEMA_VERSION:
        errors.append(f"{path}schema_version {sv} != current {SCHEMA_VERSION}")

    if status == "ok":
        if not data.get("throughput_tps") and not data.get("stats"):
            errors.append(f"{path}ok result missing throughput_tps")
        if data.get("model_repo") and "Meta-Llama-3-8B" in str(data["model_repo"]):
            if "3.1" not in str(data["model_repo"]):
                errors.append(f"{path}stale model_repo (Llama 3 → use 3.1): {data['model_repo']}")

    return errors
