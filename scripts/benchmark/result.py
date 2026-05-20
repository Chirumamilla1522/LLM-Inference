"""Benchmark result model and persistence."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from benchmark.paths import RESULTS_DIR
from benchmark.schema import SCHEMA_VERSION, WARMUP_POLICY
from optimizations import OptimizationConfig, resolve_run_params
from workloads import WorkloadProfile


def package_version(module: str) -> str:
    try:
        return subprocess.check_output(
            [sys.executable, "-m", module, "--version"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def detect_hardware(label: str | None) -> str:
    if label:
        return label
    chip = platform.processor() or "Apple Silicon"
    return f"{chip} ({platform.machine()})"


def avg(values: list[float]) -> float:
    return sum(values) / len(values)


@dataclass
class BenchmarkResult:
    timestamp: str
    hardware: str
    model_preset: str
    configuration: str
    weight_bits: int
    optimizations: dict
    model_repo: str | None
    kv_bits: int | None
    prefill_step_size: int
    prompt_tokens: int
    generation_tokens: int
    num_trials: int
    memory_gb: float
    ttft_ms: float
    throughput_tps: float
    prompt_tps: float
    mlx_version: str
    mlx_lm_version: str
    platform: str
    status: str = "ok"
    error: str | None = None
    article_id: int | None = None
    run_label: str | None = None
    benchmark_mode: str = "standard"
    draft_model_repo: str | None = None
    draft_preset: str | None = None
    num_draft_tokens: int | None = None
    draft_accept_rate: float | None = None
    prefix_cache_cold_ttft_ms: float | None = None
    prefix_cache_warm_ttft_ms: float | None = None
    prefix_system_tokens: int | None = None
    workload: dict | None = None
    schema_version: int = SCHEMA_VERSION
    trials: dict | None = None
    stats: dict | None = None
    warmup_policy: dict | None = None

    @property
    def runtime_combo_size(self) -> int:
        return int(self.optimizations.get("kv_cache", False)) + int(
            self.optimizations.get("prefill", False)
        )


def failed_result(
    *,
    model_preset: str,
    config: OptimizationConfig,
    hardware: str,
    error: str,
    prompt_tokens: int,
    generation_tokens: int,
    num_trials: int,
    workload: WorkloadProfile | None = None,
) -> BenchmarkResult:
    params = resolve_run_params(model_preset, config)
    return BenchmarkResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        hardware=hardware,
        model_preset=model_preset,
        configuration=config.label,
        weight_bits=config.weight_bits,
        optimizations=config.to_dict(),
        model_repo=params.model_repo,
        kv_bits=params.kv_bits,
        prefill_step_size=params.prefill_step_size,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        num_trials=num_trials,
        memory_gb=0.0,
        ttft_ms=0.0,
        throughput_tps=0.0,
        prompt_tps=0.0,
        mlx_version=package_version("mlx"),
        mlx_lm_version=package_version("mlx_lm"),
        platform=platform.platform(),
        status="error",
        error=error,
        workload=workload.to_metadata() if workload else None,
        schema_version=SCHEMA_VERSION,
        warmup_policy=WARMUP_POLICY,
    )


def save_result(result: BenchmarkResult, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(result), indent=2) + "\n")
    return output


def result_path(
    hardware: str,
    model_preset: str,
    config_label: str,
    *,
    output_root: Path | None = None,
    run_label: str | None = None,
) -> Path:
    safe_hw = hardware.replace(" ", "_").replace("/", "-")
    base = output_root if output_root is not None else RESULTS_DIR / safe_hw
    if output_root is not None:
        filename = f"{run_label}.json" if run_label else f"{config_label}.json"
        return base / model_preset / filename
    return base / model_preset / f"{config_label}.json"


def load_result_from_file(path: Path) -> BenchmarkResult | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    fields = BenchmarkResult.__dataclass_fields__
    return BenchmarkResult(**{k: v for k, v in data.items() if k in fields})
