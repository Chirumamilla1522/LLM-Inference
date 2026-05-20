"""Unit tests for benchmark schema and parsers (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from benchmark_schema import (  # noqa: E402
    LEGACY_CONFIG_MAP,
    build_stats_block,
    fix_stale_repo,
    infer_config_from_legacy,
    legacy_optimizations_to_modern,
    normalize_config_label,
    parse_llama_bench_output,
    trial_stats,
    validate_result,
)
from optimizations import OptimizationConfig  # noqa: E402


class TestOptimizationConfig:
    def test_fp16_label(self) -> None:
        c = OptimizationConfig.from_label("fp16")
        assert c.weight_bits == 16
        assert c.label == "fp16"

    def test_w4_kv_prefill(self) -> None:
        c = OptimizationConfig.from_label("w4+kv_cache+prefill")
        assert c.weight_bits == 4
        assert c.kv_cache and c.prefill

    def test_baseline_alias(self) -> None:
        c = OptimizationConfig.from_label("baseline")
        assert c.weight_bits == 16


class TestLegacyMigration:
    def test_normalize_baseline(self) -> None:
        assert normalize_config_label("baseline") == "fp16"

    def test_infer_quantization(self) -> None:
        data = {
            "configuration": "quantization",
            "optimizations": {"quantization": True, "kv_cache": False, "prefill": False},
        }
        assert infer_config_from_legacy(data) == "w4"

    def test_legacy_opt_modern(self) -> None:
        assert legacy_optimizations_to_modern(
            {"quantization": True, "kv_cache": True, "prefill": False}
        ) == {"weight_bits": 4, "kv_cache": True, "prefill": False}

    def test_fix_stale_llama_repo(self) -> None:
        repo = "mlx-community/Meta-Llama-3-8B-Instruct-bf16"
        fixed, changed = fix_stale_repo(repo)
        assert changed
        assert "3.1" in fixed


class TestTrialStats:
    def test_median_and_p95(self) -> None:
        s = trial_stats([10.0, 20.0, 30.0, 40.0, 100.0])
        assert s["median"] == 30.0
        assert s["p95"] >= 40.0
        assert s["n"] == 5

    def test_build_stats_block(self) -> None:
        block = build_stats_block(
            ttft_ms=[100.0, 110.0, 120.0],
            throughput_tps=[20.0, 22.0, 21.0],
            memory_gb=[5.0, 5.1, 5.2],
        )
        assert "ttft_ms" in block
        assert block["throughput_tps"]["median"] == 21.0


class TestValidateResult:
    def test_ok_minimal(self) -> None:
        data = {
            "timestamp": "t",
            "hardware": "Mac M3",
            "model_preset": "llama3-8b",
            "configuration": "w4",
            "status": "ok",
            "prompt_tokens": 512,
            "generation_tokens": 128,
            "num_trials": 3,
            "throughput_tps": 20.0,
            "schema_version": 1,
        }
        assert validate_result(data) == []

    def test_stale_repo_flagged(self) -> None:
        data = {
            "timestamp": "t",
            "hardware": "Mac M3",
            "model_preset": "llama3-8b",
            "configuration": "fp16",
            "status": "ok",
            "prompt_tokens": 512,
            "generation_tokens": 128,
            "num_trials": 1,
            "throughput_tps": 1.0,
            "model_repo": "mlx-community/Meta-Llama-3-8B-Instruct-bf16",
        }
        errs = validate_result(data)
        assert any("stale" in e for e in errs)


class TestLlamaBenchParser:
    def test_parse_pp_tg(self) -> None:
        text = """
        llama 8B Q4_K
        pp512: 240.5 t/s
        tg128: 18.2 t/s
        """
        pp, tg = parse_llama_bench_output(text)
        assert pp == pytest.approx(240.5)
        assert tg == pytest.approx(18.2)


class TestLegacyConfigMap:
    def test_all_legacy_resolve(self) -> None:
        for legacy, modern in LEGACY_CONFIG_MAP.items():
            OptimizationConfig.from_label(modern)
