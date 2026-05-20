"""Repository paths for the benchmark package."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
RESULTS_DIR = ROOT / "results"
RUN_BENCHMARK_SCRIPT = SCRIPTS_DIR / "run_benchmark.py"
