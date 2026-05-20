"""Smoke tests: CLI exits 0 without GPU."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(*args: str) -> int:
    return subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True).returncode


def test_run_benchmark_dry_run() -> None:
    assert (
        _run(
            "scripts/run_benchmark.py",
            "--dry-run",
            "--preset",
            "llama3-8b",
            "--config",
            "w4",
        )
        == 0
    )


def test_run_benchmark_list_workloads() -> None:
    assert _run("scripts/run_benchmark.py", "--list-workloads") == 0


def test_compare_runtimes_dry_run() -> None:
    assert _run("scripts/compare_runtimes.py", "--dry-run", "--hardware", "Mac M3") == 0


def test_run_article_list() -> None:
    assert _run("scripts/run_article.py", "--list") == 0


def test_validate_results_runs() -> None:
    assert _run("scripts/validate_results.py", "--hardware", "Mac M3") in (0, 1)


def test_workloads_list() -> None:
    assert _run("scripts/workloads.py") == 0
