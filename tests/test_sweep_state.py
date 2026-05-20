"""Tests for sweep checkpoint state."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from benchmark.sweep_state import (  # noqa: E402
    mark_completed,
    should_skip,
    sweep_key,
)


def test_sweep_key() -> None:
    assert sweep_key("llama3-8b", "w4") == "llama3-8b:w4"


def test_from_checkpoint_skips_completed() -> None:
    state = {"completed": {}}
    mark_completed(
        state,
        preset="llama3-8b",
        config_label="fp16",
        status="ok",
        output_path="/tmp/fp16.json",
    )
    assert should_skip(state, "llama3-8b", "fp16", from_checkpoint=True, retry_failed=False)
    assert not should_skip(state, "llama3-8b", "w4", from_checkpoint=True, retry_failed=False)


def test_retry_failed_only_runs_non_ok() -> None:
    state = {"completed": {}}
    mark_completed(
        state,
        preset="llama3-8b",
        config_label="fp16",
        status="error",
        output_path="/tmp/fp16.json",
    )
    assert not should_skip(state, "llama3-8b", "fp16", from_checkpoint=False, retry_failed=True)
    mark_completed(
        state,
        preset="llama3-8b",
        config_label="w4",
        status="ok",
        output_path="/tmp/w4.json",
    )
    assert should_skip(state, "llama3-8b", "w4", from_checkpoint=False, retry_failed=True)
