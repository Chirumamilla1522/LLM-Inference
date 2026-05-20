"""Checkpoint file for long optimization sweeps."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sweep_key(model_preset: str, config_label: str) -> str:
    return f"{model_preset}:{config_label}"


def state_path(output_root: Path) -> Path:
    return output_root / "sweep_state.json"


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": 1,
            "completed": {},
            "updated_at": None,
        }
    return json.loads(path.read_text())


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, indent=2) + "\n")


def get_entry(state: dict[str, Any], preset: str, config_label: str) -> dict[str, Any] | None:
    return state.get("completed", {}).get(sweep_key(preset, config_label))


def should_skip(
    state: dict[str, Any],
    preset: str,
    config_label: str,
    *,
    from_checkpoint: bool,
    retry_failed: bool,
) -> bool:
    if not from_checkpoint and not retry_failed:
        return False
    entry = get_entry(state, preset, config_label)
    if entry is None:
        return False
    if retry_failed:
        return entry.get("status") == "ok"
    # from_checkpoint: skip anything already recorded (ok or skipped)
    return True


def mark_completed(
    state: dict[str, Any],
    *,
    preset: str,
    config_label: str,
    status: str,
    output_path: str | None = None,
    error: str | None = None,
) -> None:
    state.setdefault("completed", {})[sweep_key(preset, config_label)] = {
        "status": status,
        "output": output_path,
        "error": error,
        "at": datetime.now(timezone.utc).isoformat(),
    }


def init_state(
    *,
    hardware: str,
    models: list[str],
    configs: list[str],
    sweep_output_root: str,
) -> dict[str, Any]:
    return {
        "version": 1,
        "hardware": hardware,
        "models": models,
        "config_order": configs,
        "sweep_output_root": sweep_output_root,
        "completed": {},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def summary_counts(state: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in state.get("completed", {}).values():
        st = entry.get("status", "unknown")
        counts[st] = counts.get(st, 0) + 1
    return counts
