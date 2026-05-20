#!/usr/bin/env python3
"""Benchmark local LLM inference on Apple Silicon via MLX."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from benchmark.cli import main

if __name__ == "__main__":
    main()
