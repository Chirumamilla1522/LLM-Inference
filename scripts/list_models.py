#!/usr/bin/env python3
"""Print all model presets, sizes, and RAM requirements."""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[0]))

from optimizations import (  # noqa: E402
    MIN_RAM_GB_BY_PRESET,
    MODEL_PARAM_LABELS,
    requires_large_machine,
    sort_presets,
    get_model_repos,
)

repos = get_model_repos()
print(f"{'Preset':<22} {'Size':<10} {'Min RAM':>8}  {'Large?':<6}  fp16 repo")
print("-" * 90)
for preset in sort_presets(list(repos)):
    min_ram = MIN_RAM_GB_BY_PRESET.get(preset, 0)
    ram = f"{min_ram:.0f} GB" if min_ram else "any"
    large = "yes" if requires_large_machine(preset) else "no"
    fp16 = repos[preset].get(16) or "—"
    print(
        f"{preset:<22} {MODEL_PARAM_LABELS.get(preset, '?'):<10} {ram:>8}  "
        f"{large:<6}  {fp16}"
    )
print(f"\nTotal: {len(repos)} presets × 16 configs = {len(repos) * 16} sweep runs")
