"""
Workload / stress profiles for benchmarks.

Dimensions:
  - task: what the workload simulates (chat, qa, summarize, complete, rag, stress)
  - data_type: input shape (random_ids, prose, json, code, repetitive, multilingual)
  - modality: text today; vision/audio reserved for future MLX multimodal models
  - pressure: 1 (light) → 5 (heavy) — combined prompt+gen length and stress axis
  - primary_stress: prefill | decode | memory | balanced
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts"


class Task(str, Enum):
    CHAT = "chat"
    QA = "qa"
    SUMMARIZE = "summarize"
    COMPLETE = "complete"
    RAG = "rag_agent"
    STRESS = "stress"
    BASELINE = "baseline"


class DataType(str, Enum):
    RANDOM_IDS = "random_ids"
    PROSE = "prose"
    JSON = "json"
    CODE = "code"
    REPETITIVE = "repetitive"
    MULTILINGUAL = "multilingual"
    MIXED = "mixed"  # json + prose


class Modality(str, Enum):
    TEXT = "text"
    VISION = "vision"  # placeholder — not runnable in mlx-lm text harness
    AUDIO = "audio"  # placeholder


class PrimaryStress(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    MEMORY = "memory"
    BALANCED = "balanced"


@dataclass(frozen=True)
class WorkloadProfile:
    id: str
    task: Task
    data_type: DataType
    modality: Modality
    pressure: int  # 1–5
    prompt_tokens: int
    generation_tokens: int
    primary_stress: PrimaryStress
    description: str
    runnable: bool = True

    def to_metadata(self) -> dict[str, Any]:
        return {
            "workload_id": self.id,
            "workload_task": self.task.value,
            "workload_data_type": self.data_type.value,
            "workload_modality": self.modality.value,
            "workload_pressure": self.pressure,
            "workload_primary_stress": self.primary_stress.value,
            "workload_description": self.description,
        }


# Low → high pressure ladder (text modality, runnable on mlx-lm).
WORKLOAD_PROFILES: dict[str, WorkloadProfile] = {
    "random_baseline": WorkloadProfile(
        id="random_baseline",
        task=Task.BASELINE,
        data_type=DataType.RANDOM_IDS,
        modality=Modality.TEXT,
        pressure=1,
        prompt_tokens=512,
        generation_tokens=128,
        primary_stress=PrimaryStress.BALANCED,
        description="Legacy harness: uniform random token IDs (config A/B only).",
    ),
    "chat_light": WorkloadProfile(
        id="chat_light",
        task=Task.CHAT,
        data_type=DataType.PROSE,
        modality=Modality.TEXT,
        pressure=1,
        prompt_tokens=128,
        generation_tokens=64,
        primary_stress=PrimaryStress.DECODE,
        description="Short chat turn — low prefill, light decode.",
    ),
    "chat_standard": WorkloadProfile(
        id="chat_standard",
        task=Task.CHAT,
        data_type=DataType.PROSE,
        modality=Modality.TEXT,
        pressure=2,
        prompt_tokens=512,
        generation_tokens=128,
        primary_stress=PrimaryStress.BALANCED,
        description="Typical instruct chat — default-shaped prompt.",
    ),
    "qa_json": WorkloadProfile(
        id="qa_json",
        task=Task.QA,
        data_type=DataType.JSON,
        modality=Modality.TEXT,
        pressure=2,
        prompt_tokens=512,
        generation_tokens=256,
        primary_stress=PrimaryStress.BALANCED,
        description="Structured JSON context + question — tokenizer punctuation stress.",
    ),
    "summarize_long": WorkloadProfile(
        id="summarize_long",
        task=Task.SUMMARIZE,
        data_type=DataType.PROSE,
        modality=Modality.TEXT,
        pressure=3,
        prompt_tokens=2048,
        generation_tokens=128,
        primary_stress=PrimaryStress.PREFILL,
        description="Long document in, short summary out — TTFT / prefill heavy.",
    ),
    "complete_code": WorkloadProfile(
        id="complete_code",
        task=Task.COMPLETE,
        data_type=DataType.CODE,
        modality=Modality.TEXT,
        pressure=3,
        prompt_tokens=256,
        generation_tokens=512,
        primary_stress=PrimaryStress.DECODE,
        description="Code prefix, long completion — sustained decode throughput.",
    ),
    "rag_agent": WorkloadProfile(
        id="rag_agent",
        task=Task.RAG,
        data_type=DataType.MIXED,
        modality=Modality.TEXT,
        pressure=4,
        prompt_tokens=4096,
        generation_tokens=256,
        primary_stress=PrimaryStress.MEMORY,
        description="RAG-style long context + medium generation — memory bandwidth.",
    ),
    "stress_prefill": WorkloadProfile(
        id="stress_prefill",
        task=Task.STRESS,
        data_type=DataType.REPETITIVE,
        modality=Modality.TEXT,
        pressure=5,
        prompt_tokens=4096,
        generation_tokens=32,
        primary_stress=PrimaryStress.PREFILL,
        description="Max prefill pressure — highly repetitive tokens (compressible pattern).",
    ),
    "stress_decode": WorkloadProfile(
        id="stress_decode",
        task=Task.STRESS,
        data_type=DataType.PROSE,
        modality=Modality.TEXT,
        pressure=5,
        prompt_tokens=64,
        generation_tokens=1024,
        primary_stress=PrimaryStress.DECODE,
        description="Max decode pressure — tiny prompt, very long generation.",
    ),
    "tokenizer_multilingual": WorkloadProfile(
        id="tokenizer_multilingual",
        task=Task.STRESS,
        data_type=DataType.MULTILINGUAL,
        modality=Modality.TEXT,
        pressure=4,
        prompt_tokens=1024,
        generation_tokens=128,
        primary_stress=PrimaryStress.PREFILL,
        description="Mixed scripts and symbols — tokenizer / embedding stress.",
    ),
    # Future modality placeholders (documented, not executed).
    "vision_placeholder": WorkloadProfile(
        id="vision_placeholder",
        task=Task.STRESS,
        data_type=DataType.PROSE,
        modality=Modality.VISION,
        pressure=3,
        prompt_tokens=512,
        generation_tokens=128,
        primary_stress=PrimaryStress.BALANCED,
        description="Reserved: image+text VLM — requires multimodal mlx-lm model.",
        runnable=False,
    ),
    "audio_placeholder": WorkloadProfile(
        id="audio_placeholder",
        task=Task.STRESS,
        data_type=DataType.PROSE,
        modality=Modality.AUDIO,
        pressure=3,
        prompt_tokens=512,
        generation_tokens=128,
        primary_stress=PrimaryStress.BALANCED,
        description="Reserved: audio+text — requires speech multimodal stack.",
        runnable=False,
    ),
}

# Default ladder for --workload-sweep (excludes placeholders and duplicate baseline).
WORKLOAD_SWEEP_ORDER: tuple[str, ...] = (
    "chat_light",
    "chat_standard",
    "qa_json",
    "summarize_long",
    "complete_code",
    "rag_agent",
    "stress_prefill",
    "stress_decode",
    "tokenizer_multilingual",
)

# Subset wired into Article 7 (context article) — workload dimension.
ARTICLE_07_WORKLOADS: tuple[str, ...] = (
    "chat_light",
    "chat_standard",
    "summarize_long",
    "complete_code",
    "rag_agent",
    "random_baseline",
)


def get_workload(workload_id: str) -> WorkloadProfile:
    if workload_id not in WORKLOAD_PROFILES:
        known = ", ".join(sorted(WORKLOAD_PROFILES))
        raise ValueError(f"Unknown workload '{workload_id}'. Known: {known}")
    return WORKLOAD_PROFILES[workload_id]


def iter_workloads(
    *,
    sweep: bool = False,
    include_non_runnable: bool = False,
) -> Iterator[WorkloadProfile]:
    order = WORKLOAD_SWEEP_ORDER if sweep else tuple(WORKLOAD_PROFILES.keys())
    seen: set[str] = set()
    for wid in order:
        if wid in seen:
            continue
        seen.add(wid)
        profile = WORKLOAD_PROFILES[wid]
        if not profile.runnable and not include_non_runnable:
            continue
        yield profile
    if sweep:
        return
    for wid, profile in WORKLOAD_PROFILES.items():
        if wid in seen:
            continue
        if not profile.runnable and not include_non_runnable:
            continue
        yield profile


def _read_fixture(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt fixture: {path}")
    return path.read_text()


def _materialize_text(profile: WorkloadProfile, seed: int) -> str:
    if profile.data_type == DataType.RANDOM_IDS:
        return ""

    if profile.data_type == DataType.PROSE:
        base = _read_fixture("prose.txt")
    elif profile.data_type == DataType.CODE:
        base = _read_fixture("code.py")
    elif profile.data_type == DataType.JSON:
        base = _read_fixture("json_template.json")
    elif profile.data_type == DataType.MULTILINGUAL:
        base = _read_fixture("multilingual.txt")
    elif profile.data_type == DataType.REPETITIVE:
        base = "token " * 64
    elif profile.data_type == DataType.MIXED:
        base = _read_fixture("json_template.json") + "\n\n" + _read_fixture("prose.txt")
    else:
        base = _read_fixture("prose.txt")

    # Slight per-trial variation without changing length much.
    return f"{base}\n<!-- seed={seed} trial={seed % 997} -->"


def _fit_token_length(token_ids: list[int], target: int) -> list[int]:
    if target <= 0:
        return []
    if not token_ids:
        return [0] * target
    if len(token_ids) >= target:
        return token_ids[:target]
    out = list(token_ids)
    while len(out) < target:
        need = target - len(out)
        out.extend(token_ids[: min(need, len(token_ids))])
    return out[:target]


def build_prompt_ids(
    tokenizer: Any,
    profile: WorkloadProfile,
    *,
    seed: int,
    prompt_tokens: int | None = None,
) -> list[int]:
    """Build prompt token IDs for a workload profile."""
    target = prompt_tokens if prompt_tokens is not None else profile.prompt_tokens

    if profile.data_type == DataType.RANDOM_IDS:
        import mlx.core as mx

        vocab_size = getattr(tokenizer, "vocab_size", None) or len(
            tokenizer.get_vocab()
        )
        mx.random.seed(seed)
        return mx.random.randint(0, vocab_size, (target,)).tolist()

    text = _materialize_text(profile, seed)
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text)
    else:
        ids = list(tokenizer(text))

    if isinstance(ids, dict):
        ids = ids.get("input_ids", [])
    return _fit_token_length(list(ids), target)


def print_workload_table() -> None:
    print("| ID | Pressure | Task | Data | Modality | Stress | p | g |")
    print("|----|----------|------|------|----------|--------|---|---|")
    for p in sorted(WORKLOAD_PROFILES.values(), key=lambda x: (x.pressure, x.id)):
        run = "yes" if p.runnable else "—"
        print(
            f"| {p.id} | {p.pressure} | {p.task.value} | {p.data_type.value} | "
            f"{p.modality.value} | {p.primary_stress.value} | "
            f"{p.prompt_tokens} | {p.generation_tokens} | ({run})"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="List workload stress profiles")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()
    if args.json:
        data = {k: asdict(v) for k, v in WORKLOAD_PROFILES.items()}
        print(json.dumps(data, indent=2, default=str))
    else:
        print_workload_table()
