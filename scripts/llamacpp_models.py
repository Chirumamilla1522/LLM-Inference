"""GGUF model paths for llama.cpp runtime comparison (Article 10)."""

from __future__ import annotations

from dataclasses import dataclass

# Hugging Face repo + GGUF filename per preset and weight_bits (MLX-aligned labels).
# Repos: https://huggingface.co/bartowski (community GGUF quantizations)
LLAMACPP_GGUF: dict[str, dict[int, tuple[str, str] | None]] = {
    "llama3-8b": {
        16: (
            "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "Meta-Llama-3.1-8B-Instruct-f16.gguf",
        ),
        8: (
            "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        ),
        4: (
            "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        ),
        2: (
            "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "Meta-Llama-3.1-8B-Instruct-Q2_K.gguf",
        ),
    },
    "mistral-7b": {
        16: (
            "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
            "Mistral-7B-Instruct-v0.3-f16.gguf",
        ),
        8: (
            "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
            "Mistral-7B-Instruct-v0.3-Q8_0.gguf",
        ),
        4: (
            "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
            "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        ),
        2: None,
    },
    "qwen-7b": {
        16: (
            "bartowski/Qwen2.5-7B-Instruct-GGUF",
            "Qwen2.5-7B-Instruct-f16.gguf",
        ),
        8: (
            "bartowski/Qwen2.5-7B-Instruct-GGUF",
            "Qwen2.5-7B-Instruct-Q8_0.gguf",
        ),
        4: (
            "bartowski/Qwen2.5-7B-Instruct-GGUF",
            "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        ),
        2: None,
    },
}

# Default Article 10 comparison matrix (preset, config label)
RUNTIME_COMPARE_PAIRS: tuple[tuple[str, str], ...] = (
    ("llama3-8b", "fp16"),
    ("llama3-8b", "w4"),
    ("mistral-7b", "w4"),
)

# Map MLX weight_bits → GGUF quant family name (for articles)
GGUF_QUANT_NAME: dict[int, str] = {
    16: "F16",
    8: "Q8_0",
    4: "Q4_K_M",
    2: "Q2_K",
}


@dataclass(frozen=True)
class GGUFSpec:
    hf_repo: str
    filename: str
    quant_name: str

    @property
    def hf_file_id(self) -> str:
        return f"{self.hf_repo}/{self.filename}"


def resolve_gguf(preset: str, weight_bits: int) -> GGUFSpec | None:
    entry = LLAMACPP_GGUF.get(preset, {}).get(weight_bits)
    if entry is None:
        return None
    repo, filename = entry
    return GGUFSpec(
        hf_repo=repo,
        filename=filename,
        quant_name=GGUF_QUANT_NAME.get(weight_bits, f"{weight_bits}bit"),
    )


def gguf_cache_path(spec: GGUFSpec, cache_dir: str | None = None) -> "Path":
    from pathlib import Path

    base = Path(cache_dir or Path.home() / ".cache" / "llama-cpp-models")
    return base / spec.hf_repo.replace("/", "--") / spec.filename
