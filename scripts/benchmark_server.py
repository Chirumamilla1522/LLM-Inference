#!/usr/bin/env python3
"""HTTP benchmark via llama-server (Article 10 serving path)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from benchmark.llama_server import find_llama_server, run_server_benchmark
from benchmark.paths import RESULTS_DIR
from compare_runtimes import ensure_gguf
from llamacpp_models import resolve_gguf
from optimizations import OptimizationConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="llama-server HTTP benchmark")
    parser.add_argument("--hardware", default="Mac M3")
    parser.add_argument("--preset", default="llama3-8b")
    parser.add_argument("--config", default="w4")
    parser.add_argument("-g", "--max-tokens", type=int, default=128)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--url",
        help="Use existing server base URL (skip spawn), e.g. http://127.0.0.1:8080",
    )
    parser.add_argument(
        "--prompt",
        default="Explain KV cache quantization for local LLM inference in two sentences.",
    )
    args = parser.parse_args()

    if args.url is None and find_llama_server() is None:
        print("llama-server not found. brew install llama.cpp")
        return 1

    config = OptimizationConfig.from_label(args.config)
    spec = resolve_gguf(args.preset, config.weight_bits)
    if spec is None:
        print(f"No GGUF for {args.preset} @ {config.weight_bits}-bit")
        return 1

    gguf = ensure_gguf(spec, None)
    safe = args.hardware.replace(" ", "_").replace("/", "-")
    out = (
        RESULTS_DIR
        / safe
        / "article_10_runtimes"
        / args.preset
        / f"{config.label}_server.json"
    )

    if args.url:
        import json
        from dataclasses import asdict
        from datetime import datetime, timezone

        from benchmark.llama_server import bench_chat_completion
        from benchmark.schema import SCHEMA_VERSION

        result = bench_chat_completion(
            base_url=args.url,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
        )
        record = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hardware": args.hardware,
            "model_preset": args.preset,
            "configuration": config.label,
            "benchmark_mode": "llama_server_http",
            "server": asdict(result),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Saved: {out}")
        return 0 if result.status == "ok" else 1

    run_server_benchmark(
        gguf_path=gguf,
        hardware=args.hardware,
        model_preset=args.preset,
        config_label=config.label,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        output_path=out,
        port=args.port,
    )
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
