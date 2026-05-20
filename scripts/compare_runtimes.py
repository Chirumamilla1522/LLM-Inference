#!/usr/bin/env python3
"""
Compare MLX (this repo) vs llama.cpp on the same preset / quant level.

Article 10 — writes results/Mac_<hardware>/article_10_runtimes/<preset>/<config>_compare.json

Requires llama.cpp on PATH: llama-bench (or llama-bench-blas).
Install: brew install llama.cpp   OR   build from https://github.com/ggml-org/llama.cpp
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from benchmark_schema import SCHEMA_VERSION, parse_llama_bench_output
from articles import get_article
from llamacpp_models import (
    GGUFSpec,
    RUNTIME_COMPARE_PAIRS,
    gguf_cache_path,
    resolve_gguf,
)
from optimizations import OptimizationConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RUN_BENCHMARK = ROOT / "scripts" / "run_benchmark.py"


def _article_output_dir(hardware: str, article) -> Path:
    safe = hardware.replace(" ", "_").replace("/", "-")
    return RESULTS_DIR / safe / article.dir_name


@dataclass
class LlamaCppResult:
    binary: str
    gguf_path: str
    hf_file: str
    quant_name: str
    prompt_tokens: int
    generation_tokens: int
    pp_tps: float | None = None
    tg_tps: float | None = None
    ngl: int = 99
    status: str = "ok"
    error: str | None = None
    raw_output: str | None = None


def find_llama_bench() -> str | None:
    for name in ("llama-bench", "llama-bench-blas"):
        path = shutil.which(name)
        if path:
            return path
    return None


def ensure_gguf(spec: GGUFSpec, cache_dir: Path | None) -> Path:
    """Download GGUF via huggingface_hub if missing."""
    path = gguf_cache_path(spec, str(cache_dir) if cache_dir else None)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=spec.hf_repo,
            filename=spec.filename,
            local_dir=path.parent,
            local_dir_use_symlinks=False,
        )
        return Path(downloaded)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {spec.hf_file_id}. "
            f"Install huggingface_hub and run: hf download {spec.hf_repo} "
            f"{spec.filename} --local-dir {path.parent}\n{exc}"
        ) from exc


def run_llama_bench(
    spec: GGUFSpec,
    gguf_path: Path,
    *,
    prompt_tokens: int,
    generation_tokens: int,
    ngl: int,
    threads: int,
) -> LlamaCppResult:
    binary = find_llama_bench()
    if binary is None:
        return LlamaCppResult(
            binary="",
            gguf_path=str(gguf_path),
            hf_file=spec.hf_file_id,
            quant_name=spec.quant_name,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            status="skipped",
            error="llama-bench not found. Install: brew install llama.cpp",
        )

    cmd = [
        binary,
        "-m",
        str(gguf_path),
        "-p",
        str(prompt_tokens),
        "-n",
        str(generation_tokens),
        "-ngl",
        str(ngl),
        "-t",
        str(threads),
        "-fa",  # flash attention when supported
    ]
    print("$", " ".join(cmd))
    proc = None
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return LlamaCppResult(
            binary=binary,
            gguf_path=str(gguf_path),
            hf_file=spec.hf_file_id,
            quant_name=spec.quant_name,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            status="error",
            error="llama-bench timed out",
        )

    pp_tps, tg_tps = parse_llama_bench_output(out)
    exit_code = proc.returncode if proc else -1
    status = "ok" if exit_code == 0 and tg_tps else "error"
    return LlamaCppResult(
        binary=binary,
        gguf_path=str(gguf_path),
        hf_file=spec.hf_file_id,
        quant_name=spec.quant_name,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        pp_tps=pp_tps,
        tg_tps=tg_tps,
        ngl=ngl,
        status=status,
        error=None if status == "ok" else f"exit {exit_code}",
        raw_output=out[-4000:] if out else None,
    )


def run_mlx(
    preset: str,
    config: OptimizationConfig,
    hardware: str,
    output_json: Path,
    args: argparse.Namespace,
) -> dict | None:
    cmd = [
        sys.executable,
        str(RUN_BENCHMARK),
        "--preset",
        preset,
        "--config",
        config.label,
        "--hardware",
        hardware,
        "-n",
        str(args.num_trials),
        "-p",
        str(args.prompt_tokens),
        "-g",
        str(args.generation_tokens),
        "-o",
        str(output_json),
    ]
    print("$", " ".join(cmd))
    code = subprocess.run(cmd, cwd=ROOT).returncode
    if not output_json.exists():
        return {"status": "error", "error": f"MLX run failed exit {code}"}
    return json.loads(output_json.read_text())


def compare_one(
    preset: str,
    config_label: str,
    hardware: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict:
    config = OptimizationConfig.from_label(config_label)
    spec = resolve_gguf(preset, config.weight_bits)

    mlx_path = out_dir / preset / f"{config_label}_mlx.json"
    mlx_path.parent.mkdir(parents=True, exist_ok=True)
    mlx_data = run_mlx(preset, config, hardware, mlx_path, args)

    llamacpp_data: dict | None = None
    if spec is not None:
        try:
            gguf_path = ensure_gguf(spec, args.gguf_cache)
            lc = run_llama_bench(
                spec,
                gguf_path,
                prompt_tokens=args.prompt_tokens,
                generation_tokens=args.generation_tokens,
                ngl=args.ngl,
                threads=args.threads,
            )
            llamacpp_data = asdict(lc)
        except Exception as exc:
            llamacpp_data = {
                "status": "error",
                "error": str(exc),
                "hf_file": spec.hf_file_id,
            }
    else:
        llamacpp_data = {
            "status": "skipped",
            "error": f"No GGUF mapping for {preset} at {config.weight_bits}-bit",
        }

    comparison: dict = {}
    if mlx_data and llamacpp_data:
        mlx_tps = mlx_data.get("throughput_tps")
        lc_tps = llamacpp_data.get("tg_tps")
        if mlx_tps and lc_tps and lc_tps > 0:
            comparison["throughput_ratio_mlx_over_llamacpp"] = mlx_tps / lc_tps
        mlx_ttft = mlx_data.get("ttft_ms")
        lc_pp = llamacpp_data.get("pp_tps")
        if mlx_ttft and lc_pp and lc_pp > 0:
            # rough: MLX ttft ms vs llama pp tokens/s → not identical but indicative
            comparison["note"] = (
                "MLX ttft_ms vs llama.cpp pp_tps measure different things; "
                "compare tg_tps to throughput_tps for decode."
            )

    record = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "article_id": 10,
        "hardware": hardware,
        "model_preset": preset,
        "configuration": config_label,
        "weight_bits": config.weight_bits,
        "prompt_tokens": args.prompt_tokens,
        "generation_tokens": args.generation_tokens,
        "runtime": "mlx_vs_llamacpp",
        "mlx": mlx_data,
        "llamacpp": llamacpp_data,
        "comparison": comparison,
    }
    out_path = out_dir / preset / f"{config_label}_compare.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved: {out_path}")
    return record


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MLX vs llama.cpp comparison (Article 10)")
    p.add_argument("--hardware", default="Mac M3")
    p.add_argument("--preset", help="Single preset (default: article 10 matrix)")
    p.add_argument("--config", help="Single config label, e.g. w4")
    p.add_argument("-n", "--num-trials", type=int, default=3)
    p.add_argument("-p", "--prompt-tokens", type=int, default=512)
    p.add_argument("-g", "--generation-tokens", type=int, default=128)
    p.add_argument("--ngl", type=int, default=99, help="GPU layers for llama.cpp (-ngl)")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument(
        "--gguf-cache",
        type=Path,
        default=None,
        help="Directory for GGUF downloads",
    )
    p.add_argument("--mlx-only", action="store_true", help="Skip llama.cpp")
    p.add_argument(
        "--with-server",
        action="store_true",
        help="Also run llama-server HTTP benchmark (spawns server per pair).",
    )
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    article = get_article(10)
    out_dir = _article_output_dir(args.hardware, article)

    pairs = RUNTIME_COMPARE_PAIRS
    if args.preset and args.config:
        pairs = ((args.preset, args.config),)

    if args.dry_run:
        print(f"Output: {out_dir}")
        for preset, cfg in pairs:
            c = OptimizationConfig.from_label(cfg)
            g = resolve_gguf(preset, c.weight_bits)
            print(f"  {preset} {cfg}: MLX + {g.hf_file_id if g else 'no GGUF'}")
        print(f"llama-bench: {find_llama_bench() or 'NOT FOUND'}")
        if args.with_server:
            from benchmark.llama_server import find_llama_server

            print(f"llama-server: {find_llama_server() or 'NOT FOUND'}")
        return

    if args.mlx_only:
        for preset, cfg in pairs:
            config = OptimizationConfig.from_label(cfg)
            mlx_path = out_dir / preset / f"{cfg}_mlx.json"
            run_mlx(preset, config, args.hardware, mlx_path, args)
        return

    failures = 0
    for preset, cfg in pairs:
        rec = compare_one(preset, cfg, args.hardware, out_dir, args)
        if rec.get("mlx", {}).get("status") != "ok" or rec.get("llamacpp", {}).get(
            "status"
        ) not in ("ok", "skipped"):
            failures += 1
        if args.with_server and (
            spec := resolve_gguf(
                preset, OptimizationConfig.from_label(cfg).weight_bits
            )
        ):
            try:
                from benchmark.llama_server import run_server_benchmark

                gguf_path = ensure_gguf(spec, args.gguf_cache)
                server_out = out_dir / preset / f"{cfg}_server.json"
                run_server_benchmark(
                    gguf_path=gguf_path,
                    hardware=args.hardware,
                    model_preset=preset,
                    config_label=cfg,
                    prompt="Summarize MLX vs llama.cpp for local inference.",
                    max_tokens=args.generation_tokens,
                    output_path=server_out,
                )
                print(f"Saved: {server_out}")
            except Exception as exc:
                print(f"Server bench failed {preset}/{cfg}: {exc}")
                failures += 1

    summary = {
        "article_id": 10,
        "hardware": args.hardware,
        "pairs": [f"{p}:{c}" for p, c in pairs],
        "llama_bench": find_llama_bench(),
        "failures": failures,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "article_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
