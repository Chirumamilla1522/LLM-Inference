"""HTTP benchmarks against llama-server (OpenAI-compatible API)."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.schema import SCHEMA_VERSION


@dataclass
class ServerBenchResult:
    status: str
    ttft_ms: float | None = None
    total_ms: float | None = None
    tokens_generated: int | None = None
    decode_tps: float | None = None
    error: str | None = None
    server_url: str = "http://127.0.0.1:8080"
    llama_server_version: str | None = None


def find_llama_server() -> str | None:
    return shutil.which("llama-server")


def wait_for_port(host: str, port: int, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def bench_chat_completion(
    *,
    base_url: str,
    prompt: str,
    max_tokens: int = 128,
    timeout: float = 600.0,
) -> ServerBenchResult:
    """POST /v1/chat/completions with stream=true; measure TTFT and decode."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    ttft_ms: float | None = None
    tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (
                    chunk.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content")
                )
                if delta:
                    tokens += 1
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
    except urllib.error.URLError as exc:
        return ServerBenchResult(
            status="error",
            error=str(exc),
            server_url=base_url,
        )

    total_ms = (time.perf_counter() - t0) * 1000
    decode_window_ms = total_ms - (ttft_ms or 0)
    decode_tps = None
    if tokens > 0 and decode_window_ms > 0:
        decode_tps = tokens / (decode_window_ms / 1000)

    return ServerBenchResult(
        status="ok" if tokens > 0 else "error",
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        tokens_generated=tokens,
        decode_tps=decode_tps,
        error=None if tokens > 0 else "no tokens in stream",
        server_url=base_url,
    )


def start_llama_server(
    *,
    gguf_path: Path,
    host: str = "127.0.0.1",
    port: int = 8080,
    ngl: int = 99,
    ctx_size: int = 8192,
) -> subprocess.Popen:
    binary = find_llama_server()
    if binary is None:
        raise RuntimeError("llama-server not found. Install: brew install llama.cpp")

    cmd = [
        binary,
        "-m",
        str(gguf_path),
        "--host",
        host,
        "--port",
        str(port),
        "-ngl",
        str(ngl),
        "-c",
        str(ctx_size),
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def run_server_benchmark(
    *,
    gguf_path: Path,
    hardware: str,
    model_preset: str,
    config_label: str,
    prompt: str,
    max_tokens: int,
    output_path: Path,
    port: int = 8080,
    manage_process: bool = True,
) -> dict[str, Any]:
    """Start server (optional), run HTTP bench, write JSON."""
    proc: subprocess.Popen | None = None
    base_url = f"http://127.0.0.1:{port}"
    try:
        if manage_process:
            proc = start_llama_server(gguf_path=gguf_path, port=port)
            if not wait_for_port("127.0.0.1", port):
                raise RuntimeError("llama-server did not open port in time")

        result = bench_chat_completion(
            base_url=base_url,
            prompt=prompt,
            max_tokens=max_tokens,
        )
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    record = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
        "model_preset": model_preset,
        "configuration": config_label,
        "benchmark_mode": "llama_server_http",
        "gguf_path": str(gguf_path),
        "server": asdict(result),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n")
    return record
