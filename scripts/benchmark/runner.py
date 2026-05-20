"""MLX inference benchmark execution."""

from __future__ import annotations

import json
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate

from benchmark.result import (
    BenchmarkResult,
    avg,
    package_version,
)
from benchmark.schema import (
    SCHEMA_VERSION,
    WARMUP_POLICY,
    build_stats_block,
    primary_metric,
)
from optimizations import (
    DEFAULT_NUM_DRAFT_TOKENS,
    OptimizationConfig,
    RunParams,
    check_hf_auth_for_repo,
    get_model_repos,
    resolve_draft_preset,
    resolve_draft_repo,
    resolve_run_params,
)
from workloads import WorkloadProfile, build_prompt_ids, get_workload


def run_trial(
    model,
    tokenizer,
    prompt: list[int],
    generation_tokens: int,
    params: RunParams,
    *,
    draft_model=None,
    num_draft_tokens: int = DEFAULT_NUM_DRAFT_TOKENS,
    prompt_cache=None,
) -> tuple[float, float, float, float, float | None]:
    tokenizer._eos_token_ids = {}
    mx.reset_peak_memory()

    ttft_ms = 0.0
    prompt_tps = 0.0
    generation_tps = 0.0
    peak_memory_gb = 0.0
    draft_accept_rate: float | None = None

    gen_kwargs: dict = {
        "max_tokens": generation_tokens,
        "prefill_step_size": params.prefill_step_size,
    }
    if params.kv_bits is not None:
        gen_kwargs["kv_bits"] = params.kv_bits
    if prompt_cache is not None:
        gen_kwargs["prompt_cache"] = prompt_cache
    if draft_model is not None:
        gen_kwargs["draft_model"] = draft_model
        gen_kwargs["num_draft_tokens"] = num_draft_tokens

    gen_count = 0
    from_draft_count = 0

    for response in stream_generate(model, tokenizer, prompt, **gen_kwargs):
        if response.generation_tokens == 1:
            ttft_ms = (response.prompt_tokens / response.prompt_tps) * 1000
            prompt_tps = response.prompt_tps
        generation_tps = response.generation_tps
        peak_memory_gb = response.peak_memory
        if draft_model is not None:
            gen_count += 1
            if response.from_draft:
                from_draft_count += 1

    if draft_model is not None and gen_count > 0:
        draft_accept_rate = from_draft_count / gen_count

    return ttft_ms, prompt_tps, generation_tps, peak_memory_gb, draft_accept_rate


def benchmark_prefix_cache(
    *,
    model_preset: str,
    config: OptimizationConfig,
    hardware: str,
    prompt_tokens: int,
    generation_tokens: int,
    num_trials: int,
    delay: int,
    seed: int,
    article_id: int | None = None,
    run_label: str | None = None,
) -> BenchmarkResult:
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache

    params = resolve_run_params(model_preset, config)
    if params.model_repo is None:
        raise ValueError(f"No model repo for {model_preset} at {config.weight_bits}-bit")

    auth_err = check_hf_auth_for_repo(params.model_repo)
    if auth_err:
        raise RuntimeError(auth_err)

    system_tokens = max(64, prompt_tokens // 2)
    user_tokens = max(32, prompt_tokens - system_tokens)

    print(f"\n{'=' * 60}")
    print(f"PREFIX CACHE  |  {model_preset}  |  {config.label}")
    print(f"system={system_tokens} user={user_tokens} tokens")
    print(f"{'=' * 60}")

    model, tokenizer = load(
        params.model_repo,
        tokenizer_config={"trust_remote_code": True},
    )
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer.get_vocab())
    mx.random.seed(seed)

    cold_ttft: list[float] = []
    warm_ttft: list[float] = []
    memory_samples: list[float] = []
    throughput_samples: list[float] = []

    gen_kwargs = {
        "prefill_step_size": params.prefill_step_size,
        "max_tokens": generation_tokens,
    }
    if params.kv_bits is not None:
        gen_kwargs["kv_bits"] = params.kv_bits

    for trial in range(1, num_trials + 1):
        if delay > 0:
            time.sleep(delay)

        mx.random.seed(seed + trial)
        system_ids = mx.random.randint(0, vocab_size, (system_tokens,)).tolist()
        user_ids = mx.random.randint(0, vocab_size, (user_tokens,)).tolist()
        full_ids = system_ids + user_ids

        mx.reset_peak_memory()
        cold_ms = 0.0
        for response in stream_generate(model, tokenizer, full_ids, **gen_kwargs):
            if response.generation_tokens == 1:
                cold_ms = (response.prompt_tokens / response.prompt_tps) * 1000
                break
        cold_ttft.append(cold_ms)

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            cache_path = tmp.name

        cache = make_prompt_cache(model)
        system_mx = mx.array(system_ids)
        for _ in generate_step(
            system_mx,
            model,
            prompt_cache=cache,
            max_tokens=1,
            prefill_step_size=params.prefill_step_size,
            kv_bits=params.kv_bits,
        ):
            pass
        save_prompt_cache(
            cache_path,
            cache,
            metadata={
                "model": params.model_repo,
                "tokenizer_config": json.dumps({"trust_remote_code": True}),
            },
        )

        loaded_cache, _meta = load_prompt_cache(cache_path, return_metadata=True)
        user_mx = mx.array(user_ids)
        mx.reset_peak_memory()
        warm_ms = 0.0
        gen_tps = 0.0
        peak_gb = 0.0
        for response in stream_generate(
            model,
            tokenizer,
            user_mx,
            prompt_cache=loaded_cache,
            **gen_kwargs,
        ):
            if response.generation_tokens == 1:
                warm_ms = (response.prompt_tokens / response.prompt_tps) * 1000
            gen_tps = response.generation_tps
            peak_gb = response.peak_memory

        warm_ttft.append(warm_ms)
        memory_samples.append(peak_gb)
        throughput_samples.append(gen_tps)
        Path(cache_path).unlink(missing_ok=True)

        print(
            f"  trial {trial}/{num_trials}: cold_ttft={cold_ttft[-1]:.1f} ms, "
            f"warm_ttft={warm_ms:.1f} ms"
        )

    del model

    return BenchmarkResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        hardware=hardware,
        model_preset=model_preset,
        configuration=config.label,
        weight_bits=config.weight_bits,
        optimizations=config.to_dict(),
        model_repo=params.model_repo,
        kv_bits=params.kv_bits,
        prefill_step_size=params.prefill_step_size,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        num_trials=num_trials,
        memory_gb=avg(memory_samples),
        ttft_ms=avg(warm_ttft),
        throughput_tps=avg(throughput_samples),
        prompt_tps=0.0,
        mlx_version=package_version("mlx"),
        mlx_lm_version=package_version("mlx_lm"),
        platform=platform.platform(),
        benchmark_mode="prefix_cache",
        article_id=article_id,
        run_label=run_label,
        prefix_cache_cold_ttft_ms=avg(cold_ttft),
        prefix_cache_warm_ttft_ms=avg(warm_ttft),
        prefix_system_tokens=system_tokens,
        schema_version=SCHEMA_VERSION,
        warmup_policy=WARMUP_POLICY,
    )


def benchmark_once(
    *,
    model_preset: str,
    config: OptimizationConfig,
    hardware: str,
    prompt_tokens: int,
    generation_tokens: int,
    num_trials: int,
    delay: int,
    seed: int,
    model_override: str | None = None,
    draft_preset: str | None = None,
    num_draft_tokens: int = DEFAULT_NUM_DRAFT_TOKENS,
    article_id: int | None = None,
    run_label: str | None = None,
    benchmark_mode: str = "standard",
    workload: WorkloadProfile | None = None,
) -> BenchmarkResult:
    params = resolve_run_params(model_preset, config)
    if model_override:
        params = RunParams(
            model_repo=model_override,
            kv_bits=params.kv_bits,
            prefill_step_size=params.prefill_step_size,
            config=config,
        )

    if params.model_repo is None:
        raise ValueError(
            f"No model repo for {model_preset} at {config.weight_bits}-bit. "
            f"Add one in models.json under overrides.{model_preset}.{config.weight_bits}"
        )

    auth_err = check_hf_auth_for_repo(params.model_repo)
    if auth_err:
        raise RuntimeError(auth_err)

    print(f"\n{'=' * 60}")
    print(f"Model: {model_preset}  |  Config: {config.label}")
    print(f"Repo:  {params.model_repo}")
    print(f"Opts:  {params.to_dict()}")
    print(f"{'=' * 60}")

    draft_model = None
    draft_repo: str | None = None
    effective_draft_preset = draft_preset or resolve_draft_preset(model_preset)
    if benchmark_mode == "speculative":
        draft_repo = resolve_draft_repo(model_preset, config.weight_bits)
        if draft_repo is None and effective_draft_preset:
            draft_repo = get_model_repos().get(effective_draft_preset, {}).get(
                config.weight_bits
            )
        if draft_repo is None:
            raise ValueError(
                f"No draft repo for {model_preset} at {config.weight_bits}-bit. "
                f"Set DRAFT_PRESET_BY_TARGET in optimizations.py."
            )
        auth_err = check_hf_auth_for_repo(draft_repo)
        if auth_err:
            raise RuntimeError(auth_err)
        print(f"Draft: {draft_repo} (num_draft_tokens={num_draft_tokens})")

    model, tokenizer = load(
        params.model_repo,
        tokenizer_config={"trust_remote_code": True},
    )
    if draft_repo:
        draft_model, draft_tokenizer = load(
            draft_repo,
            tokenizer_config={"trust_remote_code": True},
        )
        d_vocab = getattr(draft_tokenizer, "vocab_size", None) or len(
            draft_tokenizer.get_vocab()
        )
        t_vocab = getattr(tokenizer, "vocab_size", None) or len(tokenizer.get_vocab())
        if d_vocab != t_vocab:
            raise ValueError(
                f"Draft vocab ({d_vocab}) != target vocab ({t_vocab}). "
                "Pick a draft preset with a matching tokenizer."
            )

    if workload is not None and not workload.runnable:
        raise ValueError(
            f"Workload '{workload.id}' ({workload.modality.value}) is not runnable "
            "in the text mlx-lm harness yet."
        )

    effective_prompt_tokens = (
        workload.prompt_tokens if workload is not None else prompt_tokens
    )
    effective_generation_tokens = (
        workload.generation_tokens if workload is not None else generation_tokens
    )

    if workload is not None:
        print(
            f"Workload: {workload.id} (pressure={workload.pressure}, "
            f"task={workload.task.value}, data={workload.data_type.value}, "
            f"stress={workload.primary_stress.value})"
        )

    print("Warmup ...")
    warmup_prompt = build_prompt_ids(
        tokenizer,
        workload or get_workload("random_baseline"),
        seed=seed,
        prompt_tokens=effective_prompt_tokens,
    )
    run_trial(
        model,
        tokenizer,
        warmup_prompt,
        effective_generation_tokens,
        params,
        draft_model=draft_model,
        num_draft_tokens=num_draft_tokens,
    )

    ttft_samples: list[float] = []
    throughput_samples: list[float] = []
    prompt_tps_samples: list[float] = []
    memory_samples: list[float] = []
    accept_samples: list[float] = []

    for trial in range(1, num_trials + 1):
        if delay > 0:
            time.sleep(delay)
        trial_prompt = build_prompt_ids(
            tokenizer,
            workload or get_workload("random_baseline"),
            seed=seed + trial,
            prompt_tokens=effective_prompt_tokens,
        )
        ttft_ms, prompt_tps, generation_tps, peak_memory_gb, accept = run_trial(
            model,
            tokenizer,
            trial_prompt,
            effective_generation_tokens,
            params,
            draft_model=draft_model,
            num_draft_tokens=num_draft_tokens,
        )
        ttft_samples.append(ttft_ms)
        throughput_samples.append(generation_tps)
        prompt_tps_samples.append(prompt_tps)
        memory_samples.append(peak_memory_gb)
        if accept is not None:
            accept_samples.append(accept)
        extra = ""
        if accept is not None:
            extra = f", draft_accept={accept:.2%}"
        print(
            f"  trial {trial}/{num_trials}: "
            f"ttft={ttft_ms:.1f} ms, throughput={generation_tps:.1f} t/s, "
            f"memory={peak_memory_gb:.2f} GB{extra}"
        )

    del model
    if draft_model is not None:
        del draft_model

    config_label = config.label
    if benchmark_mode == "speculative":
        config_label = f"{config.label}+speculative"

    trials_block = {
        "ttft_ms": ttft_samples,
        "throughput_tps": throughput_samples,
        "memory_gb": memory_samples,
        "prompt_tps": prompt_tps_samples,
    }
    if accept_samples:
        trials_block["draft_accept_rate"] = accept_samples

    stats_block = build_stats_block(
        ttft_ms=ttft_samples,
        throughput_tps=throughput_samples,
        memory_gb=memory_samples,
        prompt_tps=prompt_tps_samples,
        draft_accept_rate=accept_samples or None,
    )
    mem_stats = stats_block.get("memory_gb", {})
    memory_primary = float(mem_stats.get("max", mem_stats.get("median", 0.0)))

    return BenchmarkResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        hardware=hardware,
        model_preset=model_preset,
        configuration=config_label,
        weight_bits=config.weight_bits,
        optimizations=config.to_dict(),
        model_repo=params.model_repo,
        kv_bits=params.kv_bits,
        prefill_step_size=params.prefill_step_size,
        prompt_tokens=effective_prompt_tokens,
        generation_tokens=effective_generation_tokens,
        num_trials=num_trials,
        memory_gb=memory_primary,
        ttft_ms=primary_metric(stats_block.get("ttft_ms")),
        throughput_tps=primary_metric(stats_block.get("throughput_tps")),
        prompt_tps=primary_metric(stats_block.get("prompt_tps")),
        mlx_version=package_version("mlx"),
        mlx_lm_version=package_version("mlx_lm"),
        platform=platform.platform(),
        article_id=article_id,
        run_label=run_label,
        benchmark_mode=benchmark_mode,
        draft_model_repo=draft_repo,
        draft_preset=effective_draft_preset,
        num_draft_tokens=num_draft_tokens if draft_repo else None,
        draft_accept_rate=primary_metric(stats_block.get("draft_accept_rate"))
        if accept_samples
        else None,
        workload=workload.to_metadata() if workload else None,
        schema_version=SCHEMA_VERSION,
        trials=trials_block,
        stats=stats_block,
        warmup_policy=WARMUP_POLICY,
    )
