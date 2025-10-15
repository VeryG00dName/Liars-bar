"""Comprehensive benchmarking utility for PPO reactive models.
This script explores a wide range of execution configurations (batch size,
sequence length, compilation mode, determinism and AMP) to surface the fastest
setup for the current GPU.  The heavy lifting happens in
``BenchmarkConfig``-driven sweeps that build fresh model instances and measure
their throughput with adaptive warmup logic to ensure stable timing numbers.
The script now runs in a manager/worker arrangement so it can reliably set
environment variables such as ``CUBLAS_WORKSPACE_CONFIG`` for highly
deterministic runs.  The top-level process enumerates benchmark configurations
and launches a fresh subprocess for each, while workers execute exactly one
configuration and append their results to a shared CSV file.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import os
import statistics
import logging
import subprocess
import time
import warnings
from argparse import Namespace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Optional, Tuple
import sys
import pandas as pd
import torch
from src.model.ppo_reactive_model import PPOReactiveModel
from src.model.ppo_reactive_model_script import PPOReactiveModelScript
DEVICE = torch.device("cuda")
# Suppress specific FutureWarning from torch.utils.checkpoint about CPU AMP autocast
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"torch\.utils\.checkpoint",
)
# Suppress the common TF32 suggestion spam from Inductor when determinism is enabled
# (we explicitly control TF32 below based on the determinism policy).
warnings.filterwarnings(
    "ignore",
    message=r"TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled\.",
    module=r"torch\._inductor\.compile_fx",
)
# Quiet excessively noisy symbolic shape warnings emitted during compilation.
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)
# Default architecture hyper-parameters for the PPO reactive model family.
DEFAULT_OBS_DIM = 9
DEFAULT_ACTION_DIM = 7
DEFAULT_NUM_AGENT_TYPES = 4
DEFAULT_NUM_EXPERTS = 8
DEFAULT_TOP_K = 2
DEFAULT_MAX_SEQ_LENGTH = 480
DEFAULT_NUM_LAYERS = 2
DEFAULT_NUM_HEADS = 4
DEFAULT_HIDDEN_DIM = 256
DEFAULT_DROPOUT = 0.1
@dataclass(frozen=True)
class BenchmarkConfig:
    """Container for a single benchmark configuration."""
    mode: str  # "eager", "compile", "script"
    batch_size: int
    seq_len: int
    use_amp: bool
    determinism_level: str  # "none", "high", "full"
    # Execution/memory features
    use_gradient_checkpointing: bool = False
    # Model hyper-parameters (kept mutable for experimentation).
    obs_dim: int = DEFAULT_OBS_DIM
    action_dim: int = DEFAULT_ACTION_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_layers: int = DEFAULT_NUM_LAYERS
    num_heads: int = DEFAULT_NUM_HEADS
    dropout_rate: float = DEFAULT_DROPOUT
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
    num_agent_types: int = DEFAULT_NUM_AGENT_TYPES
    num_experts: int = DEFAULT_NUM_EXPERTS
    top_k: int = DEFAULT_TOP_K
def _get_autocast_context():
    """Return the autocast context compatible with the installed torch."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast  # type: ignore[attr-defined]
    return torch.cuda.amp.autocast  # type: ignore[return-value]
autocast_context = _get_autocast_context()
def apply_determinism_settings(level: str) -> Dict[str, Any]:
    """Apply global determinism knobs and return the resulting state."""
    lvl = level.lower()
    if lvl not in {"none", "high", "full"}:
        raise ValueError(f"Unknown determinism level: {level}")

    # --- Reset to permissive, fast defaults ---
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Prefer TF32 for speed by default (Ampere+). Use only new APIs.
    if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
        torch.backends.cudnn.conv.fp32_precision = "tf32"

    # Enable all SDP backends; PyTorch will choose best available.
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)

    # --- Stricter modes ---
    if lvl == "high":
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False

        # Enforce IEEE fp32 for both matmul and conv via new APIs.
        if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
        if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
            torch.backends.cudnn.conv.fp32_precision = "ieee"

        # Prefer deterministic-friendly attention paths.
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    elif lvl == "full":
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Enforce IEEE fp32 for both matmul and conv via new APIs.
        if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
        if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
            torch.backends.cudnn.conv.fp32_precision = "ieee"

        # Disable non-deterministic SDP kernels.
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    return capture_backend_state()
def capture_backend_state() -> Dict[str, Any]:
    """Snapshot key Torch backend flags for reporting/debugging (new APIs only)."""
    state: Dict[str, Any] = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }

    # New TF32 controls (safe to read)
    if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        state["cuda_matmul_fp32_precision"] = torch.backends.cuda.matmul.fp32_precision
    if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
        state["cudnn_conv_fp32_precision"] = torch.backends.cudnn.conv.fp32_precision

    # Optional: global matmul precision policy (not deprecated)
    if hasattr(torch, "get_float32_matmul_precision"):
        try:
            state["torch_float32_matmul_precision"] = torch.get_float32_matmul_precision()
        except Exception:
            state["torch_float32_matmul_precision"] = None

    # SDP backend switches (safe)
    if hasattr(torch.backends.cuda, "flash_sdp_enabled"):
        state["flash_sdp"] = torch.backends.cuda.flash_sdp_enabled()
    if hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled"):
        state["mem_efficient_sdp"] = torch.backends.cuda.mem_efficient_sdp_enabled()
    if hasattr(torch.backends.cuda, "math_sdp_enabled"):
        state["math_sdp"] = torch.backends.cuda.math_sdp_enabled()

    return state
def generate_fixtures(config: BenchmarkConfig, *, seed: int = 2024) -> Dict[str, torch.Tensor]:
    """Create deterministic synthetic inputs for the requested batch/seq length."""
    if config.seq_len > config.max_seq_length:
        raise ValueError(
            f"Sequence length {config.seq_len} exceeds model max_seq_length {config.max_seq_length}"
        )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + config.batch_size * 37 + config.seq_len * 101)
    obs = torch.randn(
        config.batch_size,
        config.seq_len,
        config.obs_dim,
        dtype=torch.float32,
        generator=gen,
    )
    action_sequence = torch.randint(
        0,
        config.action_dim,
        (config.batch_size, config.seq_len),
        generator=gen,
        dtype=torch.long,
    )
    agent_types = torch.randint(
        0,
        config.num_agent_types,
        (config.batch_size, config.seq_len),
        generator=gen,
        dtype=torch.long,
    )
    # Draw per-sample valid lengths to build padding masks.
    min_len = max(1, config.seq_len // 2)
    valid_lengths = torch.randint(
        min_len,
        config.seq_len + 1,
        (config.batch_size,),
        generator=gen,
        dtype=torch.long,
    )
    arange = torch.arange(config.seq_len).unsqueeze(0).expand(config.batch_size, -1)
    padding_mask = arange >= valid_lengths.unsqueeze(1)
    action_mask_probs = torch.rand(
        config.batch_size,
        config.seq_len,
        config.action_dim,
        generator=gen,
    )
    action_masks = action_mask_probs > 0.1  # keep roughly 90% of actions valid
    fixtures = {
        "obs_sequence": obs.to(device=DEVICE),
        "action_sequence": action_sequence.to(device=DEVICE),
        "agent_types": agent_types.to(device=DEVICE),
        "positions": arange.to(device=DEVICE),
        "action_masks": action_masks.to(device=DEVICE),
        "padding_mask": padding_mask.to(device=DEVICE),
        "valid_lengths": valid_lengths.to(device=DEVICE),
    }
    return fixtures
def build_model(config: BenchmarkConfig) -> torch.nn.Module:
    """Instantiate the correct model variant for the benchmark mode."""
    model_kwargs = dict(
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
        max_seq_length=config.max_seq_length,
        num_agent_types=config.num_agent_types,
        num_experts=config.num_experts,
        top_k=config.top_k,
    )
    if config.mode == "script":
        model: torch.nn.Module = PPOReactiveModelScript(**model_kwargs)
    else:
        # Honor the configuration flag so backward benchmarks can choose
        # whether to enable gradient checkpointing during training.
        model = PPOReactiveModel(
            **model_kwargs,
            use_gradient_checkpointing=bool(config.use_gradient_checkpointing),
        )
    model = model.to(DEVICE)
    model.eval()
    if config.mode == "compile":
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build")
        model = torch.compile(model)  # type: ignore[misc]
    elif config.mode == "script":
        with torch.inference_mode():
            scripted = torch.jit.script(model)
            model = torch.jit.freeze(scripted)
    return model
def benchmark_function(
    name: str,
    func: Callable[[], Any],
    *,
    stability_threshold: float = 0.01,
    min_warmup: int = 5,
    max_warmup: int = 50,
    num_test: int = 100,
    self_timed: bool = False,
    timing_getter: Optional[Callable[[], Dict[str, List[float]]]] = None,
) -> Dict[str, Any]:
    """Benchmark ``func`` with adaptive warmup and return summary statistics.

    When ``self_timed`` is True, ``func`` is expected to record its own
    per-iteration timings in lists accessible via ``timing_getter``. In that
    case this helper will only orchestrate warmup and test loops, then compute
    and return summary statistics for forward, backward, and total times.
    """
    warmup_history: List[float] = []

    def _run_once() -> float:
        if self_timed:
            # Delegate timing to the provided function.
            func()
            torch.cuda.synchronize()
            return 0.0
        start = time.perf_counter()
        func()
        torch.cuda.synchronize()
        end = time.perf_counter()
        return end - start

    torch.cuda.synchronize()
    for _ in range(min_warmup):
        func()
    torch.cuda.synchronize()
    total_warmup = min_warmup
    # For self-timed runs, still perform adaptive warmup based on wrapper time
    # which is near-zero, but we keep the same loop structure for consistency.
    for _ in range(min_warmup, max_warmup):
        elapsed = _run_once()
        warmup_history.append(elapsed)
        total_warmup += 1
        if not self_timed and len(warmup_history) >= 5:
            mean = statistics.fmean(warmup_history[-5:])
            stdev = statistics.pstdev(warmup_history[-5:])
            if mean > 0.0 and (stdev / mean) < stability_threshold:
                break

    if self_timed:
        for _ in range(num_test):
            func()
        if timing_getter is None:
            raise ValueError("timing_getter must be provided when self_timed=True")
        buckets = timing_getter()
        fwd = buckets.get("forward_times", [])
        bwd = buckets.get("backward_times", [])
        total = buckets.get("total_times", [])
        def _stats(xs: List[float]) -> Dict[str, float]:
            if not xs:
                return {
                    "mean_ms": float("nan"),
                    "stdev_ms": float("nan"),
                    "min_ms": float("nan"),
                    "max_ms": float("nan"),
                }
            mean = statistics.fmean(xs)
            stdev = statistics.pstdev(xs) if len(xs) > 1 else 0.0
            return {
                "mean_ms": mean * 1e3,
                "stdev_ms": stdev * 1e3,
                "min_ms": min(xs) * 1e3,
                "max_ms": max(xs) * 1e3,
            }
        fwd_stats = _stats(fwd)
        bwd_stats = _stats(bwd)
        total_stats = _stats(total if total else [fi + bi for fi, bi in zip(fwd, bwd)])
        # Include legacy aggregate fields based on total time as well
        return {
            "benchmark": name,
            "num_warmup_iters": total_warmup,
            # Legacy overall fields using total timings
            "mean_ms": total_stats["mean_ms"],
            "stdev_ms": total_stats["stdev_ms"],
            "min_ms": total_stats["min_ms"],
            "max_ms": total_stats["max_ms"],
            # Detailed fields
            "mean_forward_ms": fwd_stats["mean_ms"],
            "stdev_forward_ms": fwd_stats["stdev_ms"],
            "min_forward_ms": fwd_stats["min_ms"],
            "max_forward_ms": fwd_stats["max_ms"],
            "mean_backward_ms": bwd_stats["mean_ms"],
            "stdev_backward_ms": bwd_stats["stdev_ms"],
            "min_backward_ms": bwd_stats["min_ms"],
            "max_backward_ms": bwd_stats["max_ms"],
            "mean_total_ms": total_stats["mean_ms"],
            "stdev_total_ms": total_stats["stdev_ms"],
            "min_total_ms": total_stats["min_ms"],
            "max_total_ms": total_stats["max_ms"],
        }

    # Standard path: time the function as a whole
    test_times = [_run_once() for _ in range(num_test)]
    mean = statistics.fmean(test_times)
    stdev = statistics.pstdev(test_times) if len(test_times) > 1 else 0.0
    return {
        "benchmark": name,
        "mean_ms": mean * 1e3,
        "stdev_ms": stdev * 1e3,
        "min_ms": min(test_times) * 1e3,
        "max_ms": max(test_times) * 1e3,
        "num_warmup_iters": total_warmup,
    }
def benchmark_full_forward_pass(
    model: torch.nn.Module,
    fixtures: Dict[str, torch.Tensor],
    config: BenchmarkConfig,
) -> Dict[str, Any]:
    forward_kwargs = {
        "obs_sequence": fixtures["obs_sequence"],
        "action_sequence": fixtures["action_sequence"],
        "agent_types": fixtures["agent_types"],
        "positions": fixtures["positions"],
        "padding_mask": fixtures["padding_mask"],
    }
    if "action_masks" in fixtures:
        forward_kwargs["action_masks"] = fixtures["action_masks"]
    if "valid_lengths" in fixtures:
        forward_kwargs["valid_lengths"] = fixtures["valid_lengths"]
    def _forward() -> None:
        with torch.inference_mode():
            with autocast_context(device_type="cuda", enabled=config.use_amp):
                _ = model(**forward_kwargs)
    return benchmark_function("forward", _forward)

def benchmark_forward_backward(
    model: torch.nn.Module,
    fixtures: Dict[str, torch.Tensor],
    config: BenchmarkConfig,
) -> Dict[str, Any]:
    """Benchmark combined forward + backward pass with optional checkpointing.

    Records separate timings for forward and backward phases. Optimizer step is
    omitted to focus on autograd compute and gradient checkpointing effects.
    """
    if isinstance(model, torch.jit.ScriptModule):
        raise ValueError("Backward benchmarking is not supported for scripted models")

    forward_kwargs = {
        "obs_sequence": fixtures["obs_sequence"],
        "action_sequence": fixtures["action_sequence"],
        "agent_types": fixtures["agent_types"],
        "positions": fixtures["positions"],
        "padding_mask": fixtures["padding_mask"],
    }
    if "action_masks" in fixtures:
        forward_kwargs["action_masks"] = fixtures["action_masks"]
    if "valid_lengths" in fixtures:
        forward_kwargs["valid_lengths"] = fixtures["valid_lengths"]

    # Train mode is required for gradients and optional checkpointing.
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # dummy, no step

    forward_times: List[float] = []
    backward_times: List[float] = []
    total_times: List[float] = []

    def _run_cycle() -> None:
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with autocast_context(device_type="cuda", enabled=config.use_amp):
            action_logits, _, state_values, _, _, _ = model(**forward_kwargs)  # type: ignore[misc]
            dummy_loss = action_logits.sum() + state_values.sum()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        dummy_loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        forward_times.append(t1 - t0)
        backward_times.append(t2 - t1)
        total_times.append(t2 - t0)

    def _get_buckets() -> Dict[str, List[float]]:
        return {
            "forward_times": forward_times,
            "backward_times": backward_times,
            "total_times": total_times,
        }

    try:
        return benchmark_function(
            "forward_backward",
            _run_cycle,
            self_timed=True,
            timing_getter=_get_buckets,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "benchmark": "forward_backward",
            # Legacy aggregate timing
            "mean_ms": "OOM",
            "stdev_ms": "OOM",
            "min_ms": "OOM",
            "max_ms": "OOM",
            # Detailed timing buckets
            "mean_forward_ms": "OOM",
            "stdev_forward_ms": "OOM",
            "min_forward_ms": "OOM",
            "max_forward_ms": "OOM",
            "mean_backward_ms": "OOM",
            "stdev_backward_ms": "OOM",
            "min_backward_ms": "OOM",
            "max_backward_ms": "OOM",
            "mean_total_ms": "OOM",
            "stdev_total_ms": "OOM",
            "min_total_ms": "OOM",
            "max_total_ms": "OOM",
            "num_warmup_iters": 0,
        }
def benchmark_component_functions(
    model: torch.nn.Module,
    fixtures: Dict[str, torch.Tensor],
    config: BenchmarkConfig,
) -> Iterable[Dict[str, Any]]:
    """Benchmark internal helper components when available."""
    if isinstance(model, torch.jit.ScriptModule):
        return []
    # ``torch.compile`` wraps modules inside OptimizedModule; component timings are
    # only representative for eager execution.
    if type(model).__name__ == "OptimizedModule":
        return []
    components: List[Dict[str, Any]] = []
    # Heuristic: skip component micro-benchmarks for very memory-heavy shapes.
    # These are most prone to OOM due to allocator fragmentation during
    # repeated warmups/tests and provide limited additional insight over the
    # full forward benchmark.
    is_large_batch = config.batch_size >= 128 and config.seq_len >= 384
    is_very_large_batch = config.batch_size >= 64 and config.seq_len >= 480
    if (not config.use_amp) and (is_large_batch or is_very_large_batch):
        print(
            f"INFO: Skipping component micro-benchmarks for large config: bs={config.batch_size}, seq={config.seq_len}, amp={config.use_amp}"
        )
        return components
    if not hasattr(model, "_encode_inputs"):
        return components
    def _encode_only() -> torch.Tensor:
        with torch.inference_mode():
            return model._encode_inputs(  # type: ignore[attr-defined]
                fixtures["obs_sequence"],
                fixtures["action_sequence"],
                fixtures["agent_types"],
                fixtures["positions"],
                fixtures.get("padding_mask"),
            )
    # Encode stage (timed) with OOM guard
    try:
        encoded_inputs = _encode_only()
        components.append(benchmark_function("encode_inputs", _encode_only))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        components.append({
            "benchmark": "encode_inputs",
            "mean_ms": "OOM",
            "stdev_ms": "OOM",
            "min_ms": "OOM",
            "max_ms": "OOM",
            "mean_forward_ms": "OOM",
            "stdev_forward_ms": "OOM",
            "min_forward_ms": "OOM",
            "max_forward_ms": "OOM",
            "mean_backward_ms": "OOM",
            "stdev_backward_ms": "OOM",
            "min_backward_ms": "OOM",
            "max_backward_ms": "OOM",
            "mean_total_ms": "OOM",
            "stdev_total_ms": "OOM",
            "min_total_ms": "OOM",
            "max_total_ms": "OOM",
            "num_warmup_iters": 0,
        })
        # If encoding itself OOMs, subsequent components cannot run
        return components
    # Prepare masks (small) after successful encoding
    causal_mask, key_padding = model._prepare_masks(  # type: ignore[attr-defined]
        encoded_inputs, fixtures.get("padding_mask")
    )
    def _transformer_only() -> torch.Tensor:
        with torch.inference_mode():
            return model._run_transformer(  # type: ignore[attr-defined]
                encoded_inputs,
                causal_mask=causal_mask,
                key_padding=key_padding,
            )
    # Transformer stage with OOM guard
    try:
        components.append(benchmark_function("transformer", _transformer_only))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        components.append({
            "benchmark": "transformer",
            "mean_ms": "OOM",
            "stdev_ms": "OOM",
            "min_ms": "OOM",
            "max_ms": "OOM",
            "mean_forward_ms": "OOM",
            "stdev_forward_ms": "OOM",
            "min_forward_ms": "OOM",
            "max_forward_ms": "OOM",
            "mean_backward_ms": "OOM",
            "stdev_backward_ms": "OOM",
            "min_backward_ms": "OOM",
            "max_backward_ms": "OOM",
            "mean_total_ms": "OOM",
            "stdev_total_ms": "OOM",
            "min_total_ms": "OOM",
            "max_total_ms": "OOM",
            "num_warmup_iters": 0,
        })
        return components
    # Precompute transformer output for head-only timing; guard for OOM as well
    try:
        transformer_output, _, routing = model._run_transformer(  # type: ignore[attr-defined]
            encoded_inputs, causal_mask=causal_mask, key_padding=key_padding
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        components.append({
            "benchmark": "head_outputs",
            "mean_ms": "OOM",
            "stdev_ms": "OOM",
            "min_ms": "OOM",
            "max_ms": "OOM",
            "mean_forward_ms": "OOM",
            "stdev_forward_ms": "OOM",
            "min_forward_ms": "OOM",
            "max_forward_ms": "OOM",
            "mean_backward_ms": "OOM",
            "stdev_backward_ms": "OOM",
            "min_backward_ms": "OOM",
            "max_backward_ms": "OOM",
            "mean_total_ms": "OOM",
            "stdev_total_ms": "OOM",
            "min_total_ms": "OOM",
            "max_total_ms": "OOM",
            "num_warmup_iters": 0,
        })
        return components
    def _head_only() -> torch.Tensor:
        with torch.inference_mode():
            return model._head_outputs(transformer_output, routing)  # type: ignore[attr-defined]
    components.append(benchmark_function("head_outputs", _head_only))
    return components
def run_benchmark_suite(
    model: torch.nn.Module, fixtures: Dict[str, torch.Tensor], config: BenchmarkConfig
) -> List[Dict[str, Any]]:
    """Run the set of relevant benchmarks for the given mode.

    Includes forward-only benchmark in all modes, forward+backward benchmark in
    eager/compile modes, and component micro-benchmarks only in eager mode.
    Handles KeyboardInterrupt by returning any partial results collected so far.
    """
    results: List[Dict[str, Any]] = []
    # Heuristic pruning to avoid near-certain OOM or thrash cases
    is_large_batch = config.batch_size >= 128 and config.seq_len >= 384
    is_very_large_batch = config.batch_size >= 64 and config.seq_len >= 480
    is_extreme = config.batch_size >= 128 and config.seq_len >= 480
    skip_backward_pass = (
        (not config.use_amp)
        and (
            is_extreme  # skip even if checkpointing is on
            or (
                (not bool(config.use_gradient_checkpointing))
                and (is_large_batch or is_very_large_batch)
            )
        )
    )
    try:
        results.append(benchmark_full_forward_pass(model, fixtures, config))
        if config.mode in ["eager", "compile"] and not skip_backward_pass:
            results.append(benchmark_forward_backward(model, fixtures, config))
        elif skip_backward_pass:
            print(f"INFO: Skipping backward pass for known large configuration: {config}")
        if config.mode == "eager":
            results.extend(benchmark_component_functions(model, fixtures, config))
    except KeyboardInterrupt:
        # Return whatever we've measured so far to be persisted by caller
        return results
    return results
def _default_benchmark_config() -> BenchmarkConfig:
    """Return a representative config for header computation."""
    return BenchmarkConfig(
        mode="eager",
        batch_size=1,
        seq_len=1,
        use_amp=False,
        determinism_level="none",
        use_gradient_checkpointing=False,
    )
def compute_csv_header() -> List[str]:
    """Compute the canonical CSV header shared by all workers."""
    base_config = _default_benchmark_config()
    config_fields = list(asdict(base_config).keys())
    backend_state_fields = list(apply_determinism_settings("none").keys())
    benchmark_fields = [
        "benchmark",
        # Legacy aggregate timing
        "mean_ms",
        "stdev_ms",
        "min_ms",
        "max_ms",
        # Detailed forward/backward/total timing
        "mean_forward_ms",
        "stdev_forward_ms",
        "min_forward_ms",
        "max_forward_ms",
        "mean_backward_ms",
        "stdev_backward_ms",
        "min_backward_ms",
        "max_backward_ms",
        "mean_total_ms",
        "stdev_total_ms",
        "min_total_ms",
        "max_total_ms",
        "num_warmup_iters",
    ]
    return config_fields + backend_state_fields + benchmark_fields
def initialize_results_file(path: Path, header: Sequence[str]) -> None:
    """Ensure the results CSV exists with the expected header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
def read_csv_header(path: Path) -> List[str]:
    """Read the header row from ``path``."""
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Results file {path} is empty; expected header row.") from exc
def append_rows_to_csv(rows: Sequence[Dict[str, Any]], path: Path, header: Sequence[str]) -> None:
    """Append benchmark ``rows`` to ``path`` using ``header`` ordering."""
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        for row in rows:
            writer.writerow({field: row.get(field) for field in header})
def run_single_benchmark(config: BenchmarkConfig, results_path: Path, header: Sequence[str]) -> None:
    """Execute a single configuration and append its results to ``results_path``.

    On KeyboardInterrupt, writes any partial results measured so far before
    exiting to preserve progress.
    """
    print(
        "=" * 80,
        f"\nWorker running mode={config.mode} bs={config.batch_size} seq={config.seq_len} "
        f"amp={config.use_amp} determinism={config.determinism_level} "
        f"ckpt={config.use_gradient_checkpointing}",
    )
    # Proactively cap per-process memory to force a clear OOM instead of
    # allowing oversubscription/host paging on some drivers.
    try:
        frac = float(os.environ.get("PER_PROCESS_MEM_FRAC", "0.95"))
        torch.cuda.set_per_process_memory_fraction(frac, device=0)
    except Exception:
        pass
    backend_state = apply_determinism_settings(config.determinism_level)

    # Known-stability guard: TorchScript + AMP + full determinism has caused
    # low-level crashes on some driver/PyTorch combinations (return code -11).
    # Rather than letting the worker segfault and halt the sweep, record a
    # clearly labeled skip so the manager can proceed and the CSV remains
    # consistent.
    if (
        config.mode == "script"
        and bool(config.use_amp)
        and config.determinism_level == "full"
    ):
        print(
            "INFO: Skipping config due to known instability in TorchScript + AMP + full determinism."
        )
        skipped_row = {
            **asdict(config),
            **backend_state,
            "benchmark": "skipped_known_issue",
            "mean_ms": "SKIP",
            "stdev_ms": "SKIP",
            "min_ms": "SKIP",
            "max_ms": "SKIP",
            "mean_forward_ms": "SKIP",
            "stdev_forward_ms": "SKIP",
            "min_forward_ms": "SKIP",
            "max_forward_ms": "SKIP",
            "mean_backward_ms": "SKIP",
            "stdev_backward_ms": "SKIP",
            "min_backward_ms": "SKIP",
            "max_backward_ms": "SKIP",
            "mean_total_ms": "SKIP",
            "stdev_total_ms": "SKIP",
            "min_total_ms": "SKIP",
            "max_total_ms": "SKIP",
            "num_warmup_iters": 0,
        }
        append_rows_to_csv([skipped_row], results_path, header)
        torch.cuda.empty_cache()
        return

    # Hard skip: configurations that are too large regardless of AMP/ckpt
    if config.batch_size >= 128 and config.seq_len >= 480:
        print(
            f"INFO: Skipping entire config for extreme size (bs={config.batch_size}, seq={config.seq_len})."
        )
        skipped_row = {
            **asdict(config),
            **backend_state,
            "benchmark": "skipped_config",
            "mean_ms": "SKIP",
            "stdev_ms": "SKIP",
            "min_ms": "SKIP",
            "max_ms": "SKIP",
            "mean_forward_ms": "SKIP",
            "stdev_forward_ms": "SKIP",
            "min_forward_ms": "SKIP",
            "max_forward_ms": "SKIP",
            "mean_backward_ms": "SKIP",
            "stdev_backward_ms": "SKIP",
            "min_backward_ms": "SKIP",
            "max_backward_ms": "SKIP",
            "mean_total_ms": "SKIP",
            "stdev_total_ms": "SKIP",
            "min_total_ms": "SKIP",
            "max_total_ms": "SKIP",
            "num_warmup_iters": 0,
        }
        append_rows_to_csv([skipped_row], results_path, header)
        torch.cuda.empty_cache()
        return
    model = build_model(config)
    fixtures = generate_fixtures(config)
    try:
        benchmarks = run_benchmark_suite(model, fixtures, config)
    except torch.cuda.OutOfMemoryError:
        # Catch any uncaught OOMs from deeper benchmark helpers and record a
        # single OOM row so the subprocess can exit cleanly and the manager can
        # continue with other configurations.
        torch.cuda.empty_cache()
        benchmarks = [{
            "benchmark": "suite_oom",
            "mean_ms": "OOM",
            "stdev_ms": "OOM",
            "min_ms": "OOM",
            "max_ms": "OOM",
            "mean_forward_ms": "OOM",
            "stdev_forward_ms": "OOM",
            "min_forward_ms": "OOM",
            "max_forward_ms": "OOM",
            "mean_backward_ms": "OOM",
            "stdev_backward_ms": "OOM",
            "min_backward_ms": "OOM",
            "max_backward_ms": "OOM",
            "mean_total_ms": "OOM",
            "stdev_total_ms": "OOM",
            "min_total_ms": "OOM",
            "max_total_ms": "OOM",
            "num_warmup_iters": 0,
        }]
    except KeyboardInterrupt:
        # run_benchmark_suite already attempts partial return; if interrupted
        # here, just proceed with what we have
        benchmarks = []  # nothing to append in this branch
    finally:
        del fixtures
    rows: List[Dict[str, Any]] = []
    for bench in benchmarks:
        rows.append({**asdict(config), **backend_state, **bench})
    if rows:
        append_rows_to_csv(rows, results_path, header)
    del model
    torch.cuda.empty_cache()
    
def generate_configurations(
    modes: Sequence[str],
    batch_sizes: Sequence[int],
    seq_lengths: Sequence[int],
    amp_settings: Sequence[bool],
    determinism_levels: Sequence[str],
    use_ckpt_settings: Sequence[bool],
) -> List[BenchmarkConfig]:
    """Enumerate the Cartesian product for manager orchestration."""
    configs: List[BenchmarkConfig] = []
    for mode, batch_size, seq_len, use_amp, det, use_ckpt in itertools.product(
        modes, batch_sizes, seq_lengths, amp_settings, determinism_levels, use_ckpt_settings
    ):
        configs.append(
            BenchmarkConfig(
                mode=mode,
                batch_size=batch_size,
                seq_len=seq_len,
                use_amp=use_amp,
                determinism_level=det,
                use_gradient_checkpointing=use_ckpt,
            )
        )
    return configs

def run_manager(results_path: Path, header: Sequence[str]) -> None:
    """
    Coordinate a targeted subprocess execution to test gradient checkpointing
    performance specifically for torch.compile mode.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmarking script.")

    # --- START: CUSTOMIZED TEST CONFIGURATION ---

    # 1. Isolate the test to 'compile' mode only.
    modes: List[str] = ["compile"]

    # 2. Define the exact (batch_size, seq_len) pairs to test based on the optimal map.
    PPO_MINIBATCH_SIZE_BUCKET_MAP = {
        64: 512,
        128: 512,
        256: 512,
        384: 256,
    }
    # Standard batch sizes to select from (excluding 32 as requested).
    VALID_BATCH_SIZES = [64, 128, 256, 512]
    
    targeted_configs: List[Tuple[int, int]] = []
    for seq_len, max_bs in PPO_MINIBATCH_SIZE_BUCKET_MAP.items():
        try:
            # Find the index of the largest optimal batch size.
            max_bs_index = VALID_BATCH_SIZES.index(max_bs)
            # Test all valid batch sizes up to (but not including) the largest one.
            # We add the max_bs_index itself to include the boundary case.
            batch_sizes_to_test = VALID_BATCH_SIZES[:max_bs_index + 1]
        except ValueError:
            # If max_bs is not in the list (e.g., smaller than 64), test the smallest valid one.
            batch_sizes_to_test = [VALID_BATCH_SIZES[0]]

        for bs in batch_sizes_to_test:
            targeted_configs.append((bs, seq_len))

    # 3. Test with and without AMP, as memory usage differs.
    amp_settings = [False, True]
    
    # 4. Keep the test fast by only focusing on the non-deterministic mode.
    determinism_levels = ["none"]
    
    # 5. This is the core of the test: sweep both checkpointing options.
    use_ckpt_settings = [True, False]
    
    # Generate the final list of BenchmarkConfig objects from our targeted list.
    full_configs: List[BenchmarkConfig] = []
    for (bs, sl), use_amp, use_ckpt in itertools.product(
        targeted_configs, amp_settings, use_ckpt_settings
    ):
        # We only need to iterate over one determinism level.
        full_configs.append(
            BenchmarkConfig(
                mode=modes[0],
                batch_size=bs,
                seq_len=sl,
                use_amp=use_amp,
                determinism_level=determinism_levels[0],
                use_gradient_checkpointing=use_ckpt,
            )
        )

    # --- END: CUSTOMIZED TEST CONFIGURATION ---

    # The rest of the manager logic remains the same (resuming, subprocess execution, etc.)
    completed_configs: set[tuple] = set()
    if results_path.exists() and results_path.stat().st_size > 0:
        try:
            df_existing = pd.read_csv(results_path)
            id_keys = [
                "mode", "batch_size", "seq_len", "use_amp", 
                "determinism_level", "use_gradient_checkpointing",
            ]
            keys_in_file = [k for k in id_keys if k in df_existing.columns]
            for _, row in df_existing[keys_in_file].iterrows():
                completed_configs.add(tuple(row[k] for k in keys_in_file))
        except Exception:
            completed_configs = set()
            
    def _config_key(c: BenchmarkConfig, keys: List[str]) -> tuple:
        data = asdict(c)
        return tuple(data[k] for k in keys)

    default_keys = [
        "mode", "batch_size", "seq_len", "use_amp", 
        "determinism_level", "use_gradient_checkpointing",
    ]
    keys_for_match: List[str] = default_keys
    
    if completed_configs:
        configs = [c for c in full_configs if _config_key(c, keys_for_match) not in completed_configs]
    else:
        configs = full_configs

    total = len(configs)
    skipped_due_to_csv = len(full_configs) - total
    if skipped_due_to_csv > 0:
        print(f"Resuming: {skipped_due_to_csv} configs already in CSV; {total} remaining.")

    print(f"Starting targeted gradient checkpointing benchmark with {total} configurations to test.")
    
    try:
        for index, config in enumerate(configs, start=1):
            print(f"\n{'='*80}\n[{index}/{total}] Spawning worker for config: {config}")
            env = os.environ.copy()
            if config.determinism_level in {"high", "full"}:
                env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            else:
                env.pop("CUBLAS_WORKSPACE_CONFIG", None)
            
            env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:False"
            
            cmd = [
                sys.executable,
                "-m",
                "src.tests.analyze_performance",
                "--run-with-config",
                json.dumps(asdict(config)),
                "--results-file",
                str(results_path),
            ]
            result = subprocess.run(cmd, env=env, check=False)
            if result.returncode != 0:
                print(f"Warning: Subprocess failed for config {config} with return code {result.returncode}.")

    except KeyboardInterrupt:
        print("Interrupted by user. Partial results remain saved.")

    print(f"\n{'='*80}\nBenchmark sweep complete. Results saved to {results_path}")
    
def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PPO reactive models")
    parser.add_argument(
        "--run-with-config",
        type=str,
        default=None,
        help="Internal flag used by the manager process to run a single configuration.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("performance_analysis_ckpt.csv"),
        help="Path to the CSV file where benchmark results are stored.",
    )
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    results_path = args.results_file.resolve()
    header = compute_csv_header()
    initialize_results_file(results_path, header)
    if args.run_with_config is not None:
        config_data = json.loads(args.run_with_config)
        config = BenchmarkConfig(**config_data)
        header_from_file = read_csv_header(results_path)
        run_single_benchmark(config, results_path, header_from_file)
    else:
        run_manager(results_path, header)
if __name__ == "__main__":
    main()
