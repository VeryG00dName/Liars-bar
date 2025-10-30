"""Shared test utilities for model, weights, inputs, parity and profiling.

This module centralizes helpers that were previously duplicated across
multiple tests. Utilities here are intentionally explicit and loud:
they validate assumptions and raise on unexpected states to avoid
silent test skew.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from src.misc import lb
from src.model.ppo_reactive_model import PPOReactiveModel


# ---------------------------------------------------------------------------
# Section 1: Checkpoint & Weight Loading
# ---------------------------------------------------------------------------
def load_checkpoint_weights(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    """Load model weights from a checkpoint file into a raw state_dict.

    Handles several common wrappers used in this codebase:
    - {"model_state_dict": <state_dict>}
    - {"policy_nets": {"self": <state_dict>, ...}} (legacy format)
    - Direct tensor mapping (already a state_dict)

    Args:
        checkpoint_path: Filesystem path to a torch checkpoint (``.pth``).

    Returns:
        A flat mapping of parameter name -> tensor (CPU tensors).

    Raises:
        ValueError: If no recognizable state dict mapping is found.
    """
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")

    # Direct mapping of tensors (already a state dict)
    if isinstance(checkpoint, dict) and any(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint  # type: ignore[return-value]

    if isinstance(checkpoint, dict):
        # Preferred wrapper used by our training scripts
        if "model_state_dict" in checkpoint:
            msd = checkpoint["model_state_dict"]
            if isinstance(msd, dict):
                return msd  # type: ignore[return-value]

        # Legacy multi-policy wrapper
        if "policy_nets" in checkpoint and isinstance(checkpoint["policy_nets"], dict):
            nets = checkpoint["policy_nets"]
            # Try explicit "self" key first
            if "self" in nets and isinstance(nets["self"], dict):
                return nets["self"]  # type: ignore[return-value]
            # Otherwise return the first mapping
            for _, sub in nets.items():
                if isinstance(sub, dict):
                    return sub  # type: ignore[return-value]

    raise ValueError(
        f"Could not extract model state dict from checkpoint (keys: "
        f"{list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)})"
    )


def pad_model_weights(state_dict: Dict[str, torch.Tensor], pad_obs_to: int = 16) -> Dict[str, torch.Tensor]:
    """Pad ``obs_encoder.0.weight`` for CUDA alignment if needed (in-place).

    Pads the input dimension of ``obs_encoder.0.weight`` from ``[H, I]`` to
    ``[H, pad_obs_to]`` by concatenating zeros on the right when ``I < pad_obs_to``.

    Args:
        state_dict: Model state dict to modify in-place.
        pad_obs_to: Target input dimension (default: 16).

    Returns:
        The same ``state_dict`` object (modified in-place).
    """
    key = "obs_encoder.0.weight"
    if key in state_dict:
        w = state_dict[key]
        if w.ndim == 2:
            hidden_dim, in_dim = int(w.size(0)), int(w.size(1))
            if in_dim < pad_obs_to:
                pad = pad_obs_to - in_dim
                zeros = torch.zeros(hidden_dim, pad, dtype=w.dtype, device=w.device)
                state_dict[key] = torch.cat([w, zeros], dim=1)
                print(
                    f"Padded {key} from [{hidden_dim}, {in_dim}] to "
                    f"[{hidden_dim}, {pad_obs_to}]"
                )
    return state_dict


def prepare_batched_weights(
    state_dict: Dict[str, torch.Tensor],
    num_layers: int,
    num_experts: int,
    device: str,
    to_fp16: bool = True,
) -> Dict[str, torch.Tensor]:
    """Prepare weights for C++ inference/training calls.

    Steps:
    1) Pre-stack MoE expert weights with ``lb.prestack_moe_expert_weights``.
    2) Add batch dimension ``[1, ...]`` to every tensor and move to ``device``.
    3) If ``to_fp16`` is True, convert only MoE expert weights (``.moe.experts.``)
       to ``float16``; leave other tensors in their original dtype.
    4) Create MoE pointer tensors with ``lb.create_moe_weight_pointers``.
    5) Add fixed buffers with ``lb.add_fixed_buffers``.

    Args:
        state_dict: Flat parameter mapping (CPU tensors recommended).
        num_layers: Number of transformer layers.
        num_experts: Number of experts per layer.
        device: Target device string (e.g., ``"cuda"`` or ``"cpu"``).
        to_fp16: Whether to cast MoE expert tensors to ``float16``.

    Returns:
        A fully prepared batched weight dictionary ready for C++ entrypoints.
    """
    # 1) Pre-stack MoE expert weights first (before batching)
    sd = lb.prestack_moe_expert_weights(dict(state_dict), num_layers, num_experts)

    # 2) Add batch dimension and move to device
    batched: Dict[str, torch.Tensor] = {}
    for k, t in sd.items():
        if torch.is_tensor(t):
            bt = t.unsqueeze(0).to(device)
            batched[k] = bt
        else:
            # Non-tensor entries are copied through unmodified
            batched[k] = t  # type: ignore[assignment]

    # 3) Optional: cast ONLY MoE expert tensors to fp16 for CUDA kernels
    if to_fp16:
        for k, t in list(batched.items()):
            if torch.is_tensor(t) and t.is_floating_point() and ".moe.experts." in k:
                batched[k] = t.to(torch.float16).contiguous()

    # 4) Create pointer tensors for MoE expert stacks
    batched = lb.create_moe_weight_pointers(batched, num_layers, num_experts)

    # 5) Add fixed buffers (LUTs) expected by the C++ side
    batched = lb.add_fixed_buffers(batched, device)

    return batched


# ---------------------------------------------------------------------------
# Section 2: Input Generation
# ---------------------------------------------------------------------------
def create_dummy_inputs(
    batch_size: int,
    seq_len: int,
    obs_dim: int,
    device: str,
    seed: Optional[int] = None,
    padding_ratio: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create synthetic inputs for testing and profiling.

    The generated tensors follow typical shapes used throughout tests
    and are intentionally simple (e.g., an all-valid action mask).

    Args:
        batch_size: Batch size ``B``.
        seq_len: Sequence length ``T``.
        obs_dim: Observation dimension.
        device: Device string (``"cuda"`` or ``"cpu"``).
        seed: Optional RNG seed for reproducibility.
        padding_ratio: Fraction of the sequence to mark as padding (``0.0 .. 1.0``).
        dtype: Data type for ``obs_sequence`` (default: ``torch.float32``).

    Returns:
        Tuple of tensors:
        ``(obs_sequence[B,T,obs_dim], action_sequence[B,T], agent_types[B,T], positions[B,T], action_masks[B,T,7], padding_mask[B,T])``.
    """
    gen: Optional[torch.Generator] = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

    obs_sequence = torch.randn(
        batch_size, seq_len, obs_dim, dtype=dtype, device=device, generator=gen
    )
    action_sequence = torch.randint(
        0, 11, (batch_size, seq_len), dtype=torch.long, device=device, generator=gen
    )
    agent_types = torch.randint(
        0, 3, (batch_size, seq_len), dtype=torch.long, device=device, generator=gen
    )
    positions = torch.randint(
        0, 4, (batch_size, seq_len), dtype=torch.long, device=device, generator=gen
    )
    action_masks = torch.ones(
        batch_size, seq_len, 7, dtype=torch.bool, device=device
    )
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    if padding_ratio > 0.0:
        start = max(0, min(seq_len, int(seq_len * (1.0 - padding_ratio))))
        if start < seq_len:
            padding_mask[:, start:] = True

    return (
        obs_sequence,
        action_sequence,
        agent_types,
        positions,
        action_masks,
        padding_mask,
    )


# ---------------------------------------------------------------------------
# Section 3: Tensor Comparison
# ---------------------------------------------------------------------------
def compare_tensors(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    verbose: bool = True,
) -> bool:
    """Compare two tensors with tolerances and optional detailed logging.

    Args:
        name: Label used in logs.
        actual: The produced tensor.
        expected: The reference tensor.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        verbose: If True, prints details on mismatch.

    Returns:
        True if ``actual`` is within tolerance of ``expected`` elementwise.
    """
    a = actual.detach().float().cpu()
    b = expected.detach().float().cpu()

    if a.shape != b.shape:
        if verbose:
            print(f"{name}: SHAPE MISMATCH actual={tuple(a.shape)} vs expected={tuple(b.shape)}")
        return False

    abs_diff = (a - b).abs()
    tol = atol + rtol * b.abs()
    mism = abs_diff > tol
    ok = not bool(mism.any().item())

    if verbose and not ok:
        mism_count = int(mism.sum().item())
        total = b.numel()
        max_err = float(abs_diff.max().item())
        mean_err = float(abs_diff.mean().item())
        pct = 100.0 * mism_count / max(total, 1)
        print(
            f"{name}: {pct:.2f}% mismatch, max|Δ|={max_err:.3e}, mean|Δ|={mean_err:.3e}"
        )

    return ok


def report_parity_metrics(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    rtol: float,
    atol: float,
) -> bool:
    """Always print detailed parity metrics and return pass/fail.

    Args:
        name: Label for the metric line.
        a: Actual tensor.
        b: Reference tensor.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        True if there are zero elementwise mismatches under the tolerance.
    """
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    abs_diff = (a - b).abs()
    tol = atol + rtol * b.abs()
    mism = abs_diff > tol
    mism_count = int(mism.sum().item())
    total = b.numel()
    mism_pct = 100.0 * mism_count / max(total, 1)
    max_err = float(abs_diff.max().item())
    mean_err = float(abs_diff.mean().item())
    rel_err = (abs_diff / (b.abs() + 1e-8)).clamp_max(1e6)
    mean_rel = float(rel_err.mean().item())
    print(
        f"{name:8s}: mism={mism_count}/{total} ({mism_pct:6.3f}%), "
        f"max|Δ|={max_err:.3e}, mean|Δ|={mean_err:.3e}, mean rel={mean_rel:.3e}"
    )
    return mism_count == 0


def argmax_parity(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    margin_threshold: float = 1e-3,
) -> bool:
    """Check argmax parity and report near-tie counts.

    Args:
        name: Label for the metric line.
        a: Actual logits with final dim = classes.
        b: Reference logits with final dim = classes.
        margin_threshold: Threshold below which a position is considered a near-tie.

    Returns:
        True if all argmax indices match.
    """
    a_idx = a.argmax(dim=-1)
    b_idx = b.argmax(dim=-1)
    eq = (a_idx == b_idx)
    total = int(eq.numel())
    same = int(eq.sum().item())
    pct = 100.0 * same / max(total, 1)

    def top1_margin(x: torch.Tensor) -> torch.Tensor:
        top2 = torch.topk(x, k=2, dim=-1).values
        return (top2[..., 0] - top2[..., 1]).abs()

    a_margin = top1_margin(a).detach().float().cpu()
    b_margin = top1_margin(b).detach().float().cpu()
    near_tie = int(((a_margin < margin_threshold) | (b_margin < margin_threshold)).sum().item())
    print(f"{name:8s} argmax parity: {same}/{total} ({pct:6.3f}%)  near-ties(<{margin_threshold}): {near_tie}")
    return same == total


# ---------------------------------------------------------------------------
# Section 4: Gradient Checking
# ---------------------------------------------------------------------------
def check_gradient_flow(model: torch.nn.Module, print_details: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """Verify that all parameters have finite gradients.

    Args:
        model: The model after a ``backward()`` call.
        print_details: If True, print per-parameter issues.

    Returns:
        A tuple ``(all_ok, stats)`` where ``all_ok`` is True iff no parameter
        is missing a gradient and no gradient contains NaN values. ``stats``
        contains a breakdown with keys: ``no_grad_params`` (list[str]),
        ``nan_params`` (dict[name -> nan_count]), ``total_params`` (int),
        and ``total_nan_count`` (int).
    """
    no_grad: List[str] = []
    nan_params: Dict[str, int] = {}
    total_params = 0
    total_nan = 0

    for name, p in model.named_parameters():
        total_params += 1
        if p.grad is None:
            no_grad.append(name)
            if print_details:
                print(f"⚠️  {name:50s} has no gradient")
            continue
        nan_count = int(torch.isnan(p.grad).sum().item())
        if nan_count > 0:
            nan_params[name] = nan_count
            total_nan += nan_count
            if print_details:
                print(f"❌ {name:50s} gradient has {nan_count} NaNs")

    all_ok = (len(no_grad) == 0) and (total_nan == 0)
    stats: Dict[str, Any] = {
        "no_grad_params": no_grad,
        "nan_params": nan_params,
        "total_params": total_params,
        "total_nan_count": total_nan,
    }
    return all_ok, stats


def categorize_gradient_norms(model: torch.nn.Module) -> Dict[str, List[float]]:
    """Group gradient norms by rough layer categories for quick analysis.

    Categories tracked:
    - obs_encoder
    - embeddings (any name containing "embedding")
    - gates (any name containing "gate_")
    - transformer.layers.0
    - transformer.layers.1
    - moe.gate
    - moe.experts
    - heads
    """
    buckets: Dict[str, List[float]] = {
        "obs_encoder": [],
        "embeddings": [],
        "gates": [],
        "transformer.layers.0": [],
        "transformer.layers.1": [],
        "moe.gate": [],
        "moe.experts": [],
        "heads": [],
    }

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = float(p.grad.norm().item())
        if "obs_encoder" in name:
            buckets["obs_encoder"].append(g)
        elif "embedding" in name:
            buckets["embeddings"].append(g)
        elif "gate_" in name:
            buckets["gates"].append(g)
        elif "transformer.layers.0" in name:
            if "moe.gate" in name:
                buckets["moe.gate"].append(g)
            elif "moe.experts" in name:
                buckets["moe.experts"].append(g)
            else:
                buckets["transformer.layers.0"].append(g)
        elif "transformer.layers.1" in name:
            buckets["transformer.layers.1"].append(g)
        elif "heads" in name:
            buckets["heads"].append(g)

    return buckets


# ---------------------------------------------------------------------------
# Section 5: Model Creation
# ---------------------------------------------------------------------------
def create_test_model(
    obs_dim: int = 16,
    hidden_dim: int = 256,
    num_layers: int = 2,
    num_experts: int = 8,
    top_k: int = 2,
    device: str = "cuda",
) -> PPOReactiveModel:
    """Construct a standard test model configuration.

    Args:
        obs_dim: Observation dimension.
        hidden_dim: Transformer hidden size.
        num_layers: Number of transformer layers.
        num_experts: Number of experts per MoE layer.
        top_k: Top-k experts to route tokens to.
        device: Target device (default: "cuda").

    Returns:
        ``PPOReactiveModel`` moved to ``device`` and set to ``train()`` mode.
    """
    model = PPOReactiveModel(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k
    ).to(device)
    model.train()
    return model


# ---------------------------------------------------------------------------
# Section 6: Assertions
# ---------------------------------------------------------------------------
def assert_no_nan(tensor: torch.Tensor, name: str) -> None:
    """Assert that ``tensor`` contains no NaN values.

    Raises ``AssertionError`` with the first NaN location if present.
    """
    mask = torch.isnan(tensor)
    if mask.any().item():
        idx = mask.nonzero(as_tuple=False)[0].tolist()
        raise AssertionError(f"NaN detected in '{name}' at index {idx}")


def assert_finite(tensor: torch.Tensor, name: str) -> None:
    """Assert that ``tensor`` contains no Inf values.

    Raises ``AssertionError`` with the first Inf location if present.
    """
    mask = ~torch.isfinite(tensor)
    if mask.any().item():
        idx = mask.nonzero(as_tuple=False)[0].tolist()
        raise AssertionError(f"Non-finite value detected in '{name}' at index {idx}")


def assert_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], name: str) -> None:
    """Assert that ``tensor.shape`` matches ``expected_shape`` exactly."""
    actual = tuple(int(x) for x in tensor.shape)
    exp = tuple(int(x) for x in expected_shape)
    if actual != exp:
        raise AssertionError(
            f"Shape mismatch for '{name}': actual={actual} expected={exp}"
        )


# ---------------------------------------------------------------------------
# Section 7: Performance Profiling
# ---------------------------------------------------------------------------
def profile_cuda_execution(
    func: Callable[[], Any],
    warmup: int = 10,
    iterations: int = 100,
) -> Dict[str, float]:
    """Profile a callable using CUDA events for accurate timing.

    Notes:
    - The callable should perform CUDA work (otherwise timing is meaningless).
    - This function synchronizes the default CUDA stream for stable results.

    Args:
        func: Zero-argument callable to profile.
        warmup: Number of warmup iterations (JIT, kernel cache, etc.).
        iterations: Number of timed iterations.

    Returns:
        Dict with ``total_time_ms``, ``avg_time_ms``, and ``throughput_per_sec``.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("profile_cuda_execution requires CUDA to be available")

    # Warmup
    for _ in range(max(0, int(warmup))):
        func()

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(max(1, int(iterations))):
        func()
    end.record()

    torch.cuda.synchronize()
    total_ms = float(start.elapsed_time(end))
    avg_ms = total_ms / max(1, int(iterations))
    throughput = 1000.0 / avg_ms if avg_ms > 0 else float("inf")
    return {
        "total_time_ms": total_ms,
        "avg_time_ms": avg_ms,
        "throughput_per_sec": throughput,
    }

