#!/usr/bin/env python3
"""
Eval parity: TorchScript model vs C++ forward_packed.

Usage:
  python -m src.tests.test_eval_parity_torchscript \
    --checkpoint checkpoints/test76/gen_15/final.pth \
    --script     checkpoints/test76/gen_15/final_traced.pt \
    --device     cuda

This generates synthetic test inputs and compares the four heads:
  action_logits, opp_logits, state_values, win_logits
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch

from src.misc import lb
from src.tests import test_utils as tu


def compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor, rtol=5e-3, atol=5e-4) -> bool:
    """Compare tensors with FP16-appropriate tolerances.

    FP16 has ~3 decimal digits of precision, so we use rtol=5e-3 and atol=5e-4
    to account for rounding differences between TorchScript and C++ implementations.
    """
    return tu.compare_tensors(name, a, b, rtol=rtol, atol=atol, verbose=True)


def _generate_inputs(
    batch_size: int,
    seq_len: int,
    obs_dim: int,
    device: str,
    batch_file: Path | None = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate test inputs from batch file or create synthetic."""
    if batch_file is not None and batch_file.exists():
        data = torch.load(str(batch_file), map_location=device)
        # Convert obs to FP16 for C++ kernels, keep indices as long
        obs = data['obs_sequence'].to(device).to(torch.float16)
        acts = data['action_sequence'].to(device)
        agent = data['agent_types'].to(device)
        pos = data['positions'].to(device)
        pad = data.get('padding_mask')
        if pad is not None:
            pad = pad.to(device)
        return obs, acts, agent, pos, pad

    # Generate synthetic inputs using test_utils (FP16 for C++ kernels)
    # Match profile_cpp_forward: create 9-dim obs, then pad to 16
    print(f"Generating synthetic inputs: batch={batch_size}, seq_len={seq_len}, obs_dim={obs_dim}")
    obs_9d, acts, agent, pos, _, pad = tu.create_dummy_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        obs_dim=9,  # Generate 9-dim first
        device=device,
        seed=42,
        padding_ratio=0.0,
        dtype=torch.float16
    )

    # Pad from 9 to obs_dim (usually 16) with zeros
    if obs_dim > 9:
        padding_size = obs_dim - 9
        obs = torch.cat([
            obs_9d,
            torch.zeros(batch_size, seq_len, padding_size, dtype=torch.float16, device=device)
        ], dim=-1)
    else:
        obs = obs_9d

    return obs, acts, agent, pos, pad


def run_parity(
    checkpoint: Path,
    script_pt: Path,
    device: str,
    batch_file: Path | None,
    batch_size: int,
    seq_len: int,
    obs_dim: int
) -> int:
    # Generate or load inputs
    obs, acts, agent, pos, pad = _generate_inputs(batch_size, seq_len, obs_dim, device, batch_file)

    # Policy indices (single-policy eval)
    B = obs.size(0)
    policy_indices = torch.zeros(B, dtype=torch.long, device=device)

    # Load TorchScript model and eval
    ts = torch.jit.load(str(script_pt), map_location=device)
    ts.eval()

    # Convert TorchScript model to FP16 to match C++ precision
    ts = ts.half()

    # Run TorchScript forward in FP16 to match C++ precision
    # (C++ kernels use FP16, so compare apples-to-apples)
    with torch.no_grad():
        a_ts, o_ts, v_ts, w_ts = ts(obs, acts, agent, pos, None, pad)

    # Load checkpoint weights and prepare batched weights for C++ forward
    sd = tu.load_checkpoint_weights(checkpoint)
    print(f"Loaded {len(sd)} keys from checkpoint")

    # Check for NaNs in loaded weights
    nan_keys = [k for k, v in sd.items() if torch.is_tensor(v) and torch.isnan(v).any()]
    if nan_keys:
        print(f"⚠️  WARNING: {len(nan_keys)} keys contain NaN in checkpoint: {nan_keys[:5]}")

    sd = tu.pad_model_weights(sd, pad_obs_to=obs_dim)
    bw = tu.prepare_batched_weights(sd, num_layers=2, num_experts=8, device=device)
    print(f"Prepared batched weights: {len(bw)} keys")

    # Check for NaNs in batched weights
    nan_keys = [k for k, v in bw.items() if torch.is_tensor(v) and torch.isnan(v).any()]
    if nan_keys:
        print(f"⚠️  WARNING: {len(nan_keys)} keys contain NaN in batched weights: {nan_keys[:5]}")

    # Enable MoE workspace cache (pre-allocate CUTLASS buffers)
    lb.enable_forward_moe_workspace_cache(
        num_layers=2,
        max_batch_size=batch_size,
        max_seq_length=seq_len,
        hidden_dim=256,
        top_k=2
    )

    # C++ forward
    result = lb.forward_packed(
        obs, acts, agent, pos, bw, policy_indices, pad,
        num_layers=2, num_heads=4, hidden_dim=256, num_experts=8, top_k=2,
        count_pad=4, tflag_pad=3,
    )

    # Disable cache after use
    lb.disable_forward_moe_workspace_cache()
    # Unpack: (action_logits, opp_logits, state_values, win_logits, timers_dict)
    a_cpp, o_cpp, v_cpp, w_cpp = result[:4]

    # Check for NaNs/Infs before comparison
    print("\nChecking for invalid values...")
    for name, tensor in [('action_cpp', a_cpp), ('opp_cpp', o_cpp), ('value_cpp', v_cpp), ('win_cpp', w_cpp)]:
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan or has_inf:
            print(f"⚠️  {name}: NaN={has_nan}, Inf={has_inf}")
    for name, tensor in [('action_ts', a_ts), ('opp_ts', o_ts), ('value_ts', v_ts), ('win_ts', w_ts)]:
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan or has_inf:
            print(f"⚠️  {name}: NaN={has_nan}, Inf={has_inf}")

    print("\nEval parity (TorchScript vs C++):")
    print(f"Shapes: C++ action={a_cpp.shape}, TS action={a_ts.shape}")
    ok = True
    ok &= compare_tensors('action', a_cpp, a_ts)
    ok &= compare_tensors('opp',    o_cpp, o_ts)
    ok &= compare_tensors('values', v_cpp, v_ts)
    ok &= compare_tensors('win',    w_cpp, w_ts)

    print("\nArgmax parity:")
    tu.argmax_parity('action', a_cpp, a_ts)
    tu.argmax_parity('opp',    o_cpp, o_ts)

    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description='Eval parity: TorchScript vs C++ forward')
    ap.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    ap.add_argument('--script', type=str, required=True, help='Path to TorchScript model (.pt)')
    ap.add_argument('--device', type=str, default='cuda', help='Device to run on')
    ap.add_argument('--batch', type=str, default=None, help='Optional torch.save batch file with real game data')
    ap.add_argument('--batch-size', type=int, default=4, help='Batch size for synthetic inputs')
    ap.add_argument('--seq-len', type=int, default=32, help='Sequence length for synthetic inputs')
    ap.add_argument('--obs-dim', type=int, default=16, help='Observation dimension (padded)')
    args = ap.parse_args()

    batch_path = Path(args.batch) if args.batch else None
    return run_parity(
        Path(args.checkpoint),
        Path(args.script),
        args.device,
        batch_path,
        args.batch_size,
        args.seq_len,
        args.obs_dim
    )


if __name__ == '__main__':
    raise SystemExit(main())
