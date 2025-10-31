#!/usr/bin/env python3
"""
Eval parity: TorchScript model vs C++ forward_packed on recorded inputs.

Usage:
  python -m src.tests.test_eval_parity_torchscript \
    --checkpoint checkpoints/test76/gen_15/final.pth \
    --script     checkpoints/test76/gen_15/final_traced.pt \
    --reference  reference_outputs_single.pkl \
    --device     cuda

This uses recorded inputs (not synthetic) and compares the four heads:
  action_logits, opp_logits, state_values, win_logits
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch

from src.misc import lb
from src.tests.test_cpp_layers import load_reference, load_checkpoint_weights, pad_model_weights, prepare_batched_weights
from src.tests import test_utils as tu


def compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor, rtol=1e-3, atol=1e-4) -> bool:
    return tu.compare_tensors(name, a, b, rtol=rtol, atol=atol, verbose=True)


def _load_inputs(reference_pkl: Path, batch_file: Path | None, device: str) -> Tuple[torch.Tensor, ...]:
    if batch_file is not None and batch_file.exists():
        data = torch.load(str(batch_file), map_location=device)
        obs = data['obs_sequence'].to(device)
        acts = data['action_sequence'].to(device)
        agent = data['agent_types'].to(device)
        pos = data['positions'].to(device)
        pad = data.get('padding_mask')
        if pad is not None:
            pad = pad.to(device)
        return obs, acts, agent, pos, pad
    # Fallback to pickle reference
    ref: Dict[str, torch.Tensor] = load_reference(reference_pkl)
    obs = ref['input/obs_sequence'].to(device)
    acts = ref['input/action_sequence'].to(device)
    agent = ref['input/agent_types'].to(device)
    pos = ref['input/positions'].to(device)
    pad = ref.get('input/padding_mask')
    if pad is not None:
        pad = pad.to(device)
    return obs, acts, agent, pos, pad


def run_parity(checkpoint: Path, script_pt: Path, reference_pkl: Path, device: str, batch_file: Path | None) -> int:
    # Load inputs (prefer real batch if provided)
    obs, acts, agent, pos, pad = _load_inputs(reference_pkl, batch_file, device)

    # Policy indices (single-policy eval)
    B = obs.size(0)
    policy_indices = torch.zeros(B, dtype=torch.long, device=device)

    # Load TorchScript model and eval
    ts = torch.jit.load(str(script_pt), map_location=device)
    ts.eval()
    # TorchScript model may have been authored with smaller obs_dim (e.g., 9) while
    # C++ path pads obs_encoder weight to 16 for CUDA alignment. Slice inputs for TS
    # to its expected in-dim so both paths see consistent content (pad columns are zeros).
    
    obs_ts = obs[..., :9]
    with torch.no_grad():
        a_ts, o_ts, v_ts, w_ts = ts(obs_ts, acts, agent, pos, None, pad)

    # Load checkpoint weights and prepare batched weights for C++ forward
    sd = load_checkpoint_weights(checkpoint)
    sd = pad_model_weights(sd, pad_obs_to=obs.size(-1))
    bw = prepare_batched_weights(sd, num_layers=2, num_experts=8, device=device)

    # C++ forward
    a_cpp, o_cpp, v_cpp, w_cpp = lb.forward_packed(
        obs, acts, agent, pos, bw, policy_indices, pad,
        num_layers=2, num_heads=4, hidden_dim=256, num_experts=8, top_k=2,
        count_pad=4, tflag_pad=3,
    )

    print("\nEval parity (TorchScript vs C++):")
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
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--script', type=str, required=True)
    ap.add_argument('--reference', type=str, required=True, help='reference_outputs_single.pkl')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--batch', type=str, default=None, help='Optional torch.save batch file with real game data')
    args = ap.parse_args()

    batch_path = Path(args.batch) if args.batch else None
    return run_parity(Path(args.checkpoint), Path(args.script), Path(args.reference), args.device, batch_path)


if __name__ == '__main__':
    raise SystemExit(main())
