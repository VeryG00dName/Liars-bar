#!/usr/bin/env python3
"""
Parity test: Training forward (autograd path) vs C++ inference forward.

Asserts that for identical weights and identical inputs, the four heads
(action, opp, values, win) match within a small tolerance.

Notes:
- Uses all-valid action_masks so masking does not alter logits.
- Disables TF32 for matmuls to reduce tolerance issues on Ampere+.
"""

from __future__ import annotations

import torch

from src.model.ppo_reactive_model import PPOReactiveModel
from src.misc import lb
from src.tests import test_utils as tu


def _prepare_batched_from_model(model: PPOReactiveModel, device: str = "cuda"):
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return tu.prepare_batched_weights(
        state_dict,
        num_layers=len(model.transformer.layers),
        num_experts=model.num_experts,
        device=device,
        to_fp16=True,
    )


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Model with standard dims
    model = PPOReactiveModel(
        obs_dim=16
    ).to(device)
    model.eval()

    # Inputs
    obs, acts, agent, pos, amask, pad = tu.create_dummy_inputs(device=device, batch_size=32, seq_len=64, obs_dim=16, seed=1234)

    # Training forward (autograd path)
    with torch.no_grad():
        trn = model(
            obs_sequence=obs,
            action_sequence=acts,
            agent_types=agent,
            positions=pos,
            action_masks=amask,
            padding_mask=pad,
        )
    a_tr, o_tr, v_tr, w_tr, _, _ = trn

    # C++ inference forward with the same weights
    bw = _prepare_batched_from_model(model, device=device)
    policy_indices = torch.zeros(obs.size(0), dtype=torch.long, device=device)
    a_inf, o_inf, v_inf, w_inf = lb.forward_packed(
        obs, acts, agent, pos, bw, policy_indices, pad,
        num_layers=len(model.transformer.layers),
        num_heads=4,
        hidden_dim=model.hidden_dim,
        num_experts=model.num_experts,
        top_k=model.top_k,
        count_pad=model.count_pad,
        tflag_pad=model.tflag_pad
    )

    # Metrics helper
    # Compare (prints metrics regardless, fails if any mismatches)
    print("\nParity metrics (rtol=1e-3, atol=1e-4):")
    rtol, atol = 1e-3, 1e-4
    ok_action = tu.report_parity_metrics("action", a_tr, a_inf, rtol, atol)
    ok_opp    = tu.report_parity_metrics("opp",    o_tr, o_inf, rtol, atol)
    ok_value  = tu.report_parity_metrics("values", v_tr, v_inf, rtol, atol)
    ok_win    = tu.report_parity_metrics("win",    w_tr, w_inf, rtol, atol)

    # Argmax/top-1 parity for categorical heads
    print("\nArgmax parity:")
    arg_action = tu.argmax_parity("action", a_tr, a_inf)
    arg_opp    = tu.argmax_parity("opp",    o_tr, o_inf)

    ok = ok_action and ok_opp and ok_value and ok_win
    if ok:
        print("\nTrain forward parity: OK")
        return 0
    else:
        print("\nTrain forward parity: FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
