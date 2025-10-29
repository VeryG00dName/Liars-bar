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


def create_dummy_inputs(batch_size=32, seq_len=64, obs_dim=16, device="cuda"):
    gen = torch.Generator(device=device)
    gen.manual_seed(1234)
    obs_sequence    = torch.randn(batch_size, seq_len, obs_dim, dtype=torch.float32, device=device, generator=gen)
    action_sequence = torch.randint(0, 11, (batch_size, seq_len), dtype=torch.long, device=device, generator=gen)
    agent_types     = torch.randint(0, 3,  (batch_size, seq_len), dtype=torch.long, device=device, generator=gen)
    positions       = torch.randint(0, 4,  (batch_size, seq_len), dtype=torch.long, device=device, generator=gen)
    padding_mask    = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    action_masks    = torch.ones(batch_size, seq_len, 7, dtype=torch.bool, device=device)
    return obs_sequence, action_sequence, agent_types, positions, padding_mask, action_masks


def prepare_batched_weights_from_model(model: PPOReactiveModel, device: str = "cuda"):
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    # Pad obs dim if needed inside weight utils path
    num_layers = len(model.transformer.layers)
    num_experts = model.num_experts
    sd = lb.prestack_moe_expert_weights(state_dict, num_layers, num_experts)
    batched: dict[str, torch.Tensor] = {}
    for k, t in sd.items():
        batched[k] = t.unsqueeze(0).to(device)
    # Convert ONLY MoE expert weights/biases to fp16 (kernels expect Half).
    # Keep other weights (encoder/attn/heads/LN) in float32 to match forward_packed expectations.
    if device.startswith("cuda"):
        for k, t in list(batched.items()):
            if not t.is_floating_point():
                continue
            if ".moe.experts." in k:
                batched[k] = t.to(torch.float16).contiguous()
            else:
                batched[k] = t.to(torch.float32).contiguous()
    batched = lb.create_moe_weight_pointers(batched, num_layers, num_experts)
    batched = lb.add_fixed_buffers(batched, device)
    return batched


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Model with standard dims
    model = PPOReactiveModel(
        obs_dim=16,
        use_gradient_checkpointing=True,
    ).to(device)
    model.eval()

    # Inputs
    obs, acts, agent, pos, pad, amask = create_dummy_inputs(device=device)

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
    bw = prepare_batched_weights_from_model(model, device=device)
    policy_indices = torch.zeros(obs.size(0), dtype=torch.long, device=device)
    a_inf, o_inf, v_inf, w_inf = lb.forward_packed(
        obs, acts, agent, pos, bw, policy_indices, pad,
        num_layers=len(model.transformer.layers),
        num_heads=4,
        hidden_dim=model.hidden_dim,
        num_experts=model.num_experts,
        top_k=model.top_k,
        count_pad=model.count_pad,
        tflag_pad=model.tflag_pad,
    )

    # Metrics helper
    def report(name: str, a: torch.Tensor, b: torch.Tensor, rtol=1e-3, atol=1e-4) -> bool:
        a = a.detach().float().cpu()
        b = b.detach().float().cpu()
        abs_diff = (a - b).abs()
        tol = atol + rtol * b.abs()
        mism = (abs_diff > tol)
        mism_count = int(mism.sum().item())
        total = b.numel()
        mism_pct = 100.0 * mism_count / max(total, 1)
        max_err = float(abs_diff.max().item())
        mean_err = float(abs_diff.mean().item())
        # Relative error (avoid div by zero)
        rel_err = (abs_diff / (b.abs() + 1e-8)).clamp_max(1e6)
        mean_rel = float(rel_err.mean().item())
        print(f"{name:8s}: mism={mism_count}/{total} ({mism_pct:6.3f}%), max|Δ|={max_err:.3e}, mean|Δ|={mean_err:.3e}, mean rel={mean_rel:.3e}")
        return mism_count == 0

    # Compare (prints metrics regardless, fails if any mismatches)
    print("\nParity metrics (rtol=1e-3, atol=1e-4):")
    rtol, atol = 1e-3, 1e-4
    ok_action = report("action", a_tr, a_inf, rtol, atol)
    ok_opp    = report("opp",    o_tr, o_inf, rtol, atol)
    ok_value  = report("values", v_tr, v_inf, rtol, atol)
    ok_win    = report("win",    w_tr, w_inf, rtol, atol)

    # Argmax/top-1 parity for categorical heads
    def argmax_parity(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
        # a,b: [B, T, A]
        a_idx = a.argmax(dim=-1)
        b_idx = b.argmax(dim=-1)
        eq = (a_idx == b_idx)
        total = eq.numel()
        same = int(eq.sum().item())
        pct = 100.0 * same / max(total, 1)
        # Also report how often the top-1 margin is tiny (near-tie), where flips could happen
        def top1_margin(x: torch.Tensor) -> torch.Tensor:
            # margin = top1 - top2 per position
            top2 = torch.topk(x, k=2, dim=-1).values
            return (top2[..., 0] - top2[..., 1]).abs()
        a_margin = top1_margin(a).detach().float().cpu()
        b_margin = top1_margin(b).detach().float().cpu()
        near_tie = ((a_margin < 1e-3) | (b_margin < 1e-3)).sum().item()
        print(f"{name:8s} argmax parity: {same}/{total} ({pct:6.3f}%)  near-ties(<1e-3): {near_tie}")
        return same == total

    print("\nArgmax parity:")
    arg_action = argmax_parity("action", a_tr, a_inf)
    arg_opp    = argmax_parity("opp",    o_tr, o_inf)

    ok = ok_action and ok_opp and ok_value and ok_win
    if ok:
        print("\nTrain forward parity: OK")
        return 0
    else:
        print("\nTrain forward parity: FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
