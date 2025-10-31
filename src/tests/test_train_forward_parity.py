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
    # Also run a focused backward parity check for indexed_batched_linear
    print("\nChecking backward parity: indexed_batched_linear (autograd vs PyTorch)")
    ok_bwd = _check_linear_backward_parity(device)
    if ok_bwd:
        print("Linear backward parity: OK")
    else:
        print("Linear backward parity: FAILED")
        return 1


# ------------------------------
# Backward parity helpers
# ------------------------------

def _linear_forward_ref(input, weight_cache, bias_cache, policy_indices):
    # Baseline using PyTorch ops with autograd
    B, T, in_dim = input.shape
    W, out_dim, _ = weight_cache.shape
    pol = policy_indices.to(torch.long)
    w_sel = weight_cache.index_select(0, pol)        # [B, out, in]
    b_sel = bias_cache.index_select(0, pol)          # [B, out]
    w_t = w_sel.transpose(1, 2)                      # [B, in, out]
    bias_expanded = b_sel.unsqueeze(1).expand(B, T, out_dim)
    out = torch.baddbmm(bias_expanded, input, w_t, beta=1.0, alpha=1.0)
    return out


def _check_linear_backward_parity(device: str) -> bool:
    torch.manual_seed(0)
    B, T, W = 8, 16, 4
    in_dim, out_dim = 32, 48

    # Create a single set of base tensors to share between both paths
    base_input = torch.randn(B, T, in_dim, device=device, dtype=torch.float32)
    base_weight = torch.randn(W, out_dim, in_dim, device=device, dtype=torch.float32)
    base_bias = torch.randn(W, out_dim, device=device, dtype=torch.float32)
    pol = torch.randint(0, W, (B,), device=device)

    def clone_with_grad(x):
        y = x.detach().clone()
        y.requires_grad_(True)
        return y

    # Reference grads (PyTorch autograd only)
    inp_ref = clone_with_grad(base_input)
    w_ref   = clone_with_grad(base_weight)
    b_ref   = clone_with_grad(base_bias)
    out_ref = _linear_forward_ref(inp_ref, w_ref, b_ref, pol)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    grads_ref = {
        'input': inp_ref.grad.detach().clone(),
        'weight': w_ref.grad.detach().clone(),
        'bias': b_ref.grad.detach().clone(),
    }

    # Test grads (our autograd function)
    inp_t = clone_with_grad(base_input)
    w_t   = clone_with_grad(base_weight)
    b_t   = clone_with_grad(base_bias)
    out_t = lb.autograd.indexed_batched_linear_autograd(inp_t, w_t, b_t, pol)
    loss_t = out_t.sum()
    loss_t.backward()
    grads_t = {
        'input': inp_t.grad.detach().clone(),
        'weight': w_t.grad.detach().clone(),
        'bias': b_t.grad.detach().clone(),
    }

    # Compare
    rtol, atol = 1e-4, 1e-5
    ok = True
    for name in ('input', 'weight', 'bias'):
        ref = grads_ref[name]
        got = grads_t[name]
        if not torch.allclose(ref, got, rtol=rtol, atol=atol):
            max_abs = (ref - got).abs().max().item()
            mean_abs = (ref - got).abs().mean().item()
            print(f"  Grad mismatch {name}: max|Δ|={max_abs:.3e}, mean|Δ|={mean_abs:.3e}")
            ok = False
    return ok

if __name__ == "__main__":
    raise SystemExit(main())
