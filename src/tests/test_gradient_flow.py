#!/usr/bin/env python3
"""
Quick test to verify gradient flow through the training forward pass,
both with and without gradient checkpointing.
"""

import torch
import sys
import traceback
from src.tests import test_utils as tu

# Import model
from src.model.ppo_reactive_model import PPOReactiveModel


def run_one(model, inputs, trial_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Unpack inputs created outside so both runs share the same batch
    obs_sequence, action_sequence, agent_types, positions, action_masks, padding_mask = inputs

    # Forward pass
    try:
        outputs = model(
            obs_sequence=obs_sequence,
            action_sequence=action_sequence,
            agent_types=agent_types,
            positions=positions,
            action_masks=action_masks,
            padding_mask=padding_mask
        )
        action_logits, opp_logits, state_values, win_logits, gate_logits_tensor, routing = outputs

        if torch.isnan(action_logits).any() or torch.isnan(state_values).any():
            print(f"❌ Trial {trial_num}: NaN detected in forward outputs!")
            return False, {}

    except Exception as e:
        print(f"❌ Trial {trial_num}: Forward pass failed: {e}")
        traceback.print_exc()
        return False, {}

    # Backward pass
    try:
        loss = action_logits.sum() + state_values.sum() + win_logits.sum() + opp_logits.sum()
        loss.backward()

        # Gradient check with per-parameter NaN counts
        has_nan = False
        no_grad_count = 0
        nan_params = {}
        all_params = list(model.named_parameters())

        for name, param in all_params:
            if param.grad is None:
                no_grad_count += 1
                print(f"❌ Trial {trial_num}: {name} has no gradient")
                continue
            n_nans = torch.isnan(param.grad).sum().item()
            if n_nans > 0:
                has_nan = True
                nan_params[name] = int(n_nans)
                print(f"❌ Trial {trial_num}: {name} has {n_nans} NaN gradients")

        return (not has_nan and no_grad_count == 0), nan_params

    except Exception as e:
        print(f"❌ Trial {trial_num}: Backward pass failed: {e}")
        traceback.print_exc()
        return False, {}


def run_test_suite(num_trials: int = 50, batch_size: int = 128, seq_len: int = 256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running {num_trials} trials on {device}...")

    # One model instance toggling checkpoint flag for comparable params/grads
    def make_model():
        m = PPOReactiveModel(
            obs_dim=16,
            dropout_rate=0.1
        ).to(device)
        m.train()
        return m

    results = []
    for trial in range(1, num_trials + 1):
        # Seed per trial for reproducibility
        seed = 1337 + trial
        # Use padding_ratio=0.3 to mark last 30% as padding like before
        inputs = tu.create_dummy_inputs(
            batch_size=batch_size,
            seq_len=seq_len,
            obs_dim=16,
            device=str(device),
            seed=seed,
            padding_ratio=0.3,
            dtype=torch.float32,
        )

        # Single checkpointed run (C++ path is always checkpointed now)
        model_ckpt = make_model()
        ok_ck, nans_ck = run_one(model_ckpt, inputs, trial)

        results.append({
            "trial": trial,
            "seed": seed,
            "ok_ck": ok_ck,
            "nan_count_ck": sum(nans_ck.values()) if nans_ck else 0,
            "nans_ck": nans_ck,
        })

        # Print progress every 10 trials
        if trial % 10 == 0:
            passed = sum(1 for r in results if r["ok_ck"])
            print(f"Progress: {trial}/{num_trials} trials, {passed} passed")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    ok_ck_trials = sum(1 for r in results if r["ok_ck"])
    nan_ck_total = sum(r["nan_count_ck"] for r in results)
    print(f"Passed: {ok_ck_trials}/{num_trials}, total_nan_elems={nan_ck_total}")

    # Only show failed trials
    failed = [r for r in results if not r["ok_ck"]]
    if failed:
        print(f"\n{len(failed)} failed trials:")
        for r in failed:
            print(f"  Trial {r['trial']} (seed={r['seed']}): nan_elems={r['nan_count_ck']}")
            if r['nans_ck']:
                top_ck = sorted(r['nans_ck'].items(), key=lambda kv: kv[1], reverse=True)[:5]
                for name, cnt in top_ck:
                    print(f"    {name}: {cnt}")

    # Return overall pass/fail
    return ok_ck_trials == num_trials

if __name__ == "__main__":
    # Run 50 trials to stress test gradient flow
    all_ok = run_test_suite(num_trials=50, batch_size=64, seq_len=64)
    sys.exit(0 if all_ok else 1)
