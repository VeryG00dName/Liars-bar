#!/usr/bin/env python3
"""
Profile training (forward + backward) on the autograd path.

Runs N steps with synthetic inputs and reports a simple timing breakdown
and optional operator-level profile via torch.profiler.

Usage examples:

  python -m src.tests.profile_training --steps 20 --batch-size 64 --seq-len 64 --device cuda
  python -m src.tests.profile_training --steps 10 --prof --prof-topk 25

"""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import torch
from torch import optim

from src.model.ppo_reactive_model import PPOReactiveModel
from src.tests import test_utils as tu


def run_once(model: PPOReactiveModel,
             batch: Tuple[torch.Tensor, ...],
             optimizer: optim.Optimizer,
             device: str,
             *,
             do_step: bool = True,
             clip_grad_norm: float | None = None) -> Tuple[float, float, float]:
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()

    # Forward
    f0 = time.perf_counter()
    action_logits, opp_logits, state_values, win_logits, _, _ = model(
        obs_sequence=batch[0],
        action_sequence=batch[1],
        agent_types=batch[2],
        positions=batch[3],
        action_masks=batch[4],
        padding_mask=batch[5],
    )
    torch.cuda.synchronize() if device == "cuda" else None
    f1 = time.perf_counter()

    # Backward (simple scalar loss)
    loss = (action_logits.sum() + opp_logits.sum() + state_values.sum() + win_logits.sum())
    optimizer.zero_grad(set_to_none=True)
    b0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize() if device == "cuda" else None
    b1 = time.perf_counter()

    # Optimizer step
    s0 = time.perf_counter()
    if clip_grad_norm and clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad_norm))
    if do_step:
        optimizer.step()
    torch.cuda.synchronize() if device == "cuda" else None
    s1 = time.perf_counter()

    return (f1 - f0) * 1e3, (b1 - b0) * 1e3, (s1 - s0) * 1e3  # ms


def main() -> int:
    ap = argparse.ArgumentParser(description="Profile training forward+backward")
    ap.add_argument("--steps", type=int, default=20, help="Number of measured steps")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup steps (not measured)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate (set 0 or use --no-step to skip updates)")
    ap.add_argument("--no-step", action="store_true", help="Profile forward/backward only (no optimizer.step)")
    ap.add_argument("--clip-grad", type=float, default=0.0, help="Clip grad norm if > 0")
    ap.add_argument("--prof", action="store_true", help="Enable torch.profiler for operator breakdown")
    ap.add_argument("--prof-topk", type=int, default=20, help="Rows to print from profiler table")
    args = ap.parse_args()

    device = args.device
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    # Model on device (training path uses autograd + C++ forward_packed_train)
    model = PPOReactiveModel(obs_dim=16).to(device)
    model.train()
    # Use Adam on all parameters to exercise backward graph fully
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Prepare a static batch for reproducibility and to isolate compute
    batch = tu.create_dummy_inputs(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        obs_dim=16,
        device=device,
        seed=1337,
        padding_ratio=0.3,
        dtype=dtype,
    )

    # Warmup
    for _ in range(args.warmup):
        run_once(model, batch, optimizer, device, do_step=(not args.no_step), clip_grad_norm=(args.clip_grad if args.clip_grad > 0 else None))

    # Measured runs (optionally with profiler)
    f_ms = []
    b_ms = []
    s_ms = []

    if args.prof:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for _ in range(args.steps):
                f, b, s = run_once(model, batch, optimizer, device, do_step=(not args.no_step), clip_grad_norm=(args.clip_grad if args.clip_grad > 0 else None))
                f_ms.append(f); b_ms.append(b); s_ms.append(s)
                prof.step()
        print("\nTop ops by CUDA time:")
        try:
            print(prof.key_averages().table(sort_by=("cuda_time_total" if device == "cuda" else "cpu_time_total"), row_limit=args.prof_topk))
        except Exception:
            print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=args.prof_topk))
    else:
        for _ in range(args.steps):
            f, b, s = run_once(model, batch, optimizer, device, do_step=(not args.no_step), clip_grad_norm=(args.clip_grad if args.clip_grad > 0 else None))
            f_ms.append(f); b_ms.append(b); s_ms.append(s)

    # Summary
    import statistics as stats
    def mean(xs):
        return stats.mean(xs) if xs else 0.0

    total_ms = [f + b + s for f, b, s in zip(f_ms, b_ms, s_ms)]
    print("\nTraining profile summary (ms per step):")
    print(f"  steps={args.steps}, warmup={args.warmup}, device={device}, dtype={args.dtype}")
    print(f"  forward : {mean(f_ms):8.3f}")
    print(f"  backward: {mean(b_ms):8.3f}")
    print(f"  step    : {mean(s_ms):8.3f}")
    print(f"  total   : {mean(total_ms):8.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
