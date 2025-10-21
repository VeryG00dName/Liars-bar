#!/usr/bin/env python3

import argparse
import csv
import itertools
import time

import torch
from nvmath.linalg.advanced import Matmul

def parse_args():
    p = argparse.ArgumentParser(
        description="STRICT cuBLASLt plan coverage checker using nvmath-python Matmul.plan(). "
                    "No fallbacks. Fails if planning returns 0 algorithms when --strict is set."
    )
    # Base shapes
    p.add_argument("--M", type=int, default=32)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--batch", type=int, default=512)
    # Sweeps
    p.add_argument("--sweep", action="store_true", help="Sweep shapes and dtypes (default grid uses given M,N,K,B).", default=False)
    p.add_argument("--sweep-M", type=str, default="", help="Comma-separated list of M values.")
    p.add_argument("--sweep-N", type=str, default="", help="Comma-separated list of N values.")
    p.add_argument("--sweep-K", type=str, default="", help="Comma-separated list of K values.")
    p.add_argument("--sweep-batch", type=str, default="", help="Comma-separated list of batch sizes.")
    p.add_argument("--dtypes", type=str, default="fp16,bf16,fp32", help="Comma-separated: fp16,bf16,fp32")
    # Execution / timing (optional, only after a successful plan)
    p.add_argument("--exec", action="store_true", help="Also execute via Matmul.execute() and time it.")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    # Behavior
    p.add_argument("--strict", action="store_true", help="Exit with non-zero status if any config returns 0 algorithms.")
    # Output
    p.add_argument("--csv", type=str, default="", help="Optional CSV path for results.")
    return p.parse_args()

def select_dtype(name):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]

def pretty_time(s):
    if s < 1e-6: return f"{s*1e9:.2f} ns"
    if s < 1e-3: return f"{s*1e6:.2f} us"
    if s < 1.0:  return f"{s*1e3:.2f} ms"
    return f"{s:.2f} s"

def time_exec(mm, iters, warmup):
    # Warmup
    for _ in range(warmup):
        mm.execute()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mm.execute()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

def main():
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA device required"
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)

    print("=== Environment ===")
    print(f"GPU: {torch.cuda.get_device_name(dev)} (SM {props.major}.{props.minor})")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print("===================\n")

    Ms = [int(x) for x in args.sweep_M.split(",") if x.strip()] if args.sweep_M else [args.M]
    Ns = [int(x) for x in args.sweep_N.split(",") if x.strip()] if args.sweep_N else [args.N]
    Ks = [int(x) for x in args.sweep_K.split(",") if x.strip()] if args.sweep_K else [args.K]
    Bs = [int(x) for x in args.sweep_batch.split(",") if x.strip()] if args.sweep_batch else [args.batch]
    dtypes = [s.strip() for s in args.dtypes.split(",") if s.strip()]

    if not args.sweep:
        Ms, Ns, Ks, Bs = [args.M], [args.N], [args.K], [args.batch]

    rows = []
    bad = False

    for dt in dtypes:
        dtype = select_dtype(dt)
        for M in Ms:
            for N in Ns:
                for K in Ks:
                    for B in Bs:
                        # Allocate representative operands (Torch tensors)
                        A = torch.randn(B, M, K, device=dev, dtype=dtype).contiguous()
                        Wt = torch.randn(B, K, N, device=dev, dtype=dtype).contiguous()  # transpose already applied

                        # Plan with nvmath Matmul (HARD requirement)
                        mm = Matmul(A, Wt)
                        mm.plan()
                        num_algos = len(mm.algorithms)

                        exec_time = None
                        if args.exec and num_algos > 0:
                            exec_time = time_exec(mm, args.iters, args.warmup)
                        mm.free()

                        rows.append({
                            "dtype": dt,
                            "M": M, "N": N, "K": K, "batch": B,
                            "plan_algos": num_algos,
                            "exec_time_s": exec_time,
                            "exec_time_pretty": (pretty_time(exec_time) if exec_time is not None else ""),
                        })

                        status = f"PLAN={num_algos}"
                        if args.exec and exec_time is not None:
                            status += f" exec={pretty_time(exec_time)}"
                        print(f"{dt:4s} M={M:3d} N={N:3d} K={K:3d} B={B:4d} -> {status}")
                        if num_algos == 0:
                            bad = True

    # Summary
    print("\n=== Summary ===")
    for r in rows:
        line = f"{r['dtype']:4s} M={r['M']:3d} N={r['N']:3d} K={r['K']:3d} B={r['batch']:4d}  PLAN={r['plan_algos']}"
        if r["exec_time_pretty"]:
            line += f"  EXEC={r['exec_time_pretty']}"
        print(line)

    # CSV
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\nWrote CSV to {args.csv}")

    if args.strict and bad:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
