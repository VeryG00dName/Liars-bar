#!/usr/bin/env python3
"""
Print all state_dict keys (and shapes) from a TorchScript checkpoint.

Usage:
  python scripts/print_ts_keys.py checkpoints/test97/gen_1/final_traced.pt
"""
import sys
from pathlib import Path
import torch


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/print_ts_keys.py <path_to_torchscript.pt>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    m = torch.jit.load(path, map_location="cpu")
    sd = m.state_dict()

    for k in sorted(sd.keys()):
        print(f"{k} {tuple(sd[k].shape)}")
    print(f"\nTotal keys: {len(sd)}")


if __name__ == "__main__":
    main()
