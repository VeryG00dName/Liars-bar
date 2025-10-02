#!/usr/bin/env python3
import argparse
import traceback
import torch
from src.model.ppo_reactive_model import PPOReactiveModel

def build_example_inputs(
    B: int,
    L: int,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
):
    # dtypes/shapes match your forward’s expectations
    obs_sequence    = torch.randn(B, L, obs_dim, device=device)
    action_sequence = torch.zeros(B, L, dtype=torch.long, device=device)      # (B,T)
    agent_types     = torch.zeros(B, L, dtype=torch.long, device=device)      # (B,T)
    positions       = torch.arange(L, device=device).unsqueeze(0).expand(B, L)# (B,T)
    action_masks    = torch.ones(B, L, action_dim, dtype=torch.bool, device=device)  # (B,T,A)
    padding_mask    = torch.zeros(B, L, dtype=torch.bool, device=device)      # (B,T)
    # valid_lengths uses default None — we just omit it from args
    return (
        obs_sequence,
        action_sequence,
        agent_types,
        positions,
        action_masks,
        padding_mask,
    )

def main():
    parser = argparse.ArgumentParser(description="torch.export smoke test (batch-dynamic, seq-static per bucket)")
    parser.add_argument("--obs-dim", type=int, default=9, help="Observation feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden size (must match model init if you changed it)")
    parser.add_argument("--action-dim", type=int, default=7, help="Action dimension (model default: 7)")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--buckets", type=int, nargs="+", default=[64, 128, 192, 256], help="Seq-length buckets to test")
    parser.add_argument("--test-batches", type=int, nargs="+", default=[2, 6],help="Batch sizes to test against the exported program")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--strict", type=lambda x: str(x).lower() == "true", default=False,
                        help="Use export(strict=True). Default False is more tolerant.")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Build model
    model = PPOReactiveModel(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.heads,
        num_layers=args.layers,
        max_seq_length=args.max_seq_length,
        use_gradient_checkpointing=False,  # rollout path; keep eval()
    ).to(device).eval()

    # Optional: show a quick eager forward to confirm shapes
    with torch.no_grad():
        ex_args = build_example_inputs(B=2, L=min(args.buckets), obs_dim=args.obs_dim,
                                       action_dim=args.action_dim, device=device)
        a_logits, o_logits, values = model(*ex_args)
        print(f"[INFO] Eager forward OK: action_logits {tuple(a_logits.shape)} "
              f"opp_logits {tuple(o_logits.shape)} values {tuple(values.shape)}")

    # Export per bucket with only batch dynamic
    from torch.export import export, Dim
    Bsym = Dim("batch", min=1)

    ok = True
    exported = {}

    for L in args.buckets:
        print(f"\n[INFO] === Exporting bucket T={L} (batch dynamic)… ===")
        if L > args.max_seq_length:
            print(f"[WARN] L={L} exceeds model.max_seq_length={args.max_seq_length}; skipping.")
            continue

        # Build sample inputs (B=2) for tracing
        ex_args = build_example_inputs(B=2, L=L, obs_dim=args.obs_dim,
                                       action_dim=args.action_dim, device=device)

        # dynamic shapes: ONLY dim-0 (batch) is dynamic
        dyn = {
            "obs_sequence":    {0: Bsym},
            "action_sequence": {0: Bsym},
            "agent_types":     {0: Bsym},
            "positions":       {0: Bsym},
            "action_masks":    {0: Bsym},
            "padding_mask":    {0: Bsym},
            # valid_lengths omitted -> stays default None/static
        }

        try:
            ep = export(
                model,
                args=ex_args,            # pass only the first 6 args; valid_lengths uses default None
                dynamic_shapes=dyn,
                strict=args.strict,      # strict=False is usually friendlier for encoders/decoders
            )
            exported[L] = ep
            print(f"[OK] Export success for T={L}.")
        except Exception as e:
            ok = False
            print(f"[FAIL] Export failed for T={L}.\n{type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        # Runtime check: call the exported program with multiple batch sizes
        try:
            runner = exported[L].module()  # no .eval(), .to(), etc.
            for Btest in args.test_batches:
                with torch.no_grad():
                    test_args = build_example_inputs(
                        B=Btest, L=L, obs_dim=args.obs_dim,
                        action_dim=args.action_dim, device=device
                    )
                    a_logits, o_logits, values = runner(*test_args)

                    # Shape sanity checks
                    assert a_logits.shape == (Btest, L, args.action_dim)
                    assert o_logits.shape == (Btest, L, args.action_dim)
                    assert values.shape  == (Btest, L, 1)

                    print(f"[OK] Run exported(T={L}) with B={Btest} → "
                        f"action_logits {tuple(a_logits.shape)}, "
                        f"opp_logits {tuple(o_logits.shape)}, "
                        f"values {tuple(values.shape)}")
        except Exception as e:
            ok = False
            print(f"[FAIL] Executing exported(T={L}) failed.\n{type(e).__name__}: {e}")
            traceback.print_exc()

    if ok and exported:
        print("\n[SUCCESS] All requested buckets exported and executed with dynamic batch.")
        print("         You can now AOT-compile each ExportedProgram with Inductor if desired.")
    else:
        print("\n[DONE] Some buckets failed. See logs above for the first error per bucket.")
        print("      Tips:")
        print("       • Use --strict False (default here).")
        print("       • Ensure any SDPA/backend context is set at init, not inside forward.")
        print("       • If a particular op is unexportable, tiny rewrites (e.g., boolean masks, no Python branching) help.")

if __name__ == "__main__":
    main()
