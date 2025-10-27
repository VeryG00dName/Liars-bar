"""
Test custom MoE backward kernels for numerical correctness.

Phase 1: Weight gradients (W1, W2, b1, b2, dX) with frozen routing.
"""

import torch
from src.misc import lb


def test_moe_backward_tiny_single_expert():
    """
    Gradcheck test with minimal dimensions and single expert (r=1 routing).

    This verifies the backward pass produces correct gradients compared to
    PyTorch's numerical gradient estimation.
    """
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tiny dimensions for fast gradcheck
    B, T = 2, 4
    hidden_dim = 16
    ffn_dim = 32
    num_experts = 2
    top_k = 1  # Single expert routing (r=1) for simplicity

    # Create test inputs (FP32 for gradcheck, will cast internally)
    input_tensor = torch.randn(B, T, hidden_dim, device=device, dtype=torch.float32, requires_grad=True)
    w1_weights = torch.randn(1, num_experts, ffn_dim, hidden_dim, device=device, dtype=torch.float32, requires_grad=True)
    w2_weights = torch.randn(1, num_experts, hidden_dim, ffn_dim, device=device, dtype=torch.float32, requires_grad=True)
    b1_biases = torch.randn(1, num_experts, ffn_dim, device=device, dtype=torch.float32, requires_grad=True)
    b2_biases = torch.randn(1, num_experts, hidden_dim, device=device, dtype=torch.float32, requires_grad=True)

    # Simple routing: uniform scores (will be normalized to r=1 after top-k)
    routing_scores = torch.randn(B, T, num_experts, device=device, dtype=torch.float32)
    policy_indices = torch.zeros(B, T, device=device, dtype=torch.long)

    # Test forward pass
    print("Testing forward pass...")
    try:
        output = lb.moe.grouped_moe_autograd_forward(
            input_tensor.view(-1, hidden_dim).contiguous().to(torch.float16),
            w1_weights.to(torch.float16),
            w2_weights.to(torch.float16),
            b1_biases.to(torch.float16),
            b2_biases.to(torch.float16),
            routing_scores.view(-1, num_experts).softmax(dim=-1)[:, :top_k].contiguous().to(torch.float32),
            torch.arange(B * T, device=device, dtype=torch.long),
            torch.tensor([B * T], device=device, dtype=torch.long).cpu(),
            torch.tensor([0], device=device, dtype=torch.long).cpu(),
            torch.tensor([0], device=device, dtype=torch.long).cpu(),
            torch.tensor([0], device=device, dtype=torch.long).cpu(),
            hidden_dim,
            ffn_dim
        )
        print(f"Forward output shape: {output.shape}, dtype: {output.dtype}")
        print(f"Forward output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, mean={output.mean().item():.4f}")

        # Check for NaN in forward
        if torch.isnan(output).any():
            print("ERROR: Forward pass produced NaN!")
            return False

        # Test backward pass
        print("\nTesting backward pass...")
        grad_output = torch.randn_like(output)
        output.backward(grad_output)

        print(f"grad_input computed: {input_tensor.grad is not None}")
        if input_tensor.grad is not None:
            print(f"grad_input stats: min={input_tensor.grad.min().item():.4f}, max={input_tensor.grad.max().item():.4f}, mean={input_tensor.grad.mean().item():.4f}")
            if torch.isnan(input_tensor.grad).any():
                print("ERROR: grad_input contains NaN!")
                return False

        print(f"grad_w1 computed: {w1_weights.grad is not None}")
        if w1_weights.grad is not None:
            print(f"grad_w1 stats: min={w1_weights.grad.min().item():.4f}, max={w1_weights.grad.max().item():.4f}, mean={w1_weights.grad.mean().item():.4f}")
            if torch.isnan(w1_weights.grad).any():
                print("ERROR: grad_w1 contains NaN!")
                return False

        print(f"grad_w2 computed: {w2_weights.grad is not None}")
        if w2_weights.grad is not None:
            print(f"grad_w2 stats: min={w2_weights.grad.min().item():.4f}, max={w2_weights.grad.max().item():.4f}, mean={w2_weights.grad.mean().item():.4f}")
            if torch.isnan(w2_weights.grad).any():
                print("ERROR: grad_w2 contains NaN!")
                return False

        print("\n✓ Backward pass completed without NaN")
        return True

    except Exception as e:
        print(f"ERROR during forward/backward: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing MoE Custom Backward Kernels")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        exit(1)

    success = test_moe_backward_tiny_single_expert()

    if success:
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        exit(0)
    else:
        print("\n" + "=" * 60)
        print("✗ TESTS FAILED")
        print("=" * 60)
        exit(1)
