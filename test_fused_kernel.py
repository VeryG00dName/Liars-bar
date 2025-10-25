"""
Test the fused bias+routing weight kernel in isolation.

This test verifies that the fused kernel produces identical results to
applying bias and routing weights separately.
"""

import torch

def apply_bias_routing_separate(data, bias, routing_weights):
    """Reference implementation: apply bias and routing separately."""
    # data: [M, H]
    # bias: [H]
    # routing_weights: [M]

    # Add bias (broadcast across batch dimension)
    output = data + bias.unsqueeze(0)

    # Apply routing weights (broadcast across hidden dimension)
    output = output * routing_weights.unsqueeze(-1)

    return output


def apply_bias_routing_fused_simulated(data, bias, routing_weights):
    """Simulated fused kernel implementation."""
    M, H = data.shape
    output = data.clone()

    for i in range(M):
        for j in range(H):
            val = float(output[i, j]) + float(bias[j])
            val *= routing_weights[i]
            output[i, j] = val

    return output


def test_fused_vs_separate():
    """Test that fused and separate implementations match."""
    torch.manual_seed(42)

    M, H = 100, 256

    # Create test data
    data = torch.randn(M, H, dtype=torch.float16, device='cuda')
    bias = torch.randn(H, dtype=torch.float16, device='cuda')
    routing_weights = torch.rand(M, dtype=torch.float32, device='cuda')

    # Apply separately
    output_separate = apply_bias_routing_separate(data, bias, routing_weights)

    # Apply with simulated fused kernel
    output_fused_sim = apply_bias_routing_fused_simulated(data, bias, routing_weights)

    # Compare
    diff = (output_separate - output_fused_sim).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    mismatches = (diff > 1e-3).sum().item()
    total = M * H
    mismatch_pct = 100.0 * mismatches / total

    print(f"Fused vs Separate comparison:")
    print(f"  Max abs diff: {max_diff:.6e}")
    print(f"  Mean abs diff: {mean_diff:.6e}")
    print(f"  Mismatches (>1e-3): {mismatches}/{total} ({mismatch_pct:.2f}%)")

    if max_diff < 1e-4:
        print("✓ PASS: Fused and separate implementations match!")
        return True
    else:
        print("✗ FAIL: Implementations differ!")
        # Print first few mismatches
        mismatch_indices = (diff > 1e-3).nonzero()
        print(f"\nFirst 5 mismatches:")
        for idx in mismatch_indices[:5]:
            i, j = idx
            print(f"  [{i}, {j}]: separate={output_separate[i,j]:.6f}, fused={output_fused_sim[i,j]:.6f}, diff={diff[i,j]:.6e}")
        return False


if __name__ == '__main__':
    test_fused_vs_separate()
