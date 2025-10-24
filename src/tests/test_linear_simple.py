"""Simple test to debug indexed_batched_linear."""

import torch
import numpy as np
from src.misc import lb

# Create simple test data
torch.manual_seed(42)
np.random.seed(42)

# Small dimensions for easy debugging
B = 2  # batch
T = 3  # time steps
In_Dim = 4  # input dim
Out_Dim = 5  # output dim
W = 1  # num policies

# Create input
input_tensor = torch.randn(B, T, In_Dim, device='cuda', dtype=torch.float32)
print("Input shape:", input_tensor.shape)
print("Input:\n", input_tensor)

# Create weight and bias
weight = torch.randn(Out_Dim, In_Dim, device='cuda', dtype=torch.float32)
bias = torch.randn(Out_Dim, device='cuda', dtype=torch.float32)
print("\nWeight shape:", weight.shape)
print("Weight:\n", weight)
print("\nBias shape:", bias.shape)
print("Bias:\n", bias)

# Create batched weights (add policy dimension)
weight_batched = weight.unsqueeze(0)  # [1, Out_Dim, In_Dim]
bias_batched = bias.unsqueeze(0)  # [1, Out_Dim]
print("\nWeight batched shape:", weight_batched.shape)
print("\nBias batched shape:", bias_batched.shape)

# Policy indices (all samples use policy 0)
policy_indices = torch.zeros(B, dtype=torch.long, device='cuda')
print("\nPolicy indices:", policy_indices)

# PyTorch reference
print("\n" + "="*80)
print("PyTorch Reference (F.linear)")
print("="*80)
pytorch_output = torch.nn.functional.linear(input_tensor, weight, bias)
print("PyTorch output shape:", pytorch_output.shape)
print("PyTorch output:\n", pytorch_output)

# C++ implementation
print("\n" + "="*80)
print("C++ indexed_batched_linear")
print("="*80)
cpp_output = lb.test_indexed_batched_linear(
    input_tensor,
    weight_batched,
    bias_batched,
    policy_indices
)
print("C++ output shape:", cpp_output.shape)
print("C++ output:\n", cpp_output)

# Compare
print("\n" + "="*80)
print("Comparison")
print("="*80)
diff = (pytorch_output - cpp_output).abs()
print("Max abs diff:", diff.max().item())
print("Mean abs diff:", diff.mean().item())
print("\nDifference:\n", diff)

if diff.max().item() < 1e-4:
    print("\n✅ PASS: Outputs match!")
else:
    print("\n❌ FAIL: Outputs don't match!")
    print("\nPyTorch - C++ (first sample):")
    print(pytorch_output[0] - cpp_output[0])
