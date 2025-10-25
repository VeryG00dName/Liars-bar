"""
Test index_add_ behavior to ensure we understand MoE scatter-add correctly.
"""

import torch

# Simulate: 4 tokens, 3 experts, top_k=2
# Token 0 routes to experts [0, 1]
# Token 1 routes to experts [1, 2]
# Token 2 routes to experts [0, 2]
# Token 3 routes to experts [0, 1]

top_k = 2
top_k_indices = torch.tensor([
    [0, 1],  # token 0
    [1, 2],  # token 1
    [0, 2],  # token 2
    [0, 1],  # token 3
])

routing_weights = torch.tensor([
    [0.6, 0.4],  # token 0
    [0.7, 0.3],  # token 1
    [0.5, 0.5],  # token 2
    [0.8, 0.2],  # token 3
], dtype=torch.float32)

# Flatten and sort by expert
flat_expert_indices = top_k_indices.reshape(-1)  # [0, 1, 1, 2, 0, 2, 0, 1]
flat_routing_weights = routing_weights.reshape(-1)  # [0.6, 0.4, 0.7, 0.3, 0.5, 0.5, 0.8, 0.2]

num_tokens = 4
token_indices = torch.arange(num_tokens)
expanded_token_indices = token_indices.unsqueeze(-1).expand(num_tokens, top_k).reshape(-1)
# [0, 0, 1, 1, 2, 2, 3, 3]

sort_result = torch.sort(flat_expert_indices)
sorted_expert_indices = sort_result.values
sort_order = sort_result.indices

sorted_token_indices = expanded_token_indices[sort_order]
sorted_routing_weights = flat_routing_weights[sort_order]

print("Flat expert indices:", flat_expert_indices.tolist())
print("Expanded token indices:", expanded_token_indices.tolist())
print()
print("After sorting by expert:")
print("Sorted expert indices:", sorted_expert_indices.tolist())
print("Sorted token indices:", sorted_token_indices.tolist())
print("Sorted routing weights:", sorted_routing_weights.tolist())
print()

# Simulate expert outputs (3 hidden dims for simplicity)
hidden_dim = 3
expert_outputs = torch.randn(8, hidden_dim) * sorted_routing_weights.unsqueeze(-1)

print("Expert outputs shape:", expert_outputs.shape)
print("Expert outputs (first 3):")
print(expert_outputs[:3])
print()

# Scatter-add back
moe_output = torch.zeros(num_tokens, hidden_dim)
moe_output.index_add_(0, sorted_token_indices, expert_outputs)

print("Final MoE output shape:", moe_output.shape)
print("Final MoE output:")
print(moe_output)
print()

# Verify: token 0 should have contributions from expert 0 and expert 1
print("Token 0 routes to experts [0, 1] with weights [0.6, 0.4]")
print("Token 0 should get contributions at positions where sorted_token_indices == 0")
print("Positions:", torch.where(sorted_token_indices == 0)[0].tolist())
