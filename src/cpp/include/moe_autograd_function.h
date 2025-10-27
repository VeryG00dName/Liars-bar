#pragma once

#include <torch/extension.h>

namespace lb {
namespace moe {

// C++ autograd front for grouped MoE FFN. Returns output_grouped [T, H] (Half).
// Backward computes grads for input_grouped and expert weights/biases, and
// also returns gradient for routing_weights (dr) so it can flow to gate logits.
//
// Group metadata tensors (m_sizes, policy_ids, expert_ids, token_offsets)
// must be Long on CPU.
torch::Tensor grouped_moe_autograd_forward(
    const torch::Tensor& input_grouped,             // [T, H] Half (CUDA)
    const torch::Tensor& w1_weights,                // [P, E, F, H] Half (CUDA)
    const torch::Tensor& w2_weights,                // [P, E, H, F] Half (CUDA)
    const torch::Tensor& b1_biases,                 // [P, E, F] Half (CUDA)
    const torch::Tensor& b2_biases,                 // [P, E, H] Half (CUDA)
    const torch::Tensor& routing_weights_grouped,   // [T] Float (CUDA)
    const torch::Tensor& indices_grouped,           // [T] Long (CUDA)
    const torch::Tensor& m_sizes_cpu,               // [G] Long (CPU)
    const torch::Tensor& policy_ids_cpu,            // [G] Long (CPU)
    const torch::Tensor& expert_ids_cpu,            // [G] Long (CPU)
    const torch::Tensor& token_offsets_cpu,         // [G] Long (CPU)
    int64_t hidden_dim,
    int64_t ffn_dim
);

} // namespace moe
} // namespace lb

