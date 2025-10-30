/**
 * model_forward.cpp - High-level forward orchestrators
 *
 * Top-level forward pass implementations that delegate to model layers.
 */

#include "model_forward.h"
#include "reactive_model_forward.h"

namespace lb {
namespace forward {

// Delegating to existing implementations for backward compatibility
// These will be refactored to use new model_layer functions

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward_packed(
    const torch::Tensor& obs_sequence,
    const torch::Tensor& action_sequence,
    const torch::Tensor& agent_types,
    const torch::Tensor& positions,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    const torch::optional<torch::Tensor>& padding_mask,
    int64_t num_layers,
    int64_t num_heads,
    int64_t hidden_dim,
    int64_t num_experts,
    int64_t top_k,
    int64_t count_pad,
    int64_t tflag_pad) {
    return ::forward_packed(obs_sequence, action_sequence, agent_types, positions,
                           batched_weights, policy_indices, padding_mask,
                           num_layers, num_heads, hidden_dim, num_experts, top_k,
                           count_pad, tflag_pad);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, std::unordered_map<std::string, torch::Tensor>>
forward_packed_train(
    const torch::Tensor& obs_sequence,
    const torch::Tensor& action_sequence,
    const torch::Tensor& agent_types,
    const torch::Tensor& positions,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    const torch::optional<torch::Tensor>& padding_mask,
    int64_t num_layers,
    int64_t num_heads,
    int64_t hidden_dim,
    int64_t num_experts,
    int64_t top_k,
    int64_t count_pad,
    int64_t tflag_pad) {
    return ::forward_packed_train(obs_sequence, action_sequence, agent_types, positions,
                                  batched_weights, policy_indices, padding_mask,
                                  num_layers, num_heads, hidden_dim, num_experts, top_k,
                                  count_pad, tflag_pad);
}

} // namespace forward
} // namespace lb
