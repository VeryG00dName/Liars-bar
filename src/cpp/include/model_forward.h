#pragma once

#include <torch/extension.h>
#include <tuple>
#include <unordered_map>
#include <chrono>

/**
 * model_forward.h - High-level forward orchestrators
 *
 * This file contains the top-level forward_packed* functions that orchestrate
 * full model forward passes for both training and inference.
 *
 * Naming convention: lb::forward::function_name
 */

namespace lb {
namespace forward {

/**
 * Inference forward pass (no autograd).
 *
 * This is the optimized path used during rollouts and evaluation.
 * Uses raw kernels from lb_kernels directly.
 *
 * @return Tuple of (action_logits, opp_logits, state_values, win_logits)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward_packed(
    const torch::Tensor& obs_sequence,
    const torch::Tensor& action_sequence,
    const torch::Tensor& agent_types,
    const torch::Tensor& positions,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    const torch::optional<torch::Tensor>& padding_mask = torch::nullopt,
    int64_t num_layers = 2,
    int64_t num_heads = 4,
    int64_t hidden_dim = 256,
    int64_t num_experts = 8,
    int64_t top_k = 2,
    int64_t count_pad = 4,
    int64_t tflag_pad = 3,
    std::unordered_map<std::string, std::chrono::microseconds>* timers = nullptr
);

/**
 * Training forward pass (with autograd).
 *
 * Uses autograd-enabled kernels and includes gradient checkpointing.
 * Returns routing information for auxiliary losses.
 *
 * @return Tuple of (action_logits, opp_logits, state_values, win_logits,
 *                   gate_logits_tensor, routing_info_dict)
 */
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
    int64_t tflag_pad);

} // namespace forward
} // namespace lb