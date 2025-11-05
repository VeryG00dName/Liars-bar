#pragma once

#include <torch/extension.h>
#include <tuple>
#include <unordered_map>
#include <chrono>

/**
 * model_forward.h - High-level forward orchestrators
 *
 * This file contains the top-level forward_packed function that orchestrates
 * full model inference forward passes. Training uses PyTorch autograd.
 *
 * Naming convention: lb::forward::function_name
 */

namespace lb {
namespace moe { struct MoEWorkspace; }
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
    std::unordered_map<std::string, std::chrono::microseconds>* timers = nullptr,
    lb::moe::MoEWorkspace* moe_workspaces = nullptr  // Optional per-layer workspaces
);

/**
 * Enable thread-local MoE workspace cache for forward_packed callers.
 *
 * Allocates per-layer CUTLASS workspaces sized for the provided max batch/seq.
 * If forward_packed is called with moe_workspaces == nullptr, it will use this
 * thread-local cache automatically.
 */
void enable_forward_workspace_cache(
    int64_t num_layers,
    int64_t max_batch_size,
    int64_t max_seq_length,
    int64_t hidden_dim,
    int64_t top_k);

/**
 * Disable and free the thread-local MoE workspace cache.
 */
void disable_forward_workspace_cache();

/**
 * Get the cached workspace for a given layer (or nullptr if not enabled).
 * This is used by individual layer testing functions to access the workspace cache.
 */
lb::moe::MoEWorkspace* get_cached_workspace(int64_t layer_idx);


} // namespace forward
} // namespace lb