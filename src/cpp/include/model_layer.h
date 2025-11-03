#pragma once

#include <torch/extension.h>
#include <tuple>
#include <unordered_map>
#include <chrono>

/**
 * model_layer.h - Stateless model layer implementations
 *
 * This file contains non-autograd model layer logic (attention, MoE, etc.).
 * These are pure computational functions that can be called from either
 * autograd contexts or non-autograd inference.
 *
 * Naming convention: lb::model::function_name
 */

namespace lb {
namespace moe { struct MoEWorkspace; }
namespace model {

/**
 * Decompose actions into (kind, count, table_flag) using lookup tables.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
action_decomposition(
    const torch::Tensor& action_sequence,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    const torch::optional<torch::Tensor>& padding_mask,
    int64_t count_pad,
    int64_t tflag_pad);

/**
 * Compute all embeddings (observation, action factorized, agent, position).
 */
c10::Dict<std::string, torch::Tensor>
compute_embeddings(
    const torch::Tensor& obs_sequence,
    const torch::Tensor& action_sequence,
    const torch::Tensor& agent_types,
    const torch::Tensor& positions,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    const torch::optional<torch::Tensor>& padding_mask,
    int64_t count_pad,
    int64_t tflag_pad,
    std::unordered_map<std::string, std::chrono::microseconds>* timers = nullptr);

/**
 * Compute modality gates (obs/action/agent/position) using per-policy MLPs.
 */
c10::Dict<std::string, torch::Tensor>
gating(
    const torch::Tensor& obs_embed,
    const torch::Tensor& action_embed,
    const torch::Tensor& agent_embed,
    const torch::Tensor& position_embed,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    std::unordered_map<std::string, std::chrono::microseconds>* timers = nullptr);

/**
 * Fuse gated embeddings and apply a final LayerNorm.
 */
c10::Dict<std::string, torch::Tensor>
fuse_embeddings(
    const torch::Tensor& g_obs,
    const torch::Tensor& g_action,
    const torch::Tensor& g_agent,
    const torch::Tensor& g_position,
    const torch::Tensor& obs_embed,
    const torch::Tensor& action_embed,
    const torch::Tensor& agent_embed,
    const torch::Tensor& position_embed,
    int64_t hidden_dim);

/**
 * Single transformer layer forward pass (non-autograd version).
 *
 * Used for inference and recomputation during backward pass.
 *
 * @return Tuple of (output, gate_logits, topk_indices, topk_scores)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
transformer_layer(
    const torch::Tensor& x,
    const torch::Tensor& policy_indices,
    const torch::Tensor& in_proj_weight,
    const torch::Tensor& in_proj_bias,
    const torch::Tensor& out_proj_weight,
    const torch::Tensor& out_proj_bias,
    const torch::Tensor& norm1_weight,
    const torch::Tensor& norm1_bias,
    const torch::Tensor& gate_weight,
    const torch::Tensor& gate_bias,
    const torch::Tensor& w1_all,
    const torch::Tensor& w2_all,
    const torch::Tensor& b1_all,
    const torch::Tensor& b2_all,
    const torch::Tensor& norm2_weight,
    const torch::Tensor& norm2_bias,
    int64_t num_heads,
    int64_t hidden_dim,
    int64_t top_k,
    lb::moe::MoEWorkspace* workspace = nullptr,
    std::unordered_map<std::string, std::chrono::microseconds>* timers = nullptr);

/**
 * Compute all per-expert heads.
 */
c10::Dict<std::string, torch::Tensor>
compute_heads(
    const torch::Tensor& transformer_output,
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const torch::Tensor& policy_indices,
    int64_t num_experts,
    std::unordered_map<std::string, std::chrono::microseconds>* timers = nullptr);

/**
 * Reduce per-expert head outputs using MoE routing weights.
 */
torch::Tensor reduce_expert_heads(
    const torch::Tensor& stacked,
    const torch::Tensor& topk_indices,
    const torch::Tensor& topk_scores);


/**
 * Compute MoE group metadata (m_sizes, policy_ids, expert_ids, token_offsets)
 * from sorted (expert, policy) indices using GPU ops and return CPU tensors.
 *
 * Inputs must be 1-D Long tensors sorted by (expert, policy).
 * Returns CPU-contiguous tensors suitable for C++ host-side use.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
moe_group_metadata(
    const torch::Tensor& sorted_expert_indices,
    const torch::Tensor& sorted_policy_indices);

/**
 * Device-side variant of moe_group_metadata that returns GPU tensors.
 *
 * Eliminates GPU→CPU transfer for use in device-side forward path.
 * Inputs must be 1-D Long tensors sorted by (expert, policy).
 * Returns GPU-contiguous tensors.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
moe_group_metadata_device(
    const torch::Tensor& sorted_expert_indices,
    const torch::Tensor& sorted_policy_indices);

/**
 * Build pointer table on GPU for batched expert weights.
 *
 * For a stacked tensor [P, E, ...], computes data pointers for each [p, e] slice.
 * Eliminates CPU staging and transfer. Returns GPU tensor of uint64 pointers.
 */
torch::Tensor build_ptr_table_device(const torch::Tensor& stacked);


} // namespace model
} // namespace lb
