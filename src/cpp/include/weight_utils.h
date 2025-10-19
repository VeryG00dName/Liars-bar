#pragma once

#include <torch/torch.h>
#include <ATen/core/Dict.h>
#include <string>
#include <unordered_map>

/**
 * Pre-stack MoE expert weights for efficient batched inference.
 *
 * This function takes a state_dict (weight dictionary) and reorganizes MoE expert
 * weights from individual per-expert tensors into stacked tensors optimized for
 * the MoE CUDA kernel.
 *
 * Input format (per-expert weights):
 *   - "transformer.layers.{L}.moe.experts.{E}.0.weight" : [out_dim, in_dim]
 *   - "transformer.layers.{L}.moe.experts.{E}.0.bias"   : [out_dim]
 *   - "transformer.layers.{L}.moe.experts.{E}.3.weight" : [in_dim, out_dim]
 *   - "transformer.layers.{L}.moe.experts.{E}.3.bias"   : [in_dim]
 *
 * Output format (stacked weights, added to state_dict):
 *   - "transformer.layers.{L}.moe.experts.w1" : [num_experts, B, out_dim, in_dim]
 *   - "transformer.layers.{L}.moe.experts.b1" : [num_experts, B, out_dim]
 *   - "transformer.layers.{L}.moe.experts.w2" : [num_experts, B, in_dim, out_dim]
 *   - "transformer.layers.{L}.moe.experts.b2" : [num_experts, B, in_dim]
 *
 * where B is the batch dimension (number of models being batched together).
 *
 * @param state_dict The weight dictionary to modify in-place
 * @param num_layers Number of transformer layers
 * @param num_experts Number of MoE experts per layer
 */
void prestack_moe_expert_weights(
    std::unordered_map<std::string, torch::Tensor>& state_dict,
    int64_t num_layers,
    int64_t num_experts
);

// Note: Previous C++ .pth loading helper removed.

/**
 * Batch multiple state_dicts into a single batched weight dictionary.
 *
 * Takes multiple model weight dictionaries and stacks them along a new batch
 * dimension, enabling efficient batched inference with heterogeneous models.
 *
 * All input state_dicts must have identical keys and compatible shapes.
 *
 * Input: Vector of state_dicts, each mapping "module.weight" -> [shape]
 * Output: Single state_dict mapping "module.weight" -> [B, shape]
 *
 * @param state_dicts Vector of state dictionaries to batch
 * @return Batched weight dictionary with batch dimension
 */
c10::Dict<std::string, torch::Tensor> batch_state_dicts(
    const std::vector<std::unordered_map<std::string, torch::Tensor>>& state_dicts
);

/**
 * Add fixed buffers (LUTs) to the batched weight dictionary.
 *
 * The model uses fixed lookup tables for action decomposition. These need to be
 * included in the weight dict for the forward function.
 *
 * Adds the following keys:
 *   - "lut_act_kind": [11] (action kind decomposition)
 *   - "lut_count": [11] (count decomposition)
 *   - "lut_table_flag": [11] (table flag decomposition)
 *
 * @param weights Weight dictionary to modify
 * @param device Target device for the buffers
 */
void add_fixed_buffers(
    c10::Dict<std::string, torch::Tensor>& weights,
    const torch::Device& device
);

/**
 * Split fused nn.MultiheadAttention in_proj_weight/bias into separate Q/K/V tensors
 * and add compatibility aliases by dropping the ".self_attn." segment.
 *
 * This modifies the provided weight dictionary in-place by:
 *  - For any key matching "*.self_attn.in_proj_weight": splitting [B, 3*H, H]
 *    into q_proj.weight/k_proj.weight/v_proj.weight each shaped [B, H, H].
 *  - For any key matching "*.self_attn.in_proj_bias": splitting [B, 3*H]
 *    into q_proj.bias/k_proj.bias/v_proj.bias each shaped [B, H].
 *  - Adding alias entries where the substring ".self_attn." is removed for all
 *    self-attention related keys to satisfy alternative naming conventions.
 */
void process_and_split_attention_weights(
    c10::Dict<std::string, torch::Tensor>& weights
);

/**
 * Overload for std::unordered_map-backed weight dictionaries.
 */
void process_and_split_attention_weights(
    std::unordered_map<std::string, torch::Tensor>& weights
);
