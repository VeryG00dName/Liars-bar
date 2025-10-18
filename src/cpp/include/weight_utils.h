#pragma once

#include <torch/torch.h>
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

/**
 * Load a .pth checkpoint file and extract the state_dict.
 *
 * This function loads a PyTorch checkpoint (.pth file) and extracts the model
 * weights, handling both formats:
 * 1. Direct state_dict: {"module.weight": tensor, ...}
 * 2. Wrapped checkpoint: {"policy_nets": {"agent_key": state_dict}, ...}
 *
 * @param path Path to the .pth checkpoint file
 * @param policy_key Optional key for wrapped checkpoints (e.g., "agent_0")
 * @return State dictionary mapping parameter names to tensors
 */
std::unordered_map<std::string, torch::Tensor> load_state_dict_from_pth(
    const std::string& path,
    const std::string& policy_key = ""
);

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
