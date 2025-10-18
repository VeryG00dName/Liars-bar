#include "weight_utils.h"

#include <torch/torch.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

void prestack_moe_expert_weights(
    std::unordered_map<std::string, torch::Tensor>& state_dict,
    int64_t num_layers,
    int64_t num_experts
) {
    /**
     * Pre-stack MoE expert weights for batched inference.
     *
     * For each transformer layer, gather all expert weights and stack them
     * into single tensors optimized for the MoE CUDA kernel.
     */

    // Check if state_dict already has batched weights (batch dimension present)
    // If any weight has more than 2 dimensions for linear layers, it's likely batched
    bool is_batched = false;
    for (const auto& pair : state_dict) {
        if (pair.first.find("obs_encoder.0.weight") != std::string::npos) {
            if (pair.second.dim() == 3) {  // [B, out_dim, in_dim]
                is_batched = true;
                break;
            }
        }
    }

    int64_t batch_size = 1;
    if (is_batched) {
        // Infer batch size from any weight
        for (const auto& pair : state_dict) {
            if (pair.first.find("obs_encoder.0.weight") != std::string::npos) {
                batch_size = pair.second.size(0);
                break;
            }
        }
    }

    for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        std::vector<torch::Tensor> w1_experts, b1_experts, w2_experts, b2_experts;

        for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            std::string expert_prefix =
                "transformer.layers." + std::to_string(layer_idx) +
                ".moe.experts." + std::to_string(expert_idx);

            std::string w1_key = expert_prefix + ".0.weight";
            std::string b1_key = expert_prefix + ".0.bias";
            std::string w2_key = expert_prefix + ".3.weight";
            std::string b2_key = expert_prefix + ".3.bias";

            // Check if keys exist
            if (state_dict.find(w1_key) == state_dict.end()) {
                std::stringstream ss;
                ss << "Missing expert weight key: " << w1_key;
                throw std::runtime_error(ss.str());
            }

            w1_experts.push_back(state_dict.at(w1_key));
            b1_experts.push_back(state_dict.at(b1_key));
            w2_experts.push_back(state_dict.at(w2_key));
            b2_experts.push_back(state_dict.at(b2_key));
        }

        // Stack along expert dimension (dim=0 if unbatched, dim=1 if batched)
        std::string layer_prefix = "transformer.layers." + std::to_string(layer_idx);
        int64_t stack_dim = is_batched ? 1 : 0;

        state_dict[layer_prefix + ".moe.experts.w1"] = torch::stack(w1_experts, stack_dim);
        state_dict[layer_prefix + ".moe.experts.b1"] = torch::stack(b1_experts, stack_dim);
        state_dict[layer_prefix + ".moe.experts.w2"] = torch::stack(w2_experts, stack_dim);
        state_dict[layer_prefix + ".moe.experts.b2"] = torch::stack(b2_experts, stack_dim);
    }
}

// Note: Previous C++ .pth loading helper removed; loading now handled in Python.

c10::Dict<std::string, torch::Tensor> batch_state_dicts(
    const std::vector<std::unordered_map<std::string, torch::Tensor>>& state_dicts
) {
    /**
     * Batch multiple state_dicts along a new batch dimension.
     *
     * All state_dicts must have the same keys and compatible shapes.
     */

    if (state_dicts.empty()) {
        throw std::runtime_error("Cannot batch empty state_dicts");
    }

    c10::Dict<std::string, torch::Tensor> batched;

    // Get keys from first state_dict
    const auto& first_dict = state_dicts[0];

    for (const auto& pair : first_dict) {
        const std::string& key = pair.first;
        std::vector<torch::Tensor> tensors_to_stack;

        // Gather this parameter from all state_dicts
        for (const auto& state_dict : state_dicts) {
            if (state_dict.find(key) == state_dict.end()) {
                std::stringstream ss;
                ss << "Key '" << key << "' missing in one of the state_dicts";
                throw std::runtime_error(ss.str());
            }
            tensors_to_stack.push_back(state_dict.at(key));
        }

        // Stack along batch dimension (dim=0)
        auto stacked = torch::stack(tensors_to_stack, /*dim=*/0);
        batched.insert(key, stacked);
    }

    return batched;
}

void add_fixed_buffers(
    c10::Dict<std::string, torch::Tensor>& weights,
    const torch::Device& device
) {
    /**
     * Add fixed lookup table buffers used for action decomposition.
     *
     * These are constant tensors defined in the model architecture.
     */

    // Action kind LUT: maps action ID -> action kind (1=normal, 2=challenge, 0=pad)
    auto lut_act_kind = torch::tensor(
        {1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0},
        torch::dtype(torch::kLong).device(device)
    );

    // Count LUT: maps action ID -> count (0-3, 4=pad)
    auto lut_count = torch::tensor(
        {1, 2, 3, 1, 2, 3, 0, 1, 2, 3, 4},
        torch::dtype(torch::kLong).device(device)
    );

    // Table flag LUT: maps action ID -> table flag (1=table, 2=non-table, 0=challenge, 3=pad)
    auto lut_table_flag = torch::tensor(
        {1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 3},
        torch::dtype(torch::kLong).device(device)
    );

    weights.insert("lut_act_kind", lut_act_kind);
    weights.insert("lut_count", lut_count);
    weights.insert("lut_table_flag", lut_table_flag);
}
