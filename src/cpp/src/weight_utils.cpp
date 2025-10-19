#include "weight_utils.h"

#include <torch/torch.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <cstring>

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

        // Cleanup: remove per-expert individual weight keys to avoid ambiguity
        for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            std::string expert_prefix =
                layer_prefix + ".moe.experts." + std::to_string(expert_idx);

            std::string w1_key = expert_prefix + ".0.weight";
            std::string b1_key = expert_prefix + ".0.bias";
            std::string w2_key = expert_prefix + ".3.weight";
            std::string b2_key = expert_prefix + ".3.bias";

            state_dict.erase(w1_key);
            state_dict.erase(b1_key);
            state_dict.erase(w2_key);
            state_dict.erase(b2_key);
        }
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

    // Use insert_or_assign to ensure LUTs are always set correctly
    // (insert() fails silently if key exists)
    weights.insert_or_assign("lut_act_kind", lut_act_kind);
    weights.insert_or_assign("lut_count", lut_count);
    weights.insert_or_assign("lut_table_flag", lut_table_flag);
}

// -----------------------------------------------------------------------------
// Attention weight processing utilities
// -----------------------------------------------------------------------------

namespace {
using torch::indexing::Slice;

inline std::string replace_suffix(const std::string& s,
                                  const std::string& from,
                                  const std::string& to) {
    const auto pos = s.rfind(from);
    if (pos == std::string::npos) {
        return s;
    }
    std::string out = s;
    out.replace(pos, from.size(), to);
    return out;
}

inline std::string drop_self_attn_alias(const std::string& s) {
    constexpr const char* needle = ".self_attn.";
    const auto pos = s.find(needle);
    if (pos == std::string::npos) {
        return s;
    }
    std::string out = s;
    out.replace(pos, std::strlen(needle), ".");
    return out;
}
}  // namespace

void process_and_split_attention_weights(
    c10::Dict<std::string, torch::Tensor>& weights
) {
    std::vector<std::string> original_keys;
    original_keys.reserve(weights.size());
    for (const auto& kv : weights) {
        original_keys.push_back(kv.key());
    }

    for (const auto& key : original_keys) {
        if (key.find(".self_attn.in_proj_weight") != std::string::npos) {
            if (!weights.contains(key)) continue;
            const auto& w = weights.at(key);
            if (!w.defined() || (w.dim() != 3 && w.dim() != 2)) continue;

            const int64_t chunk_dim = (w.dim() == 3) ? 1 : 0;
            if (w.size(chunk_dim) % 3 != 0) continue;
            const int64_t H = w.size(chunk_dim) / 3;

            torch::Tensor q, k, v;
            if (w.dim() == 3) {
                q = w.index({Slice(), Slice(0, H), Slice()}).contiguous();
                k = w.index({Slice(), Slice(H, 2 * H), Slice()}).contiguous();
                v = w.index({Slice(), Slice(2 * H, 3 * H), Slice()}).contiguous();
            } else {
                // Unbatched path: [3*H, H] -> three [H, H]
                q = w.index({Slice(0, H), Slice()}).contiguous();
                k = w.index({Slice(H, 2 * H), Slice()}).contiguous();
                v = w.index({Slice(2 * H, 3 * H), Slice()}).contiguous();
            }

            const auto q_key = replace_suffix(key, "in_proj_weight", "q_proj.weight");
            const auto k_key = replace_suffix(key, "in_proj_weight", "k_proj.weight");
            const auto v_key = replace_suffix(key, "in_proj_weight", "v_proj.weight");

            weights.insert(q_key, q);
            weights.insert(k_key, k);
            weights.insert(v_key, v);

            weights.insert(drop_self_attn_alias(q_key), q);
            weights.insert(drop_self_attn_alias(k_key), k);
            weights.insert(drop_self_attn_alias(v_key), v);
            continue;
        }

        if (key.find(".self_attn.in_proj_bias") != std::string::npos) {
            if (!weights.contains(key)) continue;
            const auto& b = weights.at(key);
            if (!b.defined() || (b.dim() != 2 && b.dim() != 1)) continue;

            const int64_t chunk_dim = (b.dim() == 2) ? 1 : 0;
            if (b.size(chunk_dim) % 3 != 0) continue;
            const int64_t H = b.size(chunk_dim) / 3;

            torch::Tensor q, k, v;
            if (b.dim() == 2) {
                q = b.index({Slice(), Slice(0, H)}).contiguous();
                k = b.index({Slice(), Slice(H, 2 * H)}).contiguous();
                v = b.index({Slice(), Slice(2 * H, 3 * H)}).contiguous();
            } else {
                // Unbatched path: [3*H] -> three [H]
                q = b.index({Slice(0, H)}).contiguous();
                k = b.index({Slice(H, 2 * H)}).contiguous();
                v = b.index({Slice(2 * H, 3 * H)}).contiguous();
            }

            const auto q_key = replace_suffix(key, "in_proj_bias", "q_proj.bias");
            const auto k_key = replace_suffix(key, "in_proj_bias", "k_proj.bias");
            const auto v_key = replace_suffix(key, "in_proj_bias", "v_proj.bias");

            weights.insert(q_key, q);
            weights.insert(k_key, k);
            weights.insert(v_key, v);

            weights.insert(drop_self_attn_alias(q_key), q);
            weights.insert(drop_self_attn_alias(k_key), k);
            weights.insert(drop_self_attn_alias(v_key), v);
            continue;
        }

        if (key.find(".self_attn.") != std::string::npos) {
            const auto alias = drop_self_attn_alias(key);
            if (alias != key && !weights.contains(alias)) {
                weights.insert(alias, weights.at(key));
            }
        }
    }
}

void process_and_split_attention_weights(
    std::unordered_map<std::string, torch::Tensor>& weights
) {
    std::vector<std::string> original_keys;
    original_keys.reserve(weights.size());
    for (const auto& kv : weights) {
        original_keys.push_back(kv.first);
    }

    for (const auto& key : original_keys) {
        auto it = weights.find(key);
        if (it == weights.end()) continue;

        if (key.find(".self_attn.in_proj_weight") != std::string::npos) {
            const auto& w = it->second;
            if (!w.defined() || (w.dim() != 3 && w.dim() != 2)) continue;
            const int64_t chunk_dim = (w.dim() == 3) ? 1 : 0;
            if (w.size(chunk_dim) % 3 != 0) continue;
            const int64_t H = w.size(chunk_dim) / 3;

            torch::Tensor q, k, v;
            if (w.dim() == 3) {
                q = w.index({Slice(), Slice(0, H), Slice()}).contiguous();
                k = w.index({Slice(), Slice(H, 2 * H), Slice()}).contiguous();
                v = w.index({Slice(), Slice(2 * H, 3 * H), Slice()}).contiguous();
            } else {
                q = w.index({Slice(0, H), Slice()}).contiguous();
                k = w.index({Slice(H, 2 * H), Slice()}).contiguous();
                v = w.index({Slice(2 * H, 3 * H), Slice()}).contiguous();
            }

            const auto q_key = replace_suffix(key, "in_proj_weight", "q_proj.weight");
            const auto k_key = replace_suffix(key, "in_proj_weight", "k_proj.weight");
            const auto v_key = replace_suffix(key, "in_proj_weight", "v_proj.weight");

            weights[q_key] = q;
            weights[k_key] = k;
            weights[v_key] = v;

            weights[drop_self_attn_alias(q_key)] = q;
            weights[drop_self_attn_alias(k_key)] = k;
            weights[drop_self_attn_alias(v_key)] = v;
            continue;
        }

        if (key.find(".self_attn.in_proj_bias") != std::string::npos) {
            const auto& b = it->second;
            if (!b.defined() || (b.dim() != 2 && b.dim() != 1)) continue;
            const int64_t chunk_dim = (b.dim() == 2) ? 1 : 0;
            if (b.size(chunk_dim) % 3 != 0) continue;
            const int64_t H = b.size(chunk_dim) / 3;

            torch::Tensor q, k, v;
            if (b.dim() == 2) {
                q = b.index({Slice(), Slice(0, H)}).contiguous();
                k = b.index({Slice(), Slice(H, 2 * H)}).contiguous();
                v = b.index({Slice(), Slice(2 * H, 3 * H)}).contiguous();
            } else {
                q = b.index({Slice(0, H)}).contiguous();
                k = b.index({Slice(H, 2 * H)}).contiguous();
                v = b.index({Slice(2 * H, 3 * H)}).contiguous();
            }

            const auto q_key = replace_suffix(key, "in_proj_bias", "q_proj.bias");
            const auto k_key = replace_suffix(key, "in_proj_bias", "k_proj.bias");
            const auto v_key = replace_suffix(key, "in_proj_bias", "v_proj.bias");

            weights[q_key] = q;
            weights[k_key] = k;
            weights[v_key] = v;

            weights[drop_self_attn_alias(q_key)] = q;
            weights[drop_self_attn_alias(k_key)] = k;
            weights[drop_self_attn_alias(v_key)] = v;
            continue;
        }

        if (key.find(".self_attn.") != std::string::npos) {
            const auto alias = drop_self_attn_alias(key);
            if (alias != key && !weights.count(alias)) {
                weights[alias] = it->second;
            }
        }
    }
}
