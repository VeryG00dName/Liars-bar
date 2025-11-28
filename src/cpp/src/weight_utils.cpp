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
    std::string first_stacked_key = "transformer.layers.0.moe.experts.w1";
    if (state_dict.count(first_stacked_key)) {
        return;
    }

    for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        std::vector<torch::Tensor> w1_experts, b1_experts, w2_experts, b2_experts;
        w1_experts.reserve(num_experts);
        b1_experts.reserve(num_experts);
        w2_experts.reserve(num_experts);
        b2_experts.reserve(num_experts);

        for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            std::string expert_prefix =
                "transformer.layers." + std::to_string(layer_idx) +
                ".moe.experts." + std::to_string(expert_idx);
            
            w1_experts.push_back(state_dict.at(expert_prefix + ".0.weight"));
            b1_experts.push_back(state_dict.at(expert_prefix + ".0.bias"));
            w2_experts.push_back(state_dict.at(expert_prefix + ".3.weight"));
            b2_experts.push_back(state_dict.at(expert_prefix + ".3.bias"));
        }

        std::string layer_prefix = "transformer.layers." + std::to_string(layer_idx);
        
        state_dict[layer_prefix + ".moe.experts.w1"] = torch::stack(w1_experts, 0);
        state_dict[layer_prefix + ".moe.experts.b1"] = torch::stack(b1_experts, 0);
        state_dict[layer_prefix + ".moe.experts.w2"] = torch::stack(w2_experts, 0);
        state_dict[layer_prefix + ".moe.experts.b2"] = torch::stack(b2_experts, 0);

        for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
            std::string expert_prefix = layer_prefix + ".moe.experts." + std::to_string(expert_idx);
            state_dict.erase(expert_prefix + ".0.weight");
            state_dict.erase(expert_prefix + ".0.bias");
            state_dict.erase(expert_prefix + ".3.weight");
            state_dict.erase(expert_prefix + ".3.bias");
        }
    }
}

torch::Tensor build_ptr_table_device(const torch::Tensor& stacked) {
    TORCH_CHECK(stacked.is_cuda(), "Stacked expert weights must be CUDA tensors");
    TORCH_CHECK(stacked.dim() >= 2, "Stacked expert weights must be at least 2D");

    const int64_t P = stacked.size(0);
    const int64_t E = stacked.size(1);

    auto ptr_tensor_cpu = torch::empty({P, E}, torch::dtype(torch::kUInt64).device(torch::kCPU));
    auto ptr_data = ptr_tensor_cpu.data_ptr<uint64_t>();

    for (int64_t p = 0; p < P; ++p) {
        for (int64_t e = 0; e < E; ++e) {
            auto expert_tensor = stacked.select(0, p).select(0, e);
            ptr_data[p * E + e] = reinterpret_cast<uint64_t>(expert_tensor.data_ptr());
        }
    }
    return ptr_tensor_cpu.to(stacked.device());
}

void create_moe_weight_pointers(
    std::unordered_map<std::string, torch::Tensor>& state_dict,
    int64_t num_layers,
    int64_t num_experts
) {
    for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        std::string layer_prefix = "transformer.layers." + std::to_string(layer_idx) + ".moe.experts.";
        
        const std::vector<std::string> keys = {"w1", "b1", "w2", "b2"};
        for (const auto& key : keys) {
            std::string full_key = layer_prefix + key;
            auto it = state_dict.find(full_key);
            if (it != state_dict.end()) {
                state_dict[full_key + "_ptrs"] = build_ptr_table_device(it->second);
            }
        }
    }
}

void create_moe_weight_pointers(
    c10::Dict<std::string, torch::Tensor>& state_dict,
    int64_t num_layers,
    int64_t num_experts
) {
    for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        std::string layer_prefix = "transformer.layers." + std::to_string(layer_idx) + ".moe.experts.";
        
        const std::vector<std::string> keys = {"w1", "b1", "w2", "b2"};
        for (const auto& key : keys) {
            std::string full_key = layer_prefix + key;
            auto it = state_dict.find(full_key);
            if (it != state_dict.end()) {
                state_dict.insert_or_assign(full_key + "_ptrs", build_ptr_table_device(it->value()));
            }
        }
    }
}

c10::Dict<std::string, torch::Tensor> batch_state_dicts(
    const std::vector<std::unordered_map<std::string, torch::Tensor>>& state_dicts
) {
    if (state_dicts.empty()) {
        throw std::runtime_error("Cannot batch empty state_dicts");
    }

    c10::Dict<std::string, torch::Tensor> batched;
    const auto& first_dict = state_dicts[0];

    for (const auto& pair : first_dict) {
        const std::string& key = pair.first;
        std::vector<torch::Tensor> tensors_to_stack;
        tensors_to_stack.reserve(state_dicts.size());

        for (const auto& state_dict : state_dicts) {
            tensors_to_stack.push_back(state_dict.at(key));
        }
        batched.insert(key, torch::stack(tensors_to_stack, 0));
    }
    return batched;
}

void add_fixed_buffers(
    c10::Dict<std::string, torch::Tensor>& weights,
    const torch::Device& device
) {
    // Actions 0-6: Own actions (Play 1x True, Play 2x True, Play 3x True, Play 1x False, Play 2x False, Play 3x False, Challenge)
    // Actions 7-9: Opponent actions (Opp Play 1x, Opp Play 2x, Opp Play 3x)
    // Action 10: Padding token
    // Note: act_kind: 0=play, 1=challenge
    // count: actual count (1, 2, 3) or 0 for special actions
    // table_flag: 1=true cards, 0=false cards (bluff)
    auto lut_act_kind = torch::tensor({0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0}, torch::dtype(torch::kLong).device(device));
    auto lut_count = torch::tensor({1, 2, 3, 1, 2, 3, 0, 1, 2, 3, 0}, torch::dtype(torch::kLong).device(device));
    auto lut_table_flag = torch::tensor({1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}, torch::dtype(torch::kLong).device(device));

    weights.insert_or_assign("lut_act_kind", lut_act_kind);
    weights.insert_or_assign("lut_count", lut_count);
    weights.insert_or_assign("lut_table_flag", lut_table_flag);
}

namespace {
using torch::indexing::Slice;

inline std::string replace_suffix(const std::string& s, const std::string& from, const std::string& to) {
    auto pos = s.rfind(from);
    if (pos == std::string::npos) return s;
    std::string out = s;
    out.replace(pos, from.size(), to);
    return out;
}

inline std::string drop_self_attn_alias(const std::string& s) {
    constexpr const char* needle = ".self_attn.";
    auto pos = s.find(needle);
    if (pos == std::string::npos) return s;
    std::string out = s;
    out.replace(pos, std::strlen(needle), ".");
    return out;
}
}

void process_and_split_attention_weights(c10::Dict<std::string, torch::Tensor>& weights) {
    std::vector<std::string> original_keys;
    original_keys.reserve(weights.size());
    for (const auto& kv : weights) original_keys.push_back(kv.key());

    for (const auto& key : original_keys) {
        if (key.find(".self_attn.in_proj_weight") != std::string::npos) {
            const auto& w = weights.at(key);
            const int64_t chunk_dim = (w.dim() == 3) ? 1 : 0;
            const int64_t H = w.size(chunk_dim) / 3;
            auto chunks = w.chunk(3, chunk_dim);
            auto q_key = replace_suffix(key, "in_proj_weight", "q_proj.weight");
            auto k_key = replace_suffix(key, "in_proj_weight", "k_proj.weight");
            auto v_key = replace_suffix(key, "in_proj_weight", "v_proj.weight");
            weights.insert_or_assign(q_key, chunks[0].contiguous());
            weights.insert_or_assign(k_key, chunks[1].contiguous());
            weights.insert_or_assign(v_key, chunks[2].contiguous());
            weights.insert_or_assign(drop_self_attn_alias(q_key), chunks[0].contiguous());
            weights.insert_or_assign(drop_self_attn_alias(k_key), chunks[1].contiguous());
            weights.insert_or_assign(drop_self_attn_alias(v_key), chunks[2].contiguous());
        } else if (key.find(".self_attn.in_proj_bias") != std::string::npos) {
            const auto& b = weights.at(key);
            const int64_t chunk_dim = (b.dim() == 2) ? 1 : 0;
            auto chunks = b.chunk(3, chunk_dim);
            auto q_key = replace_suffix(key, "in_proj_bias", "q_proj.bias");
            auto k_key = replace_suffix(key, "in_proj_bias", "k_proj.bias");
            auto v_key = replace_suffix(key, "in_proj_bias", "v_proj.bias");
            weights.insert_or_assign(q_key, chunks[0].contiguous());
            weights.insert_or_assign(k_key, chunks[1].contiguous());
            weights.insert_or_assign(v_key, chunks[2].contiguous());
            weights.insert_or_assign(drop_self_attn_alias(q_key), chunks[0].contiguous());
            weights.insert_or_assign(drop_self_attn_alias(k_key), chunks[1].contiguous());
            weights.insert_or_assign(drop_self_attn_alias(v_key), chunks[2].contiguous());
        } else if (key.find(".self_attn.") != std::string::npos) {
            const auto alias = drop_self_attn_alias(key);
            if (alias != key && !weights.contains(alias)) {
                weights.insert_or_assign(alias, weights.at(key));
            }
        }
    }

    // Add aliases transformer_layers.* -> transformer.layers.* for dense models
    original_keys.clear();
    original_keys.reserve(weights.size());
    for (const auto& kv : weights) original_keys.push_back(kv.key());
    for (const auto& key : original_keys) {
        constexpr const char* needle = "transformer_layers.";
        auto pos = key.find(needle);
        if (pos == 0) {
            auto alias = std::string("transformer.layers.") + key.substr(std::strlen(needle));
            weights.insert_or_assign(alias, weights.at(key));
        }
    }

    // Add aliases between dense Python naming (.ffn.) and TorchScript naming (.swiglu_ffn.)
    original_keys.clear();
    original_keys.reserve(weights.size());
    for (const auto& kv : weights) original_keys.push_back(kv.key());
    for (const auto& key : original_keys) {
        constexpr const char* ffn_token = ".ffn.";
        constexpr const char* swiglu_token = ".swiglu_ffn.";
        auto pos_ffn = key.find(ffn_token);
        if (pos_ffn != std::string::npos) {
            auto alias = key;
            alias.replace(pos_ffn, std::strlen(ffn_token), swiglu_token);
            weights.insert_or_assign(alias, weights.at(key));
            continue;
        }
        auto pos_swiglu = key.find(swiglu_token);
        if (pos_swiglu != std::string::npos) {
            auto alias = key;
            alias.replace(pos_swiglu, std::strlen(swiglu_token), ffn_token);
            if (!weights.contains(alias)) {
                weights.insert(alias, weights.at(key));
            }
        }
    }

    // Add self_attn.* aliases for q/k/v/out projections if missing
    original_keys.clear();
    original_keys.reserve(weights.size());
    for (const auto& kv : weights) original_keys.push_back(kv.key());
    const std::vector<std::pair<std::string, std::string>> attn_proj = {
        {"q_proj.weight", "self_attn.q_proj.weight"},
        {"q_proj.bias",   "self_attn.q_proj.bias"},
        {"k_proj.weight", "self_attn.k_proj.weight"},
        {"k_proj.bias",   "self_attn.k_proj.bias"},
        {"v_proj.weight", "self_attn.v_proj.weight"},
        {"v_proj.bias",   "self_attn.v_proj.bias"},
        {"out_proj.weight", "self_attn.out_proj.weight"},
        {"out_proj.bias",   "self_attn.out_proj.bias"},
    };
    for (const auto& key : original_keys) {
        for (const auto& kvp : attn_proj) {
            const auto& suffix_src = kvp.first;
            const auto& suffix_alias = kvp.second;
            if (key.size() >= suffix_src.size() && key.compare(key.size() - suffix_src.size(), suffix_src.size(), suffix_src) == 0) {
                auto alias = key.substr(0, key.size() - suffix_src.size()) + suffix_alias;
                weights.insert_or_assign(alias, weights.at(key));
            }
        }
    }

    // Final sanity pass: if fused in_proj_* still missing but split q/k/v exist, create them
    struct QKV { torch::Tensor q, k, v; };
    std::unordered_map<std::string, QKV> qkv_w, qkv_b;
    for (const auto& kv : weights) {
        const auto& key = kv.key();
        constexpr const char* q_w_suffix = ".self_attn.q_proj.weight";
        constexpr const char* k_w_suffix = ".self_attn.k_proj.weight";
        constexpr const char* v_w_suffix = ".self_attn.v_proj.weight";
        constexpr const char* q_b_suffix = ".self_attn.q_proj.bias";
        constexpr const char* k_b_suffix = ".self_attn.k_proj.bias";
        constexpr const char* v_b_suffix = ".self_attn.v_proj.bias";

        if (key.size() >= std::strlen(q_w_suffix) &&
            key.compare(key.size() - std::strlen(q_w_suffix), std::strlen(q_w_suffix), q_w_suffix) == 0) {
            auto prefix = key.substr(0, key.size() - std::strlen(q_w_suffix));
            qkv_w[prefix].q = kv.value();
        } else if (key.size() >= std::strlen(k_w_suffix) &&
                   key.compare(key.size() - std::strlen(k_w_suffix), std::strlen(k_w_suffix), k_w_suffix) == 0) {
            auto prefix = key.substr(0, key.size() - std::strlen(k_w_suffix));
            qkv_w[prefix].k = kv.value();
        } else if (key.size() >= std::strlen(v_w_suffix) &&
                   key.compare(key.size() - std::strlen(v_w_suffix), std::strlen(v_w_suffix), v_w_suffix) == 0) {
            auto prefix = key.substr(0, key.size() - std::strlen(v_w_suffix));
            qkv_w[prefix].v = kv.value();
        } else if (key.size() >= std::strlen(q_b_suffix) &&
                   key.compare(key.size() - std::strlen(q_b_suffix), std::strlen(q_b_suffix), q_b_suffix) == 0) {
            auto prefix = key.substr(0, key.size() - std::strlen(q_b_suffix));
            qkv_b[prefix].q = kv.value();
        } else if (key.size() >= std::strlen(k_b_suffix) &&
                   key.compare(key.size() - std::strlen(k_b_suffix), std::strlen(k_b_suffix), k_b_suffix) == 0) {
            auto prefix = key.substr(0, key.size() - std::strlen(k_b_suffix));
            qkv_b[prefix].k = kv.value();
        } else if (key.size() >= std::strlen(v_b_suffix) &&
                   key.compare(key.size() - std::strlen(v_b_suffix), std::strlen(v_b_suffix), v_b_suffix) == 0) {
            auto prefix = key.substr(0, key.size() - std::strlen(v_b_suffix));
            qkv_b[prefix].v = kv.value();
        }
    }

    for (const auto& kv : qkv_w) {
        const auto& prefix = kv.first;
        const auto& q = kv.second.q;
        const auto& k = kv.second.k;
        const auto& v = kv.second.v;
        auto fused_key_w = prefix + ".self_attn.in_proj_weight";
        if (!weights.contains(fused_key_w) && q.defined() && k.defined() && v.defined()) {
            const int64_t cat_dim_w = (q.dim() == 3) ? 1 : 0;
            auto in_proj_w = torch::cat({q, k, v}, cat_dim_w).contiguous();
            weights.insert_or_assign(fused_key_w, in_proj_w);
            weights.insert_or_assign(drop_self_attn_alias(fused_key_w), in_proj_w);
        }
    }

    for (const auto& kv : qkv_b) {
        const auto& prefix = kv.first;
        const auto& q = kv.second.q;
        const auto& k = kv.second.k;
        const auto& v = kv.second.v;
        auto fused_key_b = prefix + ".self_attn.in_proj_bias";
        if (!weights.contains(fused_key_b) && q.defined() && k.defined() && v.defined()) {
            const int64_t cat_dim_b = (q.dim() == 2) ? 1 : 0;
            auto in_proj_b = torch::cat({q, k, v}, cat_dim_b).contiguous();
            weights.insert_or_assign(fused_key_b, in_proj_b);
            weights.insert_or_assign(drop_self_attn_alias(fused_key_b), in_proj_b);
        }
    }
}

void process_and_split_attention_weights(std::unordered_map<std::string, torch::Tensor>& weights) {
    std::vector<std::string> original_keys;
    original_keys.reserve(weights.size());
    for (const auto& kv : weights) original_keys.push_back(kv.first);

    for (const auto& key : original_keys) {
        if (key.find(".self_attn.in_proj_weight") != std::string::npos) {
            const auto& w = weights.at(key);
            const int64_t chunk_dim = (w.dim() == 3) ? 1 : 0;
            auto chunks = w.chunk(3, chunk_dim);
            auto q_key = replace_suffix(key, "in_proj_weight", "q_proj.weight");
            auto k_key = replace_suffix(key, "in_proj_weight", "k_proj.weight");
            auto v_key = replace_suffix(key, "in_proj_weight", "v_proj.weight");
            weights[q_key] = chunks[0].contiguous();
            weights[k_key] = chunks[1].contiguous();
            weights[v_key] = chunks[2].contiguous();
            weights[drop_self_attn_alias(q_key)] = chunks[0].contiguous();
            weights[drop_self_attn_alias(k_key)] = chunks[1].contiguous();
            weights[drop_self_attn_alias(v_key)] = chunks[2].contiguous();
        } else if (key.find(".self_attn.in_proj_bias") != std::string::npos) {
            const auto& b = weights.at(key);
            const int64_t chunk_dim = (b.dim() == 2) ? 1 : 0;
            auto chunks = b.chunk(3, chunk_dim);
            auto q_key = replace_suffix(key, "in_proj_bias", "q_proj.bias");
            auto k_key = replace_suffix(key, "in_proj_bias", "k_proj.bias");
            auto v_key = replace_suffix(key, "in_proj_bias", "v_proj.bias");
            weights[q_key] = chunks[0].contiguous();
            weights[k_key] = chunks[1].contiguous();
            weights[v_key] = chunks[2].contiguous();
            weights[drop_self_attn_alias(q_key)] = chunks[0].contiguous();
            weights[drop_self_attn_alias(k_key)] = chunks[1].contiguous();
            weights[drop_self_attn_alias(v_key)] = chunks[2].contiguous();
        } else if (key.find(".self_attn.") != std::string::npos) {
            const auto alias = drop_self_attn_alias(key);
            if (alias != key) {
                weights[alias] = weights.at(key);
            }
        }
    }

    // Add aliases transformer_layers.* -> transformer.layers.* for dense models
    original_keys.clear();
    original_keys.reserve(weights.size());
    for (const auto& kv : weights) original_keys.push_back(kv.first);
    for (const auto& key : original_keys) {
        constexpr const char* needle = "transformer_layers.";
        auto pos = key.find(needle);
        if (pos == 0) {
            std::string alias = std::string("transformer.layers.") + key.substr(std::strlen(needle));
            weights[alias] = weights.at(key);
        }
    }

    // Add aliases between dense Python naming (.ffn.) and TorchScript naming (.swiglu_ffn.)
    original_keys.clear();
    original_keys.reserve(weights.size());
    for (const auto& kv : weights) original_keys.push_back(kv.first);
    for (const auto& key : original_keys) {
        constexpr const char* ffn_token = ".ffn.";
        constexpr const char* swiglu_token = ".swiglu_ffn.";
        auto pos_ffn = key.find(ffn_token);
        if (pos_ffn != std::string::npos) {
            auto alias = key;
            alias.replace(pos_ffn, std::strlen(ffn_token), swiglu_token);
            weights[alias] = weights.at(key);
            continue;
        }
        auto pos_swiglu = key.find(swiglu_token);
        if (pos_swiglu != std::string::npos) {
            auto alias = key;
            alias.replace(pos_swiglu, std::strlen(swiglu_token), ffn_token);
            if (!weights.count(alias)) {
                weights[alias] = weights.at(key);
            }
        }
    }

    // Add self_attn.* aliases for q/k/v/out projections if missing
    original_keys.clear();
    original_keys.reserve(weights.size());
    for (const auto& kv : weights) original_keys.push_back(kv.first);
    const std::vector<std::pair<std::string, std::string>> attn_proj = {
        {"q_proj.weight", "self_attn.q_proj.weight"},
        {"q_proj.bias",   "self_attn.q_proj.bias"},
        {"k_proj.weight", "self_attn.k_proj.weight"},
        {"k_proj.bias",   "self_attn.k_proj.bias"},
        {"v_proj.weight", "self_attn.v_proj.weight"},
        {"v_proj.bias",   "self_attn.v_proj.bias"},
        {"out_proj.weight", "self_attn.out_proj.weight"},
        {"out_proj.bias",   "self_attn.out_proj.bias"},
    };
    for (const auto& key : original_keys) {
        for (const auto& kvp : attn_proj) {
            const auto& suffix_src = kvp.first;
            const auto& suffix_alias = kvp.second;
            if (key.size() >= suffix_src.size() && key.compare(key.size() - suffix_src.size(), suffix_src.size(), suffix_src) == 0) {
                std::string alias = key.substr(0, key.size() - suffix_src.size()) + suffix_alias;
                weights[alias] = weights.at(key);
            }
        }
    }

    // Final sanity pass: if fused in_proj_* still missing but split q/k/v exist, create them
    struct QKV { torch::Tensor q, k, v; };
    std::unordered_map<std::string, QKV> qkv_w, qkv_b;
    for (const auto& kv : weights) {
        const auto& key = kv.first;
        constexpr const char* q_w_suffix = ".self_attn.q_proj.weight";
        constexpr const char* k_w_suffix = ".self_attn.k_proj.weight";
        constexpr const char* v_w_suffix = ".self_attn.v_proj.weight";
        constexpr const char* q_b_suffix = ".self_attn.q_proj.bias";
        constexpr const char* k_b_suffix = ".self_attn.k_proj.bias";
        constexpr const char* v_b_suffix = ".self_attn.v_proj.bias";

        if (key.size() >= std::strlen(q_w_suffix) &&
            key.compare(key.size() - std::strlen(q_w_suffix), std::strlen(q_w_suffix), q_w_suffix) == 0) {
            auto prefix = key.substr(0, key.size() - std::strlen(q_w_suffix));
            qkv_w[prefix].q = kv.second;
        } else if (key.size() >= std::strlen(k_w_suffix) &&
                   key.compare(key.size() - std::strlen(k_w_suffix), std::strlen(k_w_suffix), k_w_suffix) == 0) {
            auto prefix = key.substr(0, key.size() - std::strlen(k_w_suffix));
            qkv_w[prefix].k = kv.second;
        } else if (key.size() >= std::strlen(v_w_suffix) &&
                   key.compare(key.size() - std::strlen(v_w_suffix), std::strlen(v_w_suffix), v_w_suffix) == 0) {
            auto prefix = key.substr(0, key.size() - std::strlen(v_w_suffix));
            qkv_w[prefix].v = kv.second;
        } else if (key.size() >= std::strlen(q_b_suffix) &&
                   key.compare(key.size() - std::strlen(q_b_suffix), std::strlen(q_b_suffix), q_b_suffix) == 0) {
            auto prefix = key.substr(0, key.size() - std::strlen(q_b_suffix));
            qkv_b[prefix].q = kv.second;
        } else if (key.size() >= std::strlen(k_b_suffix) &&
                   key.compare(key.size() - std::strlen(k_b_suffix), std::strlen(k_b_suffix), k_b_suffix) == 0) {
            auto prefix = key.substr(0, key.size() - std::strlen(k_b_suffix));
            qkv_b[prefix].k = kv.second;
        } else if (key.size() >= std::strlen(v_b_suffix) &&
                   key.compare(key.size() - std::strlen(v_b_suffix), std::strlen(v_b_suffix), v_b_suffix) == 0) {
            auto prefix = key.substr(0, key.size() - std::strlen(v_b_suffix));
            qkv_b[prefix].v = kv.second;
        }
    }

    for (const auto& kv : qkv_w) {
        const auto& prefix = kv.first;
        const auto& q = kv.second.q;
        const auto& k = kv.second.k;
        const auto& v = kv.second.v;
        auto fused_key_w = prefix + ".self_attn.in_proj_weight";
        if (!weights.count(fused_key_w) && q.defined() && k.defined() && v.defined()) {
            const int64_t cat_dim_w = (q.dim() == 3) ? 1 : 0;
            auto in_proj_w = torch::cat({q, k, v}, cat_dim_w).contiguous();
            weights[fused_key_w] = in_proj_w;
            weights[drop_self_attn_alias(fused_key_w)] = in_proj_w;
        }
    }

    for (const auto& kv : qkv_b) {
        const auto& prefix = kv.first;
        const auto& q = kv.second.q;
        const auto& k = kv.second.k;
        const auto& v = kv.second.v;
        auto fused_key_b = prefix + ".self_attn.in_proj_bias";
        if (!weights.count(fused_key_b) && q.defined() && k.defined() && v.defined()) {
            const int64_t cat_dim_b = (q.dim() == 2) ? 1 : 0;
            auto in_proj_b = torch::cat({q, k, v}, cat_dim_b).contiguous();
            weights[fused_key_b] = in_proj_b;
            weights[drop_self_attn_alias(fused_key_b)] = in_proj_b;
        }
    }
}
