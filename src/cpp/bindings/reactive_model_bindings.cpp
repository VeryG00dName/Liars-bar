#include "model_forward.h"
#include "model_layer.h"
#include "weight_utils.h"
#include "lb_kernels.h"
#include "moe_cutlass_kernels.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

// Helper to convert py::dict to c10::Dict
c10::Dict<std::string, torch::Tensor> py_dict_to_c10_dict(const py::dict& py_dict) {
    c10::Dict<std::string, torch::Tensor> c10_dict;
    for (auto item : py_dict) {
        c10_dict.insert(py::cast<std::string>(item.first), py::cast<torch::Tensor>(item.second));
    }
    return c10_dict;
}

// Helper to convert c10::Dict to py::dict
py::dict c10_dict_to_py_dict(const c10::Dict<std::string, torch::Tensor>& c10_dict) {
    py::dict py_dict;
    for (const auto& pair : c10_dict) {
        py_dict[py::cast(pair.key())] = py::cast(pair.value());
    }
    return py_dict;
}

void bind_reactive_model(py::module_& m) {
    // Expose the main forward functions
    m.def("forward_packed",
        [](const torch::Tensor& obs_sequence, const torch::Tensor& action_sequence,
           const torch::Tensor& agent_types, const torch::Tensor& positions,
           py::dict weights_py, const torch::Tensor& policy_indices,
           const torch::optional<torch::Tensor>& padding_mask,
           int64_t num_layers, int64_t num_heads, int64_t hidden_dim,
           int64_t num_experts, int64_t top_k, int64_t count_pad, int64_t tflag_pad) {
            std::unordered_map<std::string, std::chrono::microseconds> timers;
            auto result = lb::forward::forward_packed(
                obs_sequence, action_sequence, agent_types, positions,
                py_dict_to_c10_dict(weights_py), policy_indices, padding_mask,
                num_layers, num_heads, hidden_dim, num_experts, top_k, count_pad, tflag_pad, &timers);
            
            // Convert timers to Python dict (microseconds as int64)
            py::dict timers_py;
            for (const auto& kv : timers) {
                timers_py[py::cast(kv.first)] = static_cast<int64_t>(kv.second.count());
            }
            
            // Return tuple: (forward_results, timers_dict)
            return py::make_tuple(
                std::get<0>(result),  // action_logits
                std::get<1>(result), // opp_logits
                std::get<2>(result), // state_values
                std::get<3>(result), // win_logits
                timers_py            // timers dict
            );
        },
        py::arg("obs_sequence"), py::arg("action_sequence"), py::arg("agent_types"),
        py::arg("positions"), py::arg("weights"), py::arg("policy_indices"),
        py::arg("padding_mask") = torch::nullopt, py::arg("num_layers") = 2,
        py::arg("num_heads") = 4, py::arg("hidden_dim") = 256,
        py::arg("num_experts") = 8, py::arg("top_k") = 2,
        py::arg("count_pad") = 4, py::arg("tflag_pad") = 3,
        "Inference forward pass for the reactive model. Returns (action_logits, opp_logits, state_values, win_logits, timers_dict).");

    // Optional: enable/disable thread-local workspace cache for forward_packed
    m.def(
        "enable_forward_moe_workspace_cache",
        [](int64_t num_layers, int64_t max_batch_size, int64_t max_seq_length, int64_t hidden_dim, int64_t top_k) {
            lb::forward::enable_forward_workspace_cache(num_layers, max_batch_size, max_seq_length, hidden_dim, top_k);
        },
        py::arg("num_layers"), py::arg("max_batch_size"), py::arg("max_seq_length"), py::arg("hidden_dim"), py::arg("top_k"),
        "Enable thread-local MoE workspace cache for C++ forward_packed.");

    m.def(
        "disable_forward_moe_workspace_cache",
        []() {
            lb::forward::disable_forward_workspace_cache();
        },
        "Disable and free the thread-local MoE workspace cache.");

    // Expose layer/diagnostic functions for testing
    m.def("action_decomposition",
        [](const torch::Tensor& action_sequence, py::dict weights_py, const torch::Tensor& policy_indices,
           const torch::optional<torch::Tensor>& padding_mask, int64_t count_pad, int64_t tflag_pad) {
            return lb::model::action_decomposition(action_sequence, py_dict_to_c10_dict(weights_py),
                policy_indices, padding_mask, count_pad, tflag_pad);
        }, py::arg("action_sequence"), py::arg("weights"), py::arg("policy_indices"),
           py::arg("padding_mask") = torch::nullopt, py::arg("count_pad") = 4, py::arg("tflag_pad") = 3);

    m.def("compute_embeddings",
        [](const torch::Tensor& obs_sequence, const torch::Tensor& action_sequence,
           const torch::Tensor& agent_types, const torch::Tensor& positions,
           py::dict weights_py, const torch::Tensor& policy_indices,
           const torch::optional<torch::Tensor>& padding_mask, int64_t count_pad, int64_t tflag_pad) {
            auto result = lb::model::compute_embeddings(obs_sequence, action_sequence, agent_types, positions,
                py_dict_to_c10_dict(weights_py), policy_indices, padding_mask, count_pad, tflag_pad);
            return c10_dict_to_py_dict(result);
        }, py::arg("obs_sequence"), py::arg("action_sequence"), py::arg("agent_types"), py::arg("positions"),
           py::arg("weights"), py::arg("policy_indices"), py::arg("padding_mask") = torch::nullopt,
           py::arg("count_pad") = 4, py::arg("tflag_pad") = 3);
    
    m.def("gating",
        [](const torch::Tensor& obs_embed, const torch::Tensor& action_embed, const torch::Tensor& agent_embed,
           const torch::Tensor& position_embed, py::dict weights_py, const torch::Tensor& policy_indices) {
            auto result = lb::model::gating(obs_embed, action_embed, agent_embed, position_embed,
                py_dict_to_c10_dict(weights_py), policy_indices);
            return c10_dict_to_py_dict(result);
        }, py::arg("obs_embed"), py::arg("action_embed"), py::arg("agent_embed"), py::arg("position_embed"),
           py::arg("weights"), py::arg("policy_indices"));

    m.def("fuse_embeddings",
        [](const torch::Tensor& g_obs, const torch::Tensor& g_action, const torch::Tensor& g_agent, const torch::Tensor& g_position,
           const torch::Tensor& obs_embed, const torch::Tensor& action_embed, const torch::Tensor& agent_embed,
           const torch::Tensor& position_embed, int64_t hidden_dim) {
            auto result = lb::model::fuse_embeddings(g_obs, g_action, g_agent, g_position, obs_embed,
                action_embed, agent_embed, position_embed, hidden_dim);
            return c10_dict_to_py_dict(result);
        }, py::arg("g_obs"), py::arg("g_action"), py::arg("g_agent"), py::arg("g_position"),
           py::arg("obs_embed"), py::arg("action_embed"), py::arg("agent_embed"), py::arg("position_embed"),
           py::arg("hidden_dim"));

    m.def("transformer_layer",
        [](const torch::Tensor& x, const torch::Tensor& policy_indices,
           py::dict weights_py, int64_t layer_idx, int64_t num_heads, int64_t hidden_dim, int64_t top_k,
           int64_t num_experts, int64_t num_policies) {
            auto weights = py_dict_to_c10_dict(weights_py);
            std::string prefix = "transformer.layers." + std::to_string(layer_idx);
            return lb::model::transformer_layer(
                x, policy_indices,
                weights.at(prefix + ".self_attn.in_proj_weight"),
                weights.at(prefix + ".self_attn.in_proj_bias"),
                weights.at(prefix + ".self_attn.out_proj.weight"),
                weights.at(prefix + ".self_attn.out_proj.bias"),
                weights.at(prefix + ".norm1.weight"),
                weights.at(prefix + ".norm1.bias"),
                weights.at(prefix + ".moe.gate.weight"),
                weights.at(prefix + ".moe.gate.bias"),
                weights.at(prefix + ".moe.experts.w1"),
                weights.at(prefix + ".moe.experts.w2"),
                weights.at(prefix + ".moe.experts.b1"),
                weights.at(prefix + ".moe.experts.b2"),
                weights.at(prefix + ".norm2.weight"),
                weights.at(prefix + ".norm2.bias"),
                num_heads, hidden_dim, top_k, num_experts, num_policies);
        },
        py::arg("x"), py::arg("policy_indices"), py::arg("weights"), py::arg("layer_idx"),
        py::arg("num_heads"), py::arg("hidden_dim"), py::arg("top_k"),
        py::arg("num_experts"), py::arg("num_policies"),
        "Single transformer layer (attention + MoE). Returns (x_next, gate_logits, topk_indices, topk_scores).");

    m.def("compute_heads",
        [](const torch::Tensor& transformer_output, py::dict weights_py, const torch::Tensor& policy_indices, int64_t num_experts) {
            auto result = lb::model::compute_heads(transformer_output, py_dict_to_c10_dict(weights_py), policy_indices, num_experts);
            return c10_dict_to_py_dict(result);
        }, py::arg("transformer_output"), py::arg("weights"), py::arg("policy_indices"), py::arg("num_experts"));

    m.def("reduce_expert_heads", &lb::model::reduce_expert_heads,
        py::arg("stacked"), py::arg("topk_indices"), py::arg("topk_scores"));

    // ========================================================================
    // Weight Utilities
    // ========================================================================
    m.def(
        "prestack_moe_expert_weights",
        [](py::dict state_dict_py, int64_t num_layers, int64_t num_experts) {
            std::unordered_map<std::string, torch::Tensor> state_dict;
            for (auto item : state_dict_py) {
                state_dict[py::cast<std::string>(item.first)] = py::cast<torch::Tensor>(item.second);
            }
            prestack_moe_expert_weights(state_dict, num_layers, num_experts);
            py::dict result;
            for (const auto& pair : state_dict) {
                result[py::cast(pair.first)] = py::cast(pair.second);
            }
            return result;
        },
        py::arg("state_dict"), py::arg("num_layers"), py::arg("num_experts"),
        "Pre-stack MoE expert weights for efficient batched inference.");

    m.def(
        "create_moe_weight_pointers",
        [](py::dict state_dict_py, int64_t num_layers, int64_t num_experts) {
            auto c10_dict = py_dict_to_c10_dict(state_dict_py);
            create_moe_weight_pointers(c10_dict, num_layers, num_experts);
            return c10_dict_to_py_dict(c10_dict);
        },
        py::arg("state_dict"), py::arg("num_layers"), py::arg("num_experts"),
        "Create pointer tensors for MoE expert weights.");

    m.def(
        "add_fixed_buffers",
        [](py::dict weights_py, const std::string& device_str) {
            auto c10_dict = py_dict_to_c10_dict(weights_py);
            torch::Device device(device_str);
            add_fixed_buffers(c10_dict, device);
            return c10_dict_to_py_dict(c10_dict);
        },
        py::arg("weights"), py::arg("device") = "cpu",
        "Add fixed lookup table buffers to the weight dictionary.");
}
