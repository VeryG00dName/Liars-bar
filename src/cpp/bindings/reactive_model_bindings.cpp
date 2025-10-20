#include "reactive_model_forward.h"
#include "weight_utils.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

void bind_reactive_model(py::module_& m) {
    // Expose the main forward function
    m.def(
        "forward_packed_cpp",
        [](const torch::Tensor& obs_sequence,
           const torch::Tensor& action_sequence,
           const torch::Tensor& agent_types,
           const torch::Tensor& positions,
           py::dict weights_py,
           const torch::Tensor& policy_indices,
           const torch::optional<torch::Tensor>& padding_mask,
           int64_t num_layers,
           int64_t num_heads,
           int64_t hidden_dim,
           int64_t num_experts,
           int64_t top_k,
           int64_t count_pad,
           int64_t tflag_pad) {
            // Convert Python dict to c10::Dict
            c10::Dict<std::string, torch::Tensor> weights;
            for (auto item : weights_py) {
                std::string key = py::cast<std::string>(item.first);
                torch::Tensor value = py::cast<torch::Tensor>(item.second);
                weights.insert(key, value);
            }

            // Call C++ function
            return forward_packed_cpp(
                obs_sequence,
                action_sequence,
                agent_types,
                positions,
                weights,
                policy_indices,
                padding_mask,
                num_layers,
                num_heads,
                hidden_dim,
                num_experts,
                top_k,
                count_pad,
                tflag_pad
            );
        },
        py::arg("obs_sequence"),
        py::arg("action_sequence"),
        py::arg("agent_types"),
        py::arg("positions"),
        py::arg("weights"),
        py::arg("policy_indices"),
        py::arg("padding_mask") = torch::nullopt,
        py::arg("num_layers") = 2,
        py::arg("num_heads") = 4,
        py::arg("hidden_dim") = 256,
        py::arg("num_experts") = 8,
        py::arg("top_k") = 2,
        py::arg("count_pad") = 4,
        py::arg("tflag_pad") = 3,
        R"doc(
        Stateless forward pass for PPOReactiveModel with batched weights.

        This function computes the forward pass using a small batched weight cache [W, ...]
        and a per-sample `policy_indices` tensor [B] to select the appropriate weights.

        Args:
            obs_sequence: Observation sequences [B, T, obs_dim]
            action_sequence: Action sequences [B, T] (long tensor)
            agent_types: Agent type IDs [B, T] (long tensor)
            positions: Position IDs [B, T] (long tensor)
            weights: Small batched weight dictionary [W, ...]
            policy_indices: Indices selecting weights per sample [B]
            padding_mask: Optional padding mask [B, T] (bool, True=padding)
            num_layers: Number of transformer layers (default: 2)
            num_heads: Number of attention heads (default: 4)
            hidden_dim: Hidden dimension size (default: 256)
            num_experts: Number of MoE experts (default: 8)
            top_k: Top-K experts to activate (default: 2)
            count_pad: Padding index for count embeddings (default: 4)
            tflag_pad: Padding index for table flag embeddings (default: 3)

        Returns:
            tuple: (action_logits, opp_logits, state_values, win_logits)
                - action_logits: [B, T, action_dim]
                - opp_logits: [B, T, action_dim]
                - state_values: [B, T, 1]
                - win_logits: [B, T, 1]
        )doc"
    );

    // Expose weight utilities
    m.def(
        "prestack_moe_expert_weights",
        [](py::dict state_dict_py, int64_t num_layers, int64_t num_experts) {
            // Convert Python dict to std::unordered_map
            std::unordered_map<std::string, torch::Tensor> state_dict;
            for (auto item : state_dict_py) {
                std::string key = py::cast<std::string>(item.first);
                torch::Tensor value = py::cast<torch::Tensor>(item.second);
                state_dict[key] = value;
            }

            // Call C++ function
            prestack_moe_expert_weights(state_dict, num_layers, num_experts);

            // Convert back to Python dict
            py::dict result;
            for (const auto& pair : state_dict) {
                result[py::cast(pair.first)] = py::cast(pair.second);
            }
            return result;
        },
        py::arg("state_dict"),
        py::arg("num_layers"),
        py::arg("num_experts"),
        R"doc(
        Pre-stack MoE expert weights for efficient batched inference.

        Reorganizes per-expert weights into stacked tensors optimized for
        the MoE CUDA kernel.

        Args:
            state_dict: Weight dictionary (will be modified in-place and returned)
            num_layers: Number of transformer layers
            num_experts: Number of MoE experts per layer

        Returns:
            dict: Modified state_dict with added stacked expert weights
        )doc"
    );

    m.def(
        "create_moe_weight_pointers",
        [](py::dict state_dict_py, int64_t num_layers, int64_t num_experts) {
            // Convert Python dict to std::unordered_map
            std::unordered_map<std::string, torch::Tensor> state_dict;
            for (auto item : state_dict_py) {
                std::string key = py::cast<std::string>(item.first);
                torch::Tensor value = py::cast<torch::Tensor>(item.second);
                state_dict[key] = value;
            }

            // Call C++ function
            create_moe_weight_pointers(state_dict, num_layers, num_experts);

            // Convert back to Python dict
            py::dict result;
            for (const auto& pair : state_dict) {
                result[py::cast(pair.first)] = py::cast(pair.second);
            }
            return result;
        },
        py::arg("state_dict"),
        py::arg("num_layers"),
        py::arg("num_experts"),
        R"doc(
        Create pointer tensors for MoE expert weights.

        This must be called AFTER prestack_moe_expert_weights() and AFTER
        weights are moved to CUDA device.

        Args:
            state_dict: Weight dictionary (will be modified in-place and returned)
            num_layers: Number of transformer layers
            num_experts: Number of MoE experts per layer

        Returns:
            dict: Modified state_dict with added pointer tensors
        )doc"
    );

    // Note: C++ .pth loading helper removed; handled in Python.

    m.def(
        "batch_state_dicts",
        [](py::list state_dicts_py) {
            // Convert list of dicts to vector of unordered_maps
            std::vector<std::unordered_map<std::string, torch::Tensor>> state_dicts;

            for (auto dict_py : state_dicts_py) {
                std::unordered_map<std::string, torch::Tensor> state_dict;
                py::dict d = py::cast<py::dict>(dict_py);

                for (auto item : d) {
                    std::string key = py::cast<std::string>(item.first);
                    torch::Tensor value = py::cast<torch::Tensor>(item.second);
                    state_dict[key] = value;
                }

                state_dicts.push_back(state_dict);
            }

            // Call C++ function
            return batch_state_dicts(state_dicts);
        },
        py::arg("state_dicts"),
        R"doc(
        Batch multiple state_dicts into a single batched weight dictionary.

        Args:
            state_dicts: List of state dictionaries to batch

        Returns:
            dict: Batched weight dictionary with batch dimension
        )doc"
    );

    m.def(
        "add_fixed_buffers",
        [](py::dict weights_py, const std::string& device_str) {
            // Convert to c10::Dict
            c10::Dict<std::string, torch::Tensor> weights;
            for (auto item : weights_py) {
                std::string key = py::cast<std::string>(item.first);
                torch::Tensor value = py::cast<torch::Tensor>(item.second);
                weights.insert(key, value);
            }

            // Parse device
            torch::Device device(device_str);

            // Call C++ function
            add_fixed_buffers(weights, device);

            // Convert back to Python dict
            py::dict result;
            for (const auto& pair : weights) {
                result[py::cast(pair.key())] = py::cast(pair.value());
            }
            return result;
        },
        py::arg("weights"),
        py::arg("device") = "cpu",
        R"doc(
        Add fixed lookup table buffers to the weight dictionary.

        Args:
            weights: Weight dictionary to modify
            device: Target device for buffers (default: "cpu")

        Returns:
            dict: Modified weight dictionary with added buffers
        )doc"
    );
}
