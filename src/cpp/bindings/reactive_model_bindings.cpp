#include "reactive_model_forward.h"
#include "weight_utils.h"
#include "indexed_kernels.h"
#include "moe_backward_kernels.h"
#include "moe_cutlass_kernels.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

void bind_reactive_model(py::module_& m) {
    // Expose the main forward function
    m.def(
        "forward_packed",
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
            return forward_packed(
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

    // Expose MoE backward function
    m.def(
        "grouped_moe_backward",
        [](const torch::Tensor& grad_output,
           const torch::Tensor& input_grouped,
           const torch::Tensor& hidden_grouped,
           const torch::Tensor& routing_weights_grouped,
           const torch::Tensor& indices_grouped,
           const torch::Tensor& w1_weights,
           const torch::Tensor& w2_weights,
           const torch::Tensor& b1_biases,
           const torch::Tensor& b2_biases,
           const std::vector<int64_t>& m_sizes,
           const std::vector<int64_t>& policy_ids,
           const std::vector<int64_t>& expert_ids,
           const std::vector<int64_t>& token_offsets,
           torch::Tensor& grad_input,
           torch::Tensor& grad_w1,
           torch::Tensor& grad_w2,
           torch::Tensor& grad_b1,
           torch::Tensor& grad_b2) {
            lb::moe::cutlass_grouped_moe_backward(
                grad_output,
                input_grouped,
                hidden_grouped,
                routing_weights_grouped,
                indices_grouped,
                w1_weights,
                w2_weights,
                b1_biases,
                b2_biases,
                m_sizes,
                policy_ids,
                expert_ids,
                token_offsets,
                grad_input,
                grad_w1,
                grad_w2,
                grad_b1,
                grad_b2
            );
        },
        py::arg("grad_output"),
        py::arg("input_grouped"),
        py::arg("hidden_grouped"),
        py::arg("routing_weights_grouped"),
        py::arg("indices_grouped"),
        py::arg("w1_weights"),
        py::arg("w2_weights"),
        py::arg("b1_biases"),
        py::arg("b2_biases"),
        py::arg("m_sizes"),
        py::arg("policy_ids"),
        py::arg("expert_ids"),
        py::arg("token_offsets"),
        py::arg("grad_input"),
        py::arg("grad_w1"),
        py::arg("grad_w2"),
        py::arg("grad_b1"),
        py::arg("grad_b2"),
        R"doc(
        Custom backward pass for grouped MoE FFN layers.

        Phase 1: Computes gradients for W1, W2, b1, b2 and input (dX).
        Routing weights are FROZEN (no routing gradients).

        Implements save-minimal policy: recomputes Z (pre-GELU activation)
        instead of saving it from forward pass.

        Args:
            grad_output: Upstream gradient [batch_size, hidden_dim] FP16
            input_grouped: Saved input in grouped order [total_tokens, hidden_dim] FP16
            hidden_grouped: Saved hidden state in grouped order [total_tokens, ffn_dim] FP16
            routing_weights_grouped: Saved routing weights [total_tokens] FP32
            indices_grouped: Mapping to original token order [total_tokens] int64
            w1_weights: W1 weights [num_policies, num_experts, ffn_dim, hidden_dim] FP16
            w2_weights: W2 weights [num_policies, num_experts, hidden_dim, ffn_dim] FP16
            b1_biases: b1 biases [num_policies, num_experts, ffn_dim] FP16
            b2_biases: b2 biases [num_policies, num_experts, hidden_dim] FP16
            m_sizes: Number of tokens per group
            policy_ids: Policy ID per group
            expert_ids: Expert ID per group
            token_offsets: Token offset per group
            grad_input: Output gradient w.r.t. input [batch_size, hidden_dim] FP16
            grad_w1: Output gradient w.r.t. W1 [num_policies, num_experts, ffn_dim, hidden_dim] FP32
            grad_w2: Output gradient w.r.t. W2 [num_policies, num_experts, hidden_dim, ffn_dim] FP32
            grad_b1: Output gradient w.r.t. b1 [num_policies, num_experts, ffn_dim] FP32
            grad_b2: Output gradient w.r.t. b2 [num_policies, num_experts, hidden_dim] FP32

        Note: All gradient tensors must be pre-allocated and zero-initialized.
        )doc"
    );

    // Expose grouped MoE forward (training) that also returns hidden_grouped
    m.def(
        "grouped_moe_forward",
        [](const torch::Tensor& input_grouped,
           const torch::Tensor& w1_weights,
           const torch::Tensor& w2_weights,
           const torch::Tensor& b1_biases,
           const torch::Tensor& b2_biases,
           const torch::Tensor& routing_weights_grouped,
           const std::vector<int64_t>& m_sizes,
           const std::vector<int64_t>& policy_ids,
           const std::vector<int64_t>& expert_ids,
           const std::vector<int64_t>& token_offsets) {
            TORCH_CHECK(input_grouped.dim() == 2, "input_grouped must be [total_tokens, hidden_dim]");
            TORCH_CHECK(routing_weights_grouped.dim() == 1, "routing_weights_grouped must be [total_tokens]");
            TORCH_CHECK(w1_weights.dim() == 4, "w1_weights must be [P,E,F,H]");
            TORCH_CHECK(w2_weights.dim() == 4, "w2_weights must be [P,E,H,F]");
            TORCH_CHECK(b1_biases.dim() == 3, "b1_biases must be [P,E,F]");
            TORCH_CHECK(b2_biases.dim() == 3, "b2_biases must be [P,E,H]");

            TORCH_CHECK(input_grouped.is_cuda(), "input_grouped must be CUDA tensor");
            TORCH_CHECK(w1_weights.is_cuda() && w2_weights.is_cuda(), "weights must be CUDA tensors");
            TORCH_CHECK(b1_biases.is_cuda() && b2_biases.is_cuda(), "biases must be CUDA tensors");
            TORCH_CHECK(routing_weights_grouped.is_cuda(), "routing_weights_grouped must be CUDA tensor");

            TORCH_CHECK(input_grouped.scalar_type() == torch::kFloat16, "input_grouped must be float16");
            TORCH_CHECK(routing_weights_grouped.scalar_type() == torch::kFloat32, "routing_weights_grouped must be float32");
            TORCH_CHECK(w1_weights.scalar_type() == torch::kFloat16, "w1_weights must be float16");
            TORCH_CHECK(w2_weights.scalar_type() == torch::kFloat16, "w2_weights must be float16");
            TORCH_CHECK(b1_biases.scalar_type() == torch::kFloat16, "b1_biases must be float16");
            TORCH_CHECK(b2_biases.scalar_type() == torch::kFloat16, "b2_biases must be float16");

            int64_t total_tokens = input_grouped.size(0);
            int64_t hidden_dim = input_grouped.size(1);
            int64_t num_policies = w1_weights.size(0);
            int64_t num_experts = w1_weights.size(1);
            int64_t ffn_dim = w1_weights.size(2);

            TORCH_CHECK(static_cast<int64_t>(m_sizes.size()) == static_cast<int64_t>(policy_ids.size()), "m_sizes/policy_ids length mismatch");
            TORCH_CHECK(static_cast<int64_t>(m_sizes.size()) == static_cast<int64_t>(expert_ids.size()), "m_sizes/expert_ids length mismatch");
            TORCH_CHECK(static_cast<int64_t>(m_sizes.size()) == static_cast<int64_t>(token_offsets.size()), "m_sizes/token_offsets length mismatch");

            // Allocate outputs
            auto opts_half = input_grouped.options().dtype(torch::kFloat16);
            torch::Tensor hidden_grouped = torch::empty({total_tokens, ffn_dim}, opts_half);
            torch::Tensor output_grouped = torch::empty({total_tokens, hidden_dim}, opts_half);

            // Build pointer arrays (host)
            std::vector<uintptr_t> input_ptrs(m_sizes.size());
            std::vector<uintptr_t> w1_ptrs(m_sizes.size());
            std::vector<uintptr_t> w2_ptrs(m_sizes.size());
            std::vector<uintptr_t> b1_ptrs(m_sizes.size());
            std::vector<uintptr_t> b2_ptrs(m_sizes.size());
            std::vector<uintptr_t> hidden_ptrs(m_sizes.size());
            std::vector<uintptr_t> output_ptrs(m_sizes.size());
            std::vector<uintptr_t> routing_ptrs(m_sizes.size());

            auto input_base = reinterpret_cast<uintptr_t>(input_grouped.data_ptr<at::Half>());
            auto hidden_base = reinterpret_cast<uintptr_t>(hidden_grouped.data_ptr<at::Half>());
            auto output_base = reinterpret_cast<uintptr_t>(output_grouped.data_ptr<at::Half>());
            auto routing_base = reinterpret_cast<uintptr_t>(routing_weights_grouped.data_ptr<float>());

            for (size_t g = 0; g < m_sizes.size(); ++g) {
                int64_t M = m_sizes[g];
                int64_t pi = policy_ids[g];
                int64_t ei = expert_ids[g];
                int64_t offset = token_offsets[g];

                TORCH_CHECK(pi >= 0 && pi < num_policies, "policy id out of range");
                TORCH_CHECK(ei >= 0 && ei < num_experts, "expert id out of range");
                TORCH_CHECK(offset >= 0 && offset + M <= total_tokens, "token offset out of range");

                input_ptrs[g]  = input_base  + static_cast<uintptr_t>(offset) * hidden_dim * sizeof(at::Half);
                hidden_ptrs[g] = hidden_base + static_cast<uintptr_t>(offset) * ffn_dim   * sizeof(at::Half);
                output_ptrs[g] = output_base + static_cast<uintptr_t>(offset) * hidden_dim * sizeof(at::Half);
                routing_ptrs[g]= routing_base+ static_cast<uintptr_t>(offset) * sizeof(float);

                auto W1 = w1_weights.index({pi, ei});
                auto W2 = w2_weights.index({pi, ei});
                auto B1 = b1_biases.index({pi, ei});
                auto B2 = b2_biases.index({pi, ei});
                TORCH_CHECK(W1.is_contiguous(), "W1 slice must be contiguous");
                TORCH_CHECK(W2.is_contiguous(), "W2 slice must be contiguous");
                TORCH_CHECK(B1.is_contiguous(), "b1 slice must be contiguous");
                TORCH_CHECK(B2.is_contiguous(), "b2 slice must be contiguous");

                w1_ptrs[g] = reinterpret_cast<uintptr_t>(W1.data_ptr<at::Half>());
                w2_ptrs[g] = reinterpret_cast<uintptr_t>(W2.data_ptr<at::Half>());
                b1_ptrs[g] = reinterpret_cast<uintptr_t>(B1.data_ptr<at::Half>());
                b2_ptrs[g] = reinterpret_cast<uintptr_t>(B2.data_ptr<at::Half>());
            }

            // Call fused CUTLASS path that also writes hidden buffer
            lb::moe::cutlass_grouped_moe_forward_with_hidden(
                input_ptrs.data(),
                w1_ptrs.data(),
                b1_ptrs.data(),
                w2_ptrs.data(),
                b2_ptrs.data(),
                hidden_ptrs.data(),
                output_ptrs.data(),
                routing_ptrs.data(),
                m_sizes.data(),
                policy_ids.data(),
                expert_ids.data(),
                token_offsets.data(),
                static_cast<int64_t>(m_sizes.size()),
                hidden_dim,
                ffn_dim
            );

            return std::make_tuple(hidden_grouped, output_grouped);
        },
        py::arg("input_grouped"),
        py::arg("w1_weights"),
        py::arg("w2_weights"),
        py::arg("b1_biases"),
        py::arg("b2_biases"),
        py::arg("routing_weights_grouped"),
        py::arg("m_sizes"),
        py::arg("policy_ids"),
        py::arg("expert_ids"),
        py::arg("token_offsets"),
        R"doc(
        Grouped MoE FFN forward used in training.

        Computes per-group FFN in FP32 accumulate and returns both hidden activations (post-GELU)
        and final outputs after W2 and routing-weight scaling. Shapes:
          - input_grouped: [total_tokens, hidden_dim] (FP16)
          - w1_weights:   [P, E, ffn_dim, hidden_dim] (FP16)
          - w2_weights:   [P, E, hidden_dim, ffn_dim] (FP16)
          - b1_biases:    [P, E, ffn_dim] (FP16)
          - b2_biases:    [P, E, hidden_dim] (FP16)
          - routing_weights_grouped: [total_tokens] (FP32)
          - m_sizes/policy_ids/expert_ids/token_offsets: lists describing groups

        Returns:
          (hidden_grouped: [total_tokens, ffn_dim] FP16,
           output_grouped: [total_tokens, hidden_dim] FP16)
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

            // Call C++ function and convert c10::Dict to Python dict
            auto batched = batch_state_dicts(state_dicts);
            py::dict result;
            for (const auto& pair : batched) {
                result[py::cast(pair.key())] = py::cast(pair.value());
            }
            return result;
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

    // ========================================================================
    // Layer-by-Layer Testing Functions
    // ========================================================================

    m.def(
        "test_action_decomposition",
        [](const torch::Tensor& action_sequence,
           py::dict weights_py,
           const torch::Tensor& policy_indices,
           const torch::optional<torch::Tensor>& padding_mask,
           int64_t count_pad,
           int64_t tflag_pad) {
            c10::Dict<std::string, torch::Tensor> weights;
            for (auto item : weights_py) {
                weights.insert(py::cast<std::string>(item.first), py::cast<torch::Tensor>(item.second));
            }
            return test_action_decomposition(action_sequence, weights, policy_indices, padding_mask, count_pad, tflag_pad);
        },
        py::arg("action_sequence"),
        py::arg("weights"),
        py::arg("policy_indices"),
        py::arg("padding_mask") = torch::nullopt,
        py::arg("count_pad") = 4,
        py::arg("tflag_pad") = 3,
        "Test action decomposition into (kind, count, table_flag)"
    );

    m.def(
        "test_embeddings",
        [](const torch::Tensor& obs_sequence,
           const torch::Tensor& act_kind_ids,
           const torch::Tensor& count_ids,
           const torch::Tensor& table_flag_ids,
           const torch::Tensor& agent_types,
           const torch::Tensor& positions,
           py::dict weights_py,
           const torch::Tensor& policy_indices) {
            c10::Dict<std::string, torch::Tensor> weights;
            for (auto item : weights_py) {
                weights.insert(py::cast<std::string>(item.first), py::cast<torch::Tensor>(item.second));
            }
            auto result = test_embeddings(obs_sequence, act_kind_ids, count_ids, table_flag_ids, agent_types, positions, weights, policy_indices);

            py::dict py_result;
            for (const auto& pair : result) {
                py_result[py::cast(pair.key())] = py::cast(pair.value());
            }
            return py_result;
        },
        py::arg("obs_sequence"),
        py::arg("act_kind_ids"),
        py::arg("count_ids"),
        py::arg("table_flag_ids"),
        py::arg("agent_types"),
        py::arg("positions"),
        py::arg("weights"),
        py::arg("policy_indices"),
        "Test all embeddings (obs, action, agent, position)"
    );

    m.def(
        "test_gating",
        [](const torch::Tensor& obs_embed,
           const torch::Tensor& action_embed,
           const torch::Tensor& agent_embed,
           const torch::Tensor& position_embed,
           py::dict weights_py,
           const torch::Tensor& policy_indices) {
            c10::Dict<std::string, torch::Tensor> weights;
            for (auto item : weights_py) {
                weights.insert(py::cast<std::string>(item.first), py::cast<torch::Tensor>(item.second));
            }
            auto result = test_gating(obs_embed, action_embed, agent_embed, position_embed, weights, policy_indices);

            py::dict py_result;
            for (const auto& pair : result) {
                py_result[py::cast(pair.key())] = py::cast(pair.value());
            }
            return py_result;
        },
        py::arg("obs_embed"),
        py::arg("action_embed"),
        py::arg("agent_embed"),
        py::arg("position_embed"),
        py::arg("weights"),
        py::arg("policy_indices"),
        "Test gating networks"
    );

    m.def(
        "test_fusion",
        [](const torch::Tensor& g_obs,
           const torch::Tensor& g_action,
           const torch::Tensor& g_agent,
           const torch::Tensor& g_position,
           const torch::Tensor& obs_embed,
           const torch::Tensor& action_embed,
           const torch::Tensor& agent_embed,
           const torch::Tensor& position_embed,
           int64_t hidden_dim) {
            auto result = test_fusion(g_obs, g_action, g_agent, g_position, obs_embed, action_embed, agent_embed, position_embed, hidden_dim);

            py::dict py_result;
            for (const auto& pair : result) {
                py_result[py::cast(pair.key())] = py::cast(pair.value());
            }
            return py_result;
        },
        py::arg("g_obs"),
        py::arg("g_action"),
        py::arg("g_agent"),
        py::arg("g_position"),
        py::arg("obs_embed"),
        py::arg("action_embed"),
        py::arg("agent_embed"),
        py::arg("position_embed"),
        py::arg("hidden_dim"),
        "Test fusion layer (gated combination + layer norm)"
    );

    m.def(
        "test_attention_layer",
        [](const torch::Tensor& x,
           py::dict weights_py,
           const torch::Tensor& policy_indices,
           const torch::optional<torch::Tensor>& padding_mask,
           int64_t layer_idx,
           int64_t num_heads,
           int64_t hidden_dim) {
            c10::Dict<std::string, torch::Tensor> weights;
            for (auto item : weights_py) {
                weights.insert(py::cast<std::string>(item.first), py::cast<torch::Tensor>(item.second));
            }
            auto result = test_attention_layer(x, weights, policy_indices, padding_mask, layer_idx, num_heads, hidden_dim);

            py::dict py_result;
            for (const auto& pair : result) {
                py_result[py::cast(pair.key())] = py::cast(pair.value());
            }
            return py_result;
        },
        py::arg("x"),
        py::arg("weights"),
        py::arg("policy_indices"),
        py::arg("padding_mask") = torch::nullopt,
        py::arg("layer_idx") = 0,
        py::arg("num_heads") = 4,
        py::arg("hidden_dim") = 256,
        "Test attention layer"
    );

    m.def(
        "test_moe_layer",
        [](const torch::Tensor& x,
           py::dict weights_py,
           const torch::Tensor& policy_indices,
           int64_t layer_idx,
           int64_t num_experts,
           int64_t top_k,
           int64_t hidden_dim) {
            c10::Dict<std::string, torch::Tensor> weights;
            for (auto item : weights_py) {
                weights.insert(py::cast<std::string>(item.first), py::cast<torch::Tensor>(item.second));
            }
            auto result = test_moe_layer(x, weights, policy_indices, layer_idx, num_experts, top_k, hidden_dim);

            py::dict py_result;
            for (const auto& pair : result) {
                py_result[py::cast(pair.key())] = py::cast(pair.value());
            }
            return py_result;
        },
        py::arg("x"),
        py::arg("weights"),
        py::arg("policy_indices"),
        py::arg("layer_idx") = 0,
        py::arg("num_experts") = 8,
        py::arg("top_k") = 2,
        py::arg("hidden_dim") = 256,
        "Test MoE layer"
    );

    // Diagnostic helpers for MoE
    m.def(
        "test_moe_routing_sort",
        [](const torch::Tensor& x,
           py::dict weights_py,
           const torch::Tensor& policy_indices,
           int64_t layer_idx,
           int64_t num_experts,
           int64_t top_k) {
            c10::Dict<std::string, torch::Tensor> weights;
            for (auto item : weights_py) {
                weights.insert(py::cast<std::string>(item.first), py::cast<torch::Tensor>(item.second));
            }
            auto result = test_moe_routing_sort(x, weights, policy_indices, layer_idx, num_experts, top_k);
            py::dict py_result;
            for (const auto& pair : result) {
                py_result[py::cast(pair.key())] = py::cast(pair.value());
            }
            return py_result;
        },
        py::arg("x"),
        py::arg("weights"),
        py::arg("policy_indices"),
        py::arg("layer_idx"),
        py::arg("num_experts"),
        py::arg("top_k"),
        "Test MoE routing and sorting stages"
    );

    m.def(
        "test_moe_group_ranges",
        &test_moe_group_ranges,
        py::arg("sorted_expert_indices"),
        py::arg("sorted_policy_indices"),
        "Build (start,count,expert,policy) group ranges from sorted indices"
    );

    m.def(
        "test_heads",
        [](const torch::Tensor& transformer_output,
           py::dict weights_py,
           const torch::Tensor& policy_indices,
           int64_t num_experts) {
            c10::Dict<std::string, torch::Tensor> weights;
            for (auto item : weights_py) {
                weights.insert(py::cast<std::string>(item.first), py::cast<torch::Tensor>(item.second));
            }
            auto result = test_heads(transformer_output, weights, policy_indices, num_experts);

            py::dict py_result;
            for (const auto& pair : result) {
                py_result[py::cast(pair.key())] = py::cast(pair.value());
            }
            return py_result;
        },
        py::arg("transformer_output"),
        py::arg("weights"),
        py::arg("policy_indices"),
        py::arg("num_experts") = 8,
        "Test per-expert head computation"
    );

    m.def(
        "reduce_expert_heads",
        &reduce_expert_heads,
        py::arg("stacked"),
        py::arg("topk_indices"),
        py::arg("topk_scores"),
        "Reduce expert heads using MoE routing weights"
    );

    // Simple test wrapper for indexed_batched_linear
    m.def(
        "test_indexed_batched_linear",
        [](const torch::Tensor& input,
           const torch::Tensor& weight_cache,
           const torch::Tensor& bias_cache,
           const torch::Tensor& policy_indices) {
            std::unordered_map<std::string, std::chrono::microseconds> timers;
            return indexed_batched_linear(input, weight_cache, bias_cache, policy_indices, timers);
        },
        py::arg("input"),
        py::arg("weight_cache"),
        py::arg("bias_cache"),
        py::arg("policy_indices"),
        "Test indexed_batched_linear directly"
    );
}
