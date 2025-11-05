#pragma once

#include <torch/torch.h>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <chrono>

#include "vec_arena.h"
#include "moe_cutlass_kernels.h"

namespace execution_core {

// Hash function for std::pair<int, int> (must be public for use in return type)
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

/**
 * Result of neural network inference for a single request.
 */
struct InferenceResult {
    uint8_t action;              // Selected action
    float log_prob;              // Log probability of the action
    float state_value;           // Value prediction
    torch::Tensor opp_logits;    // Opponent action predictions [7]
};

/**
 * Core orchestrator for batched neural network inference.
 *
 * This class implements the unified execution strategy:
 * 1. Find maximum sequence length across all requests
 * 2. Pad all tensors to this exact length (no rounding)
 * 3. Split into batches of max_inference_batch_size
 * 4. Execute forward_packed for each batch
 * 5. Return actions and auxiliary outputs
 */
class NeuralInferenceOrchestrator {
public:
    NeuralInferenceOrchestrator(
        const c10::Dict<std::string, torch::Tensor>& batched_weights,
        const std::unordered_map<int, int>& policy_id_to_index,
        int64_t max_inference_batch_size = 512,
        int64_t num_layers = 2,
        int64_t num_heads = 4,
        int64_t hidden_dim = 256,
        int64_t num_experts = 8,
        int64_t top_k = 2,
        bool use_argmax = false
    );

    ~NeuralInferenceOrchestrator();

    std::unordered_map<std::pair<int, int>, InferenceResult, pair_hash>
    run_inference(
        const std::unordered_map<int, std::vector<PolicyRequest>>& requests_by_policy
    );

    std::unordered_map<std::string, int64_t> get_timing_stats() const {
        std::unordered_map<std::string, int64_t> result;
        for (const auto& kv : timing_stats_) {
            result[kv.first] = kv.second.count();
        }
        return result;
    }

    void reset_timing_stats() {
        timing_stats_.clear();
    }

private:
    c10::Dict<std::string, torch::Tensor> batched_weights_;
    std::unordered_map<int, int> policy_id_to_index_;
    int64_t max_inference_batch_size_;
    int64_t num_layers_;
    int64_t num_heads_;
    int64_t hidden_dim_;
    int64_t num_experts_;
    int64_t top_k_;
    bool use_argmax_ = false;
    std::vector<lb::moe::MoEWorkspace> moe_workspaces_;
    mutable std::unordered_map<std::string, std::chrono::microseconds> timing_stats_;

    int64_t find_max_sequence_length(
        const std::unordered_map<int, std::vector<PolicyRequest>>& requests_by_policy
    ) const;

    std::tuple<
        torch::Tensor,  // obs_sequence
        torch::Tensor,  // action_sequence
        torch::Tensor,  // agent_types
        torch::Tensor,  // positions
        torch::Tensor,  // padding_mask
        torch::Tensor,  // policy_indices
        torch::Tensor   // valid_lengths
    >
    prepare_batch_tensors(
        const std::vector<std::tuple<int, int, const PolicyRequest*>>& requests,
        int64_t target_seq_len
    ) const;

    uint8_t sample_action(
        const torch::Tensor& logits,
        const std::array<uint8_t, 7>& mask,
        float& log_prob
    ) const;

    void ensure_workspace_capacity(
        int64_t batch_size,
        const torch::Tensor& valid_lengths);
};

}  // namespace execution_core
