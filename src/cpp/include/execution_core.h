#pragma once

#include <torch/torch.h>
#include <unordered_map>
#include <vector>
#include <cstdint>

#include "vec_arena.h"  // For PolicyRequest

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
    /**
     * Constructor.
     *
     * @param batched_weights Model weights in batched format [W, ...]
     * @param policy_id_to_index Mapping from policy_id to weight cache index
     * @param max_inference_batch_size Maximum batch size for inference
     * @param num_layers Number of transformer layers
     * @param num_heads Number of attention heads
     * @param hidden_dim Hidden dimension size
     * @param num_experts Number of MoE experts
     * @param top_k Number of experts to use per token
     */
    NeuralInferenceOrchestrator(
        const c10::Dict<std::string, torch::Tensor>& batched_weights,
        const std::unordered_map<int, int>& policy_id_to_index,
        int64_t max_inference_batch_size = 512,
        int64_t num_layers = 2,
        int64_t num_heads = 4,
        int64_t hidden_dim = 256,
        int64_t num_experts = 8,
        int64_t top_k = 2
    );

    /**
     * Run batched inference for all neural network requests.
     *
     * This is the core method that implements the "Maximal Batching Efficiency" strategy:
     * - Finds the true maximum sequence length across all requests
     * - Pads all tensors to this exact length (no bucketing/rounding)
     * - Splits requests into batches of max_inference_batch_size
     * - Executes forward_packed for each batch
     * - Returns results keyed by (policy_id, request_index)
     *
     * @param requests_by_policy Map of policy_id -> vector of PolicyRequest
     * @return Map of (policy_id, request_index) -> InferenceResult
     */
    std::unordered_map<std::pair<int, int>, InferenceResult, pair_hash>
    run_inference(
        const std::unordered_map<int, std::vector<PolicyRequest>>& requests_by_policy
    );

private:
    // Model weights and configuration
    c10::Dict<std::string, torch::Tensor> batched_weights_;
    std::unordered_map<int, int> policy_id_to_index_;

    // Inference configuration
    int64_t max_inference_batch_size_;
    int64_t num_layers_;
    int64_t num_heads_;
    int64_t hidden_dim_;
    int64_t num_experts_;
    int64_t top_k_;

    /**
     * Find the maximum sequence length across all requests.
     */
    int64_t find_max_sequence_length(
        const std::unordered_map<int, std::vector<PolicyRequest>>& requests_by_policy
    ) const;

    /**
     * Prepare batched tensors for a subset of requests.
     * All tensors will be padded to target_seq_len.
     *
     * @param requests Flat list of (policy_id, request_index, request_ptr)
     * @param target_seq_len Sequence length to pad to
     * @return Tuple of (obs_seq, action_seq, agent_types, positions, padding_mask, policy_indices)
     */
    std::tuple<
        torch::Tensor,  // obs_sequence [B, T, obs_dim]
        torch::Tensor,  // action_sequence [B, T, 3]
        torch::Tensor,  // agent_types [B, T]
        torch::Tensor,  // positions [B, T]
        torch::Tensor,  // padding_mask [B, T]
        torch::Tensor   // policy_indices [B]
    >
    prepare_batch_tensors(
        const std::vector<std::tuple<int, int, const PolicyRequest*>>& requests,
        int64_t target_seq_len
    ) const;

    /**
     * Sample action from logits using the action mask.
     *
     * @param logits Action logits [7]
     * @param mask Valid action mask [7]
     * @param log_prob Output log probability
     * @return Sampled action index
     */
    uint8_t sample_action(
        const torch::Tensor& logits,
        const std::array<uint8_t, 7>& mask,
        float& log_prob
    ) const;
};

}  // namespace execution_core
