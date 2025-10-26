#include "execution_core.h"
#include "reactive_model_forward.h"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <torch/torch.h>

namespace execution_core {

NeuralInferenceOrchestrator::NeuralInferenceOrchestrator(
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const std::unordered_map<int, int>& policy_id_to_index,
    int64_t max_inference_batch_size,
    int64_t num_layers,
    int64_t num_heads,
    int64_t hidden_dim,
    int64_t num_experts,
    int64_t top_k
)
    : batched_weights_(batched_weights),
      policy_id_to_index_(policy_id_to_index),
      max_inference_batch_size_(max_inference_batch_size),
      num_layers_(num_layers),
      num_heads_(num_heads),
      hidden_dim_(hidden_dim),
      num_experts_(num_experts),
      top_k_(top_k)
{
}

int64_t NeuralInferenceOrchestrator::find_max_sequence_length(
    const std::unordered_map<int, std::vector<PolicyRequest>>& requests_by_policy
) const {
    int64_t max_len = 0;
    for (const auto& kv : requests_by_policy) {
        for (const auto& req : kv.second) {
            max_len = std::max(max_len, static_cast<int64_t>(req.valid_len));
        }
    }
    return max_len;
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
>
NeuralInferenceOrchestrator::prepare_batch_tensors(
    const std::vector<std::tuple<int, int, const PolicyRequest*>>& requests,
    int64_t target_seq_len
) const {
    const int64_t batch_size = static_cast<int64_t>(requests.size());
    const int64_t obs_dim = 16;  // Padded observation dimension

    // Allocate output tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
    auto obs_sequence = torch::zeros({batch_size, target_seq_len, obs_dim}, options);
    auto action_sequence = torch::zeros({batch_size, target_seq_len}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    auto agent_types = torch::zeros({batch_size, target_seq_len}, options);
    auto positions = torch::zeros({batch_size, target_seq_len}, options);
    auto padding_mask = torch::ones({batch_size, target_seq_len}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    auto policy_indices = torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));

    // Fill tensors from requests
    for (int64_t i = 0; i < batch_size; ++i) {
        const auto& [policy_id, request_idx, req_ptr] = requests[i];
        const auto& req = *req_ptr;

        // Get policy index from mapping
        auto it = policy_id_to_index_.find(policy_id);
        if (it == policy_id_to_index_.end()) {
            throw std::runtime_error("Policy ID " + std::to_string(policy_id) + " not found in policy_id_to_index mapping");
        }
        policy_indices[i] = it->second;

        const int64_t seq_len = req.valid_len;

        // Copy observation sequence from std::vector<float>
        if (!req.obs_sequence.empty() && seq_len > 0) {
            // obs_sequence is [valid_len * obs_dim] flattened
            auto obs_tensor = torch::from_blob(
                const_cast<float*>(req.obs_sequence.data()),
                {seq_len, obs_dim},
                torch::TensorOptions().dtype(torch::kFloat32)
            ).to(torch::kFloat16).to(torch::kCUDA);
            obs_sequence[i].narrow(0, 0, seq_len) = obs_tensor;
        }

        // Copy action sequence from std::vector<int64_t> (no factorization - forward_packed handles that)
        if (!req.action_sequence.empty() && seq_len > 0) {
            auto action_tensor = torch::from_blob(
                const_cast<int64_t*>(req.action_sequence.data()),
                {seq_len},
                torch::TensorOptions().dtype(torch::kLong)
            ).to(torch::kCUDA);

            action_sequence[i].narrow(0, 0, seq_len).copy_(action_tensor);
        }

        // Copy agent types
        if (!req.agent_type_sequence.empty() && seq_len > 0) {
            auto agent_type_tensor = torch::from_blob(
                const_cast<int64_t*>(req.agent_type_sequence.data()),
                {seq_len},
                torch::TensorOptions().dtype(torch::kLong)
            ).to(torch::kFloat16).to(torch::kCUDA);
            agent_types[i].narrow(0, 0, seq_len) = agent_type_tensor;
        }

        // Copy positions
        if (!req.position_sequence.empty() && seq_len > 0) {
            auto position_tensor = torch::from_blob(
                const_cast<int64_t*>(req.position_sequence.data()),
                {seq_len},
                torch::TensorOptions().dtype(torch::kLong)
            ).to(torch::kFloat16).to(torch::kCUDA);
            positions[i].narrow(0, 0, seq_len) = position_tensor;
        }

        // Set padding mask (False for valid positions, True for padding)
        if (seq_len < target_seq_len) {
            padding_mask[i].narrow(0, 0, seq_len).fill_(false);
            // Positions [seq_len, target_seq_len) remain True (padding)
        } else {
            padding_mask[i].fill_(false);
        }
    }

    return std::make_tuple(
        obs_sequence,
        action_sequence,
        agent_types,
        positions,
        padding_mask,
        policy_indices
    );
}

uint8_t NeuralInferenceOrchestrator::sample_action(
    const torch::Tensor& logits,
    const std::array<uint8_t, 7>& mask,
    float& log_prob
) const {
    // logits: [7]
    auto logits_cpu = logits.to(torch::kCPU).to(torch::kFloat32);

    // Apply mask by setting invalid actions to -inf
    std::vector<float> masked_logits(7);
    for (int i = 0; i < 7; ++i) {
        if (mask[i]) {
            masked_logits[i] = logits_cpu[i].item<float>();
        } else {
            masked_logits[i] = -std::numeric_limits<float>::infinity();
        }
    }

    // Softmax with numerical stability
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < 7; ++i) {
        if (mask[i]) {
            max_logit = std::max(max_logit, masked_logits[i]);
        }
    }

    std::vector<float> probs(7);
    float sum_exp = 0.0f;
    for (int i = 0; i < 7; ++i) {
        if (mask[i]) {
            probs[i] = std::exp(masked_logits[i] - max_logit);
            sum_exp += probs[i];
        } else {
            probs[i] = 0.0f;
        }
    }

    // Normalize
    for (int i = 0; i < 7; ++i) {
        probs[i] /= sum_exp;
    }

    // Sample using categorical distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int action = dist(gen);

    // Compute log probability
    log_prob = std::log(probs[action] + 1e-8f);

    return static_cast<uint8_t>(action);
}

std::unordered_map<std::pair<int, int>, InferenceResult, pair_hash>
NeuralInferenceOrchestrator::run_inference(
    const std::unordered_map<int, std::vector<PolicyRequest>>& requests_by_policy
) {
    // Result map
    std::unordered_map<std::pair<int, int>, InferenceResult, pair_hash> results;

    // Early exit if no requests
    if (requests_by_policy.empty()) {
        return results;
    }

    // Find maximum sequence length across all requests
    int64_t max_seq_len = find_max_sequence_length(requests_by_policy);
    if (max_seq_len <= 0) {
        throw std::runtime_error("Invalid maximum sequence length: " + std::to_string(max_seq_len));
    }

    // Flatten all requests into a single vector with (policy_id, request_index, request_ptr)
    std::vector<std::tuple<int, int, const PolicyRequest*>> all_requests;
    for (const auto& kv : requests_by_policy) {
        int policy_id = kv.first;
        const auto& requests = kv.second;
        for (size_t i = 0; i < requests.size(); ++i) {
            all_requests.emplace_back(policy_id, static_cast<int>(i), &requests[i]);
        }
    }

    // Process requests in batches
    const int64_t total_requests = static_cast<int64_t>(all_requests.size());
    for (int64_t batch_start = 0; batch_start < total_requests; batch_start += max_inference_batch_size_) {
        const int64_t batch_end = std::min(batch_start + max_inference_batch_size_, total_requests);
        const int64_t actual_batch_size = batch_end - batch_start;

        // Extract batch slice
        std::vector<std::tuple<int, int, const PolicyRequest*>> batch_requests(
            all_requests.begin() + batch_start,
            all_requests.begin() + batch_end
        );

        // Prepare tensors for this batch
        auto [obs_seq, action_seq, agent_types, positions, padding_mask, policy_indices] =
            prepare_batch_tensors(batch_requests, max_seq_len);

        // Run forward_packed
        auto [action_logits, opp_logits, state_values, win_logits] = forward_packed(
            obs_seq,
            action_seq,
            agent_types,
            positions,
            batched_weights_,
            policy_indices,
            padding_mask,
            num_layers_,
            num_heads_,
            hidden_dim_,
            num_experts_,
            top_k_
        );

        // Process results for each request in the batch
        for (int64_t i = 0; i < actual_batch_size; ++i) {
            const auto& [policy_id, request_idx, req_ptr] = batch_requests[i];
            const auto& req = *req_ptr;

            // Extract logits for the last timestep
            const int64_t last_pos = req.valid_len - 1;
            auto logits_at_last_pos = action_logits[i][last_pos];  // [7]

            // Sample action
            float log_prob;
            uint8_t action = sample_action(logits_at_last_pos, req.mask, log_prob);

            // Extract value
            float value = state_values[i][last_pos].item<float>();

            // Extract opponent logits
            auto opp_logits_at_last_pos = opp_logits[i][last_pos];  // [7]

            // Store result
            InferenceResult result;
            result.action = action;
            result.log_prob = log_prob;
            result.state_value = value;
            result.opp_logits = opp_logits_at_last_pos.to(torch::kCPU);

            results[{policy_id, request_idx}] = std::move(result);
        }
    }

    return results;
}

}  // namespace execution_core
