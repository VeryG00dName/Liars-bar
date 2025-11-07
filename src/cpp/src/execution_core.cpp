#include "execution_core.h"
#include "model_forward.h"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <torch/torch.h>
#include <cuda_runtime.h>

namespace execution_core {

NeuralInferenceOrchestrator::NeuralInferenceOrchestrator(
    const c10::Dict<std::string, torch::Tensor>& batched_weights,
    const std::unordered_map<int, int>& policy_id_to_index,
    int64_t max_inference_batch_size,
    int64_t num_layers,
    int64_t num_heads,
    int64_t hidden_dim,
    int64_t num_experts,
    int64_t top_k,
    bool use_argmax
)
    : batched_weights_(batched_weights),
      policy_id_to_index_(policy_id_to_index),
      max_inference_batch_size_(max_inference_batch_size),
      num_layers_(num_layers),
      num_heads_(num_heads),
      hidden_dim_(hidden_dim),
      num_experts_(num_experts),
      top_k_(top_k),
      use_argmax_(use_argmax)
{
    // Lazily grow MoE workspaces based on observed batch characteristics.
    // For dense (non-MoE) models, these workspaces remain empty and unused.
    moe_workspaces_.resize(num_layers_);
}

NeuralInferenceOrchestrator::~NeuralInferenceOrchestrator() {
    // Clean up pre-allocated workspaces
    for (auto& ws : moe_workspaces_) {
        if (ws.hidden_buffer) cudaFree(ws.hidden_buffer);
        if (ws.workspace_w1) cudaFree(ws.workspace_w1);
        if (ws.workspace_w2) cudaFree(ws.workspace_w2);

        if (ws.problem_sizes_device_w1) cudaFree(ws.problem_sizes_device_w1);
        if (ws.ptr_A_device_w1) cudaFree(ws.ptr_A_device_w1);
        if (ws.ptr_B_device_w1) cudaFree(ws.ptr_B_device_w1);
        if (ws.ptr_C_device_w1) cudaFree(ws.ptr_C_device_w1);
        if (ws.ptr_D_device_w1) cudaFree(ws.ptr_D_device_w1);
        if (ws.lda_device_w1) cudaFree(ws.lda_device_w1);
        if (ws.ldb_device_w1) cudaFree(ws.ldb_device_w1);
        if (ws.ldc_device_w1) cudaFree(ws.ldc_device_w1);
        if (ws.ldd_device_w1) cudaFree(ws.ldd_device_w1);

        if (ws.problem_sizes_device_w2) cudaFree(ws.problem_sizes_device_w2);
        if (ws.ptr_A_device_w2) cudaFree(ws.ptr_A_device_w2);
        if (ws.ptr_B_device_w2) cudaFree(ws.ptr_B_device_w2);
        if (ws.ptr_C_device_w2) cudaFree(ws.ptr_C_device_w2);
        if (ws.ptr_D_device_w2) cudaFree(ws.ptr_D_device_w2);
        if (ws.lda_device_w2) cudaFree(ws.lda_device_w2);
        if (ws.ldb_device_w2) cudaFree(ws.ldb_device_w2);
        if (ws.ldc_device_w2) cudaFree(ws.ldc_device_w2);
        if (ws.ldd_device_w2) cudaFree(ws.ldd_device_w2);
    }
}

void NeuralInferenceOrchestrator::ensure_workspace_capacity(
    int64_t batch_size,
    const torch::Tensor& valid_lengths) {
    if (batch_size <= 0 || moe_workspaces_.empty()) {
        return;
    }

    int64_t max_valid = 0;
    if (valid_lengths.defined() && valid_lengths.numel() > 0) {
        max_valid = valid_lengths.max().item<int64_t>();
    }
    max_valid = std::max<int64_t>(1, max_valid);

    const int64_t token_capacity = batch_size * max_valid;
    for (auto& ws : moe_workspaces_) {
        lb::forward::ensure_forward_workspace_capacity(ws, token_capacity, hidden_dim_, top_k_);
    }
}

int64_t NeuralInferenceOrchestrator::find_max_sequence_length(
    const std::unordered_map<int, std::vector<PolicyRequest>>& requests_by_policy
) const {
    int64_t max_len = 1; // Start with 1 to handle empty sequences
    for (const auto& kv : requests_by_policy) {
        for (const auto& req : kv.second) {
            max_len = std::max(max_len, static_cast<int64_t>(req.valid_len));
        }
    }
    return max_len;
}

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor
>
NeuralInferenceOrchestrator::prepare_batch_tensors(
    const std::vector<std::tuple<int, int, const PolicyRequest*>>& requests,
    int64_t target_seq_len
) const {
    const int64_t batch_size = static_cast<int64_t>(requests.size());
    const int64_t obs_dim = 16;
    const bool pin_memory = true;

    // Allocate tensors on pinned CPU memory for fast async transfer
    auto opts_float_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(pin_memory);
    auto opts_long_cpu = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU).pinned_memory(pin_memory);
    auto opts_bool_cpu = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU).pinned_memory(pin_memory);

    auto obs_sequence_cpu = torch::zeros({batch_size, target_seq_len, obs_dim}, opts_float_cpu);
    auto action_sequence_cpu = torch::zeros({batch_size, target_seq_len}, opts_long_cpu);
    auto agent_types_cpu = torch::zeros({batch_size, target_seq_len}, opts_long_cpu);
    auto positions_cpu = torch::zeros({batch_size, target_seq_len}, opts_long_cpu);
    auto padding_mask_cpu = torch::ones({batch_size, target_seq_len}, opts_bool_cpu);
    auto policy_indices_cpu = torch::empty({batch_size}, opts_long_cpu);
    auto valid_lengths_cpu = torch::empty({batch_size}, opts_long_cpu);

    // Fill pinned CPU tensors
    for (int64_t i = 0; i < batch_size; ++i) {
        const auto& [policy_id, request_idx, req_ptr] = requests[i];
        const auto& req = *req_ptr;

        auto it = policy_id_to_index_.find(policy_id);
        if (it == policy_id_to_index_.end()) {
            throw std::runtime_error("Policy ID " + std::to_string(policy_id) + " not found");
        }
        policy_indices_cpu[i] = it->second;

        const int64_t seq_len = std::max<int64_t>(1, req.valid_len);
        valid_lengths_cpu[i] = seq_len;
        
        padding_mask_cpu[i].slice(0, 0, seq_len).fill_(false);

        if (req.valid_len > 0) {
            if (!req.obs_sequence.empty())
                memcpy(obs_sequence_cpu[i].data_ptr<float>(), req.obs_sequence.data(), seq_len * obs_dim * sizeof(float));
            if (!req.action_sequence.empty())
                memcpy(action_sequence_cpu[i].data_ptr<int64_t>(), req.action_sequence.data(), seq_len * sizeof(int64_t));
            if (!req.agent_type_sequence.empty())
                memcpy(agent_types_cpu[i].data_ptr<int64_t>(), req.agent_type_sequence.data(), seq_len * sizeof(int64_t));
            if (!req.position_sequence.empty())
                memcpy(positions_cpu[i].data_ptr<int64_t>(), req.position_sequence.data(), seq_len * sizeof(int64_t));
        }
    }

    // Async transfer to GPU and convert to FP16
    const auto device = torch::kCUDA;
    auto obs_sequence = obs_sequence_cpu.to(device, torch::kFloat16, true, true);
    auto action_sequence = action_sequence_cpu.to(device, true, true);
    auto agent_types = agent_types_cpu.to(device, true, true);
    auto positions = positions_cpu.to(device, true, true);
    auto padding_mask = padding_mask_cpu.to(device, true, true);
    auto policy_indices = policy_indices_cpu.to(device, true, true);
    auto valid_lengths = valid_lengths_cpu.to(device, true, true);

    return std::make_tuple(
        obs_sequence,
        action_sequence,
        agent_types,
        positions,
        padding_mask,
        policy_indices,
        valid_lengths
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

    int action = 0;
    if (use_argmax_) {
        // Greedy selection by argmax over masked logits
        float best = -std::numeric_limits<float>::infinity();
        int best_idx = 0;
        for (int i = 0; i < 7; ++i) {
            if (mask[i] && masked_logits[i] > best) {
                best = masked_logits[i];
                best_idx = i;
            }
        }
        action = best_idx;
    } else {
        // Sample using categorical distribution
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        action = dist(gen);
    }

    // Compute log probability of chosen action
    log_prob = std::log(probs[action] + 1e-8f);

    return static_cast<uint8_t>(action);
}

std::unordered_map<std::pair<int, int>, InferenceResult, pair_hash>
NeuralInferenceOrchestrator::run_inference(
    const std::unordered_map<int, std::vector<PolicyRequest>>& requests_by_policy
) {
    std::unordered_map<std::pair<int, int>, InferenceResult, pair_hash> results;
    if (requests_by_policy.empty()) return results;

    int64_t max_seq_len = find_max_sequence_length(requests_by_policy);

    std::vector<std::tuple<int, int, const PolicyRequest*>> all_requests;
    for (const auto& kv : requests_by_policy) {
        for (size_t i = 0; i < kv.second.size(); ++i) {
            all_requests.emplace_back(kv.first, static_cast<int>(i), &kv.second[i]);
        }
    }

    const int64_t total_requests = static_cast<int64_t>(all_requests.size());
    for (int64_t batch_start = 0; batch_start < total_requests; batch_start += max_inference_batch_size_) {
        const int64_t batch_end = std::min(batch_start + max_inference_batch_size_, total_requests);
        const int64_t actual_batch_size = batch_end - batch_start;

        std::vector<std::tuple<int, int, const PolicyRequest*>> batch_requests(
            all_requests.begin() + batch_start, all_requests.begin() + batch_end);

        auto [obs_seq, action_seq, agent_types, positions, padding_mask, policy_indices, valid_lengths] =
            prepare_batch_tensors(batch_requests, max_seq_len);

        ensure_workspace_capacity(obs_seq.size(0), valid_lengths);

        std::unordered_map<std::string, std::chrono::microseconds> batch_timers;
        auto [action_logits, opp_logits, state_values, win_logits] = lb::forward::forward_packed(
            obs_seq, action_seq, agent_types, positions,
            batched_weights_, policy_indices, padding_mask,
            num_layers_, num_heads_, hidden_dim_, num_experts_, top_k_,
            4, 3, &batch_timers, moe_workspaces_.data());

        for (const auto& kv : batch_timers) {
            timing_stats_[kv.first] += kv.second;
        }

        auto last_indices = (valid_lengths - 1).clamp_min(0);
        auto batch_indices = torch::arange(actual_batch_size, policy_indices.options());
        auto last_action_logits = action_logits.index({batch_indices, last_indices});
        auto last_state_values = state_values.index({batch_indices, last_indices});
        auto last_opp_logits = opp_logits.index({batch_indices, last_indices});

        for (int64_t i = 0; i < actual_batch_size; ++i) {
            const auto& [policy_id, request_idx, req_ptr] = batch_requests[i];
            
            float log_prob;
            uint8_t action = sample_action(last_action_logits[i], req_ptr->mask, log_prob);
            float value = last_state_values[i].item<float>();

            results[{policy_id, request_idx}] = {action, log_prob, value, last_opp_logits[i].to(torch::kCPU)};
        }
    }
    return results;
}

}  // namespace execution_core