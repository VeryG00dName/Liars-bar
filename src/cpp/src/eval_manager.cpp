#include "eval_manager.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <ATen/ATen.h>
#include <ATen/DeviceAccelerator.h>
#include <torch/torch.h>

#include "bots.h"
#include "torch_utils.h"
#include "weight_utils.h"

// Cache for parsed C++ bot kinds to avoid repeated string processing
std::unordered_map<std::string, EvalManager::CppBotKind> EvalManager::bot_kind_cache_;

namespace {

const torch::Device kInferenceDevice = torch::kCUDA;

std::filesystem::path metadata_path_for_model(const std::string& model_path) {
    std::filesystem::path path(model_path);
    path += ".max_seq_length";
    return path;
}

int read_max_seq_length_from_file(const std::filesystem::path& metadata_path) {
    if (metadata_path.empty()) {
        return -1;
    }
    std::ifstream stream(metadata_path);
    if (!stream.is_open()) {
        return -1;
    }
    int value = -1;
    stream >> value;
    if (!stream.good() || value <= 0) {
        return -1;
    }
    return value;
}

int fallback_max_seq_length_from_path(const std::string& path) {
    std::string lower;
    lower.reserve(path.size());
    for (char c : path) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    size_t pos = lower.find("test");
    while (pos != std::string::npos) {
        size_t idx = pos + 4;
        if (idx < lower.size() && std::isdigit(static_cast<unsigned char>(lower[idx]))) {
            int value = 0;
            bool found_digit = false;
            while (idx < lower.size() && std::isdigit(static_cast<unsigned char>(lower[idx]))) {
                value = value * 10 + (lower[idx] - '0');
                ++idx;
                found_digit = true;
            }
            if (found_digit) {
                return (value <= 62) ? 256 : 480;
            }
        }
        pos = lower.find("test", pos + 4);
    }

    return 480;
}

int infer_max_seq_length_for_model(const std::string& model_path) {
    auto metadata_path = metadata_path_for_model(model_path);
    int from_file = read_max_seq_length_from_file(metadata_path);
    if (from_file > 0) {
        return from_file;
    }
    return fallback_max_seq_length_from_path(model_path);
}

std::mt19937::result_type seed_with_optional(uint32_t seed) {
    if (seed != 0) {
        return static_cast<std::mt19937::result_type>(seed);
    }
    std::random_device rd;
    return static_cast<std::mt19937::result_type>(rd());
}

class ClassicBot : public CppBotBase {
public:
    ClassicBot() : bot_("bot") {}
    uint8_t act(const PolicyRequest& request, VecArena&) override {
        return bot_.act(request.classic_obs.data(), request.classic_obs_len, request.mask.data());
    }

private:
    bots::Classic bot_;
};

class GreedyCardSpammerBot : public CppBotBase {
public:
    GreedyCardSpammerBot() : bot_("bot") {}
    uint8_t act(const PolicyRequest& request, VecArena&) override {
        return bot_.act(request.classic_obs.data(), request.classic_obs_len, request.mask.data());
    }

private:
    bots::GreedyCardSpammer bot_;
};

class RandomAgentBot : public CppBotBase {
public:
    RandomAgentBot() : bot_("bot") {}
    uint8_t act(const PolicyRequest& request, VecArena&) override {
        return bot_.act(request.classic_obs.data(), request.classic_obs_len, request.mask.data());
    }

private:
    bots::RandomAgent bot_;
};

class SelectiveTableConservativeChallengerBot : public CppBotBase {
public:
    SelectiveTableConservativeChallengerBot() : bot_("bot") {}
    uint8_t act(const PolicyRequest& request, VecArena&) override {
        return bot_.act(request.classic_obs.data(), request.classic_obs_len, request.mask.data());
    }

private:
    bots::SelectiveTableConservativeChallenger bot_;
};

class StrategicChallengerBot : public CppBotBase {
public:
    StrategicChallengerBot(int num_players, int seat)
        : bot_("bot", num_players, seat) {}

    uint8_t act(const PolicyRequest& request, VecArena&) override {
        return bot_.act(request.classic_obs.data(), request.classic_obs_len, request.mask.data());
    }

private:
    bots::StrategicChallenger bot_;
};

class TableFirstConservativeChallengerBot : public CppBotBase {
public:
    TableFirstConservativeChallengerBot() : bot_("bot") {}
    uint8_t act(const PolicyRequest& request, VecArena&) override {
        return bot_.act(request.classic_obs.data(), request.classic_obs_len, request.mask.data());
    }

private:
    bots::TableFirstConservativeChallenger bot_;
};

class TableNonTableAgentBot : public CppBotBase {
public:
    TableNonTableAgentBot() : bot_("bot") {}
    uint8_t act(const PolicyRequest& request, VecArena&) override {
        return bot_.act(request.classic_obs.data(), request.classic_obs_len, request.mask.data());
    }

private:
    bots::TableNonTableAgent bot_;
};

}  // namespace

EvalManager::EvalManager()
    : max_env_batch_(3584), max_inference_batch_size_(512), rng_(seed_with_optional(0)) {
    if (!torch::cuda::is_available()) {
        throw std::runtime_error(
            "CUDA is not available, but the EvalManager requires it for TorchScript inference.");
    }
}

void EvalManager::set_max_env_batch(int max_batch) {
    if (max_batch <= 0) {
        throw std::invalid_argument("max_env_batch must be positive");
    }
    max_env_batch_ = max_batch;
}

void EvalManager::set_max_inference_batch_size(size_t max_batch_size) {
    if (max_batch_size == 0) {
        throw std::invalid_argument("max_inference_batch_size must be positive");
    }
    max_inference_batch_size_ = max_batch_size;
}

void EvalManager::load_model(
    int policy_id,
    const std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::string& original_path) {
    try {
        torch::NoGradGuard guard;

        // LUT keys that should be skipped (will be added by add_fixed_buffers)
        const std::unordered_set<std::string> lut_keys = {
            "lut_act_kind", "lut_count", "lut_table_flag"
        };

        // Store CPU FP32 weights as-is (skip LUTs). We'll stack on CPU and
        // transfer once in finalize_model_loading() to avoid VRAM thrashing.
        std::unordered_map<std::string, torch::Tensor> processed_dict;
        processed_dict.reserve(state_dict.size());
        for (auto& kv : state_dict) {
            if (lut_keys.find(kv.first) != lut_keys.end()) {
                continue;
            }

            auto tensor = kv.second.detach();
            if (!tensor.device().is_cpu()) {
                tensor = tensor.to(torch::kCPU, /*non_blocking=*/false, /*copy=*/true);
            }
            if (tensor.dtype() != torch::kFloat32) {
                tensor = tensor.to(torch::kFloat32);
            }
            tensor = tensor.contiguous();
            processed_dict.emplace(kv.first, std::move(tensor));
        }

        staged_state_dicts_[policy_id] = std::move(processed_dict);

        // Infer max sequence length using the original path metadata if available
        int max_seq_length = infer_max_seq_length_for_model(original_path);
        policy_max_sequence_lengths_[policy_id] = max_seq_length;
        arena_.set_policy_max_sequence_length(policy_id, max_seq_length);
        weights_finalized_ = false;
    } catch (const c10::Error& err) {
        std::cerr << "[EvalManager] Failed to process state_dict for policy " << policy_id
                  << ": " << err.what_without_backtrace() << std::endl;
        throw;
    } catch (const std::exception& err) {
        std::cerr << "[EvalManager] Failed to process state_dict for policy " << policy_id
                  << ": " << err.what() << std::endl;
        throw;
    }
}

void EvalManager::finalize_model_loading() {
    torch::NoGradGuard guard;

    batched_weight_cache_.clear();
    policy_id_to_cache_index_.clear();

    if (staged_state_dicts_.empty()) {
        weights_finalized_ = true;
        return;
    }

    std::vector<int> policy_ids;
    policy_ids.reserve(staged_state_dicts_.size());
    for (const auto& kv : staged_state_dicts_) {
        policy_ids.push_back(kv.first);
    }
    std::sort(policy_ids.begin(), policy_ids.end());

    // Pre-stack MoE expert weights in each individual state_dict before batching (CPU)
    for (auto& kv : staged_state_dicts_) {
        prestack_moe_expert_weights(kv.second, num_layers_, num_experts_);
    }

    // Build mapping policy_id -> index used during inference
    for (size_t idx = 0; idx < policy_ids.size(); ++idx) {
        policy_id_to_cache_index_[policy_ids[idx]] = static_cast<int>(idx);
    }

    // Determine keys to process from the first policy's dict
    const auto& first_dict = staged_state_dicts_.at(policy_ids[0]);
    std::vector<std::string> keys;
    keys.reserve(first_dict.size());
    for (const auto& kv : first_dict) {
        keys.push_back(kv.first);
    }

    // For each key, gather CPU tensors across policies, stack on CPU, then
    // transfer the final stacked tensor to GPU as FP16 and cache it.
    for (const auto& key : keys) {
        std::vector<torch::Tensor> cpu_tensors;
        cpu_tensors.reserve(policy_ids.size());
        for (int policy_id : policy_ids) {
            auto dict_it = staged_state_dicts_.find(policy_id);
            if (dict_it == staged_state_dicts_.end()) {
                throw std::runtime_error("Missing staged weights for policy " + std::to_string(policy_id));
            }
            auto& per_policy = dict_it->second;
            auto itw = per_policy.find(key);
            if (itw == per_policy.end()) {
                throw std::runtime_error("Missing key '" + key + "' in staged weights for policy " + std::to_string(policy_id));
            }
            cpu_tensors.push_back(itw->second);
        }

        auto stacked_cpu = torch::stack(cpu_tensors, /*dim=*/0).contiguous();
        auto stacked_gpu = stacked_cpu
                               .to(kInferenceDevice, /*non_blocking=*/false, /*copy=*/true)
                               .to(torch::kFloat16)
                               .contiguous();

        batched_weight_cache_.insert(key, stacked_gpu);

        if (stacked_gpu.device().is_cuda()) {
            auto append_ptrs = [&](const std::string& suffix) {
                if (key.size() < suffix.size() || key.rfind(suffix) != key.size() - suffix.size()) {
                    return;
                }

                TORCH_CHECK(
                    stacked_gpu.dim() >= 2,
                    "Expected stacked MoE tensor to have at least 2 dimensions"
                );
                TORCH_CHECK(
                    stacked_gpu.dtype() == torch::kFloat16,
                    "MoE tensors must be FP16 before pointer caching: " + key
                );

                const int64_t num_policies = stacked_gpu.size(0);
                const int64_t num_experts = stacked_gpu.size(1);

                // Create pointer tensor on CPU first
                auto ptr_tensor_cpu = torch::empty(
                    {num_policies, num_experts},
                    torch::dtype(torch::kUInt64).device(torch::kCPU)
                );

                const uintptr_t base_address = reinterpret_cast<uintptr_t>(stacked_gpu.data_ptr<at::Half>());
                const int64_t stride_policy = stacked_gpu.stride(0);
                const int64_t stride_expert = stacked_gpu.stride(1);
                const int64_t element_size = static_cast<int64_t>(stacked_gpu.element_size());

                auto ptr_data = ptr_tensor_cpu.data_ptr<uint64_t>();

                // Use PyTorch indexing (same as weight_utils.cpp) to get data pointers
                // The stacked tensor is contiguous, so slices [p][e] should point into it correctly
                for (int64_t p = 0; p < num_policies; ++p) {
                    for (int64_t e = 0; e < num_experts; ++e) {
                        // Get the expert tensor slice
                        auto expert_tensor = stacked_gpu[p][e];

                        ptr_data[p * num_experts + e] = reinterpret_cast<uint64_t>(
                            expert_tensor.data_ptr<at::Half>()
                        );
                    }
                }

                // Move pointer tensor to same device as weights (CUDA)
                auto ptr_tensor_gpu = ptr_tensor_cpu.to(stacked_gpu.device());
                batched_weight_cache_.insert_or_assign(key + "_ptrs", ptr_tensor_gpu);
            };

            append_ptrs(".moe.experts.w1");
            append_ptrs(".moe.experts.b1");
            append_ptrs(".moe.experts.w2");
            append_ptrs(".moe.experts.b2");
        }

        // Proactively free CPU per-policy tensors for this key to reduce RAM
        for (int policy_id : policy_ids) {
            auto dit = staged_state_dicts_.find(policy_id);
            if (dit != staged_state_dicts_.end()) {
                dit->second.erase(key);
            }
        }
    }

    // Clear temporary storage to free memory
    staged_state_dicts_.clear();  // CRITICAL: Free the per-policy tensors after stacking

    // Add fixed buffers (LUTs) - these are constant and shared across all policies
    add_fixed_buffers(batched_weight_cache_, kInferenceDevice);

    // Verify ALL 3 LUTs were added correctly
    const std::vector<std::string> lut_names = {"lut_act_kind", "lut_count", "lut_table_flag"};
    for (const auto& lut_name : lut_names) {
        if (batched_weight_cache_.contains(lut_name)) {
            auto lut = batched_weight_cache_.at(lut_name);
            fprintf(stderr, "[EvalManager] LUT verification: %s shape=[%ld], dtype=%s, device=%s\n",
                    lut_name.c_str(), lut.numel(),
                    torch::toString(lut.dtype()).c_str(), lut.device().str().c_str());
            if (lut.numel() == 11 && lut.device().is_cuda() && lut.dtype() == torch::kLong) {
                auto cpu_lut = lut.to(torch::kCPU);
                auto ptr = cpu_lut.data_ptr<int64_t>();
                fprintf(stderr, "[EvalManager]   Values: [%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld]\n",
                        ptr[0],ptr[1],ptr[2],ptr[3],ptr[4],ptr[5],ptr[6],ptr[7],ptr[8],ptr[9],ptr[10]);
            } else {
                fprintf(stderr, "[EvalManager] ERROR: %s has wrong size (%ld), device (%s), or dtype!\n",
                        lut_name.c_str(), lut.numel(), lut.device().str().c_str());
            }
        } else {
            fprintf(stderr, "[EvalManager] ERROR: %s not found in batched_weight_cache!\n", lut_name.c_str());
        }
    }

    // Unify attention weight processing and aliasing
    process_and_split_attention_weights(batched_weight_cache_);

    // Create weight pointers for MoE experts (must be after GPU transfer)
    create_moe_weight_pointers(batched_weight_cache_, num_layers_, num_experts_);

    // Create orchestrator for unified neural inference
    orchestrator_ = std::make_unique<execution_core::NeuralInferenceOrchestrator>(
        batched_weight_cache_,
        policy_id_to_cache_index_,
        max_inference_batch_size_,
        num_layers_,
        num_heads_,
        hidden_dim_,
        num_experts_,
        top_k_,
        /*use_argmax=*/false
    );

    weights_finalized_ = true;
}

void EvalManager::register_cpp_bot(int policy_id, const std::string& bot_name) {
    CppBotKind kind = parse_cpp_bot_kind(bot_name);
    auto& entry = cpp_bot_registry_[policy_id];
    entry.kind = kind;
    entry.instances.clear();
}

EvalOutcome EvalManager::run_roles(const std::vector<std::vector<int>>& roles,
                                   const std::vector<int>& lineup_indices,
                                   int num_players,
                                   uint32_t seed) {
    using Clock = std::chrono::high_resolution_clock;
    using Microseconds = std::chrono::microseconds;

    timer_total_run_roles_ = Microseconds::zero();
    timer_arena_stepping_ = Microseconds::zero();
    timer_collect_requests_ = Microseconds::zero();
    timer_model_inference_ = Microseconds::zero();
    timer_cpp_bots_ = Microseconds::zero();
    timer_hist_prep_batch_ = Microseconds::zero();
    timer_hist_prep_weights_ = Microseconds::zero();
    timer_hist_model_exec_ = Microseconds::zero();
    timer_hist_post_ = Microseconds::zero();

    // Reset detailed timers for this run
    detailed_timers_.clear();

    auto total_start = Clock::now();

    EvalOutcome outcome;

    if (roles.empty()) {
        return outcome;
    }

    if (roles.size() != lineup_indices.size()) {
        throw std::invalid_argument("roles and lineup_indices must be the same length");
    }

    if (num_players <= 0) {
        throw std::invalid_argument("num_players must be positive");
    }

    const size_t total_games = roles.size();
    for (const auto& seats : roles) {
        if (static_cast<int>(seats.size()) != num_players) {
            throw std::invalid_argument("Each role assignment must contain num_players entries");
        }
    }

    int max_lineup_index = -1;
    for (int idx : lineup_indices) {
        if (idx < 0) {
            throw std::invalid_argument("lineup_indices must be non-negative");
        }
        max_lineup_index = std::max(max_lineup_index, idx);
    }

    outcome.total_games = static_cast<int>(total_games);
    if (max_lineup_index >= 0) {
        outcome.lineups.resize(static_cast<size_t>(max_lineup_index) + 1);
    }

    uint32_t base_seed = static_cast<uint32_t>(seed_with_optional(seed));
    arena_.reset(static_cast<int>(total_games), num_players, base_seed);
    arena_.set_roles(roles);

    std::vector<uint8_t> env_completed(static_cast<size_t>(arena_.B), 0);
    int completed_games = 0;

    while (completed_games < static_cast<int>(total_games)) {
        auto cpp_start = Clock::now();
        // Advance pure C++ bot turns greedily before requesting model inference.
        for (int env_idx = 0; env_idx < arena_.B; ++env_idx) {
            if (arena_.done[env_idx]) {
                continue;
            }

            Env& env = arena_.envs[env_idx];

            while (!arena_.done[env_idx]) {
                int current_seat = env.current_player();
                int policy_id = arena_.roles[env_idx][current_seat].policy_id;

                auto bot_it = cpp_bot_registry_.find(policy_id);
                if (bot_it == cpp_bot_registry_.end()) {
                    break;  // Needs neural inference.
                }

                PolicyRequest request;
                request.env = env_idx;
                request.seat = current_seat;
                request.done = 0;
                env.valid_actions(request.mask.data());
                request.classic_obs_len = env.observe_vector(request.classic_obs.data());

                uint64_t instance_key =
                    (static_cast<uint64_t>(env_idx) << 32) ^ static_cast<uint32_t>(current_seat & 0xFFFFFFFF);
                auto& entry = bot_it->second;
                auto inst_it = entry.instances.find(instance_key);
                if (inst_it == entry.instances.end()) {
                    auto instance = make_cpp_bot_instance(entry.kind, request);
                    inst_it = entry.instances.emplace(instance_key, std::move(instance)).first;
                }

                uint8_t action = inst_it->second->act(request, arena_);
                bool over = env.step(action);
                if (over) {
                    arena_.done[env_idx] = 1;
                    if (!env_completed[static_cast<size_t>(env_idx)]) {
                        env_completed[static_cast<size_t>(env_idx)] = 1;
                        ++completed_games;
                    }
                    break;
                }
            }
        }
        auto cpp_end = Clock::now();
        timer_cpp_bots_ += std::chrono::duration_cast<Microseconds>(cpp_end - cpp_start);

        auto collect_start = Clock::now();
        auto pending = arena_.collect_requests();
        auto collect_end = Clock::now();
        timer_collect_requests_ += std::chrono::duration_cast<Microseconds>(collect_end - collect_start);
        if (pending.empty()) {
            if (completed_games == static_cast<int>(total_games)) {
                break;
            }
            throw std::runtime_error("Simulation deadlock: no pending AI requests but games remain active");
        }

        std::unordered_map<int, std::vector<uint8_t>> actions_by_policy;
        actions_by_policy.reserve(pending.size());

        std::unordered_map<int, std::vector<PolicyRequest>> neural_requests;

        for (const auto& kv : pending) {
            int policy_id = kv.first;
            const auto& requests = kv.second;

            // Check if it's a C++ bot
            if (cpp_bot_registry_.count(policy_id)) {
                actions_by_policy[policy_id] = run_cpp_bot(policy_id, requests);
                continue;
            }

            // If not a C++ bot, it must be a neural model
            if (policy_id_to_cache_index_.find(policy_id) == policy_id_to_cache_index_.end()) {
                throw std::runtime_error("No registered model for policy " + std::to_string(policy_id));
            }

            neural_requests[policy_id] = requests;
        }

        auto model_start = Clock::now();
        if (!neural_requests.empty()) {
            if (!weights_finalized_) {
                finalize_model_loading();
            }
            run_neural_inference(neural_requests, actions_by_policy);
        }
        auto model_end = Clock::now();
        timer_model_inference_ += std::chrono::duration_cast<Microseconds>(model_end - model_start);

        auto submit_start = Clock::now();
        for (const auto& kv : pending) {
            int policy_id = kv.first;
            auto it = actions_by_policy.find(policy_id);
            if (it == actions_by_policy.end()) {
                 throw std::runtime_error("Logic error: actions not found for pending policy " + std::to_string(policy_id));
            }
            arena_.submit_actions(policy_id, it->second);
            
            const auto& requests = kv.second;
            for(const auto& req : requests) {
                int env_idx = req.env;
                if (env_idx >= 0 && env_idx < arena_.B) {
                    if (arena_.done[env_idx] && !env_completed[static_cast<size_t>(env_idx)]) {
                        env_completed[static_cast<size_t>(env_idx)] = 1;
                        ++completed_games;
                    }
                }
            }
        }
        auto submit_end = Clock::now();
        timer_arena_stepping_ += std::chrono::duration_cast<Microseconds>(submit_end - submit_start);
    }

    for (size_t env_idx = 0; env_idx < total_games; ++env_idx) {
        int lineup_idx = lineup_indices[env_idx];
        if (lineup_idx < 0) {
            continue;
        }

        if (static_cast<size_t>(lineup_idx) >= outcome.lineups.size()) {
            outcome.lineups.resize(static_cast<size_t>(lineup_idx) + 1);
        }

        const auto& seats = roles[env_idx];
        auto& lineup_result = outcome.lineups[static_cast<size_t>(lineup_idx)].per_policy;

        const Env& env = arena_.envs[env_idx];
        std::vector<int> active;
        active.reserve(num_players);
        for (int seat = 0; seat < num_players; ++seat) {
            if (env.terminations[seat] == 0) {
                active.push_back(seat);
            }
        }

        int winner_seat = (active.size() == 1) ? active[0] : -1;

        for (int seat = 0; seat < num_players; ++seat) {
            int policy_id = seats[seat];
            auto& stats = lineup_result[policy_id];
            stats.num_games += 1;
            if (seat == winner_seat) {
                stats.total_wins += 1;
                stats.total_returns += 2.0;
            } else if (winner_seat >= 0) {
                stats.total_returns -= 1.0;
            }
        }

        for (int i = 0; i < num_players; ++i) {
            int pid_i = seats[i];
            for (int j = 0; j < num_players; ++j) {
                if (i == j) {
                    continue;
                }
                int pid_j = seats[j];
                std::array<int, 2> key{pid_i, pid_j};
                outcome.h2h_counts[key] += 1;
            }
        }

        if (winner_seat >= 0) {
            int winner_pid = seats[winner_seat];
            for (int seat = 0; seat < num_players; ++seat) {
                if (seat == winner_seat) {
                    continue;
                }
                int opponent_pid = seats[seat];
                if (opponent_pid == winner_pid) {
                    continue;
                }
                std::array<int, 2> key{winner_pid, opponent_pid};
                outcome.h2h_wins[key] += 1;
            }
        }
    }

    auto total_end = Clock::now();
    timer_total_run_roles_ = std::chrono::duration_cast<Microseconds>(total_end - total_start);

    return outcome;
}

void EvalManager::run_neural_inference(
    const std::unordered_map<int, std::vector<PolicyRequest>>& requests_by_policy,
    std::unordered_map<int, std::vector<uint8_t>>& out_actions) {

    if (requests_by_policy.empty()) {
        return;
    }

    using Clock = std::chrono::high_resolution_clock;
    using Microseconds = std::chrono::microseconds;

    if (!orchestrator_ || !weights_finalized_) {
        throw std::runtime_error("[EvalManager] Orchestrator not initialized - call finalize_model_loading() first");
    }

    auto t0 = Clock::now();

    // Run unified inference for all neural policies
    auto results = orchestrator_->run_inference(requests_by_policy);

    auto t1 = Clock::now();
    timer_hist_model_exec_ += std::chrono::duration_cast<Microseconds>(t1 - t0);

    // Unpack results by policy
    auto post_t0 = Clock::now();
    for (const auto& kv : requests_by_policy) {
        int policy_id = kv.first;
        const auto& requests = kv.second;

        std::vector<uint8_t> actions;
        actions.reserve(requests.size());

        for (size_t i = 0; i < requests.size(); ++i) {
            auto result_it = results.find({policy_id, static_cast<int>(i)});
            if (result_it == results.end()) {
                throw std::runtime_error("[EvalManager] Missing inference result for policy " +
                                       std::to_string(policy_id) + " request " + std::to_string(i));
            }

            const auto& result = result_it->second;
            actions.push_back(result.action);
        }

        out_actions[policy_id] = std::move(actions);
    }
    auto post_t1 = Clock::now();
    timer_hist_post_ += std::chrono::duration_cast<Microseconds>(post_t1 - post_t0);
}

std::vector<uint8_t> EvalManager::run_cpp_bot(int policy_id,
                                              const std::vector<PolicyRequest>& requests) {
    if (requests.empty()) {
        return {};
    }

    auto it = cpp_bot_registry_.find(policy_id);
    if (it == cpp_bot_registry_.end()) {
        throw std::runtime_error("No registered C++ bot for policy " + std::to_string(policy_id));
    }

    std::vector<uint8_t> actions;
    actions.reserve(requests.size());

    for (const auto& req : requests) {
        uint64_t key = (static_cast<uint64_t>(req.env) << 32)
                       ^ static_cast<uint32_t>(req.seat & 0xFFFFFFFF);
        auto& entry = it->second;
        auto inst_it = entry.instances.find(key);
        if (inst_it == entry.instances.end()) {
            auto instance = make_cpp_bot_instance(entry.kind, req);
            inst_it = entry.instances.emplace(key, std::move(instance)).first;
        }
        uint8_t action = inst_it->second->act(req, arena_);
        actions.push_back(action);
    }

    return actions;
}

std::unordered_map<std::string, int64_t> EvalManager::get_last_performance_stats() const {
    std::unordered_map<std::string, int64_t> stats;
    stats["total_run_roles_us"] = timer_total_run_roles_.count();
    stats["arena_stepping_us"] = timer_arena_stepping_.count();
    stats["collect_requests_us"] = timer_collect_requests_.count();
    stats["model_inference_us"] = timer_model_inference_.count();
    stats["cpp_bots_us"] = timer_cpp_bots_.count();
    stats["hist_prep_batch_us"] = timer_hist_prep_batch_.count();
    stats["hist_prep_weights_us"] = timer_hist_prep_weights_.count();
    stats["hist_model_exec_us"] = timer_hist_model_exec_.count();
    stats["hist_post_us"] = timer_hist_post_.count();

    // Merge detailed timers captured during model forward
    for (const auto& kv : detailed_timers_) {
        stats[kv.first] = kv.second.count();
    }
    return stats;
}

EvalManager::CppBotKind EvalManager::parse_cpp_bot_kind(const std::string& name) {
    auto cache_it = bot_kind_cache_.find(name);
    if (cache_it != bot_kind_cache_.end()) {
        return cache_it->second;
    }

    std::string lower;
    lower.reserve(name.size());
    for (char c : name) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    CppBotKind kind;
    if (lower == "classic") {
        kind = CppBotKind::Classic;
    } else if (lower == "greedycardspammer") {
        kind = CppBotKind::GreedyCardSpammer;
    } else if (lower == "randomagent") {
        kind = CppBotKind::RandomAgent;
    } else if (lower == "selectivetableconservativechallenger") {
        kind = CppBotKind::SelectiveTableConservativeChallenger;
    } else if (lower == "strategicchallenger") {
        kind = CppBotKind::StrategicChallenger;
    } else if (lower == "tablefirstconservativechallenger") {
        kind = CppBotKind::TableFirstConservativeChallenger;
    } else if (lower == "tablenontableagent") {
        kind = CppBotKind::TableNonTableAgent;
    } else {
        throw std::invalid_argument("Unknown C++ bot name: " + name);
    }

    bot_kind_cache_[name] = kind;
    return kind;
}

std::unique_ptr<CppBotBase> EvalManager::make_cpp_bot_instance(EvalManager::CppBotKind kind,
                                                               const PolicyRequest& request) {
    switch (kind) {
        case EvalManager::CppBotKind::Classic:
            return std::make_unique<ClassicBot>();
        case EvalManager::CppBotKind::GreedyCardSpammer:
            return std::make_unique<GreedyCardSpammerBot>();
        case EvalManager::CppBotKind::RandomAgent:
            return std::make_unique<RandomAgentBot>();
        case EvalManager::CppBotKind::SelectiveTableConservativeChallenger:
            return std::make_unique<SelectiveTableConservativeChallengerBot>();
        case EvalManager::CppBotKind::StrategicChallenger: {
            int num_players = 4;
            if (request.env >= 0 && request.env < static_cast<int>(arena_.envs.size())) {
                num_players = arena_.envs[request.env].num_players();
            }
            return std::make_unique<StrategicChallengerBot>(num_players, request.seat);
        }
        case EvalManager::CppBotKind::TableFirstConservativeChallenger:
            return std::make_unique<TableFirstConservativeChallengerBot>();
        case EvalManager::CppBotKind::TableNonTableAgent:
            return std::make_unique<TableNonTableAgentBot>();
    }

    throw std::runtime_error("Unhandled C++ bot kind");
}
