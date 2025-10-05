#include "rollout_manager.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include <torch/torch.h>

#include "bots.h"
#include "torch_utils.h"

namespace {
const torch::Device kInferenceDevice = torch::kCUDA;

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
}

RolloutManager::RolloutManager()
    : rng_(seed_with_optional(0)) {
    if (!torch::cuda::is_available()) {
        throw std::runtime_error(
            "CUDA is not available, but the RolloutManager requires it for historical agent inference.");
    }
    training_device_ = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    arena_.set_max_sequence_length(default_max_sequence_length_);
}

void RolloutManager::start_rollouts(int num_episodes,
                                    int num_players,
                                    const std::vector<int>& training_policy_ids,
                                    int max_batch_envs,
                                    uint32_t seed,
                                    const std::vector<int>& opponent_labels,
                                    const std::vector<double>& opponent_weights,
                                    const std::vector<std::vector<int>>& opponent_triplets) {
    target_episodes_ = num_episodes * std::max(1, num_players);
    num_players_ = num_players;
    training_policy_ids_ = training_policy_ids;
    training_policy_id_set_.clear();
    for (int id : training_policy_ids_) {
        training_policy_id_set_.insert(id);
    }
    completed_buffer_.clear();
    weighted_opponent_labels_ = opponent_labels;
    weighted_opponent_weights_ = opponent_weights;
    fixed_opponent_triplets_ = opponent_triplets;

    const int batch_guess = num_episodes;
    if (max_batch_envs > 0) {
        batch_size_ = std::min(batch_guess, max_batch_envs);
    } else {
        batch_size_ = batch_guess;
    }
    if (batch_size_ <= 0) {
        batch_size_ = std::max(1, num_episodes);
    }

    rng_.seed(seed_with_optional(seed));

    arena_.reset(batch_size_, num_players_, rng_());

    std::vector<std::vector<int>> roles;
    if (!fixed_opponent_triplets_.empty()) {
        roles.assign(batch_size_, std::vector<int>(num_players_, training_policy_id()));
        const size_t training_count = training_policy_ids_.empty() ? 1 : training_policy_ids_.size();
        std::vector<int> training_ids = training_policy_ids_;
        if (training_ids.empty()) {
            training_ids.push_back(training_policy_id());
        }
        for (int env_idx = 0; env_idx < batch_size_; ++env_idx) {
            std::vector<int> env_roles(num_players_, training_ids.front());
            for (size_t seat = 0; seat < training_count && seat < env_roles.size(); ++seat) {
                env_roles[seat] = training_ids[seat % training_ids.size()];
            }
            const auto& triplet = fixed_opponent_triplets_[env_idx % fixed_opponent_triplets_.size()];
            for (size_t seat = training_count; seat < env_roles.size(); ++seat) {
                size_t trip_idx = seat - training_count;
                if (trip_idx < triplet.size()) {
                    env_roles[seat] = triplet[trip_idx];
                }
            }
            roles[env_idx] = std::move(env_roles);
        }
    } else {
        roles = build_roles(batch_size_,
                             num_players_,
                             training_policy_ids_,
                             weighted_opponent_labels_,
                             weighted_opponent_weights_);
    }
    arena_.set_roles(roles);

    episodes_.clear();
    episodes_.resize(batch_size_);
    training_env_inactive_.assign(batch_size_, 0);
    active_training_counts_.assign(batch_size_, 0);
    for (int env_idx = 0; env_idx < batch_size_; ++env_idx) {
        episodes_[env_idx] = new_episode_tracker(env_idx, roles[env_idx]);
        active_training_counts_[env_idx] = static_cast<int>(episodes_[env_idx].training_seats.size());
    }

    for (auto& kv : cpp_bot_registry_) {
        kv.second.instances.clear();
    }
}

std::unordered_map<int, std::vector<PolicyRequest>> RolloutManager::collect_requests_for_inference() {
    std::unordered_map<int, std::vector<PolicyRequest>> learner_requests;

    while (true) {
        log_rewards_and_dones();

        const auto& raw = arena_.collect_requests();
        if (raw.empty()) {
            break;
        }

        std::vector<int> policy_ids;
        policy_ids.reserve(raw.size());
        for (const auto& kv : raw) {
            policy_ids.push_back(kv.first);
        }

        bool progressed_bot = false;
        bool bot_failure = false;
        for (int policy_id : policy_ids) {
            auto raw_it = raw.find(policy_id);
            if (raw_it == raw.end()) {
                continue;
            }

            const auto& requests = raw_it->second;
            if (requests.empty()) {
                continue;
            }

            auto cpp_it = cpp_bot_registry_.find(policy_id);
            if (cpp_it == cpp_bot_registry_.end()) {
                continue;
            }

            bool success = false;
            try {
                auto actions = run_cpp_bot(policy_id, requests);
                arena_.submit_actions(policy_id, actions);
                progressed_bot = true;
                success = true;
            } catch (const std::exception& ex) {
                std::cerr << "[RolloutManager] Native C++ bot execution failed for policy "
                          << policy_id << ": " << ex.what() << std::endl;
            } catch (...) {
                std::cerr << "[RolloutManager] Native C++ bot execution failed for policy "
                          << policy_id << ": unknown error" << std::endl;
            }

            if (!success) {
                auto& dst = learner_requests[policy_id];
                dst.insert(dst.end(), requests.begin(), requests.end());
                bot_failure = true;
            }
        }

        if (bot_failure) {
            break;
        }
        if (progressed_bot) {
            continue;
        }

        int best_policy = -1;
        size_t best_count = 0;
        const std::vector<PolicyRequest>* best_requests = nullptr;
        for (int policy_id : policy_ids) {
            if (is_training_policy(policy_id)) {
                continue;
            }

            auto raw_it = raw.find(policy_id);
            if (raw_it == raw.end()) {
                continue;
            }

            const auto& requests = raw_it->second;
            if (requests.empty()) {
                continue;
            }

            auto model_it = historical_models_.find(policy_id);
            if (model_it == historical_models_.end() || !model_it->second) {
                continue;
            }

            const size_t request_count = requests.size();
            if (best_policy < 0 || request_count > best_count) {
                best_policy = policy_id;
                best_count = request_count;
                best_requests = &requests;
            }
        }

        if (best_policy < 0 || best_requests == nullptr) {
            for (int policy_id : policy_ids) {
                auto raw_it = raw.find(policy_id);
                if (raw_it == raw.end()) {
                    continue;
                }
                const auto& requests = raw_it->second;
                if (requests.empty()) {
                    continue;
                }
                auto& dst = learner_requests[policy_id];
                dst.insert(dst.end(), requests.begin(), requests.end());
            }
            break;
        }

        auto model_it = historical_models_.find(best_policy);
        if (model_it != historical_models_.end() && model_it->second) {
            bool success = false;
            try {
                auto actions = run_historical_inference(*model_it->second, *best_requests);
                arena_.submit_actions(best_policy, actions);
                success = true;
            } catch (const std::exception& ex) {
                std::cerr << "[RolloutManager] Historical inference failed for policy " << best_policy
                          << ": " << ex.what() << std::endl;
            } catch (...) {
                std::cerr << "[RolloutManager] Historical inference failed for policy " << best_policy
                          << ": unknown error" << std::endl;
            }

            if (!success && best_requests != nullptr) {
                auto& dst = learner_requests[best_policy];
                dst.insert(dst.end(), best_requests->begin(), best_requests->end());
                break;
            }

            continue;
        }

        for (int policy_id : policy_ids) {
            auto raw_it = raw.find(policy_id);
            if (raw_it == raw.end()) {
                continue;
            }
            const auto& requests = raw_it->second;
            if (requests.empty()) {
                continue;
            }
            auto& dst = learner_requests[policy_id];
            dst.insert(dst.end(), requests.begin(), requests.end());
        }
        break;
    }

    return learner_requests;
}

void RolloutManager::submit_inference_results(int policy_id,
                                              const std::vector<uint8_t>& actions,
                                              const std::vector<float>& log_probs,
                                              const std::vector<float>& values) {
    submit_inference_results_array(policy_id,
                                   actions.empty() ? nullptr : actions.data(),
                                   actions.size(),
                                   log_probs.empty() ? nullptr : log_probs.data(),
                                   log_probs.size(),
                                   values.empty() ? nullptr : values.data(),
                                   values.size());
}

void RolloutManager::submit_inference_results_array(int policy_id,
                                                    const uint8_t* actions,
                                                    size_t action_count,
                                                    const float* log_probs,
                                                    size_t log_prob_count,
                                                    const float* values,
                                                    size_t value_count) {
    auto it_req = arena_.pending.find(policy_id);
    if (it_req != arena_.pending.end()) {
        const auto& reqs = it_req->second;

        const size_t count = std::min(reqs.size(), action_count);
        const bool has_log_probs = (log_probs != nullptr) && (log_prob_count >= count);
        const bool has_values = (values != nullptr) && (value_count >= count);

        if (is_training_policy(policy_id)) {
            for (size_t i = 0; i < count; ++i) {
                const int env_idx = reqs[i].env;
                const int seat = reqs[i].seat;
                if (env_idx < 0 || env_idx >= static_cast<int>(episodes_.size())) {
                    continue;
                }
                EpisodeTracker& tracker = episodes_[env_idx];
                if (tracker.done) {
                    continue;
                }
                auto seat_it = tracker.training_seats.find(seat);
                if (seat_it == tracker.training_seats.end()) {
                    continue;
                }
                SeatTrajectory& seat_tracker = seat_it->second;
                if (!seat_tracker.active || seat_tracker.policy_id != policy_id) {
                    continue;
                }

                const int step_idx = append_training_step(seat_tracker);
                if (step_idx >= 0 && step_idx < static_cast<int>(seat_tracker.data.our_action.size()) &&
                    actions != nullptr && i < action_count) {
                    seat_tracker.data.our_action[step_idx] = static_cast<int>(actions[i]);
                }

                const float log_prob_value = has_log_probs ? log_probs[i] : 0.0f;
                const float value_value = has_values ? values[i] : 0.0f;
                int penalties_used = 0;
                if (env_idx >= 0 && env_idx < static_cast<int>(arena_.envs.size()) &&
                    seat >= 0 && seat < arena_.envs[env_idx].num_players()) {
                    penalties_used = static_cast<int>(arena_.envs[env_idx].penalties[seat]);
                }

                if (step_idx >= 0 && step_idx < static_cast<int>(seat_tracker.data.log_prob.size())) {
                    seat_tracker.data.log_prob[step_idx] = log_prob_value;
                    seat_tracker.data.value[step_idx] = value_value;
                    seat_tracker.data.penalties_used[step_idx] = penalties_used;
                }
            }
        }
    }

    arena_.submit_actions(policy_id, actions, action_count);
}

std::vector<TrajectoryData> RolloutManager::get_completed_episodes() {
    log_rewards_and_dones();

    for (auto& tracker : episodes_) {
        if (!tracker.done && tracker.env_idx >= 0 && tracker.env_idx < static_cast<int>(arena_.done.size())) {
            if (arena_.done[tracker.env_idx]) {
                finalize_episode(tracker);
            }
        }
    }

    std::vector<TrajectoryData> out;
    if (!completed_buffer_.empty()) {
        out.swap(completed_buffer_);
        if (target_episodes_ > 0 && out.size() > static_cast<size_t>(target_episodes_)) {
            out.resize(static_cast<size_t>(target_episodes_));
        }
    }
    return out;
}

void RolloutManager::load_historical_model(int policy_id, const std::string& path) {
    try {
        auto module = std::make_shared<torch::jit::Module>(torch::jit::load(path));
        module->to(kInferenceDevice);
        module->eval();
        historical_models_[policy_id] = std::move(module);
    } catch (const c10::Error& err) {
        std::cerr << "[RolloutManager] Failed to load TorchScript module from '" << path
                  << "': " << err.what_without_backtrace() << std::endl;
        throw;
    } catch (const std::exception& err) {
        std::cerr << "[RolloutManager] Failed to load TorchScript module from '" << path
                  << "': " << err.what() << std::endl;
        throw;
    }
}

void RolloutManager::register_cpp_bot(int policy_id, const std::string& bot_name) {
    try {
        CppBotKind kind = parse_cpp_bot_kind(bot_name);
        auto& entry = cpp_bot_registry_[policy_id];
        entry.kind = kind;
        entry.instances.clear();
    } catch (const std::exception& err) {
        std::cerr << "[RolloutManager] Failed to register C++ bot '" << bot_name
                  << "' for policy " << policy_id << ": " << err.what() << std::endl;
        throw;
    }
}

void RolloutManager::set_max_sequence_length(int max_len) {
    if (max_len <= 0) {
        max_len = 1;
    }
    default_max_sequence_length_ = max_len;
    arena_.set_max_sequence_length(max_len);
}

void RolloutManager::set_policy_max_sequence_length(int policy_id, int max_len) {
    if (policy_id < 0) {
        return;
    }
    if (max_len <= 0) {
        max_len = 1;
    }
    policy_max_sequence_length_[policy_id] = max_len;
}

PreparedBatch RolloutManager::prepare_training_batch(const std::vector<PolicyRequest>& requests,
                                                     int policy_id) const {
    PreparedBatch batch;
    if (requests.empty()) {
        return batch;
    }

    const int64_t batch_size = static_cast<int64_t>(requests.size());

    // Determine the maximum sequence length allowed for this policy
    int64_t max_allowed = std::max<int64_t>(1, default_max_sequence_length_);
    auto it_limit = policy_max_sequence_length_.find(policy_id);
    if (it_limit != policy_max_sequence_length_.end()) {
        max_allowed = std::max<int64_t>(1, static_cast<int64_t>(it_limit->second));
    }

    // Find the maximum sequence length in this specific batch to determine padding
    int64_t max_len = 1;
    for (const auto& req : requests) {
        const int64_t len = std::max<int64_t>(1, std::min<int64_t>(req.valid_len, max_allowed));
        max_len = std::max(max_len, len);
    }

    torch::Device target_device = training_device_;
    if (target_device.is_cuda() && !torch::cuda::is_available()) {
        target_device = torch::Device(torch::kCPU);
    }

    // If the target is a CUDA device, we use pinned memory on the CPU for faster async transfers.
    bool pin_memory = target_device.is_cuda();

    // Allocate all tensors on the CPU.
    auto opts_float_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(pin_memory);
    auto opts_long_cpu = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(pin_memory);
    auto opts_bool_cpu = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU).pinned_memory(pin_memory);

    auto obs_sequence = torch::zeros({batch_size, max_len, OBS_DIM}, opts_float_cpu);
    auto action_sequence = torch::zeros({batch_size, max_len}, opts_long_cpu);
    auto agent_types = torch::zeros({batch_size, max_len}, opts_long_cpu);
    auto positions = torch::zeros({batch_size, max_len}, opts_long_cpu);
    auto action_masks = torch::zeros({batch_size, max_len, 7}, opts_bool_cpu);
    auto padding_mask = torch::zeros({batch_size, max_len}, opts_bool_cpu);
    auto valid_lengths = torch::zeros({batch_size}, opts_long_cpu);
    auto env_indices = torch::zeros({batch_size}, opts_long_cpu);
    auto seat_indices = torch::zeros({batch_size}, opts_long_cpu);

    // Efficiently fill the CPU tensors with data from the requests.
    for (int64_t b = 0; b < batch_size; ++b) {
        const auto& req = requests[static_cast<size_t>(b)];
        const int64_t requested_len = std::max<int64_t>(0, std::min<int64_t>(req.valid_len, max_allowed));
        const int64_t used_len = std::max<int64_t>(1, std::min<int64_t>(requested_len, max_len));

        valid_lengths[b] = used_len;
        env_indices[b] = static_cast<int64_t>(req.env);
        seat_indices[b] = static_cast<int64_t>(req.seat);

        // Get direct pointers to the tensor data for fast writing.
        float* obs_ptr = obs_sequence[b].data_ptr<float>();
        int64_t* act_ptr = action_sequence[b].data_ptr<int64_t>();
        int64_t* agent_ptr = agent_types[b].data_ptr<int64_t>();
        int64_t* pos_ptr = positions[b].data_ptr<int64_t>();
        bool* mask_ptr = action_masks[b].data_ptr<bool>();

        const float* req_obs_ptr = req.obs_sequence.empty() ? nullptr : req.obs_sequence.data();
        const int64_t* req_action_ptr = req.action_sequence.empty() ? nullptr : req.action_sequence.data();
        const int64_t* req_agent_ptr = req.agent_type_sequence.empty() ? nullptr : req.agent_type_sequence.data();
        const int64_t* req_pos_ptr = req.position_sequence.empty() ? nullptr : req.position_sequence.data();
        const uint8_t* req_mask_ptr = req.action_mask_sequence.empty() ? nullptr : req.action_mask_sequence.data();

        const int64_t obs_rows = req_obs_ptr ? static_cast<int64_t>(req.obs_sequence.size()) / OBS_DIM : 0;
        const int64_t mask_rows = req_mask_ptr ? static_cast<int64_t>(req.action_mask_sequence.size()) / 7 : 0;
        const int64_t action_rows = req_action_ptr ? static_cast<int64_t>(req.action_sequence.size()) : 0;
        const int64_t agent_rows = req_agent_ptr ? static_cast<int64_t>(req.agent_type_sequence.size()) : 0;
        const int64_t pos_rows = req_pos_ptr ? static_cast<int64_t>(req.position_sequence.size()) : 0;

        for (int64_t t = 0; t < used_len; ++t) {
            // Fill Observations
            if (t < requested_len) {
                float* dst_obs = obs_ptr + t * OBS_DIM;
                if (req_obs_ptr && t < obs_rows) {
                    std::memcpy(dst_obs, req_obs_ptr + t * OBS_DIM, sizeof(float) * OBS_DIM);
                } else {
                    std::memset(dst_obs, 0, sizeof(float) * OBS_DIM);
                }

                // Fill Actions, Agent Types, and Masks
                act_ptr[t] = (req_action_ptr && t < action_rows) ? req_action_ptr[t] : 0;
                agent_ptr[t] = (req_agent_ptr && t < agent_rows) ? req_agent_ptr[t] : 0;
                
                bool* step_mask = mask_ptr + t * 7;
                if (req_mask_ptr && t < mask_rows) {
                    const uint8_t* src_mask = req_mask_ptr + t * 7;
                    for (int j = 0; j < 7; ++j) {
                        step_mask[j] = src_mask[j] != 0;
                    }
                } else {
                    for (int j = 0; j < 7; ++j) {
                        step_mask[j] = false;
                    }
                }
            }
            
            // Fill Positions
            pos_ptr[t] = (req_pos_ptr && t < pos_rows) ? req_pos_ptr[t] : t;
        }

        // Handle the special case of the very first step in a game (length 0 request)
        if (requested_len == 0 && used_len > 0) {
            bool* step_mask = mask_ptr; // at t=0
            for (int j = 0; j < 7; ++j) {
                step_mask[j] = req.mask[j] != 0;
            }
        }

        // Fill the padding mask for parts of the tensor beyond the used length.
        bool* pad_ptr = padding_mask[b].data_ptr<bool>();
        for (int64_t t = used_len; t < max_len; ++t) {
            pad_ptr[t] = true;
        }
    }

    // Move the fully prepared tensors to the target device in a single, efficient operation.
    batch.obs_sequence = obs_sequence.to(target_device, /*non_blocking=*/pin_memory, /*copy=*/true);
    batch.action_sequence = action_sequence.to(target_device, pin_memory, true);
    batch.agent_types = agent_types.to(target_device, pin_memory, true);
    batch.positions = positions.to(target_device, pin_memory, true);
    batch.action_masks = action_masks.to(target_device, pin_memory, true);
    batch.padding_mask = padding_mask.to(target_device, pin_memory, true);
    batch.valid_lengths = valid_lengths.to(target_device, pin_memory, true);
    batch.env_indices = env_indices.to(target_device, pin_memory, true);
    batch.seat_indices = seat_indices.to(target_device, pin_memory, true);

    return batch;
}

void RolloutManager::set_training_device(const std::string& device_str) {
    try {
        torch::Device candidate(device_str);
        if (candidate.is_cuda() && !torch::cuda::is_available()) {
            std::cerr << "[RolloutManager] Requested CUDA training device but CUDA is unavailable."
                      << " Falling back to CPU." << std::endl;
            training_device_ = torch::Device(torch::kCPU);
        } else {
            training_device_ = candidate;
        }
    } catch (const std::exception& err) {
        std::cerr << "[RolloutManager] Invalid training device string '" << device_str
                  << "': " << err.what() << ". Keeping previous device." << std::endl;
    }
}

void RolloutManager::mark_training_env_inactive(int env_idx) {
    if (env_idx < 0) {
        return;
    }
    if (env_idx >= static_cast<int>(training_env_inactive_.size())) {
        training_env_inactive_.resize(env_idx + 1, 0);
    }
    if (!training_env_inactive_[env_idx]) {
        training_env_inactive_[env_idx] = 1;
        if (env_idx < static_cast<int>(arena_.done.size())) {
            arena_.done[env_idx] = 1;
        }
    }
}

void RolloutManager::finalize_seat(EpisodeTracker& tracker, SeatTrajectory& seat_tracker, Env& env) {
    if (!seat_tracker.active) {
        return;
    }

    update_penalty_rewards(seat_tracker, env.penalties);

    int our_last_step_idx = -1;
    for (int i = static_cast<int>(seat_tracker.data.agent_id.size()) - 1; i >= 0; --i) {
        if (seat_tracker.data.agent_id[i] == seat_tracker.seat) {
            our_last_step_idx = i;
            break;
        }
    }

    int alive = 0;
    for (int p = 0; p < env.num_players(); ++p) {
        if (!env.terminations[p]) {
            ++alive;
        }
    }
    const bool seat_alive = seat_tracker.seat >= 0 && seat_tracker.seat < env.num_players() &&
                            !env.terminations[seat_tracker.seat];
    const bool is_winner = seat_alive && alive == 1;
    seat_tracker.data.win = is_winner ? 1 : 0;

    if (our_last_step_idx >= 0 && our_last_step_idx < static_cast<int>(seat_tracker.data.reward.size())) {
        seat_tracker.data.reward[our_last_step_idx] += is_winner ? 1.0 : -1.0;
    }

    seat_tracker.data.episode_return =
        std::accumulate(seat_tracker.data.reward.begin(), seat_tracker.data.reward.end(), 0.0);

    completed_buffer_.push_back(std::move(seat_tracker.data));
    seat_tracker.data = TrajectoryData{};
    seat_tracker.active = false;
    seat_tracker.last_training_step_idx = -1;
    seat_tracker.last_penalties.fill(0);

    if (tracker.env_idx >= 0 && tracker.env_idx < static_cast<int>(active_training_counts_.size())) {
        int& count = active_training_counts_[tracker.env_idx];
        if (count > 0) {
            --count;
        }
        if (count <= 0) {
            mark_training_env_inactive(tracker.env_idx);
        }
    }
}

bool RolloutManager::is_training_policy(int policy_id) const {
    return training_policy_id_set_.find(policy_id) != training_policy_id_set_.end();
}

std::vector<uint8_t> RolloutManager::run_historical_inference(torch::jit::Module& module,
                                                              const std::vector<PolicyRequest>& requests) {
    if (requests.empty()) {
        return {};
    }

    torch::NoGradGuard no_grad;

    constexpr std::array<int64_t, 7> kBucketBounds{{32, 64, 96, 128, 192, 256, 480}};
    const int64_t max_limit = std::max<int64_t>(1, static_cast<int64_t>(default_max_sequence_length_));

    auto select_bucket_index = [&](int64_t length) -> size_t {
        const int64_t clamped = std::max<int64_t>(1, std::min<int64_t>(length, max_limit));
        for (size_t i = 0; i < kBucketBounds.size(); ++i) {
            if (clamped <= kBucketBounds[i]) {
                return i;
            }
        }
        return kBucketBounds.size() - 1;
    };

    std::array<std::vector<size_t>, kBucketBounds.size()> bucket_indices{};
    for (size_t idx = 0; idx < requests.size(); ++idx) {
        const auto& req = requests[idx];
        const size_t bucket = select_bucket_index(req.valid_len);
        bucket_indices[bucket].push_back(idx);
    }

    std::vector<uint8_t> chosen(requests.size(), 0);

    auto opts_long_device = torch::TensorOptions().dtype(torch::kInt64).device(kInferenceDevice);

    for (size_t bucket_idx = 0; bucket_idx < bucket_indices.size(); ++bucket_idx) {
        const auto& indices = bucket_indices[bucket_idx];
        if (indices.empty()) {
            continue;
        }
        const int64_t batch_size = static_cast<int64_t>(indices.size());
        const int64_t target_pad_len =
            std::min<int64_t>(kBucketBounds[bucket_idx], max_limit);

        auto tensor_batch =
            prepare_inference_batch(requests, target_pad_len, kInferenceDevice, indices);
        auto& obs_sequence = tensor_batch.obs_sequence;
        auto& action_sequence = tensor_batch.action_sequence;
        auto& agent_types = tensor_batch.agent_types;
        auto& positions = tensor_batch.positions;
        auto& action_masks = tensor_batch.action_masks;
        auto& padding_mask = tensor_batch.padding_mask;
        auto valid_lengths_device = tensor_batch.valid_lengths;

        std::vector<torch::jit::IValue> inputs;
        inputs.reserve(6);
        inputs.emplace_back(obs_sequence);
        inputs.emplace_back(action_sequence);
        inputs.emplace_back(agent_types);
        inputs.emplace_back(positions);
        inputs.emplace_back(action_masks);
        inputs.emplace_back(padding_mask);

        auto outputs = module.forward(inputs).toTuple();
        if (!outputs || outputs->elements().size() < 3) {
            throw std::runtime_error("Historical model returned unexpected output shape");
        }

        auto action_logits = outputs->elements()[0].toTensor().contiguous();
        (void)outputs->elements()[2].toTensor();

        auto batch_indices = torch::arange(batch_size, opts_long_device);
        auto last_indices = (valid_lengths_device - 1).clamp_min(0);
        auto last_logits = action_logits.index({batch_indices, last_indices}).contiguous();
        auto last_masks = action_masks.index({batch_indices, last_indices}).contiguous();

        auto has_legal = last_masks.any(1);
        if (!has_legal.all().item<bool>()) {
            auto fallback_indices = has_legal.logical_not().nonzero().flatten();
            for (int64_t i = 0; i < fallback_indices.size(0); ++i) {
                const int64_t row = fallback_indices[i].item<int64_t>();
                auto mask_row_tensor = last_masks.select(0, row);
                mask_row_tensor.fill_(false);
                bool assigned = false;
                const auto& req = requests[indices[static_cast<size_t>(row)]];
                for (int j = 0; j < 7; ++j) {
                    if (req.mask[j]) {
                        mask_row_tensor.index_put_({j}, true);
                        assigned = true;
                    }
                }
                if (!assigned) {
                    mask_row_tensor.fill_(true);
                }
                last_logits.select(0, row).fill_(0.0f);
            }
        }

        last_logits.masked_fill_(~last_masks, -std::numeric_limits<float>::infinity());

        auto probs = torch::softmax(last_logits, /*dim=*/1);
        auto actions_tensor = torch::multinomial(probs, /*num_samples=*/1);
        actions_tensor = actions_tensor.squeeze(-1).to(torch::kCPU);

        auto actions_ptr = actions_tensor.data_ptr<int64_t>();
        for (int64_t b = 0; b < batch_size; ++b) {
            chosen[indices[static_cast<size_t>(b)]] = static_cast<uint8_t>(actions_ptr[b]);
        }
    }

    return chosen;
}

std::vector<uint8_t> RolloutManager::run_cpp_bot(int policy_id, const std::vector<PolicyRequest>& requests) {
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
        uint64_t key = (static_cast<uint64_t>(req.env) << 32) ^ static_cast<uint32_t>(req.seat & 0xFFFFFFFF);
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

RolloutManager::CppBotKind RolloutManager::parse_cpp_bot_kind(const std::string& name) {
    std::string lower;
    lower.reserve(name.size());
    for (char c : name) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    if (lower == "classic") {
        return CppBotKind::Classic;
    } else if (lower == "greedycardspammer") {
        return CppBotKind::GreedyCardSpammer;
    } else if (lower == "randomagent") {
        return CppBotKind::RandomAgent;
    } else if (lower == "selectivetableconservativechallenger") {
        return CppBotKind::SelectiveTableConservativeChallenger;
    } else if (lower == "strategicchallenger") {
        return CppBotKind::StrategicChallenger;
    } else if (lower == "tablefirstconservativechallenger") {
        return CppBotKind::TableFirstConservativeChallenger;
    } else if (lower == "tablenontableagent") {
        return CppBotKind::TableNonTableAgent;
    }

    throw std::invalid_argument("Unknown C++ bot name: " + name);
}

std::unique_ptr<CppBotBase> RolloutManager::make_cpp_bot_instance(RolloutManager::CppBotKind kind,
                                                                  const PolicyRequest& request) {
    switch (kind) {
        case RolloutManager::CppBotKind::Classic:
            return std::make_unique<ClassicBot>();
        case RolloutManager::CppBotKind::GreedyCardSpammer:
            return std::make_unique<GreedyCardSpammerBot>();
        case RolloutManager::CppBotKind::RandomAgent:
            return std::make_unique<RandomAgentBot>();
        case RolloutManager::CppBotKind::SelectiveTableConservativeChallenger:
            return std::make_unique<SelectiveTableConservativeChallengerBot>();
        case RolloutManager::CppBotKind::StrategicChallenger: {
            int num_players = 4;
            if (request.env >= 0 && request.env < static_cast<int>(arena_.envs.size())) {
                num_players = arena_.envs[request.env].num_players();
            }
            return std::make_unique<StrategicChallengerBot>(num_players, request.seat);
        }
        case RolloutManager::CppBotKind::TableFirstConservativeChallenger:
            return std::make_unique<TableFirstConservativeChallengerBot>();
        case RolloutManager::CppBotKind::TableNonTableAgent:
            return std::make_unique<TableNonTableAgentBot>();
    }

    throw std::runtime_error("Unsupported C++ bot kind");
}

std::vector<std::vector<int>> RolloutManager::build_roles(int batch_size,
                                                          int num_players,
                                                          const std::vector<int>& training_policy_ids,
                                                          const std::vector<int>& weighted_opponents,
                                                          const std::vector<double>& opponent_weights) {
    const int fallback_training_id =
        training_policy_ids.empty() ? training_policy_id() : training_policy_ids.front();
    std::vector<std::vector<int>> roles(batch_size, std::vector<int>(num_players, fallback_training_id));
    if (batch_size <= 0 || num_players <= 0) {
        return roles;
    }

    std::vector<int> opponent_pool = weighted_opponents;
    std::vector<double> probs = opponent_weights;

    bool use_weighted = !opponent_pool.empty() && opponent_pool.size() == probs.size();
    if (use_weighted) {
        double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
        if (!std::isfinite(sum) || sum <= 1e-9) {
            use_weighted = false;
        } else {
            for (double& p : probs) {
                if (!std::isfinite(p) || p < 0.0) {
                    p = 0.0;
                }
            }
            sum = std::accumulate(probs.begin(), probs.end(), 0.0);
            if (sum <= 1e-9) {
                use_weighted = false;
            } else {
                for (double& p : probs) {
                    p /= sum;
                }
            }
        }
    }

    if (!use_weighted) {
        if (opponent_pool.empty()) {
            opponent_pool.push_back(fallback_training_id);
        }
        const size_t pool_size = opponent_pool.size();
        probs.assign(pool_size, pool_size > 0 ? 1.0 / static_cast<double>(pool_size) : 1.0);
    }

    const int training_slots =
        std::min<int>(std::max<int>(1, static_cast<int>(training_policy_ids.size())), num_players);
    const int num_opponents = std::max(0, num_players - training_slots);
    std::vector<int> seats(num_players);
    std::iota(seats.begin(), seats.end(), 0);

    std::discrete_distribution<int> sampler(probs.begin(), probs.end());

    for (int b = 0; b < batch_size; ++b) {
        std::shuffle(seats.begin(), seats.end(), rng_);
        std::vector<int> assignments = training_policy_ids;
        if (assignments.empty()) {
            assignments.push_back(fallback_training_id);
        }
        std::shuffle(assignments.begin(), assignments.end(), rng_);
        if (static_cast<int>(assignments.size()) > training_slots) {
            assignments.resize(training_slots);
        } else if (static_cast<int>(assignments.size()) < training_slots) {
            assignments.resize(training_slots, assignments.back());
        }
        for (int slot = 0; slot < training_slots; ++slot) {
            roles[b][seats[slot]] = assignments[slot];
        }

        for (int opponent_idx = 0; opponent_idx < num_opponents; ++opponent_idx) {
            const int seat = seats[opponent_idx + training_slots];
            const int choice = opponent_pool[sampler(rng_)];
            roles[b][seat] = choice;
        }
    }

    return roles;
}

EpisodeTracker RolloutManager::new_episode_tracker(int env_idx, const std::vector<int>& roles) {
    EpisodeTracker tracker;
    tracker.env_idx = env_idx;
    tracker.last_history_len = 0;
    tracker.last_processed_history_len = 0;

    for (size_t i = 0; i < roles.size(); ++i) {
        const int policy_id = roles[i];
        if (training_policy_id_set_.find(policy_id) == training_policy_id_set_.end()) {
            continue;
        }
        SeatTrajectory seat_tracker;
        seat_tracker.seat = static_cast<int>(i);
        seat_tracker.policy_id = policy_id;
        seat_tracker.active = true;
        seat_tracker.last_training_step_idx = -1;
        seat_tracker.last_penalties.fill(0);
        seat_tracker.data.env_index = env_idx;
        seat_tracker.data.training_policy_id = policy_id;
        seat_tracker.data.training_agent_seat = static_cast<int>(i);
        seat_tracker.data.player_policy_ids = roles;
        tracker.training_seats.emplace(seat_tracker.seat, std::move(seat_tracker));
    }

    return tracker;
}

int RolloutManager::append_training_step(SeatTrajectory& seat_tracker) {
    seat_tracker.data.agent_id.push_back(seat_tracker.seat);
    seat_tracker.data.our_action.push_back(-1);
    seat_tracker.data.log_prob.push_back(0.0f);
    seat_tracker.data.value.push_back(0.0f);
    seat_tracker.data.reward.push_back(0.0);
    seat_tracker.data.done.push_back(0);
    seat_tracker.data.opp_target_action.push_back(-1);
    seat_tracker.data.penalties_used.push_back(0);
    const int idx = static_cast<int>(seat_tracker.data.agent_id.size()) - 1;
    seat_tracker.last_training_step_idx = idx;
    return idx;
}

int RolloutManager::append_opponent_step(SeatTrajectory& seat_tracker, int seat) {
    seat_tracker.data.agent_id.push_back(seat);
    seat_tracker.data.our_action.push_back(-1);
    seat_tracker.data.log_prob.push_back(0.0f);
    seat_tracker.data.value.push_back(0.0f);
    seat_tracker.data.reward.push_back(0.0);
    seat_tracker.data.done.push_back(0);
    seat_tracker.data.opp_target_action.push_back(-1);
    seat_tracker.data.penalties_used.push_back(0);
    return static_cast<int>(seat_tracker.data.agent_id.size()) - 1;
}

void RolloutManager::update_penalty_rewards(SeatTrajectory& seat_tracker,
                                            const std::array<uint8_t, Env::MAX_PLAYERS>& penalties) {
    const int seat = seat_tracker.seat;
    if (!seat_tracker.active || seat < 0 || seat >= static_cast<int>(penalties.size())) {
        seat_tracker.last_penalties = penalties;
        return;
    }

    const int last_idx = seat_tracker.last_training_step_idx;
    if (last_idx < 0 || last_idx >= static_cast<int>(seat_tracker.data.reward.size())) {
        seat_tracker.last_penalties = penalties;
        return;
    }

    double delta_total = 0.0;
    for (size_t i = 0; i < penalties.size(); ++i) {
        const int diff = static_cast<int>(penalties[i]) - static_cast<int>(seat_tracker.last_penalties[i]);
        if (diff <= 0) {
            continue;
        }
        if (static_cast<int>(i) == seat) {
            delta_total -= 0.1 * diff;
        } else {
            delta_total += 0.033 * diff;
        }
    }
    if (delta_total != 0.0) {
        seat_tracker.data.reward[last_idx] += delta_total;
    }
    seat_tracker.last_penalties = penalties;
}

void RolloutManager::log_rewards_and_dones() {
    for (auto& tracker : episodes_) {
        if (tracker.done || tracker.env_idx < 0 || tracker.env_idx >= static_cast<int>(arena_.envs.size())) {
            continue;
        }
        Env& env = arena_.envs[tracker.env_idx];
        const int total_len = env.get_total_history_entries();
        const int start_idx = tracker.last_processed_history_len;
        if (start_idx < total_len) {
            auto history = env.get_history_entries_slice(start_idx, total_len);
            for (const auto& entry : history) {
                tracker.last_history_len += 1;
                const int actor = entry.player;
                for (auto& kv : tracker.training_seats) {
                    SeatTrajectory& seat_tracker = kv.second;
                    if (!seat_tracker.active) {
                        continue;
                    }
                    if (seat_tracker.seat != actor) {
                        const int idx = append_opponent_step(seat_tracker, actor);
                        if (idx >= 0 && idx < static_cast<int>(seat_tracker.data.opp_target_action.size())) {
                            seat_tracker.data.opp_target_action[idx] = static_cast<int>(entry.action);
                        }
                    }
                }
            }
            tracker.last_processed_history_len = total_len;
        }

        for (auto& kv : tracker.training_seats) {
            SeatTrajectory& seat_tracker = kv.second;
            if (!seat_tracker.active) {
                continue;
            }
            update_penalty_rewards(seat_tracker, env.penalties);
        }

        const bool env_done = arena_.done[tracker.env_idx];
        for (auto& kv : tracker.training_seats) {
            SeatTrajectory& seat_tracker = kv.second;
            if (!seat_tracker.active) {
                continue;
            }
            const bool seat_terminated = seat_tracker.seat >= 0 && seat_tracker.seat < env.num_players() &&
                                         env.terminations[seat_tracker.seat];
            if (seat_terminated || env_done) {
                finalize_seat(tracker, seat_tracker, env);
            }
        }

        bool any_active = false;
        for (const auto& kv : tracker.training_seats) {
            if (kv.second.active) {
                any_active = true;
                break;
            }
        }
        if (!any_active) {
            tracker.done = true;
        }
    }
}

void RolloutManager::finalize_episode(EpisodeTracker& tracker) {
    if (tracker.done) {
        return;
    }
    if (tracker.env_idx < 0 || tracker.env_idx >= static_cast<int>(arena_.envs.size())) {
        tracker.done = true;
        return;
    }
    Env& env = arena_.envs[tracker.env_idx];

    for (auto& kv : tracker.training_seats) {
        SeatTrajectory& seat_tracker = kv.second;
        finalize_seat(tracker, seat_tracker, env);
    }

    tracker.done = true;
}

