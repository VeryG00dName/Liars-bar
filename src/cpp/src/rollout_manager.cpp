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
}

void RolloutManager::start_rollouts(int num_episodes,
                                    int num_players,
                                    int training_policy_id,
                                    int max_batch_envs,
                                    uint32_t seed,
                                    const std::vector<int>& cpp_bots,
                                    const std::vector<int>& latest_historical_agents,
                                    const std::vector<int>& active_shadow_agents,
                                    double front_mass,
                                    double shadow_mass) {
    target_episodes_ = num_episodes;
    num_players_ = num_players;
    training_policy_id_ = training_policy_id;
    completed_buffer_.clear();
    pending_step_data_.clear();

    const int batch_guess = (arena_.B > 0) ? arena_.B : num_episodes;
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

    std::vector<int> front_pool = latest_historical_agents;

    std::vector<int> shadow_pool = active_shadow_agents;
    shadow_pool.insert(shadow_pool.end(), cpp_bots.begin(), cpp_bots.end());

    auto roles = build_roles(batch_size_,
                             num_players_,
                             training_policy_id_,
                             front_pool,
                             shadow_pool,
                             front_mass,
                             shadow_mass);
    arena_.set_roles(roles);

    episodes_.clear();
    episodes_.resize(batch_size_);
    training_env_inactive_.assign(batch_size_, 0);
    for (int env_idx = 0; env_idx < batch_size_; ++env_idx) {
        episodes_[env_idx] = new_episode_tracker(env_idx, roles[env_idx]);
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
            if (policy_id == training_policy_id_) {
                continue;
            }
            auto model_it = historical_models_.find(policy_id);
            if (model_it == historical_models_.end() || !model_it->second) {
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
            if (requests.size() > best_count) {
                best_count = requests.size();
                best_policy = policy_id;
                best_requests = &requests;
            }
        }

        if (best_policy >= 0 && best_requests != nullptr) {
            bool success = false;
            try {
                auto actions = run_historical_inference(*historical_models_[best_policy], *best_requests);
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
            if (!tracker.is_training_episode || seat != tracker.training_agent_seat ||
                policy_id != training_policy_id_) {
                continue;
            }

            const int step_idx = append_step_row(tracker, seat);
            if (step_idx >= 0 && step_idx < static_cast<int>(tracker.data.our_action.size()) &&
                actions != nullptr && i < action_count) {
                tracker.data.our_action[step_idx] = static_cast<int>(actions[i]);
            }

            PendingStepData pending{};
            pending.log_prob = (has_log_probs ? log_probs[i] : 0.0f);
            pending.value = (has_values ? values[i] : 0.0f);
            if (env_idx >= 0 && env_idx < static_cast<int>(arena_.envs.size()) &&
                seat >= 0 && seat < arena_.envs[env_idx].num_players()) {
                pending.penalties_used = static_cast<int>(arena_.envs[env_idx].penalties[seat]);
            }
            pending_step_data_[env_idx] = pending;
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

PreparedBatch RolloutManager::prepare_training_batch(const std::vector<PolicyRequest>& requests) const {
    PreparedBatch batch;
    if (requests.empty()) {
        return batch;
    }

    const int64_t batch_size = static_cast<int64_t>(requests.size());
    int64_t max_len = 1;
    for (const auto& req : requests) {
        const int64_t len = std::max<int64_t>(1, std::min<int64_t>(req.valid_len, MAX_LEN));
        max_len = std::max(max_len, len);
    }

    auto opts_float_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto opts_long_cpu = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto opts_bool_cpu = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);

    auto obs_sequence = torch::zeros({batch_size, max_len, OBS_DIM}, opts_float_cpu);
    auto action_sequence = torch::zeros({batch_size, max_len}, opts_long_cpu);
    auto agent_types = torch::zeros({batch_size, max_len}, opts_long_cpu);
    auto positions = torch::zeros({batch_size, max_len}, opts_long_cpu);
    auto action_masks = torch::zeros({batch_size, max_len, 7}, opts_bool_cpu);
    auto padding_mask = torch::zeros({batch_size, max_len}, opts_bool_cpu);
    auto valid_lengths = torch::zeros({batch_size}, opts_long_cpu);
    auto env_indices = torch::zeros({batch_size}, opts_long_cpu);
    auto seat_indices = torch::zeros({batch_size}, opts_long_cpu);

    for (int64_t b = 0; b < batch_size; ++b) {
        const auto& req = requests[static_cast<size_t>(b)];
        const int64_t requested_len = std::max<int64_t>(0, std::min<int64_t>(req.valid_len, max_len));
        const int64_t used_len = std::max<int64_t>(1, requested_len);

        valid_lengths[b] = used_len;
        env_indices[b] = static_cast<int64_t>(req.env);
        seat_indices[b] = static_cast<int64_t>(req.seat);

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

        for (int64_t t = 0; t < requested_len; ++t) {
            float* dst_obs = obs_ptr + t * OBS_DIM;
            if (req_obs_ptr && t < obs_rows) {
                std::memcpy(dst_obs, req_obs_ptr + t * OBS_DIM, sizeof(float) * OBS_DIM);
            } else {
                std::memset(dst_obs, 0, sizeof(float) * OBS_DIM);
            }

            act_ptr[t] = (req_action_ptr && t < action_rows) ? req_action_ptr[t] : 0;
            agent_ptr[t] = (req_agent_ptr && t < agent_rows) ? req_agent_ptr[t] : 0;
            pos_ptr[t] = (req_pos_ptr && t < pos_rows) ? req_pos_ptr[t] : t;

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

        if (requested_len == 0) {
            bool* step_mask = mask_ptr;
            for (int j = 0; j < 7; ++j) {
                step_mask[j] = req.mask[j] != 0;
            }
        }

        for (int64_t t = requested_len; t < used_len; ++t) {
            pos_ptr[t] = t;
        }

        bool* pad_ptr = padding_mask[b].data_ptr<bool>();
        for (int64_t t = used_len; t < max_len; ++t) {
            pad_ptr[t] = true;
        }
    }

    torch::Device target_device = training_device_;
    if (target_device.is_cuda() && !torch::cuda::is_available()) {
        target_device = torch::Device(torch::kCPU);
    }

    batch.obs_sequence = obs_sequence.to(target_device, /*non_blocking=*/false, /*copy=*/true);
    batch.action_sequence = action_sequence.to(target_device, false, true);
    batch.agent_types = agent_types.to(target_device, false, true);
    batch.positions = positions.to(target_device, false, true);
    batch.action_masks = action_masks.to(target_device, false, true);
    batch.padding_mask = padding_mask.to(target_device, false, true);
    batch.valid_lengths = valid_lengths.to(target_device, false, true);
    batch.env_indices = env_indices.to(target_device, false, true);
    batch.seat_indices = seat_indices.to(target_device, false, true);

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
    training_env_inactive_[env_idx] = 1;
    if (env_idx >= 0 && env_idx < static_cast<int>(arena_.done.size())) {
        arena_.done[env_idx] = 1;
    }
}

std::vector<uint8_t> RolloutManager::run_historical_inference(torch::jit::Module& module,
                                                              const std::vector<PolicyRequest>& requests) {
    if (requests.empty()) {
        return {};
    }

    torch::NoGradGuard no_grad;

    const int64_t batch_size = static_cast<int64_t>(requests.size());
    int64_t max_len = 1;
    for (const auto& req : requests) {
        const int64_t len = std::max<int64_t>(1, std::min<int64_t>(req.valid_len, MAX_LEN));
        max_len = std::max(max_len, len);
    }

    static const std::array<int64_t, 7> kBucketBounds{{16, 32, 64, 128, 160, 192, 256}};
    int64_t target_pad_len = max_len;
    for (int64_t bound : kBucketBounds) {
        if (max_len <= bound) {
            target_pad_len = bound;
            break;
        }
    }
    if (max_len > kBucketBounds.back()) {
        target_pad_len = std::max<int64_t>(max_len, kBucketBounds.back());
    }
    target_pad_len = std::min<int64_t>(target_pad_len, static_cast<int64_t>(MAX_LEN));

    auto opts_float_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto opts_long_cpu = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto opts_bool_cpu = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);

    auto obs_sequence = torch::zeros({batch_size, target_pad_len, OBS_DIM}, opts_float_cpu);
    auto action_sequence = torch::zeros({batch_size, target_pad_len}, opts_long_cpu);
    auto agent_types = torch::zeros({batch_size, target_pad_len}, opts_long_cpu);
    auto positions = torch::zeros({batch_size, target_pad_len}, opts_long_cpu);
    auto action_masks = torch::zeros({batch_size, target_pad_len, 7}, opts_bool_cpu);
    auto padding_mask = torch::zeros({batch_size, target_pad_len}, opts_bool_cpu);
    auto valid_lengths = torch::zeros({batch_size}, opts_long_cpu);

    for (int64_t b = 0; b < batch_size; ++b) {
        const auto& req = requests[static_cast<size_t>(b)];
        const int64_t requested_len = std::max<int64_t>(0, std::min<int64_t>(req.valid_len, target_pad_len));
        const int64_t used_len = std::max<int64_t>(1, requested_len);

        valid_lengths[b] = used_len;

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

        for (int64_t t = 0; t < requested_len; ++t) {
            float* dst_obs = obs_ptr + t * OBS_DIM;
            if (req_obs_ptr && t < obs_rows) {
                std::memcpy(dst_obs, req_obs_ptr + t * OBS_DIM, sizeof(float) * OBS_DIM);
            } else {
                std::memset(dst_obs, 0, sizeof(float) * OBS_DIM);
            }

            act_ptr[t] = (req_action_ptr && t < action_rows) ? req_action_ptr[t] : 0;
            agent_ptr[t] = (req_agent_ptr && t < agent_rows) ? req_agent_ptr[t] : 0;
            pos_ptr[t] = (req_pos_ptr && t < pos_rows) ? req_pos_ptr[t] : t;

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

        if (requested_len == 0) {
            bool* step_mask = mask_ptr;
            for (int j = 0; j < 7; ++j) {
                step_mask[j] = req.mask[j] != 0;
            }
        }

        for (int64_t t = requested_len; t < used_len; ++t) {
            pos_ptr[t] = t;
        }

        bool* pad_ptr = padding_mask[b].data_ptr<bool>();
        for (int64_t t = used_len; t < target_pad_len; ++t) {
            pad_ptr[t] = true;
        }
    }

    auto opts_long_device = torch::TensorOptions().dtype(torch::kInt64).device(kInferenceDevice);

    obs_sequence = obs_sequence.to(kInferenceDevice);
    action_sequence = action_sequence.to(kInferenceDevice);
    agent_types = agent_types.to(kInferenceDevice);
    positions = positions.to(kInferenceDevice);
    action_masks = action_masks.to(kInferenceDevice);
    padding_mask = padding_mask.to(kInferenceDevice);
    auto valid_lengths_device = valid_lengths.to(kInferenceDevice);

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
    (void)outputs->elements()[2].toTensor(); // ensure tensor materializes even if unused

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
            const auto& req = requests[static_cast<size_t>(row)];
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
    std::vector<uint8_t> chosen(static_cast<size_t>(batch_size));
    for (int64_t b = 0; b < batch_size; ++b) {
        chosen[static_cast<size_t>(b)] = static_cast<uint8_t>(actions_ptr[b]);
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
                                                          int training_policy_id,
                                                          const std::vector<int>& front_agents,
                                                          const std::vector<int>& shadow_agents,
                                                          double front_mass,
                                                          double shadow_mass) {
    std::vector<std::vector<int>> roles(batch_size, std::vector<int>(num_players, training_policy_id));
    if (batch_size <= 0 || num_players <= 0) {
        return roles;
    }

    auto unique_sorted = [](std::vector<int> values) {
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        return values;
    };

    std::vector<int> front = unique_sorted(front_agents);
    std::vector<int> shadow = unique_sorted(shadow_agents);

    if (!front.empty()) {
        std::unordered_set<int> front_set(front.begin(), front.end());
        std::vector<int> filtered;
        filtered.reserve(shadow.size());
        for (int id : shadow) {
            if (!front_set.count(id)) {
                filtered.push_back(id);
            }
        }
        shadow.swap(filtered);
    }

    std::vector<int> opponent_pool = front;
    opponent_pool.insert(opponent_pool.end(), shadow.begin(), shadow.end());
    if (opponent_pool.empty()) {
        opponent_pool.push_back(training_policy_id);
    }

    const size_t pool_size = opponent_pool.size();
    std::vector<double> probs(pool_size, 0.0);

    const double adjusted_front_mass = front.empty() ? 0.0 : front_mass;
    const double adjusted_shadow_mass = shadow.empty() ? 0.0 : shadow_mass;
    const double total_mass = adjusted_front_mass + adjusted_shadow_mass;

    if (total_mass <= 0.0 || pool_size == 0) {
        const double uniform = pool_size > 0 ? 1.0 / static_cast<double>(pool_size) : 1.0;
        std::fill(probs.begin(), probs.end(), uniform);
    } else {
        std::unordered_set<int> front_set(front.begin(), front.end());
        const double front_norm = front.empty() ? 1.0 : static_cast<double>(front.size());
        const double shadow_norm = shadow.empty() ? 1.0 : static_cast<double>(shadow.size());

        for (size_t i = 0; i < pool_size; ++i) {
            const bool is_front = front_set.count(opponent_pool[i]) > 0;
            const double bucket_mass = is_front ? adjusted_front_mass : adjusted_shadow_mass;
            const double bucket_norm = is_front ? front_norm : shadow_norm;
            probs[i] = bucket_mass / bucket_norm;
        }
        const double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
        if (sum > 0.0) {
            for (double& p : probs) {
                p /= sum;
            }
        } else {
            const double uniform = pool_size > 0 ? 1.0 / static_cast<double>(pool_size) : 1.0;
            std::fill(probs.begin(), probs.end(), uniform);
        }
    }

    const int num_opponents = std::max(0, num_players - 1);
    std::vector<int> seats(num_players);
    std::iota(seats.begin(), seats.end(), 0);

    std::discrete_distribution<int> sampler(probs.begin(), probs.end());

    for (int b = 0; b < batch_size; ++b) {
        std::shuffle(seats.begin(), seats.end(), rng_);
        const int training_seat = seats.front();
        roles[b][training_seat] = training_policy_id;

        for (int opponent_idx = 0; opponent_idx < num_opponents; ++opponent_idx) {
            const int seat = seats[opponent_idx + 1];
            const int choice = opponent_pool[sampler(rng_)];
            roles[b][seat] = choice;
        }
    }

    return roles;
}

EpisodeTracker RolloutManager::new_episode_tracker(int env_idx, const std::vector<int>& roles) {
    EpisodeTracker tracker;
    tracker.env_idx = env_idx;
    tracker.training_policy_id = training_policy_id_;
    tracker.data.env_index = env_idx;
    tracker.data.training_policy_id = training_policy_id_;
    tracker.data.player_policy_ids = roles;

    int seat = -1;
    for (size_t i = 0; i < roles.size(); ++i) {
        if (roles[i] == training_policy_id_) {
            seat = static_cast<int>(i);
            break;
        }
    }
    if (seat >= 0) {
        tracker.is_training_episode = true;
        tracker.training_agent_seat = seat;
        tracker.data.training_agent_seat = seat;
    } else {
        tracker.is_training_episode = false;
        tracker.training_agent_seat = -1;
        tracker.data.training_agent_seat = -1;
    }

    tracker.last_penalties.fill(0);

    return tracker;
}

int RolloutManager::append_step_row(EpisodeTracker& tracker, int seat) {
    tracker.data.agent_id.push_back(seat);
    tracker.data.our_action.push_back(-1);
    tracker.data.log_prob.push_back(0.0f);
    tracker.data.value.push_back(0.0f);
    tracker.data.reward.push_back(0.0);
    tracker.data.done.push_back(0);
    tracker.data.opp_target_action.push_back(-1);
    tracker.data.penalties_used.push_back(0);
    const int idx = static_cast<int>(tracker.data.agent_id.size()) - 1;
    if (seat == tracker.training_agent_seat) {
        tracker.last_training_step_idx = idx;
    }
    return idx;
}

void RolloutManager::update_penalty_rewards(EpisodeTracker& tracker, const std::array<uint8_t, Env::MAX_PLAYERS>& penalties) {
    if (!tracker.is_training_episode) {
        tracker.last_penalties = penalties;
        return;
    }

    const int seat = tracker.training_agent_seat;
    const int last_idx = tracker.last_training_step_idx;
    if (seat < 0 || last_idx < 0 || last_idx >= static_cast<int>(tracker.data.reward.size())) {
        tracker.last_penalties = penalties;
        return;
    }

    double delta_total = 0.0;
    for (size_t i = 0; i < penalties.size(); ++i) {
        const int diff = static_cast<int>(penalties[i]) - static_cast<int>(tracker.last_penalties[i]);
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
        tracker.data.reward[last_idx] += delta_total;
    }
    tracker.last_penalties = penalties;
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
                if (tracker.is_training_episode && actor == tracker.training_agent_seat) {
                    auto it_pending = pending_step_data_.find(tracker.env_idx);
                    if (it_pending != pending_step_data_.end()) {
                        const int idx = tracker.last_training_step_idx;
                        if (idx >= 0 && idx < static_cast<int>(tracker.data.log_prob.size())) {
                            tracker.data.log_prob[idx] = it_pending->second.log_prob;
                            tracker.data.value[idx] = it_pending->second.value;
                            tracker.data.penalties_used[idx] = it_pending->second.penalties_used;
                        }
                        pending_step_data_.erase(it_pending);
                    }
                } else {
                    const int idx = append_step_row(tracker, actor);
                    if (idx >= 0 && idx < static_cast<int>(tracker.data.opp_target_action.size())) {
                        tracker.data.opp_target_action[idx] = static_cast<int>(entry.action);
                    }
                }
            }
            tracker.last_processed_history_len = total_len;
        }

        update_penalty_rewards(tracker, env.penalties);

        if (tracker.is_training_episode && tracker.training_agent_seat >= 0 &&
            tracker.training_agent_seat < env.num_players()) {
            if (env.terminations[tracker.training_agent_seat]) {
                mark_training_env_inactive(tracker.env_idx);
                finalize_episode(tracker);
                continue;
            }
        }

        if (arena_.done[tracker.env_idx]) {
            finalize_episode(tracker);
        }
    }
}

void RolloutManager::finalize_episode(EpisodeTracker& tracker) {
    if (tracker.done) {
        return;
    }
    tracker.done = true;

    if (tracker.env_idx < 0 || tracker.env_idx >= static_cast<int>(arena_.envs.size())) {
        return;
    }
    Env& env = arena_.envs[tracker.env_idx];

    update_penalty_rewards(tracker, env.penalties);

    auto it_pending = pending_step_data_.find(tracker.env_idx);
    if (it_pending != pending_step_data_.end()) {
        if (!tracker.data.agent_id.empty() && tracker.last_training_step_idx >= 0 &&
            tracker.last_training_step_idx < static_cast<int>(tracker.data.agent_id.size()) &&
            tracker.data.agent_id[tracker.last_training_step_idx] == tracker.training_agent_seat) {
            const int idx = tracker.last_training_step_idx;
            tracker.data.log_prob[idx] = it_pending->second.log_prob;
            tracker.data.value[idx] = it_pending->second.value;
            tracker.data.penalties_used[idx] = it_pending->second.penalties_used;
        }
        pending_step_data_.erase(it_pending);
    }

    int our_last_step_idx = -1;
    for (int i = static_cast<int>(tracker.data.agent_id.size()) - 1; i >= 0; --i) {
        if (tracker.data.agent_id[i] == tracker.training_agent_seat) {
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
    const bool is_winner = tracker.is_training_episode && tracker.training_agent_seat >= 0 &&
                           !env.terminations[tracker.training_agent_seat] &&
                           alive == 1;
    tracker.data.win = is_winner ? 1 : 0;

    if (our_last_step_idx >= 0 && our_last_step_idx < static_cast<int>(tracker.data.reward.size())) {
        tracker.data.reward[our_last_step_idx] += is_winner ? 1.0 : -1.0;
    }

    tracker.data.episode_return = std::accumulate(tracker.data.reward.begin(), tracker.data.reward.end(), 0.0);

    if (tracker.is_training_episode) {
        mark_training_env_inactive(tracker.env_idx);
        completed_buffer_.push_back(std::move(tracker.data));
        tracker.data = TrajectoryData{};
    }
}

