#include "rollout_manager.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include <ATen/core/ivalue.h>
#include <torch/nn/functional.h>
#include <torch/serialize.h>
#include <torch/torch.h>

#include "bots.h"
#include "torch_utils.h"

// --- Start of Anonymous Namespace with Helpers ---
namespace {
const torch::Device kInferenceDevice = torch::kCUDA;

std::mt19937::result_type seed_with_optional(uint32_t seed) {
    if (seed != 0) {
        return static_cast<std::mt19937::result_type>(seed);
    }
    std::random_device rd;
    return static_cast<std::mt19937::result_type>(rd());
}

// C++ Bot Wrapper Classes
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
    StrategicChallengerBot(int num_players, int seat) : bot_("bot", num_players, seat) {}
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
}  // end anonymous namespace

// --- RolloutManager Implementation ---

RolloutManager::RolloutManager() : rng_(seed_with_optional(0)) {
    if (!torch::cuda::is_available()) {
        throw std::runtime_error(
            "CUDA is not available, but the RolloutManager requires it for historical agent "
            "inference.");
    }
    training_device_ =
        torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    arena_.set_max_sequence_length(default_max_sequence_length_);
}

void RolloutManager::start_rollouts(
    int num_episodes,
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
    kv_cache_.clear();
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
        const size_t training_count =
            training_policy_ids_.empty() ? 1 : training_policy_ids_.size();
        std::vector<int> training_ids = training_policy_ids_;
        if (training_ids.empty()) {
            training_ids.push_back(training_policy_id());
        }
        for (int env_idx = 0; env_idx < batch_size_; ++env_idx) {
            std::vector<int> env_roles(num_players_, training_ids.front());
            for (size_t seat = 0; seat < training_count && seat < env_roles.size(); ++seat) {
                env_roles[seat] = training_ids[seat % training_ids.size()];
            }
            const auto& triplet =
                fixed_opponent_triplets_[env_idx % fixed_opponent_triplets_.size()];
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
                            opponent_weights);
    }
    arena_.set_roles(roles);

    episodes_.clear();
    episodes_.resize(batch_size_);
    training_env_inactive_.assign(batch_size_, 0);
    active_training_counts_.assign(batch_size_, 0);
    for (int env_idx = 0; env_idx < batch_size_; ++env_idx) {
        episodes_[env_idx] = new_episode_tracker(env_idx, roles[env_idx]);
        active_training_counts_[env_idx] =
            static_cast<int>(episodes_[env_idx].training_seats.size());
    }

    for (auto& kv : cpp_bot_registry_) {
        kv.second.instances.clear();
    }
}

bool RolloutManager::run_rollouts_step() {
    log_rewards_and_dones();

    if (!jit_module_) {
        throw std::runtime_error("RolloutManager::run_rollouts_step called without a loaded model architecture");
    }

    // Process C++ bots first
    while (true) {
        const auto& pending = arena_.collect_requests();
        if (pending.empty()) break;
        bool progressed_bot = false;
        for (const auto& kv : pending) {
            if (cpp_bot_registry_.find(kv.first) != cpp_bot_registry_.end()) {
                auto actions = run_cpp_bot(kv.first, kv.second);
                arena_.submit_actions(kv.first, actions);
                progressed_bot = true;
            }
        }
        if (!progressed_bot) break;
        log_rewards_and_dones();
    }

    const auto& pending = arena_.collect_requests();
    if (pending.empty()) {
        log_rewards_and_dones();
        return all_episodes_complete();
    }

    // Consolidate all AI requests
    std::vector<PolicyRequest> all_requests;
    std::vector<int> all_policy_ids;
    for (const auto& kv : pending) {
        if (cpp_bot_registry_.find(kv.first) == cpp_bot_registry_.end()) {
            all_requests.insert(all_requests.end(), kv.second.begin(), kv.second.end());
            all_policy_ids.insert(all_policy_ids.end(), kv.second.size(), kv.first);
        }
    }

    if (all_requests.empty()) {
        return all_episodes_complete();
    }

    auto method = jit_module_->get_method("forward_packed");

    {
        const auto& policy_ids_in_batch = all_policy_ids;
        auto batch = prepare_inference_batch(all_requests, kInferenceDevice);
        auto weights = pack_weights_for_batch(policy_ids_in_batch);

        std::vector<c10::IValue> inputs;
        inputs.reserve(7);
        inputs.emplace_back(batch.obs_sequence);
        inputs.emplace_back(batch.action_sequence);
        inputs.emplace_back(batch.agent_types);
        inputs.emplace_back(batch.positions);
        inputs.emplace_back(weights);
        inputs.emplace_back(batch.action_masks);
        inputs.emplace_back(batch.padding_mask);

        auto result = method(inputs);
        auto tuple = result.toTuple();
        auto action_logits = tuple->elements()[0].toTensor();
        auto state_values = tuple->elements()[2].toTensor();

        const int64_t B = action_logits.size(0);
        auto lengths = batch.valid_lengths.to(kInferenceDevice, false, true).to(torch::kLong).clamp_min(1);
        auto last_indices = (lengths - 1).to(kInferenceDevice);
        auto arange_opts = torch::TensorOptions().dtype(torch::kLong).device(kInferenceDevice);
        auto batch_indices = torch::arange(B, arange_opts);

        std::vector<torch::indexing::TensorIndex> indices_vec;
        indices_vec.emplace_back(batch_indices);
        indices_vec.emplace_back(last_indices);

        auto final_logits = action_logits.index(indices_vec);
        auto final_masks = batch.action_masks.index(indices_vec);
        auto values_tensor = state_values.index(indices_vec).squeeze(-1);

        auto masked_logits = final_logits.masked_fill(final_masks.logical_not(), -std::numeric_limits<float>::infinity());
        auto probs = torch::softmax(masked_logits, -1);
        probs.nan_to_num_(0.0, 0.0, 1.0);
        auto probs_sum = probs.sum(-1, true);
        probs = probs / probs_sum.clamp_min(1e-8);

        auto actions_tensor = torch::multinomial(probs, 1).squeeze(-1);
        auto log_probs_tensor = torch::log_softmax(masked_logits, -1).gather(-1, actions_tensor.unsqueeze(-1)).squeeze(-1);

        auto actions_cpu = actions_tensor.to(torch::kCPU, false, true).to(torch::kUInt8).contiguous();
        auto log_probs_cpu = log_probs_tensor.to(torch::kCPU, false, true).to(torch::kFloat32).contiguous();
        auto values_cpu = values_tensor.to(torch::kCPU, false, true).to(torch::kFloat32).contiguous();

        const auto* actions_ptr = actions_cpu.data_ptr<uint8_t>();
        const auto* log_probs_ptr = log_probs_cpu.data_ptr<float>();
        const auto* values_ptr = values_cpu.data_ptr<float>();

        std::unordered_map<int, std::vector<uint8_t>> actions_to_submit;
        std::unordered_map<int, std::vector<float>> log_probs_map;
        std::unordered_map<int, std::vector<float>> values_map;
        std::unordered_map<int, std::vector<PolicyRequest>> requests_map;

        const int64_t N = actions_cpu.size(0);
        for (int64_t i = 0; i < N; ++i) {
            const int policy_id = policy_ids_in_batch[static_cast<size_t>(i)];
            actions_to_submit[policy_id].push_back(actions_ptr[i]);
            if (is_training_policy(policy_id)) {
                log_probs_map[policy_id].push_back(log_probs_ptr[i]);
                values_map[policy_id].push_back(values_ptr[i]);
                requests_map[policy_id].push_back(all_requests[i]);
            }
        }

        for (const auto& kv : actions_to_submit) {
            int policy_id = kv.first;
            arena_.submit_actions(policy_id, kv.second);
        }

        // Loop over the map we just built, which is guaranteed to have consistent keys
        for (const auto& kv : requests_map) {
            int policy_id = kv.first;
            apply_inference_results(policy_id, kv.second, actions_to_submit.at(policy_id), log_probs_map.at(policy_id), values_map.at(policy_id));
        }
    }

    log_rewards_and_dones();
    return all_episodes_complete();
}

std::vector<TrajectoryData> RolloutManager::get_completed_episodes() {
    log_rewards_and_dones();
    for (auto& tracker : episodes_) {
        if (!tracker.done && tracker.env_idx >= 0 &&
            tracker.env_idx < static_cast<int>(arena_.done.size())) {
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

bool RolloutManager::all_episodes_complete() const {
    for (const auto& tracker : episodes_) {
        if (!tracker.done) return false;
    }
    return true;
}

std::vector<TrajectoryData> RolloutManager::get_rollouts(
    int num_episodes,
    int num_players,
    const std::vector<int>& training_policy_ids,
    int max_batch_envs,
    uint32_t seed,
    const std::vector<std::vector<int>>& opponent_triplets) {
    // Initialize and start rollouts using fixed opponent triplets
    start_rollouts(num_episodes,
                   num_players,
                   training_policy_ids,
                   max_batch_envs,
                   seed,
                   /*opponent_labels=*/{},
                   /*opponent_weights=*/{},
                   opponent_triplets);

    // Step the environment until all episodes complete
    while (!all_episodes_complete()) {
        run_rollouts_step();
    }

    // Collect and return completed episodes
    return get_completed_episodes();
}

void RolloutManager::load_model_architecture(const std::string& path) {
    try {
        auto module = std::make_shared<torch::jit::Module>(torch::jit::load(path));
        module->to(kInferenceDevice);
        module->eval();
        jit_module_ = std::move(module);
    } catch (const c10::Error& err) {
        std::cerr << "[RolloutManager] Failed to load scripted model from '" << path
                  << "': " << err.what_without_backtrace() << std::endl;
        throw;
    }
}

void RolloutManager::load_policy_weights(int policy_id, const std::string& path) {
    try {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in.is_open()) {
            throw std::runtime_error("Unable to open weights file: " + path);
        }
        std::streamsize size = in.tellg();
        if (size < 0) {
            throw std::runtime_error("Failed to stat weights file: " + path);
        }
        in.seekg(0, std::ios::beg);
        std::vector<char> buffer(static_cast<size_t>(size));
        if (!in.read(buffer.data(), size)) {
            throw std::runtime_error("Failed to read weights file: " + path);
        }

        c10::IValue iv = torch::pickle_load(buffer);
        if (!iv.isGenericDict()) {
            throw std::runtime_error("Loaded object is not a dict for path: " + path);
        }

        auto gd = iv.toGenericDict();
        std::optional<c10::impl::GenericDict> nested_holder;
        const c10::impl::GenericDict* dict_to_use = &gd;

        const auto model_state_key = c10::IValue("model_state_dict");
        auto model_state_it = gd.find(model_state_key);
        if (model_state_it != gd.end()) {
            const auto& nested_value = model_state_it->value();
            if (!nested_value.isGenericDict()) {
                throw std::runtime_error("model_state_dict entry is not a dict in weights file: " + path);
            }
            nested_holder = nested_value.toGenericDict();
            dict_to_use = &nested_holder.value();
        }

        auto out = c10::impl::GenericDict(c10::StringType::get(), c10::TensorType::get());
        out.reserve(dict_to_use->size());
        for (const auto& item : *dict_to_use) {
            const std::string key = item.key().toStringRef();
            torch::Tensor t = item.value().toTensor().detach().to(torch::kCPU).contiguous();
            out.insert(c10::IValue(key), c10::IValue(std::move(t)));
        }

        policy_weights_.erase(policy_id);
        policy_weights_.emplace(policy_id, c10::Dict<c10::IValue, c10::IValue>(out));
    } catch (const c10::Error& err) {
        std::cerr << "[RolloutManager] Failed to load state_dict from '" << path
                  << "': " << err.what_without_backtrace() << std::endl;
        throw;
    } catch (const std::exception& ex) {
        std::cerr << "[RolloutManager] Failed to load state_dict from '" << path
                  << "': " << ex.what() << std::endl;
        throw;
    }
}

void RolloutManager::update_learner_weights(int policy_id, c10::Dict<c10::IValue, c10::IValue> state_dict) {
    policy_weights_.erase(policy_id);
    policy_weights_.emplace(policy_id, std::move(state_dict));
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
    if (max_len <= 0) max_len = 1;
    default_max_sequence_length_ = max_len;
    arena_.set_max_sequence_length(max_len);
}

void RolloutManager::set_policy_max_sequence_length(int policy_id, int max_len) {
    if (policy_id < 0) return;
    if (max_len <= 0) max_len = 1;
    policy_max_sequence_length_[policy_id] = max_len;
}

void RolloutManager::set_training_device(const std::string& device_str) {
    if (device_str == "cuda" && torch::cuda::is_available()) {
        training_device_ = torch::kCUDA;
    } else {
        training_device_ = torch::kCPU;
    }
}

void RolloutManager::mark_training_env_inactive(int env_idx) {
    if (env_idx < 0 || env_idx >= batch_size_) return;
    if (training_env_inactive_[env_idx]) return;
    training_env_inactive_[env_idx] = 1;
}

bool RolloutManager::is_training_policy(int policy_id) const {
    return training_policy_id_set_.find(policy_id) != training_policy_id_set_.end();
}

void RolloutManager::apply_inference_results(int policy_id,
                                             const std::vector<PolicyRequest>& requests,
                                             const std::vector<uint8_t>& actions,
                                             const std::vector<float>& log_probs,
                                             const std::vector<float>& values) {
    if (!is_training_policy(policy_id)) return;
    
    // The requests are now passed in directly
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto& req = requests[i];
        if (req.env < 0 || req.env >= batch_size_) continue;

        EpisodeTracker& ep = episodes_[req.env];
        auto it = ep.training_seats.find(req.seat);
        if (it == ep.training_seats.end()) continue;

        SeatTrajectory& seat_tracker = it->second;
        int step_idx = append_training_step(seat_tracker);
        if (step_idx < 0) continue;

        seat_tracker.data.our_action[step_idx] = static_cast<int>(actions[i]);
        seat_tracker.data.log_prob[step_idx] = log_probs[i];
        seat_tracker.data.value[step_idx] = values[i];
    }
}

c10::Dict<c10::IValue, c10::IValue> RolloutManager::pack_weights_for_batch(
    const std::vector<int>& policy_ids) const {
    c10::Dict<c10::IValue, c10::IValue> packed(c10::StringType::get(), c10::TensorType::get());
    if (policy_ids.empty()) return packed;

    auto ref_it = policy_weights_.find(policy_ids.front());
    if (ref_it == policy_weights_.end()) {
        throw std::runtime_error("Missing weights for policy " + std::to_string(policy_ids.front()));
    }
    const auto& reference = ref_it->second;
    packed.reserve(reference.size());

    for (const auto& kv : reference) {
        const std::string key = kv.key().toStringRef();
        std::vector<torch::Tensor> tensors;
        tensors.reserve(policy_ids.size());
        c10::IValue key_ivalue(key);

        for (int policy_id : policy_ids) {
            auto weight_it = policy_weights_.find(policy_id);
            if (weight_it == policy_weights_.end()) {
                throw std::runtime_error("Missing weights for policy " + std::to_string(policy_id));
            }
            const auto& dict = weight_it->second;
            tensors.push_back(dict.at(key_ivalue).toTensor());
        }
        auto stacked = torch::stack(tensors, 0).to(kInferenceDevice, false, true);
        packed.insert(c10::IValue(key), c10::IValue(stacked));
    }
    return packed;
}

std::vector<uint8_t> RolloutManager::run_cpp_bot(int policy_id, const std::vector<PolicyRequest>& requests) {
    if (requests.empty()) return {};
    auto it = cpp_bot_registry_.find(policy_id);
    if (it == cpp_bot_registry_.end()) {
        throw std::runtime_error("No C++ bot for policy " + std::to_string(policy_id));
    }
    std::vector<uint8_t> actions;
    actions.reserve(requests.size());
    for (const auto& req : requests) {
        uint64_t key = (static_cast<uint64_t>(req.env) << 32) ^ static_cast<uint32_t>(req.seat);
        auto& entry = it->second;
        auto inst_it = entry.instances.find(key);
        if (inst_it == entry.instances.end()) {
            auto instance = make_cpp_bot_instance(entry.kind, req);
            inst_it = entry.instances.emplace(key, std::move(instance)).first;
        }
        actions.push_back(inst_it->second->act(req, arena_));
    }
    return actions;
}

RolloutManager::CppBotKind RolloutManager::parse_cpp_bot_kind(const std::string& name) {
    std::string lower;
    lower.reserve(name.size());
    for (char c : name) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower == "classic") return CppBotKind::Classic;
    if (lower == "greedycardspammer") return CppBotKind::GreedyCardSpammer;
    if (lower == "randomagent") return CppBotKind::RandomAgent;
    if (lower == "selectivetableconservativechallenger") return CppBotKind::SelectiveTableConservativeChallenger;
    if (lower == "strategicchallenger") return CppBotKind::StrategicChallenger;
    if (lower == "tablefirstconservativechallenger") return CppBotKind::TableFirstConservativeChallenger;
    if (lower == "tablenontableagent") return CppBotKind::TableNonTableAgent;
    throw std::invalid_argument("Unknown C++ bot name: " + name);
}

std::unique_ptr<CppBotBase> RolloutManager::make_cpp_bot_instance(
    RolloutManager::CppBotKind kind, const PolicyRequest& request) {
    switch (kind) {
        case CppBotKind::Classic: return std::make_unique<ClassicBot>();
        case CppBotKind::GreedyCardSpammer: return std::make_unique<GreedyCardSpammerBot>();
        case CppBotKind::RandomAgent: return std::make_unique<RandomAgentBot>();
        case CppBotKind::SelectiveTableConservativeChallenger: return std::make_unique<SelectiveTableConservativeChallengerBot>();
        case CppBotKind::StrategicChallenger: {
            int num_players = arena_.n_players;
            return std::make_unique<StrategicChallengerBot>(num_players, request.seat);
        }
        case CppBotKind::TableFirstConservativeChallenger: return std::make_unique<TableFirstConservativeChallengerBot>();
        case CppBotKind::TableNonTableAgent: return std::make_unique<TableNonTableAgentBot>();
    }
    throw std::runtime_error("Unhandled C++ bot kind");
}

std::vector<std::vector<int>> RolloutManager::build_roles(
    int batch_size,
    int num_players,
    const std::vector<int>& training_ids,
    const std::vector<int>& opponent_labels,
    const std::vector<double>& opponent_weights) {
    std::vector<std::vector<int>> roles(batch_size, std::vector<int>(num_players));
    std::discrete_distribution<> opp_dist(opponent_weights.begin(), opponent_weights.end());
    std::uniform_int_distribution<> seat_dist(0, num_players - 1);
    for (int b = 0; b < batch_size; ++b) {
        for (int p = 0; p < num_players; ++p) {
            roles[b][p] = opponent_labels[opp_dist(rng_)];
        }
        int seat = seat_dist(rng_);
        roles[b][seat] = training_ids[rng_() % training_ids.size()];
    }
    return roles;
}

EpisodeTracker RolloutManager::new_episode_tracker(int env_idx, const std::vector<int>& roles) {
    EpisodeTracker tracker;
    tracker.env_idx = env_idx;
    for (int seat = 0; seat < num_players_; ++seat) {
        if (is_training_policy(roles[seat])) {
            SeatTrajectory st;
            st.seat = seat;
            st.policy_id = roles[seat];
            st.active = true;
            st.data.env_index = env_idx;
            st.data.training_agent_seat = seat;
            st.data.training_policy_id = roles[seat];
            st.data.player_policy_ids = roles;
            tracker.training_seats[seat] = std::move(st);
        }
    }
    return tracker;
}

int RolloutManager::append_training_step(SeatTrajectory& seat_tracker) {
    int idx = static_cast<int>(seat_tracker.data.agent_id.size());
    seat_tracker.data.agent_id.push_back(seat_tracker.seat);
    seat_tracker.data.our_action.push_back(-1);
    seat_tracker.data.log_prob.push_back(0.0f);
    seat_tracker.data.value.push_back(0.0f);
    seat_tracker.data.reward.push_back(0.0);
    seat_tracker.data.opp_target_action.push_back(-100);
    seat_tracker.last_training_step_idx = idx;
    return idx;
}

int RolloutManager::append_opponent_step(SeatTrajectory& seat_tracker, int seat) {
    int idx = static_cast<int>(seat_tracker.data.agent_id.size());
    seat_tracker.data.agent_id.push_back(seat);
    seat_tracker.data.our_action.push_back(-1);
    seat_tracker.data.log_prob.push_back(0.0f);
    seat_tracker.data.value.push_back(0.0f);
    seat_tracker.data.reward.push_back(0.0);
    seat_tracker.data.opp_target_action.push_back(-100);
    return idx;
}

void RolloutManager::log_rewards_and_dones() {
    for (int b = 0; b < batch_size_; ++b) {
        EpisodeTracker& ep = episodes_[b];
        if (ep.done) continue;
        Env& env = arena_.envs[b];
        const int total_history = env.get_total_history_entries();
        if (total_history <= ep.last_history_len) continue;

        for (auto& pair : ep.training_seats) {
            SeatTrajectory& st = pair.second;
            if (!st.active) continue;
            for (int i = ep.last_history_len; i < total_history; ++i) {
                const auto& h = env.game_history[i];
                if (h.player == st.seat) {
                    if (st.last_training_step_idx >= 0) {
                        st.data.reward[st.last_training_step_idx] += (h.action == 6) ? -0.01 : 0.0;
                    }
                } else {
                    int step_idx = append_opponent_step(st, h.player);
                    st.data.opp_target_action[step_idx] = h.action;
                }
            }
        }
        ep.last_history_len = total_history;
        if (arena_.done[b]) {
            finalize_episode(ep);
        }
    }
}

void RolloutManager::finalize_episode(EpisodeTracker& tracker) {
    if (tracker.done) return;
    tracker.done = true;
    Env& env = arena_.envs[tracker.env_idx];
    int winner = -1;
    int alive_count = 0;
    for (int p = 0; p < num_players_; ++p) {
        if (env.terminations[p] == 0) {
            winner = p;
            alive_count++;
        }
    }

    for (auto& pair : tracker.training_seats) {
        SeatTrajectory& st = pair.second;
        if (st.active && st.last_training_step_idx >= 0) {
            double reward = 0.0;
            if (alive_count == 1 && winner == st.seat) {
                reward = 1.0;
                st.data.win = 1;
            } else {
                reward = -1.0;
            }
            st.data.reward[st.last_training_step_idx] += reward;
        }
        completed_buffer_.push_back(std::move(st.data));
    }
    tracker.training_seats.clear();
}
