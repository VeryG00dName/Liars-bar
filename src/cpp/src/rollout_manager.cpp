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
#include <torch/cuda.h>
#include <torch/nn/functional.h>
#include <torch/serialize.h>
#include <torch/torch.h>
#include <torch/version.h>

#include "bots.h"
#include "torch_utils.h"

namespace {
// ... (Anonymous namespace with bot wrappers remains the same) ...
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
}  // namespace

// --- RolloutManager Implementation ---

RolloutManager::RolloutManager() : rng_(seed_with_optional(0)) {
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("CUDA is not available.");
    }
    training_device_ = torch::kCPU; // Default, can be changed
    arena_.set_max_sequence_length(default_max_sequence_length_);
}

// ... (start_rollouts, run_rollouts_step, etc. are correct from the previous step, only helpers below change)
void RolloutManager::start_rollouts(
    int num_episodes,
    int num_players,
    const std::vector<int>& training_policy_ids,
    int max_batch_envs,
    uint32_t seed,
    const std::vector<int>& opponent_labels,
    const std::vector<double>& opponent_weights,
    const std::vector<std::vector<int>>& opponent_triplets) {
    target_episodes_ = num_episodes; // num_episodes from Python is the target number of games
    num_players_ = num_players;
    training_policy_ids_ = training_policy_ids;
    training_policy_id_set_.clear();
    for (int id : training_policy_ids_) {
        training_policy_id_set_.insert(id);
    }
    completed_buffer_.clear();

    if (max_batch_envs > 0) {
        batch_size_ = std::min(num_episodes, max_batch_envs);
    } else {
        batch_size_ = num_episodes;
    }
     if (batch_size_ <= 0) {
        batch_size_ = 1;
    }

    rng_.seed(seed_with_optional(seed));
    arena_.reset(batch_size_, num_players_, rng_());

    std::vector<std::vector<int>> roles(batch_size_, std::vector<int>(num_players_));
    if (!opponent_triplets.empty()) {
        for (int i = 0; i < batch_size_; ++i) {
            std::vector<int> lineup = {training_policy_ids_[0]};
            const auto& triplet = opponent_triplets[i % opponent_triplets.size()];
            lineup.insert(lineup.end(), triplet.begin(), triplet.end());
            std::shuffle(lineup.begin(), lineup.end(), rng_);
            roles[i] = lineup;
        }
    } else {
       // Fallback logic if needed, for now assume triplets are always provided
    }
    arena_.set_roles(roles);

    episodes_.clear();
    episodes_.resize(batch_size_);
    for (int i = 0; i < batch_size_; ++i) {
        episodes_[i] = new_episode_tracker(i, roles[i]);
    }

    for (auto& kv : cpp_bot_registry_) {
        kv.second.instances.clear();
    }
}

// THIS FUNCTION IS CORRECT FROM THE PREVIOUS STEP, KEEP IT AS IS
bool RolloutManager::run_rollouts_step() {
    log_rewards_and_dones();

    if (!jit_module_) {
        throw std::runtime_error("RolloutManager::run_rollouts_step called without a loaded model architecture");
    }

    // Process C++ bots first (deep-copy requests to avoid dangling references)
    while (true) {
        std::vector<std::pair<int, std::vector<PolicyRequest>>> bot_requests_to_process;
        const auto& pending = arena_.collect_requests();
        if (pending.empty()) break;

        bool has_bots = false;
        for (const auto& kv : pending) {
            if (cpp_bot_registry_.find(kv.first) != cpp_bot_registry_.end()) {
                bot_requests_to_process.emplace_back(kv.first, kv.second);
                has_bots = true;
            }
        }

        if (!has_bots) break;

        for (const auto& kv : bot_requests_to_process) {
            auto actions = run_cpp_bot(kv.first, kv.second);
            arena_.submit_actions(kv.first, actions);
        }

        log_rewards_and_dones();
    }

    const auto& pending = arena_.collect_requests();
    if (pending.empty()) {
        log_rewards_and_dones();
        return all_episodes_complete();
    }

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

    const size_t total_requests = all_requests.size();
    std::vector<uint8_t> aggregated_actions(total_requests);
    std::vector<float> aggregated_log_probs(total_requests);
    std::vector<float> aggregated_values(total_requests);

    size_t offset = 0;
    const size_t batch_limit = static_cast<size_t>(std::max(1, inference_batch_size_));

    while (offset < total_requests) {
        const size_t end = std::min(total_requests, offset + batch_limit);
        std::vector<PolicyRequest> request_chunk(all_requests.begin() + static_cast<std::ptrdiff_t>(offset),
                                                 all_requests.begin() + static_cast<std::ptrdiff_t>(end));
        std::vector<int> policy_chunk(all_policy_ids.begin() + static_cast<std::ptrdiff_t>(offset),
                                      all_policy_ids.begin() + static_cast<std::ptrdiff_t>(end));

        auto batch = prepare_inference_batch(request_chunk, kInferenceDevice);
        auto weights = pack_weights_for_batch(policy_chunk);
        
        const auto target_dtype = torch::kBFloat16;
        batch.obs_sequence = batch.obs_sequence.to(target_dtype);

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

        for (size_t i = 0; i < static_cast<size_t>(actions_cpu.size(0)); ++i) {
            const size_t dest = offset + i;
            aggregated_actions[dest] = actions_ptr[i];
            aggregated_log_probs[dest] = log_probs_ptr[i];
            aggregated_values[dest] = values_ptr[i];
        }
        offset = end;
    }

    std::unordered_map<int, std::vector<uint8_t>> actions_to_submit;
    std::unordered_map<int, std::vector<float>> log_probs_map;
    std::unordered_map<int, std::vector<float>> values_map;
    std::unordered_map<int, std::vector<PolicyRequest>> requests_map;

    for (size_t i = 0; i < total_requests; ++i) {
        const int policy_id = all_policy_ids[i];
        actions_to_submit[policy_id].push_back(aggregated_actions[i]);
        if (is_training_policy(policy_id)) {
            log_probs_map[policy_id].push_back(aggregated_log_probs[i]);
            values_map[policy_id].push_back(aggregated_values[i]);
            requests_map[policy_id].push_back(all_requests[i]);
        }
    }

    for (const auto& kv : actions_to_submit) {
        arena_.submit_actions(kv.first, kv.second);
    }

    for (const auto& kv : requests_map) {
        int policy_id = kv.first;
        apply_inference_results(policy_id,
                                kv.second,
                                actions_to_submit.at(policy_id),
                                log_probs_map.at(policy_id),
                                values_map.at(policy_id));
    }

    log_rewards_and_dones();
    return all_episodes_complete();
}


// Unified function to add a step's data to the trajectory.
// All per-step vectors are grown here to ensure they always have the same size.
void RolloutManager::append_step(SeatTrajectory& seat_tracker, int seat, int action) {
    seat_tracker.data.agent_id.push_back(seat);
    seat_tracker.data.reward.push_back(0.0);
    seat_tracker.data.opp_target_action.push_back(action); // Store the actual action taken

    // For sparse PPO data, push placeholders. They will be overwritten later if it's our turn.
    seat_tracker.data.our_action.push_back(-1);
    seat_tracker.data.log_prob.push_back(0.0f);
    seat_tracker.data.value.push_back(0.0f);
}


void RolloutManager::apply_inference_results(int policy_id,
                                             const std::vector<PolicyRequest>& requests,
                                             const std::vector<uint8_t>& actions,
                                             const std::vector<float>& log_probs,
                                             const std::vector<float>& values) {
    if (!is_training_policy(policy_id)) return;
    
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto& req = requests[i];
        if (req.env < 0 || req.env >= batch_size_) continue;

        EpisodeTracker& ep = episodes_[req.env];
        auto it = ep.training_seats.find(req.seat);
        if (it == ep.training_seats.end()) continue;

        SeatTrajectory& seat_tracker = it->second;
        
        // Overwrite placeholders in the dense vectors at the correct index
        if (seat_tracker.last_step_idx >= 0 &&
            static_cast<size_t>(seat_tracker.last_step_idx) < seat_tracker.data.value.size()) {

            seat_tracker.data.our_action[seat_tracker.last_step_idx] = static_cast<int>(actions[i]);
            seat_tracker.data.log_prob[seat_tracker.last_step_idx] = log_probs[i];
            seat_tracker.data.value[seat_tracker.last_step_idx] = values[i];
        }
    }
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

            // Process new history entries
            for (int i = ep.last_history_len; i < total_history; ++i) {
                const auto& h = env.game_history[i];
                append_step(st, h.player, h.action);

                // If this step was our agent's turn, record its index in the dense vectors
                // so `apply_inference_results` knows where to write the model output.
                if (h.player == st.seat) {
                    st.last_step_idx = static_cast<int>(st.data.agent_id.size()) - 1;
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
        if (st.active) {
            // Apply win/loss reward to the very last step in the dense trajectory
            if (!st.data.reward.empty()) {
                if (alive_count == 1 && winner == st.seat) {
                    st.data.reward.back() += 1.0;
                    st.data.win = 1;
                } else {
                    st.data.reward.back() += -1.0;
                }
            }
           
            // Prepare the final model inputs for Python
            PolicyRequest final_request;
            int seq_cap = arena_.max_sequence_length_for_policy(st.policy_id);
            arena_.prepare_ai_sequence(env, st.seat, seq_cap, final_request);
            
            st.data.valid_len = final_request.valid_len;
            st.data.obs_sequence = std::move(final_request.obs_sequence);
            st.data.action_sequence = std::move(final_request.action_sequence);
            st.data.agent_type_sequence = std::move(final_request.agent_type_sequence);
            st.data.position_sequence = std::move(final_request.position_sequence);
            st.data.action_mask_sequence = std::move(final_request.action_mask_sequence);
        }

        completed_buffer_.push_back(std::move(st.data));
    }
    tracker.training_seats.clear();
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
    start_rollouts(num_episodes,
                   num_players,
                   training_policy_ids,
                   max_batch_envs,
                   seed,
                   {},
                   {},
                   opponent_triplets);

    while (!all_episodes_complete()) {
        run_rollouts_step();
    }

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
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Unable to open weights file: " + path);
        }
        std::vector<char> buffer((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

        c10::IValue iv = torch::pickle_load(buffer);

        const c10::impl::GenericDict* dict_to_use;
        std::optional<c10::impl::GenericDict> nested_holder;

        if (iv.isGenericDict()) {
            auto gd = iv.toGenericDict();
            auto model_state_it = gd.find(c10::IValue("model_state_dict"));
            if (model_state_it != gd.end() && model_state_it->value().isGenericDict()) {
                nested_holder = model_state_it->value().toGenericDict();
                dict_to_use = &nested_holder.value();
            } else {
                dict_to_use = &gd;
            }
        } else {
            throw std::runtime_error("Loaded object is not a dict for path: " + path);
        }

        auto out = c10::impl::GenericDict(c10::StringType::get(), c10::TensorType::get());
        out.reserve(dict_to_use->size());
        for (const auto& item : *dict_to_use) {
            const std::string key = item.key().toStringRef();
            torch::Tensor t = item.value().toTensor().detach().to(torch::kCPU).contiguous();
            out.insert(c10::IValue(key), c10::IValue(std::move(t)));
        }

        policy_weights_[policy_id] = c10::Dict<c10::IValue, c10::IValue>(out);
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
    policy_weights_[policy_id] = std::move(state_dict);
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
void RolloutManager::set_inference_batch_size(int size) {
    if (size <= 0) {
        throw std::invalid_argument("inference_batch_size must be positive");
    }
    inference_batch_size_ = size;
}
void RolloutManager::set_training_device(const std::string& device_str) {
    if (device_str == "cuda" && torch::cuda::is_available()) {
        training_device_ = torch::kCUDA;
    } else {
        training_device_ = torch::kCPU;
    }
}
bool RolloutManager::is_training_policy(int policy_id) const {
    return training_policy_id_set_.count(policy_id) > 0;
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

    const auto target_dtype = torch::kBFloat16;

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
        auto stacked = torch::stack(tensors, 0);
        if (stacked.is_floating_point()) {
            stacked = stacked.to(target_dtype);
        }
        stacked = stacked.to(kInferenceDevice, false, true);
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