#include "eval_manager.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

#include <torch/torch.h>

#include "bots.h"
#include "torch_utils.h"

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
    : max_env_batch_(512), inference_batch_size_(128), rng_(seed_with_optional(0)) {
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

void EvalManager::set_inference_batch_size(int batch_size) {
    if (batch_size <= 0) {
        throw std::invalid_argument("inference_batch_size must be positive");
    }
    inference_batch_size_ = batch_size;
}

void EvalManager::load_model(int policy_id, const std::string& path) {
    try {
        auto module = std::make_shared<torch::jit::Module>(torch::jit::load(path));
        module->to(kInferenceDevice);
        module->eval();

        torch::NoGradGuard guard;
        auto params = module->named_parameters(/*recurse=*/true);
        std::unordered_map<std::string, torch::Tensor> state_dict;
        state_dict.reserve(params.size());
        for (const auto& item : params) {
            auto tensor = item.value
                              .detach()
                              .to(kInferenceDevice, /*non_blocking=*/false, /*copy=*/true)
                              .to(torch::kFloat16)
                              .contiguous();
            state_dict.emplace(item.name, std::move(tensor));
        }

        staged_state_dicts_[policy_id] = std::move(state_dict);
        if (!representative_module_) {
            representative_module_ = std::move(module);
        }

        int max_seq_length = infer_max_seq_length_for_model(path);
        policy_max_sequence_lengths_[policy_id] = max_seq_length;
        arena_.set_policy_max_sequence_length(policy_id, max_seq_length);
        weights_finalized_ = false;
    } catch (const c10::Error& err) {
        std::cerr << "[EvalManager] Failed to load TorchScript module from '" << path
                  << "': " << err.what_without_backtrace() << std::endl;
        throw;
    } catch (const std::exception& err) {
        std::cerr << "[EvalManager] Failed to load TorchScript module from '" << path
                  << "': " << err.what() << std::endl;
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

    if (!representative_module_) {
        throw std::runtime_error(
            "No representative TorchScript module loaded before finalizing weights");
    }

    std::vector<int> policy_ids;
    policy_ids.reserve(staged_state_dicts_.size());
    for (const auto& kv : staged_state_dicts_) {
        policy_ids.push_back(kv.first);
    }
    std::sort(policy_ids.begin(), policy_ids.end());

    std::unordered_map<std::string, std::vector<torch::Tensor>> stacked;
    for (size_t idx = 0; idx < policy_ids.size(); ++idx) {
        int policy_id = policy_ids[idx];
        policy_id_to_cache_index_[policy_id] = static_cast<int>(idx);
        auto dict_it = staged_state_dicts_.find(policy_id);
        if (dict_it == staged_state_dicts_.end()) {
            throw std::runtime_error("Missing staged weights for policy " + std::to_string(policy_id));
        }
        for (const auto& kv : dict_it->second) {
            stacked[kv.first].push_back(kv.second);
        }
    }

    batched_weight_cache_.reserve(stacked.size());
    for (auto& kv : stacked) {
        batched_weight_cache_[kv.first] = torch::stack(kv.second).contiguous();
    }

    // --- START: NEW Pre-stacking Logic ---
    // Discover number of layers and experts by scanning keys
    int num_layers = 0;
    int num_experts = 0;
    for (const auto& kv : batched_weight_cache_) {
        const std::string& key = kv.first;
        const size_t pos_layer = key.find("transformer.layers.");
        const size_t pos_moe = key.find(".moe.experts.");
        if (pos_layer == std::string::npos || pos_moe == std::string::npos) continue;

        // parse layer index
        size_t layer_start = pos_layer + std::strlen("transformer.layers.");
        size_t layer_end = key.find('.', layer_start);
        if (layer_end == std::string::npos) continue;
        int layer_idx = 0;
        try {
            layer_idx = std::stoi(key.substr(layer_start, layer_end - layer_start));
        } catch (...) {
            continue;
        }
        num_layers = std::max(num_layers, layer_idx + 1);

        // parse expert index
        size_t expert_start = pos_moe + std::strlen(".moe.experts.");
        size_t expert_end = key.find('.', expert_start);
        if (expert_end == std::string::npos) continue;
        int expert_idx = 0;
        try {
            expert_idx = std::stoi(key.substr(expert_start, expert_end - expert_start));
        } catch (...) {
            continue;
        }
        num_experts = std::max(num_experts, expert_idx + 1);
    }

    // Stack expert weights into single tensors per layer: [E, B, F, H] / [E, B, H, F]
    if (num_layers > 0 && num_experts > 0) {
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            std::vector<torch::Tensor> w1_list, b1_list, w2_list, b2_list;
            w1_list.reserve(num_experts);
            b1_list.reserve(num_experts);
            w2_list.reserve(num_experts);
            b2_list.reserve(num_experts);

            bool have_all = true;
            for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
                const std::string base =
                    "transformer.layers." + std::to_string(layer_idx) + ".moe.experts." + std::to_string(expert_idx);
                const std::string w1_key = base + ".0.weight";
                const std::string b1_key = base + ".0.bias";
                const std::string w2_key = base + ".3.weight";
                const std::string b2_key = base + ".3.bias";

                auto it_w1 = batched_weight_cache_.find(w1_key);
                auto it_b1 = batched_weight_cache_.find(b1_key);
                auto it_w2 = batched_weight_cache_.find(w2_key);
                auto it_b2 = batched_weight_cache_.find(b2_key);
                if (it_w1 == batched_weight_cache_.end() || it_b1 == batched_weight_cache_.end() ||
                    it_w2 == batched_weight_cache_.end() || it_b2 == batched_weight_cache_.end()) {
                    have_all = false;
                    break;
                }
                w1_list.push_back(it_w1->second);
                b1_list.push_back(it_b1->second);
                w2_list.push_back(it_w2->second);
                b2_list.push_back(it_b2->second);
            }

            if (have_all && !w1_list.empty()) {
                const std::string prefix =
                    "transformer.layers." + std::to_string(layer_idx) + ".moe.experts.";
                batched_weight_cache_[prefix + "w1"] = torch::stack(w1_list, 0).contiguous();
                batched_weight_cache_[prefix + "b1"] = torch::stack(b1_list, 0).contiguous();
                batched_weight_cache_[prefix + "w2"] = torch::stack(w2_list, 0).contiguous();
                batched_weight_cache_[prefix + "b2"] = torch::stack(b2_list, 0).contiguous();
            }
        }
    }
    // --- END: NEW Pre-stacking Logic ---

    std::vector<std::string> original_keys;
    original_keys.reserve(batched_weight_cache_.size());
    for (const auto& kv : batched_weight_cache_) {
        original_keys.push_back(kv.first);
    }

    using torch::indexing::Slice;
    auto replace_suffix = [](const std::string& s, const std::string& from, const std::string& to) {
        const auto pos = s.rfind(from);
        if (pos == std::string::npos) {
            return s;
        }
        std::string out = s;
        out.replace(pos, from.size(), to);
        return out;
    };

    auto drop_self_attn = [](const std::string& s) {
        constexpr const char* needle = ".self_attn.";
        const auto pos = s.find(needle);
        if (pos == std::string::npos) {
            return s;
        }
        std::string out = s;
        out.replace(pos, std::strlen(needle), ".");
        return out;
    };

    for (const auto& key : original_keys) {
        if (key.find(".self_attn.in_proj_weight") != std::string::npos) {
            auto it = batched_weight_cache_.find(key);
            if (it == batched_weight_cache_.end()) {
                continue;
            }
            const auto& w = it->second;
            if (w.dim() != 3 || w.size(1) % 3 != 0) {
                continue;
            }
            const int64_t H = w.size(1) / 3;
            auto q = w.index({Slice(), Slice(0, H), Slice()}).contiguous();
            auto k = w.index({Slice(), Slice(H, 2 * H), Slice()}).contiguous();
            auto v = w.index({Slice(), Slice(2 * H, 3 * H), Slice()}).contiguous();
            const auto q_key = replace_suffix(key, "in_proj_weight", "q_proj.weight");
            const auto k_key = replace_suffix(key, "in_proj_weight", "k_proj.weight");
            const auto v_key = replace_suffix(key, "in_proj_weight", "v_proj.weight");

            batched_weight_cache_[q_key] = q;
            batched_weight_cache_[k_key] = k;
            batched_weight_cache_[v_key] = v;

            batched_weight_cache_[drop_self_attn(q_key)] = q;
            batched_weight_cache_[drop_self_attn(k_key)] = k;
            batched_weight_cache_[drop_self_attn(v_key)] = v;
            continue;
        }

        if (key.find(".self_attn.in_proj_bias") != std::string::npos) {
            auto it = batched_weight_cache_.find(key);
            if (it == batched_weight_cache_.end()) {
                continue;
            }
            const auto& b = it->second;
            if (b.dim() != 2 || b.size(1) % 3 != 0) {
                continue;
            }
            const int64_t H = b.size(1) / 3;
            auto q = b.index({Slice(), Slice(0, H)}).contiguous();
            auto k = b.index({Slice(), Slice(H, 2 * H)}).contiguous();
            auto v = b.index({Slice(), Slice(2 * H, 3 * H)}).contiguous();
            const auto q_key = replace_suffix(key, "in_proj_bias", "q_proj.bias");
            const auto k_key = replace_suffix(key, "in_proj_bias", "k_proj.bias");
            const auto v_key = replace_suffix(key, "in_proj_bias", "v_proj.bias");

            batched_weight_cache_[q_key] = q;
            batched_weight_cache_[k_key] = k;
            batched_weight_cache_[v_key] = v;

            batched_weight_cache_[drop_self_attn(q_key)] = q;
            batched_weight_cache_[drop_self_attn(k_key)] = k;
            batched_weight_cache_[drop_self_attn(v_key)] = v;
            continue;
        }

        if (key.find(".self_attn.") != std::string::npos) {
            const auto alias = drop_self_attn(key);
            if (alias != key && !batched_weight_cache_.count(alias)) {
                batched_weight_cache_[alias] = batched_weight_cache_.at(key);
            }
        }
    }

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

        std::vector<RequestRef> historical_refs;
        historical_refs.reserve(pending.size() * 4); // Pre-allocate with a reasonable guess

        for (const auto& kv : pending) {
            int policy_id = kv.first;
            const auto& requests = kv.second;

            // Check if it's a C++ bot
            if (cpp_bot_registry_.count(policy_id)) {
                actions_by_policy[policy_id] = run_cpp_bot(policy_id, requests);
                continue;
            }

            // If not a C++ bot, it must be a historical model
            if (policy_id_to_cache_index_.find(policy_id) == policy_id_to_cache_index_.end()) {
                throw std::runtime_error("No registered model for policy " + std::to_string(policy_id));
            }
            
            // Add to the batch for historical inference
            actions_by_policy.emplace(policy_id, std::vector<uint8_t>(requests.size()));
            for (size_t i = 0; i < requests.size(); ++i) {
                historical_refs.push_back({policy_id, i, &requests[i]});
            }
        }

        auto model_start = Clock::now();
        if (!historical_refs.empty()) {
            if (!weights_finalized_) {
                finalize_model_loading();
            }
            run_packed_historical_inference(historical_refs, actions_by_policy);
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

void EvalManager::run_packed_historical_inference(const std::vector<RequestRef>& refs,
                                                  std::unordered_map<int, std::vector<uint8_t>>& out_actions) {
    if (refs.empty()) {
        return;
    }

    using Clock = std::chrono::high_resolution_clock;
    using Microseconds = std::chrono::microseconds;

    if (!representative_module_) {
        throw std::runtime_error("No representative module available for inference");
    }
    auto method = representative_module_->find_method("forward_packed");
    if (!method) {
        throw std::runtime_error("Representative module does not expose forward_packed");
    }
    if (batched_weight_cache_.empty()) {
        throw std::runtime_error("Historical weight cache is empty");
    }

    torch::NoGradGuard no_grad;

    auto opts_long_device = torch::TensorOptions().dtype(torch::kInt64).device(kInferenceDevice);
    const size_t batch_limit = static_cast<size_t>(std::max(1, inference_batch_size_));

    size_t offset = 0;
    while (offset < refs.size()) {
        size_t remaining = refs.size() - offset;
        size_t take = std::min(remaining, batch_limit);

        auto prep_t0 = Clock::now();
        std::vector<PolicyRequest> batch_requests;
        batch_requests.reserve(take);
        std::vector<const RequestRef*> batch_refs;
        batch_refs.reserve(take);
        std::vector<int64_t> cache_indices;
        cache_indices.reserve(take);

        for (size_t i = 0; i < take; ++i) {
            const RequestRef& ref = refs[offset + i];
            batch_requests.push_back(*ref.request);
            batch_refs.push_back(&ref);
            auto cache_it = policy_id_to_cache_index_.find(ref.policy_id);
            if (cache_it == policy_id_to_cache_index_.end()) {
                throw std::runtime_error("Missing cache index for policy " + std::to_string(ref.policy_id));
            }
            cache_indices.push_back(static_cast<int64_t>(cache_it->second));
        }

        auto tensor_batch = prepare_inference_batch(batch_requests, kInferenceDevice);
        if (!tensor_batch.obs_sequence.defined()) {
            auto prep_t1 = Clock::now();
            timer_hist_prep_batch_ +=
                std::chrono::duration_cast<Microseconds>(prep_t1 - prep_t0);
            offset += take;
            continue;
        }

        if (tensor_batch.obs_sequence.dtype() != torch::kFloat16) {
            tensor_batch.obs_sequence = tensor_batch.obs_sequence.to(torch::kFloat16);
        }

        auto prep_t1 = Clock::now();
        timer_hist_prep_batch_ += std::chrono::duration_cast<Microseconds>(prep_t1 - prep_t0);

        auto weights_t0 = Clock::now();
        auto policy_index_tensor = torch::tensor(cache_indices, opts_long_device);

        // Select the correct weights per-request into a string->Tensor dict
        c10::Dict<std::string, torch::Tensor> weight_dict;
        weight_dict.reserve(batched_weight_cache_.size());
        for (const auto& kv : batched_weight_cache_) {
            const std::string& k = kv.first;
            // For pre-stacked MoE expert tensors with shape [E, B, ...],
            // select along policy dimension (dim=1). For all others, select along dim=0.
            bool is_moe_expert =
                (k.find(".moe.experts.w1") != std::string::npos) ||
                (k.find(".moe.experts.b1") != std::string::npos) ||
                (k.find(".moe.experts.w2") != std::string::npos) ||
                (k.find(".moe.experts.b2") != std::string::npos);
            torch::Tensor selected;
            if (is_moe_expert) {
                selected = kv.second.index_select(1, policy_index_tensor).contiguous();
            } else {
                // kv.second shape: [num_policies, ...]; select rows for this batch
                selected = kv.second.index_select(0, policy_index_tensor).contiguous();
            }
            weight_dict.insert(kv.first, selected);
        }

        auto weights_t1 = Clock::now();
        timer_hist_prep_weights_ +=
            std::chrono::duration_cast<Microseconds>(weights_t1 - weights_t0);

        auto model_t0 = Clock::now();
        std::vector<torch::jit::IValue> inputs;
        inputs.reserve(7);
        inputs.emplace_back(tensor_batch.obs_sequence);
        inputs.emplace_back(tensor_batch.action_sequence);
        inputs.emplace_back(tensor_batch.agent_types);
        inputs.emplace_back(tensor_batch.positions);
        inputs.emplace_back(weight_dict);
        inputs.emplace_back(tensor_batch.action_masks);
        inputs.emplace_back(tensor_batch.padding_mask);

        auto outputs_iv = method->operator()(inputs);
        torch::cuda::synchronize();
        auto model_t1 = Clock::now();
        timer_hist_model_exec_ +=
            std::chrono::duration_cast<Microseconds>(model_t1 - model_t0);

        auto post_t0 = Clock::now();
        auto outputs = outputs_iv.toTuple();
        if (!outputs || outputs->elements().empty()) {
            throw std::runtime_error("forward_packed returned unexpected output");
        }

        auto action_logits = outputs->elements()[0].toTensor().contiguous();
        auto valid_lengths_device = tensor_batch.valid_lengths;

        const int64_t batch_size = static_cast<int64_t>(take);
        auto batch_indices = torch::arange(batch_size, opts_long_device);
        auto last_indices = (valid_lengths_device - 1).clamp_min(0);
        auto last_logits = action_logits.index({batch_indices, last_indices}).contiguous();
        auto last_masks = tensor_batch.action_masks.index({batch_indices, last_indices}).contiguous();

        auto has_legal = last_masks.any(1);
        if (!has_legal.all().item<bool>()) {
            auto fallback_indices = has_legal.logical_not().nonzero().flatten();
            for (int64_t i = 0; i < fallback_indices.size(0); ++i) {
                const int64_t row = fallback_indices[i].item<int64_t>();
                auto mask_row_tensor = last_masks.select(0, row);
                mask_row_tensor.fill_(false);
                bool assigned = false;
                const auto& ref = *batch_refs[static_cast<size_t>(row)];
                const auto& req = *ref.request;
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
        auto probs = torch::softmax(last_logits.to(torch::kFloat32), /*dim=*/1);
        auto actions_tensor = torch::multinomial(probs, /*num_samples=*/1);
        actions_tensor = actions_tensor.squeeze(-1).to(torch::kCPU);

        auto actions_ptr = actions_tensor.data_ptr<int64_t>();
        for (size_t i = 0; i < take; ++i) {
            const auto& ref = *batch_refs[i];
            auto out_it = out_actions.find(ref.policy_id);
            if (out_it == out_actions.end()) {
                throw std::runtime_error("Missing output buffer for policy " + std::to_string(ref.policy_id));
            }
            if (ref.request_index >= out_it->second.size()) {
                out_it->second.resize(ref.request_index + 1);
            }
            out_it->second[ref.request_index] = static_cast<uint8_t>(actions_ptr[i]);
        }

        auto post_t1 = Clock::now();
        timer_hist_post_ += std::chrono::duration_cast<Microseconds>(post_t1 - post_t0);

        offset += take;
    }
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
