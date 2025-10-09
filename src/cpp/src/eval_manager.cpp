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
#include <torch/nn/functional.h>
#include <ATen/core/ivalue.h>

#include "bots.h"
#include "torch_utils.h"

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
        models_[policy_id] = std::move(module);
        int max_seq_length = infer_max_seq_length_for_model(path);
        policy_max_sequence_lengths_[policy_id] = max_seq_length;
        arena_.set_policy_max_sequence_length(policy_id, max_seq_length);
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

        const auto& pending = arena_.collect_requests();
        if (pending.empty()) {
            if (completed_games == static_cast<int>(total_games)) {
                break;
            }
            throw std::runtime_error("Simulation deadlock: no pending AI requests but games remain active");
        }

        int priority_model_id = -1;
        size_t max_requests = 0;
        for (const auto& kv : pending) {
            int policy_id = kv.first;
            if (kv.second.size() > max_requests) {
                if (models_.find(policy_id) == models_.end()) {
                    throw std::runtime_error("No registered model for policy " + std::to_string(policy_id));
                }
                priority_model_id = policy_id;
                max_requests = kv.second.size();
            }
        }

        if (priority_model_id < 0) {
            throw std::runtime_error("No eligible model found to service pending requests");
        }

        auto model_it = models_.find(priority_model_id);
        if (model_it == models_.end()) {
            throw std::runtime_error("No registered model for policy " + std::to_string(priority_model_id));
        }

        auto pending_it = pending.find(priority_model_id);
        std::vector<PolicyRequest> priority_requests = pending_it->second;

        std::vector<uint8_t> aggregated_actions;
        aggregated_actions.reserve(priority_requests.size());

        size_t offset = 0;
        const size_t batch_limit = static_cast<size_t>(std::max(1, inference_batch_size_));
        while (offset < priority_requests.size()) {
            size_t remaining = priority_requests.size() - offset;
            size_t take = std::min(remaining, batch_limit);
            std::vector<PolicyRequest> chunk(priority_requests.begin() + static_cast<std::ptrdiff_t>(offset),
                                             priority_requests.begin() + static_cast<std::ptrdiff_t>(offset + take));
            auto actions = run_model(priority_model_id, *model_it->second, chunk);
            aggregated_actions.insert(aggregated_actions.end(), actions.begin(), actions.end());
            offset += take;
        }

        arena_.submit_actions(priority_model_id, aggregated_actions);
        for (const auto& req : priority_requests) {
            int env_idx = req.env;
            if (env_idx >= 0 && env_idx < arena_.B) {
                if (arena_.done[env_idx] && !env_completed[static_cast<size_t>(env_idx)]) {
                    env_completed[static_cast<size_t>(env_idx)] = 1;
                    ++completed_games;
                }
            }
        }
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

    return outcome;
}

std::vector<uint8_t> EvalManager::run_model(int policy_id,
                                            torch::jit::Module& module,
                                            const std::vector<PolicyRequest>& requests) {
    if (requests.empty()) {
        return {};
    }

    torch::NoGradGuard no_grad;
    auto& cache_map = kv_cache_[policy_id];

    std::vector<uint8_t> chosen(requests.size());

    std::vector<size_t> without_cache;
    std::vector<size_t> with_cache;
    std::vector<CacheEntry> cached_entries;
    without_cache.reserve(requests.size());
    with_cache.reserve(requests.size());
    cached_entries.reserve(requests.size());

    for (size_t idx = 0; idx < requests.size(); ++idx) {
        const auto& req = requests[idx];
        if (req.env < 0 || req.seat < 0) {
            without_cache.push_back(idx);
            continue;
        }
        uint64_t key = seat_cache_key(req.env, req.seat);
        if (req.valid_len <= 1) {
            cache_map.erase(key);
        }
        auto it = cache_map.find(key);
        if (it == cache_map.end()) {
            without_cache.push_back(idx);
        } else {
            with_cache.push_back(idx);
            cached_entries.push_back(it->second);
        }
    }

    auto process_group = [&](const std::vector<size_t>& indices,
                             const std::vector<CacheEntry>* cache_list) {
        if (indices.empty()) {
            return c10::IValue();
        }

        int64_t pad_len = 1;
        for (size_t idx : indices) {
            pad_len = std::max<int64_t>(pad_len, std::max<int64_t>(1, requests[idx].valid_len));
        }

        auto batch = prepare_inference_batch(requests, pad_len, kInferenceDevice, indices);

        std::vector<c10::IValue> inputs;
        inputs.reserve(8);
        inputs.emplace_back(batch.obs_sequence);
        inputs.emplace_back(batch.action_sequence);
        inputs.emplace_back(batch.agent_types);
        inputs.emplace_back(batch.positions);
        inputs.emplace_back(batch.action_masks);
        inputs.emplace_back(batch.padding_mask);
        inputs.emplace_back(batch.valid_lengths);
        if (cache_list && !cache_list->empty()) {
            inputs.emplace_back(stack_kv_cache(*cache_list, kInferenceDevice));
        } else {
            inputs.emplace_back(c10::IValue());
        }

        auto outputs = module.get_method("forward_with_kv_cache")(inputs).toTuple();
        if (!outputs || outputs->elements().size() < 5) {
            throw std::runtime_error("TorchScript model returned unexpected output");
        }

        auto action_logits = outputs->elements()[0].toTensor().contiguous();
        auto new_cache = outputs->elements()[4];

        auto opts_long_device =
            torch::TensorOptions().dtype(torch::kInt64).device(kInferenceDevice);
        auto group_size = static_cast<int64_t>(indices.size());
        auto batch_indices = torch::arange(group_size, opts_long_device);
        auto last_indices = (batch.valid_lengths - 1).clamp_min(0);
        auto last_logits = action_logits.index({batch_indices, last_indices}).contiguous();
        auto last_masks = batch.action_masks.index({batch_indices, last_indices}).contiguous();

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
        for (int64_t b = 0; b < group_size; ++b) {
            chosen[indices[static_cast<size_t>(b)]] = static_cast<uint8_t>(actions_ptr[b]);
        }

        return new_cache;
    };

    auto cache_none = process_group(without_cache, nullptr);
    if (!without_cache.empty()) {
        update_kv_cache(policy_id, without_cache, requests, cache_none);
    }

    if (!with_cache.empty()) {
        auto cache_value = process_group(with_cache, &cached_entries);
        update_kv_cache(policy_id, with_cache, requests, cache_value);
    }

    return chosen;
}

uint64_t EvalManager::seat_cache_key(int env, int seat) {
    uint64_t env_part = static_cast<uint64_t>(static_cast<uint32_t>(env));
    uint64_t seat_part = static_cast<uint64_t>(static_cast<uint32_t>(seat));
    return (env_part << 32) | seat_part;
}

c10::IValue EvalManager::stack_kv_cache(const std::vector<CacheEntry>& caches,
                                        const torch::Device& device) {
    if (caches.empty()) {
        return c10::IValue();
    }

    size_t num_layers = caches.front().size();
    auto stacked = c10::impl::GenericList(c10::AnyType::get());
    stacked.reserve(num_layers);

    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        std::vector<torch::Tensor> keys;
        std::vector<torch::Tensor> values;
        std::vector<torch::Tensor> lengths;
        int64_t max_len = 0;
        for (const auto& cache : caches) {
            const auto& dict_ivalue = cache[layer_idx];
            auto dict = dict_ivalue.toGenericDict();
            auto key_tensor = dict.at("key").toTensor().to(device);
            auto value_tensor = dict.at("value").toTensor().to(device);
            auto length_tensor = dict.at("lengths").toTensor().to(device);
            max_len = std::max<int64_t>(max_len, key_tensor.size(2));
            keys.push_back(key_tensor);
            values.push_back(value_tensor);
            lengths.push_back(length_tensor);
        }

        std::vector<torch::Tensor> padded_keys;
        std::vector<torch::Tensor> padded_values;
        padded_keys.reserve(keys.size());
        padded_values.reserve(values.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            auto key_tensor = keys[i];
            auto value_tensor = values[i];
            int64_t pad_len = max_len - key_tensor.size(2);
            if (pad_len > 0) {
                key_tensor = torch::nn::functional::pad(
                    key_tensor, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_len}));
                value_tensor = torch::nn::functional::pad(
                    value_tensor, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_len}));
            }
            padded_keys.push_back(key_tensor);
            padded_values.push_back(value_tensor);
        }

        auto stacked_key = torch::cat(padded_keys, 0);
        auto stacked_value = torch::cat(padded_values, 0);
        auto stacked_lengths = torch::cat(lengths, 0);

        c10::Dict<std::string, torch::Tensor> dict;
        dict.insert("key", stacked_key);
        dict.insert("value", stacked_value);
        dict.insert("lengths", stacked_lengths);
        stacked.push_back(c10::IValue(dict));
    }

    return c10::IValue(std::move(stacked));
}

void EvalManager::update_kv_cache(int policy_id,
                                  const std::vector<size_t>& indices,
                                  const std::vector<PolicyRequest>& requests,
                                  const c10::IValue& cache_ivalue) {
    if (indices.empty() || cache_ivalue.isNone()) {
        return;
    }

    auto cache_list = cache_ivalue.toList();
    if (cache_list.size() == 0) {
        return;
    }

    auto& cache_map = kv_cache_[policy_id];
    int max_allowed = 480;
    auto it_limit = policy_max_sequence_lengths_.find(policy_id);
    if (it_limit != policy_max_sequence_lengths_.end()) {
        max_allowed = std::max(1, it_limit->second);
    }

    size_t num_layers = cache_list.size();
    auto lengths_tensor = cache_list.get(0).toGenericDict().at("lengths").toTensor();
    int64_t batch_size = lengths_tensor.size(0);

    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        if (static_cast<size_t>(batch_idx) >= indices.size()) {
            break;
        }
        size_t request_idx = indices[static_cast<size_t>(batch_idx)];
        const auto& req = requests[request_idx];
        if (req.env < 0 || req.seat < 0) {
            continue;
        }
        uint64_t key = seat_cache_key(req.env, req.seat);
        CacheEntry entry;
        entry.reserve(num_layers);

        for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            auto dict = cache_list.get(layer_idx).toGenericDict();
            auto key_tensor =
                dict.at("key").toTensor().select(0, batch_idx).unsqueeze(0).detach().clone();
            auto value_tensor =
                dict.at("value").toTensor().select(0, batch_idx).unsqueeze(0).detach().clone();
            auto length_tensor =
                dict.at("lengths").toTensor().select(0, batch_idx).unsqueeze(0).detach().clone();

            int64_t seq_len = length_tensor.item<int64_t>();
            seq_len = std::max<int64_t>(seq_len, 0);
            int64_t target_len = std::min<int64_t>(seq_len, max_allowed);
            int64_t start = seq_len - target_len;
            if (start < 0) {
                start = 0;
            }

            if (target_len < key_tensor.size(2)) {
                key_tensor = key_tensor.narrow(2, start, target_len).contiguous();
                value_tensor = value_tensor.narrow(2, start, target_len).contiguous();
            }

            key_tensor = key_tensor.contiguous();
            value_tensor = value_tensor.contiguous();
            length_tensor = length_tensor.contiguous();
            length_tensor.fill_(target_len);

            c10::Dict<std::string, torch::Tensor> layer_dict;
            layer_dict.insert("key", key_tensor);
            layer_dict.insert("value", value_tensor);
            layer_dict.insert("lengths", length_tensor);
            entry.push_back(c10::IValue(layer_dict));
        }

        cache_map[key] = std::move(entry);
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

EvalManager::CppBotKind EvalManager::parse_cpp_bot_kind(const std::string& name) {
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

