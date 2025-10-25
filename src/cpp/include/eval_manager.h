#pragma once

#include "vec_arena.h"
#include "reactive_model_forward.h"
#include "weight_utils.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

class CppBotBase {
public:
    virtual ~CppBotBase() = default;
    virtual uint8_t act(const PolicyRequest& request, VecArena& arena) = 0;
};

struct EvalAgentStats {
    int total_wins{0};
    double total_returns{0.0};
    int num_games{0};
};

struct EvalLineupResult {
    std::unordered_map<int, EvalAgentStats> per_policy;
};

struct PairKeyHash {
    std::size_t operator()(const std::array<int, 2>& key) const noexcept {
        std::size_t h1 = std::hash<int>{}(key[0]);
        std::size_t h2 = std::hash<int>{}(key[1]);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

struct EvalOutcome {
    std::vector<EvalLineupResult> lineups;
    std::unordered_map<std::array<int, 2>, int, PairKeyHash> h2h_counts;
    std::unordered_map<std::array<int, 2>, int, PairKeyHash> h2h_wins;
    int total_games{0};
};

class EvalManager {
public:
    EvalManager();

    void set_max_env_batch(int max_batch);
    void set_max_inference_batch_size(size_t max_batch_size);

    void load_model(
        int policy_id,
        const std::unordered_map<std::string, torch::Tensor>& state_dict,
        const std::string& original_path);
    void finalize_model_loading();
    void register_cpp_bot(int policy_id, const std::string& bot_name);

    EvalOutcome run_roles(const std::vector<std::vector<int>>& roles,
                          const std::vector<int>& lineup_indices,
                          int num_players,
                          uint32_t seed);

    std::unordered_map<std::string, int64_t> get_last_performance_stats() const;

private:
    enum class CppBotKind {
        Classic,
        GreedyCardSpammer,
        RandomAgent,
        SelectiveTableConservativeChallenger,
        StrategicChallenger,
        TableFirstConservativeChallenger,
        TableNonTableAgent,
    };

    struct CppBotRegistryEntry {
        CppBotKind kind{CppBotKind::Classic};
        std::unordered_map<uint64_t, std::unique_ptr<CppBotBase>> instances;
    };

    VecArena arena_;
    int max_env_batch_{512};
    size_t max_inference_batch_size_{128};
    std::mt19937 rng_;

    // Model architecture parameters (inferred from first loaded model)
    int64_t num_layers_{2};
    int64_t num_heads_{4};
    int64_t hidden_dim_{256};
    int64_t num_experts_{8};
    int64_t top_k_{2};
    int64_t count_pad_{4};
    int64_t tflag_pad_{3};

    std::unordered_map<int, std::unordered_map<std::string, torch::Tensor>> staged_state_dicts_;
    c10::Dict<std::string, torch::Tensor> batched_weight_cache_;
    std::unordered_map<int, int> policy_id_to_cache_index_;
    bool weights_finalized_{false};
    std::unordered_map<int, int> policy_max_sequence_lengths_;
    std::unordered_map<int, CppBotRegistryEntry> cpp_bot_registry_;
    static std::unordered_map<std::string, CppBotKind> bot_kind_cache_;

    // --- Performance Timers ---
    std::chrono::microseconds timer_total_run_roles_{0};
    std::chrono::microseconds timer_arena_stepping_{0};
    std::chrono::microseconds timer_collect_requests_{0};
    std::chrono::microseconds timer_model_inference_{0};
    std::chrono::microseconds timer_cpp_bots_{0};
    std::chrono::microseconds timer_hist_prep_batch_{0};
    std::chrono::microseconds timer_hist_prep_weights_{0};
    std::chrono::microseconds timer_hist_model_exec_{0};
    std::chrono::microseconds timer_hist_post_{0};

    // Fine-grained timers captured inside forward pass and kernels
    std::unordered_map<std::string, std::chrono::microseconds> detailed_timers_;

    struct RequestRef {
        int policy_id;
        size_t request_index;
        const PolicyRequest* request;
    };

    void run_packed_historical_inference(const std::vector<RequestRef>& refs,
                                         std::unordered_map<int, std::vector<uint8_t>>& out_actions);
    std::vector<uint8_t> run_cpp_bot(int policy_id, const std::vector<PolicyRequest>& requests);
    static CppBotKind parse_cpp_bot_kind(const std::string& name);
    std::unique_ptr<CppBotBase> make_cpp_bot_instance(CppBotKind kind,
                                                      const PolicyRequest& request);
};
