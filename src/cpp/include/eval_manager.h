#pragma once

#include "vec_arena.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <torch/script.h>

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
    void set_inference_batch_size(int batch_size);

    void load_model(int policy_id, const std::string& path);
    void register_cpp_bot(int policy_id, const std::string& bot_name);

    EvalOutcome run_roles(const std::vector<std::vector<int>>& roles,
                          const std::vector<int>& lineup_indices,
                          int num_players,
                          uint32_t seed);

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
    int inference_batch_size_{128};
    std::mt19937 rng_;
    std::unordered_map<int, std::shared_ptr<torch::jit::Module>> models_;
    std::unordered_map<int, int> policy_max_sequence_lengths_;
    std::unordered_map<int, CppBotRegistryEntry> cpp_bot_registry_;

    std::vector<uint8_t> run_model(torch::jit::Module& module,
                                   const std::vector<PolicyRequest>& requests);
    std::vector<uint8_t> run_cpp_bot(int policy_id, const std::vector<PolicyRequest>& requests);
    static CppBotKind parse_cpp_bot_kind(const std::string& name);
    std::unique_ptr<CppBotBase> make_cpp_bot_instance(CppBotKind kind,
                                                      const PolicyRequest& request);
};

