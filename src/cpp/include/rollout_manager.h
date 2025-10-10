#pragma once

#include "vec_arena.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <torch/script.h>

// Base class for simple, stateless C++ bots.
class CppBotBase {
public:
    virtual ~CppBotBase() = default;
    virtual uint8_t act(const PolicyRequest& request, VecArena& arena) = 0;
};

// Holds all data for a single completed trajectory from one agent's perspective.
struct TrajectoryData {
    int env_index{-1};
    int training_policy_id{-1};
    int training_agent_seat{-1};
    std::vector<int> player_policy_ids;

    // Sparse, step-by-step data (length = total game steps).
    std::vector<int> agent_id;
    std::vector<double> reward;
    std::vector<int> opp_target_action;
    std::vector<float> value;
    
    // Dense data, only for steps taken by this agent.
    std::vector<int> our_action;
    std::vector<float> log_prob;
    
    int win{0};
    
    // Sequences for the autoregressive model input.
    int valid_len{0};
    std::vector<float> obs_sequence;
    std::vector<int64_t> action_sequence;
    std::vector<int64_t> agent_type_sequence;
    std::vector<int64_t> position_sequence;
    std::vector<uint8_t> action_mask_sequence;
};

// Tracks the state of a single training agent within an episode.
struct SeatTrajectory {
    int seat{-1};
    int policy_id{-1};
    bool active{false};
    std::array<uint8_t, Env::MAX_PLAYERS> last_penalties{};
    TrajectoryData data;
};

// Tracks the overall state of a single episode/environment.
struct EpisodeTracker {
    int env_idx{-1};
    bool done{false};
    int last_history_len{0};
    std::unordered_map<int, SeatTrajectory> training_seats;
};

class RolloutManager {
public:
    RolloutManager();

    // High-level function to run a batch of rollouts from start to finish.
    std::vector<TrajectoryData> get_rollouts(
        int num_episodes,
        int num_players,
        const std::vector<int>& training_policy_ids,
        int max_batch_envs,
        uint32_t seed,
        const std::vector<std::vector<int>>& opponent_triplets);

    // Low-level API for stepping through rollouts manually (used by get_rollouts).
    void start_rollouts(int num_episodes,
                        int num_players,
                        const std::vector<int>& training_policy_ids,
                        int max_batch_envs = -1,
                        uint32_t seed = 0,
                        const std::vector<int>& opponent_labels = {},
                        const std::vector<double>& opponent_weights = {},
                        const std::vector<std::vector<int>>& opponent_triplets = {});
    bool run_rollouts_step();
    bool all_episodes_complete() const;
    std::vector<TrajectoryData> get_completed_episodes();

    // Configuration and setup.
    void load_model_architecture(const std::string& path);
    void load_policy_weights(int policy_id, const std::string& path);
    void update_learner_weights(int policy_id, c10::Dict<c10::IValue, c10::IValue> state_dict);
    void register_cpp_bot(int policy_id, const std::string& bot_name);
    void set_training_device(const std::string& device_str);
    void set_max_sequence_length(int max_len);
    void set_policy_max_sequence_length(int policy_id, int max_len);
    void set_inference_batch_size(int size);

    bool is_training_policy(int policy_id) const;
    const std::vector<int>& training_policy_ids() const { return training_policy_ids_; }
    int training_policy_id() const {
        return training_policy_ids_.empty() ? -1 : training_policy_ids_.front();
    }
    const torch::Device& training_device() const { return training_device_; }

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
    int target_episodes_{0};
    int batch_size_{0};
    int num_players_{0};
    std::vector<int> training_policy_ids_{};
    std::unordered_set<int> training_policy_id_set_{};
    std::mt19937 rng_;
    torch::Device training_device_{torch::kCPU};
    int default_max_sequence_length_{DEFAULT_MAX_LEN};
    std::unordered_map<int, int> policy_max_sequence_length_;

    std::vector<EpisodeTracker> episodes_;
    std::vector<TrajectoryData> completed_buffer_;
    std::shared_ptr<torch::jit::Module> jit_module_;
    std::unordered_map<int, c10::Dict<c10::IValue, c10::IValue>> policy_weights_;
    std::unordered_map<int, CppBotRegistryEntry> cpp_bot_registry_;
    
    int inference_batch_size_{256};

    // Helper functions for the rollout loop.
    EpisodeTracker new_episode_tracker(int env_idx, const std::vector<int>& roles);
    void append_step(SeatTrajectory& seat_tracker, const HistoryEntry& h);
    void append_new_history_entries();
    void maybe_finalize_episodes();
    void finalize_episode(EpisodeTracker& tracker);
    
    std::vector<uint8_t> run_cpp_bot(int policy_id, const std::vector<PolicyRequest>& requests);
    static CppBotKind parse_cpp_bot_kind(const std::string& name);
    std::unique_ptr<CppBotBase> make_cpp_bot_instance(CppBotKind kind,
                                                      const PolicyRequest& request);
    
    void apply_inference_results(int policy_id,
                                 const std::vector<PolicyRequest>& requests,
                                 const std::vector<uint8_t>& actions,
                                 const std::vector<float>& log_probs,
                                 const std::vector<float>& values);
    
    c10::Dict<c10::IValue, c10::IValue> pack_weights_for_batch(const std::vector<int>& policy_ids) const;
};