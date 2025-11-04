#pragma once

#include "vec_arena.h"
#include "execution_core.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <torch/torch.h>

class CppBotBase {
public:
    virtual ~CppBotBase() = default;
    virtual uint8_t act(const PolicyRequest& request, VecArena& arena) = 0;
};

struct TrajectoryData {
    int env_index{-1};
    int training_policy_id{-1};
    int training_agent_seat{-1};
    int agent_index{-1};  // Stable ID for this agent (use instead of trajectory_id)
    std::vector<int> player_policy_ids;

    std::vector<int> agent_id;
    std::vector<int> our_action;
    std::vector<float> log_prob;
    std::vector<float> value;
    std::vector<double> reward;
    std::vector<int> opp_target_action;
    int win{0};

    // Last model input for training (saved before final inference)
    torch::Tensor last_obs_sequence;
    torch::Tensor last_action_sequence;
    torch::Tensor last_agent_types;
    torch::Tensor last_positions;
    torch::Tensor last_action_masks;
};

struct SeatTrajectory {
    int seat{-1};  // This IS agent_index (stable ID)
    int policy_id{-1};
    // Removed trajectory_id - seat is already agent_index!
    bool active{false};
    int last_training_step_idx{-1};
    std::array<uint8_t, Env::MAX_PLAYERS> last_penalties{};
    TrajectoryData data;

    // Last model input (for training with C++ forward pass)
    torch::Tensor last_obs_sequence;
    torch::Tensor last_action_sequence;
    torch::Tensor last_agent_types;
    torch::Tensor last_positions;
    torch::Tensor last_action_masks;
};

struct EpisodeTracker {
    int env_idx{-1};
    bool done{false};
    int last_history_len{0};
    int last_processed_history_len{0};
    std::unordered_map<int, SeatTrajectory> training_seats;
};

struct PreparedBatch {
    torch::Tensor obs_sequence;
    torch::Tensor action_sequence;
    torch::Tensor agent_types;
    torch::Tensor positions;
    torch::Tensor action_masks;
    torch::Tensor padding_mask;
    torch::Tensor valid_lengths;
    torch::Tensor env_indices;
    torch::Tensor seat_indices;
};

class RolloutManager {
public:
    RolloutManager();

    // New unified API - runs entire rollout internally
    std::vector<TrajectoryData> run_rollouts(
        int num_episodes,
        int num_players,
        const std::vector<int>& training_policy_ids,
        int max_batch_envs = -1,
        uint32_t seed = 0,
        const std::vector<std::vector<int>>& opponent_triplets = {},
        double shuffle_percentage = 0.0);

    // Legacy API for backwards compatibility
    void start_rollouts(int num_episodes,
                        int num_players,
                        const std::vector<int>& training_policy_ids,
                        int max_batch_envs = -1,
                        uint32_t seed = 0,
                        const std::vector<int>& opponent_labels = {},
                        const std::vector<double>& opponent_weights = {},
                        const std::vector<std::vector<int>>& opponent_triplets = {},
                        double shuffle_percentage = 0.0);

    std::unordered_map<int, std::vector<PolicyRequest>> collect_requests_for_inference();

    void submit_inference_results(int policy_id,
                                  const std::vector<uint8_t>& actions,
                                  const std::vector<float>& log_probs = {},
                                  const std::vector<float>& values = {});

    void submit_inference_results_array(int policy_id,
                                        const uint8_t* actions,
                                        size_t action_count,
                                        const float* log_probs,
                                        size_t log_prob_count,
                                        const float* values,
                                        size_t value_count);

    std::vector<TrajectoryData> get_completed_episodes();

    void load_model(int policy_id,
                    const std::unordered_map<std::string, torch::Tensor>& state_dict,
                    const std::string& original_path = "");
    void finalize_model_loading();
    void register_cpp_bot(int policy_id, const std::string& bot_name);

    PreparedBatch prepare_training_batch(const std::vector<PolicyRequest>& requests,
                                         int policy_id) const;
    void set_training_device(const std::string& device_str);
    void set_max_sequence_length(int max_len);
    void set_policy_max_sequence_length(int policy_id, int max_len);
    void set_use_greedy_stepping(bool use_greedy);
    void load_historical_model(int policy_id, const std::string& path);
    std::unordered_map<std::string, int64_t> get_performance_stats() const;
    /**
     * Get accumulated timing statistics from forward passes.
     * Returns a map of operation name -> total microseconds.
     */
    std::unordered_map<std::string, int64_t> get_timing_stats() const;
    int training_policy_id() const {
        return training_policy_ids_.empty() ? -1 : training_policy_ids_.front();
    }
    const std::vector<int>& training_policy_ids() const { return training_policy_ids_; }
    bool is_training_policy(int policy_id) const;

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

    // Model architecture parameters (inferred from first loaded model)
    int64_t num_layers_{2};
    int64_t num_heads_{4};
    int64_t hidden_dim_{256};
    int64_t num_experts_{8};
    int64_t top_k_{2};
    int64_t max_inference_batch_size_{512};

    // Batched weights for all policies (learner + historical + bots use IDs)
    std::unordered_map<int, std::unordered_map<std::string, torch::Tensor>> staged_state_dicts_;
    c10::Dict<std::string, torch::Tensor> batched_weight_cache_;
    std::unordered_map<int, int> policy_id_to_cache_index_;
    bool weights_finalized_{false};

    std::unique_ptr<execution_core::NeuralInferenceOrchestrator> orchestrator_;
    std::unordered_map<int, CppBotRegistryEntry> cpp_bot_registry_;
    static std::unordered_map<std::string, CppBotKind> bot_kind_cache_;
    std::vector<uint8_t> training_env_inactive_;
    std::vector<int> active_training_counts_;
    std::vector<int> weighted_opponent_labels_;
    std::vector<double> weighted_opponent_weights_;
    std::vector<std::vector<int>> fixed_opponent_triplets_;
    
    // Seat shuffling is configured per Env via Env::shuffle_seats_each_round
    
    // Legacy mode: greedy stepping and torch.jit.Module for historical models
    bool use_greedy_stepping_{false};
    std::unordered_map<int, std::shared_ptr<torch::jit::Module>> historical_models_;

    // --- Profiling Timers ---
    std::chrono::microseconds timer_total_collect_{0};
    std::chrono::microseconds timer_log_rewards_{0};
    std::chrono::microseconds timer_cpp_bots_{0};
    std::chrono::microseconds timer_neural_inference_{0};

    std::vector<std::vector<int>> build_roles(int batch_size,
                                              int num_players,
                                              const std::vector<int>& training_policy_ids,
                                              const std::vector<int>& weighted_opponents,
                                              const std::vector<double>& opponent_weights);
    EpisodeTracker new_episode_tracker(int env_idx, const std::vector<int>& roles);
    int append_training_step(SeatTrajectory& seat_tracker, int env_idx);
    int append_opponent_step(SeatTrajectory& seat_tracker, int opponent_agent_index);
    void update_penalty_rewards(SeatTrajectory& seat_tracker,
                                const std::array<uint8_t, Env::MAX_PLAYERS>& penalties);
    void log_rewards_and_dones();
    void finalize_episode(EpisodeTracker& tracker);
    void mark_training_env_inactive(int env_idx);
    void finalize_seat(EpisodeTracker& tracker, SeatTrajectory& seat_tracker, Env& env);
    // Seat shuffling handled within Env

    void run_neural_inference(
        const std::unordered_map<int, std::vector<PolicyRequest>>& requests_by_policy,
        std::unordered_map<int, std::vector<uint8_t>>& out_actions,
        std::unordered_map<int, std::vector<float>>& out_log_probs,
        std::unordered_map<int, std::vector<float>>& out_values);

    std::vector<uint8_t> run_cpp_bot(int policy_id, const std::vector<PolicyRequest>& requests);
    std::vector<uint8_t> run_historical_inference(torch::jit::Module& module,
                                                  const std::vector<PolicyRequest>& requests);
    static CppBotKind parse_cpp_bot_kind(const std::string& name);
    std::unique_ptr<CppBotBase> make_cpp_bot_instance(CppBotKind kind,
                                                      const PolicyRequest& request);
};
