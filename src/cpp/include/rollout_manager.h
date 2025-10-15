#pragma once

#include "vec_arena.h"

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

#include <torch/script.h>

class CppBotBase {
public:
    virtual ~CppBotBase() = default;
    virtual uint8_t act(const PolicyRequest& request, VecArena& arena) = 0;
};

struct TrajectoryData {
    int env_index{-1};
    int training_policy_id{-1};
    int training_agent_seat{-1};
    std::vector<int> player_policy_ids;

    std::vector<int> agent_id;
    std::vector<int> our_action;
    std::vector<float> log_prob;
    std::vector<float> value;
    std::vector<double> reward;
    std::vector<int> opp_target_action;
    int win{0};
};

struct SeatTrajectory {
    int seat{-1};
    int policy_id{-1};
    bool active{false};
    int last_training_step_idx{-1};
    std::array<uint8_t, Env::MAX_PLAYERS> last_penalties{};
    TrajectoryData data;
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

    void start_rollouts(int num_episodes,
                        int num_players,
                        const std::vector<int>& training_policy_ids,
                        int max_batch_envs = -1,
                        uint32_t seed = 0,
                        const std::vector<int>& opponent_labels = {},
                        const std::vector<double>& opponent_weights = {},
                        const std::vector<std::vector<int>>& opponent_triplets = {});

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

    void load_historical_model(int policy_id, const std::string& path);
    void register_cpp_bot(int policy_id, const std::string& bot_name);

    PreparedBatch prepare_training_batch(const std::vector<PolicyRequest>& requests,
                                         int policy_id) const;
    void set_training_device(const std::string& device_str);
    void set_max_sequence_length(int max_len);
    void set_policy_max_sequence_length(int policy_id, int max_len);
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
    struct HistoricalModelEntry {
        std::shared_ptr<torch::jit::Module> module;
        int cache_index{-1};
        // Cache for single-policy weight dict (when batch contains only this policy)
        c10::Dict<std::string, torch::Tensor> single_policy_weight_dict;
        bool single_policy_dict_cached{false};
    };

    std::unordered_map<int, HistoricalModelEntry> historical_models_;
    std::vector<int> historical_weight_policy_order_;
    std::unordered_map<std::string, torch::Tensor> historical_weight_cache_fp16_;
    bool historical_weight_cache_dirty_{false};
    std::unordered_map<int, CppBotRegistryEntry> cpp_bot_registry_;
    static std::unordered_map<std::string, CppBotKind> bot_kind_cache_;
    std::vector<uint8_t> training_env_inactive_;
    std::vector<int> active_training_counts_;
    std::vector<int> weighted_opponent_labels_;
    std::vector<double> weighted_opponent_weights_;
    std::vector<std::vector<int>> fixed_opponent_triplets_;

    // --- Profiling Timers ---
    std::chrono::microseconds timer_total_collect_{0};
    std::chrono::microseconds timer_log_rewards_{0};
    std::chrono::microseconds timer_cpp_bots_{0};
    std::chrono::microseconds timer_historical_total_{0};
    std::chrono::microseconds timer_hist_prep_{0};
    std::chrono::microseconds timer_hist_weights_{0};
    std::chrono::microseconds timer_hist_model_{0};
    std::chrono::microseconds timer_hist_post_{0};

    std::vector<std::vector<int>> build_roles(int batch_size,
                                              int num_players,
                                              const std::vector<int>& training_policy_ids,
                                              const std::vector<int>& weighted_opponents,
                                              const std::vector<double>& opponent_weights);
    EpisodeTracker new_episode_tracker(int env_idx, const std::vector<int>& roles);
    int append_training_step(SeatTrajectory& seat_tracker);
    int append_opponent_step(SeatTrajectory& seat_tracker, int seat);
    void update_penalty_rewards(SeatTrajectory& seat_tracker,
                                const std::array<uint8_t, Env::MAX_PLAYERS>& penalties);
    void log_rewards_and_dones();
    void finalize_episode(EpisodeTracker& tracker);
    void mark_training_env_inactive(int env_idx);
    void finalize_seat(EpisodeTracker& tracker, SeatTrajectory& seat_tracker, Env& env);
    void rebuild_historical_weight_cache();
    std::unordered_map<int, std::vector<uint8_t>> run_batched_historical_inference(
        const std::vector<std::pair<int, const std::vector<PolicyRequest>*>>& grouped,
        std::chrono::microseconds& prep_time_acc,
        std::chrono::microseconds& weight_time_acc,
        std::chrono::microseconds& model_time_acc,
        std::chrono::microseconds& post_time_acc);
    std::vector<uint8_t> run_historical_inference(torch::jit::Module& module,
                                                  const std::vector<PolicyRequest>& requests);
    std::vector<uint8_t> run_cpp_bot(int policy_id, const std::vector<PolicyRequest>& requests);
    static CppBotKind parse_cpp_bot_kind(const std::string& name);
    std::unique_ptr<CppBotBase> make_cpp_bot_instance(CppBotKind kind,
                                                      const PolicyRequest& request);
};

