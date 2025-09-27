#pragma once

#include "vec_arena.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
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
    std::vector<uint8_t> done;
    std::vector<int> opp_target_action;
    std::vector<int> penalties_used;

    double episode_return{0.0};
    int win{0};
};

struct EpisodeTracker {
    int env_idx{-1};
    bool done{false};
    bool is_training_episode{false};
    int training_policy_id{-1};
    int training_agent_seat{-1};

    int last_history_len{0};
    int last_processed_history_len{0};
    int last_training_step_idx{-1};
    std::array<uint8_t, Env::MAX_PLAYERS> last_penalties{};

    TrajectoryData data;
};

struct PendingStepData {
    float log_prob{0.0f};
    float value{0.0f};
    int penalties_used{0};
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
                        int training_policy_id,
                        int max_batch_envs = -1,
                        uint32_t seed = 0,
                        const std::vector<int>& cpp_bots = {},
                        const std::vector<int>& opponent_labels = {},
                        const std::vector<double>& opponent_weights = {},
                        int newest_opponent_label = -1);

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

    PreparedBatch prepare_training_batch(const std::vector<PolicyRequest>& requests) const;
    void set_training_device(const std::string& device_str);
    int training_policy_id() const { return training_policy_id_; }

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
    int training_policy_id_{-1};
    std::mt19937 rng_;
    torch::Device training_device_{torch::kCPU};

    std::vector<EpisodeTracker> episodes_;
    std::unordered_map<int, PendingStepData> pending_step_data_;
    std::vector<TrajectoryData> completed_buffer_;
    std::unordered_map<int, std::shared_ptr<torch::jit::Module>> historical_models_;
    std::unordered_map<int, CppBotRegistryEntry> cpp_bot_registry_;

    std::vector<std::vector<int>> build_roles(int batch_size,
                                              int num_players,
                                              int training_policy_id,
                                              const std::vector<int>& opponent_labels,
                                              const std::vector<double>& opponent_weights,
                                              int newest_label);
    EpisodeTracker new_episode_tracker(int env_idx, const std::vector<int>& roles);
    int append_step_row(EpisodeTracker& tracker, int seat);
    void update_penalty_rewards(EpisodeTracker& tracker, const std::array<uint8_t, Env::MAX_PLAYERS>& penalties);
    void log_rewards_and_dones();
    void finalize_episode(EpisodeTracker& tracker);
    std::vector<uint8_t> run_historical_inference(torch::jit::Module& module,
                                                  const std::vector<PolicyRequest>& requests);
    std::vector<uint8_t> run_cpp_bot(int policy_id, const std::vector<PolicyRequest>& requests);
    static CppBotKind parse_cpp_bot_kind(const std::string& name);
    std::unique_ptr<CppBotBase> make_cpp_bot_instance(CppBotKind kind,
                                                      const PolicyRequest& request);
};

