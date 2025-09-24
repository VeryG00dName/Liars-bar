#pragma once

#include "vec_arena.h"

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include <torch/script.h>

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

class RolloutManager {
public:
    RolloutManager();

    void start_rollouts(int num_episodes,
                        int num_players,
                        int training_policy_id,
                        int max_batch_envs = -1,
                        uint32_t seed = 0,
                        const std::vector<int>& cpp_bots = {},
                        const std::vector<int>& latest_historical_agents = {},
                        const std::vector<int>& active_shadow_agents = {},
                        double front_mass = 0.0,
                        double shadow_mass = 0.0);

    std::unordered_map<int, std::vector<PolicyRequest>> collect_requests_for_inference();

    void submit_inference_results(int policy_id,
                                  const std::vector<uint8_t>& actions,
                                  const std::vector<float>& log_probs = {},
                                  const std::vector<float>& values = {});

    std::vector<TrajectoryData> get_completed_episodes();

    void load_historical_model(int policy_id, const std::string& path);

private:
    VecArena arena_;
    int target_episodes_{0};
    int batch_size_{0};
    int num_players_{0};
    int training_policy_id_{-1};
    std::mt19937 rng_;

    std::vector<EpisodeTracker> episodes_;
    std::unordered_map<int, PendingStepData> pending_step_data_;
    std::vector<TrajectoryData> completed_buffer_;
    std::unordered_map<int, std::shared_ptr<torch::jit::Module>> historical_models_;

    std::vector<std::vector<int>> build_roles(int batch_size,
                                              int num_players,
                                              int training_policy_id,
                                              const std::vector<int>& front_agents,
                                              const std::vector<int>& shadow_agents,
                                              double front_mass,
                                              double shadow_mass);
    EpisodeTracker new_episode_tracker(int env_idx, const std::vector<int>& roles);
    int append_step_row(EpisodeTracker& tracker, int seat);
    void update_penalty_rewards(EpisodeTracker& tracker, const std::array<uint8_t, Env::MAX_PLAYERS>& penalties);
    void log_rewards_and_dones();
    void finalize_episode(EpisodeTracker& tracker);
    std::vector<uint8_t> run_historical_inference(torch::jit::Module& module,
                                                  const std::vector<PolicyRequest>& requests);
};

