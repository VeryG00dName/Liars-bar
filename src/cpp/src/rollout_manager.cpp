#include "rollout_manager.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace {
std::mt19937::result_type seed_with_optional(uint32_t seed) {
    if (seed != 0) {
        return static_cast<std::mt19937::result_type>(seed);
    }
    std::random_device rd;
    return static_cast<std::mt19937::result_type>(rd());
}
}

RolloutManager::RolloutManager()
    : rng_(seed_with_optional(0)) {}

void RolloutManager::start_rollouts(int num_episodes,
                                    int num_players,
                                    int training_policy_id,
                                    int max_batch_envs,
                                    uint32_t seed,
                                    const std::vector<int>& cpp_bots,
                                    const std::vector<int>& latest_historical_agents,
                                    const std::vector<int>& active_shadow_agents,
                                    double front_mass,
                                    double shadow_mass) {
    target_episodes_ = num_episodes;
    num_players_ = num_players;
    training_policy_id_ = training_policy_id;
    completed_buffer_.clear();
    pending_step_data_.clear();

    const int batch_guess = (arena_.B > 0) ? arena_.B : num_episodes;
    if (max_batch_envs > 0) {
        batch_size_ = std::min(batch_guess, max_batch_envs);
    } else {
        batch_size_ = batch_guess;
    }
    if (batch_size_ <= 0) {
        batch_size_ = std::max(1, num_episodes);
    }

    rng_.seed(seed_with_optional(seed));

    arena_.reset(batch_size_, num_players_, rng_());

    std::vector<int> front_pool = cpp_bots;
    front_pool.insert(front_pool.end(), latest_historical_agents.begin(), latest_historical_agents.end());

    auto roles = build_roles(batch_size_,
                             num_players_,
                             training_policy_id_,
                             front_pool,
                             active_shadow_agents,
                             front_mass,
                             shadow_mass);
    arena_.set_roles(roles);

    episodes_.clear();
    episodes_.resize(batch_size_);
    for (int env_idx = 0; env_idx < batch_size_; ++env_idx) {
        episodes_[env_idx] = new_episode_tracker(env_idx, roles[env_idx]);
    }
}

std::unordered_map<int, std::vector<PolicyRequest>> RolloutManager::collect_requests_for_inference() {
    log_rewards_and_dones();

    const auto& raw = arena_.collect_requests();
    return raw;
}

void RolloutManager::submit_inference_results(int policy_id,
                                              const std::vector<uint8_t>& actions,
                                              const std::vector<float>& log_probs,
                                              const std::vector<float>& values) {
    auto it_req = arena_.pending.find(policy_id);
    if (it_req == arena_.pending.end()) {
        return;
    }
    const auto& reqs = it_req->second;

    const size_t count = std::min(reqs.size(), actions.size());
    const bool has_log_probs = log_probs.size() == count;
    const bool has_values = values.size() == count;

    for (size_t i = 0; i < count; ++i) {
        const int env_idx = reqs[i].env;
        const int seat = reqs[i].seat;
        if (env_idx < 0 || env_idx >= static_cast<int>(episodes_.size())) {
            continue;
        }
        EpisodeTracker& tracker = episodes_[env_idx];
        if (tracker.done) {
            continue;
        }
        if (!tracker.is_training_episode || seat != tracker.training_agent_seat ||
            policy_id != training_policy_id_) {
            continue;
        }

        const int step_idx = append_step_row(tracker, seat);
        if (step_idx >= 0 && step_idx < static_cast<int>(tracker.data.our_action.size())) {
            tracker.data.our_action[step_idx] = static_cast<int>(actions[i]);
        }

        PendingStepData pending{};
        pending.log_prob = (has_log_probs ? log_probs[i] : 0.0f);
        pending.value = (has_values ? values[i] : 0.0f);
        if (env_idx >= 0 && env_idx < static_cast<int>(arena_.envs.size()) &&
            seat >= 0 && seat < arena_.envs[env_idx].num_players()) {
            pending.penalties_used = static_cast<int>(arena_.envs[env_idx].penalties[seat]);
        }
        pending_step_data_[env_idx] = pending;
    }

    arena_.submit_actions(policy_id, actions);
}

std::vector<TrajectoryData> RolloutManager::get_completed_episodes() {
    log_rewards_and_dones();

    for (auto& tracker : episodes_) {
        if (!tracker.done && tracker.env_idx >= 0 && tracker.env_idx < static_cast<int>(arena_.done.size())) {
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

std::vector<std::vector<int>> RolloutManager::build_roles(int batch_size,
                                                          int num_players,
                                                          int training_policy_id,
                                                          const std::vector<int>& front_agents,
                                                          const std::vector<int>& shadow_agents,
                                                          double front_mass,
                                                          double shadow_mass) {
    std::vector<std::vector<int>> roles(batch_size, std::vector<int>(num_players, training_policy_id));
    if (batch_size <= 0 || num_players <= 0) {
        return roles;
    }

    auto unique_sorted = [](std::vector<int> values) {
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        return values;
    };

    std::vector<int> front = unique_sorted(front_agents);
    std::vector<int> shadow = unique_sorted(shadow_agents);

    if (!front.empty()) {
        std::unordered_set<int> front_set(front.begin(), front.end());
        std::vector<int> filtered;
        filtered.reserve(shadow.size());
        for (int id : shadow) {
            if (!front_set.count(id)) {
                filtered.push_back(id);
            }
        }
        shadow.swap(filtered);
    }

    std::vector<int> opponent_pool = front;
    opponent_pool.insert(opponent_pool.end(), shadow.begin(), shadow.end());
    if (opponent_pool.empty()) {
        opponent_pool.push_back(training_policy_id);
    }

    const size_t pool_size = opponent_pool.size();
    std::vector<double> probs(pool_size, 0.0);

    const double adjusted_front_mass = front.empty() ? 0.0 : front_mass;
    const double adjusted_shadow_mass = shadow.empty() ? 0.0 : shadow_mass;
    const double total_mass = adjusted_front_mass + adjusted_shadow_mass;

    if (total_mass <= 0.0 || pool_size == 0) {
        const double uniform = pool_size > 0 ? 1.0 / static_cast<double>(pool_size) : 1.0;
        std::fill(probs.begin(), probs.end(), uniform);
    } else {
        std::unordered_set<int> front_set(front.begin(), front.end());
        const double front_norm = front.empty() ? 1.0 : static_cast<double>(front.size());
        const double shadow_norm = shadow.empty() ? 1.0 : static_cast<double>(shadow.size());

        for (size_t i = 0; i < pool_size; ++i) {
            const bool is_front = front_set.count(opponent_pool[i]) > 0;
            const double bucket_mass = is_front ? adjusted_front_mass : adjusted_shadow_mass;
            const double bucket_norm = is_front ? front_norm : shadow_norm;
            probs[i] = bucket_mass / bucket_norm;
        }
        const double sum = std::accumulate(probs.begin(), probs.end(), 0.0);
        if (sum > 0.0) {
            for (double& p : probs) {
                p /= sum;
            }
        } else {
            const double uniform = pool_size > 0 ? 1.0 / static_cast<double>(pool_size) : 1.0;
            std::fill(probs.begin(), probs.end(), uniform);
        }
    }

    const int num_opponents = std::max(0, num_players - 1);
    std::vector<int> seats(num_players);
    std::iota(seats.begin(), seats.end(), 0);

    std::discrete_distribution<int> sampler(probs.begin(), probs.end());

    for (int b = 0; b < batch_size; ++b) {
        std::shuffle(seats.begin(), seats.end(), rng_);
        const int training_seat = seats.front();
        roles[b][training_seat] = training_policy_id;

        for (int opponent_idx = 0; opponent_idx < num_opponents; ++opponent_idx) {
            const int seat = seats[opponent_idx + 1];
            const int choice = opponent_pool[sampler(rng_)];
            roles[b][seat] = choice;
        }
    }

    return roles;
}

EpisodeTracker RolloutManager::new_episode_tracker(int env_idx, const std::vector<int>& roles) {
    EpisodeTracker tracker;
    tracker.env_idx = env_idx;
    tracker.training_policy_id = training_policy_id_;
    tracker.data.env_index = env_idx;
    tracker.data.training_policy_id = training_policy_id_;
    tracker.data.player_policy_ids = roles;

    int seat = -1;
    for (size_t i = 0; i < roles.size(); ++i) {
        if (roles[i] == training_policy_id_) {
            seat = static_cast<int>(i);
            break;
        }
    }
    if (seat >= 0) {
        tracker.is_training_episode = true;
        tracker.training_agent_seat = seat;
        tracker.data.training_agent_seat = seat;
    } else {
        tracker.is_training_episode = false;
        tracker.training_agent_seat = -1;
        tracker.data.training_agent_seat = -1;
    }

    tracker.last_penalties.fill(0);

    return tracker;
}

int RolloutManager::append_step_row(EpisodeTracker& tracker, int seat) {
    tracker.data.agent_id.push_back(seat);
    tracker.data.our_action.push_back(-1);
    tracker.data.log_prob.push_back(0.0f);
    tracker.data.value.push_back(0.0f);
    tracker.data.reward.push_back(0.0);
    tracker.data.done.push_back(0);
    tracker.data.opp_target_action.push_back(-1);
    tracker.data.penalties_used.push_back(0);
    const int idx = static_cast<int>(tracker.data.agent_id.size()) - 1;
    if (seat == tracker.training_agent_seat) {
        tracker.last_training_step_idx = idx;
    }
    return idx;
}

void RolloutManager::update_penalty_rewards(EpisodeTracker& tracker, const std::array<uint8_t, Env::MAX_PLAYERS>& penalties) {
    if (!tracker.is_training_episode) {
        tracker.last_penalties = penalties;
        return;
    }

    const int seat = tracker.training_agent_seat;
    const int last_idx = tracker.last_training_step_idx;
    if (seat < 0 || last_idx < 0 || last_idx >= static_cast<int>(tracker.data.reward.size())) {
        tracker.last_penalties = penalties;
        return;
    }

    double delta_total = 0.0;
    for (size_t i = 0; i < penalties.size(); ++i) {
        const int diff = static_cast<int>(penalties[i]) - static_cast<int>(tracker.last_penalties[i]);
        if (diff <= 0) {
            continue;
        }
        if (static_cast<int>(i) == seat) {
            delta_total -= 0.1 * diff;
        } else {
            delta_total += 0.033 * diff;
        }
    }
    if (delta_total != 0.0) {
        tracker.data.reward[last_idx] += delta_total;
    }
    tracker.last_penalties = penalties;
}

void RolloutManager::log_rewards_and_dones() {
    for (auto& tracker : episodes_) {
        if (tracker.done || tracker.env_idx < 0 || tracker.env_idx >= static_cast<int>(arena_.envs.size())) {
            continue;
        }
        Env& env = arena_.envs[tracker.env_idx];
        const int total_len = env.get_total_history_entries();
        const int start_idx = tracker.last_processed_history_len;
        if (start_idx < total_len) {
            auto history = env.get_history_entries_slice(start_idx, total_len);
            for (const auto& entry : history) {
                tracker.last_history_len += 1;
                const int actor = entry.player;
                if (tracker.is_training_episode && actor == tracker.training_agent_seat) {
                    auto it_pending = pending_step_data_.find(tracker.env_idx);
                    if (it_pending != pending_step_data_.end()) {
                        const int idx = tracker.last_training_step_idx;
                        if (idx >= 0 && idx < static_cast<int>(tracker.data.log_prob.size())) {
                            tracker.data.log_prob[idx] = it_pending->second.log_prob;
                            tracker.data.value[idx] = it_pending->second.value;
                            tracker.data.penalties_used[idx] = it_pending->second.penalties_used;
                        }
                        pending_step_data_.erase(it_pending);
                    }
                } else {
                    const int idx = append_step_row(tracker, actor);
                    if (idx >= 0 && idx < static_cast<int>(tracker.data.opp_target_action.size())) {
                        tracker.data.opp_target_action[idx] = static_cast<int>(entry.action);
                    }
                }
            }
            tracker.last_processed_history_len = total_len;
        }

        update_penalty_rewards(tracker, env.penalties);

        if (arena_.done[tracker.env_idx]) {
            finalize_episode(tracker);
        }
    }
}

void RolloutManager::finalize_episode(EpisodeTracker& tracker) {
    if (tracker.done) {
        return;
    }
    tracker.done = true;

    if (tracker.env_idx < 0 || tracker.env_idx >= static_cast<int>(arena_.envs.size())) {
        return;
    }
    Env& env = arena_.envs[tracker.env_idx];

    update_penalty_rewards(tracker, env.penalties);

    auto it_pending = pending_step_data_.find(tracker.env_idx);
    if (it_pending != pending_step_data_.end()) {
        if (!tracker.data.agent_id.empty() && tracker.last_training_step_idx >= 0 &&
            tracker.last_training_step_idx < static_cast<int>(tracker.data.agent_id.size()) &&
            tracker.data.agent_id[tracker.last_training_step_idx] == tracker.training_agent_seat) {
            const int idx = tracker.last_training_step_idx;
            tracker.data.log_prob[idx] = it_pending->second.log_prob;
            tracker.data.value[idx] = it_pending->second.value;
            tracker.data.penalties_used[idx] = it_pending->second.penalties_used;
        }
        pending_step_data_.erase(it_pending);
    }

    int our_last_step_idx = -1;
    for (int i = static_cast<int>(tracker.data.agent_id.size()) - 1; i >= 0; --i) {
        if (tracker.data.agent_id[i] == tracker.training_agent_seat) {
            our_last_step_idx = i;
            break;
        }
    }

    int alive = 0;
    for (int p = 0; p < env.num_players(); ++p) {
        if (!env.terminations[p]) {
            ++alive;
        }
    }
    const bool is_winner = tracker.is_training_episode && tracker.training_agent_seat >= 0 &&
                           !env.terminations[tracker.training_agent_seat] &&
                           alive == 1;
    tracker.data.win = is_winner ? 1 : 0;

    if (our_last_step_idx >= 0 && our_last_step_idx < static_cast<int>(tracker.data.reward.size())) {
        tracker.data.reward[our_last_step_idx] += is_winner ? 1.0 : -1.0;
    }

    tracker.data.episode_return = std::accumulate(tracker.data.reward.begin(), tracker.data.reward.end(), 0.0);

    if (tracker.is_training_episode) {
        completed_buffer_.push_back(std::move(tracker.data));
        tracker.data = TrajectoryData{};
    }
}

