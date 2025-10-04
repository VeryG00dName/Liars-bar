#include "vec_arena.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <cstdio>
#include <cstring>
// ---------- small helpers ----------
uint8_t VecArena::first_valid(const uint8_t mask[7]) {
    for (int i = 0; i < 7; ++i) if (mask[i]) return (uint8_t)i;
    return 6; // default challenge if somehow none
}

void VecArena::fill_mask_for_current(const Env& e, uint8_t m[7]) {
    e.valid_actions(m);
}

void VecArena::prepare_ai_sequence(const Env& e, int ai_seat, int seq_cap, PolicyRequest& out) const {
    const int n_players = e.num_players();
    const int total = static_cast<int>(e.game_history.size());
    const int capped_max = std::max(1, seq_cap);
    const int start = std::max(0, total - (capped_max - 1));
    const int history_span = std::max(0, total - start);
    const int max_history = std::min(capped_max - 1, history_span);
    const int max_valid = std::max(1, std::min(capped_max, history_span + 1));

    out.obs_sequence.resize(static_cast<size_t>(max_valid) * OBS_DIM);
    out.action_sequence.resize(max_valid);
    out.agent_type_sequence.resize(max_valid);
    out.position_sequence.resize(max_valid);
    out.action_mask_sequence.resize(static_cast<size_t>(max_valid) * 7);
    std::fill(out.action_mask_sequence.begin(), out.action_mask_sequence.end(), 0);

    auto transform_action = [&](int actor_seat, uint8_t action) -> int64_t {
        int relative_agent_id = (actor_seat - ai_seat + n_players) % n_players;
        if (relative_agent_id == 0) {
            return static_cast<int64_t>(action);
        }
        if (action <= 5) {
            return static_cast<int64_t>(7 + (action % 3));
        }
        return static_cast<int64_t>(action);
    };

    float* obs_ptr = out.obs_sequence.data();
    int64_t* action_ptr = out.action_sequence.data();
    int64_t* agent_ptr = out.agent_type_sequence.data();
    int64_t* pos_ptr = out.position_sequence.data();
    uint8_t* mask_ptr = out.action_mask_sequence.data();

    int idx = 0;
    const int history_start = total - max_history;
    for (int i = history_start; i < total && idx < max_valid; ++i) {
        const HistoryEntry& h = e.game_history[i];
        const auto& obs = h.observations[ai_seat];
        const int obs_take = std::min<int>(OBS_DIM, static_cast<int>(obs.size()));
        float* dst_obs = obs_ptr + idx * OBS_DIM;
        if (obs_take > 0) {
            std::memcpy(dst_obs, obs.data(), sizeof(float) * obs_take);
            if (obs_take < OBS_DIM) {
                std::fill(dst_obs + obs_take, dst_obs + OBS_DIM, 0.0f);
            }
        } else {
            std::fill(dst_obs, dst_obs + OBS_DIM, 0.0f);
        }

        int rel = (h.player - ai_seat + n_players) % n_players;
        agent_ptr[idx] = rel;
        if (idx == 0) {
            action_ptr[idx] = 10; // PAD first token
        } else {
            const HistoryEntry& prev_h = e.game_history[i - 1];
            action_ptr[idx] = transform_action(prev_h.player, prev_h.action);
        }

        if (rel == 0) {
            std::memcpy(mask_ptr + idx * 7, h.mask.data(), sizeof(uint8_t) * 7);
        }

        pos_ptr[idx] = idx;
        ++idx;
    }

    if (idx < max_valid) {
        float cur_obs[OBS_DIM];
        e.observe_vector_newerest(ai_seat, cur_obs);
        std::memcpy(obs_ptr + idx * OBS_DIM, cur_obs, sizeof(float) * OBS_DIM);
        agent_ptr[idx] = 0; // me
        if (total > 0) {
            const HistoryEntry& last_h = e.game_history[total - 1];
            action_ptr[idx] = transform_action(last_h.player, last_h.action);
        } else {
            action_ptr[idx] = 10;
        }
        e.valid_actions(mask_ptr + idx * 7);
        pos_ptr[idx] = idx;
        ++idx;
    }

    out.valid_len = std::max(1, idx);
    out.action_sequence.resize(out.valid_len);
    out.agent_type_sequence.resize(out.valid_len);
    out.position_sequence.resize(out.valid_len);
    out.action_mask_sequence.resize(static_cast<size_t>(out.valid_len) * 7);
    out.obs_sequence.resize(static_cast<size_t>(out.valid_len) * OBS_DIM);
}

// ---------- API ----------
void VecArena::reset(int B_, int n_players_, uint32_t seed0) {
    B = B_;
    n_players = n_players_;
    base_seed = seed0;
    envs.assign(B, Env{});
    done.assign(B, 0);
    roles.assign(B, std::vector<Role>(n_players));
    pending.clear();

    for (int b = 0; b < B; ++b) envs[b].reset(n_players, seed0 + (uint32_t)b);
}

void VecArena::set_max_sequence_length(int max_len) {
    if (max_len <= 0) {
        max_sequence_length = 1;
    } else {
        max_sequence_length = max_len;
    }
}

void VecArena::set_policy_max_sequence_length(int policy_id, int max_len) {
    if (policy_id < 0) {
        return;
    }
    if (max_len <= 0) {
        policy_max_sequence_lengths.erase(policy_id);
        return;
    }
    policy_max_sequence_lengths[policy_id] = max_len;
}

int VecArena::max_sequence_length_for_policy(int policy_id) const {
    auto it = policy_max_sequence_lengths.find(policy_id);
    if (it != policy_max_sequence_lengths.end() && it->second > 0) {
        return it->second;
    }
    return max_sequence_length;
}

void VecArena::set_roles(const std::vector<std::vector<int>>& policy_ids_per_env) {
    roles.assign(B, std::vector<Role>(n_players));
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < n_players; ++s) {
            roles[b][s].policy_id = policy_ids_per_env[b][s];
        }
    }
    pending.clear();
}

void VecArena::advance_env_until_policy_or_done(
  int env_index,
  std::unordered_map<int, std::vector<PolicyRequest>>& out)
{
  if (done[env_index]) return;
  Env& e = envs[env_index];

  // Large but finite limit so you don't hang forever if a bug slips in
  constexpr int64_t ITER_BUDGET = 200000;  // tune as needed
  int64_t iters = 0;

  for (;;) {
    if (++iters > ITER_BUDGET) {
      // --- DEBUG DUMP ---
      uint8_t m[7]; e.valid_actions(m);
      float legacy[3 + Env::MAX_PLAYERS]; int L = e.observe_vector(legacy);
      fprintf(stderr,
        "[WATCHDOG] env=%d stuck. cur=%d alive(term=0)=%d  mask=[%d%d%d%d%d%d%d] "
        "pending=%u claimant=%d count=%u bluff=%u last=%u force=%u fclaim=%d L=%d\n",
        env_index, e.current_player(),
        [&](){int a=0;for(int p=0;p<e.num_players();++p) if(!e.terminations[p]) ++a; return a;}(),
        m[0],m[1],m[2],m[3],m[4],m[5],m[6],
        e.pending_exists, e.pending_claimant, e.pending_count, e.pending_bluff,
        e.last_action_count, e.force_challenge_mode, e.force_claimant, L);
      // Optional: dump hands/penalties
      for (int p=0;p<e.num_players();++p) {
        fprintf(stderr, "  seat %d: len=%u pen=%u lim=%u term=%u round_out=%u\n",
          p, e.hand_len[p], e.penalties[p], e.penalty_limits[p], e.terminations[p], e.round_eliminated[p]);
      }
      throw std::runtime_error("advance_env_until_policy_or_done watchdog tripped");
    }

    int alive = 0;
    for (int p = 0; p < e.num_players(); ++p) if (!e.terminations[p]) ++alive;
    if (alive <= 1) { done[env_index] = 1; return; }

    int cur = e.current_player();
    int policy_id = roles[env_index][cur].policy_id;

    PolicyRequest req;
    req.env = env_index; req.seat = cur; req.done = 0;
    fill_mask_for_current(e, req.mask.data());
    if (policy_id < 7) {
        // classic obs for C++ bots
        req.classic_obs_len = e.observe_vector(req.classic_obs.data());
    } else {
        int seq_cap = max_sequence_length_for_policy(policy_id);
        prepare_ai_sequence(e, cur, seq_cap, req);
    }
    out[policy_id].push_back(std::move(req));
    return;
  }
}

const std::unordered_map<int, std::vector<PolicyRequest>>& VecArena::collect_requests() {
    pending.clear();

    for (int b = 0; b < B; ++b) {
        if (done[b]) continue;
        advance_env_until_policy_or_done(b, pending);
    }

    return pending;
}

void VecArena::submit_actions(int policy_id, const std::vector<uint8_t>& actions) {
    submit_actions(policy_id, actions.data(), actions.size());
}

void VecArena::submit_actions(int policy_id, const uint8_t* actions, size_t count) {
    auto it = pending.find(policy_id);
    if (it == pending.end()) return;
    auto& reqs = it->second;
    const size_t K = reqs.size();
    const size_t T = std::min(K, count);

    for (size_t i = 0; i < T; ++i) {
        auto& r = reqs[i];
        Env& e = envs[r.env];

        if (e.current_player() != r.seat) {
            continue;
        }
        uint8_t mask[7]; e.valid_actions(mask);
        uint8_t a = actions ? actions[i] : 6;
        if (a > 6 || !mask[a]) a = first_valid(mask);
        bool over = e.step(a);
        if (over) done[r.env] = 1;
    }

    pending.erase(it);
}
