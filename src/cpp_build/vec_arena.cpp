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

void VecArena::prepare_ai_sequence(const Env& e, int ai_seat, PolicyRequest& out) const {
    const int n_players = e.num_players();
    const int total = (int)e.game_history.size();
    const int start = std::max(0, total - (MAX_LEN - 1));
    int idx = 0;

    auto transform_action = [&](int actor_seat, uint8_t action) -> int64_t {
        int relative_agent_id = (actor_seat - ai_seat + n_players) % n_players;
        if (relative_agent_id == 0) {
            return (int64_t)action;
        } else {
            if (action <= 5) return (int64_t)(7 + (action % 3));
            return (int64_t)action;
        }
    };

    for (int i = start; i < total && idx < MAX_LEN; ++i) {
        const HistoryEntry& h = e.game_history[i];
        const auto& obs = h.observations[ai_seat];
        const int obs_take = std::min<int>(OBS_DIM, static_cast<int>(obs.size()));
        if (obs_take > 0) {
            std::memcpy(out.obs_sequence[idx], obs.data(), sizeof(float) * obs_take);
        }
        int rel = (h.player - ai_seat + n_players) % n_players;
        out.agent_type_sequence[idx] = rel;
        if (idx == 0) {
            out.action_sequence[idx] = 10; // PAD first token
        } else {
            const HistoryEntry& prev_h = e.game_history[i - 1];
            out.action_sequence[idx] = transform_action(prev_h.player, prev_h.action);
        }
        // For opponent rows, store zeros; trainer will only use our rows' masks
        if (rel == 0) {
            std::memcpy(out.action_mask_sequence[idx], h.mask.data(), sizeof(uint8_t) * 7);
        } else {
            std::memset(out.action_mask_sequence[idx], 0, sizeof(uint8_t) * 7);
        }
        out.position_sequence[idx] = idx;
        ++idx;
    }
    // Current step observation
    if (idx < MAX_LEN) {
        float cur_obs[OBS_DIM];
        e.observe_vector_newerest(ai_seat, cur_obs);
        std::memcpy(out.obs_sequence[idx], cur_obs, sizeof(float) * OBS_DIM);
        out.agent_type_sequence[idx] = 0; // me
        if (total > 0) {
            const HistoryEntry& last_h = e.game_history[total - 1];
            out.action_sequence[idx] = transform_action(last_h.player, last_h.action);
        } else {
            out.action_sequence[idx] = 10;
        }
        e.valid_actions(out.action_mask_sequence[idx]);
        out.position_sequence[idx] = idx;
        out.valid_len = idx + 1;
    } else {
        out.valid_len = MAX_LEN;
    }
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
    fill_mask_for_current(e, req.mask);
    if (policy_id < 7) {
        // classic obs for C++ bots
        req.classic_obs_len = e.observe_vector(req.classic_obs);
    } else {
        prepare_ai_sequence(e, cur, req);
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
    auto it = pending.find(policy_id);
    if (it == pending.end()) return;
    auto& reqs = it->second;
    const size_t K = reqs.size();
    const size_t T = std::min(K, actions.size());

    for (size_t i = 0; i < T; ++i) {
        auto& r = reqs[i];
        Env& e = envs[r.env];

        if (e.current_player() != r.seat) {
            continue;
        }
        uint8_t mask[7]; e.valid_actions(mask);
        uint8_t a = actions[i];
        if (a > 6 || !mask[a]) a = first_valid(mask);
        bool over = e.step(a);
        if (over) done[r.env] = 1;
    }

    pending.erase(it);
}
