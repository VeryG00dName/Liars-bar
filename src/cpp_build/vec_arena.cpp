#include "vec_arena.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
// ---------- small helpers ----------
uint8_t VecArena::first_valid(const uint8_t mask[7]) {
    for (int i = 0; i < 7; ++i) if (mask[i]) return (uint8_t)i;
    return 6; // default challenge if somehow none
}

void VecArena::fill_mask_for_current(const Env& e, uint8_t m[7]) {
    e.valid_actions(m);
}

void VecArena::newerest_obs_for_seat(const Env& e, int seat, std::vector<float>& out_obs) const {
    const int D = obs_dim();
    out_obs.resize(D);
    // Env::observe_vector_newerest(agent_index, float* out) returns exactly D entries:
    //   [2 hand comps] + [n_players-1 opponent sizes] + [n_players penalties]
    e.observe_vector_newerest(seat, out_obs.data());
}

uint8_t VecArena::run_bot_turn(Env& e, const BotFn& fn) {
    uint8_t mask[7]; e.valid_actions(mask);

    // Use the classic layout (includes last_action_count at obs[2])
    float obs_buf[3 + Env::MAX_PLAYERS];
    int len = e.observe_vector(obs_buf);

    uint8_t a = 6;
    if (fn) a = fn(obs_buf, len, mask);
    if (a > 6 || !mask[a]) a = first_valid(mask);
    e.step(a);
    return a;
}

static uint32_t simple_mix(uint32_t base, int env, int seat) {
    return base ^ (0x9e3779b9u * (uint32_t)(env + 1)) ^ (0x7f4a7c15u * (uint32_t)(seat + 1));
}

// ---------- API ----------
void VecArena::reset(int B_, int n_players_, uint32_t seed0) {
    B = B_;
    n_players = n_players_;
    base_seed = seed0;
    envs.assign(B, Env{});
    done.assign(B, 0);
    roles.assign(B, std::vector<Role>(n_players));
    bot_fns.assign(B, std::vector<BotFn>(n_players));
    pending.clear();

    for (int b = 0; b < B; ++b) envs[b].reset(n_players, seed0 + (uint32_t)b);
}

VecArena::BotFn VecArena::make_bot_fn(BotKind kind, int env_index, int seat) {
    // Construct a C++ bot object captured by value inside the lambda.
    // For stateful bots (Random, TNT), the lambda MUST be mutable.
    // Signature: uint8_t(const float*, int, const uint8_t[7])
    switch (kind) {
    case BotKind::Classic: {
        bots::Classic bot("Classic");
        return [bot](const float* obs, int len, const uint8_t m[7]) mutable -> uint8_t {
            return bot.act(obs, len, m);
            };
    }
    case BotKind::GreedyCardSpammer: {
        bots::GreedyCardSpammer bot("Greedy");
        return [bot](const float* obs, int len, const uint8_t m[7]) mutable -> uint8_t {
            return bot.act(obs, len, m);
            };
    }
    case BotKind::TableFirstConservativeChallenger: {
        bots::TableFirstConservativeChallenger bot("TFCC");
        return [bot](const float* obs, int len, const uint8_t m[7]) mutable -> uint8_t {
            return bot.act(obs, len, m);
            };
    }
    case BotKind::SelectiveTableConservativeChallenger: {
        bots::SelectiveTableConservativeChallenger bot("STCC");
        return [bot](const float* obs, int len, const uint8_t m[7]) mutable -> uint8_t {
            return bot.act(obs, len, m);
            };
    }
    case BotKind::TableNonTableAgent: {
        bots::TableNonTableAgent bot("TNT");
        return [bot](const float* obs, int len, const uint8_t m[7]) mutable -> uint8_t {
            return bot.act(obs, len, m);
            };
    }
    case BotKind::StrategicChallenger: {
        // Needs num_players + agent_index
        bots::StrategicChallenger bot("Strat", n_players, seat);
        return [bot](const float* obs, int len, const uint8_t m[7]) mutable -> uint8_t {
            return bot.act(obs, len, m);
            };
    }
    case BotKind::RandomAgent: {
        bots::RandomAgent bot("Rand");
        bot.set_seed(simple_mix(base_seed, env_index, seat));
        return [bot](const float* obs, int len, const uint8_t m[7]) mutable -> uint8_t {
            return bot.act(obs, len, m);
            };
    }
    default: {
        bots::Classic bot("Classic");
        return [bot](const float* obs, int len, const uint8_t m[7]) mutable -> uint8_t {
            return bot.act(obs, len, m);
            };
    }
    }
}

void VecArena::set_roles(const std::vector<std::vector<Role>>& roles_per_env) {
    roles = roles_per_env;
    bot_fns.assign(B, std::vector<BotFn>(n_players));

    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < n_players; ++s) {
            if (roles[b][s].type == RoleType::BotCpp) {
                bot_fns[b][s] = make_bot_fn(roles[b][s].bot_kind, b, s);
            }
            else {
                bot_fns[b][s] = BotFn(); // empty; controlled by Python
            }
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
    const Role& r = roles[env_index][cur];
    if (r.type == RoleType::Policy) {
      PolicyRequest req;
      req.env = env_index; req.seat = cur;
      newerest_obs_for_seat(e, cur, req.obs);
      fill_mask_for_current(e, req.mask);
      req.done = 0;
      out[r.policy_id].push_back(std::move(req));
      return; // hand control to Python
    } else {
      (void)run_bot_turn(e, bot_fns[env_index][cur]);
    }
  }
}

std::unordered_map<int, std::vector<PolicyRequest>> VecArena::collect_requests() {
    pending.clear();
    std::unordered_map<int, std::vector<PolicyRequest>> grouped;

    for (int b = 0; b < B; ++b) {
        if (done[b]) continue;
        advance_env_until_policy_or_done(b, grouped);
    }

    // Keep a copy for submit_actions matching
    pending = grouped;
    return grouped;
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