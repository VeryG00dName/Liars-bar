// ps.cpp
#include "ps.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstring>

// -------- Debug toggle --------
#ifndef LB_PS_DEBUG
#define LB_PS_DEBUG 0   // set to 1 to enable PSDBG prints
#endif

#if LB_PS_DEBUG
#define PSDBG(...) do { std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); std::fflush(stderr); } while(0)
#else
#define PSDBG(...) ((void)0)
#endif

static inline void print_mask7(const char* tag, const uint8_t m[7]) {
    PSDBG("%s mask=[%u %u %u %u %u %u %u]", tag, m[0], m[1], m[2], m[3], m[4], m[5], m[6]);
}

// ---------- ctor & setters ----------
PerfectSearch::PerfectSearch(int me, const std::vector<BotFn>& bot_fns)
    : me_(me), bot_fns_(bot_fns)
{
    // default search order (same semantics as before)
    sim_order_ = { 6, 3, 5, 4, 0, 2, 1 };
    swap_heuristic_ = true;
    v5_penalize_uncontested_3table_ = false;
    v5_penalty_value_ = -2000.0f;
}

void PerfectSearch::set_sim_order(const std::vector<uint8_t>& order) {
    if (!order.empty()) sim_order_ = order;
}
void PerfectSearch::set_swap_heuristic(bool enable) {
    swap_heuristic_ = enable;
}
void PerfectSearch::set_v5_penalty(bool enable, float value) {
    v5_penalize_uncontested_3table_ = enable;
    v5_penalty_value_ = value;
}

// ---------- small utils ----------
inline bool PerfectSearch::game_over(const Env& e) {
    return alive_players(e) <= 1;
}

inline int PerfectSearch::alive_players(const Env& e) {
    int alive = 0;
    for (int p = 0; p < e.num_players(); ++p)
        if (!e.terminations[p]) ++alive;
    return alive;
}

inline int PerfectSearch::count_ones_from_obs(const Env& e) {
    float obs[3 + Env::MAX_PLAYERS];
    int len = e.observe_vector(obs);
    (void)len;
    // observe_vector returns RAW counts: obs[0] = ones in my hand
    return (int)std::lround(obs[0]);
}

uint8_t PerfectSearch::pick_opponent_action(const Env& e, const BotFn& fn) const {
    uint8_t mask[7]; e.valid_actions(mask);
    float obs[3 + Env::MAX_PLAYERS];
    int len = e.observe_vector(obs);

    uint8_t a = 6;
    if (fn) a = fn(obs, len, mask);

    // ensure validity
    if (a > 6 || !mask[a]) {
        for (int i = 0; i < 7; ++i) if (mask[i]) { a = (uint8_t)i; break; }
    }
    return a;
}

// ---------- core simulation ----------
float PerfectSearch::simulate(
    Env env,
    uint8_t first_action,
    int /*depth*/,
    std::vector<std::pair<int, uint8_t>>& seq_out
) {
    PSDBG("[sim] enter first_action=%u curP=%d", (unsigned)first_action, env.current_player());

    const int my_index = me_;

    // Initial ones (table cards) in *my* hand before applying first_action
    const int initial_ones = count_ones_from_obs(env);

    // First action properties
    const bool first_is_non_table = (first_action >= 3 && first_action <= 5);
    const int  first_k = (int)(first_action % 3) + 1;     // 1..3
    const uint8_t first_equiv_table = (first_action >= 3 && first_action <= 5)
        ? (uint8_t)(first_action - 3)            // 3->0, 4->1, 5->2
        : first_action;
    PSDBG("[sim] first_k=%d first_equiv_table=%u initial_ones=%d",
        first_k, (unsigned)first_equiv_table, initial_ones);

    // Plan starts with our first step
    seq_out.clear();
    seq_out.push_back({ my_index, first_action });

    const int my_pen_before = env.penalties[my_index];
    const bool my_term_before = (env.terminations[my_index] != 0);

    // Track last play for heuristics
    int  last_play_by = -1;
    int  last_play_k = 0;     // 1..3
    bool last_play_non_table = false;

    // Apply our first action
    env.step(first_action);
    last_play_by = my_index;
    last_play_k = (int)(first_action % 3) + 1;
    last_play_non_table = (first_action >= 3);

    // immediate outcomes after first action
    if (!my_term_before && env.terminations[my_index]) {
        PSDBG("[sim] self-elim right after first action");
        return LOSE_VALUE;
    }
    if (env.penalties[my_index] > my_pen_before) {
        PSDBG("[sim] self-penalty after first action");
        // constant severity (no env.get_penalty_limit() dependency)
        return (my_pen_before >= 2) ? -5000.0f : -1000.0f;
    }
    if (game_over(env)) {
        int winner = -1, alive = 0;
        for (int p = 0; p < env.num_players(); ++p) if (!env.terminations[p]) { winner = p; ++alive; }
        return (alive == 1 && winner == my_index) ? WIN_VALUE : LOSE_VALUE;
    }

    // main loop: continue until a penalty occurs or game over
    int steps = 0;
    const int MAX_STEPS = 512; // just-in-case guard; typical rounds end on penalty
    while (steps++ < MAX_STEPS) {
        int p = env.current_player();
        PSDBG("[sim] step=%d curP=%d", steps, p);

        if (p == my_index) {
            // Our turn: branch on actions using configured order
            uint8_t mask[7]; env.valid_actions(mask);
            bool any = false; for (int i = 0; i < 7; ++i) any |= (mask[i] != 0);
            if (!any) { PSDBG("[sim] our-turn: no valid -> -50"); return -50.0f; }
            print_mask7("[sim] our-turn", mask);

            float best_v = -std::numeric_limits<float>::infinity();
            std::vector<std::pair<int, uint8_t>> best_seq, tmp_seq;

            for (uint8_t a : sim_order_) {
                if (!mask[a]) continue;
                PSDBG("[sim] recurse a=%u", (unsigned)a);
                Env branch = env; // copy
                float v = simulate(branch, a, /*depth+1*/ 0, tmp_seq);
                if (v >= OPP_PENALTY_THRESHOLD) {
                    // prioritize early opponent penalty
                    seq_out.insert(seq_out.end(), tmp_seq.begin(), tmp_seq.end());
                    PSDBG("[sim] early-exit opponent-penalty via a=%u (v=%.1f)", (unsigned)a, v);
                    return v;
                }
                if (v > best_v) { best_v = v; best_seq = tmp_seq; }
            }
            seq_out.insert(seq_out.end(), best_seq.begin(), best_seq.end());
            return best_v;

        }
        else {
            // Opponent turn
            const BotFn& fn = (p >= 0 && p < (int)bot_fns_.size()) ? bot_fns_[p] : BotFn();
            const int opp_pen_before = env.penalties[p];

            // Choose action
            uint8_t a = pick_opponent_action(env, fn);

            // ----- V5 rule: if our last play was 3 table cards and opponent could challenge but didn't
            if (v5_penalize_uncontested_3table_
                && last_play_by == my_index
                && !last_play_non_table
                && last_play_k == 3) {
                uint8_t mtmp[7]; env.valid_actions(mtmp);
                const bool could_challenge = (mtmp[6] != 0);
                const bool did_not_challenge = (a != 6);
                if (could_challenge && did_not_challenge) {
                    seq_out.push_back({ p, a });
                    PSDBG("[sim] V5 penalty: uncontested 3-table -> %.1f", v5_penalty_value_);
                    return v5_penalty_value_;
                }
            }

            // ----- Swap heuristic: if opponent challenges our non-table bluff and we
            // had enough ones initially to have played table instead, treat as their penalty
            if (swap_heuristic_
                && a == 6
                && last_play_by == my_index
                && last_play_non_table) {
                if (initial_ones >= first_k) {
                    // pretend our first action was table; store the swap in the sequence
                    if (!seq_out.empty() && seq_out[0].first == my_index) {
                        seq_out[0].second = first_equiv_table;
                    }
                    seq_out.push_back({ p, a });
                    PSDBG("[sim] swap-heuristic triggers -> opponent penalized (threshold)");
                    return OPP_PENALTY_THRESHOLD;
                }
                else {
                    // real bluff punished -> strongly negative
                    seq_out.push_back({ p, a });
                    PSDBG("[sim] our bluff caught -> -5000");
                    return -5000.0f;
                }
            }

            // apply opponent action
            uint8_t mask_dbg[7]; env.valid_actions(mask_dbg);
            print_mask7("[sim] opp-turn", mask_dbg);
            PSDBG("[sim] opp p=%d picks a=%u", p, (unsigned)a);

            env.step(a);
            seq_out.push_back({ p, a });

            // immediate penalty checks
            if (env.penalties[p] > opp_pen_before) {
                PSDBG("[sim] opponent penalized immediately");
                return OPP_PENALTY_THRESHOLD;
            }
            if (env.penalties[my_index] > my_pen_before) {
                PSDBG("[sim] we were penalized by opponent action");
                return (my_pen_before >= 2) ? -5000.0f : -1000.0f;
            }

            // update last-play info if opponent played cards
            if (a <= 5) {
                last_play_by = p;
                last_play_k = (int)(a % 3) + 1;
                last_play_non_table = (a >= 3);
            }

            // game over?
            if (game_over(env)) {
                int winner = -1, alive = 0;
                for (int q = 0; q < env.num_players(); ++q) if (!env.terminations[q]) { winner = q; ++alive; }
                PSDBG("[sim] game over on opponent branch (winner=%d alive=%d)", winner, alive);
                return (alive == 1 && winner == my_index) ? WIN_VALUE : LOSE_VALUE;
            }
        }
    }

    // should be very rare
    PSDBG("[sim] safety MAX_STEPS reached -> -10");
    return -10.0f;
}

// ---------- search ----------
uint8_t PerfectSearch::search(const Env& base, float* out_value) {
    plan_.clear(); plan_pos_ = 0;

    uint8_t mask[7]; base.valid_actions(mask);
    float o[3 + Env::MAX_PLAYERS]; (void)base.observe_vector(o);
    PSDBG("[search] me=%d curP=%d players=%d ones=%.0f zeros=%.0f last=%.0f",
        me_, base.current_player(), base.num_players(), o[0], o[1], o[2]);
    print_mask7("[search] our-turn", mask);

    // Build ordered candidates from configurable order
    std::vector<uint8_t> candidates;
    candidates.reserve(7);
    for (uint8_t a : sim_order_) if (mask[a]) candidates.push_back(a);
    if (candidates.empty()) {
        if (out_value) *out_value = -50.0f;
        return 6; // default to challenge
    }

    #if LB_PS_DEBUG
        std::fprintf(stderr, "[search] candidates:");
        for (auto a : candidates) std::fprintf(stderr, " %u", (unsigned)a);
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
    #endif

    float best_v = -std::numeric_limits<float>::infinity();
    uint8_t best_a = candidates[0];
    std::vector<std::pair<int, uint8_t>> best_seq, tmp_seq;

    for (uint8_t a : candidates) {
        Env env_copy = base;
        tmp_seq.clear();
        float v = simulate(env_copy, a, 0, tmp_seq);

        if (v >= OPP_PENALTY_THRESHOLD) {
            best_v = v;
            best_a = tmp_seq.empty() ? a : tmp_seq.front().second; // respect swap-heuristic
            best_seq = tmp_seq;
            PSDBG("[search] early-exit on opponent-penalty via a=%u (v=%.1f)", (unsigned)best_a, v);
            break;
        }
        if (v > best_v) { best_v = v; best_a = a; best_seq = tmp_seq; }
    }

    // store plan (excluding our first step)
    if (!best_seq.empty()) {
        uint8_t returned_first = best_seq.front().second;
        if (returned_first != best_a) best_a = returned_first;
        if (best_seq.size() > 1) {
            plan_.assign(best_seq.begin() + 1, best_seq.end());
            PSDBG("[search] plan_len=%zu", plan_.size());
        }
    }
    else {
        plan_.clear();
    }

    if (out_value) *out_value = best_v;
    return best_a;
}

// ---------- plan follow ----------
bool PerfectSearch::next_planned_action(int agent, const Env& live, uint8_t* action_out) {
    if (!action_out) return false;
    if (plan_pos_ >= plan_.size()) return false;

    auto exp = plan_[plan_pos_];
    if (exp.first != agent) { plan_.clear(); plan_pos_ = 0; return false; }

    uint8_t mask[7]; live.valid_actions(mask);
    if (exp.second > 6 || !mask[exp.second]) { plan_.clear(); plan_pos_ = 0; return false; }

    *action_out = exp.second;
    ++plan_pos_;
    return true;
}
