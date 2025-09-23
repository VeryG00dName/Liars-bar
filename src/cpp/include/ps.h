#pragma once
#include <cstdint>
#include <vector>
#include <utility>
#include "bare_env.h"
#include <functional>

struct PerfectSearch {
    using BotFn = std::function<uint8_t(const float* obs, int len, const uint8_t mask[7])>;

    explicit PerfectSearch(int me, const std::vector<BotFn>& bot_fns);

    // ---- configurable knobs ----
    void set_sim_order(const std::vector<uint8_t>& order);
    void set_swap_heuristic(bool enable);
    void set_v5_penalty(bool enable, float value);

    // search API
    uint8_t search(const Env& base, float* out_value = nullptr);
    bool next_planned_action(int agent, const Env& live, uint8_t* action_out);

private:
    int me_;
    std::vector<BotFn> bot_fns_;

    // plan of (agent, action)
    std::vector<std::pair<int, uint8_t>> plan_;
    size_t plan_pos_ = 0;

    // config
    std::vector<uint8_t> sim_order_;          // default: {6,3,5,4,0,2,1}
    bool swap_heuristic_ = true;               // default: enabled
    bool v5_penalize_uncontested_3table_ = false;
    float v5_penalty_value_ = -2000.0f;

    // constants
    static constexpr float WIN_VALUE = 5000.0f;
    static constexpr float LOSE_VALUE = -5000.0f;
    static constexpr float OPP_PENALTY_THRESHOLD = 1500.0f;

    // utils
    uint8_t pick_opponent_action(const Env& e, const BotFn& fn) const;
    inline bool game_over(const Env& e);
    inline int  alive_players(const Env& e);
    inline int  count_ones_from_obs(const Env& e);

    float simulate(Env env, uint8_t first_action, int depth,
        std::vector<std::pair<int, uint8_t>>& seq_out);
};
