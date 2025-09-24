#pragma once
#include "bare_env.h"
#include "roles.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

static constexpr int MAX_LEN = 255;
static constexpr int OBS_DIM = 2 + (Env::MAX_PLAYERS - 1) + Env::MAX_PLAYERS; // newerest dim

struct PolicyRequest {
        int env = -1;    // env index [0..B)
        int seat = -1;   // seat in [0..n_players)
        std::array<uint8_t, 7> mask{};
        uint8_t done = 0;  // 1 if env already terminal

        // For classic C++ bots
        std::array<float, 3 + Env::MAX_PLAYERS> classic_obs{};
        int   classic_obs_len = 0;

        // For AI models (pre-built sequence)
        std::vector<float> obs_sequence;           // [valid_len, OBS_DIM]
        std::vector<int64_t> action_sequence;      // [valid_len]
        std::vector<int64_t> agent_type_sequence;  // [valid_len]
        std::vector<int64_t> position_sequence;    // [valid_len]
        std::vector<uint8_t> action_mask_sequence; // [valid_len, 7]
        int     valid_len = 0;
};

struct VecArena {
	// Config
	int B = 0;
	int n_players = 4;
	uint32_t base_seed = 0;
	// State
	std::vector<Env> envs;
	std::vector<uint8_t> done;                   // per-env terminal flag

	// Roles per env per seat
        std::vector<std::vector<Role>> roles;

        // Pending requests (to match submit_actions)
        std::unordered_map<int, std::vector<PolicyRequest>> pending; // policy_id -> requests (order matters)

	// ---- API ----
        void reset(int B_, int n_players_, uint32_t seed0);
        void set_roles(const std::vector<std::vector<int>>& policy_ids_per_env);

        // Advance everything until any POLICY seat needs an action.
        // Returns grouped requests per policy_id.
        const std::unordered_map<int, std::vector<PolicyRequest>>& collect_requests();

	// Submit batched actions for a specific policy_id (must match order & count of last collect_requests()).
        void submit_actions(int policy_id, const std::vector<uint8_t>& actions);
        void submit_actions(int policy_id, const uint8_t* actions, size_t count);

	// Observation dimensionality for newerest
	int obs_dim() const { return 2 + (n_players - 1) + n_players; }

private:
	// Helpers
        static uint8_t first_valid(const uint8_t mask[7]);
        static void fill_mask_for_current(const Env& e, uint8_t m[7]);
        void prepare_ai_sequence(const Env& e, int ai_seat, PolicyRequest& out) const;
        void advance_env_until_policy_or_done(int env_index, std::unordered_map<int, std::vector<PolicyRequest>>& out);
};
