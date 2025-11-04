#pragma once
#include "bare_env.h"
#include "roles.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

static constexpr int DEFAULT_MAX_LEN = 480;
// Base newerest dim = 2 (hand counts) + (MAX_PLAYERS-1) opponent hand sizes + MAX_PLAYERS penalties
static constexpr int BASE_OBS_DIM = 2 + (Env::MAX_PLAYERS - 1) + Env::MAX_PLAYERS;
// Pad to a multiple of 8 to ensure Tensor Core friendly FP16 GEMMs
static constexpr int OBS_DIM = ((BASE_OBS_DIM + 7) / 8) * 8;

struct PolicyRequest {
        int env = -1;    // env index [0..B)
        int seat = -1;   // seat in [0..n_players) - physical seat (may change with shuffling)
        int agent_index = -1;  // stable ID for this agent (use instead of trajectory_id)
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
        int max_sequence_length = DEFAULT_MAX_LEN;
        std::unordered_map<int, int> policy_max_sequence_lengths;
	// State
	std::vector<Env> envs;
	std::vector<uint8_t> done;                   // per-env terminal flag

	// Roles per env per agent_index (stable ID, not physical seat!)
        std::vector<std::unordered_map<int, Role>> roles;  // [env_idx][agent_index] -> Role

        // Pending requests (to match submit_actions)
        std::unordered_map<int, std::vector<PolicyRequest>> pending; // policy_id -> requests (order matters)

	// ---- API ----
        void reset(int B_, int n_players_, uint32_t seed0);
        void set_roles(const std::vector<std::unordered_map<int, Role>>& roles_by_agent_index);
        void set_max_sequence_length(int max_len);
        void set_policy_max_sequence_length(int policy_id, int max_len);

        // Advance everything until any POLICY seat needs an action.
        // Returns grouped requests per policy_id.
        const std::unordered_map<int, std::vector<PolicyRequest>>& collect_requests();

	// Submit batched actions for a specific policy_id (must match order & count of last collect_requests()).
        void submit_actions(int policy_id, const std::vector<uint8_t>& actions);
        void submit_actions(int policy_id, const uint8_t* actions, size_t count);

	// Observation dimensionality for newerest (padded)
	int obs_dim() const { return OBS_DIM; }

private:
	// Helpers
        static uint8_t first_valid(const uint8_t mask[7]);
        static void fill_mask_for_current(const Env& e, uint8_t m[7]);
        void prepare_ai_sequence(const Env& e, int ai_seat, int seq_cap, PolicyRequest& out) const;
        void advance_env_until_policy_or_done(int env_index, std::unordered_map<int, std::vector<PolicyRequest>>& out);
        int max_sequence_length_for_policy(int policy_id) const;
};
