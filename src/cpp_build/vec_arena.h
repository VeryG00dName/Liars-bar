#pragma once
#include "bare_env.h"
#include "bots.h"
#include "roles.h"

#include <vector>
#include <functional>
#include <unordered_map>
#include <cstdint>

struct PolicyRequest {
	int env = -1;    // env index [0..B)
	int seat = -1;   // seat in [0..n_players)
	std::vector<float> obs; // newerest: size = 2 + (n_players-1) + n_players
	uint8_t mask[7]{ 0,0,0,0,0,0,0 };
	uint8_t done = 0;  // 1 if env already terminal
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

	// Fast C++ bot functors (per env per seat)
	using BotFn = std::function<uint8_t(const float*, int, const uint8_t[7])>;
	std::vector<std::vector<BotFn>> bot_fns;

	// Pending requests (to match submit_actions)
	std::unordered_map<int, std::vector<PolicyRequest>> pending; // policy_id -> requests (order matters)

	// ---- API ----
	void reset(int B_, int n_players_, uint32_t seed0);
	void set_roles(const std::vector<std::vector<Role>>& roles_per_env);

	// Advance everything until any POLICY seat needs an action.
	// Returns grouped requests per policy_id.
	std::unordered_map<int, std::vector<PolicyRequest>> collect_requests();

	// Submit batched actions for a specific policy_id (must match order & count of last collect_requests()).
	void submit_actions(int policy_id, const std::vector<uint8_t>& actions);

	// Observation dimensionality for newerest
	int obs_dim() const { return 2 + (n_players - 1) + n_players; }

private:
	// Helpers
	static uint8_t first_valid(const uint8_t mask[7]);
	static void fill_mask_for_current(const Env& e, uint8_t m[7]);
	void newerest_obs_for_seat(const Env& e, int seat, std::vector<float>& out_obs) const;
	uint8_t run_bot_turn(Env& e, const BotFn& fn);  // returns action applied
	void advance_env_until_policy_or_done(int env_index, std::unordered_map<int, std::vector<PolicyRequest>>& out);
	BotFn make_bot_fn(BotKind kind, int env_index, int seat);
};