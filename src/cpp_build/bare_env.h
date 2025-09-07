#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include <string>
struct VecArena;
struct HistoryEntry {
  int player = -1;               // seat index who acted
  uint8_t action = 6;            // 0..6 (0..5 = Play, 6 = Challenge)
  int step = 0;                  // global step counter

  // pre-action newerest observations per agent: obs[agent] is a flat vector<float>
  std::vector<std::vector<float>> observations;

  // pre-action mask for the acting player
  std::array<uint8_t,7> mask{};
};

class Env {
	friend struct VecArena;
public:
  // Tunables
  static constexpr int MAX_PLAYERS = 4;
  static constexpr int HAND_CAP    = 5;   // fixed hand size per round

  // ---- Public state ----
  std::array<std::array<uint8_t, HAND_CAP>, MAX_PLAYERS> hands{};
  std::array<uint8_t, MAX_PLAYERS> hand_len{};
  std::array<uint8_t, MAX_PLAYERS> penalties{};
  std::array<uint8_t, MAX_PLAYERS> terminations{};
  std::array<uint8_t, MAX_PLAYERS> round_eliminated{};

  // ---- NEW: History ----
  int global_step = 0;
  std::vector<HistoryEntry> game_history;

  // ---- Lifecycle ----
  void reset(int players, uint32_t seed = 0xC001BEEF);
  void set_seed(uint32_t seed);
  bool step(uint8_t action);
  void valid_actions(uint8_t out_mask[7]) const;
  int observe_vector(float out[3 + MAX_PLAYERS]) const;
  int observe_vector_newerest(int agent_index, float* out) const;
  int current_player() const { return cur; }
  int num_players()   const { return n_players; }

private:
  int n_players = 0;
  int cur = 0;
  uint32_t rng = 0x9E3779B9u;

  uint8_t pending_exists = 0;
  int     pending_claimant = -1;
  uint8_t pending_count = 0;
  uint8_t pending_bluff = 0;
  uint8_t last_action_count = 0;

  uint8_t force_challenge_mode = 0;
  int8_t  force_claimant = -1;
  int starter_hint = -1;

  std::array<uint8_t, MAX_PLAYERS> penalty_limits{};

  // Helpers
  inline uint32_t rnd();
  template<int N>
  inline void shuffle(std::array<uint8_t,N>& a);
  inline int  count_in_hand(int p, uint8_t v) const;
  inline void remove_k(int p, uint8_t v, int k);
  inline bool advance_to_next_active();
  inline void penalize(int p);
  void start_round();
  bool round_and_game_checks();
};