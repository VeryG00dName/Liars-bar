
#pragma once
#include <cstdint>
#include <vector>
#include <cmath>

namespace bots {

int round_to_int(float x);

// 0..2 -> play 1..3 table(1s), 3..5 -> play 1..3 non-table(0s), 6 -> challenge

struct GreedyCardSpammer {
  const char* name;
  explicit GreedyCardSpammer(const char* n);
  uint8_t act(const float* obs, int len, const uint8_t mask[7]) const;
};

struct TableFirstConservativeChallenger {
  const char* name;
  explicit TableFirstConservativeChallenger(const char* n);
  uint8_t act(const float* obs, int len, const uint8_t mask[7]) const;
};

struct StrategicChallenger {
  const char* name;
  int num_players;
  int agent_index;
  StrategicChallenger(const char* n, int n_players, int idx);
  uint8_t act(const float* obs, int len, const uint8_t mask[7]) const;
};

struct SelectiveTableConservativeChallenger {
  const char* name;
  explicit SelectiveTableConservativeChallenger(const char* n);
  uint8_t act(const float* obs, int len, const uint8_t mask[7]) const;
};

struct TableNonTableAgent {
  const char* name;
  bool commit_to_table;
  explicit TableNonTableAgent(const char* n);
  uint8_t act(const float* obs, int len, const uint8_t mask[7]);
};

struct Classic {
  const char* name;
  explicit Classic(const char* n);
  uint8_t act(const float* obs, int len, const uint8_t mask[7]) const;
};

struct RandomAgent {
  const char* name;
  uint32_t rng;
  explicit RandomAgent(const char* n);
  void set_seed(uint32_t s);
  static uint32_t xorshift32(uint32_t& s);
  uint8_t act(const float* obs, int len, const uint8_t mask[7]);
};

} // namespace bots
