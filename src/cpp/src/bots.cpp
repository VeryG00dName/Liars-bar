
#include "bots.h"
#include <algorithm>

namespace bots {

int round_to_int(float x) { return static_cast<int>(std::lround(x)); }

// -------- GreedyCardSpammer --------
GreedyCardSpammer::GreedyCardSpammer(const char* n) : name(n) {}
uint8_t GreedyCardSpammer::act(const float* obs, int /*len*/, const uint8_t mask[7]) const {
  int table_cards = round_to_int(obs[0]);
  int non_table_cards = round_to_int(obs[1]);
  for (int action : {5,4,3}) { int need = action - 2; if (mask[action] && non_table_cards >= need) return (uint8_t)action; }
  for (int action : {2,1,0}) { int need = action + 1; if (mask[action] && table_cards >= need) return (uint8_t)action; }
  return 6;
}

// -------- TableFirstConservativeChallenger --------
TableFirstConservativeChallenger::TableFirstConservativeChallenger(const char* n) : name(n) {}
uint8_t TableFirstConservativeChallenger::act(const float* obs, int /*len*/, const uint8_t mask[7]) const {
  int table_count = round_to_int(obs[0]);
  int non_table_count = round_to_int(obs[1]);
  int total = table_count + non_table_count;
  if (total == 1) return 6;
  for (int action : {2,1,0}) {
    int required = (action % 3) + 1;
    if (mask[action] && table_count >= required) return (uint8_t)action;
  }
  if (mask[3] && non_table_count >= 1) return 3;
  return 6;
}

// -------- StrategicChallenger --------
StrategicChallenger::StrategicChallenger(const char* n, int n_players, int idx)
: name(n), num_players(n_players), agent_index(idx) {}
uint8_t StrategicChallenger::act(const float* obs, int len, const uint8_t mask[7]) const {
  int table_cards = round_to_int(obs[0]);
  int non_table_cards = round_to_int(obs[1]);
  int last_action_count = round_to_int(obs[2]);
  if (last_action_count >= 2) return 6;

  std::vector<int> active_counts;
  active_counts.reserve(num_players);
  for (int i = 0; i < num_players && (3 + i) < len; ++i)
    active_counts.push_back(round_to_int(obs[3 + i]));
  std::vector<int> nonzero;
  for (int c : active_counts) if (c > 0) nonzero.push_back(c);
  if ((int)nonzero.size() == 2) {
    int my_cards = (agent_index < (int)active_counts.size()) ? active_counts[agent_index] : 0;
    int opponent_cards = 0; for (int c : active_counts) opponent_cards += c; opponent_cards -= my_cards;
    if (my_cards == 2 && opponent_cards == 1) return 6;
  }
  if (mask[3] && non_table_cards >= 1) return 3;
  if (mask[0] && table_cards >= 1) return 0;
  return 6;
}

// -------- SelectiveTableConservativeChallenger --------
SelectiveTableConservativeChallenger::SelectiveTableConservativeChallenger(const char* n) : name(n) {}
uint8_t SelectiveTableConservativeChallenger::act(const float* obs, int /*len*/, const uint8_t mask[7]) const {
  int table_count = round_to_int(obs[0]);
  int non_table_count = round_to_int(obs[1]);
  int total = table_count + non_table_count;
  if (total == 1) return 6;
  if (table_count == 1) {
    if (non_table_count > 1) { if (mask[3]) return 3; }
    else if (non_table_count == 1) { if (mask[0]) return 0; }
  }
  for (int action : {2,1,0}) {
    int required = (action % 3) + 1;
    if (mask[action] && table_count >= required) return (uint8_t)action;
  }
  if (mask[3] && non_table_count >= 1) return 3;
  return 6;
}

// -------- TableNonTableAgent --------
TableNonTableAgent::TableNonTableAgent(const char* n) : name(n), commit_to_table(false) {}
uint8_t TableNonTableAgent::act(const float* obs, int /*len*/, const uint8_t mask[7]) {
  int table_cards = round_to_int(obs[0]);
  int non_table_cards = round_to_int(obs[1]);

  if (table_cards >= 3 && mask[2]) { commit_to_table = false; return 2; }
  if (commit_to_table) {
    if (table_cards > 0 && mask[0]) return 0;
    commit_to_table = false;
  }
  if (table_cards == 2 && mask[0]) { commit_to_table = true; return 0; }
  if (non_table_cards >= 3 && mask[4]) return 4;
  if (table_cards == 1) {
    if (non_table_cards > 0 && mask[3]) return 3;
    if (mask[0]) return 0;
  }
  if (table_cards == 0) {
    if (non_table_cards >= 1 && mask[3]) return 3;
  }
  return 6;
}

// -------- Classic (Liar's Bar logic) --------
Classic::Classic(const char* n) : name(n) {}
uint8_t Classic::act(const float* obs, int len, const uint8_t mask[7]) const {
  int table_cards = round_to_int(obs[0]);
  int non_table_cards = round_to_int(obs[1]);
  int my_cards = table_cards + non_table_cards;
  int last_action_count = round_to_int(obs[2]);

  int total_active_cards = 0;
  int active_players = 0;
  for (int i=3; i<len; ++i) {
    int c = round_to_int(obs[i]);
    if (c > 0) { ++active_players; total_active_cards += c; }
  }

  if (mask[6] && last_action_count > 1) return 6;
  if (mask[6] && active_players == 2) {
    int opponent_cards = total_active_cards - my_cards;
    if (opponent_cards == 1) return 6;
  }
  if (table_cards > 0 && mask[0]) return 0;
  if (non_table_cards > 0 && mask[3]) return 3;
  return 6;
}

// -------- RandomAgent --------
RandomAgent::RandomAgent(const char* n) : name(n), rng(0xDEADBEEF) {}
void RandomAgent::set_seed(uint32_t s) { rng = s ? s : 0xDEADBEEF; }
uint32_t RandomAgent::xorshift32(uint32_t& s) {
  uint32_t x = s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; s = x; return s;
}
uint8_t RandomAgent::act(const float* /*obs*/, int /*len*/, const uint8_t mask[7]) {
  int ids[7], m=0;
  for (int i=0;i<7;++i) if (mask[i]) ids[m++]=i;
  if (m==0) return 6;
  int pick = (int)(xorshift32(rng) % (uint32_t)m);
  return (uint8_t)ids[pick];
}

// -------- ParametricBot --------
ParametricBot::ParametricBot(const char* n) 
  : name(n), rng(0xDEADBEEF), has_personality(false), 
    bluff_aggressiveness(0.0f), volume_aggressiveness(0.0f), challenge_threshold(0.0f) {}

void ParametricBot::set_seed(uint32_t s) { rng = s ? s : 0xDEADBEEF; }

void ParametricBot::reset_personality() {
  auto rand_float = [&](uint32_t& s) {
    return (float)(RandomAgent::xorshift32(s)) / 4294967295.0f;
  };
  bluff_aggressiveness = rand_float(rng);
  volume_aggressiveness = rand_float(rng);
  challenge_threshold = rand_float(rng);
  has_personality = true;
}

uint8_t ParametricBot::act(const float* obs, int /*len*/, const uint8_t mask[7]) {
  if (!has_personality) reset_personality();

  int last_action_count = round_to_int(obs[2]);
  
  // 1. Challenge logic
  if (mask[6]) {
    float challenge_prob = (last_action_count / 5.0f) * challenge_threshold;
    float r = (float)(RandomAgent::xorshift32(rng)) / 4294967295.0f;
    if (r < challenge_prob) return 6;
  }

  // 2. Action scoring
  std::vector<int> possible_actions;
  for (int i=0; i<6; ++i) if (mask[i]) possible_actions.push_back(i);
  
  if (possible_actions.empty()) return 6;

  int best_action = -1;
  float best_score = -1e9f;

  for (int action : possible_actions) {
    float score = 0.0f;
    bool is_bluff = (action >= 3);
    int card_count = 0;
    if (action == 0 || action == 3) card_count = 1;
    else if (action == 1 || action == 4) card_count = 2;
    else if (action == 2 || action == 5) card_count = 3;

    // Bluff bias
    if (is_bluff) score += bluff_aggressiveness * 10.0f;
    else score += (1.0f - bluff_aggressiveness) * 10.0f;

    // Volume bias
    float norm_count = (card_count - 1.0f) / 2.0f;
    float dist_score = 1.0f - std::abs(norm_count - volume_aggressiveness);
    score += dist_score * 5.0f;

    // Noise
    float noise = (float)(RandomAgent::xorshift32(rng)) / 4294967295.0f;
    score += noise * 2.0f;

    if (score > best_score) {
      best_score = score;
      best_action = action;
    }
  }
  
  return (uint8_t)best_action;
}

} // namespace bots
