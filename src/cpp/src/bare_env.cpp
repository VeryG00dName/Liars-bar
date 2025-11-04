#include "bare_env.h"
#include <algorithm>

// ---------------- PRNG ----------------
inline uint32_t Env::rnd() {
    uint32_t x = rng;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng = x; return x;
}

// ---------------- Fisher–Yates shuffle ----------------
template<std::size_t N>
inline void Env::shuffle(std::array<uint8_t, N>& a) {
    for (int i = N - 1; i > 0; --i) {
        int j = static_cast<int>(rnd() % static_cast<uint32_t>(i + 1));
        uint8_t t = a[i]; a[i] = a[j]; a[j] = t;
    }
}

// ---------------- Hand helpers ----------------
inline int Env::count_in_hand(int p, uint8_t v) const {
    int c = 0, L = hand_len[p];
    for (int i = 0; i < L; ++i) c += (hands[p][i] == v);
    return c;
}

inline void Env::remove_k(int p, uint8_t v, int k) {
    int L = hand_len[p];
    int w = 0;
    for (int i = 0; i < L; ++i) {
        uint8_t keep = !(k > 0 && hands[p][i] == v);
        if (keep) hands[p][w++] = hands[p][i];
        else --k;
    }
    hand_len[p] = static_cast<uint8_t>(w);
}

// ---------------- Turn helpers ----------------
inline bool Env::advance_to_next_active() {
    for (int step = 0; step < n_players; ++step) {
        cur = (cur + 1) % n_players;
        if (!terminations[cur] && !round_eliminated[cur]) return true;
    }
    return false; // no active players found
}

inline void Env::penalize(int p) {
    if (p < 0) {
        std::fprintf(stderr,
            "[BUG] penalize called with p=%d (pending_claimant may be unset)\n", p);
        return;
    }
    if (++penalties[p] >= penalty_limits[p]) {
        terminations[p] = 1;
    }
}

// ---------------- Round lifecycle ----------------
void Env::start_round() {
    // Optional seat shuffling at round start (except the very first round after reset)
    if (shuffle_seats_each_round && global_step > 0) {
        // Shuffle seats 1..(n_players-1); keep seat 0 fixed
        if (n_players > 1) {
            std::vector<int> seats_to_shuffle;
            seats_to_shuffle.reserve(std::max(0, n_players - 1));
            for (int s = 1; s < n_players; ++s) seats_to_shuffle.push_back(s);
            // Fisher–Yates using Env RNG
            for (int i = static_cast<int>(seats_to_shuffle.size()) - 1; i > 0; --i) {
                int j = static_cast<int>(rnd() % static_cast<uint32_t>(i + 1));
                std::swap(seats_to_shuffle[i], seats_to_shuffle[j]);
            }

            // Build full permutation: logical -> new physical
            // logical seat 0 stays at physical 0; others follow seats_to_shuffle order
            std::array<int, MAX_PLAYERS> perm_full;
            for (int i = 0; i < MAX_PLAYERS; ++i) perm_full[i] = i; // init identity
            perm_full[0] = 0;
            for (int logical = 1; logical < n_players; ++logical) {
                perm_full[logical] = seats_to_shuffle[logical - 1];
            }

            // Apply permutation to agent_index mapping
            std::array<int, MAX_PLAYERS> old_agent_index = agent_index;
            for (int logical = 0; logical < n_players; ++logical) {
                int new_physical = perm_full[logical];
                agent_index[new_physical] = old_agent_index[logical];
            }

            // Apply permutation to persistent per-player state so it moves with agents
            std::array<uint8_t, MAX_PLAYERS> old_penalties = penalties;
            std::array<uint8_t, MAX_PLAYERS> old_terminations = terminations;
            std::array<uint8_t, MAX_PLAYERS> old_limits = penalty_limits;
            for (int logical = 0; logical < n_players; ++logical) {
                int new_physical = perm_full[logical];
                penalties[new_physical] = old_penalties[logical];
                terminations[new_physical] = old_terminations[logical];
                penalty_limits[new_physical] = old_limits[logical];
            }

            // Preserve who should start next round if previously set: map old seat to new seat
            if (starter_hint >= 0 && starter_hint < n_players) {
                starter_hint = perm_full[starter_hint];
            }
        }
    }

    for (int p = 0; p < n_players; ++p) round_eliminated[p] = 0;
    pending_exists = 0; pending_claimant = -1; pending_count = 0; pending_bluff = 0;
    last_action_count = 0;
    force_challenge_mode = 0; force_claimant = -1;

    std::array<uint8_t, 20> deck{};
    for (int i = 0; i < 8; ++i)  deck[i] = 1;
    for (int i = 8; i < 20; ++i) deck[i] = 0;
    shuffle(deck);

    int idx = 0;
    for (int p = 0; p < n_players; ++p) {
        if (terminations[p]) { hand_len[p] = 0; continue; }
        for (int i = 0; i < HAND_CAP; ++i) hands[p][i] = deck[idx++];
        hand_len[p] = HAND_CAP;
    }

    int start_from = (starter_hint >= 0 ? starter_hint : 0);
    cur = start_from;
    for (int t = 0; t < n_players; ++t) {
        if (!terminations[cur]) break;
        cur = (cur + 1) % n_players;
    }
    starter_hint = -1;
}

void Env::reset(int players, uint32_t seed) {
    n_players = players;
    rng = seed ? seed : 0xC001BEEF;
    penalties.fill(0);
    terminations.fill(0);
    round_eliminated.fill(0);
    hand_len.fill(0);
    starter_hint = -1;
    force_challenge_mode = 0;
    force_claimant = -1;

    for (int p = 0; p < n_players; ++p) {
        penalty_limits[p] = static_cast<uint8_t>((rnd() % 6u) + 1u);
    }

    // NEW: clear history and step counter
    global_step = 0;
    game_history.clear();

    start_round();
}

void Env::set_seed(uint32_t seed) { rng = seed ? seed : 0xC001BEEF; }

// ---------------- Observations / Masks ----------------
void Env::valid_actions(uint8_t out[7]) const {
    if (terminations[cur] || round_eliminated[cur]) { for (int i = 0; i < 7; ++i) out[i] = 0; return; }
    const int c1 = count_in_hand(cur, 1);
    const int c0 = count_in_hand(cur, 0);
    out[0] = (c1 >= 1); out[1] = (c1 >= 2); out[2] = (c1 >= 3);
    out[3] = (c0 >= 1); out[4] = (c0 >= 2); out[5] = (c0 >= 3);
    out[6] = pending_exists;
    if (pending_exists && force_challenge_mode && cur != force_claimant) {
        out[0] = out[1] = out[2] = out[3] = out[4] = out[5] = 0;
        out[6] = 1;
    }
}

int Env::observe_vector(float out[3 + MAX_PLAYERS]) const {
    int idx = 0;
    if (cur < 0 || cur >= n_players) return 0; // Safety check
    out[idx++] = static_cast<float>(count_in_hand(cur, 1));
    out[idx++] = static_cast<float>(count_in_hand(cur, 0));
    out[idx++] = static_cast<float>(last_action_count);
    for (int p = 0; p < n_players; ++p) {
        const float live = (!terminations[p] && !round_eliminated[p]) ? static_cast<float>(hand_len[p]) : 0.0f;
        out[idx++] = live;
    }
    return idx;
}

int Env::observe_vector_newerest(int agent_index, float* out) const {
    // Base dimension and padded dimension (to multiple of 8)
    const int baseD = 2 + (n_players - 1) + n_players;
    const int D = ((baseD + 7) / 8) * 8;
    if (out) {
        int idx = 0;

        // 0) Observer's own features (unchanged)
        out[idx++] = static_cast<float>(count_in_hand(agent_index, 1)) / 5.0f;
        out[idx++] = static_cast<float>(count_in_hand(agent_index, 0)) / 5.0f;

        // Build a rotated player order: self first, then next in turn order
        // order[0] == agent_index, order[1] == (agent_index+1) % n_players, ...
        for (int i = 1; i < n_players; ++i) {
            const int p = (agent_index + i) % n_players;
            const int sz = (!terminations[p] && !round_eliminated[p])
                ? static_cast<int>(hand_len[p])
                : 0;
            // 1) Opponent hand sizes in rotated order (self excluded)
            out[idx++] = static_cast<float>(sz) / 5.0f;
        }

        // 2) Penalties in the same rotated order, but include self (observer is first)
        for (int i = 0; i < n_players; ++i) {
            const int p = (agent_index + i) % n_players;
            out[idx++] = static_cast<float>(penalties[p]) / 6.0f;
        }

        // Zero-pad remaining slots up to padded D
        for (; idx < D; ++idx) {
            out[idx] = 0.0f;
        }
    }
    return D;
}

// ---------------- Round / Game checks ----------------
bool Env::round_and_game_checks() {
    // Game over iff ≤1 non-terminated players remain.
    int alive = 0;
    for (int p = 0; p < n_players; ++p)
        if (!terminations[p]) ++alive;
    return (alive <= 1);
}

// ---------------- Step ----------------
bool Env::step(uint8_t a) {
    // ---- History Logging: Capture pre-action state ----
    HistoryEntry H;
    H.player = cur;
    H.action = a;
    H.step = global_step;
    valid_actions(H.mask.data());
    const int baseD = 2 + (n_players - 1) + n_players;
    const int D = ((baseD + 7) / 8) * 8;
    H.observations.resize(n_players, std::vector<float>(D));
    for (int p = 0; p < n_players; ++p) {
        observe_vector_newerest(p, H.observations[p].data());
    }

    // ---- Action Logic ----
    bool game_is_over = false;
    const int player = cur;

    if (a > 6 || !H.mask[a]) { // Invalid action
        penalize(cur);
        if (!advance_to_next_active()) start_round();
        game_is_over = round_and_game_checks();
    }
    else if (a <= 5) { // Play cards
        const int k = (a % 3) + 1;
        const uint8_t want = (a <= 2) ? 1u : 0u;
        remove_k(player, want, k);
        pending_exists = 1; pending_claimant = player;
        pending_count = static_cast<uint8_t>(k); pending_bluff = (want == 0);
        last_action_count = static_cast<uint8_t>(k);

        if (hand_len[player] == 0) {
            int alive = 0; for (int q = 0; q < n_players; ++q) if (!terminations[q] && !round_eliminated[q]) ++alive;
            if (alive == 2) { force_challenge_mode = 1; force_claimant = player; }
            else { round_eliminated[player] = 1; }
        }
        if (!advance_to_next_active()) start_round();
        game_is_over = false;
    }
    else { // a == 6: Challenge
        if (!pending_exists) { // Invalid challenge
            penalize(player);
            if (!advance_to_next_active()) start_round();
            game_is_over = round_and_game_checks();
        }
        else { // Valid challenge
            starter_hint = player;
            if (pending_bluff) {
                penalize(pending_claimant);
            }
            else {
                penalize(player);
            }

            if (round_and_game_checks()) {
                game_is_over = true;
            }
            else {
                start_round();
                game_is_over = false;
            }
        }
    }

    // ---- Finalize Step: Push history ----
    game_history.push_back(std::move(H));
    global_step++;
    return game_is_over;
}

int Env::get_total_history_entries() const {
    return static_cast<int>(game_history.size());
}

std::vector<HistoryEntry> Env::get_history_entries_slice(int start_index, int end_index) const {
    const int total = static_cast<int>(game_history.size());
    if (start_index < 0) start_index = 0;
    if (end_index < start_index) end_index = start_index;
    if (end_index > total) end_index = total;

    std::vector<HistoryEntry> out;
    const int count = end_index - start_index;
    if (count <= 0) {
        return out;
    }
    out.reserve(count);
    for (int i = start_index; i < end_index; ++i) {
        out.push_back(game_history[i]);
    }
    return out;
}

// explicit template instantiation
template void Env::shuffle<20>(std::array<uint8_t, 20>&);
