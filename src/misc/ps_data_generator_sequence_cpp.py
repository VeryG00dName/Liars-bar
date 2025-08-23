
#!/usr/bin/env python3
# ps_data_generator_sequence_cpp.py - generate sequence data using C++ Env/Bots/PerfectSearch via pybind
import os, time, logging, random, argparse, pickle, json, datetime
from collections import defaultdict

# --- Import C++ bindings (adjust names if your pybind module differs) ---

from src.misc import lb as cpp  # preferred short name
from tqdm import tqdm

try:
    # Optional config; if absent we use sane defaults
    from src import config as py_config
    DEFAULT_NUM_PLAYERS = getattr(py_config, "NUM_PLAYERS", 4)
except Exception:
    DEFAULT_NUM_PLAYERS = 4

CARD_COUNT_MAPPING = {1: 7, 2: 8, 3: 9}  # for transformed_action field (compat)

def setup_logging(log_file=None, level=logging.INFO):
    logger = logging.getLogger("ps_gen_cpp")
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def create_output_dir(base_dir: str) -> str:
    if os.path.basename(base_dir) == "ps_autoreg_data":
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(base_dir, f"ps_autoreg_data_{ts}")
    else:
        out = base_dir
    os.makedirs(out, exist_ok=True)
    return out

def append_to_data_file(items, file_path: str):
    # Append-friendly: write one pickle per item (no full reload)
    with open(file_path, 'ab') as f:
        for it in items:
            pickle.dump(it, f)

def winner_index(env: "cpp.Env") -> int:
    w, alive = -1, 0
    n = env.num_players()
    for p in range(n):
        if not env.terminations[p]:
            w = p; alive += 1
    return w if alive == 1 else -1

def build_opponent_pool(n_players: int):
    """
    Return a dict of factory lambdas for C++ bots.
    Keys are string tags; values are callables producing bot instances.
    """
    pool = {
        "RandomAgent":               lambda name, **kw: cpp.RandomAgent(name),
        "GreedyCardSpammer":         lambda name, **kw: cpp.GreedyCardSpammer(name),
        "TableFirstConservativeChallenger": lambda name, **kw: cpp.TableFirstConservativeChallenger(name),
        "SelectiveTableConservativeChallenger": lambda name, **kw: cpp.SelectiveTableConservativeChallenger(name),
        "TableNonTableAgent":        lambda name, **kw: cpp.TableNonTableAgent(name),
        "Classic":                   lambda name, **kw: cpp.Classic(name),
        "StrategicChallenger":       lambda name, **kw: cpp.StrategicChallenger(name, n_players, kw.get("agent_index", 0)),
    }
    return pool

def create_belief_vector(opponent_types, current_opponents):
    # For now, belief is just the opponent type tags in seating order
    return [info["name"] for _, info in current_opponents.items()]

def generate_data(
    num_episodes=1000,
    output_dir="./ps_autoreg_data",
    save_frequency=100,
    verbose=False,
    seed=42,
    num_players=DEFAULT_NUM_PLAYERS
):
    """
    Generate training data using C++ PerfectSearch + hardcoded C++ bots.
    Produces one full-game sequence per episode.
    """
    logger = setup_logging(os.path.join(output_dir, 'generation.log'), logging.INFO if verbose else logging.WARNING)

    random.seed(seed)

    main_data_file = os.path.join(output_dir, "ps_autoreg_data.pkl")
    os.makedirs(output_dir, exist_ok=True)

    # Build opponent pool & agent names
    opponent_pool = build_opponent_pool(num_players)
    opponent_types = list(opponent_pool.keys())

    player_names = [f"player_{i}" for i in range(num_players)]
    AGENT_ID_MAP = {name: i for i, name in enumerate(player_names)}

    # Stats
    stats = {
        "episodes": 0, "steps": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
        "total_saved_sequences": 0, "opponent_combinations": defaultdict(int),
        "action_distribution": defaultdict(int),
        "avg_sequence_length": 0.0, "avg_search_time": 0.0, "simulation_count": 0,
        "failed_searches": 0, "start_time": time.time()
    }

    batch = []

    for episode in tqdm(range(num_episodes), desc="Generating games"):
        episode_seed = seed + episode

        # Choose opponents for seats 1..N-1
        chosen = random.sample(opponent_types, num_players - 1)
        stats["opponent_combinations"]["_vs_".join(chosen)] += 1

        # Instantiate C++ Env (random penalty limit happens inside reset)
        env = cpp.Env()
        env.reset(num_players, episode_seed)

        # Instantiate bots per seat
        current_opponents = {}
        bot_objs = [None] * num_players
        for seat in range(num_players):
            name = player_names[seat]
            if seat != 0:
                tag = chosen[seat - 1]
                factory = opponent_pool[tag]
                if tag == "StrategicChallenger":
                    bot = factory(name, agent_index=seat)
                else:
                    bot = factory(name)
                bot_objs[seat] = bot
                current_opponents[name] = {"instance": bot, "name": tag, "type": "hardcoded"}

        ps = cpp.PerfectSearch(0, bot_objs)

        # Game loop
        seq = []
        step_no = 0
        game_over = False
        while not game_over and step_no < 2000:
            step_no += 1
            p = env.current_player()
            mask = env.valid_actions()            # mask for current player (decision)
            obs_legacy = env.observe_vector()     # legacy obs for bots/PS decisions
            obs_train  = env.observe_newerest(0)  # NEW: newerest obs for learning agent (seat 0)

            step = {
                "agent_id": AGENT_ID_MAP[player_names[p]],
                "step": step_no,
                "belief": create_belief_vector(chosen, current_opponents),
                "observation": [round(x, 2) for x in obs_train],  # always log player_0 newerest
            }

            if p == 0:
                step["action_mask"] = list(mask)
                used_plan, a = ps.next_planned_action(p, env)
                if used_plan:
                    step["action_source"] = "PS Plan"
                else:
                    t0 = time.time()
                    a, value = ps.search(env)
                    dt = time.time() - t0
                    stats["avg_search_time"] = (stats["avg_search_time"] * stats["simulation_count"] + dt) / (stats["simulation_count"] + 1)
                    stats["simulation_count"] += 1
                    step["search_value"] = float(value)
                    step["action_source"] = "PS Search"
            else:
                used_plan, a = ps.next_planned_action(p, env)
                if used_plan:
                    step["action_source"] = "PS Plan"
                else:
                    bot = bot_objs[p]
                    if bot is None:
                        a = next((i for i, m in enumerate(mask) if m), 6)
                        step["action_source"] = "Fallback"
                    else:
                        a = bot.act(obs_legacy, len(obs_legacy), mask)
                        if a > 6 or not mask[a]:
                            a = next((i for i, m in enumerate(mask) if m), 6)
                        step["action_source"] = f"Opponent Model ({current_opponents[player_names[p]]['name']})"

            # record distribution and any play metadata
            stats["action_distribution"][f"{a}"] += 1

            step["action"] = a

            # apply step
            game_over = env.step(a)
            seq.append(step)
            stats["steps"] += 1

        # Final outcome
        win_idx = winner_index(env)
        penalties = {player_names[i]: int(env.penalties[i]) for i in range(num_players)}
        result = 100.0 if win_idx == 0 else (-100.0 if win_idx != -1 else 0.0)

        game_data = {
            "game_id": episode,
            "sequence": seq,
            "game_outcome": {
                "winner": (player_names[win_idx] if win_idx != -1 else None),
                "penalties": penalties,
                "result": result
            }
        }

        # Stats update
        if win_idx == 0: stats["wins"] += 1
        elif win_idx != -1: stats["losses"] += 1
        stats["episodes"] += 1
        stats["win_rate"] = stats["wins"] / max(1, stats["episodes"])

        seq_len = len(seq)
        stats["avg_sequence_length"] = ((stats["avg_sequence_length"] * (stats["episodes"] - 1)) + seq_len) / stats["episodes"]

        batch.append(game_data)
        if (episode + 1) % save_frequency == 0:
            append_to_data_file(batch, main_data_file)
            stats["total_saved_sequences"] += len(batch)
            batch = []

    if batch:
        append_to_data_file(batch, main_data_file)
        stats["total_saved_sequences"] += len(batch)

    # Save stats
    stats_file = os.path.join(output_dir, "stats_final.json")
    with open(stats_file, 'w') as f:
        json.dump({k: (dict(v) if isinstance(v, defaultdict) else v) for k,v in stats.items()}, f, indent=2)

    # Summary
    total_time = time.time() - stats["start_time"]
    stats["total_time"] = total_time
    total_sequences = stats["total_saved_sequences"]
    stats["sequences_per_episode"] = total_sequences / max(1, stats["episodes"])
    stats["steps_per_sequence"] = stats["steps"] / max(1, total_sequences)
    stats["episodes_per_second"] = stats["episodes"] / max(1, total_time)

    logger.info("\\n===== Data Generation Summary (C++ backend) =====")
    logger.info(f"Episodes: {stats['episodes']}")
    logger.info(f"Saved sequences: {stats['total_saved_sequences']} ({stats['sequences_per_episode']:.2f}/ep)")
    logger.info(f"Avg seq length: {stats['avg_sequence_length']:.2f}")
    logger.info(f"Win rate: {stats['win_rate']:.4f} ({stats['wins']}/{stats['episodes']})")
    logger.info(f"Total time: {total_time:.2f}s ({stats['episodes_per_second']:.2f} eps/s)")
    logger.info(f"Avg search time: {stats['avg_search_time']:.3f}s over {stats['simulation_count']} searches")

    return stats

def main():
    ap = argparse.ArgumentParser(description="Generate PS data with C++ env/bots via pybind")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--output-dir", type=str, default="./ps_autoreg_data")
    ap.add_argument("--save-frequency", type=int, default=100)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--players", type=int, default=DEFAULT_NUM_PLAYERS)
    args = ap.parse_args()

    out_dir = create_output_dir(args.output_dir)
    stats = generate_data(
        num_episodes=args.episodes,
        output_dir=out_dir,
        save_frequency=args.save_frequency,
        verbose=args.verbose,
        seed=args.seed,
        num_players=args.players
    )

    print(f"\\nData generation complete. Output saved to {out_dir}")
    print(f"Generated {stats['total_saved_sequences']} sequences from {stats['episodes']} episodes")
    print(f"Average sequence length: {stats['avg_sequence_length']:.2f} steps")
    print(f"Win rate: {stats['win_rate']:.4f}")

if __name__ == "__main__":
    main()
