#src/tests/agent_sequence_debug.py
"""Generate PS games and compare AutoregressiveAgentFull perception against
training pipeline labels, across many episodes with a summary at the end.
"""
import argparse
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
from typing import Dict, List
import io
import contextlib
from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import trange
from tqdm import tqdm
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.env.liars_deck_env_utils_2 import decode_action
from src import config

# Opponent models and PerfectSearch utilities
from src.model.hard_coded_agents import (
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    StrategicChallenger,
    SelectiveTableConservativeChallenger,
    RandomAgent,
    TableNonTableAgent,
    Classic,
)
from src.training.train_utils import load_specific_historical_models
from src.model.ps_v3 import PerfectSearch

# Agent and training dataset utilities
from src.agents.autoregressive_agent_full import AutoregressiveAgentFull
from src.training.train_autoregressive_model_full import (
    collate_variable_length_sequences,
    create_opponent_mapping,
)

AGENT_ID_MAP = {"player_0": 0, "player_1": 1, "player_2": 2}
CARD_COUNT_MAPPING = {1: 7, 2: 8, 3: 9}
TRANSFORM_MAP = {
    0: 7, 3: 7,
    1: 8, 4: 8,
    2: 9, 5: 9,
    6: 6,   # keep challenge as-is
    10: 6,  # normalize if 10 ever appears
}

def setup_logging(level=logging.INFO):
    logger = logging.getLogger("PSSequenceDebugger")
    logger.setLevel(level)
    if not logger.handlers:
        fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger

def load_opponent_pool(include_historical=True):
    pool = {
        "RandomAgent": RandomAgent,
        "GreedyCardSpammer": GreedyCardSpammer,
        "TableFirstConservativeChallenger": TableFirstConservativeChallenger,
        "SelectiveTableConservativeChallenger": SelectiveTableConservativeChallenger,
        "TableNonTableAgent": TableNonTableAgent,
        "StrategicChallenger": StrategicChallenger,
        "Classic": Classic,
    }
    if include_historical:
        try:
            models = load_specific_historical_models(config.HISTORICAL_MODEL_DIR, "cpu")
            for model_instance, identifier in models:
                pool[f"Historical_{identifier}"] = model_instance
        except Exception as e:  # best effort
            logging.getLogger("PSSequenceDebugger").warning(
                f"Failed loading historical models: {e}")
    return pool

def setup_opponents(opponent_pool, opponent_types, agent_names):
    current = {}
    models = {}
    for agent_name, opponent_type in zip(agent_names, opponent_types):
        opponent_cls = opponent_pool[opponent_type]
        if opponent_type.startswith("Historical_"):
            inst = opponent_cls
        else:
            if opponent_type == "StrategicChallenger":
                agent_index = int(agent_name.split("_")[1])
                inst = opponent_cls(agent_name=agent_name,
                                    num_players=config.NUM_PLAYERS,
                                    agent_index=agent_index)
            else:
                inst = opponent_cls(agent_name=agent_name)
        current[agent_name] = {"instance": inst, "name": opponent_type}
        models[agent_name] = inst
    return current, models

def create_belief_vector(current_opponents):
    return [info["name"] for _, info in current_opponents.items()]

class AutoregressiveGameDataset(Dataset):
    """
    Dataset for sequence-based autoregressive game model training.
    Processes round sequences into tensors for model training, handling
    variable-length sequences and using externally provided belief vectors.
    """

    def __init__(
        self,
        data,
        opponent_mapping,
        num_opponent_types,
        device,
        max_seq_length=100,
    ):
        self.sequences = []
        self.opponent_mapping = opponent_mapping
        self.num_opponent_types = num_opponent_types
        self.device = device
        self.max_seq_length = max_seq_length
        
        TRANSFORM_MAP = {0:7, 3:7, 1:8, 4:8, 2:9, 5:9}

        # Debug counters
        self.obs_trimmed_count = 0
        self.total_sequences = 0
        self.sequence_lengths = []

        def convert_old_obs_to_new(obs_7d, agent_id=0):
            hand_vec = obs_7d[:2]
            hand_sizes = obs_7d[4:]
            opp_hand_sizes = [hand_sizes[i] for i in range(3) if i != agent_id]
            return np.round(np.concatenate([hand_vec, opp_hand_sizes]).astype(np.float32), 2)

        for round_data in data:
            sequence = round_data["sequence"]
            seq_len = len(sequence)
            if seq_len > max_seq_length:
                continue

            self.total_sequences += 1
            self.sequence_lengths.append(seq_len)

            # Detect "final step is a self-challenge" once per sequence
            last = sequence[-1] if seq_len > 0 else None
            final_is_self_challenge = bool(
                last and (last.get("agent_id", 0) == 0) and (last.get("action") in (6, 10))
            )

            raw_actions = []
            raw_target_actions = []

            for i, step in enumerate(sequence):
                is_train = step.get("is_training_agent", step.get("agent_id", 0) == 0)

                if "action" in step:
                    a = step["action"]
                    b = step["action"]
                elif is_train and "expert_action" in step:
                    a = step["chosen_action"]
                    b = step["expert_action"]
                else:
                    a = 0
                    b = 0

                # Transform opponents’ plays to 7/8/9 for inputs
                if not is_train and a not in (6, 10):
                    a = TRANSFORM_MAP.get(a, a)

                # Normalize challenge 10 -> 6 defensively
                a = 6 if a == 10 else a
                b = 6 if b == 10 else b

                # Retro-correct previous token when a challenge appears,
                # EXCEPT when this is the final step AND it is a self-challenge
                if a == 6 and raw_actions:
                    do_retro = True
                    if final_is_self_challenge and (i == seq_len - 1) and is_train:
                        do_retro = False
                    if do_retro:
                        raw_actions[-1] = raw_target_actions[-1]

                raw_target_actions.append(b)
                raw_actions.append(a)

            PAD = 0
            input_actions = [PAD] + raw_actions[:-1]
            target_actions = raw_target_actions.copy()
            
            obs_list = []
            action_mask_list = []
            agent_type_list = []
            position_list = []
            belief_list = []
            has_belief = False
            latest_belief_vector = None

            LABELS = {
                "GreedyCardSpammer": 1, "StrategicChallenger": 4,
                "TableNonTableAgent": 6, "Classic": 0,
                "TableFirstConservativeChallenger": 5,
                "SelectiveTableConservativeChallenger": 3,
                "RandomAgent": 2,
                "Historical_Version_E_player_1": 9,
                "Historical_Version_C_player_0": 8,
                "Historical_Version_A_player_2": 7
            }

            for i, step in enumerate(sequence):
                # Determine agent type based on ID
                # 0: Training Agent, 1: Opponent 0, 2: Opponent 1
                agent_id = step.get("agent_id", 0)
                agent_type_list.append(agent_id)
                position_list.append(i)

                # Observations are only provided for the training agent (ID 0)
                if agent_id == 0:
                    obs = np.array(step["observation"], dtype=np.float32)
                    if obs.shape[0] == 7:
                        obs = convert_old_obs_to_new(obs, agent_id=0)
                    elif obs.shape[0] != 4:
                        print(f"⚠️ Unexpected obs shape at step {i}: {obs.shape}, skipping sequence.")
                        obs = np.zeros(4, dtype=np.float32)
                        self.obs_trimmed_count += 1
                    obs_list.append(obs)
                else:
                    obs_list.append(np.zeros(4, dtype=np.float32))

                # Action masks are only for the training agent
                if agent_id == 0 and "action_mask" in step:
                    action_mask_list.append(step["action_mask"])
                else:
                    action_mask_list.append([0] * 7)

                # Belief targets (agent types for opponent 0 and 1)
                if "belief" in step:
                    has_belief = True
                    names = step["belief"]
                    full_belief = []

                    # We expect belief to be a list of 2 opponent names
                    for opp_idx in range(2):
                        if opp_idx < len(names):
                            name = names[opp_idx]
                            idx = LABELS.get(name, 0)
                            full_belief.append(idx)
                        else:
                            # Missing opponent → fallback to 0 (Classic)
                            full_belief.append(0)

                    latest_belief_vector = np.array(full_belief, dtype=np.int64)  # shape: [2]
                    belief_list.append(latest_belief_vector)

                elif has_belief and latest_belief_vector is not None:
                    belief_list.append(latest_belief_vector)

            # Convert to tensors
            obs_tensor        = torch.tensor(np.stack(obs_list),       dtype=torch.float32, device=device)
            action_tensor     = torch.tensor(input_actions,            dtype=torch.long,    device=device)
            target_tensor     = torch.tensor(target_actions,           dtype=torch.long,    device=device)
            mask_tensor       = torch.tensor(np.array(action_mask_list), dtype=torch.bool,  device=device)
            agent_type_tensor = torch.tensor(agent_type_list,          dtype=torch.long,    device=device)
            position_tensor   = torch.tensor(position_list,            dtype=torch.long,    device=device)
            
            belief_tensor = None
            if has_belief and latest_belief_vector is not None:
                belief_tensor = torch.tensor(np.stack(belief_list), dtype=torch.long, device=device)
            
            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1
            )

            seq_dict = {
                "obs":            obs_tensor,
                "action":         action_tensor,
                "target_action":  target_tensor,
                "action_mask":    mask_tensor,
                "agent_type":     agent_type_tensor,
                "position":       position_tensor,
                "attention_mask": attention_mask,
                "length":         seq_len,
                "round_id":       round_data.get("round_id", round_data.get("game_id", None)),
                "belief":         belief_tensor
            }

            self.sequences.append(seq_dict)

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def compare_tensors(agent_tensor: torch.Tensor, truth_tensor: torch.Tensor, name: str, *, quiet: bool=False) -> bool:
    """Compare two tensors and log differences; optionally quiet."""
    try:
        if agent_tensor.shape != truth_tensor.shape:
            logging.error(f"{name}: shape mismatch {agent_tensor.shape} vs {truth_tensor.shape}")
            if not quiet:
                print(f"\n=== {name} AGENT TENSOR ===\n{agent_tensor}")
                print(f"\n=== {name} TRUTH TENSOR ===\n{truth_tensor}")
            return False

        if not torch.allclose(agent_tensor.cpu(), truth_tensor.cpu(), atol=1e-5):
            logging.error(f"{name}: value mismatch")
            if not quiet:
                print(f"\n=== {name} AGENT TENSOR ===\n{agent_tensor}")
                print(f"\n=== {name} TRUTH TENSOR ===\n{truth_tensor}")
            return False

    except RuntimeError as e:
        logging.exception(f"{name}: tensor comparison failed with error: {e}")
        if not quiet:
            print(f"\n=== {name} AGENT TENSOR ===\n{agent_tensor}")
            print(f"\n=== {name} TRUTH TENSOR ===\n{truth_tensor}")
        raise
    return True

def compare_histories(agent_hist: List[Dict[str, any]], game_seq: List[Dict[str, any]]) -> bool:
    """History check with end-of-episode challenge handling & guards."""
    ok = True
    history_to_compare = agent_hist[:-1]
    game_seq_to_compare = game_seq[:-1]

    if len(history_to_compare) != len(game_seq_to_compare):
        logging.warning(f"Compared history length {len(history_to_compare)} != game data length {len(game_seq_to_compare)}")

    # Is the final step a training-agent challenge?
    def is_training_challenge(step):
        return step is not None and step.get("agent_id") == 0 and step.get("action") in (6, 10)
    last_is_training_challenge = bool(game_seq and is_training_challenge(game_seq[-1]))

    for i, (h, g) in enumerate(zip(history_to_compare, game_seq_to_compare)):
        hid = AGENT_ID_MAP.get(h.get("agent_id_env"))
        gid = g.get("agent_id")
        if hid != gid:
            logging.warning(f"Step {i}: agent_id mismatch {hid} != {gid}")
            ok = False

        # Look-ahead on FULL trimmed seq
        next_step = game_seq[i + 1] if i + 1 < len(game_seq) else None
        next_is_challenge = next_step is not None and next_step.get("action") in (6, 10)

        # If next is challenge: normally de-transform, except if it's the final training challenge
        use_transformed = True
        if next_is_challenge:
            if last_is_training_challenge and (i + 1 == len(game_seq) - 1) and next_step.get("agent_id") == 0:
                use_transformed = True
            else:
                use_transformed = False

        true_action = g.get("transformed_action", g.get("action")) if use_transformed else g.get("action")
        agent_action = h.get("action")
        if agent_action != true_action:
            logging.warning(f"Step {i}: action mismatch Agent={agent_action} != Truth={true_action} (orig: {g.get('action')})")
            ok = False

        # Only compare obs/mask when BOTH sides are player 0 and the fields exist
        if hid == 0 and gid == 0:
            obs_a = h.get("observation"); obs_b = g.get("observation")
            if obs_a is not None and obs_b is not None:
                obs_a = np.array(obs_a, dtype=np.float32)
                obs_b = np.array(obs_b, dtype=np.float32)
                if not np.allclose(obs_a, obs_b, atol=1e-2):
                    logging.warning(f"Step {i}: observation mismatch {obs_a} vs {obs_b}")
                    ok = False
            mask_a = h.get("action_mask"); mask_b = g.get("action_mask")
            if mask_a is not None and mask_b is not None:
                mask_a = np.array(mask_a, dtype=np.int64)
                mask_b = np.array(mask_b, dtype=np.int64)
                if not np.array_equal(mask_a, mask_b):
                    logging.warning(f"Step {i}: action_mask mismatch {mask_a} vs {mask_b}")
                    ok = False
    return ok

def run_episode(env, ps, agent, current_opponents, selected_opponents):
    training_agent = "player_0"
    game_data = {"game_id": 0, "sequence": []}
    step = 0
    while not all(env.terminations.values()):
        step += 1
        current_agent = env.agent_selection
        step_data = {"agent_id": AGENT_ID_MAP[current_agent], "step": step}
        step_data["belief"] = create_belief_vector(current_opponents)
        if current_agent == training_agent:
            obs_curr = env.observe(current_agent, newest=True)[current_agent]
            step_data["observation"] = np.round(obs_curr, 2).tolist()
            step_data["action_mask"] = env.infos[current_agent].get("action_mask", [0] * 7)
            # allow the autoregressive agent to process this step
            agent_picked_action = agent.get_action(env, current_agent, obs_curr, env.infos[current_agent], {})
            step_data["model_action"] = int(agent_picked_action)
            planned = ps.get_next_agent_action(current_agent)
            if planned is not None:
                best_action = planned
            else:
                current_state = env.get_state()
                _, best_action, _ = ps.search(current_state)
            step_data["action"] = best_action
            action_type, _, count = decode_action(best_action)
            if action_type == "Play" and count is not None:
                step_data["card_count"] = count
            env.step(best_action)
        else:
            planned = ps.get_next_agent_action(current_agent)
            if planned is not None:
                best_action = planned
            else:
                opp_model = current_opponents[current_agent]["instance"]
                obs_opp = env.observe(current_agent, newer=True)[current_agent]
                mask = env.infos[current_agent]["action_mask"]
                if hasattr(opp_model, "play_turn"):
                    best_action = opp_model.play_turn(obs_opp, mask, table_card=env.table_card)
                else:
                    best_action = mask.index(1)
            step_data["action"] = best_action
            step_data["transformed_action"] = TRANSFORM_MAP.get(best_action, best_action)
            action_type, _, count = decode_action(best_action)
            if action_type == "Play" and count is not None:
                step_data["card_count"] = count
            env.step(best_action)
        game_data["sequence"].append(step_data)
    game_data["game_outcome"] = {"winner": env.winner}
    return game_data

# ---- Helper to mirror dataset action construction (retro-correct + shift) ----
def build_actions_like_dataset(seq):
    PAD = 0
    raw_actions = []
    raw_target_actions = []

    for step in seq:
        aid = step.get("agent_id", 0)
        a = step["action"]; b = step["action"]
        if aid != 0 and a not in (6, 10):
            a = TRANSFORM_MAP.get(a, a)
        if a in (6, 10) and raw_actions:
            raw_actions[-1] = raw_target_actions[-1]
        raw_target_actions.append(6 if b == 10 else b)
        raw_actions.append(6 if a == 10 else a)

    input_actions = [PAD] + raw_actions[:-1]
    return input_actions

def build_truth_batch_quiet(game_data, opponent_mapping, device, max_seq_length):
    """Create dataset & batch while silencing its prints/tqdm."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        dataset = AutoregressiveGameDataset(
            data=[game_data],
            opponent_mapping=opponent_mapping,
            num_opponent_types=len(opponent_mapping),
            device=device,
            max_seq_length=max_seq_length,
        )
        batch = collate_variable_length_sequences([dataset[0]])
    return batch

def main():
    parser = argparse.ArgumentParser(description="PS generator with agent debug")
    parser.add_argument("--agent-checkpoint", required=True, help="Path to AR agent checkpoint")
    parser.add_argument("--data-dir", default="./ps_autoreg_data", help="Directory for opponent mapping")
    parser.add_argument("--max-seq-length", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None,
                        help="If set, Episode 1 uses seed=42+seed; Episode k uses 42+seed+(k-1). "
                             "If unset, seed=42+k.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.INFO if args.verbose else logging.WARNING
    logger = setup_logging(level)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = AutoregressiveAgentFull(device=device, player_id="player_0")
    if os.path.exists(args.agent_checkpoint):
        ckpt = torch.load(args.agent_checkpoint, map_location=device, weights_only=False)
        key = next((k for k in ckpt.get("policy_nets", {}) if "autoregressive" in k.lower()), "policy_net_0")
        state_dict_source = ckpt.get("model_state_dict", ckpt)
        agent.load_models_from_checkpoint({"policy_nets": {key: state_dict_source}}, key)
    else:
        logger.error("Checkpoint not found")
        return

    opponent_pool = load_opponent_pool(include_historical=False)
    opponent_types = list(opponent_pool.keys())
    opponent_agent_names = ["player_1", "player_2"]

    # Build opponent mapping ONCE
    opponent_mapping = create_opponent_mapping(args.data_dir)

    episodes_failed: Dict[int, List[str]] = {}
    quiet = not args.verbose
    
    global_match = 0
    global_total = 0
    
    for ep_idx in trange(args.episodes, desc="Episodes"):
        episode_num = ep_idx + 1
        # Seed policy:
        # - if args.seed is provided: seed = 42 + args.seed + (episode_num)   (lets you replay "Episode N" by passing N)
        # - else: seed = 42 + episode_num
        if args.seed is None:
            seed = 42 + episode_num
        else:
            seed = 42 + args.seed

        if args.verbose:
            logger.info(f"=== Episode {episode_num} | Seed {seed} ===")

        # Per-episode seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        selected = random.sample(opponent_types, len(opponent_agent_names))
        current_opponents, opponent_models = setup_opponents(opponent_pool, selected, opponent_agent_names)

        env = LiarsDeckEnv(num_players=config.NUM_PLAYERS)
        obs, infos = env.reset(seed=seed)
        ps = PerfectSearch(env=env, training_agent="player_0", opponent_models=opponent_models)
        game_data = run_episode(env, ps, agent, current_opponents, selected)

        # Trim after last agent_id == 0
        last_agent0_index = max(i for i, s in enumerate(game_data["sequence"]) if s.get("agent_id") == 0)
        game_data["sequence"] = game_data["sequence"][: last_agent0_index + 1]

        # --- Model vs PS agreement (our turns only) ---
        our_steps = [s for s in game_data["sequence"] if s.get("agent_id") == 0 and "model_action" in s]
        matched = sum(int(s["model_action"] == s["action"]) for s in our_steps)
        total = len(our_steps)
        ep_acc = (matched / total) if total else float("nan")
        if args.verbose:
            print(f"[Episode {episode_num}] Model vs PS (our turns): {matched}/{total} = {ep_acc:.2%}")
        
        global_match += matched
        global_total += total
        
        # Sync speculative last self action
        if agent.sequence_history:
            hist_last = agent.sequence_history[-1]
            game_last = game_data["sequence"][-1]
            hid = AGENT_ID_MAP.get(hist_last.get("agent_id_env"))
            if hid == 0 and hist_last.get("action") != game_last.get("action"):
                hist_last["action"] = game_last.get("action")

        # history check
        _ = compare_histories(agent.sequence_history, game_data["sequence"])

        # Build dataset batch (truth) quietly
        truth_batch = build_truth_batch_quiet(
            game_data,
            opponent_mapping=opponent_mapping,
            device=device,
            max_seq_length=args.max_seq_length,
        )

        # Build agent inputs, then overwrite action_sequence to match dataset logic exactly
        agent_input = agent._prepare_model_input(agent.sequence_history)

        # Compare tensors; collect failures
        key_map = {
            "obs_sequence": "obs",
            "action_sequence": "action",
            "agent_types": "agent_type",
            "positions": "position",
        }
        failed_keys = []
        for a_key, t_key in key_map.items():
            if not compare_tensors(agent_input[a_key], truth_batch[t_key], a_key, quiet=quiet):
                failed_keys.append(a_key)

        if failed_keys:
            episodes_failed[episode_num] = failed_keys

    # ---- Summary ----
    total = args.episodes
    failed = len(episodes_failed)
    passed = total - failed
    print(f"\nSummary: PASSED {passed} / {total} episodes")
    if failed:
        print("Failed episodes (with failing tensors):")
        for ep, keys in sorted(episodes_failed.items()):
            print(f"  Episode {ep}: {', '.join(keys)}")
    if global_total:
        print(f"\n== Overall model-vs-PS agreement (our turns): {global_match}/{global_total} = {global_match/global_total:.2%}")
    else:
        print("\n== Overall model-vs-PS agreement: no agent turns recorded")
if __name__ == "__main__":
    main()
