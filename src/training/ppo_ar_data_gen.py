import torch
from typing import Dict, Any, List, Optional

# -------------------------
# Episode container helpers
# -------------------------

def _new_episode(env, players: Dict[str, Any], training_agent_env_id: str,
                 opponent_label_map: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Create a fresh episode dict (one list per field; one entry per env step)."""
    # figure out which env IDs are opponents (stable order)
    opp_env_ids = [aid for aid in env.possible_agents if aid != training_agent_env_id]

    # "true" opponent labels for belief targets (either global int labels or IDs you can map later)
    if opponent_label_map is not None:
        true_opponent_labels = tuple(
            opponent_label_map[players[aid].get_player_id()] for aid in opp_env_ids
        )
    else:
        true_opponent_labels = tuple(players[aid].get_player_id() for aid in opp_env_ids)

    return {
        # meta
        "training_agent_env_id": training_agent_env_id,
        "opponent_env_ids": tuple(opp_env_ids),
        "true_opponent_labels": true_opponent_labels,  # for belief targets on our turns

        # per-step tracks (aligned 1:1 with env steps)
        "agent_id": [],
        "our_action": [],
        "log_prob": [],
        "value": [],
        "reward": [],
        "done": [],
        "opp_target_action": [],
        "belief_pred0": [],
        "belief_pred1": [],
        "belief_pred2": [],
        "belief_tgt0": [],
        "belief_tgt1": [],
        "belief_tgt2": [],

        # Kept for backward compatibility with your PPO loss gate:
        # we now store only a boolean marker (True) when a prediction exists,
        # and leave None otherwise. No logits are ever stored.
        "opp_pred_logits": [],

        # snapshot of model inputs at game end (CPU tensors)
        "model_input": None,
        "episode_return": 0.0,   # NEW: will fill at end
        "win": 0,                # NEW: 1/0 flag
    }

def _append_step_row(ep: Dict[str, Any], agent_id_env: str) -> int:
    """Append a new step with defaults; return its index."""
    ep["agent_id"].append(agent_id_env)

    ep["our_action"].append(None)
    ep["log_prob"].append(None)
    ep["value"].append(None)
    ep["reward"].append(0.0)
    ep["done"].append(False)

    ep["opp_target_action"].append(None)

    ep["belief_pred0"].append(None)
    ep["belief_pred1"].append(None)
    ep["belief_pred2"].append(None)
    ep["belief_tgt0"].append(None)
    ep["belief_tgt1"].append(None)
    ep["belief_tgt2"].append(None)

    ep["opp_pred_logits"].append(None)  # now used as a boolean flag only
    return len(ep["agent_id"]) - 1


# -------------------------
# Utilities
# -------------------------

def _to_scalar_int(x):
    """Convert possible tensor/list/float to int (top-1 if vector), else None."""
    if x is None:
        return None
    if torch.is_tensor(x):
        if x.ndim == 0:
            return int(x.item())
        # vector logits/probs -> argmax class id
        return int(torch.argmax(x, dim=-1).item())
    if isinstance(x, (int,)):
        return int(x)
    if isinstance(x, float):
        return int(x)
    # unknown type -> drop
    return None


# -------------------------
# Main collector
# -------------------------

def collect_training_sequences(
    env,
    device: torch.device,
    players_in_this_game: Dict[str, Any],
    episodes: int = 1,
    training_agent_env_id: str = "player_0",
    opponent_label_map: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Play `episodes` games and return a list[episode_dict].

    Notes:
      - Rollout runs under torch.inference_mode() to avoid building graphs.
      - No CUDA tensors are stored in the episode dict.
      - 'opp_pred_logits' is now a boolean flag (True when a pred exists), never logits.
    """
    sequences: List[Dict[str, Any]] = []

    for game_idx in range(episodes):
        # Reset env & agents
        env.reset(seed=game_idx)
        for agent in players_in_this_game.values():
            # If your wrapper needs device/num_players context, it's set at construction.
            agent.reset()

        ep = _new_episode(env, players_in_this_game, training_agent_env_id, opponent_label_map)

        game_active = True
        with torch.inference_mode():
            while game_active and env.agent_selection is not None:
                agent_id_env = env.agent_selection

                # Check termination/truncation for current agent BEFORE any calls and advance if needed
                if env.terminations.get(agent_id_env, False) or env.truncations.get(agent_id_env, False):
                    env.step(None)
                    game_active = bool(env.agents)
                    continue

                # Safe to read observation/info now
                observation = env.observe(agent_id_env)
                info = env.infos.get(agent_id_env, {})

                row = _append_step_row(ep, agent_id_env)

                if agent_id_env == training_agent_env_id:
                    # ---- OUR TURN (training=True) ----
                    (action, log_prob, value,
                     b0, b1, b2,
                     opp1, opp2, opp3) = players_in_this_game[agent_id_env].get_action(
                        env, agent_id_env, observation, info, training=True
                    )

                    # Fill our step with Python scalars
                    ep["our_action"][row] = _to_scalar_int(action)
                    ep["log_prob"][row]   = float(log_prob) if log_prob is not None else None
                    ep["value"][row]      = float(value)    if value    is not None else None

                    # Belief preds as top-1 class ids (or None)
                    ep["belief_pred0"][row] = _to_scalar_int(b0)
                    ep["belief_pred1"][row] = _to_scalar_int(b1)
                    ep["belief_pred2"][row] = _to_scalar_int(b2)

                    # Belief targets (static per-episode)
                    tl = ep["true_opponent_labels"]
                    if len(tl) >= 1: ep["belief_tgt0"][row] = tl[0]
                    if len(tl) >= 2: ep["belief_tgt1"][row] = tl[1]
                    if len(tl) >= 3: ep["belief_tgt2"][row] = tl[2]

                    # Step env
                    env.step(int(ep["our_action"][row]))

                    # Rewards/done from dicts (post-step)
                    ep["reward"][row] = float(env.rewards.get(training_agent_env_id, 0.0))
                    ep["done"][row] = bool(
                        env.terminations.get(training_agent_env_id, False)
                        or env.truncations.get(training_agent_env_id, False)
                    )

                    # Retro-fill opponent prediction *flags* (True) onto prior opponent rows (1..3 back)
                    # (We do NOT store logits to avoid VRAM/memory blow-up.)
                    for i, opp_pred in enumerate((opp1, opp2, opp3), start=1):
                        if opp_pred is None:
                            continue
                        target_row = row - i
                        if target_row >= 0 and ep["agent_id"][target_row] != training_agent_env_id:
                            ep["opp_pred_logits"][target_row] = True  # boolean marker only

                else:
                    # ---- OPPONENT TURN ----
                    action = players_in_this_game[agent_id_env].get_action(
                        env, agent_id_env, observation, info
                    )
                    ep["opp_target_action"][row] = _to_scalar_int(action)

                    env.step(int(ep["opp_target_action"][row]))

                    # Store training-agent terminal state on this row if it happened here
                    
                    ep["done"][row] = bool(
                        env.terminations.get(training_agent_env_id, False)
                        or env.truncations.get(training_agent_env_id, False)
                    )
                    if ep["done"][row]:
                        ep["reward"][row] = float(env.rewards.get(training_agent_env_id, 0.0))
                # loop control
                game_active = bool(env.agents)

        # ---------- End of game: snapshot the learner's model inputs ----------
        learner = players_in_this_game[training_agent_env_id]
        model_input = learner._prepare_model_input(learner.sequence_history)
        ep["model_input"] = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in model_input.items()}

        # --- robust Win/Return finalize ---
        total_return = float(sum(ep["reward"]))
        ep["episode_return"] = total_return

        # Prefer env-provided winner if available; otherwise fall back to reward sum
        win_from_env = None
        info = env.infos.get(training_agent_env_id, {})
        if "win" in info:
            # If your env sets infos[agent]["win"] on terminal
            win_from_env = 1 if info["win"] else 0
        elif hasattr(env, "winners") and env.winners is not None:
            # PettingZoo-style list/tuple of winners
            try:
                win_from_env = 1 if training_agent_env_id in env.winners else 0
            except Exception:
                win_from_env = None

        # Fallback: current behavior (reward==1 on win)
        ep["win"] = win_from_env if win_from_env is not None else (1 if total_return > 0.0 else 0)

        sequences.append(ep)

    return sequences
