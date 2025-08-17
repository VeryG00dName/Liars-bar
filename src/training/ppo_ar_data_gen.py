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
        "opp_pred_logits": [],  # retro-filled onto opponent steps

        # snapshot of model inputs at game end
        "model_input": None,
    }

def _append_step_row(ep: Dict[str, Any], agent_id_env: str) -> int:
    """Append a new step with defaults; return its index."""
    ep["agent_id"].append(agent_id_env)

    ep["our_action"].append(None)
    ep["log_prob"].append(None)
    ep["value"].append(None)
    ep["reward"].append(0.0)    # or None if you prefer
    ep["done"].append(False)

    ep["opp_target_action"].append(None)

    ep["belief_pred0"].append(None)
    ep["belief_pred1"].append(None)
    ep["belief_pred2"].append(None)
    ep["belief_tgt0"].append(None)
    ep["belief_tgt1"].append(None)
    ep["belief_tgt2"].append(None)

    ep["opp_pred_logits"].append(None)
    return len(ep["agent_id"]) - 1


# -------------------------
# Main collector
# -------------------------

def collect_training_sequences(
    env,
    device: torch.device,
    players_in_this_game: Dict[str, Any],
    episodes: int = 1,
    training_agent_env_id: str = "player_0",
    opponent_label_map: Optional[Dict[str, int]] = None
) -> List[Dict[str, Any]]:
    """
    Play `episodes` games and return a list[episode_dict].
    Each episode_dict contains:
      - step-aligned lists (one entry per env step)
      - 'true_opponent_labels' (for belief targets)
      - 'model_input' snapshot produced by the learner's _prepare_model_input(history)
    """
    sequences: List[Dict[str, Any]] = []

    for game_idx in range(episodes):
        # Reset env & all agents
        env.reset(seed=game_idx)
        for agent in players_in_this_game.values():
            agent.reset()

        ep = _new_episode(env, players_in_this_game, training_agent_env_id, opponent_label_map)

        game_active = True
        while game_active and env.agent_selection is not None:
            agent_id_env = env.agent_selection
            observation = env.observe(agent_id_env)
            _, _, terminated, truncated, info = env.last()
            if terminated or truncated:
                # advance pettingzoo
                env.step(None)
                continue

            row = _append_step_row(ep, agent_id_env)

            if agent_id_env == training_agent_env_id:
                # ---- OUR TURN (training=True) ----
                # Expecting: action, log_prob, value, b0, b1, b2, opp1, opp2, opp3
                (action, log_prob, value,
                 b0, b1, b2,
                 opp1, opp2, opp3) = players_in_this_game[agent_id_env].get_action(
                    env, agent_id_env, observation, info, training=True
                )

                # per-step fields for our turn
                ep["our_action"][row] = int(action) if not isinstance(action, int) else action
                ep["log_prob"][row]   = float(log_prob)
                ep["value"][row]      = float(value)

                ep["belief_pred0"][row] = b0
                ep["belief_pred1"][row] = b1
                ep["belief_pred2"][row] = b2

                # belief targets are the static true opponent labels (put them only on our turn)
                tl = ep["true_opponent_labels"]
                if len(tl) >= 1: ep["belief_tgt0"][row] = tl[0]
                if len(tl) >= 2: ep["belief_tgt1"][row] = tl[1]
                if len(tl) >= 3: ep["belief_tgt2"][row] = tl[2]

                # step the env and read our reward/done
                env.step(action)
                _, reward, terminated, truncated, _ = env.last()
                ep["reward"][row] = float(reward)
                ep["done"][row]   = bool(terminated or truncated)

                # ---- Retro-fill: write our opponent-prediction logits onto prior opponent steps
                prev_preds = (opp1, opp2, opp3)
                for i, opp_pred in enumerate(prev_preds, start=1):
                    if opp_pred is None:
                        continue
                    target_row = row - i
                    if target_row >= 0 and ep["agent_id"][target_row] != training_agent_env_id:
                        ep["opp_pred_logits"][target_row] = opp_pred

            else:
                # ---- OPPONENT TURN ----
                # Do NOT pass training kwarg to other agents.
                action = players_in_this_game[agent_id_env].get_action(
                    env, agent_id_env, observation, info
                )
                # record the *target* (ground-truth) opponent action for this step
                ep["opp_target_action"][row] = int(action) if not isinstance(action, int) else action

                env.step(action)
                _, _, terminated, truncated, _ = env.last()
                ep["done"][row] = bool(terminated or truncated)

            # loop control
            game_active = bool(env.agents)

        # ---------- End of game: snapshot the learner's model inputs ----------
        learner = players_in_this_game[training_agent_env_id]

        # prepare model input from the learner's sequence history
        model_input = learner._prepare_model_input(learner.sequence_history)

        # store CPU copies to keep VRAM free
        ep["model_input"] = {
            k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in model_input.items()
        }

        sequences.append(ep)

    return sequences
