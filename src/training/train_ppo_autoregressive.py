# src/training/train_ppo_autoregressive.py

import os
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

# Env & project imports
from src.env.liars_deck_env_core import LiarsDeckEnv
from src import config

# Model & agent
from src.model.ppo_autoregressive_model import PPOAutoregressiveModel
from src.agents.autoregressive_ppo_agent import PPOAutoregressiveAgent

# Hardcoded bots
from src.model.hard_coded_agents import (
    RandomAgent,
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    SelectiveTableConservativeChallenger,
    TableNonTableAgent,
    StrategicChallenger,
    Classic,
)

# Data generator (collector)
from src.training.ppo_ar_data_gen import collect_training_sequences

# Utilities

from src.training.train_extras import set_seed

from src.training.train_utils import compute_gae

from src.agents.hardcoded_agent_wrapper import HardcodedAgentWrapper


# --------------------------------------------------------------------------------------
# Helpers to build opponents and the per-episode player map
# --------------------------------------------------------------------------------------

def _instantiate_hardcoded(hc_class, hc_name: str, env_id: str, device: torch.device):
    """Instantiate a hardcoded opponent, with graceful fallback if ctor signature differs."""
    # Determine agent index from env_id (e.g., "player_2" -> 2)
    try:
        agent_index = int(env_id.split('_')[-1])
    except Exception:
        agent_index = 1

    # Try (name, num_players, agent_index) then fallback to (name)
    try:
        inst = hc_class(hc_name, 4, agent_index)
    except TypeError:
        inst = hc_class(hc_name)

    # Wrap if wrapper exists
    player_id = f"Hardcoded_{hc_name}"
    return HardcodedAgentWrapper(inst, device, player_id)


def build_players(
    device: torch.device,
    training_agent: PPOAutoregressiveAgent,
    opponent_types: List[str],
    opponent_configs: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Build the env_id -> agent map for one game, with player_0 as the learner.
    Returns (players_in_game, opponent_label_map) for belief targets.
    """
    players: Dict[str, Any] = {"player_0": training_agent}

    # Map from opponent *player_id* (stable name) to integer class label
    opponent_label_map: Dict[str, int] = {}

    # Instantiate opponents for env_ids player_1..player_{NUM_PLAYERS-1}
    assert len(opponent_types) == len(opponent_configs), "opponent_types and opponent_configs must align"
    for i in range(1, 4):
        env_id = f"player_{i}"
        opp_type = opponent_types[i - 1]
        opp_cfg = opponent_configs[i - 1]

        if opp_type == "hardcoded":
            hc_class = opp_cfg["class"]
            hc_name = opp_cfg["name"]
            agent = _instantiate_hardcoded(hc_class, hc_name, env_id, device)
        elif opp_type == "historical":
            # TODO: load a frozen PPO agent; stub for now
            raise NotImplementedError("Historical agent loading not yet implemented.")
        elif opp_type == "training":
            # Reuse the current learner as a self-play opponent (rare, but allowed)
            agent = training_agent
        else:
            raise ValueError(f"Unknown opponent type: {opp_type}")

        players[env_id] = agent
        # Use .get_player_id() if available, else fall back to class name
        try:
            pid = agent.get_player_id()
        except Exception:
            pid = getattr(agent, "player_id", opp_cfg.get("name", str(agent)))
        opponent_label_map[pid] = len(opponent_label_map)

    return players, opponent_label_map


# --------------------------------------------------------------------------------------
# Loss builders
# --------------------------------------------------------------------------------------

def ppo_losses_for_episode(
    model: PPOAutoregressiveModel,
    episode: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute PPO+auxiliary losses for a single episode dict produced by collect_training_sequences.
    Returns (total_loss, scalars)
    """
    # Move model inputs to device and forward once
    mi = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in episode["model_input"].items()}
    action_logits, opp_logits, state_values, b0_logits, b1_logits, b2_logits = model(**mi)

    # Shapes: [1, T, ...]
    T = action_logits.size(1)
    agent_types = mi["agent_types"].squeeze(0)  # [T]
    our_mask = (agent_types == 0)
    opp_mask = ~our_mask

    # Indices for our/opponent steps
    our_idx = torch.nonzero(our_mask, as_tuple=False).squeeze(-1)

    scalars = {
        "n_our_steps": float(our_idx.numel()),
        "n_total_steps": float(T),
    }

    total_loss = torch.zeros((), device=device)

    # ---------------- Actor-Critic on our steps ----------------
    if our_idx.numel() > 0:
        # Gather episode data at our steps
        actions = torch.tensor([episode["our_action"][i] for i in our_idx.tolist()], device=device, dtype=torch.long)
        old_logp = torch.tensor([episode["log_prob"][i] for i in our_idx.tolist()], device=device, dtype=torch.float32)
        rewards = [float(episode["reward"][i]) for i in our_idx.tolist()]
        dones   = [bool(episode["done"][i])   for i in our_idx.tolist()]

        logits_at = action_logits[0, our_idx, :]  # [N, A]
        values_at = state_values[0, our_idx, 0]   # [N]

        # Build next_values for GAE: shift left, last=0 (or last value if not done)
        next_vals = values_at.detach().clone()
        if next_vals.numel() > 1:
            next_vals[:-1] = values_at.detach()[1:]
        next_vals[-1] = 0.0

        adv, ret = compute_gae(rewards, dones, values_at.detach().cpu().tolist(), next_vals.detach().cpu().tolist(), config.GAMMA, config.GAE_LAMBDA)
        advantages = torch.tensor(adv, dtype=torch.float32, device=device)
        returns    = torch.tensor(ret, dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        dist = torch.distributions.Categorical(logits=logits_at)
        new_logp = dist.log_prob(actions)
        ratio = (new_logp - old_logp).exp()

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss  = F.mse_loss(values_at, returns)
        entropy_loss = -dist.entropy().mean()

        total_loss = total_loss + policy_loss + 0.5 * value_loss + config.INIT_ENTROPY_COEF * entropy_loss

        scalars.update({
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float((-entropy_loss).detach().cpu()),
        })

        # ---------------- Belief loss on our steps ----------------
        # Use belief targets if available (they are integers 0..K-1)
        def _belief_ce(b_logits, key):
            targets = []
            idxs = []
            for i in our_idx.tolist():
                tgt = episode[key][i]
                if tgt is not None:
                    targets.append(int(tgt))
                    idxs.append(i)
            if not targets:
                return torch.zeros((), device=device)
            t = torch.tensor(targets, dtype=torch.long, device=device)
            l = b_logits[0, torch.tensor(idxs, dtype=torch.long, device=device), :]
            return F.cross_entropy(l, t)

        b0_loss = _belief_ce(b0_logits, "belief_tgt0")
        b1_loss = _belief_ce(b1_logits, "belief_tgt1")
        b2_loss = _belief_ce(b2_logits, "belief_tgt2")
        belief_loss = b0_loss + b1_loss + b2_loss
        total_loss = total_loss + config.AUX_LOSS_WEIGHT * belief_loss
        scalars.update({"belief_loss": float(belief_loss.detach().cpu())})

    # ---------------- Opponent action prediction loss ----------------
    # Only at opponent steps that have a retro-filled prediction (mask via episode["opp_pred_logits"]) and a target action
    opp_idxs = []
    opp_targets = []
    for i in range(T):
        if not opp_mask[i]:
            continue
        tgt = episode["opp_target_action"][i]
        pred_flag = episode["opp_pred_logits"][i] is not None
        if (tgt is not None) and pred_flag:
            opp_idxs.append(i)
            opp_targets.append(int(tgt))

    if len(opp_idxs) > 0:
        opp_logits_sel = opp_logits[0, torch.tensor(opp_idxs, device=device), :]
        opp_targets_t = torch.tensor(opp_targets, dtype=torch.long, device=device)
        opp_loss = F.cross_entropy(opp_logits_sel, opp_targets_t)
        total_loss = total_loss + config.AUX_LOSS_WEIGHT * opp_loss
        scalars.update({"opp_loss": float(opp_loss.detach().cpu()), "n_opp_supervised": float(len(opp_idxs))})
    else:
        scalars.update({"opp_loss": 0.0, "n_opp_supervised": 0.0})

    return total_loss, scalars


# --------------------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------------------

def train(
    num_updates: int = 1000,
    episodes_per_update: int = 8,
    k_epochs: int = 2,
    checkpoint_dir: Optional[str] = None,
):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    device = torch.device(config.DEVICE if hasattr(config, "DEVICE") else ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(getattr(config, "SEED", 42))

    # ---- Env & model ----
    env = LiarsDeckEnv(num_players=4)

    obs_dim = 9
    action_dim = 7

    model = PPOAutoregressiveModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        belief_dim=64,
        hidden_dim=256,
        max_seq_length=320,
        num_agent_types=4,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, eps=1e-5)

    learner = PPOAutoregressiveAgent(device, "TrainAgent_v1")
    learner.set_model(model)

    # ---- Opponents (example: all hardcoded; replace with your configs as needed) ----
    # Build a simple rotating set of hardcoded opponents
    hc_pool = { # Hardcoded agents ...
             "Classic": Classic, "GreedyCardSpammer": GreedyCardSpammer, "RandomAgent": RandomAgent,
             "SelectiveTableConservativeChallenger": SelectiveTableConservativeChallenger,
             "StrategicChallenger": StrategicChallenger, "TableFirstConservativeChallenger": TableFirstConservativeChallenger,
             "TableNonTableAgent": TableNonTableAgent
        }

    def build_opponent_cfgs(start: int = 0) -> Tuple[List[str], List[Dict[str, Any]]]:
        n = 4 - 1
        picks = [hc_pool[(start + i) % len(hc_pool)] for i in range(n)]
        opponent_types = ["hardcoded"] * n
        opponent_configs = [{"class": cls, "name": name} for cls, name in picks]
        return opponent_types, opponent_configs

    global_step = 0
    for update in range(1, num_updates + 1):
        opponent_types, opponent_configs = build_opponent_cfgs(update)
        players_in_game, opponent_label_map = build_players(device, learner, opponent_types, opponent_configs)

        # ---- Collect episodes ----
        episodes = collect_training_sequences(
            env=env,
            device=device,
            players_in_this_game=players_in_game,
            episodes=episodes_per_update,
            training_agent_env_id="player_0",
            opponent_label_map=opponent_label_map,
        )

        # ---- PPO update ----
        model.train()
        epoch_loss = 0.0
        n_loss_terms = 0

        for epoch in range(k_epochs):
            for ep in episodes:
                loss, scalars = ppo_losses_for_episode(model, ep, device)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=getattr(config, "MAX_NORM", 0.5))
                optimizer.step()

                epoch_loss += float(loss.detach().cpu())
                n_loss_terms += 1

        model.eval()

        # ---- Logging ----
        avg_loss = epoch_loss / max(1, n_loss_terms)
        logging.info(f"Update {update}/{num_updates} | episodes={episodes_per_update} | avg_loss={avg_loss:.4f}")

        # ---- Checkpoint ----
        if checkpoint_dir and (update % getattr(config, "CHECKPOINT_INTERVAL", 200) == 0):
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, f"arppo_update_{update}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "update": update,
            }, path)
            logging.info(f"Saved checkpoint to {path}")


if __name__ == "__main__":
    train(
        num_updates=1000,
        episodes_per_update=8,
        k_epochs=getattr(config, "K_EPOCHS", 2),
        checkpoint_dir=os.path.join(getattr(config, "CHECKPOINT_DIR", "checkpoints"), "arppo"),
    )
