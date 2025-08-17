# src/training/train_ppo_autoregressive.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

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
from src.training.train_utils import compute_gae  # (rewards, dones, values, next_values, gamma, lam)

from src.agents.hardcoded_agent_wrapper import HardcodedAgentWrapper


# --------------------------------------------------------------------------------------
# Belief mapping (extend this when adding frozen/historical models)
# --------------------------------------------------------------------------------------
BELIEF_LABELS: Dict[str, int] = {
    "GreedyCardSpammer": 1,
    "StrategicChallenger": 4,
    "TableNonTableAgent": 6,
    "Classic": 0,
    "TableFirstConservativeChallenger": 5,
    "SelectiveTableConservativeChallenger": 3,
    "RandomAgent": 2,
}


# --------------------------------------------------------------------------------------
# Helpers to build opponents and the per-episode player map
# --------------------------------------------------------------------------------------

def _instantiate_hardcoded(hc_class, hc_name: str, env_id: str, device: torch.device):
    """Instantiate a hardcoded opponent and wrap to BaseAgent via HardcodedAgentWrapper."""
    # Try to provide context (name, num_players, agent_index); fallback to just name
    try:
        agent_index = int(env_id.split('_')[-1])
    except Exception:
        agent_index = 1

    try:
        inst = hc_class(hc_name, config.NUM_PLAYERS, agent_index)
    except TypeError:
        inst = hc_class(hc_name)

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
    Returns (players_in_game, opponent_label_map) where opponent_label_map maps
    **player_id -> belief class id** per BELIEF_LABELS.
    """
    assert len(opponent_types) == len(opponent_configs), "opponent_types and opponent_configs must align"

    players: Dict[str, Any] = {"player_0": training_agent}
    opponent_label_map: Dict[str, int] = {}

    for i in range(1, config.NUM_PLAYERS):
        env_id = f"player_{i}"
        opp_type = opponent_types[i - 1]
        opp_cfg = opponent_configs[i - 1]

        if opp_type == "hardcoded":
            hc_class = opp_cfg["class"]
            hc_name = opp_cfg["name"]  # must match key in BELIEF_LABELS
            agent = _instantiate_hardcoded(hc_class, hc_name, env_id, device)

            # Map this opponent's player_id to the belief class id
            if hc_name not in BELIEF_LABELS:
                raise KeyError(f"Hardcoded agent '{hc_name}' missing from BELIEF_LABELS")
            label = BELIEF_LABELS[hc_name]
            pid = agent.get_player_id()  # e.g., "Hardcoded_GreedyCardSpammer"
            opponent_label_map[pid] = label

        elif opp_type == "historical":
            # TODO: instantiate frozen PPO agent and add to BELIEF_LABELS when available
            raise NotImplementedError("Historical agent loading not yet implemented.")

        elif opp_type == "training":
            # Self-play (rare): use learner; you must also assign a belief label
            agent = training_agent
            pid = agent.get_player_id() if hasattr(agent, "get_player_id") else "TrainAgent"
            if pid not in opponent_label_map:
                raise NotImplementedError("Belief label for training/self-play opponent is undefined. Add to BELIEF_LABELS and map here.")

        else:
            raise ValueError(f"Unknown opponent type: {opp_type}")

        players[env_id] = agent

    return players, opponent_label_map


# --------------------------------------------------------------------------------------
# Loss builders (+ PPO metrics/accuracies for TB)
# --------------------------------------------------------------------------------------

def _accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.numel() == 0 or targets.numel() == 0:
        return 0.0
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        return float((preds == targets).float().mean().item())


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

    our_idx = torch.nonzero(our_mask, as_tuple=False).squeeze(-1)

    scalars = {
        "n_our_steps": float(our_idx.numel()),
        "n_total_steps": float(T),
        "episode_return": float(sum(episode["reward"])),
        "episode_len": float(len(episode["reward"])),
    }

    total_loss = torch.zeros((), device=device)

    # ---------------- Actor-Critic on our steps ----------------
    if our_idx.numel() > 0:
        actions = torch.tensor([episode["our_action"][i] for i in our_idx.tolist()], device=device, dtype=torch.long)
        old_logp = torch.tensor([episode["log_prob"][i] for i in our_idx.tolist()], device=device, dtype=torch.float32)
        rewards = [float(episode["reward"][i]) for i in our_idx.tolist()]
        dones   = [bool(episode["done"][i])   for i in our_idx.tolist()]

        logits_at = action_logits[0, our_idx, :]  # [N, A]
        values_at = state_values[0, our_idx, 0]   # [N]

        # Build next_values for GAE (shifted values, last = 0)
        next_vals = values_at.detach().clone()
        if next_vals.numel() > 1:
            next_vals[:-1] = values_at.detach()[1:]
        next_vals[-1] = 0.0

        adv, ret = compute_gae(
            rewards,
            dones,
            values_at.detach().cpu().tolist(),
            next_vals.detach().cpu().tolist(),
            config.GAMMA,
            config.GAE_LAMBDA,
        )
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
        entropy_term = dist.entropy().mean()
        entropy_loss = -entropy_term

        total_loss = total_loss + policy_loss + 0.5 * value_loss + config.INIT_ENTROPY_COEF * entropy_loss

        # PPO diagnostics
        approx_kl = torch.mean(old_logp - new_logp).detach()
        clipfrac = ((ratio - 1.0).abs() > config.EPS_CLIP).float().mean().detach()

        scalars.update({
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy_term.detach().cpu()),
            "approx_kl": float(approx_kl.cpu()),
            "clip_fraction": float(clipfrac.cpu()),
        })

        # Policy action accuracy on our steps (against taken action)
        scalars["agent_action_acc"] = _accuracy_from_logits(logits_at.detach(), actions.detach())

        # ---------------- Belief loss on our steps ----------------
        def _belief_ce_and_acc(b_logits, key):
            targets, idxs = [], []
            for i in our_idx.tolist():
                tgt = episode[key][i]
                if tgt is not None:
                    targets.append(int(tgt))
                    idxs.append(i)
            if not targets:
                return torch.zeros((), device=device), 0.0
            t = torch.tensor(targets, dtype=torch.long, device=device)
            l = b_logits[0, torch.tensor(idxs, dtype=torch.long, device=device), :]
            acc = _accuracy_from_logits(l.detach(), t.detach())
            return F.cross_entropy(l, t), acc

        b0_loss, acc_b0 = _belief_ce_and_acc(b0_logits, "belief_tgt0")
        b1_loss, acc_b1 = _belief_ce_and_acc(b1_logits, "belief_tgt1")
        b2_loss, acc_b2 = _belief_ce_and_acc(b2_logits, "belief_tgt2")
        belief_loss = b0_loss + b1_loss + b2_loss
        total_loss = total_loss + config.AUX_LOSS_WEIGHT * belief_loss
        scalars.update({
            "belief_loss": float(belief_loss.detach().cpu()),
            "belief_acc_0": float(acc_b0),
            "belief_acc_1": float(acc_b1),
            "belief_acc_2": float(acc_b2),
        })

    # ---------------- Opponent action prediction loss & accuracy ----------------
    opp_idxs, opp_targets = [], []
    for i in range(T):
        if not opp_mask[i]:
            continue
        tgt = episode["opp_target_action"][i]
        pred_flag = episode["opp_pred_logits"][i] is not None
        if (tgt is not None) and pred_flag:
            opp_idxs.append(i)
            opp_targets.append(int(tgt))

    if len(opp_idxs) > 0:
        idx_tensor = torch.tensor(opp_idxs, device=device)
        opp_logits_sel = opp_logits[0, idx_tensor, :]
        opp_targets_t = torch.tensor(opp_targets, dtype=torch.long, device=device)
        opp_loss = F.cross_entropy(opp_logits_sel, opp_targets_t)
        total_loss = total_loss + config.AUX_LOSS_WEIGHT * opp_loss
        scalars.update({
            "opp_loss": float(opp_loss.detach().cpu()),
            "n_opp_supervised": float(len(opp_idxs)),
            "opp_action_acc": _accuracy_from_logits(opp_logits_sel.detach(), opp_targets_t.detach()),
        })
    else:
        scalars.update({"opp_loss": 0.0, "n_opp_supervised": 0.0, "opp_action_acc": 0.0})

    return total_loss, scalars


# --------------------------------------------------------------------------------------
# Training loop (with TensorBoard logging similar to supervised version)
# --------------------------------------------------------------------------------------

def train(
    num_updates: int = 1000,
    episodes_per_update: int = 8,
    k_epochs: int = 2,
    checkpoint_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    device = torch.device(getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(getattr(config, "SEED", 42))

    # ---- Logging / TB ----
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    logging.info(f"TensorBoard logdir: {log_dir}")

    # ---- Env & model ----
    env = LiarsDeckEnv(num_players=getattr(config, "NUM_PLAYERS", 4))

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

    # ---- Opponent rotation (less confusing setup) ----
    HC_POOL: List[Tuple[str, Any]] = [
        ("Classic", Classic),
        ("GreedyCardSpammer", GreedyCardSpammer),
        ("RandomAgent", RandomAgent),
        ("SelectiveTableConservativeChallenger", SelectiveTableConservativeChallenger),
        ("StrategicChallenger", StrategicChallenger),
        ("TableFirstConservativeChallenger", TableFirstConservativeChallenger),
        ("TableNonTableAgent", TableNonTableAgent),
    ]

    def build_opponent_cfgs(start: int = 0) -> Tuple[List[str], List[Dict[str, Any]]]:
        n = getattr(config, "NUM_PLAYERS", 4) - 1
        picks = [HC_POOL[(start + i) % len(HC_POOL)] for i in range(n)]
        opponent_types = ["hardcoded"] * n
        opponent_configs = [{"class": cls, "name": name} for (name, cls) in picks]
        return opponent_types, opponent_configs

    global_step = 0
    for update in range(1, num_updates + 1):
        update_start = time.time()

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
        agg: Dict[str, float] = {}
        n_loss_terms = 0

        for _ in range(k_epochs):
            for ep in episodes:
                loss, scalars = ppo_losses_for_episode(model, ep, device)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=getattr(config, "MAX_NORM", 0.5))
                optimizer.step()

                # aggregate scalars
                for k, v in scalars.items():
                    agg[k] = agg.get(k, 0.0) + float(v)
                agg["total_loss"] = agg.get("total_loss", 0.0) + float(loss.detach().cpu())
                n_loss_terms += 1
                global_step += 1

        model.eval()

        # ---- Averaging over all loss terms (episodes * k_epochs) ----
        def avg(name, default=0.0):
            return (agg.get(name, 0.0) / max(1, n_loss_terms)) if name in agg else default

        # ---- Logging ----
        avg_total_loss = avg("total_loss")
        logging.info(f"Update {update}/{num_updates} | episodes={episodes_per_update} | "
                     f"avg_loss={avg_total_loss:.4f} | time={time.time()-update_start:.2f}s")

        # Compute returns & lengths per episode for TB (based on collection)
        ep_returns = [sum(ep["reward"]) for ep in episodes]
        ep_lens = [len(ep["reward"]) for ep in episodes]
        mean_return = float(sum(ep_returns) / len(ep_returns)) if ep_returns else 0.0
        mean_ep_len = float(sum(ep_lens) / len(ep_lens)) if ep_lens else 0.0

        # ---- TensorBoard scalars (mirroring supervised style + PPO-specific) ----
        writer.add_scalar("Loss/Total", avg_total_loss, update)
        writer.add_scalar("Loss/Policy", avg("policy_loss"), update)
        writer.add_scalar("Loss/Value", avg("value_loss"), update)
        writer.add_scalar("Loss/Aux/Opponent", avg("opp_loss"), update)
        writer.add_scalar("Loss/Aux/Belief", avg("belief_loss"), update)
        writer.add_scalar("Policy/Entropy", avg("entropy"), update)
        writer.add_scalar("Policy/ApproxKL", avg("approx_kl"), update)
        writer.add_scalar("Policy/ClipFraction", avg("clip_fraction"), update)

        writer.add_scalar("Acc/OpponentAction", avg("opp_action_acc"), update)
        writer.add_scalar("Acc/Belief0", avg("belief_acc_0"), update)
        writer.add_scalar("Acc/Belief1", avg("belief_acc_1"), update)
        writer.add_scalar("Acc/Belief2", avg("belief_acc_2"), update)
        writer.add_scalar("Acc/AgentActionProxy", avg("agent_action_acc"), update)

        writer.add_scalar("Rollout/EpisodeReturnMean", mean_return, update)
        writer.add_scalar("Rollout/EpisodeLenMean", mean_ep_len, update)
        writer.add_scalar("Rollout/OurStepsPerEpisodeMean", avg("n_our_steps"), update)
        writer.add_scalar("Rollout/TotalStepsPerEpisodeMean", avg("n_total_steps"), update)
        writer.add_scalar("Supervision/OppStepsWithTargets", avg("n_opp_supervised"), update)

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

    writer.close()


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_autoreg_{timestamp}"
    log_dir = os.path.join("logs", run_name)
    ckpt_dir = os.path.join(getattr(config, "CHECKPOINT_DIR", "checkpoints"), run_name)
    train(
        num_updates=1000,
        episodes_per_update=10,
        k_epochs=getattr(config, "K_EPOCHS", 2),
        checkpoint_dir=ckpt_dir,
        log_dir=log_dir,
    )
