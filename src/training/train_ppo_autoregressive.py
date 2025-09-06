# src/training/train_ppo_autoregressive.py

import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import random

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import torch.amp as amp
# Env & project imports
from src.misc import lb
from src import config

# Model & agent
from src.model.ppo_autoregressive_model import PPOAutoregressiveModel
from src.agents.batch_autoregressive_ppo_agent import BatchPPOAutoregressiveAgent

# High-performance vectorized data collector
from src.training.vec_ppo_rollout import PPOVecRolloutManager

# Utilities
from src.training.train_extras import set_seed
from src.training.train_utils import compute_gae

# --------------------------------------------------------------------------------------
# Belief mapping (maps BotKind enum value to belief index)
# --------------------------------------------------------------------------------------
BELIEF_LABELS_FROM_KIND: Dict[int, int] = {
    lb.BotKind.GreedyCardSpammer.value: 1, lb.BotKind.StrategicChallenger.value: 4,
    lb.BotKind.TableNonTableAgent.value: 6, lb.BotKind.Classic.value: 0,
    lb.BotKind.TableFirstConservativeChallenger.value: 5,
    lb.BotKind.SelectiveTableConservativeChallenger.value: 3, lb.BotKind.RandomAgent.value: 2,
}

# --------------------------------------------------------------------------------------
# Loss builders
# --------------------------------------------------------------------------------------
def _accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.numel() == 0 or targets.numel() == 0: return 0.0
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        return float((preds == targets).float().mean().item())

def ppo_losses_for_episode(
    model: PPOAutoregressiveModel,
    episode: Dict[str, Any],
    device: torch.device,
    sl_teacher: Optional[PPOAutoregressiveModel] = None,
    bc_kl_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:

    # ---- 0) Unpack model input exactly as fed to the network ----
    mi = {k: (v.to(device) if torch.is_tensor(v) else v)
          for k, v in episode["model_input"].items()}

    action_seq    = mi["action_sequence"]                       # [1, L]
    agent_types   = mi["agent_types"]                           # [1, L]
    valid_lengths = mi.get("valid_lengths",
                           torch.tensor([action_seq.size(1)], device=device))  # [1]
    action_masks  = mi.get("action_masks", None)                # [1, L, A] or None

    # Local (truncated) views in model space
    valid_len        = int(valid_lengths[0].item())
    agent_types_1d   = agent_types[0, :valid_len]               # [L]
    actions_1d       = action_seq [0, :valid_len]               # [L]
    masks_2d         = action_masks[0, :valid_len] if action_masks is not None else None  # [L, A] or None

    # ---- Student forward on exactly the same input ----
    action_logits, opp_logits, state_values, b0, b1, b2 = model(**mi)  # [1, L, ...]
    action_logits = action_logits[0, :valid_len, :]            # [L, A]
    opp_logits    = opp_logits[0, :valid_len, :] if opp_logits is not None else None
    state_values  = state_values[0, :valid_len].squeeze(-1)    # [L]

    # ---- 1) Indices for OUR turns in local space (0..valid_len-1) ----
    our_pos = (agent_types_1d == 0).nonzero(as_tuple=False).squeeze(-1).long()  # [K]
    K = int(our_pos.numel())

    scalars = {
        "n_our_steps": float(len(episode["our_action"]) - episode["our_action"].count(None)),
        "n_total_steps": float(valid_len),
        "episode_return": float(episode.get("episode_return", 0.0))
    }

    # Graph-carrying zero (so backward() never breaks) in "no-learnable-steps" cases
    if K == 0:
        total_loss = next(model.parameters()).sum() * 0.0
        scalars.update({
            "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "approx_kl": 0.0, "clip_fraction": 0.0,
            "opp_loss": 0.0, "opp_action_acc": 0.0,
            "belief_loss": 0.0, "belief_acc_0": 0.0, "belief_acc_1": 0.0, "belief_acc_2": 0.0,
        })
        return total_loss, scalars

    # ---- 2) Build episode-aligned lists at OUR steps ----
    our_steps_ep_idx = [i for i, seat in enumerate(episode["agent_id"])
                        if seat == episode["training_agent_seat"]]

    # Align counts; we’ll learn on min(K, len(ep rows))
    K_ep = min(K, len(our_steps_ep_idx))
    if K_ep == 0:
        total_loss = next(model.parameters()).sum() * 0.0
        scalars.update({
            "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "approx_kl": 0.0, "clip_fraction": 0.0,
            "opp_loss": 0.0, "opp_action_acc": 0.0,
            "belief_loss": 0.0, "belief_acc_0": 0.0, "belief_acc_1": 0.0, "belief_acc_2": 0.0,
        })
        return total_loss, scalars

    # Slice model-side positions to K_ep
    posK = our_pos[:K_ep]                           # [K_ep]
    next_pos = posK + 1                             # candidate next indices
    has_next = (next_pos < valid_len)               # [K_ep] bool

    # Episode-side tensors
    actions_t  = torch.tensor([episode["our_action"][i] for i in our_steps_ep_idx[:K_ep]],
                              dtype=torch.long, device=device)                # [K_ep]
    old_logp_t = torch.tensor([episode["log_prob"][i]  for i in our_steps_ep_idx[:K_ep]],
                              dtype=torch.float32, device=device)             # [K_ep]
    rewards    = [float(episode["reward"][i]) for i in our_steps_ep_idx[:K_ep]]

    # ---- 3) Gather logits/values at action states; legal-mask if provided ----
    logits_at = action_logits.index_select(0, posK).float()                   # [K_ep, A]
    if masks_2d is not None:
        mask_at = masks_2d.index_select(0, posK)                              # [K_ep, A] bool
        logits_at = logits_at.masked_fill(~mask_at, float("-inf"))

    values_at = state_values.index_select(0, posK)                            # [K_ep]
    next_values_full = torch.zeros_like(values_at)
    if has_next.any():
        next_values_full[has_next] = state_values.index_select(0, next_pos[has_next])

    # Dones: terminal iff there is no next token within valid_len (true terminal → no bootstrap)
    dones = (~has_next).tolist()

    # ---- 4) GAE ----
    adv, ret = compute_gae(
        rewards, dones,
        values_at.detach().cpu().tolist(),
        next_values_full.detach().cpu().tolist(),
        config.GAMMA, config.GAE_LAMBDA,
    )
    advantages = torch.tensor(adv, dtype=torch.float32, device=device)
    returns    = torch.tensor(ret, dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    # ---- 5) PPO objective ----
    dist     = torch.distributions.Categorical(logits=logits_at)
    new_logp = dist.log_prob(actions_t)
    ratio    = (new_logp - old_logp_t).exp()
    surr1    = ratio * advantages
    surr2    = torch.clamp(ratio, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP) * advantages
    policy_loss  = -torch.min(surr1, surr2).mean()
    value_loss   = F.mse_loss(values_at, returns)
    entropy_loss = -dist.entropy().mean()

    total_loss = policy_loss + 0.5 * value_loss + config.INIT_ENTROPY_COEF * entropy_loss

    # ---- 6) Teacher KL (run teacher here, same input/indices/masks) ----
    if (bc_kl_weight > 0.0) and (sl_teacher is not None):
        with torch.no_grad():
            t_action_logits, *_ = sl_teacher(**mi)                            # [1, L, A]
            t_action_logits = t_action_logits[0, :valid_len, :]               # [L, A]
            t_logits_at = t_action_logits.index_select(0, posK).float()       # [K_ep, A]
            if action_masks is not None:
                t_logits_at = t_logits_at.masked_fill(~mask_at, float("-inf"))
        dist_sl = torch.distributions.Categorical(logits=t_logits_at)
        bc_kl   = torch.distributions.kl_divergence(dist, dist_sl).mean()
        total_loss = total_loss + bc_kl_weight * bc_kl
        scalars["bc_kl"] = float(bc_kl.detach().cpu())

    approx_kl  = torch.mean(old_logp_t - new_logp).detach()
    clipfrac   = ((ratio - 1.0).abs() > config.EPS_CLIP).float().mean().detach()
    scalars.update({
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss":  float(value_loss.detach().cpu()),
        "entropy":     float((-entropy_loss).detach().cpu()),
        "approx_kl":   float(approx_kl.cpu()),
        "clip_fraction": float(clipfrac.cpu()),
    })

    # ---- 7) Belief heads (optional) supervised on OUR steps with valid targets ----
    def _belief_ce_and_acc(b_logits, key_tgt):
        if b_logits is None:
            return torch.zeros((), device=device), 0.0
        # Collect targets aligned to OUR step list, then truncate to K_ep
        targets, keep_idx = [], []
        for i_ep, step_idx in enumerate(our_steps_ep_idx[:K_ep]):
            tgt = episode[key_tgt][step_idx]
            if tgt is not None:
                lab = BELIEF_LABELS_FROM_KIND.get(tgt)
                if lab is not None:
                    targets.append(lab); keep_idx.append(i_ep)
        if not targets:
            return torch.zeros((), device=device), 0.0
        target_t  = torch.tensor(targets, dtype=torch.long, device=device)
        keep_idx  = torch.as_tensor(keep_idx, dtype=torch.long, device=device)
        logits_sel = b_logits[0, :valid_len, :].index_select(0, posK).index_select(0, keep_idx)
        acc = _accuracy_from_logits(logits_sel.detach(), target_t.detach())
        return F.cross_entropy(logits_sel, target_t), acc

    b0_loss, acc_b0 = _belief_ce_and_acc(b0, "belief_tgt0")
    b1_loss, acc_b1 = _belief_ce_and_acc(b1, "belief_tgt1")
    b2_loss, acc_b2 = _belief_ce_and_acc(b2, "belief_tgt2")
    belief_loss = b0_loss + b1_loss + b2_loss
    total_loss  = total_loss + getattr(config, "AUX_BELIEF_WEIGHT", 0.5) * belief_loss
    scalars.update({
        "belief_loss": float(belief_loss.detach().cpu()),
        "belief_acc_0": acc_b0,
        "belief_acc_1": acc_b1,
        "belief_acc_2": acc_b2,
    })

    # ---- 8) Opponent supervised auxiliary (keep within valid_len) ----
    opp_idx = (agent_types_1d != 0).nonzero(as_tuple=False).squeeze(-1)   # [N_opp]
    if opp_logits is not None and opp_idx.numel() > 0:
        opp_ep_idx = [i for i, seat in enumerate(episode["agent_id"])
                      if seat != episode["training_agent_seat"]]
        opp_targets = [episode["opp_target_action"][i] for i in opp_ep_idx
                       if episode["opp_target_action"][i] is not None]
        M = min(len(opp_targets), opp_idx.numel())
        if M > 0:
            opp_logits_sel = opp_logits.index_select(0, opp_idx[:M])       # [M, A]
            opp_targets_t  = torch.tensor(opp_targets[:M], dtype=torch.long, device=device)
            opp_loss = F.cross_entropy(opp_logits_sel, opp_targets_t)
            total_loss = total_loss + getattr(config, "AUX_OPP_WEIGHT", 0.5) * opp_loss
            scalars.update({
                "opp_loss": float(opp_loss.detach().cpu()),
                "n_opp_supervised": float(M),
                "opp_action_acc": _accuracy_from_logits(opp_logits_sel.detach(), opp_targets_t.detach()),
            })
        else:
            scalars.update({"opp_loss": 0.0, "n_opp_supervised": 0.0, "opp_action_acc": 0.0})
    else:
        scalars.update({"opp_loss": 0.0, "n_opp_supervised": 0.0, "opp_action_acc": 0.0})

    return total_loss, scalars


# --------------------------------------------------------------------------------------
# Training loop (This function is correct and remains unchanged)
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
    scaler = amp.GradScaler(device=device, enabled=(device.type == 'cuda'))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    logging.info(f"TensorBoard logdir: {log_dir}")
    arena = lb.VecArena()
    CKPT_PATH = r"checkpoints\autoreg_20250823_224827\autoreg_model_final.pth"
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    learner = BatchPPOAutoregressiveAgent(device, "TrainAgent_v1")
    checkpoint = {"policy_nets": {"agent_model": checkpoint["model_state_dict"]}}
    agent_key = next(iter(checkpoint["policy_nets"]))
    learner.load_models_from_checkpoint(checkpoint, agent_key)
    model = learner.model
    sl_teacher = copy.deepcopy(model).eval()
    for p in sl_teacher.parameters(): p.requires_grad = False
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, eps=1e-5)
    logging.info(f"Loaded SL checkpoint: {CKPT_PATH}")
    policies = {0: learner}
    rollout_manager = PPOVecRolloutManager(arena, policies, device)
    HC_POOL = [
        lb.BotKind.Classic, lb.BotKind.GreedyCardSpammer, lb.BotKind.RandomAgent,
        lb.BotKind.SelectiveTableConservativeChallenger, lb.BotKind.StrategicChallenger,
        lb.BotKind.TableFirstConservativeChallenger, lb.BotKind.TableNonTableAgent,
    ]
    for update in range(1, num_updates + 1):
        update_start = time.time()
        model.eval()
        episodes = rollout_manager.collect_episodes(
            num_episodes=episodes_per_update, num_players=config.NUM_PLAYERS,
            training_policy_id=0, opponent_pool=HC_POOL
        )
        if not episodes:
            logging.warning(f"Update {update}/{num_updates}: No episodes collected. Skipping update.")
            continue
        logging.info(f"Update {update}/{num_updates} | Collected {len(episodes)} episodes | Starting training...")

        # No precompute step
        model.train()
        agg, n_loss_terms = {}, 0

        for _ in range(k_epochs):
            random.shuffle(episodes)
            for ep in episodes:
                with amp.autocast(device_type=device.type, dtype=torch.float16):
                    loss, scalars = ppo_losses_for_episode(
                        model, ep, device,
                        sl_teacher=sl_teacher,
                        bc_kl_weight=getattr(config, "BC_KL_WEIGHT", 0.0),
                    )
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=getattr(config, "MAX_NORM", 0.5))
                scaler.step(optimizer)
                scaler.update()

                for k, v in scalars.items():
                    agg[k] = agg.get(k, 0.0) + float(v)
                agg["total_loss"] = agg.get("total_loss", 0.0) + float(loss.detach().cpu())
                n_loss_terms += 1
        avg = lambda name: (agg.get(name, 0.0) / max(1, n_loss_terms))
        avg_total_loss = avg("total_loss")
        logging.info(f"Update {update}/{num_updates} training complete | avg_loss={avg_total_loss:.4f} | time={time.time()-update_start:.2f}s")
        win_rate = sum(ep["win"] for ep in episodes) / len(episodes)
        writer.add_scalar("Loss/Total", avg_total_loss, update)
        writer.add_scalar("Loss/Policy", avg("policy_loss"), update)
        writer.add_scalar("Loss/Value", avg("value_loss"), update)
        writer.add_scalar("Loss/Aux/Opponent", avg("opp_loss"), update)
        writer.add_scalar("Loss/Aux/Belief", avg("belief_loss"), update)
        writer.add_scalar("Policy/Entropy", avg("entropy"), update)
        writer.add_scalar("Policy/ApproxKL", avg("approx_kl"), update)
        writer.add_scalar("Policy/ClipFraction", avg("clip_fraction"), update)
        writer.add_scalar("Policy/SL_KL", avg("bc_kl"), update)
        writer.add_scalar("Acc/OpponentAction", avg("opp_action_acc"), update)
        writer.add_scalar("Acc/Belief0", avg("belief_acc_0"), update)
        writer.add_scalar("Acc/Belief1", avg("belief_acc_1"), update)
        writer.add_scalar("Acc/Belief2", avg("belief_acc_2"), update)
        writer.add_scalar("Rollout/WinRate", win_rate, update)
        writer.add_scalar("Rollout/EpisodeReturnMean", sum(ep["episode_return"] for ep in episodes) / len(episodes), update)
        writer.add_scalar("Rollout/EpisodeLenMean", sum(len(ep["reward"]) for ep in episodes) / len(episodes), update)
        if checkpoint_dir and (update % getattr(config, "CHECKPOINT_INTERVAL", 200) == 0):
            os.makedirs(checkpoint_dir, exist_ok=True)
            path = os.path.join(checkpoint_dir, f"arppo_update_{update}.pth")
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "update": update}, path)
            logging.info(f"Saved checkpoint to {path}")
    writer.close()

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_autoreg_{timestamp}"
    log_dir = os.path.join("logs", run_name)
    ckpt_dir = os.path.join(getattr(config, "CHECKPOINT_DIR", "checkpoints"), run_name)
    train(
        num_updates=2000,
        episodes_per_update=256,
        k_epochs=config.K_EPOCHS,
        checkpoint_dir=ckpt_dir,
        log_dir=log_dir,
    )