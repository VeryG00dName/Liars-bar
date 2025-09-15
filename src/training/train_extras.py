# src/training/train_extras.py

import random
import numpy as np
import torch
from src import config
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
import torch.nn.functional as F

def set_seed(seed=42):
    """
    Sets the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def convert_memory_to_features(memory, response_mapping, action_mapping):
    """
    Convert the opponent memory (a list of events) to a list of 4-dimensional feature vectors.
    Each event is expected to be a dictionary with keys: "response", "triggering_action", "penalties", and "card_count".
    """
    features = []
    for event in memory:
        if not isinstance(event, dict):
            raise ValueError(f"Memory event is not a dictionary: {event}. Please fix the data generation.")
        resp = event.get("response", "")
        act = event.get("triggering_action", "")
        penalties = float(event.get("penalties", 0))
        card_count = float(event.get("card_count", 0))
        # Map the categorical features using the provided mappings.
        resp_val = float(response_mapping.get(resp, 0))
        act_val = float(action_mapping.get(act, 0))
        features.append([resp_val, act_val, penalties, card_count])
    return features

def convert_memory_to_features2(memory, response_mapping, action_mapping):
    """
    Convert the opponent memory (a list of events) to a list of 5-dimensional feature vectors.
    Each event is expected to be a dictionary with keys: "response", "triggering_action", 
    "penalties", "card_count", and optionally "challenge_success".
    
    challenge_success will be:
    - 1.0 if the challenge was successful (play was a bluff)
    - 0.0 if the challenge was unsuccessful (play was honest)
    - -1.0 if not applicable (e.g., for Play actions where no challenge occurred)
    """
    features = []
    for event in memory:
        if not isinstance(event, dict):
            raise ValueError(f"Memory event is not a dictionary: {event}. Please fix the data generation.")
            
        resp = event["response"]
        act = event["triggering_action"]
        penalties = float(event["penalties"])
        card_count = float(event["card_count"])
        
        # Get challenge_success value, use -1.0 as placeholder when None
        challenge_success_val = -1.0
        if event["challenge_success"] is not None:
            challenge_success_val = 1.0 if event["challenge_success"] else 0.0
        
        # Map the categorical features using the provided mappings
        resp_val = float(response_mapping.get(resp, 0))
        act_val = float(action_mapping.get(act, 0))
        
        features.append([resp_val, act_val, penalties, card_count, challenge_success_val])
        
    return features

def extract_obp_features_from_action(action_entry):
    """
    Extracts features from a single opponent action entry suitable for OBP input.
    """
    atype_onehot = [0.0, 0.0, 0.0]
    if action_entry['action_type'] == "Play":
        atype_onehot[1] = 1.0
    elif action_entry['action_type'] == "Challenge":
        atype_onehot[2] = 1.0
    else:
        atype_onehot[0] = 1.0

    count_val = 0.0
    if action_entry['count'] is not None:
        count_val = float(action_entry['count']) / 5.0

    features = atype_onehot + [count_val]
    return features


def extract_obp_training_data(env):
    """
    Extract (features, memory_embedding, label) triplets for OBP training from private_opponent_histories.
    The memory_embedding is computed from memory events via the transformer.
    """
    training_data = []
    # Assume that response2idx, action2idx, event_encoder, and strategy_transformer are loaded
    global response2idx, action2idx, event_encoder, strategy_transformer
    for agent in env.possible_agents:
        for entry in env.private_opponent_histories[agent]:
            if entry['action_type'] == "Play" and entry['was_bluff'] is not None:
                features = extract_obp_features_from_action(entry)
                label = 1 if entry['was_bluff'] else 0
                if 'memory_events' in entry and entry['memory_events']:
                    features_list = convert_memory_to_features(entry['memory_events'], response2idx, action2idx)
                    if features_list:
                        feature_tensor = torch.tensor(features_list, dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            projected = event_encoder(feature_tensor)
                            memory_embedding, _ = strategy_transformer(projected)
                        # Convert to a list (or keep as tensor)
                        memory_embedding = memory_embedding.squeeze(0).cpu().detach().numpy().tolist()
                    else:
                        memory_embedding = [0.0] * config.STRATEGY_DIM
                else:
                    memory_embedding = [0.0] * config.STRATEGY_DIM
                training_data.append((features, memory_embedding, label))
    return training_data


def run_obp_inference(obp_model, obs_array, device, num_players, memory_embeddings):
    """
    Run OBP inference on public opponent features.
    memory_embeddings: a list of memory embedding tensors (one per opponent) to be passed to OBP.
    """
    if obp_model is None:
        num_opponents = num_players - 1
        return [0.0] * num_opponents

    num_opponents = num_players - 1
    opp_feature_dim = 4  # (bluff_freq removed)

    hand_vector_length = 2
    last_action_val_length = 1
    active_players_length = num_players
    non_opponent_features_length = hand_vector_length + last_action_val_length + active_players_length

    obp_probs = []
    for i in range(num_opponents):
        start_idx = non_opponent_features_length + (i * opp_feature_dim)
        end_idx = start_idx + opp_feature_dim
        opp_vec = obs_array[start_idx:end_idx]
        opp_vec_tensor = torch.tensor(opp_vec, dtype=torch.float32, device=device).unsqueeze(0)
        # Pass the corresponding memory embedding (assumed to be a tensor of shape [1, STRATEGY_DIM])
        logits = obp_model(opp_vec_tensor, memory_embeddings[i])
        probs = torch.softmax(logits, dim=-1)
        bluff_prob = probs[0, 1].item()
        obp_probs.append(bluff_prob)
    return obp_probs

def collect_strategy_embeddings(agents, opponents, embeddings_dict):
    """
    Collect strategy embeddings for visualization.

    Args:
        agents: List of agent IDs.
        opponents: List of opponent IDs/types.
        embeddings_dict: Dictionary mapping (agent, opponent) tuples to embeddings.

    Returns:
        X: Array of embeddings.
        labels: List of (agent, opponent) pairs corresponding to each embedding.
    """
    X = []
    labels = []
    
    for agent in agents:
        for opponent in opponents:
            if (agent, opponent) in embeddings_dict:
                X.append(embeddings_dict[(agent, opponent)])
                labels.append((agent, opponent))
    
    return np.array(X), labels

REFERENCE_PCA = None

def initialize_reference_pca(reference_embeddings):
    """
    Initialize the global reference PCA using the provided reference embeddings.

    Args:
        reference_embeddings (numpy.ndarray): A 2D array of reference embeddings.
    """
    global REFERENCE_PCA
    if REFERENCE_PCA is None:
        REFERENCE_PCA = PCA(n_components=2)
        REFERENCE_PCA.fit(reference_embeddings)

def visualize_strategy_embeddings(writer, embeddings_dict, agents, opponents, episode, method='tsne', reference_embeddings=None):
    """
    Visualize strategy embeddings using dimensionality reduction.
    If method is 'pca', a fixed PCA transform is computed from reference_embeddings.
    The legend is placed outside the plot.

    Args:
        writer: TensorBoard writer.
        embeddings_dict (dict): Mapping (agent, opponent) -> embedding (numpy array).
        agents (list): List of agent IDs.
        opponents (list): List of opponent IDs/types.
        episode (int): Current episode number.
        method (str): 'pca' or 'tsne'.
        reference_embeddings (numpy.ndarray): 2D array of embeddings for initializing PCA.
    """
    X, labels = collect_strategy_embeddings(agents, opponents, embeddings_dict)
    if len(X) < 2:
        return  # Need at least 2 points

    if method == 'pca':
        if reference_embeddings is None:
            raise ValueError("Reference embeddings must be provided for PCA.")
        initialize_reference_pca(reference_embeddings)
        X_2d = REFERENCE_PCA.transform(X)
    else:
        reducer = TSNE(n_components=2, perplexity=min(30, len(X)-1) if len(X) > 1 else 1)
        X_2d = reducer.fit_transform(X)

    plt.figure(figsize=(10, 8))
    # Instead of unsorted sets, we use sorted lists so the ordering is consistent.
    unique_agents = sorted(set(label[0] for label in labels))
    unique_opponents = sorted(set(label[1] for label in labels))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_agents)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
    agent_to_color = {agent: colors[i] for i, agent in enumerate(unique_agents)}
    opponent_to_marker = {opponent: markers[i % len(markers)] for i, opponent in enumerate(unique_opponents)}

    for (agent, opponent), point in zip(labels, X_2d):
        plt.scatter(point[0], point[1], color=agent_to_color[agent],
                    marker=opponent_to_marker[opponent], s=100)

    agent_patches = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=agent_to_color[agent], markersize=10,
                      label=f'Agent {agent}') for agent in unique_agents]
    opponent_patches = [plt.Line2D([0], [0], marker=opponent_to_marker[opponent],
                         color='black', markersize=10,
                         label=f'Opponent {opponent}') for opponent in unique_opponents]
    
    plt.legend(handles=agent_patches + opponent_patches,
               bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplots_adjust(right=0.7)
    
    plt.title(f'Strategy Embeddings ({method.upper()}) - Episode {episode}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image_array = np.array(image)
    writer.add_image(f'Strategy_Embeddings_{method.upper()}', image_array, episode, dataformats='HWC')
    plt.close()


EPS_CLIP          = float(getattr(config, "EPS_CLIP", 0.2))
ENT_COEF          = float(getattr(config, "INIT_ENTROPY_COEF", 0.005))
TRINAL_DELTA1     = float(getattr(config, "TRINAL_DELTA1", 1.8))
GAMMA             = float(getattr(config, "GAMMA", 0.974))
GAE_LAMBDA        = float(getattr(config, "GAE_LAMBDA", 0.98))

# --- Loss Function Weights ---
VALUE_WEIGHT      = float(getattr(config, "VALUE_WEIGHT", 1.0))
AUX_BELIEF_WEIGHT = float(getattr(config, "AUX_BELIEF_WEIGHT", 0.3))
AUX_OPP_WEIGHT    = float(getattr(config, "AUX_OPP_WEIGHT", 0.5))
BC_KL_WEIGHT      = float(getattr(config, "BC_KL_WEIGHT", 0.002))

# --- Stakes-Based Value Clipping Hyperparameters ---
EPS_V                  = float(getattr(config, "EPS_V", 0.9))
RET_STD_EMA_DECAY      = float(getattr(config, "RET_STD_EMA_DECAY", 0.99))
STAKES_CHALLENGE_BASE  = float(getattr(config, "STAKES_CHALLENGE_BASE", 4.0))
STAKES_BASE_EXP        = float(getattr(config, "STAKES_BASE_EXP", 1.0))
STAKES_PEN_NORM        = float(getattr(config, "STAKES_PEN_NORM", 4.0))
STAKES_PEN_EXP         = float(getattr(config, "STAKES_PEN_EXP", 1.0))
STAKES_CLIP_MIN        = float(getattr(config, "STAKES_CLIP_MIN", 0.5))
STAKES_CLIP_MAX        = float(getattr(config, "STAKES_CLIP_MAX", 3.5))
def _cards_base_from_action(action_ids: torch.Tensor) -> torch.Tensor:
    base = ((action_ids % 3) + 1).to(torch.float32)
    base = torch.where(action_ids == 6,
                       torch.full_like(base, STAKES_CHALLENGE_BASE, dtype=base.dtype),
                       base)
    hi = max(STAKES_CHALLENGE_BASE, 3.0)
    return torch.clamp(base, 1.0, hi).pow(STAKES_BASE_EXP)

def _stakes_multiplier_public(action_ids: torch.Tensor, penalties_used: torch.Tensor) -> torch.Tensor:
    base = _cards_base_from_action(action_ids)
    pen  = penalties_used.to(torch.float32).clamp_min(0.0)
    pen_factor = (1.0 + pen / max(STAKES_PEN_NORM, 1.0)) ** STAKES_PEN_EXP
    mult = base * pen_factor
    return torch.clamp(mult, STAKES_CLIP_MIN, STAKES_CLIP_MAX)

def _value_loss_with_stakes_clip_public(
    v_pred: torch.Tensor,
    returns: torch.Tensor,
    action_ids: torch.Tensor,
    penalties_used: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched-safe stakes-aware value loss with clipping of the *target*.
    Accepts matching shapes (e.g., [N] or [B, T]) for all tensors.

    Uses an EMA of the return std stored on `config.RET_STD_EMA` to scale the clip range.
    Returns:
      (mse_loss, clip_frac) where clip_frac is the fraction of samples whose targets were clipped.
    """
    # Ensure fp32 for the math; shapes propagate
    v_pred  = v_pred.to(torch.float32)
    returns = returns.to(torch.float32)

    with torch.no_grad():
        r_flat = returns.reshape(-1)
        n = int(r_flat.numel())
        if n < 2:
            batch_std = 1.0
        else:
            nz = (r_flat.abs() > 1e-8)
            if nz.float().mean().item() >= 0.2:  # enough non-zeros → robust std
                batch_std = r_flat[nz].std(unbiased=False).clamp(min=1e-3).item()
            else:
                batch_std = 1.0

        # Smooth std via EMA (module-level state in config)
        prev_ema = config.RET_STD_EMA
        new_ema  = RET_STD_EMA_DECAY * prev_ema + (1.0 - RET_STD_EMA_DECAY) * batch_std
        config.RET_STD_EMA = float(new_ema)
        ret_scale = config.RET_STD_EMA

    # Stakes multiplier derived from public info (same shape as inputs)
    stakes = _stakes_multiplier_public(action_ids, penalties_used).to(torch.float32)

    # Per-sample clip band scaled by stakes and EMA’d return std
    delta = EPS_V * stakes * ret_scale
    lower = -delta
    upper =  delta

    with torch.no_grad():
        clip_mask = (returns < lower) | (returns > upper)
        clip_frac = clip_mask.float().mean()

    target = torch.clamp(returns, min=lower, max=upper)
    loss = torch.nn.functional.mse_loss(v_pred, target)
    return loss, clip_frac

# ---------------------- Batched PPO loss (graph-safe) ----------------------
def ppo_losses_batched(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    sl_teacher: Optional[torch.nn.Module] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Fully batched PPO objective, updated to use boolean masks and a 
    correct, efficient advantage calculation for sparse terminal rewards.
    """
    # --- Step 1: Unpack batch (unchanged) ---
    mi = batch["mi"]
    our_mask = batch["our_mask"].bool()         # [B, L_pad] - For our agent's steps
    opp_mask = batch["opp_mask"].bool()         # [B, L_pad] - For opponent's steps
    actions = batch["actions"].long()           # [B, L_pad]
    old_logp = batch["old_logp"].float()        # [B, L_pad]
    rewards = batch["rewards"].float()          # [B, L_pad]
    penalties_used = batch["penalties_used"]    # [B, L_pad]
    padding_mask = mi["padding_mask"].bool()    # [B, L_pad]

    # --- Step 2: Model Forward Pass (unchanged) ---
    outs = model(**mi)
    action_logits = outs[0]
    opp_logits = outs[1]
    values_full = outs[2].squeeze(-1).to(torch.float32)
    b0 = outs[3] if len(outs) > 3 else None
    b1 = outs[4] if len(outs) > 4 else None
    b2 = outs[5] if len(outs) > 5 else None
    
    B, L_pad = our_mask.shape

    # --- Step 3: Advantage & Return Calculation for Terminal Rewards ---
    with torch.no_grad():
        # The reward tensor 'rewards' is sparse. It's 0 everywhere except for one step
        # (our last action) which has +1.0 or -1.0.
        
        # We need to find the discounted returns (G_t) for all steps.
        # We can do this with a single backward pass (dynamic programming).
        returns = torch.zeros_like(rewards)
        next_return = torch.zeros(B, device=rewards.device)
        
        for t in reversed(range(L_pad)):
            # If this step is padded, it's not part of the episode. Reset and continue.
            is_padded = padding_mask[:, t]
            
            # The return at this step is its own reward + discounted next return
            # This works because all r_t are 0 until the final step.
            current_return = rewards[:, t] + GAMMA * next_return
            
            # Update next_return for the step *before* this one
            next_return = torch.where(is_padded, 0.0, current_return)
            
            # Store the return
            returns[:, t] = current_return

        # Advantages are now simple: A(t) = G(t) - V(s_t)
        # We only care about advantages at the steps we took action.
        advantages = returns - values_full
        advantages = advantages.where(~padding_mask, 0.0) # Zero out pad advantages

    # Normalize advantages over valid 'our' steps
    if our_mask.any():
        adv_masked = advantages[our_mask]
        adv_mean = adv_masked.mean()
        adv_std = adv_masked.std().clamp_min(1e-8)
        advantages = (advantages - adv_mean) / adv_std
    
    # --- Step 4: PPO Objective using Masks ---
    valid_logits = action_logits[our_mask]
    valid_actions = actions[our_mask]
    valid_old_logp = old_logp[our_mask]
    valid_advantages = advantages[our_mask]
    
    dist = torch.distributions.Categorical(logits=valid_logits)
    valid_entropy = dist.entropy()
    new_logp = dist.log_prob(valid_actions)
    
    ratio = (new_logp - valid_old_logp).exp()
    
    surr1 = ratio * valid_advantages
    surr2 = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * valid_advantages
    
    # Check for empty batch to prevent .mean() on empty tensor
    if valid_advantages.numel() > 0:
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -valid_entropy.mean() * ENT_COEF
    else:
        policy_loss = torch.tensor(0.0, device=values_full.device)
        entropy_loss = torch.tensor(0.0, device=values_full.device)

    # --- Step 5: Value Loss using Masks ---
    valid_returns = returns[our_mask]
    valid_values = values_full[our_mask]
    
    if valid_values.numel() > 0:
        value_loss, vclip_frac = _value_loss_with_stakes_clip_public(
            v_pred=valid_values,
            returns=valid_returns,
            action_ids=valid_actions,
            penalties_used=penalties_used[our_mask].long(),
        )
    else:
        value_loss = torch.tensor(0.0, device=values_full.device)
        vclip_frac = torch.tensor(0.0, device=values_full.device)
    
    # --- Step 6: Auxiliary Losses using Masks (Filled in) ---
    belief_loss_total = torch.tensor(0.0, device=values_full.device)
    acc0, acc1, acc2 = [torch.tensor(0.0, device=values_full.device) for _ in range(3)]

    # Helper for calculating aux loss & acc safely
    def _compute_aux(logits, targets, mask):
        if logits is None:
            return torch.tensor(0.0), torch.tensor(0.0) # Return dummy tensors with no device needed
        if not mask.any():
            return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)
        
        valid_logits = logits[mask]
        valid_targets = targets[mask]
        
        if valid_targets.numel() == 0:
            return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)
            
        loss = F.cross_entropy(valid_logits, valid_targets, ignore_index=-100)
        
        with torch.no_grad():
            valid_preds = valid_logits.argmax(dim=-1)
            valid_mask = valid_targets != -100
            if valid_mask.any():
                acc = (valid_preds[valid_mask] == valid_targets[valid_mask]).float().mean()
            else:
                acc = torch.tensor(0.0, device=logits.device)
                
        return loss, acc

    # Belief Loss
    b0_loss, acc0 = _compute_aux(b0, batch.get("belief_tgt0"), our_mask)
    b1_loss, acc1 = _compute_aux(b1, batch.get("belief_tgt1"), our_mask)
    b2_loss, acc2 = _compute_aux(b2, batch.get("belief_tgt2"), our_mask)
    belief_loss_total = b0_loss + b1_loss + b2_loss # Sum the losses

    # Opponent Action Loss
    opp_loss, opp_acc = _compute_aux(opp_logits, batch.get("opp_targets"), opp_mask)

    # --- Final Combination ---
    total = policy_loss + VALUE_WEIGHT * value_loss + entropy_loss \
            + AUX_BELIEF_WEIGHT * belief_loss_total + AUX_OPP_WEIGHT * opp_loss
            
    # --- Metrics (calculated on valid data) ---
    with torch.no_grad():
        if new_logp.numel() > 0:
            clipfrac = ((ratio - 1.0).abs() > EPS_CLIP).float().mean()
            approx_kl = (valid_old_logp - new_logp).mean()
            ent_mean = valid_entropy.mean()
        else:
            clipfrac = approx_kl = ent_mean = torch.tensor(0.0, device=values_full.device)

        metrics = {
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": ent_mean.detach(),
            "approx_kl": approx_kl.detach(),
            "clip_fraction": clipfrac.detach(),
            "value_clip_frac": vclip_frac.detach(),
            "belief_loss": (belief_loss_total / 3.0).detach(), # Average loss per head
            "opp_loss": opp_loss.detach(),
            "belief_acc_0": acc0.detach(),
            "belief_acc_1": acc1.detach(),
            "belief_acc_2": acc2.detach(),
            "opp_action_acc": opp_acc.detach(),
        }
    
    return total, metrics


def _collate_batch(
    episodes: List[Dict[str, Any]],
    L_max: Optional[int] = None,
    pin_memory: bool = False,
    ignore_index: int = -100,
    oracle_mode: bool = False
) -> Dict[str, torch.Tensor]:
    """
    CPU-side collation. Returns tensors on CPU so _to_device_batch(...) moves them.
    Correctly handles indexing on a single, unified timeline.
    """
    IGN = int(ignore_index)
    
    valid_episodes = [ep for ep in episodes if ep.get("model_input") and ep["model_input"].get("valid_lengths") is not None]
    B = len(valid_episodes)
    if B == 0:
        return {}

    raw_lens = [int(ep["model_input"]["valid_lengths"].item()) for ep in valid_episodes]
    L_pad = L_max if L_max is not None else max(raw_lens)
    if L_pad == 0: return {}

    def _pad_trim(v: torch.Tensor, L_tgt: int):
        L = v.size(1)
        if L == L_tgt: return v
        if L > L_tgt:  return v[:, :L_tgt, ...]
        pad_len = L_tgt - L
        pad_shape = list(v.shape); pad_shape[1] = pad_len
        z = torch.zeros(pad_shape, dtype=v.dtype, device=v.device)
        return torch.cat([v, z], dim=1)
    
    EXPECTED_MI_KEYS = {"obs_sequence", "action_sequence", "agent_types", "positions", "action_masks"}
    mi_batch: Dict[str, torch.Tensor] = {}
    for k in sorted(list(EXPECTED_MI_KEYS)):
        vs = [ep["model_input"].get(k) for ep in valid_episodes]
        if any(v is None for v in vs): continue
        padded = [_pad_trim(v, L_pad) for v in vs]
        mi_batch[k] = torch.cat(padded, dim=0).contiguous()
    
    valid_lengths = torch.tensor([min(l, L_pad) for l in raw_lens], dtype=torch.long)
    padding_mask = torch.arange(L_pad)[None, :] >= valid_lengths[:, None]
    mi_batch["padding_mask"] = padding_mask
    mi_batch["valid_lengths"] = valid_lengths
    
    agent_types = mi_batch["agent_types"].long()
    
    T_full = L_pad
    def _pm(x: torch.Tensor): return x.pin_memory() if pin_memory else x

    our_mask, actions, old_logp, rewards, pen_used = [_pm(t) for t in [
        torch.zeros((B, T_full), dtype=torch.bool), torch.full((B, T_full), IGN, dtype=torch.long),
        torch.zeros((B, T_full), dtype=torch.float32), torch.zeros((B, T_full), dtype=torch.float32),
        torch.zeros((B, T_full), dtype=torch.long)]]
    belief_tgt0, belief_tgt1, belief_tgt2 = [_pm(torch.full((B, T_full), IGN, dtype=torch.long)) for _ in range(3)]
    opp_mask, opp_targets = _pm(torch.zeros((B, T_full), dtype=torch.bool)), _pm(torch.full((B, T_full), IGN, dtype=torch.long))
    opp_belief_mask, opp_belief_tgt0, opp_belief_tgt1, opp_belief_tgt2 = [_pm(t) for t in [
        torch.zeros((B, T_full), dtype=torch.bool), torch.full((B, T_full), IGN, dtype=torch.long),
        torch.full((B, T_full), IGN, dtype=torch.long), torch.full((B, T_full), IGN, dtype=torch.long)]]
    
    for b, ep in enumerate(valid_episodes):
        true_len_mi = raw_lens[b]
        true_len_ep = len(ep["agent_id"]) # The number of completed steps
        
        # We iterate over the COMPLETED steps. The final observation token in mi_batch is ignored.
        for t_ep in range(true_len_ep):
            agent_seat_in_ep = ep["agent_id"][t_ep]
            is_our_turn = (agent_seat_in_ep == ep["training_agent_seat"])
            
            # The position 't_ep' is also the index in the unpadded model_input sequence
            pos_in_mi = t_ep

            if pos_in_mi >= T_full: continue # Should not happen if L_max is large enough

            # Check if this step corresponds to the correct agent type in the model input
            # This is a critical sanity check
            mi_agent_type = agent_types[b, pos_in_mi].item()
            is_our_turn_in_mi = (mi_agent_type == 0)

            if is_our_turn != is_our_turn_in_mi:
                # This would indicate a major desync between rollout and C++ data prep
                continue

            if is_our_turn:
                if ep["log_prob"][t_ep] is not None:
                    our_mask[b, pos_in_mi] = True
                    actions[b, pos_in_mi] = int(ep["our_action"][t_ep])
                    old_logp[b, pos_in_mi] = float(ep["log_prob"][t_ep])
                    rewards[b, pos_in_mi] = float(ep["reward"][t_ep])
                    pen_used[b, pos_in_mi] = int(ep["penalties_used"][t_ep])
                    b0, b1, b2 = ep["belief_tgt0"][t_ep], ep["belief_tgt1"][t_ep], ep["belief_tgt2"][t_ep]
                    if b0 is not None: belief_tgt0[b, pos_in_mi] = int(b0)
                    if b1 is not None: belief_tgt1[b, pos_in_mi] = int(b1)
                    if b2 is not None: belief_tgt2[b, pos_in_mi] = int(b2)
            else: # Opponent turn
                if ep["opp_target_action"][t_ep] is not None:
                    opp_mask[b, pos_in_mi] = True
                    opp_targets[b, pos_in_mi] = int(ep["opp_target_action"][t_ep])
                
                if oracle_mode:
                    b0, b1, b2 = ep["opp_belief_tgt0"][t_ep], ep["opp_belief_tgt1"][t_ep], ep["opp_belief_tgt2"][t_ep]
                    if all(x is not None for x in [b0, b1, b2]):
                        opp_belief_mask[b, pos_in_mi] = True
                        opp_belief_tgt0[b, pos_in_mi] = int(b0)
                        opp_belief_tgt1[b, pos_in_mi] = int(b1)
                        opp_belief_tgt2[b, pos_in_mi] = int(b2)
    
    final_dict = {
        "mi": mi_batch, "our_mask": our_mask, "actions": actions,
        "old_logp": old_logp, "rewards": rewards, "penalties_used": pen_used,
        "our_action_mask": mi_batch.get("action_masks"),
        "belief_tgt0": belief_tgt0, "belief_tgt1": belief_tgt1, "belief_tgt2": belief_tgt2,
        "opp_mask": opp_mask, "opp_targets": opp_targets,
    }

    if oracle_mode:
        final_dict["opp_belief_tgt0"] = opp_belief_tgt0
        final_dict["opp_belief_tgt1"] = opp_belief_tgt1
        final_dict["opp_belief_tgt2"] = opp_belief_tgt2
        final_dict["opp_belief_mask"] = opp_belief_mask
        
    return final_dict


def _to_device_batch(batch_cpu: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move a collated CPU batch (with nested 'mi' dict) to device."""
    if not batch_cpu:
        return {}
        
    out = {}
    # Move main MI dict
    out["mi"] = {k: v.to(device, non_blocking=True) for k, v in batch_cpu["mi"].items()}

    # Move all other tensors
    for key, tensor in batch_cpu.items():
        if key != "mi":
            if torch.is_tensor(tensor):
                out[key] = tensor.to(device, non_blocking=True)
            else:
                out[key] = tensor # Handle non-tensor data like our_action_mask=None
                
    return out
