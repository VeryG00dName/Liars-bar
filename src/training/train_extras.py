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
    Fully batched PPO objective with:
      • irregular-step GAE computed inside (from batch['rewards'] & model values)
      • stakes-based value target clipping (optional)
      • belief-head CE (batched, masked; -100 ignored)
      • opponent action CE (batched; NO action masking; -100 ignored)
      • optional teacher KL at OUR steps

    Requires in batch:
      mi, our_idx [B,T], mask [B,T], actions [B,T], old_logp [B,T],
      rewards [B,T], penalties_used [B,T],
      our_action_mask [B,L,A] or None,
      belief_tgt{0,1,2}, belief{0,1,2}_mask,
      opp_idx [B,To], opp_targets [B,To], opp_have_label [B,To]
    """
    mi = batch["mi"]
    our_idx = batch["our_idx"].long()     # [B, T]
    our_mask = batch["mask"].bool()       # [B, T]
    actions = batch["actions"].long()     # [B, T]
    old_logp = batch["old_logp"].float()  # [B, T]
    rewards = batch["rewards"].float()    # [B, T]

    outs = model(**mi)
    action_logits = outs[0]                                # [B, L, A]
    opp_logits    = outs[1] if len(outs) > 1 else None     # [B, L, A] or None
    values_full   = outs[2].squeeze(-1).to(torch.float32)  # [B, L]
    b0 = outs[3] if len(outs) > 3 else None                # [B, L, C0] or None
    b1 = outs[4] if len(outs) > 4 else None
    b2 = outs[5] if len(outs) > 5 else None

    B, T = our_idx.shape
    A = action_logits.size(-1)

    # ---- Gather OUR-step logits ----
    logits_at = action_logits.gather(1, our_idx.unsqueeze(-1).expand(-1, -1, A))  # [B,T,A]

    def _neg_inf_like(x: torch.Tensor) -> torch.Tensor:
        # returns scalar tensor on same device/dtype with the most negative finite value
        return torch.tensor(torch.finfo(x.dtype).min, dtype=x.dtype, device=x.device)
    
    # Apply legality mask for OUR steps only (if provided)
    if batch.get("our_action_mask", None) is not None:
        step_mask = batch["our_action_mask"].gather(  # [B,T,A]
            1, our_idx.unsqueeze(-1).expand(-1, -1, A)
        )
        invalid_rows = (~step_mask).all(dim=-1)  # [B,T]
        if invalid_rows.any():
            fb_cols = logits_at[invalid_rows].argmax(dim=-1)
            step_mask[invalid_rows] = False
            step_mask[invalid_rows, fb_cols] = True
        logits_at = logits_at.masked_fill(~step_mask, _neg_inf_like(logits_at))

    logits_at = torch.nan_to_num(logits_at, nan=0.0, posinf=0.0, neginf=float(torch.finfo(logits_at.dtype).min))
    values_at = values_full.gather(1, our_idx)  # [B,T]
    values_at = torch.nan_to_num(values_at, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Build "next" indices & gaps for irregular-step GAE ----
    # next_idx[b,t] = our_idx[b,t+1], except last where no next
    next_idx = torch.full_like(our_idx, -1)
    if T > 1:
        next_idx[:, :-1] = our_idx[:, 1:]
    has_next = next_idx.ge(0) & our_mask  # [B,T]

    # gaps = (next - current), clamped >= 1; zero where no next
    gaps = torch.zeros_like(our_idx, dtype=torch.long)
    valid_gap = has_next & our_mask
    gaps[valid_gap] = (next_idx[valid_gap] - our_idx[valid_gap]).clamp_min(1)


    gamma_gap = (GAMMA ** gaps.to(torch.float32))   # [B,T]
    lam_gap   = (GAE_LAMBDA   ** gaps.to(torch.float32))   # [B,T]

    # ---- Irregular-step GAE (vectorized backward over T) ----
    advantages = torch.zeros_like(values_at)
    lastgaelam = torch.zeros((B,), device=values_at.device, dtype=torch.float32)

    for t in reversed(range(T)):
        g  = torch.where(has_next[:, t], gamma_gap[:, t], torch.zeros_like(gamma_gap[:, t]))
        gl = torch.where(has_next[:, t], gamma_gap[:, t] * lam_gap[:, t], torch.zeros_like(gamma_gap[:, t]))
        L = values_full.size(1)
        idx_safe = next_idx[:, t].clamp(0, L - 1)  # <-- clamp_max added
        nv = torch.where(
            has_next[:, t],
            values_full.gather(1, idx_safe.unsqueeze(-1)).squeeze(-1),
            torch.zeros_like(values_at[:, t]),
        )
        delta = rewards[:, t] + g * nv - values_at[:, t]
        lastgaelam = delta + gl * lastgaelam
        advantages[:, t] = lastgaelam

        # reset the accumulator where this time-step is invalid (keeps masked mean clean)
        lastgaelam = torch.where(our_mask[:, t], lastgaelam, lastgaelam * 0.0)

    returns = (advantages + values_at)
    # Normalize advantages using only valid positions
    m = our_mask.to(torch.float32)
    adv_mean = (advantages * m).sum() / m.sum().clamp_min(1.0)
    adv_var  = ((advantages - adv_mean) ** 2 * m).sum() / m.sum().clamp_min(1.0)
    adv_std  = adv_var.clamp_min(1e-8).sqrt()
    advantages = (advantages - adv_mean) / adv_std

    # ---- PPO objective (masked) ----
    dist = torch.distributions.Categorical(logits=logits_at)
    new_logp = dist.log_prob(actions).to(torch.float32)  # [B,T]
    entropy  = dist.entropy().to(torch.float32)          # [B,T]

    def masked_mean(x: torch.Tensor) -> torch.Tensor:
        w = our_mask.to(x.dtype)
        return (x * w).sum() / w.sum().clamp_min(1.0)

    log_ratio = (new_logp - old_logp).clamp(min=-60.0, max=60.0)
    ratio = log_ratio.exp()

    clipped_std = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP)
    clipped_neg = torch.clamp(ratio, 1.0 - EPS_CLIP, TRINAL_DELTA1)
    r_clipped = torch.where(advantages < 0, clipped_neg, clipped_std)
    surr1 = ratio * advantages
    surr2 = r_clipped * advantages
    policy_loss = -masked_mean(torch.min(surr1, surr2))
    with torch.no_grad():
        neg_mask = (advantages < 0) & our_mask
        trinal_clip_neg_frac = ((ratio > (1.0 + EPS_CLIP)) & neg_mask).float()
        trinal_clip_neg_frac = trinal_clip_neg_frac.sum() / neg_mask.float().sum().clamp_min(1.0)

    ent_mean = masked_mean(entropy)
    entropy_loss = -ent_mean * ENT_COEF
    approx_kl = masked_mean(old_logp - new_logp)
    clipfrac  = masked_mean(((ratio - 1.0).abs() > EPS_CLIP).float())

    # ---- Value loss ----
    value_loss, vclip_frac = _value_loss_with_stakes_clip_public(
        v_pred=values_at[our_mask],
        returns=returns[our_mask],
        action_ids=actions[our_mask],
        penalties_used=batch["penalties_used"][our_mask].long(),
    )
    
    total = policy_loss + VALUE_WEIGHT * value_loss + entropy_loss

    metrics: Dict[str, torch.Tensor] = {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": ent_mean.detach(),
        "approx_kl": approx_kl.detach(),
        "clip_fraction": clipfrac.detach(),
        "trinal_clip_neg_frac": trinal_clip_neg_frac.detach(),
        "value_clip_frac": vclip_frac.detach(),
    }

    # ---- Teacher KL (optional) ----
    if (BC_KL_WEIGHT > 0.0) and (sl_teacher is not None):
        with torch.no_grad():
            t_outs = sl_teacher(**mi)
            t_logits = t_outs[0]  # [B, L, A]
            t_logits_at = t_logits.gather(1, our_idx.unsqueeze(-1).expand(-1, -1, A))
            if batch.get("our_action_mask", None) is not None:
                step_mask = batch["our_action_mask"].gather(1, our_idx.unsqueeze(-1).expand(-1, -1, A))
                t_logits_at = t_logits_at.masked_fill(~step_mask, _neg_inf_like(t_logits_at))
            t_logits_at = torch.nan_to_num(t_logits_at, nan=0.0, posinf=0.0, neginf=float(torch.finfo(t_logits_at.dtype).min))
        dist_sl = torch.distributions.Categorical(logits=t_logits_at)
        bc_kl = torch.distributions.kl_divergence(dist, dist_sl)  # [B,T]
        bc_kl = masked_mean(bc_kl)
        total = total + BC_KL_WEIGHT * bc_kl
        metrics["bc_kl"] = bc_kl.detach()
    else:
        metrics["bc_kl"] = torch.zeros((), device=logits_at.device)

    # ---- Aux: belief heads (batched, masked; -100 ignored) ----
    def _belief_aux(b_logits: Optional[torch.Tensor],
                    tgt: Optional[torch.Tensor],
                    msk: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if b_logits is None or tgt is None or msk is None:
            z = torch.zeros((), device=values_full.device)
            return z, z
        C = b_logits.size(-1)
        b_sel = b_logits.gather(1, our_idx.unsqueeze(-1).expand(-1, -1, C))  # [B,T,C]
        ce = torch.nn.functional.cross_entropy(
            b_sel.flatten(0,1), tgt.view(-1),
            ignore_index=-100, reduction="none"
        ).view(B, T)
        valid = (tgt != -100) & msk & our_mask
        w = valid.to(ce.dtype)
        loss = (ce * w).sum() / w.sum().clamp_min(1.0)
        with torch.no_grad():
            pred = b_sel.argmax(dim=-1)
            acc = ( ((pred == tgt) & valid).sum().to(torch.float32) /
                    valid.sum().clamp_min(1) )
        return loss, acc

    b0_loss, acc0 = _belief_aux(b0, batch.get("belief_tgt0"), batch.get("belief0_mask"))
    b1_loss, acc1 = _belief_aux(b1, batch.get("belief_tgt1"), batch.get("belief1_mask"))
    b2_loss, acc2 = _belief_aux(b2, batch.get("belief_tgt2"), batch.get("belief2_mask"))
    belief_loss = b0_loss + b1_loss + b2_loss
    if AUX_BELIEF_WEIGHT > 0.0:
        total = total + AUX_BELIEF_WEIGHT * belief_loss
    metrics["belief_loss"] = belief_loss.detach()
    metrics["belief_acc_0"] = acc0.detach()
    metrics["belief_acc_1"] = acc1.detach()
    metrics["belief_acc_2"] = acc2.detach()

    # ---- Aux: opponent action supervision (NO masking; -100 ignored) ----
    opp_loss = torch.zeros((), device=values_full.device)
    opp_acc  = torch.zeros((), device=values_full.device)
    if AUX_OPP_WEIGHT > 0.0 and (opp_logits is not None) and ("opp_idx" in batch):
        if batch["opp_idx"].numel() > 0:
            To = batch["opp_idx"].size(1)
            A_opp = opp_logits.size(-1)
            opp_sel = opp_logits.gather(
                1, batch["opp_idx"].unsqueeze(-1).expand(-1, -1, A_opp)
            )  # [B,To,A]
            ce_opp = torch.nn.functional.cross_entropy(
                opp_sel.flatten(0,1), batch["opp_targets"].view(-1),
                ignore_index=-100, reduction="none"
            ).view(B, To)
            w = batch["opp_have_label"].to(ce_opp.dtype)
            if w.sum() > 0:
                opp_loss = (ce_opp * w).sum() / w.sum().clamp_min(1.0)
                with torch.no_grad():
                    pred = opp_sel.argmax(dim=-1)
                    corr = ((pred == batch["opp_targets"]) & batch["opp_have_label"]).sum().to(torch.float32)
                    opp_acc = corr / batch["opp_have_label"].sum().clamp_min(1)
    if AUX_OPP_WEIGHT > 0.0:
        total = total + AUX_OPP_WEIGHT * opp_loss
    metrics["opp_loss"] = opp_loss.detach()
    metrics["opp_action_acc"] = opp_acc.detach()

    return total, metrics


def _collate_batch(
    episodes: List[Dict[str, Any]],
    L_max: Optional[int] = None,
    pin_memory: bool = False,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """
    CPU-side collation. Returns tensors on CPU so _to_device_batch(...) moves them.

    Outputs:
      mi: dict with only time-major tensors (dim>=2) padded to L_pad, plus 'valid_lengths' [B]
      our_idx [B,T], mask [B,T], actions [B,T], old_logp [B,T], rewards [B,T],
      penalties_used [B,T], our_action_mask [B,L_pad,A] or None,
      belief_tgt{0,1,2} [B,T], belief{0,1,2}_mask [B,T],
      opp_idx [B,To], opp_targets [B,To], opp_have_label [B,To]
      padding_mask [B,L_pad] (True where padded)
    """
    IGN = int(ignore_index)
    B = len(episodes)
    if B == 0:
        raise ValueError("Empty batch.")

    # -------- discover per-episode true sequence lengths --------
    raw_lens: List[int] = []
    for ep in episodes:
        mi = ep["model_input"]
        # prefer the length saved during acting (correct per-episode length)
        if "valid_lengths" in mi and torch.is_tensor(mi["valid_lengths"]):
            # acting stored [B] but here B==1 per-episode snapshot
            L_true = int(mi["valid_lengths"].view(-1)[0].item())
            raw_lens.append(L_true)
        else:
            # fallback: infer from the longest [1, L, ...] tensor
            L_found = None
            for v in mi.values():
                if torch.is_tensor(v) and v.dim() >= 2 and v.size(0) == 1:
                    L_found = int(v.size(1)); break
            if L_found is None:
                raise ValueError("Cannot infer sequence length for an episode.")
            raw_lens.append(L_found)

    # Choose padding length
    L_batch_max = max(raw_lens) if raw_lens else 0
    L_pad = int(L_max) if (L_max is not None) else L_batch_max
    if L_pad <= 0:
        L_pad = L_batch_max

    # -------- helper: pad/trim only tensors with a time dimension (dim >= 2) --------
    def _pad_trim(v: torch.Tensor, L_tgt: int) -> torch.Tensor:
        L = v.size(1)
        if L == L_tgt: return v
        if L > L_tgt:  return v[:, -L_tgt:, ...]
        pad_len = L_tgt - L
        pad_shape = list(v.shape); pad_shape[1] = pad_len
        z = torch.zeros(pad_shape, dtype=v.dtype, device=v.device)
        return torch.cat([v, z], dim=1)

    # -------- build batched model inputs (time-major tensors only) --------
    all_keys = set()
    for ep in episodes:
        all_keys |= set(ep["model_input"].keys())

    # we will REBUILD both 'valid_lengths' and 'padding_mask' — exclude the cached mask
    skip_keys = {"padding_mask", "valid_lengths"}
    mi_batch: Dict[str, torch.Tensor] = {}
    for k in sorted(all_keys - skip_keys):
        proto = next((ep["model_input"][k] for ep in episodes
                      if k in ep["model_input"]
                      and torch.is_tensor(ep["model_input"][k])
                      and ep["model_input"][k].dim() >= 2), None)
        if proto is None:
            continue

        vs = []
        for b, ep in enumerate(episodes):
            if k in ep["model_input"]:
                v = ep["model_input"][k]
            else:
                Lb = raw_lens[b]
                shape = list(proto.shape)
                shape[0] = 1
                shape[1] = Lb
                v = proto.new_zeros(shape)
            vs.append(v)

        padded = [_pad_trim(v, L_pad) for v in vs]   # each [1, L_pad, ...]
        cat = torch.cat(padded, dim=0).contiguous()  # [B, L_pad, ...]
        if pin_memory: cat = cat.pin_memory()
        mi_batch[k] = cat

    # ---- REBUILD valid_lengths and padding_mask from the true lengths ----
    valid_lengths = torch.tensor([min(l, L_pad) for l in raw_lens], dtype=torch.long)
    if pin_memory: valid_lengths = valid_lengths.pin_memory()
    mi_batch["valid_lengths"] = valid_lengths  # [B]

    padding_mask = torch.zeros((B, L_pad), dtype=torch.bool)
    for b, Lb in enumerate(valid_lengths.tolist()):
        if Lb < L_pad:
            padding_mask[b, Lb:] = True          # True = PAD
    if pin_memory: padding_mask = padding_mask.pin_memory()
    mi_batch["padding_mask"] = padding_mask     # [B, L_pad]

    # Require agent_types for actor/opp selection
    if "agent_types" not in mi_batch:
        raise ValueError("model_input must include 'agent_types' with dim>=2 (batched [B, L]).")
    agent_types = mi_batch["agent_types"].long()  # [B, L_pad]

    # Optional legality mask for OUR steps; we'll zero it past each valid length
    our_action_mask = None
    if "action_masks" in mi_batch:
        m = mi_batch["action_masks"].bool()  # [B, L_pad, A]
        # trim mask beyond valid lengths so padding is never considered legal
        for b in range(B):
            Lb = int(valid_lengths[b].item())
            if Lb < m.size(1):
                m[b, Lb:, :].fill_(False)
        our_action_mask = m

    # -------- build OUR/OPP timestep indices using ONLY valid tokens --------
    #
    # `agent_types` is expected to mark the training agent with 0 and opponents
    # with non-zero values.  This mapping should align with `ep["agent_id"]`
    # which records the acting seat at each step.  If these sources disagree
    # the downstream indexing will be incorrect.
    our_pos_lists: List[torch.Tensor] = []
    opp_pos_lists: List[torch.Tensor] = []
    for b in range(B):
        Lb = int(valid_lengths[b].item())
        slice_end = max(Lb - 1, 0)
        at = agent_types[b, :slice_end].detach().cpu().numpy()
        our_pos = torch.from_numpy((at == 0).nonzero()[0]).long()
        opp_pos = torch.from_numpy((at != 0).nonzero()[0]).long()

        # Sanity check: align with the same trimmed window from episode metadata
        ep = episodes[b]
        seat = ep.get("training_agent_seat")
        if seat is None:
            raise ValueError("Episode is missing 'training_agent_seat'.")
        # Compute how many steps were trimmed from the left by _pad_trim
        raw_Lb = raw_lens[b]
        offset_b = max(raw_Lb - Lb, 0)
        agent_ids_full = ep.get("agent_id", [])
        agent_ids_sliced = agent_ids_full[offset_b: offset_b + slice_end]
        expected_count = sum(1 for sid in agent_ids_sliced if sid == seat)
        if expected_count != int(our_pos.numel()):
            raise ValueError(
                "agent_types mismatch with agent_id: got "
                f"{int(our_pos.numel())} our steps but found "
                f"{expected_count} occurrences of training_agent_seat {seat}"
            )

        our_pos_lists.append(our_pos)
        opp_pos_lists.append(opp_pos)

    T  = max((int(x.numel()) for x in our_pos_lists), default=0)
    To = max((int(x.numel()) for x in opp_pos_lists), default=0)

    # -------- allocate supervision tensors (CPU) --------
    def _pm(x: torch.Tensor) -> torch.Tensor:
        return x.pin_memory() if pin_memory else x

    our_idx    = _pm(torch.zeros((B, T),  dtype=torch.long))
    our_mask   = _pm(torch.zeros((B, T),  dtype=torch.bool))
    actions    = _pm(torch.zeros((B, T),  dtype=torch.long))
    old_logp   = _pm(torch.zeros((B, T),  dtype=torch.float32))
    rewards    = _pm(torch.zeros((B, T),  dtype=torch.float32))
    pen_used   = _pm(torch.zeros((B, T),  dtype=torch.long))

    belief_tgt0 = _pm(torch.full((B, T), IGN, dtype=torch.long))
    belief_tgt1 = _pm(torch.full((B, T), IGN, dtype=torch.long))
    belief_tgt2 = _pm(torch.full((B, T), IGN, dtype=torch.long))
    belief0_mask = _pm(torch.zeros((B, T), dtype=torch.bool))
    belief1_mask = _pm(torch.zeros((B, T), dtype=torch.bool))
    belief2_mask = _pm(torch.zeros((B, T), dtype=torch.bool))

    opp_idx        = _pm(torch.zeros((B, To),  dtype=torch.long))
    opp_targets    = _pm(torch.full((B, To), IGN, dtype=torch.long))
    opp_have_label = _pm(torch.zeros((B, To),  dtype=torch.bool))

    # -------- fill from episodes (only real steps) --------
    for b, ep in enumerate(episodes):
        Lb = int(valid_lengths[b].item())
        slice_end = max(Lb - 1, 0)
        offset = max(raw_lens[b] - Lb, 0)

        # OUR timeline
        our_pos = our_pos_lists[b]
        # Build absolute episode indices for our steps within the trimmed window
        seat = ep["training_agent_seat"]
        agent_ids_full = ep.get("agent_id", [])
        agent_ids_sliced = agent_ids_full[offset: offset + slice_end]
        our_ep_idx = [offset + t for t, sid in enumerate(agent_ids_sliced) if sid == seat]
        K_true = len(our_ep_idx)
        K_pos = int(our_pos.numel())
        K_fill = min(T, K_true, K_pos)
        if K_fill > 0:
            our_idx[b, :K_fill] = our_pos[:K_fill]
            our_mask[b, :K_fill] = True
            for t_local in range(K_fill):
                step_ep = our_ep_idx[t_local]

                a  = ep["our_action"][step_ep] if step_ep < len(ep["our_action"]) else None
                lp = ep["log_prob"][step_ep]   if step_ep < len(ep["log_prob"])   else None
                rw = ep["reward"][step_ep]     if step_ep < len(ep["reward"])     else 0.0
                pu = ep["penalties_used"][step_ep] if step_ep < len(ep["penalties_used"]) else 0

                if a is not None:  actions[b, t_local] = int(a)
                if lp is not None: old_logp[b, t_local] = float(lp)
                rewards[b, t_local]  = float(rw)
                pen_used[b, t_local] = int(pu)

                lb0 = ep.get("belief_tgt0", [None]*len(ep["agent_id"]))[step_ep]
                lb1 = ep.get("belief_tgt1", [None]*len(ep["agent_id"]))[step_ep]
                lb2 = ep.get("belief_tgt2", [None]*len(ep["agent_id"]))[step_ep]
                if lb0 is not None: belief_tgt0[b, t_local] = int(lb0); belief0_mask[b, t_local] = True
                if lb1 is not None: belief_tgt1[b, t_local] = int(lb1); belief1_mask[b, t_local] = True
                if lb2 is not None: belief_tgt2[b, t_local] = int(lb2); belief2_mask[b, t_local] = True

        # OPP timeline (labels optional)
        opp_pos = opp_pos_lists[b]
        # Build absolute episode indices for opponent steps within the trimmed window
        opp_ep_idx = [offset + t for t, sid in enumerate(agent_ids_sliced) if sid != seat]
        M = int(opp_pos.numel())
        M_true = len(opp_ep_idx)
        M_fill = min(To, M_true, M)
        if M_fill > 0:
            opp_idx[b, :M_fill] = opp_pos[:M_fill]
            for t_local in range(M_fill):
                step_ep = opp_ep_idx[t_local]
                tgt = ep.get("opp_target_action", [None]*len(ep["agent_id"]))[step_ep]
                if tgt is not None:
                    opp_targets[b, t_local] = int(tgt)
                    opp_have_label[b, t_local] = True

    return {
        "mi": mi_batch,
        "our_idx": our_idx,
        "mask": our_mask,
        "actions": actions,
        "old_logp": old_logp,
        "rewards": rewards,
        "penalties_used": pen_used,
        "our_action_mask": our_action_mask,

        "belief_tgt0": belief_tgt0, "belief_tgt1": belief_tgt1, "belief_tgt2": belief_tgt2,
        "belief0_mask": belief0_mask, "belief1_mask": belief1_mask, "belief2_mask": belief2_mask,

        "opp_idx": opp_idx, "opp_targets": opp_targets, "opp_have_label": opp_have_label,
    }

def _to_device_batch(batch_cpu: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move a collated CPU batch (with nested 'mi' dict) to device."""
    mi_dev = {k: v.to(device, non_blocking=True) for k, v in batch_cpu["mi"].items()}
    oam = batch_cpu.get("our_action_mask", None)
    oam_dev = oam.to(device, non_blocking=True) if (oam is not None) else None
    out = {
        "mi": mi_dev,
        "our_idx":        batch_cpu["our_idx"].to(device, non_blocking=True),
        "mask":           batch_cpu["mask"].to(device, non_blocking=True),
        "actions":        batch_cpu["actions"].to(device, non_blocking=True),
        "old_logp":       batch_cpu["old_logp"].to(device, non_blocking=True),
        "rewards":        batch_cpu["rewards"].to(device, non_blocking=True),
        "penalties_used": batch_cpu["penalties_used"].to(device, non_blocking=True),
        "our_action_mask": oam_dev,
        "belief_tgt0":    batch_cpu["belief_tgt0"].to(device, non_blocking=True),
        "belief_tgt1":    batch_cpu["belief_tgt1"].to(device, non_blocking=True),
        "belief_tgt2":    batch_cpu["belief_tgt2"].to(device, non_blocking=True),
        "belief0_mask":   batch_cpu["belief0_mask"].to(device, non_blocking=True),
        "belief1_mask":   batch_cpu["belief1_mask"].to(device, non_blocking=True),
        "belief2_mask":   batch_cpu["belief2_mask"].to(device, non_blocking=True),
        "opp_idx":        batch_cpu["opp_idx"].to(device, non_blocking=True),
        "opp_targets":    batch_cpu["opp_targets"].to(device, non_blocking=True),
        "opp_have_label": batch_cpu["opp_have_label"].to(device, non_blocking=True),
    }
    return out
