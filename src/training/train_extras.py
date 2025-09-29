# src/training/train_extras.py

import random
from collections import Counter
from contextlib import contextmanager
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple, Iterable, Union, Hashable, Callable, Set
import math
import numpy as np
import torch
from src import config
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
import torch.nn.functional as F
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from PIL import Image
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
import umap.umap_ as umap


_EMBED_LABEL_MAP: Dict[Hashable, int] = {}


_PERSISTENT_PCA: Dict[str, Dict[str, Any]] = {}


# Expected model_input keys collated during batching. Keeping this tuple at module
# scope avoids repeatedly constructing temporary containers in the hot _collate_batch
# path during PPO training.
_COLLATE_EXPECTED_MI_KEYS: Tuple[str, ...] = (
    "action_masks",
    "action_sequence",
    "agent_types",
    "obs_sequence",
    "positions",
)


def _use_heldout_agents() -> bool:
    return bool(getattr(config, "USE_HELDOUT_AGENT", True))


def _remap_embed_labels(labels: List[Any]) -> List[int]:
    """Map arbitrary opponent identifiers to stable sequential integers."""
    remapped: List[int] = []
    for lab in labels:
        key: Hashable
        if isinstance(lab, np.generic):
            key = lab.item()
        elif isinstance(lab, torch.Tensor):
            key = lab.item()
        elif isinstance(lab, (list, tuple)):
            key = tuple(lab)
        else:
            key = lab  # assume hashable
        if key not in _EMBED_LABEL_MAP:
            _EMBED_LABEL_MAP[key] = len(_EMBED_LABEL_MAP)
        remapped.append(int(_EMBED_LABEL_MAP[key]))
    return remapped


def _persistent_pca_project(
    X: np.ndarray,
    key: str,
    step: int,
    *,
    allow_fit: bool = True,
    n_components: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """Project ``X`` using a PCA that is fitted at most once per ``key``."""
    if not isinstance(X, np.ndarray):
        X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.size == 0:
        return None

    max_comp = int(min(X.shape[0], X.shape[1]))
    if max_comp <= 0:
        return None

    entry = _PERSISTENT_PCA.get(key)
    if entry is None:
        if not allow_fit:
            return None
        if n_components is None:
            desired = min(4, max_comp)
            n_comp = max(1, desired)
        else:
            n_comp = max(1, min(int(n_components), max_comp))
        pca = PCA(n_components=n_comp, random_state=0)
        Xp = pca.fit_transform(X)
        entry = {
            "model": pca,
            "n_components": int(pca.n_components_),
            "fit_step": int(step),
            "fit_dim": int(X.shape[1]),
            "explained_variance_ratio": pca.explained_variance_ratio_.copy(),
        }
        _PERSISTENT_PCA[key] = entry
    else:
        pca = entry.get("model")
        if pca is None:
            if not allow_fit:
                return None
            return _persistent_pca_project(X, key, step, allow_fit=True, n_components=n_components)
        try:
            Xp = pca.transform(X)
        except Exception as exc:  # pragma: no cover - defensive fallback
            if not allow_fit:
                print(f"[viz][pca] transform failed for key '{key}': {exc}")
                return None
            max_comp = int(min(X.shape[0], X.shape[1]))
            if n_components is None:
                n_comp = max(1, min(entry.get("n_components", 2), max_comp))
            else:
                n_comp = max(1, min(int(n_components), max_comp))
            pca = PCA(n_components=n_comp, random_state=0)
            Xp = pca.fit_transform(X)
            entry = {
                "model": pca,
                "n_components": int(pca.n_components_),
                "fit_step": int(step),
                "fit_dim": int(X.shape[1]),
                "explained_variance_ratio": pca.explained_variance_ratio_.copy(),
                "refit_reason": str(exc),
                "refit_step": int(step),
            }
            _PERSISTENT_PCA[key] = entry

    entry["last_step"] = int(step)
    return Xp, entry

def set_seed(seed=42):
    """
    Sets the seed for reproducibility.
    """
    # Python / NumPy RNGs
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    # Torch RNGs (CPU & CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Disable non-deterministic kernel selection / precision trade-offs
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    torch.set_float32_matmul_precision("medium")

    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    # Enforce deterministic algorithm usage (raises if unavailable)
    torch.use_deterministic_algorithms(True, warn_only=True)

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

def _coerce_opponent_input(
    data: Union[
        Dict[Any, np.ndarray],                           # {opponent_id: embedding_vec}
        Tuple[Iterable[Iterable[float]], Iterable[Any]]  # (X, opponent_labels)
    ]
) -> Tuple[np.ndarray, list]:
    """Return (X [N,D], opp_labels [N])."""
    if isinstance(data, dict):
        labels, rows = [], []
        for opp, emb in data.items():
            labels.append(opp)
            rows.append(np.asarray(emb, dtype=np.float32))
        if not rows:
            return np.empty((0, 2), dtype=np.float32), []
        return np.stack(rows, axis=0), labels
    if isinstance(data, tuple) and len(data) == 2:
        X, labels = data
        return np.asarray(X, dtype=np.float32), list(labels)
    raise TypeError("Pass a dict {opponent_id: embedding} OR a tuple (X, opponent_labels).")


def _balanced_subsample_indices(
    labels: np.ndarray,
    per_label_cap: int,
    max_points: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """Return balanced subsample indices for labels."""
    if labels.size == 0:
        return np.empty((0,), dtype=np.int64)

    rng = np.random.default_rng(seed)
    take = []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if per_label_cap and idx.size > per_label_cap:
            idx = rng.choice(idx, size=per_label_cap, replace=False)
        take.append(idx)
    idxs = np.concatenate(take) if take else np.empty((0,), dtype=np.int64)
    if max_points and idxs.size > max_points:
        idxs = rng.choice(idxs, size=max_points, replace=False)
    return np.sort(idxs)


def _ellipse_from_cov(
    center: np.ndarray,
    cov: np.ndarray,
    color,
    alpha: float = 0.2,
) -> Optional[Ellipse]:
    if not np.isfinite(cov).all():
        return None
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    vals = np.maximum(vals, 1e-9)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * np.sqrt(vals)
    ell = Ellipse(xy=center, width=width, height=height, angle=theta,
                  edgecolor=color, facecolor=color, alpha=alpha, linewidth=2.0)
    return ell


def save_interactive_3d(
    X: np.ndarray,
    labels: np.ndarray,
    path_html: str,
    method: str = "pca",
    per_label_cap: int = 600,
    title_prefix: Optional[str] = None,
) -> Optional[str]:
    """Save an interactive 3D PCA HTML scatter plot of embeddings."""
    X = np.asarray(X)
    labels = np.asarray([str(l) for l in labels])
    if X.ndim != 2 or X.shape[0] == 0:
        return None

    idxs = _balanced_subsample_indices(labels, per_label_cap=per_label_cap)
    if idxs.size == 0:
        return None
    Xs = X[idxs]
    Ls = labels[idxs]

    pca = PCA(n_components=min(3, Xs.shape[1]), random_state=0)
    X3 = pca.fit_transform(Xs)

    fig = go.Figure()
    uniq = sorted(set(Ls.tolist()), key=lambda x: str(x))
    for lab in uniq:
        mask = Ls == lab
        fig.add_trace(go.Scatter3d(
            x=X3[mask, 0], y=X3[mask, 1], z=X3[mask, 2],
            mode="markers",
            marker=dict(size=3, opacity=0.8),
            name=f"Opp {lab}",
            hovertext=[f"Opp {lab}"] * int(mask.sum()),
            hoverinfo="text",
        ))

    evr = pca.explained_variance_ratio_
    ttl = (
        f"3D PCA (EVR: {evr[0]:.2f}, {evr[1]:.2f}, {evr[2]:.2f})"
        if evr.size >= 3 else f"3D PCA — N={Xs.shape[0]}"
    )
    full_title = (f"{title_prefix} — " if title_prefix else "") + ttl + f" — N={Xs.shape[0]}"
    fig.update_layout(
        title=full_title,
        showlegend=True,
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(path_html, include_plotlyjs="cdn")
    return path_html


def embedding_quality_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
    max_points: int = 20000,
    per_label_cap: int = 2000,
) -> dict:
    """Compute label-aware quality metrics in the original embedding space."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray([str(l) for l in labels])
    if X.ndim != 2 or X.shape[0] == 0:
        return {}

    sel = _balanced_subsample_indices(y, per_label_cap=per_label_cap, max_points=max_points)
    if sel.size == 0:
        return {}
    Xs, ys = X[sel], y[sel]

    norms = np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-9
    Xn = Xs / norms

    nn = NearestNeighbors(n_neighbors=min(k + 1, Xn.shape[0]), metric="cosine", algorithm="brute")
    nn.fit(Xn)
    _, idxs = nn.kneighbors(Xn, return_distance=True)
    idxs = idxs[:, 1:]
    nbr_labels = ys[idxs]
    preds = []
    for row in nbr_labels:
        most_common = Counter(row).most_common(1)[0][0]
        preds.append(most_common)
    knn_acc = float(np.mean(np.asarray(preds, dtype=ys.dtype) == ys))

    s_sel = min(5000, Xn.shape[0])
    rng = np.random.default_rng(0)
    pick = rng.choice(np.arange(Xn.shape[0]), size=s_sel, replace=False)
    sil = float(silhouette_score(Xn[pick], ys[pick], metric="cosine")) if s_sel >= 30 else float("nan")

    ch = float(calinski_harabasz_score(Xn, ys))
    db = float(davies_bouldin_score(Xn, ys))

    labs = np.unique(ys)
    cents = []
    intra = []
    for lab in labs:
        Xi = Xn[ys == lab]
        if Xi.size == 0:
            continue
        c = Xi.mean(axis=0)
        norm = np.linalg.norm(c)
        if norm > 0:
            c = c / norm
        cents.append(c)
        intra.append(1.0 - float((Xi @ c).mean()))
    C = np.stack(cents) if cents else np.zeros((1, Xn.shape[1]), np.float32)
    if C.shape[0] > 1:
        sim = C @ C.T
        iu = np.triu_indices(C.shape[0], k=1)
        between = float(np.mean(1.0 - sim[iu]))
    else:
        between = float("nan")
    within = float(np.mean(intra)) if intra else float("nan")
    bw_ratio = (between / (within + 1e-9)) if np.isfinite(between) and np.isfinite(within) else float("nan")

    pca = PCA(n_components=min(16, Xn.shape[1]), random_state=0).fit(Xn)
    evr = pca.explained_variance_ratio_
    cum_evr_4 = float(evr[:4].sum()) if evr.size >= 4 else float(evr.sum())
    cum_evr_8 = float(evr[:8].sum()) if evr.size >= 8 else float(evr.sum())
    cum_evr_16 = float(evr[:16].sum()) if evr.size >= 16 else float(evr.sum())

    return {
        f"knn_acc@{k}": knn_acc,
        "silhouette_cosine": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "centroid_within_cos": within,
        "centroid_between_cos": between,
        "between_over_within": bw_ratio,
        "pca_evr": evr,
        "pca_cum_evr_4": cum_evr_4,
        "pca_cum_evr_8": cum_evr_8,
        "pca_cum_evr_16": cum_evr_16,
        "N_eval": int(Xn.shape[0]),
        "n_labels": int(labs.size),
    }

def _visualize_pca_panels(
    writer,
    X: np.ndarray,
    opp_labels: list,
    step: int,
    title_prefix: str,
    *,
    pca_key: Optional[str] = None,
    allow_fit: bool = True,
    tag_name: str = "Embeddings/PCA multi-view",
    legend_label_fn: Optional[Callable[[str], str]] = None,
    legend_title: str = "Opponents",
    show_legend: bool = True,
    scatter_alpha: float = 0.25,
    scatter_cap: int = 400,
    point_size: float = 8.0,
    cmap_name: str = "tab20",
    equal_aspect: bool = True,
    auto_hide_legend_over: int = 24,
) -> dict:
    label_list = [str(l) for l in opp_labels]
    labels_arr = np.asarray(label_list)
    N, D = X.shape

    fit_step = step
    if pca_key:
        proj = _persistent_pca_project(X, pca_key, step, allow_fit=allow_fit)
        if proj is None:
            return {}
        Xp, entry = proj
        evr = np.asarray(entry.get("explained_variance_ratio", ()), dtype=np.float32)
        n_comp = int(entry.get("n_components", Xp.shape[1]))
        fit_step = int(entry.get("fit_step", step))
    else:
        n_comp = int(max(1, min(4, D, N)))
        pca = PCA(n_components=n_comp, random_state=0)
        Xp = pca.fit_transform(X)
        evr = pca.explained_variance_ratio_

    if Xp.ndim != 2 or Xp.shape[0] == 0:
        return {"pca_evr": evr, "pca_fit_step": fit_step}

    n_comp = int(min(n_comp, Xp.shape[1]))
    pairs = [(0, 1), (1, 2), (2, 3)]
    valid_pairs = [pair for pair in pairs if pair[1] < n_comp]
    if not valid_pairs:
        return {"pca_evr": evr, "pca_fit_step": fit_step}

    fig, axes = plt.subplots(1, len(valid_pairs), figsize=(6 * len(valid_pairs), 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    uniq = sorted(set(label_list))
    if legend_label_fn is None:
        legend_label_fn = lambda lab: lab  # type: ignore[return-value]
    cmap = plt.cm.get_cmap(cmap_name, max(1, len(uniq)))
    scatter_cap = max(1, int(scatter_cap))
    rng = np.random.default_rng(0)
    legend_handles: List[Line2D] = []
    # Auto-hide legend when there are too many classes to keep layout readable
    if show_legend and len(uniq) > int(auto_hide_legend_over):
        show_legend = False

    legend_added = False
    for ax, (i, j) in zip(axes, valid_pairs):
        ax.set_title(f"PC{i+1} vs PC{j+1}")
        xi = evr[i] if i < len(evr) else float("nan")
        xlbl = f"PC{i+1} ({xi*100:.1f}% var)" if np.isfinite(xi) else f"PC{i+1}"
        yj = evr[j] if j < len(evr) else float("nan")
        ylbl = f"PC{j+1} ({yj*100:.1f}% var)" if np.isfinite(yj) else f"PC{j+1}"
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        ax.grid(True, linestyle="--", alpha=0.25)

        for idx, lab in enumerate(uniq):
            mask = labels_arr == lab
            if not np.any(mask):
                continue
            pts = Xp[mask][:, [i, j]]
            if pts.shape[0] == 0:
                continue
            color = cmap(idx % cmap.N)

            show_pts = pts
            if pts.shape[0] > scatter_cap:
                take = rng.choice(pts.shape[0], size=scatter_cap, replace=False)
                show_pts = pts[take]
            ax.scatter(
                show_pts[:, 0],
                show_pts[:, 1],
                color=color,
                s=point_size,
                alpha=scatter_alpha,
                linewidths=0,
            )

            if show_legend and i == valid_pairs[0][0] and j == valid_pairs[0][1]:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=color,
                        markeredgecolor="none",
                        linewidth=0,
                        markersize=max(point_size / 1.5, 4.0),
                        label=legend_label_fn(lab),
                    )
                )

        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")

    if show_legend and legend_handles:
        axes[0].legend(
            handles=legend_handles,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            title=legend_title,
        )
        legend_added = True

    evr_pad = np.pad(evr, (0, max(0, 4 - evr.size)), constant_values=np.nan)
    evr_text = ", ".join(
        [
            f"PC{k+1}: {val:.2f}" if np.isfinite(val) else f"PC{k+1}: n/a"
            for k, val in enumerate(evr_pad[:4])
        ]
    )
    fig.suptitle(
        f"{title_prefix} — PCA multi-view — step {step}\n{evr_text} (fit @ step {fit_step})"
    )

    if evr.size >= 3 and np.isfinite(evr[:3]).all():
        pc3_ratio = float(evr[2] / max(evr[0] + evr[1], 1e-9))
        fig.text(0.5, 0.02, f"PC3 vs PC1-2 EVR ratio: {pc3_ratio:.3f}", ha="center")
    else:
        pc3_ratio = float("nan")

    # Layout: if there is no legend, expand to full width. Otherwise leave room
    # on the right for the legend box outside axes[0].
    if legend_added:
        fig.tight_layout(rect=[0, 0.05, 0.88, 0.92])
    else:
        fig.tight_layout(rect=[0, 0.05, 1.0, 0.92])

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    image = Image.open(buf)
    writer.add_image(tag_name, np.array(image), step, dataformats="HWC")
    plt.close(fig)

    return {
        "pca_evr": evr,
        "pc3_vs_pc12": pc3_ratio,
        "n_components": n_comp,
        "pairs": valid_pairs,
        "pca_fit_step": fit_step,
        "pca_key": pca_key,
    }

def visualize_opponent_embeddings_all(
    writer,
    data,                      # dict {opp: emb} OR (X, opp_labels)
    step: int,
    methods: list | tuple | None = None,
    pca_dim: int = 50,
    title_prefix: str = "Opponent Embeddings",
    pca_key: Optional[str] = None,
):
    """
    Call visualize_opponent_embeddings() for a list of methods.
    Default set is chosen based on whether UMAP is available.
    """
    if methods is None:
        methods = ["pca_panels"]

    results = {}
    for m in methods:
        try:
            out = visualize_opponent_embeddings(
                writer,
                data=data,
                step=step,
                method=m,
                pca_dim=pca_dim,
                title_prefix=title_prefix,
                pca_key=pca_key,
            )
            if out is not None:
                results[m] = out
        except Exception as e:
            # keep training going even if a reducer fails
            print(f"[viz][{m}] failed: {e}")
    return results

def visualize_opponent_embeddings(
    writer,
    data,                       # dict {opp: emb} OR (X, opp_labels)
    step: int,
    method: str = "pca_panels",  # 'pca_panels' | 'pca_tsne' | 'pca_umap' | 'pca' | 'tsne' | 'umap'
    pca_dim: int = 50,
    title_prefix: str = "Opponent Embeddings",
    pca_key: Optional[str] = None,
):
    # ---- coerce input ----
    X, opp_labels = _coerce_opponent_input(data)
    if X.shape[0] < 2:
        return None  # need ≥2 points

    N, D = X.shape
    if method == "pca_panels":
        return _visualize_pca_panels(
            writer,
            X,
            opp_labels,
            step,
            title_prefix,
            pca_key=pca_key,
            legend_label_fn=lambda lab: f"Opp {lab}",
            legend_title="Opponents",
        )

    # ---- optional PCA pre-step for mixed methods ----
    use_pca_prefix = method in ("pca_tsne", "pca_umap")
    if use_pca_prefix:
        d = int(max(2, min(pca_dim, D, N - 1)))  # safe for small N
        X_low = PCA(n_components=d, random_state=0).fit_transform(X)
    else:
        X_low = X

    # ---- final 2D reducer ----
    if method in ("pca_tsne", "tsne"):
        perplexity = max(5, min(30, N - 1))
        reducer = TSNE(n_components=2, perplexity=perplexity, init="pca",
                       learning_rate="auto", random_state=0)
        X_2d = reducer.fit_transform(X_low)
        method_name = "PCA→t-SNE" if method == "pca_tsne" else "t-SNE"

    elif method in ("pca_umap", "umap"):
        reducer = umap.UMAP(n_components=2, n_neighbors=min(15, N-1), min_dist=0.1,
                            metric="cosine", random_state=0, n_jobs=1)
        X_2d = reducer.fit_transform(X_low)
        method_name = "PCA→UMAP" if method == "pca_umap" else "UMAP"

    elif method == "pca":
        X_2d = PCA(n_components=2, random_state=0).fit_transform(X)
        method_name = "PCA"

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'pca_tsne', 'pca_umap', 'pca', 'tsne', or 'umap'.")

    # ---- styling: color per opponent ----
    uniq_opps = sorted(set(opp_labels), key=lambda x: str(x))
    colors = plt.cm.rainbow(np.linspace(0, 1, max(1, len(uniq_opps))))
    color_map = {o: colors[i % len(colors)] for i, o in enumerate(uniq_opps)}

    plt.figure(figsize=(10, 8))
    for (x, y), opp in zip(X_2d, opp_labels):
        plt.scatter(x, y, color=color_map.get(opp, "black"), s=60, alpha=0.9)

    legend_handles = [plt.Line2D([0],[0], marker='o', color='w',
                          markerfacecolor=color_map[o], markersize=8,
                          label=f'Opp {o}') for o in uniq_opps]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1),
               loc='upper left', borderaxespad=0.)
    plt.subplots_adjust(right=0.78)

    plt.title(f'{title_prefix} — {method_name} — step {step}')
    plt.xlabel('Dim 1'); plt.ylabel('Dim 2')
    plt.grid(True, linestyle='--', alpha=0.35)

    # ---- to TensorBoard ----
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf)
    writer.add_image(f'Embeddings/{method_name}', np.array(image), step, dataformats='HWC')
    plt.close()


def log_strategy_pca_views(
    writer,
    *,
    step: int,
    pca_key: str,
    opponent_codes: Optional[np.ndarray] = None,
    opponent_labels: Optional[Iterable[Any]] = None,
    brick_vectors: Optional[np.ndarray] = None,
    activation_codes: Optional[np.ndarray] = None,
    activation_labels: Optional[Iterable[Any]] = None,
    title_prefix: str = "Strategy Code",
) -> Dict[str, Dict[str, Any]]:
    """Log opponent, brick, and activation projections that share one persistent PCA."""

    results: Dict[str, Dict[str, Any]] = {}

    if opponent_codes is not None and opponent_labels is not None:
        opp_labels_list = [str(l) for l in opponent_labels]
        if len(opponent_codes) == len(opp_labels_list) and opp_labels_list:
            res = _visualize_pca_panels(
                writer,
                np.asarray(opponent_codes, dtype=np.float32),
                opp_labels_list,
                step,
                title_prefix,
                pca_key=pca_key,
                allow_fit=True,
                legend_label_fn=lambda lab: f"Opp {lab}",
                legend_title="Opponents",
                scatter_alpha=0.35,
                scatter_cap=600,
            )
            results["opponents"] = res

    if brick_vectors is not None and np.asarray(brick_vectors).ndim == 2:
        if pca_key not in _PERSISTENT_PCA and opponent_codes is None:
            pass  # need a fitted PCA first
        else:
            bricks_np = np.asarray(brick_vectors, dtype=np.float32)
            labels = [str(i) for i in range(bricks_np.shape[0])]
            res = _visualize_pca_panels(
                writer,
                bricks_np,
                labels,
                step,
                f"{title_prefix} — Dictionary Bricks",
                pca_key=pca_key,
                allow_fit=False,
                legend_label_fn=lambda lab: f"Brick {lab}",
                legend_title="Bricks",
                scatter_alpha=0.9,
                scatter_cap=max(64, bricks_np.shape[0]),
                point_size=30.0,
                tag_name="Embeddings/PCA bricks",
                show_legend=False,
                equal_aspect=False,
            )
            results["bricks"] = res

    if activation_codes is not None:
        act_np = np.asarray(activation_codes, dtype=np.float32)
        if act_np.ndim == 2 and act_np.shape[0] > 0:
            if activation_labels is None:
                activation_labels = ["activation"] * act_np.shape[0]
            act_labels = [str(l) for l in activation_labels]
            res = _visualize_pca_panels(
                writer,
                act_np,
                act_labels,
                step,
                f"{title_prefix} — Activation Samples",
                pca_key=pca_key,
                allow_fit=False,
                legend_label_fn=lambda lab: f"Top {lab}",
                legend_title="Top Brick",
                show_legend=False,
                scatter_alpha=0.18,
                scatter_cap=2000,
                point_size=6.0,
                tag_name="Embeddings/PCA activation samples",
                equal_aspect=False,
            )
            results["activations"] = res

    return results

EPS_CLIP          = float(getattr(config, "EPS_CLIP", 0.2))
ENT_COEF          = float(getattr(config, "INIT_ENTROPY_COEF", 0.005))
TRINAL_DELTA1     = float(getattr(config, "TRINAL_DELTA1", 1.8))
GAMMA             = float(getattr(config, "GAMMA", 0.974))
GAE_LAMBDA        = float(getattr(config, "GAE_LAMBDA", 0.98))

# --- Loss Function Weights ---
VALUE_WEIGHT      = float(getattr(config, "VALUE_WEIGHT", 1.0))
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
def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.to(x.dtype)
    denom = w.sum().clamp_min(1.0)
    return (x * w).sum() / denom


def _normalize_advantages(advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.to(advantages.dtype)
    denom = w.sum()
    if denom.item() == 0:
        return torch.zeros_like(advantages)
    mean = (advantages * w).sum() / denom
    var = ((advantages - mean).pow(2) * w).sum() / denom
    std = torch.sqrt(var + 1e-8)
    norm = (advantages - mean) / (std + 1e-8)
    return norm * w


def _brick_decorrelation_penalty(bricks: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
    if bricks is None or bricks.ndim != 2 or bricks.size(0) == 0:
        return torch.zeros((), device=device)
    eps = 1e-6
    norm_bricks = bricks / (bricks.norm(dim=1, keepdim=True) + eps)
    gram = norm_bricks @ norm_bricks.t()
    eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
    diff = gram - eye
    return diff.pow(2).sum()


def _dictionary_regularizers(
    embedding_tuple: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    mi: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    *,
    update_num: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[np.ndarray], Dict[str, Any]]:
    sample_tensor = None
    for v in mi.values():
        if torch.is_tensor(v):
            sample_tensor = v
            break
    device = sample_tensor.device if sample_tensor is not None else torch.device("cpu")
    zero = torch.zeros((), device=device)

    l1_sparsity_loss = zero.clone()
    usage_balance_loss = zero.clone()
    brick_diversity_loss = zero.clone()
    decor_penalty = zero.clone()
    avg_brick_usage_np: Optional[np.ndarray] = None
    metrics_extra: Dict[str, Any] = {}
    selected_token_mask: Optional[torch.Tensor] = None
    opp_mask_for_tokens: Optional[torch.Tensor] = None
    should_log_embeddings = (update_num % int(getattr(config, "EMBED_LOG_INTERVAL", 50)) == 0)

    if not (isinstance(embedding_tuple, tuple) and len(embedding_tuple) == 3):
        return l1_sparsity_loss, usage_balance_loss, brick_diversity_loss, decor_penalty, avg_brick_usage_np, metrics_extra

    strategy_code, activations, bricks = embedding_tuple

    if activations is not None:
        agent_types = mi.get("agent_types")
        padding_mask = mi.get("padding_mask")
        if agent_types is not None:
            opp_mask = (agent_types != 0)
            if padding_mask is not None:
                opp_mask = opp_mask & (~padding_mask.bool())
            opp_mask_for_tokens = opp_mask
            if opp_mask.any():
                opp_activations = activations[opp_mask]
                l1_sparsity_loss = opp_activations.abs().mean()

                brick_sums = opp_activations.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                probs = torch.nan_to_num(opp_activations / brick_sums, nan=0.0, posinf=0.0, neginf=0.0)
                log_probs = torch.log(probs + 1e-8)
                num_bricks = probs.size(-1)
                if num_bricks > 0:
                    log_uniform = math.log(1.0 / num_bricks)
                    usage_balance_loss = (probs * (log_probs - log_uniform)).sum(dim=-1).mean()

                avg_brick_usage = opp_activations.mean(dim=0)
                avg_brick_usage_np = avg_brick_usage.detach().cpu().float().numpy()


    if bricks is not None and bricks.ndim == 2 and bricks.size(0) > 1:
        norm_bricks = F.normalize(bricks, dim=-1)
        sim_matrix = norm_bricks @ norm_bricks.t()
        eye = torch.eye(sim_matrix.size(0), device=sim_matrix.device, dtype=sim_matrix.dtype)
        off_diag = sim_matrix - eye
        denom = max(sim_matrix.size(0) * (sim_matrix.size(0) - 1), 1)
        brick_diversity_loss = off_diag.pow(2).sum() / denom
        decor_penalty = _brick_decorrelation_penalty(bricks, bricks.device)
    else:
        decor_penalty = torch.zeros((), device=device)

    if not should_log_embeddings:
        return (
            l1_sparsity_loss,
            usage_balance_loss,
            brick_diversity_loss,
            decor_penalty,
            avg_brick_usage_np,
            metrics_extra,
        )

    if strategy_code is not None:
        agent_types = mi.get("agent_types")
        padding_mask = mi.get("padding_mask")
        opp_labels_by_seat = batch.get("opp_labels_by_seat")
        opp_seat_ids = batch.get("opp_seat_ids")
        opp_last_idx = batch.get("opp_last_token_idx")
        opp_last_adjacent = batch.get("opp_last_token_is_adjacent")

        if (
            agent_types is not None
            and opp_labels_by_seat is not None
            and opp_seat_ids is not None
            and opp_last_idx is not None
            and opp_last_adjacent is not None
        ):
            B_emb, num_seats = opp_labels_by_seat.shape
            _, L_seq, D_emb = strategy_code.shape
            seat_valid = opp_seat_ids >= 0
            seat_labels = opp_labels_by_seat.clone().masked_fill(~seat_valid, -1)

            seat_embeds = torch.full(
                (B_emb, num_seats, D_emb),
                float("nan"),
                dtype=strategy_code.dtype,
                device=strategy_code.device,
            )
            seat_counts = torch.zeros(
                (B_emb, num_seats),
                dtype=strategy_code.dtype,
                device=strategy_code.device,
            )

            idx_shape_ok = (
                opp_last_idx.dim() == 2
                and opp_last_idx.size(0) == B_emb
                and opp_last_idx.size(1) == num_seats
            )
            adj_shape_ok = (
                opp_last_adjacent.dim() == 2
                and opp_last_adjacent.size(0) == B_emb
                and opp_last_adjacent.size(1) == num_seats
            )

            if idx_shape_ok and adj_shape_ok and L_seq > 0:
                idx_clamped = opp_last_idx.clamp(min=0, max=max(L_seq - 1, 0))
                adjacency_mask = (
                    opp_last_adjacent.bool()
                    & seat_valid
                    & (opp_last_idx >= 0)
                    & (idx_clamped < L_seq)
                )
                if adjacency_mask.any():
                    flat_idx = adjacency_mask.nonzero(as_tuple=False)
                    b_sel = flat_idx[:, 0]
                    seat_sel = flat_idx[:, 1]
                    tok_sel = idx_clamped[adjacency_mask]
                    gathered = strategy_code[b_sel, tok_sel, :]
                    seat_embeds[b_sel, seat_sel, :] = gathered
                    seat_counts[adjacency_mask] = 1.0
                    selected_token_mask = torch.zeros(
                        (B_emb, L_seq), dtype=torch.bool, device=strategy_code.device
                    )
                    selected_token_mask[b_sel, tok_sel] = True

            seat_counts = seat_counts.masked_fill(~seat_valid, 0.0)

            metrics_extra["opp_embeds_batch"] = (
                seat_embeds.detach().cpu().float().numpy(),
                seat_labels.detach().cpu().numpy(),
                seat_counts.detach().cpu().float().numpy(),
            )

            valid_flat_mask = (seat_counts > 0) & (seat_labels >= 0)
            if valid_flat_mask.any():
                embeds_tensor = seat_embeds[valid_flat_mask].detach().cpu().float()
                labels_tensor = seat_labels[valid_flat_mask].detach().cpu()
                labels_list = labels_tensor.tolist()
                metrics_extra["opp_embeds_flat"] = embeds_tensor.numpy()
                metrics_extra["opp_labels_flat"] = labels_list
                metrics_extra["opp_labels_flat_original"] = labels_list

        mask_for_tokens = selected_token_mask if selected_token_mask is not None else opp_mask_for_tokens
        if mask_for_tokens is not None:
            mask_bool = mask_for_tokens.bool()
            if mask_bool.any():
                max_tokens = int(getattr(config, "PCA_TOKEN_SAMPLE", 2048))
                with torch.no_grad():
                    strat_det = strategy_code.detach()
                    opp_codes_flat = strat_det[mask_bool]
                    if opp_codes_flat.numel() > 0:
                        opp_act_flat = None
                        if activations is not None:
                            opp_act_flat = activations.detach()[mask_bool]
                        if opp_codes_flat.size(0) > max_tokens:
                            perm = torch.randperm(opp_codes_flat.size(0), device=opp_codes_flat.device)[:max_tokens]
                            opp_codes_flat = opp_codes_flat[perm]
                            if opp_act_flat is not None:
                                opp_act_flat = opp_act_flat[perm]
                        metrics_extra["opp_strategy_codes_tokens"] = (
                            opp_codes_flat.cpu().float().numpy()
                        )
                        if opp_act_flat is not None:
                            metrics_extra["opp_activation_top_idx"] = (
                                opp_act_flat.argmax(dim=-1).cpu().numpy()
                            )

    return (
        l1_sparsity_loss,
        usage_balance_loss,
        brick_diversity_loss,
        decor_penalty,
        avg_brick_usage_np,
        metrics_extra,
    )


@contextmanager
def _temporarily_freeze_heads(model: torch.nn.Module):
    params: List[Tuple[torch.nn.Parameter, bool]] = []
    try:
        for name in ("action_head", "value_head", "opp_action_head"):
            module = getattr(model, name, None)
            if module is None:
                continue
            for p in module.parameters(recurse=True):
                params.append((p, p.requires_grad))
                p.requires_grad_(False)
        yield
    finally:
        for p, req in params:
            p.requires_grad_(req)


def _single_pass_ppo(
    outs: Tuple[Any, ...],
    *,
    batch: Dict[str, torch.Tensor],
    mi: Dict[str, torch.Tensor],
    our_idx: torch.Tensor,
    our_mask: torch.Tensor,
    actions: torch.Tensor,
    old_logp: torch.Tensor,
    rewards: torch.Tensor,
    penalties_used: torch.Tensor,
    our_action_mask: Optional[torch.Tensor],
    step_mask: torch.Tensor,
    episode_mask: torch.Tensor,
    sl_teacher: Optional[torch.nn.Module],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    action_logits = outs[0]
    opp_logits = outs[1] if len(outs) > 1 else None
    values_full = outs[2].squeeze(-1).to(torch.float32)
    embedding_tuple = outs[3] if len(outs) > 3 else None

    B, T = our_idx.shape
    A = action_logits.size(-1)
    device = values_full.device

    logits_at = action_logits.gather(1, our_idx.unsqueeze(-1).expand(-1, -1, A))

    def _neg_inf_like(x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(torch.finfo(x.dtype).min, dtype=x.dtype, device=x.device)

    if our_action_mask is not None:
        step_mask_full = our_action_mask.gather(1, our_idx.unsqueeze(-1).expand(-1, -1, A))
        invalid_rows = (~step_mask_full).all(dim=-1)
        if invalid_rows.any():
            fb_cols = logits_at[invalid_rows].argmax(dim=-1)
            step_mask_full[invalid_rows] = False
            step_mask_full[invalid_rows, fb_cols] = True
        logits_at = logits_at.masked_fill(~step_mask_full, _neg_inf_like(logits_at))

    logits_at = torch.nan_to_num(
        logits_at, nan=0.0, posinf=0.0, neginf=float(torch.finfo(logits_at.dtype).min)
    )
    values_at = values_full.gather(1, our_idx)
    values_at = torch.nan_to_num(values_at, nan=0.0, posinf=0.0, neginf=0.0)

    rewards = rewards.where(our_mask, torch.zeros_like(rewards))

    next_idx = torch.zeros_like(our_idx)
    if T > 1:
        next_idx[:, :-1] = our_idx[:, 1:]
    L = values_full.size(1)
    idx_safe = next_idx.clamp(0, max(L - 1, 0))
    next_values = torch.take_along_dim(values_full, idx_safe, dim=1)

    has_next = torch.zeros_like(our_mask)
    if T > 1:
        has_next[:, :-1] = our_mask[:, 1:]
    has_next = has_next & our_mask

    gap_steps = (next_idx - our_idx).clamp_min(1).to(torch.float32)
    gap_steps = torch.where(has_next, gap_steps, torch.zeros_like(gap_steps))
    log_gamma = math.log(GAMMA)
    log_lam = math.log(GAE_LAMBDA)
    gamma_gap = torch.where(has_next, torch.exp(log_gamma * gap_steps), torch.zeros_like(gap_steps))
    lam_gap = torch.where(has_next, torch.exp(log_lam * gap_steps), torch.zeros_like(gap_steps))

    next_values = torch.where(has_next, next_values, torch.zeros_like(next_values))
    delta = rewards + gamma_gap * next_values - values_at
    delta = delta.where(our_mask, torch.zeros_like(delta))
    discount = gamma_gap * lam_gap

    advantages = torch.zeros_like(values_at)
    lastgaelam = torch.zeros((B,), device=device, dtype=torch.float32)
    for t in reversed(range(T)):
        lastgaelam = delta[:, t] + discount[:, t] * lastgaelam
        lastgaelam = torch.where(our_mask[:, t], lastgaelam, torch.zeros_like(lastgaelam))
        advantages[:, t] = lastgaelam
    returns = advantages + values_at

    adv_norm = _normalize_advantages(advantages, step_mask)

    dist = torch.distributions.Categorical(logits=logits_at)
    actions_for_log_prob = actions.masked_fill(~our_mask, 0)
    new_logp = dist.log_prob(actions_for_log_prob).to(torch.float32)
    entropy = dist.entropy().to(torch.float32)
    new_logp = new_logp.where(our_mask, torch.zeros_like(new_logp))
    entropy = entropy.where(our_mask, torch.zeros_like(entropy))

    log_ratio = (new_logp - old_logp).clamp(min=-60.0, max=60.0)
    ratio = log_ratio.exp()
    clipped_std = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP)
    clipped_neg = torch.clamp(ratio, 1.0 - EPS_CLIP, TRINAL_DELTA1)
    r_clipped = torch.where(advantages < 0, clipped_neg, clipped_std)
    surr1 = ratio * adv_norm
    surr2 = r_clipped * adv_norm
    policy_loss = -_masked_mean(torch.min(surr1, surr2), step_mask)

    with torch.no_grad():
        neg_mask = (advantages < 0) & step_mask
        trinal_clip_neg_frac = ((ratio > (1.0 + EPS_CLIP)) & neg_mask).float()
        denom_neg = neg_mask.float().sum().clamp_min(1.0)
        trinal_clip_neg_frac = trinal_clip_neg_frac.sum() / denom_neg

    ent_mean = _masked_mean(entropy, step_mask)
    entropy_loss = -ent_mean * ENT_COEF
    approx_kl = _masked_mean(old_logp - new_logp, step_mask)
    clipfrac = _masked_mean(((ratio - 1.0).abs() > EPS_CLIP).float(), step_mask)

    if step_mask.any():
        value_loss, vclip_frac = _value_loss_with_stakes_clip_public(
            v_pred=values_at[step_mask],
            returns=returns[step_mask],
            action_ids=actions[step_mask],
            penalties_used=penalties_used[step_mask].long(),
        )
    else:
        value_loss = torch.zeros((), device=device)
        vclip_frac = torch.zeros((), device=device)

    total = policy_loss + VALUE_WEIGHT * value_loss + entropy_loss

    opp_loss = torch.zeros((), device=device)
    opp_acc = torch.zeros((), device=device)
    opp_idx = batch.get("opp_idx")
    opp_targets = batch.get("opp_targets")
    opp_have_label = batch.get("opp_have_label")
    if (
        AUX_OPP_WEIGHT > 0.0
        and (opp_logits is not None)
        and (opp_idx is not None)
        and episode_mask.any()
    ):
        if opp_idx.numel() > 0:
            B_sel, L_sel, A_opp = opp_logits.shape
            To = opp_idx.size(1)
            opp_sel = opp_logits.gather(1, opp_idx.unsqueeze(-1).expand(-1, -1, A_opp))
            ce_opp = torch.nn.functional.cross_entropy(
                opp_sel.reshape(-1, A_opp),
                opp_targets.view(-1)
                if opp_targets is not None
                else torch.full((B_sel * To,), -100, device=device, dtype=torch.long),
                ignore_index=-100,
                reduction="none",
            ).view(B_sel, To)

            if opp_have_label is not None:
                w = (opp_have_label & episode_mask.unsqueeze(1)).to(ce_opp.dtype)
                if w.sum() > 0:
                    opp_loss = (ce_opp * w).sum() / w.sum().clamp_min(1.0)
                    with torch.no_grad():
                        pred = opp_sel.argmax(dim=-1)
                        corr = ((pred == opp_targets) & opp_have_label & episode_mask.unsqueeze(1)).sum().to(torch.float32)
                        denom = (opp_have_label & episode_mask.unsqueeze(1)).sum().clamp_min(1)
                        opp_acc = corr / denom.to(torch.float32)

    if AUX_OPP_WEIGHT > 0.0:
        total = total + AUX_OPP_WEIGHT * opp_loss

    bc_kl = torch.zeros((), device=device)
    if (BC_KL_WEIGHT > 0.0) and (sl_teacher is not None) and step_mask.any():
        with torch.no_grad():
            t_outs = sl_teacher(**mi)
            t_logits = t_outs[0]
            t_logits_at = t_logits.gather(1, our_idx.unsqueeze(-1).expand(-1, -1, A))
            if our_action_mask is not None:
                step_mask_full = our_action_mask.gather(1, our_idx.unsqueeze(-1).expand(-1, -1, A))
                t_logits_at = t_logits_at.masked_fill(~step_mask_full, _neg_inf_like(t_logits_at))
            t_logits_at = torch.nan_to_num(
                t_logits_at,
                nan=0.0,
                posinf=0.0,
                neginf=float(torch.finfo(t_logits_at.dtype).min),
            )
        dist_sl = torch.distributions.Categorical(logits=t_logits_at)
        bc_kl_val = torch.distributions.kl_divergence(dist, dist_sl)
        bc_kl = _masked_mean(bc_kl_val, step_mask)
        total = total + BC_KL_WEIGHT * bc_kl

    metrics: Dict[str, torch.Tensor] = {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": ent_mean.detach(),
        "approx_kl": approx_kl.detach(),
        "clip_fraction": clipfrac.detach(),
        "trinal_clip_neg_frac": trinal_clip_neg_frac.detach(),
        "value_clip_frac": vclip_frac.detach(),
        "opp_loss": opp_loss.detach(),
        "opp_action_acc": opp_acc.detach(),
        "bc_kl": bc_kl.detach(),
    }

    return total, metrics, embedding_tuple


def ppo_losses_batched(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    sl_teacher: Optional[torch.nn.Module] = None,
    *,
    update_num: int = 0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Batched PPO objective with compositional pressure support.
    """
    mi = batch["mi"]
    our_idx = batch["our_idx"].long()
    our_mask = batch["mask"].bool()
    actions = batch["actions"].long()
    old_logp = batch["old_logp"].float()
    rewards = batch["rewards"].float()
    penalties_used = batch["penalties_used"].long()
    our_action_mask = batch.get("our_action_mask")
    heldout_episode_mask = batch.get("heldout_episode_mask")
    use_heldout = _use_heldout_agents()
    if not use_heldout:
        heldout_episode_mask = torch.zeros(our_idx.size(0), dtype=torch.bool, device=our_idx.device)
    elif heldout_episode_mask is None:
        heldout_episode_mask = torch.zeros(our_idx.size(0), dtype=torch.bool, device=our_idx.device)

    dropout_p = getattr(config, "DROPOUT_P", 0.25)
    train_episode_mask = ~heldout_episode_mask
    train_step_mask = our_mask & train_episode_mask.unsqueeze(1)
    heldout_step_mask = our_mask & heldout_episode_mask.unsqueeze(1)

    outs_train = model(**{**mi, "return_embeddings": True, "dropout_p": dropout_p})
    train_loss, train_metrics, embedding_tuple = _single_pass_ppo(
        outs_train,
        batch=batch,
        mi=mi,
        our_idx=our_idx,
        our_mask=our_mask,
        actions=actions,
        old_logp=old_logp,
        rewards=rewards,
        penalties_used=penalties_used,
        our_action_mask=our_action_mask,
        step_mask=train_step_mask,
        episode_mask=train_episode_mask,
        sl_teacher=sl_teacher,
    )

    (
        l1_sparsity_loss,
        usage_balance_loss,
        brick_diversity_loss,
        decor_penalty,
        avg_brick_usage_np,
        embed_metrics,
    ) = _dictionary_regularizers(embedding_tuple, mi, batch, update_num=update_num)

    metrics: Dict[str, Any] = dict(train_metrics)
    metrics.update(embed_metrics)
    metrics["l1_sparsity_loss"] = l1_sparsity_loss.detach()
    metrics["usage_balance_loss"] = usage_balance_loss.detach()
    metrics["brick_diversity_loss"] = brick_diversity_loss.detach()
    metrics["brick_decorrelation_loss"] = decor_penalty.detach()
    if avg_brick_usage_np is not None:
        metrics["avg_brick_usage_np"] = avg_brick_usage_np

    # Effective DCP weight scales with held-out prevalence, then tuned by DCP_LOSS_WEIGHT
    # Change from per-episode to per-token weighting: use proportion of held-out
    # training tokens (our_mask) rather than count of episodes.
    dcp_tune = float(getattr(config, "DCP_LOSS_WEIGHT", 1.0))
    with torch.no_grad():
        total_tokens = float(our_mask.sum().item())
        heldout_tokens = float(heldout_step_mask.sum().item())
        heldout_frac_tokens = (heldout_tokens / max(total_tokens, 1.0)) if total_tokens > 0 else 0.0
        # Keep episode fraction as a diagnostic for backwards-compat dashboards
        B_eps = float(heldout_episode_mask.numel()) if heldout_episode_mask is not None else 0.0
        heldout_eps = float(heldout_episode_mask.sum().item()) if heldout_episode_mask is not None else 0.0
        heldout_frac = (heldout_eps / max(B_eps, 1.0)) if B_eps > 0 else 0.0
    lambda_cp = dcp_tune * heldout_frac_tokens
    decor_weight = float(getattr(config, "BRICK_DECORRELATION_WEIGHT", 0.0))

    dcp_loss = torch.zeros_like(train_loss)
    if heldout_step_mask.any():
        with _temporarily_freeze_heads(model):
            outs_cp = model(**{**mi, "return_embeddings": True, "dropout_p": dropout_p})
        dcp_loss, dcp_metrics, _ = _single_pass_ppo(
            outs_cp,
            batch=batch,
            mi=mi,
            our_idx=our_idx,
            our_mask=our_mask,
            actions=actions,
            old_logp=old_logp,
            rewards=rewards,
            penalties_used=penalties_used,
            our_action_mask=our_action_mask,
            step_mask=heldout_step_mask,
            episode_mask=heldout_episode_mask,
            sl_teacher=sl_teacher,
        )
        metrics.update({f"dcp_{k}": v for k, v in dcp_metrics.items()})
        metrics["dcp_total_loss"] = dcp_loss.detach()
    else:
        metrics["dcp_total_loss"] = dcp_loss.detach()

    total_loss = train_loss + lambda_cp * dcp_loss
    total_loss = total_loss + getattr(config, "L1_SPARSITY_WEIGHT", 0.0) * l1_sparsity_loss
    total_loss = total_loss + getattr(config, "USAGE_BALANCE_WEIGHT", 0.0) * usage_balance_loss
    total_loss = total_loss + getattr(config, "BRICK_DIVERSITY_WEIGHT", 0.0) * brick_diversity_loss
    total_loss = total_loss + decor_weight * decor_penalty

    metrics["dcp_weighted_loss"] = (lambda_cp * dcp_loss).detach()
    metrics["dcp_weight_eff"] = torch.tensor(lambda_cp, device=train_loss.device)
    metrics["heldout_token_frac"] = torch.tensor(heldout_frac_tokens, device=train_loss.device)
    metrics["heldout_episode_frac"] = torch.tensor(heldout_frac, device=train_loss.device)
    metrics["total_loss"] = total_loss.detach()

    return total_loss, metrics


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
      opp_idx [B,To], opp_targets [B,To], opp_have_label [B,To]
      opp_last_token_idx [B,num_opponents], opp_last_token_is_adjacent [B,num_opponents]
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

    # -------- build batched model inputs (time-major tensors only) --------
    mi_batch: Dict[str, torch.Tensor] = {}
    for k in _COLLATE_EXPECTED_MI_KEYS:
        vs = [ep["model_input"].get(k) for ep in episodes]

        valid_vs = [v for v in vs if v is not None and torch.is_tensor(v) and v.dim() >= 2]
        if not valid_vs:
            continue

        if len(valid_vs) != len(vs):
            print(f"Warning: Key '{k}' missing in some episodes, skipping for this batch.")
            continue

        first = valid_vs[0]
        if first.dim() < 2:
            raise ValueError(f"model_input['{k}'] must be at least 2D per episode.")

        out_shape = list(first.shape)
        out_shape[0] = len(vs)
        out_shape[1] = L_pad
        cat = first.new_zeros(out_shape)

        for b, v in enumerate(vs):
            if not torch.is_tensor(v):
                raise ValueError(f"model_input['{k}'] missing tensor for episode {b}.")
            if v.dim() < 2:
                raise ValueError(f"model_input['{k}'] must be at least 2D per episode.")
            if v.size(0) != 1:
                raise ValueError(
                    f"model_input['{k}'] expected leading batch size 1 per episode, got {v.size(0)}."
                )
            Lb = min(v.size(1), L_pad)
            if Lb > 0:
                cat[b, :Lb].copy_(v[0, :Lb].contiguous())

        cat = cat.contiguous()
        if pin_memory:
            cat = cat.pin_memory()
        mi_batch[k] = cat

    # ---- REBUILD valid_lengths and padding_mask from the true lengths ----
    valid_lengths = torch.tensor([min(l, L_pad) for l in raw_lens], dtype=torch.long)
    if pin_memory: valid_lengths = valid_lengths.pin_memory()
    mi_batch["valid_lengths"] = valid_lengths  # [B]

    if L_pad > 0:
        token_range = torch.arange(L_pad, dtype=torch.long)
        padding_mask = token_range.unsqueeze(0) >= valid_lengths.unsqueeze(1)
    else:
        token_range = torch.arange(0, dtype=torch.long)
        padding_mask = torch.zeros((B, 0), dtype=torch.bool)
    if pin_memory:
        padding_mask = padding_mask.pin_memory()
    mi_batch["padding_mask"] = padding_mask     # [B, L_pad]

    # Require agent_types for actor/opp selection
    if "agent_types" not in mi_batch:
        raise ValueError("model_input must include 'agent_types' with dim>=2 (batched [B, L]).")
    agent_types = mi_batch["agent_types"].long()  # [B, L_pad]

    valid_token_mask = ~padding_mask
    our_token_mask_full = (agent_types == 0) & valid_token_mask
    opp_token_mask_full = (agent_types != 0) & valid_token_mask

    our_counts = our_token_mask_full.sum(dim=1)
    opp_counts = opp_token_mask_full.sum(dim=1)

    T = int(our_counts.max().item()) if our_counts.numel() > 0 else 0
    To = int(opp_counts.max().item()) if opp_counts.numel() > 0 else 0

    if L_pad > 0:
        token_idx = token_range.unsqueeze(0).expand(B, -1)
        sentinel_idx = torch.full_like(token_idx, L_pad)
    else:
        token_idx = torch.zeros((B, 0), dtype=torch.long)
        sentinel_idx = token_idx.clone()

    def _mk_idx(mask: torch.Tensor, counts: torch.Tensor, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if max_len <= 0:
            empty_idx = torch.zeros((B, 0), dtype=torch.long)
            empty_mask = torch.zeros((B, 0), dtype=torch.bool)
            if pin_memory:
                empty_idx = empty_idx.pin_memory()
                empty_mask = empty_mask.pin_memory()
            return empty_idx, empty_mask

        sorted_idx = torch.sort(torch.where(mask, token_idx, sentinel_idx), dim=1).values[:, :max_len]
        sorted_idx = sorted_idx.contiguous()
        slot_mask = torch.arange(max_len, dtype=torch.long).unsqueeze(0) < counts.unsqueeze(1)
        sorted_idx = sorted_idx.clamp_max(max(L_pad - 1, 0))
        sorted_idx = torch.where(slot_mask, sorted_idx, torch.zeros_like(sorted_idx))
        slot_mask = slot_mask.to(torch.bool)
        if pin_memory:
            sorted_idx = sorted_idx.pin_memory()
            slot_mask = slot_mask.pin_memory()
        return sorted_idx, slot_mask

    our_idx_tensor, our_mask_tensor = _mk_idx(our_token_mask_full, our_counts, T)
    opp_idx_tensor, _ = _mk_idx(opp_token_mask_full, opp_counts, To)

    # Optional legality mask for OUR steps; we'll zero it past each valid length
    our_action_mask = None
    if "action_masks" in mi_batch:
        m = mi_batch["action_masks"].bool()
        if m.size(1) != L_pad:
            new_shape = (m.size(0), L_pad, m.size(2))
            new_m = m.new_zeros(new_shape)
            Lb = min(m.size(1), L_pad)
            if Lb > 0:
                new_m[:, :Lb].copy_(m[:, :Lb])
            m = new_m
        valid_mask = valid_token_mask.unsqueeze(-1)
        m = m & valid_mask
        if pin_memory:
            m = m.pin_memory()
        our_action_mask = m

    opponent_counts: List[int] = []

    # Labels <= 6 are classic C++ bots; only consider held-out labels > 6
    BOT_MAX_ID = 6

    def _label_to_int(label: Any) -> Optional[int]:
        if label is None:
            return None
        if isinstance(label, (int, np.integer)):
            return int(label)
        if isinstance(label, torch.Tensor):
            return int(label.item())
        try:
            return int(label)
        except (TypeError, ValueError):
            try:
                if hasattr(label, "item"):
                    return int(label.item())
            except Exception:
                pass
        return None

    use_heldout = _use_heldout_agents()
    if use_heldout:
        # --- REVISED: Efficient Held-out Label Selection (Single Pass) ---
        heldout_label = None
        episode_opponent_labels_sets: List[Set[int]] = []

        # First pass: find the max opponent label across the batch AND cache each episode's opponents
        for ep in episodes:
            training_seat = int(ep.get("training_agent_seat", -1))
            # Prefer true_opponent_labels if available, else derive from player_labels
            opp_labels_source = ep.get("true_opponent_labels", ep.get("player_labels", ()))

            current_ep_opp_labels = set()
            for i, lab in enumerate(opp_labels_source):
                # If using player_labels, skip the training agent
                if "true_opponent_labels" not in ep and i == training_seat:
                    continue
                lab_i = _label_to_int(lab)
                if lab_i is not None and lab_i != ep.get("training_agent_label") and lab_i > BOT_MAX_ID:
                    current_ep_opp_labels.add(lab_i)

            episode_opponent_labels_sets.append(current_ep_opp_labels)

            if current_ep_opp_labels:
                ep_max_label = max(current_ep_opp_labels)
                if heldout_label is None or ep_max_label > heldout_label:
                    heldout_label = ep_max_label

        # Second pass (fast): build the boolean mask based on the determined heldout_label
        if heldout_label is not None:
            heldout_flags = [heldout_label in opp_set for opp_set in episode_opponent_labels_sets]
        else:
            heldout_flags = [False] * len(episodes)
    else:
        heldout_flags = [False] * len(episodes)
    
    num_opponents = 0
    for ep in episodes:
        count = sum(1 for seat_idx in range(len(ep.get("player_labels", ()))) if seat_idx != ep.get("training_agent_seat", -1))
        opponent_counts.append(count)
    if opponent_counts:
        num_opponents = max(opponent_counts)
    # --- END REVISION ---

    # -------- allocate supervision tensors (CPU) --------
    def _pm(x: torch.Tensor) -> torch.Tensor:
        return x.pin_memory() if pin_memory and hasattr(x, "is_pinned") and not x.is_pinned() else x

    our_idx    = _pm(our_idx_tensor)
    our_mask   = _pm(our_mask_tensor)
    actions    = _pm(torch.full((B, T), IGN, dtype=torch.long))
    old_logp   = _pm(torch.zeros((B, T),  dtype=torch.float32))
    rewards    = _pm(torch.zeros((B, T),  dtype=torch.float32))
    pen_used   = _pm(torch.zeros((B, T),  dtype=torch.long))

    # Opponent action supervision (unchanged)
    opp_idx        = _pm(opp_idx_tensor)
    opp_targets    = _pm(torch.full((B, To), IGN, dtype=torch.long))
    opp_have_label = _pm(torch.zeros((B, To),  dtype=torch.bool))

    if our_action_mask is not None:
        our_action_mask = _pm(our_action_mask)

    opp_labels_by_seat = torch.full((B, num_opponents), IGN, dtype=torch.long)
    opp_seat_ids = torch.full((B, num_opponents), -1, dtype=torch.long)
    opp_last_token_idx = torch.full((B, num_opponents), -1, dtype=torch.long)
    opp_last_token_is_adjacent = torch.zeros((B, num_opponents), dtype=torch.bool)
    heldout_episode_mask = _pm(torch.tensor(heldout_flags, dtype=torch.bool))

    def _int_array(seq: Any, invalid_fill: int = -1) -> np.ndarray:
        if seq is None:
            return np.empty(0, dtype=np.int64)
        if isinstance(seq, np.ndarray):
            arr = seq
        elif torch.is_tensor(seq):
            arr = seq.detach().cpu().numpy()
        else:
            arr = np.asarray(seq)
        if arr.dtype == np.object_:
            flat = [
                invalid_fill
                if (x is None or (isinstance(x, float) and np.isnan(x)))
                else int(x)
                for x in arr.tolist()
            ]
            arr = np.asarray(flat, dtype=np.int64)
        else:
            arr = arr.astype(np.int64, copy=False)
        return arr.reshape(-1)

    def _float_array(seq: Any) -> np.ndarray:
        if seq is None:
            return np.empty(0, dtype=np.float32)
        if isinstance(seq, np.ndarray):
            arr = seq
        elif torch.is_tensor(seq):
            arr = seq.detach().cpu().numpy()
        else:
            arr = np.asarray(seq)
        if arr.dtype == np.object_:
            arr = np.asarray(
                [0.0 if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x) for x in arr.tolist()],
                dtype=np.float32,
            )
        else:
            arr = arr.astype(np.float32, copy=False)
        return arr.reshape(-1)

    # -------- fill from episodes (only real steps) --------
    for b, ep in enumerate(episodes):
        training_seat = int(ep.get("training_agent_seat", -1))
        agent_id_seq = _int_array(ep.get("agent_id"), invalid_fill=-1)

        # ===== OUR timeline =====
        our_ep_idx = np.nonzero(agent_id_seq == training_seat)[0]
        max_steps = min(T, our_ep_idx.size, int(our_counts[b].item())) if our_counts.numel() > 0 else 0

        if max_steps < our_mask.size(1):
            our_mask[b, max_steps:] = False
            if max_steps < our_idx.size(1):
                our_idx[b, max_steps:] = 0

        if max_steps > 0:
            steps = our_ep_idx[:max_steps]

            # Log probabilities
            lp_src = _float_array(ep.get("log_prob"))
            lp_dest = np.zeros(max_steps, dtype=np.float32)
            if lp_src.size > 0:
                valid_lp = steps < lp_src.shape[0]
                if np.any(valid_lp):
                    lp_dest[valid_lp] = lp_src[steps[valid_lp]]
            old_logp[b, :max_steps] = torch.from_numpy(lp_dest)

            # Rewards
            rw_src = _float_array(ep.get("reward"))
            rw_dest = np.zeros(max_steps, dtype=np.float32)
            if rw_src.size > 0:
                valid_rw = steps < rw_src.shape[0]
                if np.any(valid_rw):
                    rw_dest[valid_rw] = rw_src[steps[valid_rw]]
            rewards[b, :max_steps] = torch.from_numpy(rw_dest)

            # Penalties used
            pu_src = _int_array(ep.get("penalties_used"), invalid_fill=0)
            pu_dest = np.zeros(max_steps, dtype=np.int64)
            if pu_src.size > 0:
                valid_pu = steps < pu_src.shape[0]
                if np.any(valid_pu):
                    pu_dest[valid_pu] = pu_src[steps[valid_pu]]
            pen_used[b, :max_steps] = torch.from_numpy(pu_dest)

            # Actions (respect IGN sentinel)
            act_src = _int_array(ep.get("our_action"), invalid_fill=-1)
            act_dest = np.full(max_steps, IGN, dtype=np.int64)
            if act_src.size > 0:
                valid_act = steps < act_src.shape[0]
                if np.any(valid_act):
                    sel = act_src[steps[valid_act]]
                    valid_vals = sel >= 0
                    if np.any(valid_vals):
                        idx = np.nonzero(valid_act)[0][valid_vals]
                        act_dest[idx] = sel[valid_vals]
            actions[b, :max_steps] = torch.from_numpy(act_dest)

        # ===== OPP timeline =====
        opp_ep_idx = np.nonzero(agent_id_seq != training_seat)[0]
        M_fill = min(To, opp_ep_idx.size, int(opp_counts[b].item())) if opp_counts.numel() > 0 else 0
        if M_fill > 0:
            # Episode metadata we already saved
            player_labels = tuple(ep.get("player_labels", ()))  # absolute seat -> label

            # Record per-seat labels for visualization later
            opp_entries: List[Tuple[int, int]] = []
            for seat_idx, lbl in enumerate(player_labels):
                if seat_idx == training_seat:
                    continue
                if lbl is None:
                    opp_entries.append((seat_idx, IGN))
                else:
                    opp_entries.append((seat_idx, int(lbl)))
            seat_pos_map: Dict[int, int] = {}
            for j, (seat_idx, lbl) in enumerate(opp_entries[:num_opponents]):
                opp_seat_ids[b, j] = seat_idx
                if lbl != IGN:
                    opp_labels_by_seat[b, j] = lbl
                seat_pos_map[seat_idx] = j

            steps = opp_ep_idx[:M_fill]

            if seat_pos_map and agent_id_seq.size > 0 and L_pad > 0 and training_seat >= 0:
                seq_len = int(agent_id_seq.shape[0])
                max_token_index = min(L_pad, seq_len)
                for seat_idx, col in seat_pos_map.items():
                    occ = np.nonzero(agent_id_seq == seat_idx)[0]
                    if occ.size == 0:
                        continue
                    last_idx = int(occ[-1])
                    if last_idx >= max_token_index:
                        continue
                    opp_last_token_idx[b, col] = last_idx
                    adjacent = False
                    if last_idx > 0:
                        prev_agent = int(agent_id_seq[last_idx - 1])
                        adjacent = (prev_agent == training_seat)
                    if not adjacent and (last_idx + 1) < seq_len:
                        next_agent = int(agent_id_seq[last_idx + 1])
                        adjacent = (next_agent == training_seat)
                    if adjacent:
                        opp_last_token_is_adjacent[b, col] = True

            tgt_src = _int_array(ep.get("opp_target_action"), invalid_fill=-1)
            tgt_dest = np.full(M_fill, IGN, dtype=np.int64)
            have_dest = np.zeros(M_fill, dtype=np.bool_)
            if tgt_src.size > 0:
                valid_tgt = steps < tgt_src.shape[0]
                if np.any(valid_tgt):
                    sel = tgt_src[steps[valid_tgt]]
                    valid_vals = sel >= 0
                    if np.any(valid_vals):
                        idx = np.nonzero(valid_tgt)[0][valid_vals]
                        tgt_dest[idx] = sel[valid_vals]
                        have_dest[idx] = True

            opp_targets[b, :M_fill] = torch.from_numpy(tgt_dest)
            opp_have_label[b, :M_fill] = torch.from_numpy(have_dest)

    return {
        "mi": mi_batch,
        "our_idx": our_idx,
        "mask": our_mask,
        "actions": actions,
        "old_logp": old_logp,
        "rewards": rewards,
        "penalties_used": pen_used,
        "our_action_mask": our_action_mask,
        "opp_idx":        opp_idx,
        "opp_targets":    opp_targets,
        "opp_have_label": opp_have_label,
        "opp_labels_by_seat": _pm(opp_labels_by_seat),
        "opp_seat_ids": _pm(opp_seat_ids),
        "opp_last_token_idx": _pm(opp_last_token_idx),
        "opp_last_token_is_adjacent": _pm(opp_last_token_is_adjacent),
        "heldout_episode_mask": heldout_episode_mask,
    }

def _to_device_batch(batch_cpu: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move a collated CPU batch (with nested 'mi' dict) to device."""
    mi_dev = {k: v.to(device, non_blocking=True) for k, v in batch_cpu["mi"].items()}
    oam = batch_cpu.get("our_action_mask", None)
    oam_dev = oam.to(device, non_blocking=True) if (oam is not None) else None
    out = {
        "mi":              mi_dev,
        "our_idx":         batch_cpu["our_idx"].to(device, non_blocking=True),
        "mask":            batch_cpu["mask"].to(device, non_blocking=True),
        "actions":         batch_cpu["actions"].to(device, non_blocking=True),
        "old_logp":        batch_cpu["old_logp"].to(device, non_blocking=True),
        "rewards":         batch_cpu["rewards"].to(device, non_blocking=True),
        "penalties_used":  batch_cpu["penalties_used"].to(device, non_blocking=True),
        "our_action_mask": oam_dev,
        # Opponent action supervision
        "opp_idx":        batch_cpu["opp_idx"].to(device, non_blocking=True),
        "opp_targets":    batch_cpu["opp_targets"].to(device, non_blocking=True),
        "opp_have_label": batch_cpu["opp_have_label"].to(device, non_blocking=True),
        "opp_labels_by_seat": batch_cpu["opp_labels_by_seat"].to(device, non_blocking=True),
        "opp_seat_ids":      batch_cpu["opp_seat_ids"].to(device, non_blocking=True),
        "opp_last_token_idx": batch_cpu["opp_last_token_idx"].to(device, non_blocking=True),
        "opp_last_token_is_adjacent": batch_cpu["opp_last_token_is_adjacent"].to(device, non_blocking=True),
        "heldout_episode_mask": batch_cpu["heldout_episode_mask"].to(device, non_blocking=True),
    }
    return out
