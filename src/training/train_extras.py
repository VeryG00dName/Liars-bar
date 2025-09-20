# src/training/train_extras.py

import random
from collections import Counter
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple, Iterable, Union, Hashable
import itertools
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
    fig.update_layout(
        title=ttl + f" — N={Xs.shape[0]}",
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
) -> dict:
    label_list = [str(l) for l in opp_labels]
    labels_arr = np.asarray(label_list)
    N, D = X.shape
    n_comp = int(max(2, min(4, D, N)))
    pca = PCA(n_components=n_comp, random_state=0)
    Xp = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_

    pairs = [(0, 1), (1, 2), (2, 3)]
    valid_pairs = [pair for pair in pairs if pair[1] < n_comp]
    if not valid_pairs:
        return {"pca_evr": evr}

    fig, axes = plt.subplots(1, len(valid_pairs), figsize=(6 * len(valid_pairs), 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    uniq = sorted(set(label_list))
    cmap = plt.cm.get_cmap("tab20", max(1, len(uniq)))
    scatter_cap = 400
    rng = np.random.default_rng(0)
    legend_handles: List[Line2D] = []

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
            ax.scatter(show_pts[:, 0], show_pts[:, 1], color=color, s=8, alpha=0.25, linewidths=0)

            centroid = pts.mean(axis=0)
            ax.scatter(
                centroid[0],
                centroid[1],
                marker="X",
                s=80,
                color=color,
                edgecolor="black",
                linewidths=0.8,
            )
            if i == valid_pairs[0][0] and j == valid_pairs[0][1]:
                legend_handles.append(Line2D([0], [0], marker="X", color="w",
                                             markerfacecolor=color, markeredgecolor="black",
                                             linewidth=0, markersize=9, label=f"Opp {lab}"))

            if pts.shape[0] >= 3:
                cov = np.cov(pts, rowvar=False)
                ell = _ellipse_from_cov(centroid, cov, color)
                if ell is not None:
                    ax.add_patch(ell)

        ax.set_aspect("equal", adjustable="box")

    if legend_handles:
        axes[0].legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left",
                       borderaxespad=0.0, title="Opponents")

    evr_pad = np.pad(evr, (0, max(0, 4 - evr.size)), constant_values=np.nan)
    evr_text = ", ".join(
        [
            f"PC{k+1}: {val:.2f}" if np.isfinite(val) else f"PC{k+1}: n/a"
            for k, val in enumerate(evr_pad[:4])
        ]
    )
    fig.suptitle(f"{title_prefix} — PCA multi-view — step {step}\n{evr_text}")

    if evr.size >= 3 and np.isfinite(evr[:3]).all():
        pc3_ratio = float(evr[2] / max(evr[0] + evr[1], 1e-9))
        fig.text(0.5, 0.02, f"PC3 vs PC1-2 EVR ratio: {pc3_ratio:.3f}", ha="center")
    else:
        pc3_ratio = float("nan")

    fig.tight_layout(rect=[0, 0.05, 1, 0.92])

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    image = Image.open(buf)
    writer.add_image("Embeddings/PCA multi-view", np.array(image), step, dataformats="HWC")
    plt.close(fig)

    return {
        "pca_evr": evr,
        "pc3_vs_pc12": pc3_ratio,
        "n_components": n_comp,
        "pairs": valid_pairs,
    }

def visualize_opponent_embeddings_all(
    writer,
    data,                      # dict {opp: emb} OR (X, opp_labels)
    step: int,
    methods: list | tuple | None = None,
    pca_dim: int = 50,
    title_prefix: str = "Opponent Embeddings",
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
    title_prefix: str = "Opponent Embeddings"
):
    # ---- coerce input ----
    X, opp_labels = _coerce_opponent_input(data)
    if X.shape[0] < 2:
        return None  # need ≥2 points

    N, D = X.shape
    if method == "pca_panels":
        return _visualize_pca_panels(writer, X, opp_labels, step, title_prefix)

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
def ppo_losses_batched(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    sl_teacher: Optional[torch.nn.Module] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Fully batched PPO objective with:
      • irregular-step GAE computed inside (from batch['rewards'] & model values)
      • stakes-based value target clipping (optional)
      • opponent action CE (batched; NO action masking; -100 ignored)
      • optional teacher KL at OUR steps

    Requires in batch:
      mi, our_idx [B,T], mask [B,T], actions [B,T], old_logp [B,T],
      rewards [B,T], penalties_used [B,T],
      our_action_mask [B,L,A] or None,
      opp_idx [B,To], opp_targets [B,To], opp_have_label [B,To]
    """
    mi = batch["mi"]
    our_idx = batch["our_idx"].long()     # [B, T]
    our_mask = batch["mask"].bool()       # [B, T]
    actions = batch["actions"].long()     # [B, T]
    old_logp = batch["old_logp"].float()  # [B, T]
    rewards = batch["rewards"].float()    # [B, T]

    outs = model(**{**mi, "return_embeddings": True, "dropout_p": getattr(config, "DROPOUT_P", 0.25)})
    action_logits = outs[0]                                # [B, L, A]
    opp_logits    = outs[1] if len(outs) > 1 else None     # [B, L, A] or None
    values_full   = outs[2].squeeze(-1).to(torch.float32)  # [B, L]
    embedding_tuple = outs[3] if len(outs) > 3 else None   # (strategy_code, activations, bricks)

    strategy_code = None
    activations = None
    bricks = None
    if isinstance(embedding_tuple, tuple) and len(embedding_tuple) == 3:
        strategy_code, activations, bricks = embedding_tuple

    B, T = our_idx.shape
    A = action_logits.size(-1)

    # ---- Gather OUR-step logits ----
    logits_at = action_logits.gather(1, our_idx.unsqueeze(-1).expand(-1, -1, A))  # [B,T,A]

    def _neg_inf_like(x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(torch.finfo(x.dtype).min, dtype=x.dtype, device=x.device)
    
    # Apply legality mask for OUR steps only (if provided)
    if batch.get("our_action_mask", None) is not None:
        step_mask = batch["our_action_mask"].gather(
            1, our_idx.unsqueeze(-1).expand(-1, -1, A)
        )
        invalid_rows = (~step_mask).all(dim=-1)
        if invalid_rows.any():
            fb_cols = logits_at[invalid_rows].argmax(dim=-1)
            step_mask[invalid_rows] = False
            step_mask[invalid_rows, fb_cols] = True
        logits_at = logits_at.masked_fill(~step_mask, _neg_inf_like(logits_at))

    logits_at = torch.nan_to_num(logits_at, nan=0.0, posinf=0.0, neginf=float(torch.finfo(logits_at.dtype).min))
    values_at = values_full.gather(1, our_idx)
    values_at = torch.nan_to_num(values_at, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Build "next" indices & gaps for irregular-step GAE ----
    next_idx = torch.full_like(our_idx, -1)
    if T > 1:
        next_idx[:, :-1] = our_idx[:, 1:]

    has_next = torch.zeros_like(our_mask)
    if T > 1:
        has_next[:, :-1] = our_mask[:, 1:]

    gaps = torch.zeros_like(our_idx, dtype=torch.long)
    valid_gap = has_next & our_mask
    gaps[valid_gap] = (next_idx[valid_gap] - our_idx[valid_gap]).clamp_min(1)
    gamma_gap = (GAMMA ** gaps.to(torch.float32))
    lam_gap   = (GAE_LAMBDA ** gaps.to(torch.float32))

    # ---- Irregular-step GAE (vectorized backward over T) ----
    with torch.no_grad():
        advantages = torch.zeros_like(values_at)
        lastgaelam = torch.zeros((B,), device=values_at.device, dtype=torch.float32)
        for t in reversed(range(T)):
            g  = torch.where(has_next[:, t], gamma_gap[:, t], torch.zeros_like(gamma_gap[:, t]))
            gl = torch.where(has_next[:, t], gamma_gap[:, t] * lam_gap[:, t], torch.zeros_like(gamma_gap[:, t]))
            L = values_full.size(1)
            idx_safe = next_idx[:, t].clamp(0, L - 1)
            nv = torch.where(
                has_next[:, t],
                values_full.gather(1, idx_safe.unsqueeze(-1)).squeeze(-1),
                torch.zeros_like(values_at[:, t]),
            )
            delta = rewards[:, t] + g * nv - values_at[:, t]
            lastgaelam = delta + gl * lastgaelam
            advantages[:, t] = lastgaelam
            lastgaelam = torch.where(our_mask[:, t], lastgaelam, lastgaelam * 0.0)
        returns = advantages + values_at

    # Normalize advantages using only valid positions
    m = our_mask.to(torch.float32)
    adv_sum = (advantages * m).sum()
    m_sum = m.sum().clamp_min(1.0)
    adv_mean = adv_sum / m_sum
    adv_var  = ((advantages - adv_mean).pow(2) * m).sum() / m_sum
    adv_std  = torch.sqrt(adv_var)
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    # ---- PPO objective (masked) ----
    dist = torch.distributions.Categorical(logits=logits_at)
    
    # Replace padded actions (-100) with a valid placeholder (0) to prevent crash.
    actions_for_log_prob = actions.masked_fill(~our_mask, 0)
    
    # Calculate log_prob and entropy. The results for padded steps are garbage.
    new_logp = dist.log_prob(actions_for_log_prob).to(torch.float32)
    entropy  = dist.entropy().to(torch.float32)

    # CRITICAL FIX: Zero out the garbage values for padded steps.
    new_logp = new_logp.where(our_mask, 0.0)
    entropy = entropy.where(our_mask, 0.0)
    # old_logp is already correctly zero for padded steps from the collate function.

    def masked_mean(x: torch.Tensor) -> torch.Tensor:
        w = our_mask.to(x.dtype)
        return (x * w).sum() / w.sum().clamp_min(1.0)

    # The rest of the calculation is now safe.
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
    # Ensure we only compute value loss on valid, unpadded steps
    if our_mask.any():
        value_loss, vclip_frac = _value_loss_with_stakes_clip_public(
            v_pred=values_at[our_mask],
            returns=returns[our_mask],
            action_ids=actions[our_mask],
            penalties_used=batch["penalties_used"][our_mask].long(),
        )
    else: # Handle empty batch case
        value_loss = torch.tensor(0.0, device=values_at.device)
        vclip_frac = torch.tensor(0.0, device=values_at.device)
    
    total = policy_loss + VALUE_WEIGHT * value_loss + entropy_loss

    l1_sparsity_loss = torch.zeros((), device=values_full.device)
    usage_balance_loss = torch.zeros((), device=values_full.device)
    brick_diversity_loss = torch.zeros((), device=values_full.device)
    avg_brick_usage_np = None

    if activations is not None:
        agent_types = mi.get("agent_types")
        padding_mask = mi.get("padding_mask")
        if agent_types is not None:
            opp_mask = (agent_types != 0)
            if padding_mask is not None:
                opp_mask = opp_mask & (~padding_mask.bool())
            if opp_mask.any():
                opp_activations = activations[opp_mask]
                l1_sparsity_loss = opp_activations.abs().mean()

                brick_sums = opp_activations.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                probs = opp_activations / brick_sums
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
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

    total = (
        total
        + getattr(config, "L1_SPARSITY_WEIGHT", 0.0) * l1_sparsity_loss
        + getattr(config, "USAGE_BALANCE_WEIGHT", 0.0) * usage_balance_loss
        + getattr(config, "BRICK_DIVERSITY_WEIGHT", 0.0) * brick_diversity_loss
    )

    metrics: Dict[str, torch.Tensor] = {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": ent_mean.detach(),
        "approx_kl": approx_kl.detach(),
        "clip_fraction": clipfrac.detach(),
        "trinal_clip_neg_frac": trinal_clip_neg_frac.detach(),
        "value_clip_frac": vclip_frac.detach(),
    }

    metrics["l1_sparsity_loss"] = l1_sparsity_loss.detach()
    metrics["usage_balance_loss"] = usage_balance_loss.detach()
    metrics["brick_diversity_loss"] = brick_diversity_loss.detach()
    if avg_brick_usage_np is not None:
        metrics["avg_brick_usage_np"] = avg_brick_usage_np

    # ---- Opponent embedding summaries (for visualization) ----
    if strategy_code is not None:
        agent_types = mi.get("agent_types")
        padding_mask = mi.get("padding_mask")
        opp_labels_by_seat = batch.get("opp_labels_by_seat")
        opp_seat_ids = batch.get("opp_seat_ids")
        opp_idx = batch.get("opp_idx")
        opp_have_label = batch.get("opp_have_label")

        if (
            agent_types is not None
            and opp_labels_by_seat is not None
            and opp_seat_ids is not None
            and opp_idx is not None
        ):
            B_emb, To = opp_idx.shape
            _, _, D_emb = strategy_code.shape

            idx = opp_idx.long().clamp_min(0)
            tok_embeds = strategy_code.gather(1, idx.unsqueeze(-1).expand(-1, -1, D_emb))
            seat_tokens = agent_types.gather(1, idx)

            valid_tok = (opp_idx >= 0)
            if opp_have_label is not None:
                valid_tok = valid_tok & opp_have_label
            if padding_mask is not None:
                pad_at = padding_mask.gather(1, idx.clamp_max(padding_mask.size(1) - 1))
                valid_tok = valid_tok & (~pad_at.bool())

            num_seats = opp_labels_by_seat.size(1)
            seat_embeds = torch.full(
                (B_emb, num_seats, D_emb),
                float("nan"),
                device=tok_embeds.device,
                dtype=tok_embeds.dtype,
            )
            seat_counts = torch.zeros(
                (B_emb, num_seats), device=tok_embeds.device, dtype=tok_embeds.dtype
            )
            seat_labels = opp_labels_by_seat.clone()
            seat_labels = seat_labels.masked_fill(seat_labels < 0, -1)

            embeds_flat: List[torch.Tensor] = []
            labels_flat: List[int] = []

            for b in range(B_emb):
                for slot in range(num_seats):
                    seat_id = int(opp_seat_ids[b, slot].item())
                    if seat_id < 0:
                        seat_labels[b, slot] = -1
                        continue

                    mask = (seat_tokens[b] == seat_id) & valid_tok[b]
                    count = int(mask.sum().item())
                    if count == 0:
                        seat_labels[b, slot] = -1
                        continue

                    embed_vec = tok_embeds[b, mask].mean(dim=0)
                    seat_embeds[b, slot] = embed_vec
                    seat_counts[b, slot] = float(count)

                    label_val = int(seat_labels[b, slot].item())
                    if label_val >= 0 and not torch.isnan(embed_vec).any():
                        embeds_flat.append(embed_vec.detach().cpu().float())
                        labels_flat.append(label_val)

            metrics["opp_embeds_batch"] = (
                seat_embeds.detach().cpu().float().numpy(),
                seat_labels.detach().cpu().numpy(),
                seat_counts.detach().cpu().float().numpy(),
            )

            if embeds_flat:
                embeds_tensor = torch.stack(embeds_flat, dim=0)
                metrics["opp_embeds_flat"] = embeds_tensor.numpy()
                metrics["opp_labels_flat"] = labels_flat
                metrics["opp_labels_flat_original"] = labels_flat

    # ---- Teacher KL (optional) ----
    if (BC_KL_WEIGHT > 0.0) and (sl_teacher is not None):
        with torch.no_grad():
            t_outs = sl_teacher(**mi)
            t_logits = t_outs[0]
            t_logits_at = t_logits.gather(1, our_idx.unsqueeze(-1).expand(-1, -1, A))
            if batch.get("our_action_mask", None) is not None:
                step_mask = batch["our_action_mask"].gather(1, our_idx.unsqueeze(-1).expand(-1, -1, A))
                t_logits_at = t_logits_at.masked_fill(~step_mask, _neg_inf_like(t_logits_at))
            t_logits_at = torch.nan_to_num(t_logits_at, nan=0.0, posinf=0.0, neginf=float(torch.finfo(t_logits_at.dtype).min))
        dist_sl = torch.distributions.Categorical(logits=t_logits_at)
        bc_kl = torch.distributions.kl_divergence(dist, dist_sl)
        bc_kl = masked_mean(bc_kl)
        total = total + BC_KL_WEIGHT * bc_kl
        metrics["bc_kl"] = bc_kl.detach()
    else:
        metrics["bc_kl"] = torch.zeros((), device=logits_at.device)

    # =========================
    # Shared opponent timeline
    # =========================
    opp_idx        = batch.get("opp_idx", None)            # [B, To]
    opp_targets    = batch.get("opp_targets", None)        # [B, To]
    opp_have_label = batch.get("opp_have_label", None)     # [B, To]
    device = values_full.device

    # ---- Aux: opponent action supervision (re-use opp_idx/targets/mask) ----
    opp_loss = torch.zeros((), device=device)
    opp_acc  = torch.zeros((), device=device)
    if AUX_OPP_WEIGHT > 0.0 and (opp_logits is not None) and (opp_idx is not None):
        if opp_idx.numel() > 0:
            B, L, A_opp = opp_logits.shape
            To = opp_idx.size(1)
            opp_sel = opp_logits.gather(1, opp_idx.unsqueeze(-1).expand(-1, -1, A_opp))  # [B, To, A]
            ce_opp = torch.nn.functional.cross_entropy(
                opp_sel.reshape(-1, A_opp),
                opp_targets.view(-1) if opp_targets is not None else torch.full((B*To,), -100, device=device, dtype=torch.long),
                ignore_index=-100, reduction="none"
            ).view(B, To)

            if opp_have_label is not None:
                w = opp_have_label.to(ce_opp.dtype)
                if w.sum() > 0:
                    opp_loss = (ce_opp * w).sum() / w.sum().clamp_min(1.0)
                    with torch.no_grad():
                        pred = opp_sel.argmax(dim=-1)
                        corr = ((pred == opp_targets) & opp_have_label).sum().to(torch.float32)
                        opp_acc = corr / opp_have_label.sum().clamp_min(1).to(torch.float32)

    if AUX_OPP_WEIGHT > 0.0:
        total = total + AUX_OPP_WEIGHT * opp_loss
    metrics["opp_loss"]        = opp_loss.detach()
    metrics["opp_action_acc"]  = opp_acc.detach()

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
        if L > L_tgt:  return v[:, :L_tgt, ...]
        pad_len = L_tgt - L
        pad_shape = list(v.shape); pad_shape[1] = pad_len
        z = torch.zeros(pad_shape, dtype=v.dtype, device=v.device)
        return torch.cat([v, z], dim=1)

    # -------- build batched model inputs (time-major tensors only) --------
    # --- FIX: ROBUST KEY HANDLING ---
    EXPECTED_MI_KEYS = {
        "obs_sequence", "action_sequence", "agent_types",
        "positions", "action_masks"
    }

    mi_batch: Dict[str, torch.Tensor] = {}
    for k in sorted(list(EXPECTED_MI_KEYS)):
        vs = [ep["model_input"].get(k) for ep in episodes]

        valid_vs = [v for v in vs if v is not None and torch.is_tensor(v) and v.dim() >= 2]
        if not valid_vs:
            continue

        if len(valid_vs) != len(vs):
            print(f"Warning: Key '{k}' missing in some episodes, skipping for this batch.")
            continue

        padded = [_pad_trim(v, L_pad) for v in vs]
        cat = torch.cat(padded, dim=0).contiguous()
        if pin_memory:
            cat = cat.pin_memory()
        mi_batch[k] = cat
    # --- END FIX ---

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
    our_pos_lists: List[torch.Tensor] = []
    opp_pos_lists: List[torch.Tensor] = []
    for b in range(B):
        Lb = int(valid_lengths[b].item())
        at = agent_types[b, :Lb].detach().cpu().numpy()  # slice to true length
        our_pos_lists.append(torch.from_numpy((at == 0).nonzero()[0]).long())
        opp_pos_lists.append(torch.from_numpy((at != 0).nonzero()[0]).long())

    T  = max((int(x.numel()) for x in our_pos_lists), default=0)
    To = max((int(x.numel()) for x in opp_pos_lists), default=0)

    opponent_counts: List[int] = []
    for ep in episodes:
        player_labels = tuple(ep.get("player_labels", ()))
        training_seat = ep.get("training_agent_seat", -1)
        count = sum(1 for seat_idx in range(len(player_labels)) if seat_idx != training_seat)
        opponent_counts.append(count)
    num_opponents = max(opponent_counts + [0])

    # -------- allocate supervision tensors (CPU) --------
    def _pm(x: torch.Tensor) -> torch.Tensor:
        return x.pin_memory() if pin_memory else x

    our_idx    = _pm(torch.zeros((B, T),  dtype=torch.long))
    our_mask   = _pm(torch.zeros((B, T),  dtype=torch.bool))
    actions    = _pm(torch.full((B, T), IGN, dtype=torch.long))
    old_logp   = _pm(torch.zeros((B, T),  dtype=torch.float32))
    rewards    = _pm(torch.zeros((B, T),  dtype=torch.float32))
    pen_used   = _pm(torch.zeros((B, T),  dtype=torch.long))

    # Opponent action supervision (unchanged)
    opp_idx        = _pm(torch.zeros((B, To),  dtype=torch.long))
    opp_targets    = _pm(torch.full((B, To), IGN, dtype=torch.long))
    opp_have_label = _pm(torch.zeros((B, To),  dtype=torch.bool))

    opp_labels_by_seat = _pm(torch.full((B, num_opponents), IGN, dtype=torch.long))
    opp_seat_ids = _pm(torch.full((B, num_opponents), -1, dtype=torch.long))

    # -------- fill from episodes (only real steps) --------
    for b, ep in enumerate(episodes):
        # ===== OUR timeline (unchanged) =====
        our_pos = our_pos_lists[b]
        K = int(our_pos.numel())
        our_ep_idx = [i for i, seat in enumerate(ep["agent_id"]) if seat == ep["training_agent_seat"]]

        for t_local in range(min(T, K)):
            if t_local >= len(our_ep_idx):
                break
            step_ep = our_ep_idx[t_local]
            lp = ep["log_prob"][step_ep] if step_ep < len(ep["log_prob"]) else None
            if lp is None:
                continue

            our_mask[b, t_local] = True
            our_idx[b, t_local]  = our_pos[t_local]

            a  = ep["our_action"][step_ep] if step_ep < len(ep["our_action"]) else None
            rw = ep["reward"][step_ep]     if step_ep < len(ep["reward"])     else 0.0
            pu = ep["penalties_used"][step_ep] if step_ep < len(ep["penalties_used"]) else 0

            if a is not None:
                actions[b, t_local] = int(a)
            old_logp[b, t_local] = float(lp)
            rewards[b, t_local]  = float(rw)
            pen_used[b, t_local] = int(pu)

        # ===== OPP timeline =====
        opp_pos = opp_pos_lists[b]
        M = int(opp_pos.numel())
        M_fill = min(To, M)
        if M_fill > 0:
            opp_idx[b, :M_fill] = opp_pos[:M_fill]

            # Episode metadata we already saved
            player_labels = tuple(ep.get("player_labels", ()))  # absolute seat -> label
            agent_id_seq  = ep["agent_id"]                      # per-step absolute seat index
            training_seat = ep.get("training_agent_seat", -1)

            # Record per-seat labels for visualization later
            opp_entries: List[Tuple[int, int]] = []
            for seat_idx, lbl in enumerate(player_labels):
                if seat_idx == training_seat:
                    continue
                if lbl is None:
                    opp_entries.append((seat_idx, IGN))
                else:
                    opp_entries.append((seat_idx, int(lbl)))
            for j, (seat_idx, lbl) in enumerate(opp_entries[:num_opponents]):
                opp_seat_ids[b, j] = seat_idx
                if lbl != IGN:
                    opp_labels_by_seat[b, j] = lbl

            # Indices of opponent steps in episode timeline
            opp_ep_idx = [i for i, seat in enumerate(agent_id_seq) if seat != training_seat]

            for t_local in range(M_fill):
                if t_local >= len(opp_ep_idx):
                    break
                step_ep = opp_ep_idx[t_local]

                # Opponent action supervision (unchanged)
                tgt = ep.get("opp_target_action", [None]*len(agent_id_seq))[step_ep]
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
        "opp_idx":        opp_idx,
        "opp_targets":    opp_targets,
        "opp_have_label": opp_have_label,
        "opp_labels_by_seat": opp_labels_by_seat,
        "opp_seat_ids": opp_seat_ids,
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
    }
    return out
