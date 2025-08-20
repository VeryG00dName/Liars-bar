"""
run_attn_collect.py
Play LiarsDeck games with your PPOAutoregressiveModel and dump *raw* tensors + per-layer/head
last-token attention per step for an expert-mode viewer.

Usage:
  python run_attn_collect.py \
      --ckpt checkpoints/autoreg_xxx/autoreg_model_final.pth \
      --episodes 5 \
      --save-dir attn_logs

Assumptions:
- Your codebase layout matches the training scripts (imports below).
- Hardcoded agents & env exist as in your repo.
- This script does *not* modify your model weights; it temporarily runs the
  encoder layers with need_weights=True to extract attention.

Outputs:
- JSON files: save_dir/episode_YYYYmmdd_HHMMSS_<seed>.json
"""

import argparse, json, os, time, math, random
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Project imports (same as train_ppo_autoregressive.py) ----
from src.env.liars_deck_env_core import LiarsDeckEnv
from src import config

from src.model.ppo_autoregressive_model import PPOAutoregressiveModel

from src.model.hard_coded_agents import (
    RandomAgent,
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    SelectiveTableConservativeChallenger,
    TableNonTableAgent,
    StrategicChallenger,
    Classic,
)

from src.agents.hardcoded_agent_wrapper import HardcodedAgentWrapper


# -------------------------
# Belief labels (copied from training)
# -------------------------
BELIEF_LABELS: Dict[str, int] = {
    "GreedyCardSpammer": 1,
    "StrategicChallenger": 4,
    "TableNonTableAgent": 6,
    "Classic": 0,
    "TableFirstConservativeChallenger": 5,
    "SelectiveTableConservativeChallenger": 3,
    "RandomAgent": 2,
}


# -------------------------
# Opponents builder (trimmed from training)
# -------------------------
def _instantiate_hardcoded(hc_class, hc_name: str, env_id: str, device: torch.device):
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


def build_players_for_game(
    device: torch.device,
    learner_agent: "VizAgent",
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Build env_id -> agent for a single game:
      player_0 = learner; others = random sample of hardcoded.
    Returns (players, opponent_label_map) used for belief targets.
    """
    players: Dict[str, Any] = {"player_0": learner_agent}
    opponent_label_map: Dict[str, int] = {}

    HC_POOL: List[Tuple[str, Any]] = [
        ("Classic", Classic),
        ("GreedyCardSpammer", GreedyCardSpammer),
        ("RandomAgent", RandomAgent),
        ("SelectiveTableConservativeChallenger", SelectiveTableConservativeChallenger),
        ("StrategicChallenger", StrategicChallenger),
        ("TableFirstConservativeChallenger", TableFirstConservativeChallenger),
        ("TableNonTableAgent", TableNonTableAgent),
    ]

    n_needed = getattr(config, "NUM_PLAYERS", 4) - 1
    picks = random.sample(HC_POOL, n_needed)

    for i, (name, cls) in enumerate(picks, start=1):
        env_id = f"player_{i}"
        agent = _instantiate_hardcoded(cls, name, env_id, device)
        players[env_id] = agent

        if name not in BELIEF_LABELS:
            raise KeyError(f"Hardcoded agent '{name}' missing BELIEF_LABELS mapping.")
        opponent_label_map[agent.get_player_id()] = BELIEF_LABELS[name]

    return players, opponent_label_map


# -------------------------
# Attention-enabled forward utilities
# -------------------------
def encoder_forward_with_attn(encoder: nn.TransformerEncoder, src: torch.Tensor,
                              attn_mask: torch.Tensor = None,
                              key_padding_mask: torch.Tensor = None):
    """
    Re-implements nn.TransformerEncoder forward pass so we can force
    self_attn(..., need_weights=True, average_attn_weights=False) per layer.
    Returns: (output, attn_list) with attn_list = [B,H,T,T] per layer.
    """
    output = src
    attn_list = []
    for layer in encoder.layers:
        # norm_first=False assumption (matches your instantiation)
        attn_out, attn_weights = layer.self_attn(
            output, output, output,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )  # attn_out: [B,T,C], attn_weights: [B,H,T,T]

        output = output + layer.dropout1(attn_out)
        output = layer.norm1(output)

        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(output))))
        output = output + layer.dropout2(ff)
        output = layer.norm2(output)

        attn_list.append(attn_weights)

    if encoder.norm is not None:
        output = encoder.norm(output)
    return output, attn_list


def extract_last_token_attn(attn_list: List[torch.Tensor], batch_idx: int = 0) -> Dict[str, torch.Tensor]:
    """
    Given per-layer attention [B,H,T,T], take the last query row (T-1,:) for each layer.
    Returns {f"L{idx}": [H, T]}.
    """
    out = {}
    for i, A in enumerate(attn_list):
        out[f"L{i}"] = A[batch_idx, :, -1, :].detach().cpu()  # [H,T]
    return out


# -------------------------
# Learner agent that produces actions + attention
# -------------------------
class VizAgent:
    def __init__(self, device: torch.device, model: PPOAutoregressiveModel):
        self.device = device
        self.model = model.eval()
        # dims
        self.obs_dim = model.obs_dim
        self.action_dim = model.action_dim
        self.max_seq_length = model.max_seq_length
        # runtime state
        self.env_agent_id_map: Dict[str, int] = None
        self.sequence_history: List[Dict[str, Any]] = []
        self._obs_by_step: Dict[int, Tuple[List[float], List[int]]] = {}
        self._last_seen_step = -1

    def reset(self):
        self.sequence_history.clear()
        self.env_agent_id_map = None
        self._obs_by_step.clear()
        self._last_seen_step = -1

    def _revealed_token_from_play(self, e: Dict[str, Any]) -> int:
        cnt = int(e.get("count") or 1)
        base = 0 if e.get("card_category", "table") == "table" else 3
        return base + (cnt - 1)

    def _rebuild_history_from_gh(self, env, me: str) -> List[Dict[str, Any]]:
        gh = list(getattr(env, "game_history", []))
        seq: List[Dict[str, Any]] = []
        HIDDEN_MAP = {1: 7, 2: 8, 3: 9}

        for e in gh:
            a_type = e.get("action_type")
            actor  = e.get("player")
            step   = int(e.get("step"))

            if actor == me and step in self._obs_by_step:
                obs, mask = self._obs_by_step[step]
            else:
                obs  = [0.0] * int(self.obs_dim)
                mask = [0]   * int(self.action_dim)

            if a_type == "Play":
                cnt = int(e.get("count") or 1)
                if actor == me:
                    action = self._revealed_token_from_play(e)
                else:
                    action = HIDDEN_MAP.get(cnt, 7)
                seq.append({
                    "agent_id_env": actor,
                    "action": action,
                    "observation": obs,
                    "action_mask": mask if actor == me else [0]*int(self.action_dim),
                })
            elif a_type == "Challenge":
                seq.append({
                    "agent_id_env": actor,
                    "action": 6,
                    "observation": obs,
                    "action_mask": mask if actor == me else [0]*int(self.action_dim),
                })
        return seq

    def _prepare_model_input(self, history: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        PAD = 0
        filtered = list(history)
        # left-shift actions
        raw_actions = [step.get("action", PAD) for step in filtered]
        input_actions = [PAD] + raw_actions[:-1]

        current_seq_len = len(filtered)
        max_len = self.max_seq_length
        valid_len = min(current_seq_len, max_len)
        if current_seq_len > max_len:
            filtered      = filtered[-max_len:]
            input_actions = input_actions[-max_len:]

        obs_seq         = torch.zeros((1, valid_len, self.obs_dim), dtype=torch.float32, device=self.device)
        action_seq      = torch.tensor(input_actions[:valid_len], dtype=torch.long, device=self.device).unsqueeze(0)
        agent_type_seq  = torch.ones ((1, valid_len), dtype=torch.long, device=self.device)  # default: opponent
        pos_seq         = torch.arange(valid_len, dtype=torch.long, device=self.device).unsqueeze(0)
        action_mask_seq = torch.zeros((1, valid_len, self.action_dim), dtype=torch.bool, device=self.device)
        padding_mask    = torch.zeros(1, valid_len, dtype=torch.bool, device=self.device)

        for i, step in enumerate(filtered[:valid_len]):
            agent_type = self.env_agent_id_map[step["agent_id_env"]]
            agent_type_seq[0, i] = agent_type
            if agent_type == 0:
                obs_seq[0, i] = torch.tensor(step["observation"], dtype=torch.float32, device=self.device)
                action_mask_seq[0, i] = torch.tensor(step.get("action_mask", [0]*self.action_dim), dtype=torch.bool, device=self.device)

        return {
            "obs_sequence":   obs_seq,
            "action_sequence":action_seq,
            "agent_types":    agent_type_seq,
            "positions":      pos_seq,
            "action_masks":   action_mask_seq,
            "padding_mask":   padding_mask,
            "valid_lengths":  torch.tensor([valid_len], device=self.device),
        }

    @torch.inference_mode()
    def act_with_attn(self, env, agent_id_env: str, info: Dict[str, Any]) -> Tuple[int, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        # map seats once
        if self.env_agent_id_map is None:
            if agent_id_env == "player_0":
                self.env_agent_id_map = {"player_0": 0, "player_1": 1, "player_2": 2, "player_3": 3}
                # if 4p, player_3 will map to 2 as well (share embedding) – extend if you have num_agent_types=4
                if getattr(config, "NUM_PLAYERS", 4) >= 4:
                    self.env_agent_id_map["player_3"] = 3
            else:
                others = [a for a in env.possible_agents if a != agent_id_env]
                self.env_agent_id_map = {agent_id_env: 0}
                for i, opp in enumerate(others, start=1):
                    if i <= 2:
                        self.env_agent_id_map[opp] = i

        # detect next gh step for caching
        gh = list(getattr(env, "game_history", []))
        next_step = (gh[-1]["step"] + 1) if gh else 1
        obs_curr = env.observe(agent_id_env, newerest=True)[agent_id_env]
        self._obs_by_step[next_step] = (
            obs_curr,
            list(info.get("action_mask", [0]*self.action_dim))
        )

        # rebuild history and append current row (no action yet)
        self.sequence_history = self._rebuild_history_from_gh(env, agent_id_env)
        self.sequence_history.append({
            "agent_id_env": agent_id_env,
            "observation": obs_curr,
            "action_mask": list(info.get("action_mask", [0]*self.action_dim)),
        })

        mi = self._prepare_model_input(self.sequence_history)
        # --- custom forward to capture attention ---
        encoded = self.model._encode_inputs(
            mi["obs_sequence"], mi["action_sequence"],
            mi["agent_types"], mi["positions"], mi["action_masks"],
            padding_mask=mi["padding_mask"]
        )
        T = encoded.size(1)
        causal = self.model.causal_bool_mask_full[:T, :T]
        enc_out, attn_list = encoder_forward_with_attn(self.model.transformer, encoded, causal, mi["padding_mask"])
        belief_hidden  = F.relu(self.model.belief_fc(enc_out))
        fused = torch.cat([enc_out, belief_hidden], dim=-1)

        action_logits = self.model.action_head(fused)
        opp_logits    = self.model.opp_action_head(fused)
        state_values  = self.model.value_head(fused)

        last_idx = mi["valid_lengths"][0].item() - 1
        logits_t = action_logits[0, last_idx]
        mask_t = torch.tensor(info["action_mask"], dtype=torch.bool, device=self.device)
        masked_logits = logits_t.masked_fill(~mask_t, float("-inf"))
        action = int(torch.argmax(masked_logits).item())

        # last-token attention per layer
        attn_last = extract_last_token_attn(attn_list, 0)  # {L*: [H,T]}

        # For convenience in logs, return belief top-1 (optional)
        b0 = self.model.belief_head_op0(belief_hidden)[0, last_idx]
        b1 = self.model.belief_head_op1(belief_hidden)[0, last_idx]
        b2 = self.model.belief_head_op2(belief_hidden)[0, last_idx]

        return action, attn_last, masked_logits.detach().cpu(), state_values[0, last_idx, 0].detach().cpu(), torch.stack([b0, b1, b2]).detach().cpu()


# -------------------------
# Episode runner + JSON logger
# -------------------------
def run_and_log(ckpt_path: str, episodes: int, save_dir: str, device_str: str = None):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(save_dir, exist_ok=True)

    # ---- load model ----
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd   = ckpt.get("model_state_dict", ckpt)  # allow raw sd too

    # NOTE: adjust dims if needed; these should match your checkpoint
    model = PPOAutoregressiveModel(
        obs_dim=9,
        action_dim=7,
        belief_dim=64,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        max_seq_length=320,
        num_agent_types=4
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    learner = VizAgent(device, model)

    # ---- env ----
    env = LiarsDeckEnv(num_players=4)

    for epi in range(episodes):
        # seed for reproducibility
        env.reset(seed=epi)
        for a in getattr(env, "agents", []):
            pass  # wrappers reset internally on first use if needed
        learner.reset()

        players, _ = build_players_for_game(device, learner)

        # episode record
        ep = {"meta": {"time": int(time.time()), "seed": epi},
              "steps": []}

        active = True
        with torch.inference_mode():
            while active and env.agent_selection is not None:
                aid = env.agent_selection

                # If terminated/truncated, advance None
                if env.terminations.get(aid, False) or env.truncations.get(aid, False):
                    env.step(None)
                    active = bool(env.agents)
                    continue

                info = env.infos.get(aid, {})
                if aid == "player_0":
                    # our turn
                    act, attn_last, masked_logits, value, belief_vecs = learner.act_with_attn(env, aid, info)

                    # step
                    env.step(int(act))

                    # observation snapshot (raw)
                    obs_curr = learner._obs_by_step[max(learner._obs_by_step.keys())][0]  # just stored

                    ep["steps"].append({
                        "t": len(ep["steps"]),
                        "actor": 0,
                        "action_token": int(act),
                        "obs": list(map(float, obs_curr)),
                        "mask": list(map(int, info.get("action_mask", [0]*model.action_dim))),
                        "logits": masked_logits.tolist(),
                        "value": float(value.item() if torch.is_tensor(value) else value),
                        "beliefs": belief_vecs.softmax(dim=-1).tolist(),  # 3 x belief_dim
                        "attn_last": {k: v.tolist() for k, v in attn_last.items()},
                    })
                else:
                    # opponents
                    agent = players[aid]
                    observation = env.observe(aid)
                    opp_action = agent.get_action(env, aid, observation, info)
                    env.step(int(opp_action))

                    ep["steps"].append({
                        "t": len(ep["steps"]),
                        "actor": 1 if aid.endswith("1") else 2,
                        "action_token": int(opp_action),
                        "obs": [],
                        "mask": [],
                        "logits": [],
                        "value": None,
                        "beliefs": [],
                        "attn_last": {},  # we only care about our decision steps for now
                    })

                active = bool(env.agents)

        # save json
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(save_dir, f"episode_{ts}_{epi}.json")
        with open(out_path, "w") as f:
            json.dump(ep, f)
        print(f"[saved] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pth (contains model_state_dict).")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--save-dir", type=str, default="attn_logs")
    ap.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu' (default auto)")
    args = ap.parse_args()
    run_and_log(args.ckpt, args.episodes, args.save_dir, args.device)


if __name__ == "__main__":
    main()