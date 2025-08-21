#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_attn_collect.py (v2.2-viz)
- Logs raw tensors as before
- Logs model-input row (mi_row)
- ALSO logs Query and Key vectors for Neuron View.
"""

import math
import argparse, json, os, time, random
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.env.liars_deck_env_core import LiarsDeckEnv
from src import config

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

if config.NUM_PLAYERS == 3:
    from src.model.autoregressive_model_full import AutoregressiveGameModelFull as aimodel
else:
    # This is the model we are instrumenting
    from src.model.ppo_autoregressive_model import PPOAutoregressiveModel as aimodel

BELIEF_LABELS: Dict[str, int] = {
    "GreedyCardSpammer": 1, "StrategicChallenger": 4, "TableNonTableAgent": 6, "Classic": 0,
    "TableFirstConservativeChallenger": 5, "SelectiveTableConservativeChallenger": 3, "RandomAgent": 2,
}

def _instantiate_hardcoded(hc_class, hc_name: str, env_id: str, device: torch.device):
    try: agent_index = int(env_id.split('_')[-1])
    except Exception: agent_index = 1
    try: inst = hc_class(hc_name, config.NUM_PLAYERS, agent_index)
    except TypeError: inst = hc_class(hc_name)
    player_id = f"Hardcoded_{hc_name}"
    return HardcodedAgentWrapper(inst, device, player_id)

def build_players_for_game(device: torch.device, learner_agent: "VizAgent"):
    players: Dict[str, Any] = {"player_0": learner_agent}
    opponent_label_map: Dict[str, int] = {}
    HC_POOL = [
        ("Classic", Classic), ("GreedyCardSpammer", GreedyCardSpammer), ("RandomAgent", RandomAgent),
        ("SelectiveTableConservativeChallenger", SelectiveTableConservativeChallenger),
        ("StrategicChallenger", StrategicChallenger), ("TableFirstConservativeChallenger", TableFirstConservativeChallenger),
        ("TableNonTableAgent", TableNonTableAgent),
    ]
    n_needed = getattr(config, "NUM_PLAYERS", 4) - 1
    picks = random.sample(HC_POOL, n_needed)
    for i, (name, cls) in enumerate(picks, start=1):
        env_id = f"player_{i}"
        agent = _instantiate_hardcoded(cls, name, env_id, device)
        players[env_id] = agent
        opponent_label_map[agent.get_player_id()] = BELIEF_LABELS[name]
    return players, opponent_label_map

def encoder_forward_with_viz_data(encoder: nn.TransformerEncoder, src: torch.Tensor,
                                  attn_mask: torch.Tensor = None,
                                  key_padding_mask: torch.Tensor = None):
    """
    A custom TransformerEncoder forward pass that intercepts Q, K, and Attention.
    This version manually implements attention to get the weight matrix.
    """
    output = src
    viz_data_by_layer = []

    for mod in encoder.layers:
        sa = mod.self_attn
        
        # --- Manual Multi-Head Attention block to capture Q, K, and weights ---
        q, k, v = F.linear(output, sa.in_proj_weight, sa.in_proj_bias).chunk(3, dim=-1)
        
        q = q.view(output.shape[0], output.shape[1], sa.num_heads, sa.head_dim).transpose(1, 2)
        k = k.view(output.shape[0], output.shape[1], sa.num_heads, sa.head_dim).transpose(1, 2)
        v = v.view(output.shape[0], output.shape[1], sa.num_heads, sa.head_dim).transpose(1, 2)

        # 1. Scaled dot-product
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(sa.head_dim)

        # 2. Apply causal attention mask
        if attn_mask is not None:
            # Ensure mask has correct shape and type for broadcasting
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0) # [T, T] -> [1, T, T]
            attn_scores = attn_scores.masked_fill(attn_mask == True, float('-inf'))

        # 3. Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 4. Apply weights to V
        attn_output_raw = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output_raw = attn_output_raw.transpose(1, 2).contiguous().view(output.shape)
        self_attn_output = sa.out_proj(attn_output_raw)
        
        # --- Store visualization data for this layer ---
        viz_data_by_layer.append({'attn': attn_weights, 'q': q, 'k': k})

        # --- Remainder of the TransformerLayer logic (Post-Norm) ---
        # 1. Residual connection and dropout
        output = output + mod.dropout1(self_attn_output)
        # 2. Normalization
        output = mod.norm1(output)
        # 3. Feed-forward network
        ff_output = mod.linear2(mod.dropout(mod.activation(mod.linear1(output))))
        # 4. Residual connection and dropout
        output = output + mod.dropout2(ff_output)
        # 5. Normalization
        output = mod.norm2(output)

    if encoder.norm is not None:
        output = encoder.norm(output)

    return output, viz_data_by_layer


def extract_viz_data(viz_data_by_layer, batch_idx=0):
    """
    Extracts attention, query, and key vectors for the last token from the raw viz data.
    """
    attn_last_out = {}
    neuron_data_out = {}
    for i, layer_data in enumerate(viz_data_by_layer):
        # Attention to the last token: [H, 1, T] -> [H, T]
        attn_last_out[f"L{i}"] = layer_data['attn'][batch_idx, :, -1, :].detach().cpu()

        # Q vector for the last token: [B, H, T, D_h] -> [H, D_h]
        q_last = layer_data['q'][batch_idx, :, -1, :].detach().cpu()
        # K vectors for ALL tokens: [B, H, T, D_h] -> [H, T, D_h]
        k_all = layer_data['k'][batch_idx, :, :, :].detach().cpu()
        neuron_data_out[f"L{i}"] = {'q_last': q_last, 'k_all': k_all}

    return attn_last_out, neuron_data_out


class VizAgent:
    def __init__(self, device: torch.device, model: aimodel):
        self.device = device
        self.model = model.eval()
        self.obs_dim = model.obs_dim
        self.action_dim = model.action_dim
        self.max_seq_length = model.max_seq_length
        self.env_agent_id_map = None
        self.sequence_history: List[Dict[str, Any]] = []
        self._obs_by_step: Dict[int, Tuple[List[float], List[int]]] = {}

    def reset(self):
        self.sequence_history.clear()
        self.env_agent_id_map = None
        self._obs_by_step.clear()

    def _revealed_token_from_play(self, e):
        cnt = int(e.get("count") or 1)
        base = 0 if e.get("card_category", "table") == "table" else 3
        return base + (cnt - 1)

    def _rebuild_history_from_gh(self, env, me: str):
        gh = list(getattr(env, "game_history", []))
        seq = []
        HIDDEN_MAP = {1: 7, 2: 8, 3: 9}
        for e in gh:
            a_type, actor, step = e.get("action_type"), e.get("player"), int(e.get("step"))
            obs, mask = self._obs_by_step.get(step, ([0.0] * int(self.obs_dim), [0] * int(self.action_dim)))
            action = -1
            if a_type == "Play":
                cnt = int(e.get("count") or 1)
                action = self._revealed_token_from_play(e) if actor == me else HIDDEN_MAP.get(cnt, 7)
            elif a_type == "Challenge":
                action = 6
            if action != -1:
                seq.append({"agent_id_env": actor, "action": action, "observation": obs, "action_mask": mask})
        return seq

    def _prepare_model_input(self, history):
        PAD = 0
        filtered = list(history)
        raw_actions = [step.get("action", PAD) for step in filtered]
        input_actions = [PAD] + raw_actions[:-1]
        current_seq_len = len(filtered)
        max_len = self.max_seq_length
        valid_len = min(current_seq_len, max_len)
        if current_seq_len > max_len:
            filtered, input_actions = filtered[-max_len:], input_actions[-max_len:]

        obs_seq = torch.zeros((1, valid_len, self.obs_dim), dtype=torch.float32, device=self.device)
        action_seq = torch.tensor(input_actions[:valid_len], dtype=torch.long, device=self.device).unsqueeze(0)
        agent_type_seq = torch.ones((1, valid_len), dtype=torch.long, device=self.device)
        pos_seq = torch.arange(valid_len, dtype=torch.long, device=self.device).unsqueeze(0)
        action_mask_seq = torch.zeros((1, valid_len, self.action_dim), dtype=torch.bool, device=self.device)
        padding_mask = torch.zeros(1, valid_len, dtype=torch.bool, device=self.device)

        for i, step in enumerate(filtered[:valid_len]):
            agent_type = self.env_agent_id_map[step["agent_id_env"]]
            agent_type_seq[0, i] = agent_type
            if agent_type == 0:
                obs_seq[0, i] = torch.tensor(step["observation"], dtype=torch.float32, device=self.device)
                action_mask_seq[0, i] = torch.tensor(step.get("action_mask", [0]*self.action_dim), dtype=torch.bool, device=self.device)
        return {"obs_sequence": obs_seq, "action_sequence": action_seq, "agent_types": agent_type_seq,
                "positions": pos_seq, "action_masks": action_mask_seq, "padding_mask": padding_mask,
                "valid_lengths": torch.tensor([valid_len], device=self.device)}

    @torch.inference_mode()
    def act_with_attn(self, env, agent_id_env: str, info):
        if self.env_agent_id_map is None:
            self.env_agent_id_map = {p: i for i, p in enumerate(env.possible_agents)}

        gh = list(getattr(env, "game_history", []))
        next_step = (gh[-1]["step"] + 1) if gh else 1
        obs_curr = env.observe(agent_id_env, newerest=True)[agent_id_env]
        self._obs_by_step[next_step] = (obs_curr, list(info.get("action_mask", [0]*self.action_dim)))

        self.sequence_history = self._rebuild_history_from_gh(env, agent_id_env)
        self.sequence_history.append({"agent_id_env": agent_id_env, "observation": obs_curr, "action_mask": list(info.get("action_mask", [0]*self.action_dim))})

        mi = self._prepare_model_input(self.sequence_history)
        encoded = self.model._encode_inputs(mi["obs_sequence"], mi["action_sequence"], mi["agent_types"], mi["positions"], mi["action_masks"], padding_mask=mi["padding_mask"])
        
        T = encoded.size(1)
        causal = self.model.causal_bool_mask_full[:T, :T]
        
        # *** USE NEW FORWARD FUNCTION ***
        enc_out, viz_data = encoder_forward_with_viz_data(self.model.transformer, encoded, causal, mi["padding_mask"])
        
        belief_hidden = F.relu(self.model.belief_fc(enc_out))
        fused = torch.cat([enc_out, belief_hidden], dim=-1)
        action_logits, opp_logits, state_values = self.model.action_head(fused), self.model.opp_action_head(fused), self.model.value_head(fused)

        last_idx = mi["valid_lengths"][0].item() - 1
        logits_t = action_logits[0, last_idx]
        mask_t = torch.tensor(info["action_mask"], dtype=torch.bool, device=self.device)
        masked_logits = logits_t.masked_fill(~mask_t, float("-inf"))
        action = int(torch.argmax(masked_logits).item())

        # *** EXTRACT ALL VIZ DATA (ATTN, Q, K) ***
        attn_last, neuron_data = extract_viz_data(viz_data, 0)

        mi_row = {"t": int(last_idx), "agent_type": int(mi["agent_types"][0, last_idx].item()),
                  "prev_action_token": int(mi["action_sequence"][0, last_idx].item()),
                  "obs": mi["obs_sequence"][0, last_idx].detach().cpu().tolist(),
                  "mask": mi["action_masks"][0, last_idx].int().detach().cpu().tolist()}

        b0, b1 = self.model.belief_head_op0(belief_hidden)[0, last_idx], self.model.belief_head_op1(belief_hidden)[0, last_idx]
        b2 = self.model.belief_head_op2(belief_hidden)[0, last_idx] if config.NUM_PLAYERS == 4 else torch.zeros_like(b1)
        
        return action, attn_last, neuron_data, masked_logits.detach().cpu(), state_values[0, last_idx, 0].detach().cpu(), torch.stack([b0, b1, b2]).detach().cpu(), mi_row


def run_and_log(ckpt_path: str, episodes: int, save_dir: str, device_str: str = None):
    # (This function remains largely the same, but the call to act_with_attn and data saving is updated)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(save_dir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd   = ckpt.get("model_state_dict", ckpt)
    model_cls = aimodel
    model_args = {
        "obs_dim": 9 if config.NUM_PLAYERS != 3 else 4, "action_dim": 7, "belief_dim": 64 if config.NUM_PLAYERS != 3 else 10,
        "hidden_dim": 256, "num_heads": 4, "num_layers": 2, "max_seq_length": 320 if config.NUM_PLAYERS != 3 else 100,
        "num_agent_types": 4 if config.NUM_PLAYERS != 3 else 3
    }
    model = model_cls(**model_args).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    learner = VizAgent(device, model)
    env = LiarsDeckEnv(num_players=getattr(config, "NUM_PLAYERS", 4))

    for epi in range(episodes):
        env.reset(seed=42+epi)
        learner.reset()
        players, _ = build_players_for_game(device, learner)
        ep = {"meta": {"time": int(time.time()), "seed": epi}, "steps": []}

        active = True
        with torch.inference_mode():
            while active and env.agent_selection is not None:
                aid = env.agent_selection
                if env.terminations.get(aid, False) or env.truncations.get(aid, False):
                    env.step(None); active = bool(env.agents); continue
                env.observe(aid)
                info = env.infos.get(aid, {})
                if aid == "player_0":
                    
                    act, attn_last, neuron_data, logits, value, beliefs, mi_row = learner.act_with_attn(env, aid, info)
                    env.step(int(act))
                    obs_curr = learner._obs_by_step[max(learner._obs_by_step.keys())][0]
                    
                    # Convert neuron_data tensors to lists for JSON serialization
                    neuron_data_serializable = {
                        ln: {'q_last': v['q_last'].tolist(), 'k_all': v['k_all'].tolist()}
                        for ln, v in neuron_data.items()
                    }

                    ep["steps"].append({
                        "t": len(ep["steps"]), "actor": 0, "action_token": int(act), "mi_row": mi_row,
                        "obs": list(map(float, obs_curr)), "mask": list(map(int, info.get("action_mask", [0]*model.action_dim))),
                        "logits": logits.tolist(), "value": float(value.item()), "beliefs": beliefs.softmax(dim=-1).tolist(),
                        "attn_last": {k: v.tolist() for k, v in attn_last.items()},
                        "neuron_data": neuron_data_serializable # <-- NEW DATA
                    })
                else:
                    agent = players[aid]
                    opp_action = agent.get_action(env, aid, env.observe(aid), info)
                    env.step(int(opp_action))
                    prev_action = ep["steps"][-1]["action_token"] if ep["steps"] else 0
                    agent_type = env.possible_agents.index(aid)
                    ep["steps"].append({
                        "t": len(ep["steps"]), "actor": agent_type, "action_token": int(opp_action),
                        "mi_row": {"t": len(ep["steps"]), "agent_type": agent_type, "prev_action_token": int(prev_action),
                                   "obs": [0.0]*int(model.obs_dim), "mask": [0]*int(model.action_dim)},
                        "obs": [], "mask": [], "logits": [], "value": None, "beliefs": [], "attn_last": {}, "neuron_data": {}
                    })
                active = bool(env.agents)

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(save_dir, f"episode_{ts}_{epi}.json")
        with open(out_path, "w") as f:
            json.dump(ep, f)
        print(f"[saved] {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--save-dir", type=str, default="attn_logs")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    run_and_log(args.ckpt, args.episodes, args.save_dir, args.device)

if __name__ == "__main__":
    main()