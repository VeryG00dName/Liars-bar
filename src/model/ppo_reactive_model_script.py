# src/model/ppo_reactive_model_script.py
import copy
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

class PPOReactiveModelScript(nn.Module):
    """
    A simplified, monolithic autoregressive model for PPO that operates
    reactively based on the full game history.

    This version is specifically designed to be compatible with torch.jit.script
    for deployment as a fast historical agent in C++.
    
    Action ID Semantics (factorization):
      action_sequence (int 0..10) -> decomposed into:
        act_kind   ∈ {PAD=0, PLAY=1, CHALLENGE=2}
        count      ∈ {NONE=0, 1=1, 2=2, 3=3, PAD=4}
        table_flag ∈ {NA=0, TABLE=1, NON_TABLE=2, PAD=3}
    """
    def __init__(self,
                 obs_dim: int,
                 action_dim: int = 7,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout_rate: float = 0.1,
                 max_seq_length: int = 480,
                 num_agent_types: int = 4):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        self.count_pad = 4  # Matches lut_count
        self.tflag_pad = 3  # Matches lut_table_flag
        
        self.register_buffer(
            "causal_bool_mask_full",
            torch.triu(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool), 1)
        )

        # === Input Encoders ===
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.act_kind_embedding = nn.Embedding(3, hidden_dim, padding_idx=0)
        self.count_embedding = nn.Embedding(5, hidden_dim, padding_idx=self.count_pad)
        self.table_flag_embedding = nn.Embedding(4, hidden_dim, padding_idx=self.tflag_pad)
        self.agent_embedding = nn.Embedding(num_agent_types, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_dim)
        
        # === Gating Layers (Independent) ===
        def make_gate_net(h_dim: int):
            return nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.Tanh(),
                nn.Linear(h_dim, h_dim),
                nn.Sigmoid()
            )

        self.gate_obs = make_gate_net(hidden_dim)
        self.gate_action = make_gate_net(hidden_dim)
        self.gate_agent = make_gate_net(hidden_dim)
        self.gate_position = make_gate_net(hidden_dim)

        # === Factorization Look-up Tables ===
        self.register_buffer("lut_act_kind",   torch.tensor([1,1,1,1,1,1,2,1,1,1,0], dtype=torch.long))
        self.register_buffer("lut_count",      torch.tensor([1,2,3,1,2,3,0,1,2,3,4], dtype=torch.long))
        self.register_buffer("lut_table_flag", torch.tensor([1,1,1,2,2,2,0,0,0,0,3], dtype=torch.long))

        expert_ffn_dim = hidden_dim * 2
        self.num_experts = 8
        self.top_k = 2

        class ScriptTopKMoE(nn.Module):
            def __init__(self, hidden_dim: int, expert_dim: int, num_experts: int, top_k: int, dropout: float):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.num_experts = num_experts
                self.top_k = top_k
                self.gate = nn.Linear(hidden_dim, num_experts)
                experts: List[nn.Module] = []
                for _ in range(num_experts):
                    experts.append(nn.Sequential(
                        nn.Linear(hidden_dim, expert_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(expert_dim, hidden_dim),
                        nn.Dropout(dropout),
                    ))
                self.experts = nn.ModuleList(experts)

            def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                gate_logits = self.gate(x)
                gate_probs = torch.softmax(gate_logits, dim=-1)
                top_scores, top_indices = torch.topk(gate_probs, self.top_k, dim=-1)
                top_weights = top_scores / top_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
                gather_idx = top_indices.unsqueeze(-1).expand(*top_indices.shape, self.hidden_dim)
                top_outputs = torch.gather(expert_outputs, 2, gather_idx)
                combined = (top_outputs * top_weights.unsqueeze(-1)).sum(dim=2)
                routing = {
                    "gate_logits": gate_logits,
                    "topk_indices": top_indices,
                    "topk_scores": top_weights,
                }
                return combined, routing

        class ScriptMoEEncoderLayer(nn.Module):
            def __init__(self, hidden_dim: int, nhead: int, dropout: float, num_experts: int, top_k: int, expert_dim: int):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
                self.dropout1 = nn.Dropout(dropout)
                self.dropout2 = nn.Dropout(dropout)
                self.norm1 = nn.LayerNorm(hidden_dim)
                self.norm2 = nn.LayerNorm(hidden_dim)
                self.moe = ScriptTopKMoE(hidden_dim, expert_dim, num_experts, top_k, dropout)

            def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor], src_key_padding_mask: Optional[torch.Tensor]):
                attn_output, _ = self.self_attn(
                    src,
                    src,
                    src,
                    attn_mask=src_mask,
                    key_padding_mask=src_key_padding_mask,
                    need_weights=False,
                )
                src = self.norm1(src + self.dropout1(attn_output))
                moe_output, routing = self.moe(src)
                src = self.norm2(src + self.dropout2(moe_output))
                return src, routing

        base_layer = ScriptMoEEncoderLayer(hidden_dim, num_heads, dropout_rate, self.num_experts, self.top_k, expert_ffn_dim)
        layers: List[nn.Module] = []
        for _ in range(num_layers):
            layers.append(copy.deepcopy(base_layer))
        self.transformer_layers = nn.ModuleList(layers)
        self.transformer_norm = nn.LayerNorm(hidden_dim)

        def make_head(out_dim: int) -> nn.ModuleList:
            return nn.ModuleList([nn.Linear(hidden_dim, out_dim) for _ in range(self.num_experts)])

        self.action_heads = make_head(action_dim)
        self.reward_stream_heads = make_head(1)
        self.win_prob_heads = make_head(1)
        self.opp_action_heads = make_head(action_dim)

    # -------------------------- utils --------------------------
    def _decompose_actions(self, action_sequence: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = action_sequence.long()
        act_kind, count, tflag = self.lut_act_kind[a], self.lut_count[a], self.lut_table_flag[a]
        if padding_mask is not None:
            act_pad = torch.zeros_like(act_kind)
            count_pad = torch.full_like(count, self.count_pad, dtype=torch.long)
            tflag_pad = torch.full_like(tflag, self.tflag_pad, dtype=torch.long)
            act_kind = torch.where(padding_mask, act_pad,   act_kind)
            count    = torch.where(padding_mask, count_pad, count)
            tflag    = torch.where(padding_mask, tflag_pad, tflag)
        return act_kind, count, tflag

    def _encode_inputs(self, obs_sequence: torch.Tensor, action_sequence: torch.Tensor, agent_types: torch.Tensor, positions: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Get individual embeddings
        obs_embed = self.obs_encoder(obs_sequence)
        act_kind_ids, count_ids, table_flag_ids = self._decompose_actions(action_sequence, padding_mask)
        action_embed = (self.act_kind_embedding(act_kind_ids) +
                        self.count_embedding(count_ids) +
                        self.table_flag_embedding(table_flag_ids))
        agent_embed = self.agent_embedding(agent_types)
        position_embed = self.position_embedding(positions)

        # Compute independent gates for each embedding
        g_obs = self.gate_obs(obs_embed)
        g_action = self.gate_action(action_embed)
        g_agent = self.gate_agent(agent_embed)
        g_position = self.gate_position(position_embed)

        # Apply gates and sum to combine embeddings
        fused = (g_obs * obs_embed +
                    g_action * action_embed +
                    g_agent * agent_embed +
                    g_position * position_embed)
        combined = nn.functional.layer_norm(fused, (self.hidden_dim,))
        return combined

    # -------------------------- forward --------------------------
    def forward(self,
                obs_sequence: torch.Tensor,
                action_sequence: torch.Tensor,
                agent_types: torch.Tensor, 
                positions: torch.Tensor,
                action_masks: torch.Tensor,
                padding_mask: torch.Tensor,
                valid_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        encoded_inputs = self._encode_inputs(obs_sequence, action_sequence, agent_types, positions, padding_mask)

        T = encoded_inputs.size(1)
        causal_mask = self.causal_bool_mask_full[:T, :T]
        if causal_mask.device != encoded_inputs.device:
            causal_mask = causal_mask.to(encoded_inputs.device)
        
        key_padding = padding_mask.to(torch.bool)
        
        routing: Dict[str, torch.Tensor] = {}
        output = encoded_inputs
        for layer in self.transformer_layers:
            output, routing = layer(output, causal_mask, key_padding)
        transformer_output = self.transformer_norm(output)

        final_indices = routing.get("topk_indices")
        final_scores = routing.get("topk_scores")

        def combine_head(heads: nn.ModuleList) -> torch.Tensor:
            all_outputs = torch.stack([head(transformer_output) for head in heads], dim=2)
            if final_indices is None or final_scores is None:
                return all_outputs.mean(dim=2)
            gather_idx = final_indices.unsqueeze(-1).expand(*final_indices.shape, all_outputs.size(-1))
            top_outputs = torch.gather(all_outputs, 2, gather_idx)
            return (top_outputs * final_scores.unsqueeze(-1)).sum(dim=2)

        action_logits = combine_head(self.action_heads)
        state_values = combine_head(self.reward_stream_heads)
        win_logits = combine_head(self.win_prob_heads)
        opp_logits = combine_head(self.opp_action_heads)

        return action_logits, opp_logits, state_values, win_logits