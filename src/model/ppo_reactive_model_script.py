# src/model/ppo_reactive_model_script.py
import torch
import torch.nn as nn
from typing import Optional, Tuple

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

        # === Transformer Backbone ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        # === Output heads ===
        self.action_head        = nn.Linear(hidden_dim, action_dim)
        self.reward_stream_head = nn.Linear(hidden_dim, 1)
        self.win_prob_head      = nn.Linear(hidden_dim, 1)
        self.opp_action_head    = nn.Linear(hidden_dim, action_dim)

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
        
        transformer_output = self.transformer(
            encoded_inputs,
            mask=causal_mask,
            src_key_padding_mask=key_padding,
            is_causal=True
        )

        action_logits = self.action_head(transformer_output)
        state_values = self.reward_stream_head(transformer_output)
        win_logits = self.win_prob_head(transformer_output)
        opp_logits = self.opp_action_head(transformer_output)

        return action_logits, opp_logits, state_values, win_logits