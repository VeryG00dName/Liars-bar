# Create this new file: src/model/belief_oracle.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.ppo_fused_model import FiLMLayer

class BeliefOracle(nn.Module):
    """
    A transformer-based model dedicated to predicting an agent's beliefs about its opponents.
    It takes the same input sequence as the main PPO agent.
    Its output is the predicted belief logits for each of the three opponent slots.
    """
    def __init__(self,
                 obs_dim,
                 belief_dim=64,
                 hidden_dim=256,
                 num_heads=4,
                 num_layers=2,
                 dropout_rate=0.1,
                 max_seq_length=256,
                 num_agent_types=4):
        super().__init__()
        self.obs_dim = obs_dim
        self.belief_dim = belief_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        self.count_pad = 4
        self.tflag_pad = 3
        
        self.register_buffer(
            "causal_bool_mask_full",
            torch.triu(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool), 1)
        )

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout_rate)
        )
        self.act_kind_embedding   = nn.Embedding(3, hidden_dim, padding_idx=0)
        self.count_embedding = nn.Embedding(5, hidden_dim, padding_idx=self.count_pad)
        self.table_flag_embedding = nn.Embedding(4, hidden_dim, padding_idx=self.tflag_pad)
        self.agent_embedding = nn.Embedding(num_agent_types, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_dim)

        self.register_buffer("lut_act_kind",   torch.tensor([1,1,1,1,1,1,2,1,1,1,0], dtype=torch.long))
        self.register_buffer("lut_count",      torch.tensor([1,2,3,1,2,3,0,1,2,3,4], dtype=torch.long))
        self.register_buffer("lut_table_flag", torch.tensor([1,1,1,2,2,2,0,0,0,0,3], dtype=torch.long))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        # === Belief Head (This is the ONLY output head) ===
        self.belief_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.opponent_position_embedding = nn.Embedding(3, 16)
        self.belief_film_layer = FiLMLayer(input_dim=hidden_dim, cond_dim=16)
        self.belief_head_shared = nn.Linear(hidden_dim, hidden_dim)
        
        self.belief_head_0 = nn.Linear(hidden_dim, self.belief_dim)
        self.belief_head_1 = nn.Linear(hidden_dim, self.belief_dim)
        self.belief_head_2 = nn.Linear(hidden_dim, self.belief_dim)

    @torch.no_grad()
    def _decompose_actions(self, action_sequence, padding_mask=None):
        a = action_sequence.long()
        act_kind, count, tflag = self.lut_act_kind[a], self.lut_count[a], self.lut_table_flag[a]
        if padding_mask is not None:
            act_pad, count_pad, tflag_pad = torch.zeros_like(act_kind), torch.full_like(count, self.count_pad), torch.full_like(tflag, self.tflag_pad)
            act_kind, count, tflag = torch.where(padding_mask, act_pad, act_kind), torch.where(padding_mask, count_pad, count), torch.where(padding_mask, tflag_pad, tflag)
        return act_kind, count, tflag

    def _encode_inputs(self, obs_sequence, action_sequence, agent_types, positions, padding_mask):
        act_kind_ids, count_ids, table_flag_ids = self._decompose_actions(action_sequence, padding_mask)
        combined = (self.obs_encoder(obs_sequence) + self.act_kind_embedding(act_kind_ids) + self.count_embedding(count_ids) +
                    self.table_flag_embedding(table_flag_ids) + self.agent_embedding(agent_types) + self.position_embedding(positions))
        return combined

    def forward(self, obs_sequence, action_sequence, agent_types, positions, padding_mask, **kwargs):
        encoded_inputs = self._encode_inputs(obs_sequence, action_sequence, agent_types, positions, padding_mask)
        T = encoded_inputs.size(1)
        causal_mask = self.causal_bool_mask_full[:T, :T]
        
        transformer_output = self.transformer(
            encoded_inputs, mask=causal_mask,
            src_key_padding_mask=padding_mask.bool() if padding_mask is not None else None,
            is_causal=True,
        )

        B, _, _ = transformer_output.shape
        device = transformer_output.device
        
        belief_hidden = self.belief_fc(transformer_output)

        opp_indices = torch.arange(3, device=device).view(1, 1, 3).expand(B, T, -1)
        pos_embeds = self.opponent_position_embedding(opp_indices)
        belief_hidden_tiled = belief_hidden.unsqueeze(2).expand(-1, -1, 3, -1)
        modulated_hidden = self.belief_film_layer(belief_hidden_tiled, pos_embeds)
        
        out_logits = self.belief_head_shared(F.relu(modulated_hidden))

        logits0 = self.belief_head_0(out_logits[:, :, 0, :])
        logits1 = self.belief_head_1(out_logits[:, :, 1, :])
        logits2 = self.belief_head_2(out_logits[:, :, 2, :])

        # Return the intermediate embedding and final logits
        return {
            "embedding": belief_hidden,
            "logits_opp0": logits0,
            "logits_opp1": logits1,
            "logits_opp2": logits2
        }