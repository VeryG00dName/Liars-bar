# src/model/ppo_embedding_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOEmbeddingModel(nn.Module):
    """
    A unified, monolithic autoregressive model for PPO.
    
    This version combines the best features discussed:
    - A fixed-size belief head for up to `belief_dim` opponent types.
    - An improved, non-linear belief feature extractor (`belief_fc`).
    - A robust "Projected Fusion" mechanism that integrates belief context
      before making policy and value decisions.
    - Maintained backward compatibility with original naming conventions.
    """
    def __init__(self,
                 obs_dim,
                 action_dim=7,
                 belief_dim=64,
                 hidden_dim=256,
                 num_heads=4,
                 num_layers=2,
                 dropout_rate=0.1,
                 max_seq_length=256,
                 num_agent_types=4):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.belief_dim = belief_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        self.count_pad = 4
        self.tflag_pad = 3
        
        self.register_buffer(
            "causal_bool_mask_full",
            torch.triu(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool), 1)
        )

        # === Input Encoders ===
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout_rate)
        )
        self.act_kind_embedding   = nn.Embedding(3, hidden_dim, padding_idx=0)
        self.count_embedding = nn.Embedding(5, hidden_dim, padding_idx=self.count_pad)
        self.table_flag_embedding = nn.Embedding(4, hidden_dim, padding_idx=self.tflag_pad)
        self.agent_embedding = nn.Embedding(num_agent_types, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_dim)

        # === Factorization Look-up Tables ===
        self.register_buffer("lut_act_kind",   torch.tensor([1,1,1,1,1,1,2,1,1,1,0], dtype=torch.long))
        self.register_buffer("lut_count",      torch.tensor([1,2,3,1,2,3,0,1,2,3,4], dtype=torch.long))
        self.register_buffer("lut_table_flag", torch.tensor([1,1,1,2,2,2,0,0,0,0,3], dtype=torch.long))

        # === Transformer Backbone ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate, activation='silu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        # === Belief Heads (Integrated) ===
        # Upgraded belief feature extractor before FiLM and final projection
        self.belief_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.opponent_position_embedding = nn.Embedding(3, 16) # Conditioning for 3 opponent slots
        self.belief_film_layer = FiLMLayer(input_dim=hidden_dim, cond_dim=16)
        self.belief_head_shared = nn.Linear(hidden_dim, self.belief_dim)

        # === Fused Policy and Value Heads ===
        # The input is the main hidden_dim PLUS the intermediate belief_hidden dim.
        fused_input_dim = hidden_dim + hidden_dim
        
        # This layer processes the FUSED representation
        self.policy_value_feature_extractor = nn.Sequential(
            nn.Linear(fused_input_dim, hidden_dim), nn.SiLU(),
        )
        self.action_head     = nn.Linear(hidden_dim, action_dim)
        self.opp_action_head = nn.Linear(hidden_dim, action_dim)
        self.value_head      = nn.Linear(hidden_dim, 1)

    @torch.no_grad()
    def _decompose_actions(self, action_sequence, padding_mask=None):
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

    def _encode_inputs(self, obs_sequence, action_sequence, agent_types, positions, padding_mask):
        act_kind_ids, count_ids, table_flag_ids = self._decompose_actions(action_sequence, padding_mask)
        combined = (self.obs_encoder(obs_sequence) +
                    self.act_kind_embedding(act_kind_ids) +
                    self.count_embedding(count_ids) +
                    self.table_flag_embedding(table_flag_ids) +
                    self.agent_embedding(agent_types) +
                    self.position_embedding(positions))
        return combined

    def forward(self, obs_sequence, action_sequence, agent_types, positions, action_masks, padding_mask, **kwargs):
        encoded_inputs = self._encode_inputs(obs_sequence, action_sequence, agent_types, positions, padding_mask)
        T = encoded_inputs.size(1)
        causal_mask = self.causal_bool_mask_full[:T, :T]
        
        transformer_output = self.transformer(
            encoded_inputs, mask=causal_mask,
            src_key_padding_mask=padding_mask.bool() if padding_mask is not None else None,
            is_causal=True,
        )

        # --- Step 1: Compute Belief Logits ---
        B, _, _ = transformer_output.shape
        device = transformer_output.device
        
        # The intermediate representation used for both belief and fusion
        belief_hidden = self.belief_fc(transformer_output) # Shape: [B, T, D_hidden]

        # Use FiLM to create opponent-specific belief features
        opp_indices = torch.arange(3, device=device).view(1, 1, 3).expand(B, T, -1)
        pos_embeds = self.opponent_position_embedding(opp_indices)
        belief_hidden_tiled = belief_hidden.unsqueeze(2).expand(-1, -1, 3, -1)
        modulated_hidden = self.belief_film_layer(belief_hidden_tiled, pos_embeds)
        
        # Project to final belief logits
        out_logits = self.belief_head_shared(F.relu(modulated_hidden)) # Shape: [B, T, 3, D_belief]

        # --- Step 2: Fuse Belief Information with Transformer Output ---
        fused_representation = torch.cat([transformer_output, belief_hidden], dim=-1) # Shape: [B, T, 2 * D_hidden]

        # --- Step 3: Policy and Value Heads on Fused Representation ---
        pv_features = self.policy_value_feature_extractor(fused_representation)
        action_logits = self.action_head(pv_features)
        opp_logits = self.opp_action_head(pv_features)
        state_values = self.value_head(pv_features)

        # --- Step 4: Apply Action Mask ---
        LARGE_NEG = torch.finfo(action_logits.dtype).min / 4.0
        if action_masks is not None:
            our_turns = (agent_types == 0).unsqueeze(-1)
            invalid = (~action_masks.bool()) & our_turns
            action_logits = action_logits.masked_fill(invalid, LARGE_NEG)

        # Return belief logits for the 3 opponents, split for the loss function
        return (action_logits, opp_logits, state_values,
                out_logits[:, :, 0, :],
                out_logits[:, :, 1, :],
                out_logits[:, :, 2, :])

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation Layer.
    """
    def __init__(self, input_dim, cond_dim):
        super().__init__()
        self.cond_projection = nn.Linear(cond_dim, input_dim * 2)

    def forward(self, main_input, cond_input):
        gamma_beta = self.cond_projection(cond_input)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        return gamma * main_input + beta