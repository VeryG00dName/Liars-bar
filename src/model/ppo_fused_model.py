# src/model/ppo_fused_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOFusedModel(nn.Module):
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
            dropout=dropout_rate, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        # === Belief Heads (Integrated) ===
        # Upgraded belief feature extractor before FiLM and final projection
        self.belief_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.belief_head = nn.Linear(hidden_dim, belief_dim)
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

    def forward(
        self,
        obs_sequence, action_sequence, agent_types, positions,
        action_masks, padding_mask,
        *,                               # make extra args keyword-only
        return_embeddings: bool = False,
        **kwargs
    ):
        encoded_inputs = self._encode_inputs(obs_sequence, action_sequence, agent_types, positions, padding_mask)
        T = encoded_inputs.size(1)
        causal_mask = self.causal_bool_mask_full[:T, :T]
        transformer_output = self.transformer(
            encoded_inputs, mask=causal_mask,
            src_key_padding_mask=padding_mask.bool() if padding_mask is not None else None,
            is_causal=True,
        )

        # belief_hidden drives opponent head
        belief_hidden = self.belief_fc(transformer_output)            # [B,T,D]
        fused_representation = torch.cat([transformer_output, belief_hidden], dim=-1)
        pv_features = self.policy_value_feature_extractor(fused_representation)
        action_logits = self.action_head(pv_features)                  # [B,T,7]
        opp_logits    = self.opp_action_head(belief_hidden)           # [B,T,7]
        state_values  = self.value_head(pv_features)                  # [B,T,1]
        belief_logits = self.belief_head(belief_hidden.detach())   # [B,T,belief_dim]
        LARGE_NEG = torch.finfo(action_logits.dtype).min / 4.0
        if action_masks is not None:
            our_turns = (agent_types == 0).unsqueeze(-1)              # [B,T,1]
            invalid = (~action_masks.bool()) & our_turns
            action_logits = action_logits.masked_fill(invalid, LARGE_NEG)

        if return_embeddings:
            return (action_logits, opp_logits, state_values, belief_logits, belief_hidden.detach())
        else:
            return (action_logits, opp_logits, state_values, belief_logits)

    # ===== Convenience helpers for planning & challenge evaluation =====

    @staticmethod
    def _last_nonself_index(agent_types: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        agent_types: [B, T] with {0=self, 1/2/3=opponents}
        padding_mask: [B, T] True for PAD (optional). If provided, pads are ignored.

        Returns:
            idx: [B] int64, index of the last timestep t' < T where an opponent acted.
                 If none exists (edge case), returns T-1 clamped to valid (so you can guard on a mask).
        """
        B, T = agent_types.shape
        nonself = (agent_types != 0)
        if padding_mask is not None:
            nonself &= ~padding_mask.bool()

        # Find last True along time: reverse, argmax of first True, then convert back
        rev = torch.flip(nonself.to(torch.int32), dims=[1])
        pos_from_end = rev.argmax(dim=1)  # [B], 0 if last element is True, else distance
        # If a row has no True at all, argmax is 0 but nonself is all False. Detect that:
        has_any = nonself.any(dim=1)
        # Convert back to forward index
        idx = (T - 1) - pos_from_end
        # For rows with no opponent, just point at 0 to stay in-bounds (caller should check has_any)
        idx = torch.where(has_any, idx, torch.zeros_like(idx))
        return idx  # [B]

    @staticmethod
    def _batch_gather_last_step(tensorBTD: torch.Tensor, idxB: torch.Tensor) -> torch.Tensor:
        """
        tensorBTD: [B, T, D]  (e.g., opp_logits)
        idxB:      [B] indices to gather along T
        Returns:   [B, D]
        """
        B, T, D = tensorBTD.shape
        gather_idx = idxB.view(B, 1, 1).expand(B, 1, D)
        out = torch.gather(tensorBTD, dim=1, index=gather_idx).squeeze(1)
        return out  # [B, D]

    def last_opponent_logits_before_us(
        self,
        opp_logits: torch.Tensor,
        agent_types: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get opponent logits for the *immediately previous* opponent timestep (per batch row).

        Args:
            opp_logits:   [B, T, A]
            agent_types:  [B, T] with {0=self, 1/2/3=opponents}
            padding_mask: [B, T] True for PAD (optional)

        Returns:
            prev_opp_logits: [B, A]
            has_prev_opp:    [B] bool  (False if no opponent step exists before our current step)
        """
        idx = self._last_nonself_index(agent_types, padding_mask)      # [B]
        prev_opp_logits = self._batch_gather_last_step(opp_logits, idx)  # [B, A]
        has_prev_opp = (agent_types.gather(dim=1, index=idx.view(-1,1)).squeeze(1) != 0)
        return prev_opp_logits, has_prev_opp


class FiLMLayer(nn.Module):
    """
    (Unused now) Feature-wise Linear Modulation Layer.
    Kept here in case you want to reintroduce conditioning later.
    """
    def __init__(self, input_dim, cond_dim):
        super().__init__()
        self.cond_projection = nn.Linear(cond_dim, input_dim * 2)

    def forward(self, main_input, cond_input):
        gamma_beta = self.cond_projection(cond_input)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        return gamma * main_input + beta
