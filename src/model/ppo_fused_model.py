# src/model/ppo_fused_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class StrategyDictionary(nn.Module):
    """
    Learns a shared dictionary of strategy "bricks" and produces a sparse,
    compositional strategy code for a given history.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_bricks: int, brick_dim: int):
        super().__init__()
        self.num_bricks = num_bricks
        self.brick_dim = brick_dim

        # The shared dictionary of "bricks" (learnable parameters)
        self.bricks = nn.Parameter(torch.randn(num_bricks, brick_dim))
        # Initialize bricks to be orthogonal to encourage diversity from the start
        nn.init.orthogonal_(self.bricks)

        # Encoder to produce activations from the transformer's output
        self.activation_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bricks)
        )

    def forward(self, transformer_output: torch.Tensor):
        # transformer_output shape: [B, T, D_transformer]
        
        # 1. Produce raw activations for each brick
        raw_activations = self.activation_encoder(transformer_output)
        
        # 2. Ensure activations are non-negative to act as weights
        # Softplus is a smooth version of ReLU, which is good here.
        activations = F.softplus(raw_activations)  # Shape: [B, T, num_bricks]
        
        # 3. Combine bricks using activations to form the strategy code
        # Matmul: (B, T, num_bricks) @ (num_bricks, brick_dim) -> (B, T, brick_dim)
        
        return activations, self.bricks


class PPOFusedModel(nn.Module):
    """
    A unified, monolithic autoregressive model for PPO, updated with a
    Sparse Dictionary representation for opponent strategy.
    
    This version replaces the dense `belief_fc` with a compositional
    strategy mechanism designed to learn "building blocks" of behavior.
    """
    def __init__(self,
                 obs_dim,
                 action_dim=7,
                 hidden_dim=256,
                 num_heads=4,
                 num_layers=2,
                 dropout_rate=0.1,
                 max_seq_length=256,
                 num_agent_types=4,
                 # --- New args for the Strategy Dictionary ---
                 num_bricks=32,
                 brick_dim=32):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        self.count_pad = 4
        self.tflag_pad = 3
        
        self.register_buffer(
            "causal_bool_mask_full",
            torch.triu(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool), 1)
        )

        # === Input Encoders (Unchanged) ===
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout_rate)
        )
        self.act_kind_embedding   = nn.Embedding(3, hidden_dim, padding_idx=0)
        self.count_embedding = nn.Embedding(5, hidden_dim, padding_idx=self.count_pad)
        self.table_flag_embedding = nn.Embedding(4, hidden_dim, padding_idx=self.tflag_pad)
        self.agent_embedding = nn.Embedding(num_agent_types, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_dim)

        # === Factorization Look-up Tables (Unchanged) ===
        self.register_buffer("lut_act_kind",   torch.tensor([1,1,1,1,1,1,2,1,1,1,0], dtype=torch.long))
        self.register_buffer("lut_count",      torch.tensor([1,2,3,1,2,3,0,1,2,3,4], dtype=torch.long))
        self.register_buffer("lut_table_flag", torch.tensor([1,1,1,2,2,2,0,0,0,0,3], dtype=torch.long))

        # === Transformer Backbone (Unchanged) ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        # === NEW: Strategy Dictionary (Replaces belief_fc) ===
        self.strategy_dictionary = StrategyDictionary(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_bricks=num_bricks,
            brick_dim=brick_dim
        )

        # === Heads (Updated to use the new strategy code) ===
        # FiLM layer now conditioned on the strategy code's dimension
        self.pv_film = StrategyFiLM(feat_dim=hidden_dim, cond_dim=brick_dim, use_ln=True)

        # Policy and Value heads operate on the FiLM-modulated features
        self.action_head     = nn.Linear(hidden_dim, action_dim)
        self.value_head      = nn.Linear(hidden_dim, 1)

        # Opponent action head operates directly on the strategy code
        self.opp_action_head = nn.Linear(brick_dim, action_dim)


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
        dropout_p: float = 0.25,         # Add dropout probability as an argument
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

        # Get the raw activations from the dictionary
        activations, bricks = self.strategy_dictionary(transformer_output)

        # --- APPLY DROPOUT HERE ---
        # Apply dropout to the activations before they are used.
        # This should only be active during training.
        activations_reg = F.dropout(activations, p=dropout_p, training=self.training)
        
        # compute the strategy code using the regularized activations
        strategy_code = torch.matmul(activations_reg, bricks)
        
        # --- Policy/Value Stream ---
        pv_features = self.pv_film(transformer_output, strategy_code)
        action_logits = self.action_head(pv_features)
        state_values = self.value_head(pv_features)

        # --- Opponent Modeling Stream ---
        # Opponent prediction also uses the regularized strategy code
        opp_logits = self.opp_action_head(strategy_code)
        
        # Apply action mask for our turns
        LARGE_NEG = torch.finfo(action_logits.dtype).min / 4.0
        if action_masks is not None:
            our_turns = (agent_types == 0).unsqueeze(-1)
            invalid = (~action_masks.bool()) & our_turns
            action_logits = action_logits.masked_fill(invalid, LARGE_NEG)

        if return_embeddings:
            # We return the original, non-dropped-out activations for the regularization losses
            # The losses should be based on the model's "intent", not the noisy version.
            strategy_code_probe = torch.matmul(activations, bricks).detach()
            embedding_tuple = (strategy_code_probe, activations, bricks)
            return (action_logits, opp_logits, state_values, embedding_tuple)
        else:
            return (action_logits, opp_logits, state_values)

    # ===== Convenience helpers (Unchanged) =====
    @staticmethod
    def _last_nonself_index(agent_types: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T = agent_types.shape
        nonself = (agent_types != 0)
        if padding_mask is not None:
            nonself &= ~padding_mask.bool()
        rev = torch.flip(nonself.to(torch.int32), dims=[1])
        pos_from_end = rev.argmax(dim=1)
        has_any = nonself.any(dim=1)
        idx = (T - 1) - pos_from_end
        idx = torch.where(has_any, idx, torch.zeros_like(idx))
        return idx

    @staticmethod
    def _batch_gather_last_step(tensorBTD: torch.Tensor, idxB: torch.Tensor) -> torch.Tensor:
        B, T, D = tensorBTD.shape
        gather_idx = idxB.view(B, 1, 1).expand(B, 1, D)
        out = torch.gather(tensorBTD, dim=1, index=gather_idx).squeeze(1)
        return out

    def last_opponent_logits_before_us(
        self,
        opp_logits: torch.Tensor,
        agent_types: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self._last_nonself_index(agent_types, padding_mask)
        prev_opp_logits = self._batch_gather_last_step(opp_logits, idx)
        has_prev_opp = (agent_types.gather(dim=1, index=idx.view(-1,1)).squeeze(1) != 0)
        return prev_opp_logits, has_prev_opp


class StrategyFiLM(nn.Module):
    """
    Feature-wise Linear Modulation with safe initialization.
    (Unchanged, but now takes a condition from the Strategy Dictionary)
    """
    def __init__(self, feat_dim: int, cond_dim: int, use_ln: bool = True):
        super().__init__()
        hidden = max(64, cond_dim // 2)
        self.to_gamma = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, feat_dim)
        )
        self.to_beta  = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, feat_dim)
        )
        if use_ln:
            self.ln = nn.LayerNorm(feat_dim)
        else:
            self.ln = nn.Identity()
            
        # Optional: Add a learnable gate for stability, as discussed
        # self.gate = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.to_gamma[-1].weight); nn.init.zeros_(self.to_gamma[-1].bias)
        nn.init.zeros_(self.to_beta[-1].weight);  nn.init.zeros_(self.to_beta[-1].bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z[:, None, :]

        gamma = torch.tanh(self.to_gamma(z))
        beta  = self.to_beta(z)
        
        # If using the gate:
        # gate_s = torch.sigmoid(self.gate) # or just self.gate if you want unbounded
        # gamma = gamma * gate_s
        # beta = beta * gate_s
        
        h_mod = h * (1.0 + gamma) + beta
        return self.ln(h_mod)