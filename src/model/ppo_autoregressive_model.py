# src/model/ppo_autoregressive_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOAutoregressiveModel(nn.Module):
    """
    Action ID semantics (0..9) → factorization:
      0..2 : PLAY,  count=(id%3)+1, table=TABLE
      3..5 : PLAY,  count=(id%3)+1, table=NON_TABLE
      6    : CHALLENGE, count=NONE, table=NA
      7..9 : PLAY,  count=id-6,    table=NA   (unknown cards; only count known)

    Factor vocabularies (indices):
      act_kind   ∈ {PAD=0, PLAY=1, CHALLENGE=2}
      count      ∈ {NONE=0, 1=1, 2=2, 3=3}
      table_flag ∈ {NA=0, TABLE=1, NON_TABLE=2}
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
        
        # Causal attention mask (upper triangular = future is masked)
        self.register_buffer(
            "causal_bool_mask_full",
            torch.triu(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool), 1)
        )

        # === Input encoders ===
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # --- Factorized action embeddings ---

        self.act_kind_embedding   = nn.Embedding(3, hidden_dim, padding_idx=0)  # 0=PAD, 1=PLAY, 2=CHALLENGE
        self.count_embedding = nn.Embedding(5, hidden_dim, padding_idx=self.count_pad)
        self.table_flag_embedding = nn.Embedding(4, hidden_dim, padding_idx=self.tflag_pad)

        # Agent / position embeddings
        self.agent_embedding = nn.Embedding(num_agent_types, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_dim)

        # Fast LUTs to decompose 0..9 → factor IDs
        # act_kind: 0=PAD, 1=PLAY, 2=CHALLENGE
        self.register_buffer("lut_act_kind",   torch.tensor([1,1,1,1,1,1,2,1,1,1,0], dtype=torch.long))
        # count: 0=NONE, 1,2,3
        self.register_buffer("lut_count",      torch.tensor([1,2,3,1,2,3,0,1,2,3,4], dtype=torch.long))
        # table_flag: 0=NA, 1=TABLE, 2=NON_TABLE
        self.register_buffer("lut_table_flag", torch.tensor([1,1,1,2,2,2,0,0,0,0,3], dtype=torch.long))

        # === Transformer ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        # === Output heads ===
        self.action_head     = nn.Linear(hidden_dim * 2, action_dim)  # self (agent 0)
        self.opp_action_head = nn.Linear(hidden_dim * 2, action_dim)  # opponents (agent 1/2)
        self.value_head      = nn.Linear(hidden_dim * 2, 1)

        # Belief heads
        self.belief_fc      = nn.Linear(hidden_dim, hidden_dim)
        self.belief_head_op0 = nn.Linear(hidden_dim, belief_dim)
        self.belief_head_op1 = nn.Linear(hidden_dim, belief_dim)
        self.belief_head_op2 = nn.Linear(hidden_dim, belief_dim)
    # -------------------------- utils --------------------------

    @torch.no_grad()
    def _decompose_actions(self, action_sequence, agent_types=None, padding_mask=None):
        """
        action_sequence: LongTensor [B,T] with values in {0..9}
        Returns (act_kind_ids, count_ids, table_flag_ids) as Long [B,T]
        """
        a = action_sequence.long()

        act_kind = self.lut_act_kind[a]
        count    = self.lut_count[a]
        tflag    = self.lut_table_flag[a]

        if padding_mask is not None:
            # Zero-out factors on padded steps so embeddings use padding_idx=0 (zero vector)
            # Where padding_mask == True, fill with each factor's PAD index
            act_pad   = torch.zeros_like(act_kind)                                  # 0
            count_pad = torch.full_like(count, self.count_pad, dtype=torch.long)    # 4
            tflag_pad = torch.full_like(tflag, self.tflag_pad, dtype=torch.long)    # 3

            act_kind = torch.where(padding_mask, act_pad,   act_kind)
            count    = torch.where(padding_mask, count_pad, count)
            tflag    = torch.where(padding_mask, tflag_pad, tflag)
        
        return act_kind, count, tflag

    def _encode_inputs(self, obs_sequence, action_sequence=None,
                       agent_types=None, positions=None, action_masks=None,
                       padding_mask=None):
        """
        Build token embeddings per time step by summing:
          - factorized action embeddings (act_kind + count + table_flag)
          - agent embedding
          - position embedding
          - obs encoder output (only on our turns: agent_types==0)
        """
        batch_size, seq_len = action_sequence.shape
        device = action_sequence.device

        combined = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)

        # Factorize the actions via LUTs
        act_kind_ids, count_ids, table_flag_ids = self._decompose_actions(
            action_sequence, agent_types=agent_types, padding_mask=padding_mask
        )

        # Sum factor embeddings
        combined += self.act_kind_embedding(act_kind_ids)
        combined += self.count_embedding(count_ids)
        combined += self.table_flag_embedding(table_flag_ids)

        # Add agent + position
        combined += self.agent_embedding(agent_types)
        combined += self.position_embedding(positions)

        # Add observation encoding
        combined += self.obs_encoder(obs_sequence)

        return combined

    # -------------------------- forward --------------------------

    def forward(self, obs_sequence=None, action_sequence=None,
                agent_types=None, positions=None, action_masks=None,
                padding_mask=None, valid_lengths=None):

        encoded_inputs = self._encode_inputs(
            obs_sequence, action_sequence,
            agent_types, positions, action_masks,
            padding_mask=padding_mask,
        )

        # encoded_inputs: [B, T, D] or [T, B, D]; padding_mask: [B, T] (True = pad)
        T = encoded_inputs.size(1) if encoded_inputs.dim() == 3 else encoded_inputs.size(0)

        # ensure pad mask is boolean and on the same device
        if padding_mask is not None:
            if padding_mask.dtype is not torch.bool:
                padding_mask = padding_mask.bool()
            # padding_mask should already be on the same device as inputs; assert to catch mistakes early
            assert padding_mask.device == encoded_inputs.device, \
                f"padding_mask on {padding_mask.device}, inputs on {encoded_inputs.device}"

        # slice the preallocated GPU boolean causal mask
        causal_mask = self.causal_bool_mask_full[:T, :T]
        # assert same device to avoid surprises
        assert causal_mask.device == encoded_inputs.device, \
            f"causal_mask on {causal_mask.device}, inputs on {encoded_inputs.device}"

        # pass BOTH mask and is_causal=True (PyTorch’s MHA expects a mask when is_causal=True)
        transformer_output = self.transformer(
            encoded_inputs,
            mask=causal_mask,                 # bool, on CUDA
            src_key_padding_mask=padding_mask,
            is_causal=True,
        )

        # ---- beliefs ----
        belief_hidden  = F.relu(self.belief_fc(transformer_output))
        belief_logits_0 = self.belief_head_op0(belief_hidden)
        belief_logits_1 = self.belief_head_op1(belief_hidden)
        belief_logits_2 = self.belief_head_op2(belief_hidden)
        # ---- fuse & heads ----
        fused_output = torch.cat([transformer_output, belief_hidden], dim=-1)

        action_logits = self.action_head(fused_output)     # [B,T,7]
        opp_logits    = self.opp_action_head(fused_output) # [B,T,7]
        state_values  = self.value_head(fused_output)      # [B,T,1]

        # ---- apply action mask ONLY on our turns ----
        # action_masks: [B,T,7] with True=legal
        if (action_masks is not None) and (agent_types is not None):
            if action_masks.dtype != torch.bool:
                action_masks = action_masks.bool()

            our_turns = (agent_types == 0)                           # [B,T]
            has_any_legal = (action_masks.any(dim=-1)) & our_turns   # [B,T]
            gate = has_any_legal.unsqueeze(-1)                       # [B,T,1]
            invalid = (~action_masks) & our_turns.unsqueeze(-1)      # [B,T,7]

            LARGE_NEG = -1e4
            action_logits = torch.where(
                gate,
                action_logits.masked_fill(invalid, LARGE_NEG),
                action_logits
            )

        return action_logits, opp_logits, state_values, belief_logits_0, belief_logits_1, belief_logits_2