import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoregressiveGameModelFull(nn.Module):
    def __init__(self,
                 obs_dim,
                 action_dim=7,
                 belief_dim=10,
                 hidden_dim=256,
                 num_heads=4,
                 num_layers=2,
                 dropout_rate=0.1,
                 max_seq_length=100,
                 num_agent_types=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.belief_dim = belief_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length

        self.extended_action_dim = action_dim + 3

        self.register_buffer(
            "causal_bool_mask_full",
            torch.triu(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool), 1)
        )
        
        # === Input Encoders ===
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        self.action_embedding = nn.Embedding(self.extended_action_dim, hidden_dim)
        # Use num_agent_types (3) for the embedding
        self.agent_embedding = nn.Embedding(num_agent_types, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_dim)

        self.reveal_embedding = nn.Embedding(self.action_dim + 1, hidden_dim)  # indices 0..6 plus 7: NO_REVEAL

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

        # Output Heads
        # Action head for the training agent (0)
        self.action_head = nn.Linear(hidden_dim*2, action_dim) 
        # Opponent action head (for types 1 and 2)
        self.opp_action_head = nn.Linear(hidden_dim*2, action_dim)
        # Value head (not used)
        self.value_head = nn.Linear(hidden_dim*2, 1)

        # Belief prediction
        self.belief_fc = nn.Linear(hidden_dim, hidden_dim)
        self.belief_head_op0 = nn.Linear(hidden_dim, belief_dim)
        self.belief_head_op1 = nn.Linear(hidden_dim, belief_dim)

    def _encode_inputs(self, obs_sequence, action_sequence=None,
                   agent_types=None, positions=None, action_masks=None,
                   reveal_sequence=None):
        batch_size, seq_len = action_sequence.shape
        device = action_sequence.device
        combined = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)

        combined += self.action_embedding(action_sequence)
        combined += self.agent_embedding(agent_types)
        combined += self.position_embedding(positions)

        if reveal_sequence is not None:
            combined += self.reveal_embedding(reveal_sequence)

        if obs_sequence is not None:
            training_agent_mask = (agent_types == 0).unsqueeze(-1)
            obs_encoded = self.obs_encoder(obs_sequence)
            combined += (obs_encoded * training_agent_mask)
        return combined

    def forward(self, obs_sequence=None, action_sequence=None,
            agent_types=None, positions=None, action_masks=None, padding_mask=None, reveal_sequence=None, valid_lengths=None):

        encoded_inputs = self._encode_inputs(
            obs_sequence, action_sequence,
            agent_types, positions, action_masks,
            reveal_sequence=reveal_sequence,
        )

        T = encoded_inputs.size(1)
        causal_mask = self.causal_bool_mask_full[:T, :T]

        transformer_output = self.transformer(
            encoded_inputs,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        # ---- beliefs (unchanged) ----
        belief_hidden = F.relu(self.belief_fc(transformer_output))

        belief_logits_0 = self.belief_head_op0(belief_hidden)
        belief_logits_1 = self.belief_head_op1(belief_hidden)

        # ---- fuse & heads ----
        fused_output = torch.cat([transformer_output, belief_hidden], dim=-1)

        action_logits = self.action_head(fused_output)    # [B,T,7]
        opp_logits    = self.opp_action_head(fused_output)
        state_values  = self.value_head(fused_output)

        # ---- apply action mask ONLY on our turns ----
        # action_masks: [B,T,7] with True=legal (as produced by your dataset)
        if (action_masks is not None) and (agent_types is not None):
            # ensure boolean
            if action_masks.dtype != torch.bool:
                action_masks = action_masks.bool()

            our_turns = (agent_types == 0)                        # [B,T]
            # rows where we *have* at least one legal action (avoid all -inf)
            has_any_legal = (action_masks.any(dim=-1)) & our_turns  # [B,T]

            # Build a full [B,T,1] broadcast gate
            gate = has_any_legal.unsqueeze(-1)                    # [B,T,1]

            # Invalid where: it's our turn AND not legal
            invalid = (~action_masks) & our_turns.unsqueeze(-1)   # [B,T,7]

            # Only mask rows that have at least one legal action; otherwise skip to avoid all -inf
            # Use a large negative constant for stability instead of -inf
            LARGE_NEG = torch.finfo(action_logits.dtype).min
            action_logits = torch.where(
                gate,                                            # per-row switch
                action_logits.masked_fill(invalid, LARGE_NEG),   # mask invalid on our rows with any legal
                action_logits                                    # leave as-is otherwise (opponent turns / degenerate rows)
            )

        return action_logits, opp_logits, state_values, belief_logits_0, belief_logits_1