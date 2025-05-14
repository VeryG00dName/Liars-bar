import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


class AutoregressiveGameModel(nn.Module):
    """
    Modified autoregressive model with opponent profiling via GRU and belief head.
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 7,
        hidden_dim: int = 256,
        gru_hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        max_seq_length: int = 16,
        num_opponent_types: int = 3,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.max_seq_length = max_seq_length
        self.num_opponent_types = num_opponent_types

        # Input encoders
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Embeddings
        self.action_embedding = nn.Embedding(action_dim, hidden_dim)
        self.agent_id_embedding = nn.Embedding(10, hidden_dim)  # assume up to 10 players
        self.position_embedding = nn.Embedding(max_seq_length, hidden_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Opponent profiling GRU and belief head
        self.opponent_profile_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
        )
        self.opponent_belief_head = nn.Linear(gru_hidden_dim, num_opponent_types)

        # Action and value heads
        combined_dim = hidden_dim + 2 * gru_hidden_dim
        self.action_head = nn.Linear(combined_dim, action_dim)
        self.opp_action_head = nn.Linear(combined_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def _encode_inputs(
        self,
        obs_sequence: Optional[torch.Tensor],
        action_sequence: torch.Tensor,
        agent_ids: torch.Tensor,
        positions: torch.Tensor,
        training_agent_perspective_id: int,
        action_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = action_sequence.shape
        device = action_sequence.device

        combined = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)

        # Action embedding
        combined += self.action_embedding(action_sequence)
        # Agent ID embedding
        combined += self.agent_id_embedding(agent_ids)
        # Position embedding
        combined += self.position_embedding(positions)

        # Observation encoding for training agent turns
        if obs_sequence is not None:
            training_mask = (agent_ids == training_agent_perspective_id).unsqueeze(-1)
            obs_encoded = self.obs_encoder(obs_sequence)
            combined += obs_encoded * training_mask

        return combined

    def forward(
        self,
        obs_sequence: Optional[torch.Tensor],
        action_sequence: torch.Tensor,
        agent_ids: torch.Tensor,
        positions: torch.Tensor,
        initial_gru_h_opp0: torch.Tensor,
        initial_gru_h_opp1: torch.Tensor,
        training_agent_perspective_id: int,
        opponent_ids_map: Union[List[List[int]], torch.Tensor],
        action_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,  # action_logits
        torch.Tensor,  # opponent action logits
        torch.Tensor,  # state values
        torch.Tensor,  # belief logits opp0
        torch.Tensor,  # belief logits opp1
        torch.Tensor,  # final hidden opp0
        torch.Tensor,  # final hidden opp1
    ]:
        batch_size, seq_len = action_sequence.shape
        device = action_sequence.device

        # Encode inputs and transformer
        encoded = self._encode_inputs(
            obs_sequence,
            action_sequence,
            agent_ids,
            positions,
            training_agent_perspective_id,
            action_masks,
        )
        causal_mask = generate_causal_mask(seq_len, device)
        transformer_out = self.transformer(encoded, mask=causal_mask)

        # Prepare outputs
        action_logits_list, opp_action_logits_list = [], []
        belief_logits_opp0_list, belief_logits_opp1_list = [], []
        value_list = []

        # Initialize GRU states
        current_h_opp0 = initial_gru_h_opp0.clone()
        current_h_opp1 = initial_gru_h_opp1.clone()

        for t in range(seq_len):
            trans_t = transformer_out[:, t, :]
            ids_t = agent_ids[:, t]
            is_train_mask = (ids_t == training_agent_perspective_id)
            is_opp_mask = ~is_train_mask

            # placeholders
            step_act = torch.zeros(batch_size, self.action_dim, device=device)
            step_opp_act = torch.zeros(batch_size, self.action_dim, device=device)
            step_val = torch.zeros(batch_size, 1, device=device)
            step_belief0 = torch.zeros(batch_size, self.num_opponent_types, device=device)
            step_belief1 = torch.zeros(batch_size, self.num_opponent_types, device=device)

            # Training agent turn: update GRUs, predict
            if is_train_mask.any():
                active_idx = is_train_mask.nonzero(as_tuple=False).squeeze(-1)
                out_active = trans_t[active_idx]

                h0 = current_h_opp0[:, active_idx, :]
                h1 = current_h_opp1[:, active_idx, :]
                gru0_out, next_h0 = self.opponent_profile_gru(out_active.unsqueeze(1), h0)
                gru1_out, next_h1 = self.opponent_profile_gru(out_active.unsqueeze(1), h1)
                gru0 = gru0_out.squeeze(1)
                gru1 = gru1_out.squeeze(1)

                # update states
                current_h_opp0[:, active_idx, :] = next_h0
                current_h_opp1[:, active_idx, :] = next_h1

                # beliefs
                b0 = self.opponent_belief_head(gru0)
                b1 = self.opponent_belief_head(gru1)
                step_belief0[active_idx] = b0
                step_belief1[active_idx] = b1

                # agent action
                inp = torch.cat([out_active, gru0, gru1], dim=-1)
                step_act[active_idx] = self.action_head(inp)

                # value
                step_val[active_idx] = self.value_head(out_active)

            # Opponent turn: predict opponent action
            if is_opp_mask.any():
                opp_idx = is_opp_mask.nonzero(as_tuple=False).squeeze(-1)
                out_opp = trans_t[opp_idx]
                # profile features
                profile0 = current_h_opp0.squeeze(0)[opp_idx]
                profile1 = current_h_opp1.squeeze(0)[opp_idx]
                inp_opp = torch.cat([out_opp, profile0, profile1], dim=-1)
                step_opp_act[opp_idx] = self.opp_action_head(inp_opp)
                step_val[opp_idx] = self.value_head(out_opp)

            action_logits_list.append(step_act)
            opp_action_logits_list.append(step_opp_act)
            belief_logits_opp0_list.append(step_belief0)
            belief_logits_opp1_list.append(step_belief1)
            value_list.append(step_val)

        # Stack outputs
        action_logits = torch.stack(action_logits_list, dim=1)
        opp_logits = torch.stack(opp_action_logits_list, dim=1)
        belief0 = torch.stack(belief_logits_opp0_list, dim=1)
        belief1 = torch.stack(belief_logits_opp1_list, dim=1)
        values = torch.stack(value_list, dim=1)

        return action_logits, opp_logits, values, belief0, belief1, current_h_opp0.detach(), current_h_opp1.detach()
