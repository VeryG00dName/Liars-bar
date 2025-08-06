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
                 max_seq_length=20,
                 num_agent_types=3): # Added num_agent_types (0: Agent, 1: Opponent 0, 2: Opponent 1)
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.belief_dim = belief_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length

        self.extended_action_dim = action_dim + 3

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
        self.value_head = nn.Linear(hidden_dim*2, 1)

        # Belief prediction (one shared head, run twice with one-hot opponent indicator)
        self.belief_fc = nn.Linear(hidden_dim, hidden_dim)
        self.belief_head = nn.Linear(hidden_dim + 2, belief_dim) # Add 2 for one-hot opponent ID

        self.register_buffer("onehot_0_base", torch.tensor([1, 0], dtype=torch.float32))
        self.register_buffer("onehot_1_base", torch.tensor([0, 1], dtype=torch.float32))

        self.null_token = nn.Parameter(torch.zeros(hidden_dim))

    def _encode_inputs(self, obs_sequence, action_sequence=None,
                      agent_types=None, positions=None, action_masks=None):
        """
        Encode all inputs into a unified sequence representation.
        """
        batch_size, seq_len = action_sequence.shape
        device = action_sequence.device
        
        # Initialize combined representation with zeros
        combined = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)
        
        # 1. Embed actions (all steps)
        action_embedded = self.action_embedding(action_sequence)
        combined += action_embedded
        
        # 2. Embed agent types (0, 1, or 2)
        agent_embedded = self.agent_embedding(agent_types)
        combined += agent_embedded
        
        # 3. Embed positions (all steps)
        position_embedded = self.position_embedding(positions)
        combined += position_embedded
        
        # 4. Encode observations (only for training agent turns, agent_type=0)
        if obs_sequence is not None:
            # Create a mask for training agent turns (agent_type 0)
            training_agent_mask = (agent_types == 0).unsqueeze(-1)
            
            # Encode observations
            obs_encoded = self.obs_encoder(obs_sequence)
            
            # Only add observation encoding for training agent turns
            combined += (obs_encoded * training_agent_mask)
        
        return combined

    def _generate_causal_mask(self, seq_len, device):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )

    def _generate_padding_mask(self, seq_len, valid_lens, device):
        batch_size = valid_lens.size(0)
        # We need to ensure valid_lens is on the same device as the sequence.
        valid_lens_device = valid_lens.to(device) 
        
        # Create a mask where True indicates padding (should be ignored by attention)
        mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) >= valid_lens_device.unsqueeze(1)
        return mask

    def forward(self, obs_sequence=None, action_sequence=None,
                agent_types=None, positions=None, action_masks=None, valid_lengths=None):

        batch_size, seq_len = action_sequence.shape
        device = action_sequence.device

        encoded_inputs = self._encode_inputs(
            obs_sequence, action_sequence,
            agent_types, positions, action_masks
        )

        # Causal mask ensures we only attend to past inputs
        causal_mask = self._generate_causal_mask(seq_len, device)
        
        # Padding mask handles variable sequence lengths
        padding_mask = self._generate_padding_mask(seq_len, valid_lengths, device) if valid_lengths is not None else None

        transformer_output = self.transformer(
            encoded_inputs,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        # Shared hidden layer for belief prediction
        # Belief prediction is done based on the agent's current hidden state.
        belief_hidden = F.relu(self.belief_fc(transformer_output))

        # One-hot encoding for the target opponent slot (0 or 1)
        # We assume the model predicts beliefs for both opponent slots simultaneously.
        onehot_0 = self.onehot_0_base.view(1, 1, 2).expand(batch_size, seq_len, 2)
        onehot_1 = self.onehot_1_base.view(1, 1, 2).expand(batch_size, seq_len, 2)

        belief_input_0 = torch.cat([belief_hidden, onehot_0], dim=-1)
        belief_input_1 = torch.cat([belief_hidden, onehot_1], dim=-1)

        belief_logits_0 = self.belief_head(belief_input_0)
        belief_logits_1 = self.belief_head(belief_input_1)

        # Inject belief influence into transformer output (e.g., using belief hidden)
        fused_output = torch.cat([transformer_output, belief_hidden], dim=-1)

        action_logits = self.action_head(fused_output)
        opp_logits = self.opp_action_head(fused_output)
        state_values = self.value_head(fused_output)

        if action_masks is not None:
            # Mask actions logits for the training agent (type 0)
            agent_mask = (agent_types == 0).unsqueeze(-1).expand_as(action_logits)
            action_logits = torch.where(agent_mask, action_logits.masked_fill(~action_masks.bool(), float('-inf')), action_logits)

        return action_logits, opp_logits, state_values, belief_logits_0, belief_logits_1