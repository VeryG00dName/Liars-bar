import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoregressiveGameModel(nn.Module):
    """
    Autoregressive model for Liar's Deck that predicts next actions sequentially.
    
    The model takes a round sequence history and predicts the next action in the sequence.
    It can handle both agent actions and opponent actions, with special handling for opponent
    actions where only card counts may be observed during evaluation.
    
    Features:
    - Transformer-based architecture for sequence modeling
    - Separate embeddings for observations, actions, agent types, and positions
    - Support for variable-length sequences with causal masking
    - Action prediction with proper masking of invalid actions
    - Support for both explicit actions (0-6)
    """
    def __init__(self,
                 obs_dim,
                 action_dim=7,
                 belief_dim=None,
                 hidden_dim=256,
                 num_heads=4,
                 num_layers=2,
                 dropout_rate=0.1,
                 max_seq_length=20):
        """
        Initialize the autoregressive game model.
        
        Args:
            obs_dim: Dimension of observation vectors
            action_dim: Number of possible actions (typically 7 for Liar's Deck)
            belief_dim: Dimension of belief vectors. If ``None`` the model will not
                use external belief inputs. This allows training on full game
                sequences without a separate belief model while remaining
                backwards compatible with older checkpoints that expect a belief
                vector.
            hidden_dim: Hidden dimension for transformer and other layers
            num_heads: Number of attention heads in transformer
            num_layers: Number of transformer layers
            dropout_rate: Dropout rate for regularization
            max_seq_length: Maximum sequence length to support
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.belief_dim = belief_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        
        # Extended action space for card counts and challenge token
        # Regular actions: 0-6
        # Card count representations: 7=1 card, 8=2 cards, 9=3 cards
        self.extended_action_dim = action_dim + 3
        
        # === Input Encoders ===
        # Observation encoder (only for training agent turns)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Belief encoder for external beliefs from a separate belief model
        if belief_dim is not None:
            self.belief_encoder = nn.Sequential(
                nn.Linear(belief_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
        else:
            self.belief_encoder = None
        
        # === Embeddings ===
        # Action embedding (includes special tokens for card count)
        self.action_embedding = nn.Embedding(self.extended_action_dim, hidden_dim)
        
        # Agent type embedding (0=training agent, 1=opponent)
        self.agent_embedding = nn.Embedding(2, hidden_dim)
        
        # Position embedding for sequence positions
        self.position_embedding = nn.Embedding(max_seq_length, hidden_dim)
        
        # === Transformer Encoder ===
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
        
        # === Output Heads ===
        # Main output: predict actions in original action space (0-6)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # opponent output: predict actions in original action space (0-6)
        self.opp_action_head  = nn.Linear(hidden_dim, action_dim)
        
        # Value prediction head
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # === Null/Padding Token ===
        # Learnable null token for padding (initialized as zeros)
        self.null_token = nn.Parameter(torch.zeros(hidden_dim))
    
    def _encode_inputs(self, obs_sequence, belief_sequence=None, action_sequence=None,
                      agent_types=None, positions=None, action_masks=None):
        """
        Encode all inputs into a unified sequence representation.
        
        Args:
            obs_sequence: Tensor of shape ``[batch_size, seq_len, obs_dim]`` or ``None``
            belief_sequence: Optional tensor of shape ``[batch_size, seq_len, belief_dim]``
            action_sequence: Tensor of shape ``[batch_size, seq_len]`` with previous actions
            agent_types: Tensor of shape ``[batch_size, seq_len]`` where 0 denotes the
                training agent and 1 an opponent
            positions: Tensor of shape ``[batch_size, seq_len]`` indicating positions
            action_masks: Optional tensor of shape ``[batch_size, seq_len, action_dim]``
        
        Returns:
            Tensor of shape [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len = action_sequence.shape
        device = action_sequence.device
        
        # Initialize combined representation with zeros
        combined = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)
        
        # 1. Embed actions (all steps)
        action_embedded = self.action_embedding(action_sequence)
        combined = combined + action_embedded
        
        # 2. Embed agent types (all steps)
        agent_embedded = self.agent_embedding(agent_types)
        combined = combined + agent_embedded
        
        # 3. Embed positions (all steps)
        position_embedded = self.position_embedding(positions)
        combined = combined + position_embedded
        
        # 4. Encode observations (only for training agent turns)
        if obs_sequence is not None:
            # Create a mask for training agent turns
            training_agent_mask = (agent_types == 0).unsqueeze(-1)
            
            # Encode observations
            obs_encoded = self.obs_encoder(obs_sequence)
            
            # Only add observation encoding for training agent turns
            combined = combined + (obs_encoded * training_agent_mask)
        
        # 5. Encode beliefs (if provided)
        if belief_sequence is not None and self.belief_encoder is not None:
            belief_encoded = self.belief_encoder(belief_sequence)
            combined = combined + belief_encoded
        
        return combined
    
    def _generate_causal_mask(self, seq_len, device):
        """
        Generate a causal mask for transformer to prevent looking at future tokens.
        
        Args:
            seq_len: Length of the sequence
            device: Device to place the mask on
        
        Returns:
            Tensor of shape [seq_len, seq_len] with 1s in upper triangle
        """
        # Create a mask that prevents attending to future positions
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
    
    def _generate_padding_mask(self, seq_len, valid_lens, device):
        """
        Generate padding mask for sequences of different lengths.
        
        Args:
            seq_len: Maximum sequence length
            valid_lens: Tensor of shape [batch_size] with valid lengths
            device: Device to place the mask on
        
        Returns:
            Tensor of shape [batch_size, seq_len] with True for padding positions
        """
        batch_size = valid_lens.size(0)
        mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) >= valid_lens.unsqueeze(1)
        return mask
    
    def forward(self, obs_sequence=None, belief_sequence=None, action_sequence=None,
                agent_types=None, positions=None, action_masks=None, valid_lengths=None):
        """Forward pass through the model.

        Args:
            obs_sequence: Tensor of observations ``[batch, seq, obs_dim]`` or ``None``.
            belief_sequence: Optional tensor of beliefs ``[batch, seq, belief_dim]``.
            action_sequence: Tensor of previous actions ``[batch, seq]``.
            agent_types: Tensor indicating agent type ``[batch, seq]`` (0=ours, 1=opponent).
            positions: Tensor of positions in sequence ``[batch, seq]``.
            action_masks: Optional tensor of action masks ``[batch, seq, action_dim]``.
            valid_lengths: Optional tensor of valid sequence lengths ``[batch]``.

        Returns:
            Tuple ``(action_logits, opp_logits, state_values)`` where each tensor has
            shape ``[batch, seq, ...]``.
        """
        batch_size, seq_len = action_sequence.shape
        device = action_sequence.device
        
        # 1. Encode all inputs
        encoded_inputs = self._encode_inputs(
            obs_sequence, belief_sequence, action_sequence, 
            agent_types, positions, action_masks
        )
        
        # 2. Generate attention masks
        # Causal mask to prevent looking at future tokens
        causal_mask = self._generate_causal_mask(seq_len, device)
        
        # Padding mask for variable length sequences (if provided)
        padding_mask = None
        if valid_lengths is not None:
            padding_mask = self._generate_padding_mask(seq_len, valid_lengths, device)
        
        # 3. Process through transformer
        transformer_output = self.transformer(
            encoded_inputs,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        
        # 4. Generate outputs
        # Standard action predictions (0-6)
        action_logits = self.action_head(transformer_output)
        
        opp_logits  = self.opp_action_head(transformer_output)
        
        # Value predictions
        state_values = self.value_head(transformer_output)
        
        # Apply action masks if provided
        if action_masks is not None:
            action_logits = action_logits.masked_fill(~action_masks.bool(), float('-inf'))
        
        return action_logits, opp_logits, state_values
