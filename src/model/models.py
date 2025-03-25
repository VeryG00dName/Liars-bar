# src/model/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StackedObservationConvModel(nn.Module):
    """
    Neural network that processes a stack of previous observations using 1D convolutions.
    
    Takes in a tensor of shape (batch_size, N, obs_dim) where:
    - batch_size: Number of examples in the batch
    - N: Number of historical observations to include
    - obs_dim: Dimension of each observation
    
    Returns policy logits and state value for PPO algorithm.
    """
    def __init__(self, obs_dim, num_actions, hidden_dim=256, num_obs_stack=10):
        super(StackedObservationConvModel, self).__init__()
        
        # 1D Convolutional layers over the stacked observations
        self.conv_layers = nn.Sequential(
            # First conv layer: (batch_size, num_obs_stack, obs_dim) -> (batch_size, hidden_dim//2, obs_dim)
            nn.Conv1d(in_channels=num_obs_stack, out_channels=hidden_dim//2, kernel_size=3, padding=1),
            nn.GELU(),
            
            # Second conv layer: (batch_size, hidden_dim//2, obs_dim) -> (batch_size, hidden_dim, obs_dim)
            nn.Conv1d(in_channels=hidden_dim//2, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            
            # Global pooling across the obs_dim dimension
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Fully connected layers after convolution and pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        
        # Policy head - outputs action logits
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        
        # Value head - outputs state value estimate
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """
        Args:
            x: Stacked observations tensor of shape (batch_size, N, obs_dim)
        
        Returns:
            policy_logits: Action logits of shape (batch_size, num_actions)
            state_value: State value of shape (batch_size, 1)
        """
        # Input shape: (batch_size, N, obs_dim)
        # Conv1d expects: (batch_size, channels, length) where channels=N and length=obs_dim
        # So no need to permute
        
        # Process through conv layers
        x = self.conv_layers(x)  # Output: (batch_size, hidden_dim, 1)
        x = x.squeeze(-1)        # Remove last dimension: (batch_size, hidden_dim)
        
        # Process through fully connected layers
        features = self.fc_layers(x)
        
        # Get policy logits and state value
        policy_logits = self.policy_head(features)
        state_value = self.value_head(features)
        
        return policy_logits, state_value


class TransformerMemoryModel(nn.Module):
    """
    Neural network that combines current observation with a sequence of game history
    using a Transformer encoder.
    
    Takes in:
    - obs_input: Current observation tensor of shape (batch_size, obs_dim)
    - memory_input: Game history tensor of shape (batch_size, seq_len, 2) where each
      entry is a [player_id, action_code] pair
    
    Returns policy logits and state value for PPO algorithm.
    """
    def __init__(
        self,
        obs_dim,
        num_actions,
        hidden_dim=256,
        embedding_dim=32,
        num_players=5,
        num_action_codes=5,
        memory_seq_len=50,
        num_heads=4,
        num_transformer_layers=2,
        dropout=0.2
    ):
        super(TransformerMemoryModel, self).__init__()
        
        # Embedding layers for memory sequence
        self.player_embedding = nn.Embedding(num_players + 1, embedding_dim, padding_idx=0)
        self.action_embedding = nn.Embedding(num_action_codes + 1, embedding_dim, padding_idx=0)
        
        # Positional encoding for transformer
        self.pos_encoder = PositionalEncoding(embedding_dim * 2, dropout, max_len=memory_seq_len)
        
        # Transformer encoder for memory sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim * 2,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        # Learnable [CLS] token for sequence pooling
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim * 2))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Observation processing
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Combined processing
        self.combined_encoder = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Policy head - outputs action logits
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        
        # Value head - outputs state value estimate
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs_input, memory_input):
        """
        Args:
            obs_input: Current observation tensor of shape (batch_size, obs_dim)
            memory_input: Game history tensor of shape (batch_size, seq_len, 2)
                          where each entry is [player_id, action_code]
        
        Returns:
            policy_logits: Action logits of shape (batch_size, num_actions)
            state_value: State value of shape (batch_size, 1)
        """
        batch_size = obs_input.size(0)
        
        # Process observation
        obs_features = self.obs_encoder(obs_input)  # (batch_size, hidden_dim)
        
        # Process memory sequence - extract player IDs and action codes
        player_ids = memory_input[:, :, 0].long()   # (batch_size, seq_len)
        action_codes = memory_input[:, :, 1].long() # (batch_size, seq_len)
        
        # Get embeddings
        player_embeds = self.player_embedding(player_ids)  # (batch_size, seq_len, embedding_dim)
        action_embeds = self.action_embedding(action_codes)  # (batch_size, seq_len, embedding_dim)
        
        # Combine embeddings (concatenate)
        memory_embeds = torch.cat([player_embeds, action_embeds], dim=2)  # (batch_size, seq_len, embedding_dim*2)
        
        # Add positional encoding
        memory_embeds = self.pos_encoder(memory_embeds)
        
        # Add CLS token at the beginning
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        memory_embeds = torch.cat([cls_tokens, memory_embeds], dim=1)
        
        # Process through transformer
        memory_features = self.transformer_encoder(memory_embeds)
        
        # Extract CLS token output as sequence representation
        memory_features = memory_features[:, 0]  # (batch_size, embedding_dim*2)
        
        # Combine observation and memory features
        combined_features = torch.cat([obs_features, memory_features], dim=1)
        features = self.combined_encoder(combined_features)
        
        # Get policy logits and state value
        policy_logits = self.policy_head(features)
        state_value = self.value_head(features)
        
        return policy_logits, state_value


class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the token embeddings for transformer.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)