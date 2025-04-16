# src/model/new_models.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleExpertPolicyNetwork(nn.Module):
    """
    A simplified expert network that uses a single hidden layer.
    
    It performs:
      1. A linear transformation from input_dim to hidden_dim, with GELU activation.
      2. Optional dropout and layer normalization.
      3. Optionally, a one-layer LSTM (applied to a single time step).
      4. A final linear layer to produce action logits.
      5. Softmax over logits to produce action probabilities.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, use_lstm=True, use_dropout=True, use_layer_norm=True):
        super(SingleExpertPolicyNetwork, self).__init__()
        self.use_lstm = use_lstm
        self.use_dropout = use_dropout
        self.use_layer_norm = use_layer_norm
        
        # A single hidden layer.
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layer (if used) and final classification layer.
        if self.use_lstm:
            # Using a one-layer LSTM for a single time-step.
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
            self.fc_out = nn.Linear(hidden_dim, output_dim)
        else:
            self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.3)
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, hidden_state=None):
        # x: (batch_size, input_dim)
        x = F.gelu(self.fc1(x))
        if self.use_layer_norm:
            x = self.layer_norm(x)
        if self.use_dropout:
            x = self.dropout(x)
        
        # Optionally use LSTM.
        if self.use_lstm:
            # Add a time dimension (sequence length 1).
            x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
            x, hidden_state = self.lstm(x, hidden_state)
            x = x.squeeze(1)    # (batch_size, hidden_dim)
        
        action_logits = self.fc_out(x)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs, hidden_state

class PolicyNetwork(nn.Module):
    """
    Mixture-of-Experts policy network that uses an external expert selection.
    
    Instead of computing gating internally, this network expects a second input,
    'expert_index', which indicates which expert to use. The network then simply
    passes the observation through the selected expert to produce the action probabilities.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts,
                 use_lstm=True, use_dropout=True, use_layer_norm=True):
        super(PolicyNetwork, self).__init__()
        self.num_experts = num_experts
        
        # Create a list of experts.
        self.experts = nn.ModuleList([
            SingleExpertPolicyNetwork(input_dim, hidden_dim, output_dim, use_lstm, use_dropout, use_layer_norm)
            for _ in range(num_experts)
        ])
    
    def forward(self, x, expert_index, hidden_state=None):
        """
        Args:
            x (Tensor): Observation of shape (batch_size, input_dim).
            expert_index (int): The index of the expert to use.
            hidden_state (optional): LSTM hidden state for the selected expert.
            
        Returns:
            action_probs (Tensor): (batch_size, output_dim) action probability distribution.
            hidden_state: LSTM hidden state from the selected expert.
        """
        if isinstance(expert_index, tuple):
        # Use the first expert in the tuple by default
            expert_index = expert_index[0]
        if expert_index < 0 or expert_index >= self.num_experts:
            raise ValueError(f"Expert index {expert_index} is out of range.")
        expert = self.experts[expert_index]
        return expert(x, hidden_state)

    
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_dropout=True, use_layer_norm=True):
        super(ValueNetwork, self).__init__()
        self.use_dropout = use_dropout
        self.use_layer_norm = use_layer_norm

        # Enhanced layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Regularization
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.3)
        if self.use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(hidden_dim)
            self.layer_norm2 = nn.LayerNorm(hidden_dim)
            self.layer_norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        if self.use_layer_norm:
            x = self.layer_norm1(x)
        if self.use_dropout:
            x = self.dropout(x)
        
        x = F.gelu(self.fc2(x))
        if self.use_layer_norm:
            x = self.layer_norm2(x)
        if self.use_dropout:
            x = self.dropout(x)
        
        x = F.gelu(self.fc3(x))
        if self.use_layer_norm:
            x = self.layer_norm3(x)
        if self.use_dropout:
            x = self.dropout(x)
        
        state_value = self.value_head(x)
        return state_value

class OpponentBehaviorPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2, memory_dim=0):
        """
        memory_dim: Dimension of the transformer-based memory embedding.
        """
        super(OpponentBehaviorPredictor, self).__init__()
        # The new input dimension is the sum of the original input and the memory embedding.
        combined_dim = input_dim + memory_dim
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_layer = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(0.3)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim // 2)

    def forward(self, x, memory_embedding):
        # Expect that memory_embedding is provided (for training OBP).
        # Concatenate along the last dimension.
        x = torch.cat([x, memory_embedding], dim=-1)
        x = F.gelu(self.fc1(x))
        x = self.layer_norm1(x)
        x = self.dropout(x)
        
        x = F.gelu(self.fc2(x))
        x = self.layer_norm2(x)
        x = self.dropout(x)
        
        x = F.gelu(self.fc3(x))
        x = self.layer_norm3(x)
        x = self.dropout(x)
        
        logits = self.output_layer(x)
        return logits
    
class PositionalEncoding(nn.Module):
    """
    Implements the standard sinusoidal positional encoding.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)  # shape: (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encodings added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class StrategyTransformer(nn.Module):
    """
    Transformer model that compresses sequences of action tokens into a fixed-size strategy embedding.
    """
    def __init__(
        self,
        num_tokens,             # Vocabulary size for action tokens.
        token_embedding_dim,    # Dimension of token embeddings.
        nhead,                  # Number of attention heads.
        num_layers,             # Number of Transformer encoder layers.
        strategy_dim,           # Desired dimension for the final strategy embedding (e.g. 5 or 10).
        num_classes=25,         # Number of classes for the classification head.
        dropout=0.1,
        use_cls_token=True      # If True, a learnable [CLS] token is prepended for pooling.
    ):
        super(StrategyTransformer, self).__init__()
        self.use_cls_token = use_cls_token
        self.token_embedding = nn.Embedding(num_tokens, token_embedding_dim)
        self.pos_encoder = PositionalEncoding(token_embedding_dim, dropout=dropout)

        if self.use_cls_token:
            # Learnable classification token that will serve as the summary representation.
            self.cls_token = nn.Parameter(torch.zeros(1, 1, token_embedding_dim))

        # ✅ Update: Enable batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_embedding_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Projects the pooled output into the fixed-size strategy embedding.
        self.strategy_head = nn.Linear(token_embedding_dim, strategy_dim)
        # Classification head used only during training.
        self.classification_head = nn.Linear(strategy_dim, num_classes)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        if self.use_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)
        self.strategy_head.weight.data.uniform_(-initrange, initrange)
        self.strategy_head.bias.data.zero_()
        self.classification_head.weight.data.uniform_(-initrange, initrange)
        self.classification_head.bias.data.zero_()

    def forward(self, src):
        """
        Args:
            src: Tensor of shape (batch_size, seq_length) containing token indices.
        Returns:
            strategy_embedding: Tensor of shape (batch_size, strategy_dim) – the compressed representation.
            classification_logits: Tensor of shape (batch_size, num_classes) for training purposes, or None if classification_head is disabled.
        """
        batch_size = src.size(0)
        
        # Embed tokens: (batch_size, seq_length, token_embedding_dim)
        x = self.token_embedding(src)

        if self.use_cls_token:
            # Prepend the [CLS] token to each sequence.
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, token_embedding_dim)
            x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, 1 + seq_length, token_embedding_dim)

        # Add positional encoding.
        x = self.pos_encoder(x)

        # ✅ Remove transpose(0,1), since batch_first=True now
        encoded = self.transformer_encoder(x)  # Now works with (batch_size, seq_length, token_embedding_dim)

        if self.use_cls_token:
            pooled = encoded[:, 0, :]  # Use CLS token representation
        else:
            pooled = encoded.mean(dim=1)  # Mean pooling over sequence

        strategy_embedding = self.strategy_head(pooled)

        if self.classification_head is not None:
            classification_logits = self.classification_head(strategy_embedding)
        else:
            classification_logits = None

        return strategy_embedding, classification_logits