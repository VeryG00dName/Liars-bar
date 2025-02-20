import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Policy Network: Specialized Agent (No Auxiliary Classifier)
# ----------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_lstm=True, use_dropout=True, use_layer_norm=True):
        super(PolicyNetwork, self).__init__()
        self.use_lstm = use_lstm
        self.use_dropout = use_dropout
        self.use_layer_norm = use_layer_norm

        # Core layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        if self.use_lstm:
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
            self.fc4 = nn.Linear(hidden_dim, output_dim)
        else:
            self.fc4 = nn.Linear(hidden_dim, output_dim)

        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.3)
        if self.use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(hidden_dim)
            self.layer_norm2 = nn.LayerNorm(hidden_dim)
            self.layer_norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, hidden_state=None):
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
        
        if self.use_lstm:
            # Add sequence dimension if needed
            x = x.unsqueeze(1)
            x, hidden_state = self.lstm(x, hidden_state)
            x = x.squeeze(1)
            action_logits = self.fc4(x)
        else:
            action_logits = self.fc4(x)
        
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs, hidden_state

# ----------------------------
# Value Network
# ----------------------------
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_dropout=True, use_layer_norm=True):
        super(ValueNetwork, self).__init__()
        self.use_dropout = use_dropout
        self.use_layer_norm = use_layer_norm
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
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
    
# ----------------------------
# Opponent Behavior Predictor
# ----------------------------
    
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
    
# ----------------------------
# Strategy Transformer for Opponent Memory Embedding
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class StrategyTransformer(nn.Module):
    """
    Transformer that compresses sequences of opponent events into a fixed-size strategy embedding.
    """
    def __init__(self, num_tokens, token_embedding_dim, nhead, num_layers, strategy_dim, dropout=0.1, use_cls_token=True):
        super(StrategyTransformer, self).__init__()
        self.use_cls_token = use_cls_token
        self.token_embedding = nn.Embedding(num_tokens, token_embedding_dim)
        self.pos_encoder = PositionalEncoding(token_embedding_dim, dropout=dropout)
        
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, token_embedding_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_embedding_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.strategy_head = nn.Linear(token_embedding_dim, strategy_dim)
        # Classification head removed as per updated design.
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        if self.use_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)
        self.strategy_head.weight.data.uniform_(-initrange, initrange)
        self.strategy_head.bias.data.zero_()

    def forward(self, src):
        # src: (batch, seq_len) token indices
        x = self.token_embedding(src)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoder(x)
        # Transformer expects (seq_len, batch, embedding_dim)
        x = x.transpose(0, 1)
        encoded = self.transformer_encoder(x)
        encoded = encoded.transpose(0, 1)
        if self.use_cls_token:
            pooled = encoded[:, 0, :]
        else:
            pooled = encoded.mean(dim=1)
        strategy_embedding = self.strategy_head(pooled)
        return strategy_embedding
