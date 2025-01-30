# src/model/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_lstm=False, use_dropout=False, use_layer_norm=False):
        """
        Initializes the Policy Network.
        
        Args:
            input_dim (int): Dimension of input observations.
            hidden_dim (int): Number of units in hidden layers.
            output_dim (int): Number of possible actions.
            use_lstm (bool): Whether to include LSTM layers for sequential data.
            use_dropout (bool): Whether to apply dropout for regularization.
            use_layer_norm (bool): Whether to apply layer normalization.
        """
        super(PolicyNetwork, self).__init__()
        self.use_lstm = use_lstm
        self.use_dropout = use_dropout
        self.use_layer_norm = use_layer_norm

        # Define the network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if self.use_lstm:
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
        else:
            self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Optional components
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.2)
        if self.use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(hidden_dim)
            self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, hidden_state=None):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input observation tensor.
            hidden_state (tuple): Hidden and cell states for LSTM.
        
        Returns:
            torch.Tensor: Action probabilities.
            tuple: Updated hidden and cell states (if LSTM is used).
        """
        x = F.relu(self.fc1(x))
        if self.use_layer_norm:
            x = self.layer_norm1(x)
        if self.use_dropout:
            x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        if self.use_layer_norm:
            x = self.layer_norm2(x)
        if self.use_dropout:
            x = self.dropout(x)
        
        if self.use_lstm:
            # Reshape for LSTM: (batch, seq_len, input_size)
            x = x.unsqueeze(1)  # Assuming seq_len=1 for each step
            x, hidden_state = self.lstm(x, hidden_state)
            x = x.squeeze(1)  # Remove seq_len dimension
            action_logits = self.fc3(x)
            action_probs = F.softmax(action_logits, dim=-1)
            return action_probs, hidden_state
        else:
            action_logits = self.fc3(x)
            action_probs = F.softmax(action_logits, dim=-1)
            return action_probs, None  # No hidden state

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_dropout=False, use_layer_norm=False):
        """
        Initializes the Value Network.
        
        Args:
            input_dim (int): Dimension of input observations.
            hidden_dim (int): Number of units in hidden layers.
            use_dropout (bool): Whether to apply dropout for regularization.
            use_layer_norm (bool): Whether to apply layer normalization.
        """
        super(ValueNetwork, self).__init__()
        self.use_dropout = use_dropout
        self.use_layer_norm = use_layer_norm

        # Define the network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Optional components
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.2)
        if self.use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(hidden_dim)
            self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input observation tensor.
        
        Returns:
            torch.Tensor: State value estimate.
        """
        x = F.relu(self.fc1(x))
        if self.use_layer_norm:
            x = self.layer_norm1(x)
        if self.use_dropout:
            x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        if self.use_layer_norm:
            x = self.layer_norm2(x)
        if self.use_dropout:
            x = self.dropout(x)
        
        state_value = self.value_head(x)
        return state_value  # Output a single scalar value

class OpponentBehaviorPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        """
        Initializes the Opponent Behavior Predictor.
        
        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Number of units in hidden layers.
            output_dim (int): Number of output classes (e.g., lie or not).
        """
        super(OpponentBehaviorPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input feature tensor.
        
        Returns:
            torch.Tensor: Logits for each class.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.output_layer(x)
        return logits  # For CrossEntropyLoss
