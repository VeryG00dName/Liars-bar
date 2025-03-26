# src/model/models.py
import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


class StackedObservationConvModel(nn.Module):
    """
    Neural network that processes a stack of previous observations using 1D convolutions.
    
    Takes in a tensor of shape (batch_size, N, obs_dim) where:
    - batch_size: Number of examples in the batch
    - N: Number of historical observations to include
    - obs_dim: Dimension of each observation
    
    Features dual policy and value heads with a gating network to decide which head to use.
    """
    def __init__(self, obs_dim, num_actions, hidden_dim=256, num_obs_stack=50):
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
        
        # Two policy heads - output action logits
        self.policy_head1 = nn.Linear(hidden_dim, num_actions)
        self.policy_head2 = nn.Linear(hidden_dim, num_actions)
        
        # Two value heads - output state value estimates
        self.value_head1 = nn.Linear(hidden_dim, 1)
        self.value_head2 = nn.Linear(hidden_dim, 1)
        
        # Gating network - determines which policy/value head to use
        self.gating_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 2),  # Outputs 2 values, one for each head
            nn.Softmax(dim=1)  # Normalize to probabilities that sum to 1
        )
        
        # Next observation prediction head
        self.next_obs_head = nn.Linear(hidden_dim, obs_dim)
    
    def forward(self, x):
        """
        Args:
            x: Stacked observations tensor of shape (batch_size, N, obs_dim)
        
        Returns:
            policy_logits: Action logits of shape (batch_size, num_actions)
            state_value: State value of shape (batch_size, 1)
            next_obs_pred: Predicted next observation of shape (batch_size, obs_dim)
            gate_weights: Gate probabilities of shape (batch_size, 2)
        """
        # Process through conv layers
        x = self.conv_layers(x)  # Output: (batch_size, hidden_dim, 1)
        x = x.squeeze(-1)        # Remove last dimension: (batch_size, hidden_dim)
        
        # Process through fully connected layers
        features = self.fc_layers(x)
        
        # Get gate probabilities
        gate_weights = self.gating_network(features)  # shape: (batch_size, 2)
        
        # Get policy logits from both heads
        policy_logits1 = self.policy_head1(features)
        policy_logits2 = self.policy_head2(features)
        
        # Get state values from both heads
        state_value1 = self.value_head1(features)
        state_value2 = self.value_head2(features)
        
        # Blend the policy logits and state values using the gate weights
        policy_logits = gate_weights[:, 0:1] * policy_logits1 + gate_weights[:, 1:2] * policy_logits2
        state_value = gate_weights[:, 0:1] * state_value1 + gate_weights[:, 1:2] * state_value2
        
        # Next observation prediction
        next_obs_pred = self.next_obs_head(features)
        
        return policy_logits, state_value, next_obs_pred, gate_weights