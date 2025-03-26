# src/models/shen_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BeliefSpacePolicy(nn.Module):
    """
    Policy network that takes both observation and belief over opponent types as input.
    Maps combined (belief + observation) to action logits and state value.
    """
    def __init__(self, belief_dim, obs_dim, hidden_dim, output_dim):
        super().__init__()
        # Combined belief and observation as input
        self.network = nn.Sequential(
            nn.Linear(belief_dim + obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Policy head (output action logits)
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        
        # Value head for critic
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, 1)
        )
    
    def forward(self, obs, belief):
        """
        Forward pass through the belief-space policy network.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            belief: Belief tensor of shape (batch_size, belief_dim)
            
        Returns:
            action_logits: Policy logits of shape (batch_size, output_dim)
            state_value: Value prediction of shape (batch_size, 1)
            game_state_pred: Game state prediction (if game_state_head is set)
        """
        # Concatenate belief and observation
        combined = torch.cat([obs, belief], dim=1)
        
        # Process through shared network
        features = self.network(combined)
        
        # Get action logits from policy head
        action_logits = self.policy_head(features)
        
        # Get state value from value head
        state_value = self.value_head(features)
    
        return action_logits, state_value


class OpponentBeliefModel(nn.Module):
    """
    Model for updating beliefs about opponent types based on observations.
    Uses Bayesian-inspired updates to maintain a belief distribution.
    """
    def __init__(self, obs_dim, num_opponent_types, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Network to update belief based on observation and current belief
        self.belief_update = nn.Sequential(
            nn.Linear(hidden_dim + num_opponent_types, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_opponent_types)
        )
    
    def forward(self, obs, current_belief):
        """
        Forward pass to update belief based on new observation.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            current_belief: Current belief tensor of shape (batch_size, num_opponent_types)
            
        Returns:
            updated_belief: Updated belief of shape (batch_size, num_opponent_types)
        """
        # Encode observation
        obs_features = self.encoder(obs)
        
        # Combine with current belief
        combined = torch.cat([obs_features, current_belief], dim=1)
        
        # Output logits for belief update
        belief_logits = self.belief_update(combined)
        
        # Convert to probability distribution
        updated_belief = F.softmax(belief_logits, dim=1)
        
        return updated_belief