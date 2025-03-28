# src/models/shen_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BeliefSpacePolicy(nn.Module):
    """
    Policy network that takes both observation and belief over opponent types as input.
    Maps combined (belief + observation) to action logits and state value.
    
    Enhanced with numerical stability safeguards to prevent NaN/Inf values.
    """
    def __init__(self, belief_dim, obs_dim, hidden_dim, output_dim):
        super().__init__()
        # Store dimensions for validation during forward pass
        self.belief_dim = belief_dim
        self.obs_dim = obs_dim
        self.total_input_dim = belief_dim + obs_dim
        
        self.network = nn.Sequential(
            nn.Linear(self.total_input_dim, hidden_dim),
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
        Forward pass through the belief-space policy network with numerical stability safeguards.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
            belief: Belief tensor of shape (batch_size, belief_dim)
            
        Returns:
            action_logits: Policy logits of shape (batch_size, output_dim)
            state_value: Value prediction of shape (batch_size, 1)
        """
        # Validate input dimensions
        if obs.size(1) + belief.size(1) != self.total_input_dim:
            # Handle dimension mismatch - try to adapt
            total_size = obs.size(1) + belief.size(1)
            if total_size > self.total_input_dim:
                # Truncate to match expected dimensions
                if obs.size(1) > self.obs_dim:
                    obs = obs[:, :self.obs_dim]
                if belief.size(1) > self.belief_dim:
                    belief = belief[:, :self.belief_dim]
            elif total_size < self.total_input_dim:
                # Pad with zeros to match expected dimensions
                missing = self.total_input_dim - total_size
                padding = torch.zeros((obs.size(0), missing), device=obs.device)
                # Append padding to belief (safer than observation)
                belief = torch.cat([belief, padding], dim=1)
        
        # Ensure both tensors have proper values (no NaN/Inf)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        belief = torch.nan_to_num(belief, nan=1.0/belief.size(1), posinf=1.0, neginf=0.0)
        
        # Normalize belief tensor to ensure it sums to 1
        belief_sum = belief.sum(dim=1, keepdim=True)
        belief = torch.where(belief_sum > 0, belief / belief_sum, 
                            torch.ones_like(belief) / belief.size(1))
        
        # Concatenate belief and observation
        combined = torch.cat([obs, belief], dim=1)
        
        # Process through network with additional safeguards
        features = self.network(combined)
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Get action logits from policy head
        action_logits = self.policy_head(features)
        action_logits = torch.clamp(action_logits, min=-10.0, max=10.0)  # Prevent extreme values
        
        # Get state value from value head
        state_value = self.value_head(features)
        state_value = torch.clamp(state_value, min=-100.0, max=100.0)  # Reasonable value range
        
        return action_logits, state_value


class OpponentBeliefModel(nn.Module):
    """
    Model for updating beliefs about opponent types based on observations.
    Uses Bayesian-inspired updates to maintain a belief distribution.
    
    Enhanced with numerical stability safeguards to prevent NaN/Inf values.
    """
    def __init__(self, obs_dim, num_opponent_types, hidden_dim):
        super().__init__()
        # Store dimensions for validation
        self.obs_dim = obs_dim
        self.num_opponent_types = num_opponent_types
        
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
        # Validate and handle input dimensions
        if obs.size(1) != self.obs_dim:
            if obs.size(1) > self.obs_dim:
                # Truncate observation
                obs = obs[:, :self.obs_dim]
            else:
                # Pad observation
                padding = torch.zeros((obs.size(0), self.obs_dim - obs.size(1)), device=obs.device)
                obs = torch.cat([obs, padding], dim=1)
        
        if current_belief.size(1) != self.num_opponent_types:
            if current_belief.size(1) > self.num_opponent_types:
                # Truncate belief
                current_belief = current_belief[:, :self.num_opponent_types]
            else:
                # Pad belief
                padding = torch.zeros((current_belief.size(0), 
                                      self.num_opponent_types - current_belief.size(1)), 
                                      device=current_belief.device)
                current_belief = torch.cat([current_belief, padding], dim=1)
        
        # Apply numeric safeguards
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize belief to ensure it sums to 1
        belief_sum = current_belief.sum(dim=1, keepdim=True)
        current_belief = torch.where(
            belief_sum > 0, 
            current_belief / belief_sum,
            torch.ones_like(current_belief) / current_belief.size(1)
        )
        
        # Encode observation
        obs_features = self.encoder(obs)
        obs_features = torch.nan_to_num(obs_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Combine with current belief
        combined = torch.cat([obs_features, current_belief], dim=1)
        
        # Output logits for belief update
        belief_logits = self.belief_update(combined)
        belief_logits = torch.clamp(belief_logits, min=-10.0, max=10.0)  # Prevent extreme values
        
        # Convert to probability distribution with added stability
        # Add a small epsilon to prevent underflow
        belief_logits_stable = belief_logits - belief_logits.max(dim=1, keepdim=True)[0]
        belief_logits_stable = torch.clamp(belief_logits_stable, min=-20.0)  # Prevent extreme negative values
        
        # Use softmax with additional safeguards
        exp_logits = torch.exp(belief_logits_stable)
        exp_sum = exp_logits.sum(dim=1, keepdim=True)
        updated_belief = exp_logits / (exp_sum + 1e-10)  # Add small epsilon to prevent division by zero
        
        # Final safety check
        updated_belief = torch.nan_to_num(updated_belief, nan=1.0/self.num_opponent_types, 
                                         posinf=1.0, neginf=0.0)
        
        # Renormalize if needed
        belief_sum = updated_belief.sum(dim=1, keepdim=True)
        updated_belief = torch.where(
            belief_sum > 0,
            updated_belief / belief_sum,
            torch.ones_like(updated_belief) / updated_belief.size(1)
        )
        
        return updated_belief