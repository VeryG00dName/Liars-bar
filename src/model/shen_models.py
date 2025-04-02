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
    Model for updating beliefs about opponent types based on sequences of opponent memory events.
    Uses a sequence model to process memory events and updates belief distribution.
    
    Enhanced with numerical stability safeguards to prevent NaN/Inf values.
    """
    def __init__(self, event_feature_dim=5, max_seq_length=400, hidden_dim=128, num_opponent_types=10):
        super().__init__()
        # Store dimensions for validation
        self.event_feature_dim = event_feature_dim  # 5 features per event
        self.max_seq_length = max_seq_length  # Up to 200 events
        self.num_opponent_types = num_opponent_types
        self.hidden_dim = hidden_dim
        
        # Event embedding layer
        self.event_embedding = nn.Linear(event_feature_dim, hidden_dim)
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),  # Bidirectional LSTM gives 2*hidden_dim
            nn.Tanh()
        )
        
        # Network to update belief based on sequence representation and current belief
        self.belief_update = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_opponent_types, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_opponent_types)
        )
    
    def forward(self, event_sequences, current_belief, sequence_lengths=None):
        """
        Forward pass to update belief based on sequences of opponent memory events.
        
        Args:
            event_sequences: Tensor of shape (batch_size, seq_length, event_feature_dim)
                containing sequences of event features
            current_belief: Current belief tensor of shape (batch_size, num_opponent_types)
            sequence_lengths: Optional tensor of shape (batch_size) containing actual
                length of each sequence, used for packing/padding
            
        Returns:
            updated_belief: Updated belief of shape (batch_size, num_opponent_types)
        """
        batch_size = event_sequences.size(0)
        seq_length = event_sequences.size(1)
        
        # Handle empty sequences gracefully
        if seq_length == 0:
            # Just return the current belief with some noise to encourage exploration
            noise = torch.randn_like(current_belief) * 0.01
            return current_belief + noise
        
        # Apply numeric safeguards
        event_sequences = torch.nan_to_num(event_sequences, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize belief to ensure it sums to 1
        belief_sum = current_belief.sum(dim=1, keepdim=True)
        current_belief = torch.where(
            belief_sum > 0, 
            current_belief / belief_sum,
            torch.ones_like(current_belief) / current_belief.size(1)
        )
        
        # Embed each event in the sequence
        # (batch_size, seq_length, event_feature_dim) -> (batch_size, seq_length, hidden_dim)
        embedded_events = self.event_embedding(event_sequences)
        embedded_events = F.gelu(embedded_events)
        
        # Handle variable-length sequences
        if sequence_lengths is not None:
            # Pack the sequences for efficient processing
            packed_events = nn.utils.rnn.pack_padded_sequence(
                embedded_events, 
                sequence_lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
            
            # Process through LSTM
            packed_outputs, (hidden, _) = self.lstm(packed_events)
            
            # Unpack the sequences
            lstm_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            # If no sequence lengths provided, just process through LSTM
            lstm_outputs, (hidden, _) = self.lstm(embedded_events)
        
        # Apply attention mechanism to focus on relevant events
        attention_weights = self.attention(lstm_outputs)
        
        # Normalize attention weights
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention to get a weighted sum of the LSTM outputs
        # (batch_size, seq_length, hidden_dim*2) * (batch_size, seq_length, 1) -> (batch_size, hidden_dim*2)
        context_vector = torch.sum(lstm_outputs * attention_weights, dim=1)
        
        # Combine with current belief
        combined = torch.cat([context_vector, current_belief], dim=1)
        
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