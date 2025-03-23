# src/model/rebel_models.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CFRValueNetwork(nn.Module):
    """
    Estimates counterfactual values for belief states and outputs counterfactual regret estimates.
    Updated to handle public/private belief separation.
    """
    def __init__(self, input_dim, belief_dim, hidden_dim, action_dim, output_dim=1):
        super(CFRValueNetwork, self).__init__()
        
        # Determine dimensions for public and private parts of the observation
        self.public_dim = input_dim - 2  # Subtract dimensions for player's hand
        self.private_dim = 2  # Player's hand (table cards, non-table cards)
        self.action_dim = action_dim
        
        # Total input dimension includes public obs, private obs, and belief state
        combined_dim = self.public_dim + self.private_dim + belief_dim
        
        # Shared feature extractor
        self.shared_network = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Separate heads for value and regret estimation
        self.value_head = nn.Linear(hidden_dim, output_dim)
        self.regret_head = nn.Linear(hidden_dim, action_dim)
        
        # Add a variance estimation head for early termination decisions
        self.variance_head = nn.Linear(hidden_dim, output_dim)
        
        # Cache for previously computed values
        self._value_cache = {}
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Orthogonal initialization for better gradient flow
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs, beliefs):
        """
        Forward pass through the value network.
        Optimized for computational efficiency.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            beliefs: Belief state tensor [batch_size, belief_dim]
            
        Returns:
            value: Estimated counterfactual value [batch_size, 1]
            regrets: Counterfactual regret estimates [batch_size, action_dim]
            variance: Estimated variance of the value [batch_size, 1]
        """
        # Check cache for prior computation results during evaluation
        if not self.training:
            cache_key = None
            if isinstance(obs, torch.Tensor) and isinstance(beliefs, torch.Tensor):
                # Create a cache key from observation and beliefs
                obs_hash = hash(obs.detach().cpu().numpy().tobytes())
                belief_hash = hash(beliefs.detach().cpu().numpy().tobytes())
                cache_key = hash((obs_hash, belief_hash))
                
                if cache_key in self._value_cache:
                    return self._value_cache[cache_key]
        
        # Flatten beliefs for input to network
        batch_size = obs.size(0)
        beliefs_flat = beliefs.reshape(batch_size, -1)
        
        # Concatenate observations and beliefs
        combined_input = torch.cat([obs, beliefs_flat], dim=-1)
        
        # Forward pass through shared network
        features = self.shared_network(combined_input)
        
        # Get outputs from all heads
        value = self.value_head(features)
        regrets = self.regret_head(features)
        variance = torch.abs(self.variance_head(features))  # Make sure variance is positive
        
        # Cache result during evaluation
        if not self.training and cache_key is not None:
            # Limit cache size
            if len(self._value_cache) > 1000:
                self._value_cache = {}
            self._value_cache[cache_key] = (value, regrets, variance)
        
        return value, regrets, variance
    
    def evaluate_public_state(self, public_obs, beliefs):
        """
        Evaluate the value of a public state without private information.
        Optimized for subgame solving at public state nodes.
        
        Args:
            public_obs: Public observation tensor [batch_size, public_dim]
            beliefs: Belief state tensor [batch_size, belief_dim]
            
        Returns:
            value: Estimated value [batch_size, 1]
            regrets: Estimated regrets [batch_size, action_dim]
            variance: Estimated variance [batch_size, 1]
        """
        # Create dummy private observation (zeros)
        batch_size = public_obs.size(0)
        device = public_obs.device
        private_obs = torch.zeros(batch_size, self.private_dim, device=device)
        
        # Combine with public observation
        combined_obs = torch.cat([private_obs, public_obs], dim=1)
        
        # Use regular forward pass
        return self.forward(combined_obs, beliefs)

    def batch_evaluate(self, observations, beliefs_list):
        """
        Efficiently evaluate a batch of observations and beliefs.
        
        Args:
            observations: List of observation tensors
            beliefs_list: List of belief state tensors
            
        Returns:
            List of (value, regrets, variance) tuples
        """
        # Batch size handling
        batch_size = 16  # Configurable based on GPU memory
        results = []
        
        # Process in batches for efficiency
        for i in range(0, len(observations), batch_size):
            batch_obs = torch.cat(observations[i:i+batch_size], dim=0)
            batch_beliefs = torch.cat(beliefs_list[i:i+batch_size], dim=0)
            
            with torch.no_grad():
                batch_values, batch_regrets, batch_variances = self.forward(batch_obs, batch_beliefs)
            
            # Split results back to individual entries
            for j in range(len(batch_values)):
                results.append((
                    batch_values[j:j+1],
                    batch_regrets[j:j+1],
                    batch_variances[j:j+1]
                ))
                
            # Early termination if we've processed everything
            if i + batch_size >= len(observations):
                break
                
        return results

class RebelPolicyNetwork(nn.Module):
    """
    Policy network specialized for belief-based decision making.
    Optimized for computational efficiency.
    """
    def __init__(self, obs_dim, belief_dim, hidden_dim, action_dim, 
                 use_residual=True, use_layer_norm=True, dropout_rate=0.2):
        super(RebelPolicyNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.belief_dim = belief_dim
        self.action_dim = action_dim
        self.use_residual = use_residual
        
        # Determine dimensions for public and private parts of the observation
        self.public_dim = obs_dim - 2  # Subtract dimensions for player's hand
        self.private_dim = 2  # Player's hand (table cards, non-table cards)
        
        # Process public observation features
        self.public_encoder = nn.Sequential(
            nn.Linear(self.public_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2) if use_layer_norm else nn.Identity(),
            nn.Dropout(dropout_rate)
        )
        
        # Process private observation features
        self.private_encoder = nn.Sequential(
            nn.Linear(self.private_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2) if use_layer_norm else nn.Identity(),
            nn.Dropout(dropout_rate)
        )
        
        # Process belief features
        self.belief_encoder = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.Dropout(dropout_rate)
        )
        
        # Joint processing of all features
        self.joint_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.Dropout(dropout_rate)
        )
        
        # Action prediction head with residual connection
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, action_dim)
        )
        self.residual_proj = nn.Linear(hidden_dim, action_dim)
        
        # Value prediction (auxiliary output)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2) if use_layer_norm else nn.Identity(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Auxiliary search policy head (for integrating search feedback)
        self.search_policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Cache for forward pass results
        self._policy_cache = {}
        
        self._init_weights()
    
    def _init_weights(self):
        # Orthogonal initialization for better gradient flow
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def split_observation(self, x):
        """
        Split observation into public and private components.
        Optimized for efficiency.
        
        Args:
            x: Full observation tensor [batch_size, obs_dim]
            
        Returns:
            (public_obs, private_obs): Tuple of public and private observation tensors
        """
        # Handle tensor input
        if isinstance(x, torch.Tensor):
            # First two elements are the player's hand information
            private_obs = x[:, :self.private_dim]
            public_obs = x[:, self.private_dim:]
            return public_obs, private_obs
            
        # Handle numpy input
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float()
            if x_tensor.dim() == 1:
                x_tensor = x_tensor.unsqueeze(0)
            private_obs = x_tensor[:, :self.private_dim]
            public_obs = x_tensor[:, self.private_dim:]
            return public_obs, private_obs
        
        # Fallback
        x_tensor = torch.tensor(x, dtype=torch.float)
        if x_tensor.dim() == 1:
            x_tensor = x_tensor.unsqueeze(0)
        private_obs = x_tensor[:, :self.private_dim]
        public_obs = x_tensor[:, self.private_dim:]
        return public_obs, private_obs
    
    def forward(self, obs, beliefs=None, hidden_state=None):
        """
        Forward pass through the policy network.
        Optimized for computational efficiency.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            beliefs: Belief state tensor [batch_size, belief_dim] or None
            hidden_state: Not used, included for API compatibility
            
        Returns:
            action_probs: Action probabilities [batch_size, action_dim]
            value: State value estimate [batch_size, 1]
            search_policy_probs: Auxiliary search-derived action probabilities [batch_size, action_dim]
        """
        # Check cache during evaluation
        if not self.training:
            cache_key = None
            if isinstance(obs, torch.Tensor):
                obs_hash = hash(obs.detach().cpu().numpy().tobytes())
                belief_hash = hash(beliefs.detach().cpu().numpy().tobytes()) if beliefs is not None else 0
                cache_key = hash((obs_hash, belief_hash))
                
                if cache_key in self._policy_cache:
                    return self._policy_cache[cache_key]
        
        batch_size = obs.size(0)
        
        # Split observation into public and private parts
        public_obs, private_obs = self.split_observation(obs)
        
        # Process each part separately
        public_features = self.public_encoder(public_obs)
        private_features = self.private_encoder(private_obs)
        
        # Combine observation features
        obs_features = torch.cat([public_features, private_features], dim=1)
        
        # Process beliefs if provided
        if beliefs is not None:
            if beliefs.dim() > 2:
                beliefs_flat = beliefs.reshape(batch_size, -1)
            else:
                beliefs_flat = beliefs
            belief_features = self.belief_encoder(beliefs_flat)
            
            # Combine all features
            combined_features = torch.cat([obs_features, belief_features], dim=1)
            joint_features = self.joint_encoder(combined_features)
        else:
            # If no beliefs provided, duplicate observation features
            combined_features = torch.cat([obs_features, obs_features], dim=1)
            joint_features = self.joint_encoder(combined_features)
        
        # Action prediction with residual connection
        if self.use_residual:
            action_logits = self.action_head(joint_features) + self.residual_proj(joint_features)
        else:
            action_logits = self.action_head(joint_features)
        
        # Apply softmax for probabilities
        action_probs = F.softmax(action_logits, dim=1)
        
        # Value prediction
        value = self.value_head(joint_features)
        
        # Auxiliary search policy head
        search_policy_logits = self.search_policy_head(joint_features)
        search_policy_probs = F.softmax(search_policy_logits, dim=1)
        
        # Cache results during evaluation
        if not self.training and cache_key is not None:
            if len(self._policy_cache) > 1000:  # Limit cache size
                self._policy_cache = {}
            self._policy_cache[cache_key] = (action_probs, value, search_policy_probs)
        
        return action_probs, value, search_policy_probs
    
    def public_policy(self, public_obs, beliefs=None):
        """
        Compute policy based only on public information.
        Optimized for blueprint strategy generation.
        
        Args:
            public_obs: Public observation tensor
            beliefs: Belief state tensor or None
            
        Returns:
            action_probs: Action probabilities
            value: State value estimate
            search_policy_probs: Auxiliary search policy
        """
        # Check cache during evaluation
        if not self.training:
            cache_key = None
            if isinstance(public_obs, torch.Tensor):
                obs_hash = hash(public_obs.detach().cpu().numpy().tobytes())
                belief_hash = hash(beliefs.detach().cpu().numpy().tobytes()) if beliefs is not None else 0
                cache_key = hash(("public", obs_hash, belief_hash))
                
                if cache_key in self._policy_cache:
                    return self._policy_cache[cache_key]
        
        batch_size = public_obs.size(0)
        device = public_obs.device
        
        # Create dummy private observations (zeros)
        private_obs = torch.zeros(batch_size, self.private_dim, device=device)
        
        # Process public features
        public_features = self.public_encoder(public_obs)
        private_features = self.private_encoder(private_obs)
        
        # Combine observation features
        obs_features = torch.cat([public_features, private_features], dim=1)
        
        # Process beliefs if provided
        if beliefs is not None:
            if beliefs.dim() > 2:
                beliefs_flat = beliefs.reshape(batch_size, -1)
            else:
                beliefs_flat = beliefs
            belief_features = self.belief_encoder(beliefs_flat)
            
            # Combine features
            combined_features = torch.cat([obs_features, belief_features], dim=1)
            joint_features = self.joint_encoder(combined_features)
        else:
            # Fallback to using observation features twice
            combined_features = torch.cat([obs_features, obs_features], dim=1)
            joint_features = self.joint_encoder(combined_features)
        
        # Action prediction with residual connection
        if self.use_residual:
            action_logits = self.action_head(joint_features) + self.residual_proj(joint_features)
        else:
            action_logits = self.action_head(joint_features)
        
        # Apply softmax for probabilities
        action_probs = F.softmax(action_logits, dim=1)
        
        # Value prediction
        value = self.value_head(joint_features)
        
        # Auxiliary search policy head
        search_policy_logits = self.search_policy_head(joint_features)
        search_policy_probs = F.softmax(search_policy_logits, dim=1)
        
        # Cache results during evaluation
        if not self.training and cache_key is not None:
            if len(self._policy_cache) > 1000:  # Limit cache size
                self._policy_cache = {}
            self._policy_cache[cache_key] = (action_probs, value, search_policy_probs)
        
        return action_probs, value, search_policy_probs
    
    def act(self, observation, beliefs=None, temperature=1.0):
        """
        Choose an action based on observation and beliefs.
        Optimized for efficient action selection.
        
        Args:
            observation: Current observation tensor.
            beliefs: Current belief state.
            temperature: Temperature parameter for exploration (lower = more greedy)
            
        Returns:
            action: Selected action.
            action_prob: Probability of the selected action.
            state_value: Estimated state value.
        """
        with torch.no_grad():
            # Convert to tensor if needed
            if not isinstance(observation, torch.Tensor):
                observation = torch.FloatTensor(observation).unsqueeze(0)
            
            # Get policy and value
            action_probs, state_value, _ = self.forward(observation, beliefs)
            
            # Apply temperature
            if temperature != 1.0:
                # Use temperature to control exploration
                logits = torch.log(action_probs + 1e-10) / temperature
                action_probs = F.softmax(logits, dim=-1)
            
            # Sample from distribution
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
            action_prob = action_probs[0, action].item()
            
        return action, action_prob, state_value.item()

    def batch_forward(self, observations, beliefs_list=None):
        """
        Process multiple observations and beliefs efficiently in a batch.
        
        Args:
            observations: List of observation tensors
            beliefs_list: List of belief tensors (optional)
            
        Returns:
            List of (action_probs, value, search_policy_probs) tuples
        """
        batch_size = 16  # Configurable
        results = []
        
        # Create batches
        for i in range(0, len(observations), batch_size):
            # Get current batch
            batch_obs = torch.cat(observations[i:i+batch_size], dim=0)
            
            # Handle beliefs if provided
            if beliefs_list is not None:
                batch_beliefs = torch.cat(beliefs_list[i:i+batch_size], dim=0)
            else:
                batch_beliefs = None
                
            # Process batch
            with torch.no_grad():
                batch_probs, batch_values, batch_search_probs = self.forward(batch_obs, batch_beliefs)
                
            # Split results
            for j in range(len(batch_probs)):
                results.append((
                    batch_probs[j:j+1],
                    batch_values[j:j+1],
                    batch_search_probs[j:j+1]
                ))
                
            # Stop if we've processed everything
            if i + batch_size >= len(observations):
                break
                
        return results

class ActionProbabilityModel(nn.Module):
    """
    Neural model to predict action probabilities for all possible actions.
    Updated to work with transformer embeddings.
    """
    def __init__(self, input_dim=16, hidden_dim=128):  # Updated input_dim for new features (including 5-dim embeddings)
        super(ActionProbabilityModel, self).__init__()
        self.input_dim = input_dim

        # Adjust network to handle new input dimension
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 7)  # Output for all 7 possible actions
        )

    def forward(self, features):
        """
        Predict action probabilities from state features.

        Args:
            features: Tensor of state features including transformer embeddings

        Returns:
            Tensor of probabilities for all 7 possible actions
        """
        logits = self.network(features)
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def extract_features(self, action_type, count, hand, table_card,
                        hand_size, penalty_ratio, transformer_embeddings=None,
                        opponent_idx=None, last_action=None,
                        last_action_agent=None, last_action_bluff=None):
        """
        Extract relevant features using transformer embeddings instead of opponent memory.
        """
        features = []

        # Action type (one-hot encoded)
        is_play = 1.0 if action_type == "Play" else 0.0
        is_challenge = 1.0 if action_type == "Challenge" else 0.0
        features.extend([is_play, is_challenge])

        # Play count (normalized)
        play_count = count if count is not None else 0
        features.append(play_count / 3.0)  # Normalize to [0,1]

        # Hand size (normalized)
        features.append(hand_size / 5.0)  # Normalize to [0,1]

        # Penalty ratio
        features.append(penalty_ratio)

        # Table card information
        if hand and table_card:
            # Count table cards in hand
            table_card_count = sum(1 for c in hand if c == table_card or c == "Joker")
            non_table_card_count = hand_size - table_card_count

            # Normalized card counts
            features.append(table_card_count / max(1, hand_size))
            features.append(non_table_card_count / max(1, hand_size))
        else:
            features.extend([0.0, 0.0])

        # Last action information
        if last_action is not None:
            features.append(last_action / 6.0)  # Normalize to [0,1]

            # Was the last action a bluff?
            if last_action_bluff is not None:
                features.append(1.0 if last_action_bluff else 0.0)
            else:
                features.append(0.5)  # Unknown
        else:
            features.extend([0.0, 0.5])

        # Transformer embeddings for opponent
        if transformer_embeddings is not None and opponent_idx is not None:
            # Check if we have a list of embeddings or a flattened array
            if isinstance(transformer_embeddings, list) and opponent_idx < len(transformer_embeddings):
                # Get the specific opponent's embedding
                opponent_embedding = transformer_embeddings[opponent_idx]
                if isinstance(opponent_embedding, np.ndarray) or isinstance(opponent_embedding, torch.Tensor):
                    # Flatten if needed
                    if len(opponent_embedding.shape) > 1:
                        opponent_embedding = opponent_embedding.flatten()
                    features.extend(opponent_embedding)
                else:
                    # Default embedding if format isn't as expected
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # 5-dim default
            elif isinstance(transformer_embeddings, (np.ndarray, torch.Tensor)):
                # If it's a tensor with all embeddings combined
                # Extract the specific opponent's section
                num_opponents = len(transformer_embeddings) // 5  # Assuming 5-dim per opponent
                if opponent_idx < num_opponents:
                    start_idx = opponent_idx * 5
                    end_idx = start_idx + 5
                    opponent_embedding = transformer_embeddings[start_idx:end_idx]
                    features.extend(opponent_embedding)
                else:
                    # Default embedding if index out of range
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                # Default embedding if format isn't as expected
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            # Default transformer embedding if none provided
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # 5-dim default

        return torch.tensor(features, dtype=torch.float)
    
class ActionProbabilityDataCollector:
    """
    Collects data for training the action probability model.
    """
    def __init__(self):
        self.data = []
     
    def extract_features(self, action_type, count, hand, table_card, hand_size, 
                         penalty_ratio, transformer_embeddings=None, opponent_idx=None, 
                         last_action=None, last_action_agent=None, last_action_bluff=None):
        """
        Extract relevant features with transformer embeddings.
        """
        features = []
        
        # Action type (one-hot encoded)
        is_play = 1.0 if action_type == "Play" else 0.0
        is_challenge = 1.0 if action_type == "Challenge" else 0.0
        features.extend([is_play, is_challenge])
        
        # Play count (normalized)
        play_count = count if count is not None else 0
        features.append(play_count / 3.0)  # Normalize to [0,1]
        
        # Hand size (normalized)
        features.append(hand_size / 5.0)  # Normalize to [0,1]
        
        # Penalty ratio
        features.append(penalty_ratio)
        
        # Table card information
        if hand and table_card:
            # Count table cards in hand
            table_card_count = sum(1 for c in hand if c == table_card or c == "Joker")
            non_table_card_count = hand_size - table_card_count
            
            # Normalized card counts
            features.append(table_card_count / max(1, hand_size))
            features.append(non_table_card_count / max(1, hand_size))
        else:
            features.extend([0.0, 0.0])
        
        # Last action information
        if last_action is not None:
            features.append(last_action / 6.0)  # Normalize to [0,1]
            
            # Was the last action a bluff?
            if last_action_bluff is not None:
                features.append(1.0 if last_action_bluff else 0.0)
            else:
                features.append(0.5)  # Unknown
        else:
            features.extend([0.0, 0.5])
        
        # Transformer embeddings for opponent
        if transformer_embeddings is not None and opponent_idx is not None:
            # Check if we have a list of embeddings or a flattened array
            if isinstance(transformer_embeddings, list) and opponent_idx < len(transformer_embeddings):
                # Get the specific opponent's embedding
                opponent_embedding = transformer_embeddings[opponent_idx]
                if isinstance(opponent_embedding, (np.ndarray, torch.Tensor)):
                    # Flatten if needed
                    if hasattr(opponent_embedding, 'shape') and len(opponent_embedding.shape) > 1:
                        if isinstance(opponent_embedding, torch.Tensor):
                            opponent_embedding = opponent_embedding.flatten().detach().cpu().numpy()
                        else:
                            opponent_embedding = opponent_embedding.flatten()
                    features.extend(opponent_embedding)
                else:
                    # Default embedding if format isn't as expected
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # 5-dim default
            else:
                # Default embedding if index out of range
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            # Default transformer embedding if none provided
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # 5-dim default
            
        # Return as a list, not as a tensor
        return features

    def record_action(self, action_type, count, hand, table_card, was_bluff=None, 
                      hand_size=None, penalty_ratio=None, transformer_embeddings=None,
                      opponent_idx=None, specific_action=None, last_action=None,
                      last_action_agent=None, last_action_bluff=None):
        """
        Record an action with transformer embeddings instead of opponent memory.
        """
        # Determine specific action if not provided
        if specific_action is None:
            if action_type == "Challenge":
                specific_action = 6
            elif action_type == "Play" and count is not None:
                # Determine if this was a table or non-table play based on was_bluff
                if was_bluff is False:
                    # Truthful play of table cards
                    specific_action = count - 1  # Maps count 1,2,3 to actions 0,1,2
                elif was_bluff is True:
                    # Bluffing with non-table cards
                    specific_action = count + 2  # Maps count 1,2,3 to actions 3,4,5
                else:
                    # Unknown if bluff or not, cannot determine exact action
                    return
            else:
                # Cannot determine exact action
                return
        
        # Count cards in hand
        if hand_size is None:
            hand_size = len(hand) if hand else 0
            
        # Extract features using transformer embeddings
        features = self.extract_features(
            action_type=action_type,
            count=count,
            hand=hand,
            table_card=table_card,
            hand_size=hand_size,
            penalty_ratio=penalty_ratio,
            transformer_embeddings=transformer_embeddings,
            opponent_idx=opponent_idx,
            last_action=last_action,
            last_action_agent=last_action_agent,
            last_action_bluff=last_action_bluff
        )
        
        # Create target as one-hot encoded action
        target = [0.0] * 7
        if 0 <= specific_action < 7:  # Ensure action is valid
            target[specific_action] = 1.0
        
        self.data.append({
            'features': features,  # This is now a list, not a tensor
            'target': target,
            'meta': {
                'action_type': action_type,
                'count': count,
                'hand_size': hand_size,
                'was_bluff': was_bluff,
                'specific_action': specific_action,
                'last_action': last_action,
                'last_action_agent': last_action_agent
            }
        })
        
    def get_training_data(self):
        """Get collected data for training."""
        if not self.data:
            return None, None
            
        # Each item in self.data['features'] is already a list, not a tensor
        features = torch.tensor([d['features'] for d in self.data], dtype=torch.float)
        targets = torch.tensor([d['target'] for d in self.data], dtype=torch.float)
        return features, targets
        
    def record_action(self, action_type, count, hand, table_card, was_bluff=None, 
                  hand_size=None, penalty_ratio=None, transformer_embeddings=None,
                  opponent_idx=None, specific_action=None, last_action=None,
                  last_action_agent=None, last_action_bluff=None):
        """
        Record an action with transformer embeddings instead of opponent memory.
        """
        # Skip if we don't have the actual action and it can't be determined
        if specific_action is None:
            if action_type == "Challenge":
                specific_action = 6
            elif action_type == "Play" and count is not None:
                # Determine if this was a table or non-table play based on was_bluff
                if was_bluff is False:
                    # Truthful play of table cards
                    specific_action = count - 1  # Maps count 1,2,3 to actions 0,1,2
                elif was_bluff is True:
                    # Bluffing with non-table cards
                    specific_action = count + 2  # Maps count 1,2,3 to actions 3,4,5
                else:
                    # Unknown if bluff or not, cannot determine exact action
                    return
            else:
                # Cannot determine exact action
                return
        
        # Count cards in hand
        if hand_size is None:
            hand_size = len(hand) if hand else 0
            
        # Extract features using transformer embeddings
        features = self.extract_features(
            action_type=action_type,
            count=count,
            hand=hand,
            table_card=table_card,
            hand_size=hand_size,
            penalty_ratio=penalty_ratio,
            transformer_embeddings=transformer_embeddings,
            opponent_idx=opponent_idx,
            last_action=last_action,
            last_action_agent=last_action_agent,
            last_action_bluff=last_action_bluff
        )
        
        # Create target as one-hot encoded action
        target = [0.0] * 7
        if 0 <= specific_action < 7:  # Ensure action is valid
            target[specific_action] = 1.0
        
        self.data.append({
            'features': features,
            'target': target,
            'meta': {
                'action_type': action_type,
                'count': count,
                'hand_size': hand_size,
                'was_bluff': was_bluff,
                'specific_action': specific_action,
                'last_action': last_action,
                'last_action_agent': last_action_agent
            }
        })
        