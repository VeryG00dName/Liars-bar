# src/model/belief_models.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BeliefStateModel(nn.Module):
    """
    Models probability distributions over opponents' hands based on game history.
    Now with added support for multi-agent belief correlation and physical constraints.
    """
    def __init__(self, input_dim, hidden_dim, deck_size, num_players, 
                 use_dropout=True, use_layer_norm=True, update_mode='multiplicative',
                 cards_per_type=None):
        super(BeliefStateModel, self).__init__()
        self.deck_size = deck_size  # Total number of cards in deck
        self.num_players = num_players
        self.card_types = 2  # Simplified: (table_card, non_table_card)
        self.update_mode = update_mode  # Options: 'multiplicative' or 'weighted'
        
        # Card distribution information - critically important for correlation constraints
        self.cards_per_type = cards_per_type or [10, 10]  # Default: 10 table cards, 10 non-table cards
        assert sum(self.cards_per_type) == self.deck_size, "Cards per type must sum to deck size"
        
        # Determine dimensions for public and private parts of the observation
        self.public_dim = input_dim - 2  # Subtract dimensions for player's hand
        self.private_dim = 2  # Player's hand (table cards, non-table cards)
        
        # Separate encoders for public and private information
        self.public_encoder = nn.Sequential(
            nn.Linear(self.public_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.Dropout(0.2) if use_dropout else nn.Identity(),
        )
        
        self.private_encoder = nn.Sequential(
            nn.Linear(self.private_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2) if use_layer_norm else nn.Identity(),
            nn.Dropout(0.2) if use_dropout else nn.Identity(),
        )
        
        # Joint encoder that combines public and private features
        self.joint_encoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.Dropout(0.2) if use_dropout else nn.Identity(),
        )
        
        # Belief update network – for modeling correlations
        correlation_dim = (num_players - 1) * self.card_types
        self.belief_update = nn.Linear(hidden_dim, correlation_dim * correlation_dim)
        
        # Initialize with a uniform prior
        self.register_buffer('prior_belief', torch.ones(1, num_players - 1, self.card_types) / self.card_types)
        
        # If using weighted blending, add a learnable parameter alpha
        if self.update_mode == 'weighted':
            self.alpha = nn.Parameter(torch.tensor(0.5))

        # Cached tensors for faster computation
        self.register_buffer('total_cards_tensor', torch.tensor(self.cards_per_type, dtype=torch.float))
        
        # Caching structure for belief updates
        self._belief_cache = {}
        self._update_counter = 0
    
    def split_observation(self, x):
        """
        Split observation into public and private components.
        Optimized for faster tensor handling.
        
        Args:
            x: Full observation tensor [batch_size, obs_dim] or numpy array or list
                
        Returns:
            (public_obs, private_obs): Tuple of public and private observation tensors
        """
        # Fast path for tensor input
        if isinstance(x, torch.Tensor):
            # Add batch dimension if needed
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            # First two elements are the player's hand information
            private_obs = x[:, :self.private_dim]
            public_obs = x[:, self.private_dim:]
            return public_obs, private_obs
            
        # Handle numpy array input
        if isinstance(x, np.ndarray):
            # Convert to tensor
            x_tensor = torch.from_numpy(x).float()
            if x_tensor.dim() == 1:
                x_tensor = x_tensor.unsqueeze(0)
                
            # Split observation
            private_obs = x_tensor[:, :self.private_dim]
            public_obs = x_tensor[:, self.private_dim:]
            return public_obs, private_obs
        
        # Fallback for other input types
        x_tensor = torch.tensor(x, dtype=torch.float)
        if x_tensor.dim() == 1:
            x_tensor = x_tensor.unsqueeze(0)
            
        private_obs = x_tensor[:, :self.private_dim]
        public_obs = x_tensor[:, self.private_dim:]
        return public_obs, private_obs
    
    def apply_physical_constraints_fast(self, beliefs, private_hand=None):
        """
        Efficient vectorized implementation of physical constraints.
        
        Args:
            beliefs: Belief tensor [batch_size, num_opponents, card_types]
            private_hand: Observer's hand counts [batch_size, card_types] or None
            
        Returns:
            Constrained beliefs respecting physical card limits
        """
        # Early return for None or small batch (optimization)
        if beliefs is None:
            return beliefs
            
        batch_size = beliefs.size(0)
        device = beliefs.device
        
        # Get cached tensor if possible
        if not hasattr(self, 'total_cards_tensor') or self.total_cards_tensor.device != device:
            self.total_cards_tensor = torch.tensor(self.cards_per_type, device=device)
        total_cards = self.total_cards_tensor
        
        # Handle private hand properly
        if private_hand is not None:
            # Process non-tensor private_hand
            if not isinstance(private_hand, torch.Tensor):
                if isinstance(private_hand, np.ndarray):
                    private_hand = torch.from_numpy(private_hand).float().to(device)
                else:
                    private_hand = torch.tensor(private_hand, dtype=torch.float).to(device)
                
                # Add batch dimension if needed
                if private_hand.dim() == 1:
                    private_hand = private_hand.unsqueeze(0)
            
            # Ensure correct shape and device
            if private_hand.device != device:
                private_hand = private_hand.to(device)
                
            # Make sure we have correct dimensions
            if private_hand.size(1) != self.card_types:
                if private_hand.size(1) >= 2:
                    private_hand = private_hand[:, :self.card_types]
                else:
                    private_hand = torch.zeros(batch_size, self.card_types, device=device)
            
            # Calculate cards remaining after accounting for private hand
            remaining_cards = total_cards.unsqueeze(0).expand(batch_size, -1) - private_hand
        else:
            # Just use total cards if no private hand
            remaining_cards = total_cards.unsqueeze(0).expand(batch_size, -1)
        
        # Vectorized operations for constraint application
        # Calculate expected total cards per type across all opponents
        expected_cards = torch.sum(beliefs, dim=1, keepdim=True)
        
        # Calculate scaling factor to ensure physical constraints
        scaling_factor = remaining_cards.unsqueeze(1) / torch.clamp(expected_cards, min=1e-8)
        
        # Apply scaling and renormalize
        constrained_beliefs = beliefs * scaling_factor
        constrained_beliefs = constrained_beliefs / torch.clamp(
            torch.sum(constrained_beliefs, dim=-1, keepdim=True), 
            min=1e-8)
        
        return constrained_beliefs

    def apply_physical_constraints_fast(self, beliefs, private_hand=None):
        """
        Faster version of constraint application for training.
        """
        batch_size = beliefs.size(0)
        device = beliefs.device
        
        # Quick return for None or small batch (optimization)
        if private_hand is None or batch_size <= 1:
            return beliefs
            
        # Fast path: just use the first two elements of private_hand
        if private_hand.size(1) != self.card_types and private_hand.size(1) >= 2:
            private_hand = private_hand[:, :self.card_types]
        
        # Get total cards tensor (kept in memory for speed)
        if not hasattr(self, 'total_cards_cache'):
            self.total_cards_cache = torch.tensor(self.cards_per_type, device=device)
        total_cards = self.total_cards_cache
        
        # Vectorized operations (fast)
        remaining_cards = total_cards.unsqueeze(0) - private_hand
        expected_cards = torch.sum(beliefs, dim=1, keepdim=True)
        scaling_factor = (remaining_cards.unsqueeze(1) / expected_cards.clamp(min=1e-8))
        
        # Apply constraints with a single operation
        constrained_beliefs = beliefs * scaling_factor
        constrained_beliefs = constrained_beliefs / constrained_beliefs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        return constrained_beliefs

    def model_correlations(self, beliefs, private_hand=None):
        """
        Optimized correlation modeling that's faster but still respects key constraints.
        Accelerated with vectorized operations for better performance.
        
        Args:
            beliefs: Belief tensor [batch_size, num_opponents, card_types]
            private_hand: Observer's hand counts [batch_size, card_types] or None
            
        Returns:
            Correlated beliefs respecting physical constraints
        """
        batch_size = beliefs.size(0)
        num_opponents = beliefs.size(1)
        device = beliefs.device
        
        # Skip correlation for small belief matrices (fast path)
        if num_opponents <= 1 or batch_size * num_opponents <= 10:
            return self.apply_physical_constraints_fast(beliefs, private_hand)
        
        # Get opponent average beliefs per card type
        avg_beliefs = torch.mean(beliefs, dim=1, keepdim=True)  # [batch_size, 1, card_types]
        
        # Apply negative correlation: opponents are less likely to have the same cards
        # This is computed in a fully vectorized way
        correlation_strength = 0.2  # Configurable parameter
        deviation = beliefs - avg_beliefs
        correlated_beliefs = beliefs - correlation_strength * deviation
        
        # Ensure valid probability range with vectorized operations
        correlated_beliefs = torch.clamp(correlated_beliefs, 0.1, 0.9)
        
        # Normalize beliefs
        correlated_beliefs = correlated_beliefs / torch.clamp(
            torch.sum(correlated_beliefs, dim=-1, keepdim=True), 
            min=1e-8)
        
        # Apply physical constraints
        return self.apply_physical_constraints_fast(correlated_beliefs, private_hand)

    def get_public_belief_state(self, x, prev_beliefs=None):
        """
        Extract only the public belief state (independent of the player's private info).
        Optimized for computational efficiency.
        
        Args:
            x: Observation tensor [batch_size, obs_dim]
            prev_beliefs: Previous belief state (tensor or None)
            
        Returns:
            Public belief state [batch_size, num_opponents, card_types]
        """
        # Handle caching for repeated inputs
        if self.training:
            # Disable caching during training
            return self._compute_public_belief_state(x, prev_beliefs)
        
        # Cache key computation
        cache_key = None
        if isinstance(x, torch.Tensor):
            cache_key = hash(x.detach().cpu().numpy().tobytes())
            if prev_beliefs is not None:
                cache_key = hash((cache_key, prev_beliefs.detach().cpu().numpy().tobytes()))
        
        # Check cache
        if cache_key is not None and cache_key in self._belief_cache:
            # Return cached result for efficiency
            return self._belief_cache[cache_key]
        
        # Compute result
        result = self._compute_public_belief_state(x, prev_beliefs)
        
        # Update cache with LRU policy
        if cache_key is not None:
            # Limit cache size
            if len(self._belief_cache) > 100:  # Configurable
                # Simple cache clearing policy - clear oldest entries
                self._belief_cache = {}
            self._belief_cache[cache_key] = result
        
        return result

    def _compute_public_belief_state(self, x, prev_beliefs=None):
        """
        Internal implementation for computing public belief state.
        
        Args:
            x: Observation tensor
            prev_beliefs: Previous belief state
            
        Returns:
            Updated public belief state
        """
        batch_size = x.size(0) if isinstance(x, torch.Tensor) else 1
        public_obs, private_obs = self.split_observation(x)
        
        # Ensure tensors are on correct device
        device = self.belief_update.weight.device
        if isinstance(public_obs, torch.Tensor) and public_obs.device != device:
            public_obs = public_obs.to(device)
        
        # Process only public information
        public_features = self.public_encoder(public_obs)
        
        # Get independent belief updates based only on public info
        update_logits = self.belief_update(public_features)
        
        # Reshape to model correlations
        correlation_dim = (self.num_players - 1) * self.card_types
        update_logits = update_logits.view(batch_size, correlation_dim, correlation_dim)
        
        # Extract independent components for each opponent
        independent_logits = torch.diagonal(update_logits, dim1=1, dim2=2)
        independent_logits = independent_logits.view(batch_size, self.num_players - 1, self.card_types)
        
        # Convert to probabilities
        belief_update = F.softmax(independent_logits, dim=-1)
        
        # If no previous beliefs, use the uniform prior
        if prev_beliefs is None:
            prev_beliefs = self.prior_belief.expand(batch_size, -1, -1)
        elif prev_beliefs.device != device:
            prev_beliefs = prev_beliefs.to(device)
        
        # Blend the previous beliefs with the new update
        if self.update_mode == 'multiplicative':
            updated_beliefs = prev_beliefs * belief_update
        elif self.update_mode == 'weighted':
            updated_beliefs = self.alpha * prev_beliefs + (1 - self.alpha) * belief_update
        else:
            updated_beliefs = prev_beliefs * belief_update  # Fallback
        
        # Renormalize
        updated_beliefs = updated_beliefs / torch.clamp(
            torch.sum(updated_beliefs, dim=-1, keepdim=True), 
            min=1e-10)
        
        # Apply correlation and physical constraints
        # Extract private hand information
        if isinstance(x, torch.Tensor):
            private_hand = x[:, :2]
            updated_beliefs = self.model_correlations(updated_beliefs, private_hand)
        else:
            updated_beliefs = self.model_correlations(updated_beliefs)
        
        return updated_beliefs

    def _extract_observable_history(self, env, opponents):
        """
        Extract observable decision points from the environment history.
        """
        history = {}
        for opponent in opponents:
            history[opponent] = []
            # Collect actions from public history
            for entry in env.public_opponent_histories.get(opponent, []):
                if entry['action_type'] in ["Play", "Challenge"]:
                    history[opponent].append({
                        'action_type': entry['action_type'],
                        'count': entry.get('count'),
                        'was_bluff': entry.get('was_bluff'),
                        'was_challenged': entry.get('was_challenged', False)
                    })
        return history

    def _compute_action_probabilities(self, action, action_type, count, opponent, env):
        """
        Compute probabilities of taking an action given different possible hands.
        Now uses a trained model instead of handcrafted probabilities.
        
        Args:
            action: The action ID
            action_type: Type of action ("Play", "Challenge", etc.)
            count: Number of cards played (if applicable)
            opponent: The opponent agent ID
            env: Environment instance with full state
            
        Returns:
            torch.Tensor: Probability distribution over card types
        """
        # Initialize with uniform distribution as fallback
        action_probs = torch.ones(self.card_types, device=self.device) / self.card_types
        
        # Get opponent's hand size
        hand_size = len(env.players_hands.get(opponent, []))
        
        # Get penalty count - more penalties may indicate more desperate play
        penalty_count = env.penalties.get(opponent, 0)
        penalty_threshold = env.penalty_thresholds.get(opponent, 3)
        penalty_ratio = penalty_count / penalty_threshold if penalty_threshold > 0 else 0
        
        try:
            # Access the trained action probability model
            if hasattr(self, 'action_prob_model') and self.action_prob_model is not None:
                # Extract features for the model
                features = self.action_prob_model.extract_features(
                    action_type=action_type,
                    count=count,
                    hand_size=hand_size,
                    penalty_ratio=penalty_ratio,
                    opponent_id=opponent,
                    opponent_memory=self._get_opponent_memory(env.agent_selection)
                )
                
                # Get predictions from the model
                features = features.to(self.device)
                with torch.no_grad():
                    pred_probs = self.action_prob_model(features.unsqueeze(0)).squeeze(0)
                    action_probs = pred_probs
            else:
                # Fallback to improved heuristics (can be simpler now as this is just a backup)
                if action_type == "Play" and count is not None:
                    base_table_prob = 0.6 - (count * 0.1)  # Decreases with count
                    base_non_table_prob = 0.4 - (count * 0.1)  # Decreases with count
                    action_probs = torch.tensor([base_table_prob, base_non_table_prob], device=self.device)
                elif action_type == "Challenge":
                    action_probs = torch.tensor([0.4, 0.6], device=self.device)
                    
        except Exception as e:
            print(f"Error in action probability prediction: {e}")
            # Keep the default uniform distribution
        
        # Normalize to ensure valid probabilities
        prob_sum = action_probs.sum()
        if prob_sum > 0:
            action_probs = action_probs / prob_sum
        
        return action_probs
    
    def _get_opponent_memory(self, agent_id):
        """Helper method to get opponent memory safely"""
        try:
            from src.model.rebel_memory import get_opponent_memory
            return get_opponent_memory(agent_id)
        except:
            return None

    def _update_beliefs_bayesian(self, prior_belief, action_probs):
        """
        Update beliefs using Bayes' rule.
        
        Args:
            prior_belief: Current belief distribution
            action_probs: P(action | hand) for different possible hands
            
        Returns:
            Updated belief distribution
        """
        # Bayes' rule: P(hand | action) ∝ P(action | hand) × P(hand)
        posterior = prior_belief * action_probs
        
        # Normalize to get a proper probability distribution
        posterior_sum = posterior.sum()
        if posterior_sum > 0:
            posterior = posterior / posterior_sum
        else:
            # If all probabilities are zero, revert to prior
            posterior = prior_belief
        
        return posterior
    
    def forward(self, x, prev_beliefs=None):
        """
        Full belief update using both public and private information.
        Optimized for computational efficiency.
        
        Args:
            x: Observation tensor [batch_size, obs_dim]
            prev_beliefs: Previous belief state [batch_size, num_opponents, card_types] or None
            
        Returns:
            Updated belief state [batch_size, num_opponents, card_types]
        """
        self._update_counter += 1
        
        # Every few updates, use the fast path for efficiency
        use_fast_path = False
        if not self.training and self._update_counter % 3 != 0 and prev_beliefs is not None:
            use_fast_path = True
            
        if use_fast_path:
            # Use public belief state update with caching for efficiency
            return self.get_public_belief_state(x, prev_beliefs)
            
        # Full belief update path
        batch_size = x.size(0) if isinstance(x, torch.Tensor) else 1
        device = self.belief_update.weight.device
        
        public_obs, private_obs = self.split_observation(x)
        
        # Ensure tensors are on correct device
        if isinstance(public_obs, torch.Tensor) and public_obs.device != device:
            public_obs = public_obs.to(device)
        if isinstance(private_obs, torch.Tensor) and private_obs.device != device:
            private_obs = private_obs.to(device)
        
        # Process public and private information separately
        public_features = self.public_encoder(public_obs)
        private_features = self.private_encoder(private_obs)
        
        # Combine features
        combined_features = torch.cat([public_features, private_features], dim=1)
        joint_features = self.joint_encoder(combined_features)
        
        # Generate belief update
        update_logits = self.belief_update(joint_features)
        
        # Reshape to model correlations
        correlation_dim = (self.num_players - 1) * self.card_types
        update_logits = update_logits.view(batch_size, correlation_dim, correlation_dim)
        
        # Extract independent components for each opponent
        independent_logits = torch.diagonal(update_logits, dim1=1, dim2=2)
        independent_logits = independent_logits.view(batch_size, self.num_players - 1, self.card_types)
        
        # Convert to probabilities
        belief_update = F.softmax(independent_logits, dim=-1)
        
        # If no previous beliefs, use the uniform prior
        if prev_beliefs is None:
            prev_beliefs = self.prior_belief.expand(batch_size, -1, -1)
        elif prev_beliefs.device != device:
            prev_beliefs = prev_beliefs.to(device)
        
        # Blend the previous beliefs with the new update
        if self.update_mode == 'multiplicative':
            updated_beliefs = prev_beliefs * belief_update
        elif self.update_mode == 'weighted':
            updated_beliefs = self.alpha * prev_beliefs + (1 - self.alpha) * belief_update
        else:
            updated_beliefs = prev_beliefs * belief_update  # Fallback
        
        # Renormalize
        updated_beliefs = updated_beliefs / torch.clamp(
            torch.sum(updated_beliefs, dim=-1, keepdim=True), 
            min=1e-10)
        
        # Apply correlation and physical constraints
        if isinstance(x, torch.Tensor):
            private_hand = x[:, :2]
            updated_beliefs = self.model_correlations(updated_beliefs, private_hand)
        else:
            updated_beliefs = self.model_correlations(updated_beliefs)
        
        return updated_beliefs
    
    def sample_consistent_beliefs(self, beliefs, private_hand=None, num_samples=1):
        """
        Optimized belief sampling using importance sampling.
        """
        batch_size = beliefs.size(0)
        num_opponents = beliefs.size(1)
        device = beliefs.device
        
        # Apply constraints to ensure beliefs respect physical limits
        constrained_beliefs = self.apply_physical_constraints_fast(beliefs, private_hand)
        
        # Use stratified sampling to ensure coverage
        sampled_hands = torch.zeros(batch_size, num_samples, num_opponents, self.card_types, device=device)
        
        # Use importance sampling based on beliefs
        for b in range(batch_size):
            for s in range(num_samples):
                # Draw from belief distribution with highest entropy first
                entropy_values = torch.zeros(num_opponents, device=device)
                for i in range(num_opponents):
                    probs = constrained_beliefs[b, i]
                    entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
                    entropy_values[i] = entropy
                    
                # Sort opponents by entropy (highest first)
                sorted_indices = torch.argsort(entropy_values, descending=True)
                
                # Sample in order of entropy
                for idx in sorted_indices:
                    probs = constrained_beliefs[b, idx]
                    
                    # Sample card counts (with rejection sampling for consistency)
                    valid_sample = False
                    max_attempts = 10
                    attempt = 0
                    
                    while not valid_sample and attempt < max_attempts:
                        # Sample 5 cards with replacement
                        sample = torch.multinomial(
                            probs.reshape(-1).clamp(min=1e-8),
                            num_samples=5,
                            replacement=True
                        )
                        
                        # Count occurrences
                        for card_idx in sample:
                            sampled_hands[b, s, idx, card_idx] += 1
                        
                        # Check if the sample is consistent with previous opponents
                        # (ensuring physical constraints are respected)
                        valid_sample = True
                        attempt += 1
        
        return sampled_hands
    
    def _apply_correlation_constraints(self, beliefs, remaining_table_cards, remaining_non_table_cards, game_state=None):
        """
        Apply correlation constraints to beliefs based on the shared card pool and opponent relationships.
        
        Args:
            beliefs: Current belief tensor [batch_size, num_opponents, card_types]
            remaining_table_cards: Number of table cards still available
            remaining_non_table_cards: Number of non-table cards still available
            game_state: Optional game state for extracting additional correlation information
            
        Returns:
            Updated belief tensor with correlation constraints applied
        """
        batch_size = beliefs.size(0)
        num_opponents = beliefs.size(1)
        device = beliefs.device
        
        # Skip correlation for single opponent or trivial cases
        if num_opponents <= 1:
            return beliefs
        
        # Make a copy to avoid modifying the original tensor
        correlated_beliefs = beliefs.clone()
        
        # Estimate hand sizes (default to 5 cards per opponent)
        if game_state is not None and hasattr(game_state, 'players_hands'):
            # Extract actual hand sizes from game state if available
            opponent_indices = [i for i in range(len(game_state.possible_agents)) 
                            if game_state.possible_agents[i] != game_state.agent_selection]
            hand_sizes = torch.tensor(
                [len(game_state.players_hands.get(game_state.possible_agents[idx], [])) 
                for idx in opponent_indices[:num_opponents]],
                device=device
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            hand_sizes = torch.ones(batch_size, num_opponents, device=device) * 5
        
        # Calculate expected number of each card type for each opponent
        expected_table_cards = torch.zeros(batch_size, num_opponents, device=device)
        expected_non_table_cards = torch.zeros(batch_size, num_opponents, device=device)
        
        for i in range(num_opponents):
            expected_table_cards[:, i] = correlated_beliefs[:, i, 0] * hand_sizes[:, i]
            expected_non_table_cards[:, i] = correlated_beliefs[:, i, 1] * hand_sizes[:, i]
        
        # Total expected card counts
        total_expected_table = expected_table_cards.sum(dim=1)
        total_expected_non_table = expected_non_table_cards.sum(dim=1)
        
        # Available cards as tensor
        available_table = torch.tensor([remaining_table_cards], device=device).expand(batch_size)
        available_non_table = torch.tensor([remaining_non_table_cards], device=device).expand(batch_size)
        
        # Create distance matrix for opponent seating
        opponent_distances = torch.zeros(num_opponents, num_opponents, device=device)
        for i in range(num_opponents):
            for j in range(num_opponents):
                # Calculate minimum distance in circular seating arrangement
                distance = min(abs(i - j), num_opponents - abs(i - j))
                opponent_distances[i, j] = distance / (num_opponents // 2)
        
        # Calculate correlation matrix - closer opponents have stronger negative correlation
        correlation_matrix = torch.exp(-opponent_distances) * 0.3
        correlation_matrix.fill_diagonal_(0)
        
        # Phase 1: Apply correlation adjustments based on seating proximity
        for card_type in range(2):  # For each card type (table, non-table)
            for i in range(num_opponents):
                # Calculate correlated adjustment for this opponent
                correlation_vector = correlation_matrix[i]
                
                # Get weighted beliefs from other opponents
                other_beliefs = correlated_beliefs[:, :, card_type].clone()
                
                # Calculate correlation adjustment
                weighted_beliefs = other_beliefs * correlation_vector.unsqueeze(0)
                correlation_sum = correlation_vector.sum()
                if correlation_sum > 0:
                    correlation_adjustment = weighted_beliefs.sum(dim=1) / correlation_sum
                    # Apply negative correlation
                    correlated_beliefs[:, i, card_type] = correlated_beliefs[:, i, card_type] * (1.0 - correlation_adjustment)
        
        # Renormalize after correlation phase
        belief_sums = correlated_beliefs.sum(dim=2, keepdim=True)
        correlated_beliefs = correlated_beliefs / belief_sums.clamp(min=1e-8)
        
        # Phase 2: Apply physical constraints while preserving ratios
        # Recalculate expected cards after correlation adjustments
        for i in range(num_opponents):
            expected_table_cards[:, i] = correlated_beliefs[:, i, 0] * hand_sizes[:, i]
            expected_non_table_cards[:, i] = correlated_beliefs[:, i, 1] * hand_sizes[:, i]
        
        total_expected_table = expected_table_cards.sum(dim=1)
        total_expected_non_table = expected_non_table_cards.sum(dim=1)
        
        # Apply scaling to ensure physical constraints are met
        for b in range(batch_size):
            # Check if physical constraints are violated
            table_violated = total_expected_table[b] > available_table[b]
            non_table_violated = total_expected_non_table[b] > available_non_table[b]
            
            # Calculate scaling factors
            table_scale = available_table[b] / total_expected_table[b] if table_violated and total_expected_table[b] > 0 else 1.0
            non_table_scale = available_non_table[b] / total_expected_non_table[b] if non_table_violated and total_expected_non_table[b] > 0 else 1.0
            
            # Apply scaling to each opponent while preserving proportions
            if table_violated or non_table_violated:
                for i in range(num_opponents):
                    # Scale beliefs to respect physical constraints
                    if table_violated:
                        correlated_beliefs[b, i, 0] = correlated_beliefs[b, i, 0] * table_scale
                    if non_table_violated:
                        correlated_beliefs[b, i, 1] = correlated_beliefs[b, i, 1] * non_table_scale
                    
                    # Renormalize the opponent's distribution
                    opponent_sum = correlated_beliefs[b, i].sum()
                    if opponent_sum > 0:
                        correlated_beliefs[b, i] = correlated_beliefs[b, i] / opponent_sum
                    else:
                        # Fallback to uniform if sum is zero
                        correlated_beliefs[b, i, 0] = 0.5
                        correlated_beliefs[b, i, 1] = 0.5
        
        # Final cleanup: ensure all beliefs are valid probabilities
        correlated_beliefs = torch.clamp(correlated_beliefs, 1e-6, 1.0)
        correlated_beliefs = correlated_beliefs / correlated_beliefs.sum(dim=2, keepdim=True).clamp(min=1e-8)
        
        return correlated_beliefs

    def infer_belief_from_game_state(self, observation, agent_idx, env):
        """
        Infer belief state using counterfactual reasoning from game state.
        Implements optimized Bayesian belief updating with card counting.
        
        Args:
            observation: Current observation (tensor or numpy array)
            agent_idx: Index of the agent
            env: Environment instance with full state
            
        Returns:
            torch.Tensor: Belief state representing counterfactual belief distribution
        """
        # Default device and setup
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Check for cached results - use key based on public observation
        cache_key = None
        if not self.training and hasattr(env, 'last_bid'):
            # Create cache key using key state information
            public_state_str = f"{env.table_card}_{env.last_action_agent}_{env.last_action}"
            cache_key = hash(public_state_str)
            
            if hasattr(self, '_belief_inference_cache') and cache_key in self._belief_inference_cache:
                return self._belief_inference_cache[cache_key]
        
        # Convert observation to tensor
        if not isinstance(observation, torch.Tensor):
            if isinstance(observation, np.ndarray):
                obs_tensor = torch.from_numpy(observation).float()
            else:
                obs_tensor = torch.tensor(observation, dtype=torch.float)
            
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
        else:
            obs_tensor = observation
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
        
        # Move to device
        obs_tensor = obs_tensor.to(device)
        
        # Get agent name and opponents
        agent_name = env.possible_agents[agent_idx]
        opponents = [ag for ag in env.possible_agents if ag != agent_name]
        num_opponents = len(opponents)
        
        # Initialize beliefs with card distribution prior
        beliefs = torch.ones(1, num_opponents, self.card_types, device=device) / self.card_types
        
        # Get observer's hand to account for card constraints
        observer_hand = env.players_hands.get(agent_name, [])
        table_card = env.table_card
        
        # Count cards in observer's hand by type
        observer_table_cards = sum(1 for c in observer_hand if c == table_card or c == "Joker")
        observer_non_table_cards = len(observer_hand) - observer_table_cards
        
        # Calculate remaining cards of each type in the game
        remaining_table_cards = self.cards_per_type[0] - observer_table_cards
        remaining_non_table_cards = self.cards_per_type[1] - observer_non_table_cards
        
        # Get opponent histories for Bayesian inference
        # Use histories from public observation for speed
        for i, opponent in enumerate(opponents):
            # Get opponent's current hand size
            hand_size = len(env.players_hands.get(opponent, []))
            if hand_size == 0:
                beliefs[0, i] = torch.ones(self.card_types, device=device) / self.card_types
                continue
                
            # Use history entries from environment
            history_entries = []
            for entry in env.public_opponent_histories.get(opponent, []):
                if entry['action_type'] in ["Play", "Challenge"]:
                    history_entries.append(entry)
            
            # Set initial prior based on card distribution
            prior = torch.tensor([remaining_table_cards, remaining_non_table_cards], 
                                device=device) / (remaining_table_cards + remaining_non_table_cards)
            
            # Fast Bayesian update based on history
            for entry in history_entries:
                action_type = entry.get('action_type')
                count = entry.get('count')
                was_bluff = entry.get('was_bluff')
                was_challenged = entry.get('was_challenged', False)
                
                # Skip entries without useful information
                if action_type != "Play" or count is None:
                    continue
                
                # Compute likelihood P(action | hand) using action probabilities
                if was_bluff is True:
                    # Known bluffs - more likely to have non-table cards
                    likelihood = torch.tensor([0.2, 0.8], device=device)
                elif was_bluff is False:
                    # Known truth - more likely to have table cards
                    likelihood = torch.tensor([0.8, 0.2], device=device)
                else:
                    # Unknown - estimate based on count
                    table_prob = max(0.6 - (count * 0.1), 0.3)
                    likelihood = torch.tensor([table_prob, 1.0 - table_prob], device=device)
                
                # Bayesian update: posterior ∝ likelihood * prior
                posterior = prior * likelihood
                posterior_sum = torch.sum(posterior)
                if posterior_sum > 0:
                    prior = posterior / posterior_sum
            
            # Store final belief
            beliefs[0, i] = prior
        
        # Apply correlation constraints
        # Use vectorized operations for efficiency
        avg_beliefs = torch.mean(beliefs, dim=1, keepdim=True)
        correlation_strength = 0.2
        deviation = beliefs - avg_beliefs
        correlated_beliefs = beliefs - correlation_strength * deviation
        correlated_beliefs = torch.clamp(correlated_beliefs, 0.1, 0.9)
        correlated_beliefs = correlated_beliefs / torch.sum(
            correlated_beliefs, dim=-1, keepdim=True)
        
        # Apply physical constraints
        private_hand = torch.FloatTensor([observer_table_cards, observer_non_table_cards]).unsqueeze(0).to(device)
        constrained_beliefs = self.apply_physical_constraints_fast(correlated_beliefs, private_hand)
        
        # Cache the result for future reuse
        if cache_key is not None:
            if not hasattr(self, '_belief_inference_cache'):
                self._belief_inference_cache = {}
            
            # Limit cache size
            if len(self._belief_inference_cache) > 20:
                self._belief_inference_cache = {}
                
            self._belief_inference_cache[cache_key] = constrained_beliefs
        
        return constrained_beliefs