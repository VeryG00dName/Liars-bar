# src/model/rebel_models.py
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
        
        # Belief update network – extended to model correlations
        # Output shape: [batch, num_players * card_types * num_players * card_types]
        # This allows modeling joint distributions between all players
        correlation_dim = (num_players - 1) * self.card_types
        self.belief_update = nn.Linear(hidden_dim, correlation_dim * correlation_dim)
        
        # Initialize with a uniform prior
        self.register_buffer('prior_belief', torch.ones(1, num_players - 1, self.card_types) / self.card_types)
        
        # If using weighted blending, add a learnable parameter alpha
        if self.update_mode == 'weighted':
            self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def split_observation(self, x):
        """
        Split observation into public and private components.
        
        Args:
            x: Full observation tensor [batch_size, obs_dim] or numpy array or list
                
        Returns:
            (public_obs, private_obs): Tuple of public and private observation tensors
        """
        # Convert to tensor if not already
        if not isinstance(x, torch.Tensor):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            else:
                x = torch.tensor(x, dtype=torch.float)
            
            # Add batch dimension if needed
            if x.dim() == 1:
                x = x.unsqueeze(0)
        
        # First two elements are the player's hand information (table cards, non-table cards)
        private_obs = x[:, :self.private_dim]
        
        # Remaining elements are public information
        public_obs = x[:, self.private_dim:]
        
        return public_obs, private_obs
    
    def apply_physical_constraints(self, beliefs, private_hand=None):
        """
        Apply physical constraints to ensure the beliefs respect card limits.
        
        Args:
            beliefs: Belief tensor [batch_size, num_opponents, card_types]
            private_hand: Observer's hand counts [batch_size, card_types] or None
            
        Returns:
            Constrained beliefs respecting physical card limits
        """
        batch_size = beliefs.size(0)
        device = beliefs.device
        
        # Clone the beliefs to avoid modifying the original
        constrained_beliefs = beliefs.clone()
        
        # Get the total card counts from our configuration
        total_cards = torch.tensor(self.cards_per_type, device=device)
        
        # Account for the cards in the observer's hand
        if private_hand is not None:
            # Convert private_hand to tensor if needed
            if not isinstance(private_hand, torch.Tensor):
                if isinstance(private_hand, np.ndarray):
                    private_hand = torch.from_numpy(private_hand).float().to(device)
                else:
                    private_hand = torch.tensor(private_hand, dtype=torch.float).to(device)
                
                # Add batch dimension if needed
                if private_hand.dim() == 1:
                    private_hand = private_hand.unsqueeze(0)
                
            # Ensure private_hand has the right shape
            if private_hand.size(1) != self.card_types:
                # If it's the observation, use only first two elements
                if private_hand.size(1) >= 2:
                    private_hand = private_hand[:, :self.card_types]
                else:
                    private_hand = torch.zeros(batch_size, self.card_types, device=device)
            
            remaining_cards = total_cards.unsqueeze(0) - private_hand
        else:
            remaining_cards = total_cards.unsqueeze(0).expand(batch_size, -1)
        
        # Calculate the expected number of cards per type for each opponent
        # This treats the beliefs as probabilities of having cards
        expected_cards = torch.sum(constrained_beliefs, dim=1)  # [batch_size, card_types]
        
        # Scale beliefs to match the physical constraints
        # If expected cards > remaining cards, scale down; if < remaining cards, scale up
        scaling_factor = (remaining_cards / expected_cards.clamp(min=1e-8)).unsqueeze(1)
        constrained_beliefs = constrained_beliefs * scaling_factor
        
        # Normalize each opponent's distribution
        constrained_beliefs = constrained_beliefs / constrained_beliefs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
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
        """
        batch_size = beliefs.size(0)
        num_opponents = beliefs.size(1)
        device = beliefs.device
        
        # Skip correlation for small belief matrices (fast path)
        if num_opponents <= 1 or batch_size * num_opponents <= 10:
            return self.apply_physical_constraints_fast(beliefs, private_hand)
        
        # Apply a simpler, vectorized correlation model
        # Instead of pairwise adjustments, use a single matrix operation
        correlated_beliefs = beliefs.clone()
        
        # Get opponent average beliefs per card type
        avg_beliefs = beliefs.mean(dim=1, keepdim=True)  # [batch_size, 1, card_types]
        
        # Apply negative correlation: push away from the average (vectorized)
        correlation_strength = 0.2  # Slightly reduced for faster convergence
        deviation = beliefs - avg_beliefs
        correlated_beliefs = beliefs - correlation_strength * deviation
        
        # Clamp values to valid probability range (vectorized)
        correlated_beliefs = torch.clamp(correlated_beliefs, 0.1, 0.9)
        
        # Normalize (vectorized)
        correlated_beliefs = correlated_beliefs / correlated_beliefs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # Fast physical constraints
        return self.apply_physical_constraints_fast(correlated_beliefs, private_hand)

    def get_public_belief_state(self, x, prev_beliefs=None):
        """
        Extract only the public belief state (independent of the player's private info).
        Now with correlation constraints.
        
        Args:
            x: Observation tensor [batch_size, obs_dim]
            prev_beliefs: Previous belief state or None
            
        Returns:
            Public belief state [batch_size, num_opponents, card_types]
        """
        batch_size = x.size(0)
        public_obs, private_obs = self.split_observation(x)
        
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
        
        # Blend the previous beliefs with the new update
        if self.update_mode == 'multiplicative':
            updated_beliefs = prev_beliefs * belief_update
        elif self.update_mode == 'weighted':
            updated_beliefs = self.alpha * prev_beliefs + (1 - self.alpha) * belief_update
        else:
            updated_beliefs = prev_beliefs * belief_update  # Fallback
        
        # Renormalize
        updated_beliefs = updated_beliefs / (updated_beliefs.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Apply correlation and physical constraints
        # Extract private hand information - first two elements are the table cards and non-table cards
        private_hand = x[:, :2]
        updated_beliefs = self.model_correlations(updated_beliefs, private_hand)
        
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
        Now with correlation constraints and device consistency checks.
        
        Args:
            x: Observation tensor [batch_size, obs_dim]
            prev_beliefs: Previous belief state [batch_size, num_opponents, card_types] or None
            
        Returns:
            Updated belief state [batch_size, num_opponents, card_types]
        """
        batch_size = x.size(0)
        device = x.device  # Get the device of input tensor
        public_obs, private_obs = self.split_observation(x)
        
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
        
        # Ensure prev_beliefs is on the same device as belief_update
        prev_beliefs = prev_beliefs.to(device)
        
        # Blend the previous beliefs with the new update
        if self.update_mode == 'multiplicative':
            updated_beliefs = prev_beliefs * belief_update
        elif self.update_mode == 'weighted':
            updated_beliefs = self.alpha * prev_beliefs + (1 - self.alpha) * belief_update
        else:
            updated_beliefs = prev_beliefs * belief_update  # Fallback
        
        # Renormalize
        updated_beliefs = updated_beliefs / (updated_beliefs.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Apply correlation and physical constraints
        updated_beliefs = self.model_correlations(updated_beliefs, private_obs)
        
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
        Implements proper Bayesian belief updating with card counting
        and physical constraints.
        
        Args:
            observation: Current observation (tensor or numpy array)
            agent_idx: Index of the agent
            env: Environment instance with full state
            
        Returns:
            torch.Tensor: Belief state representing counterfactual belief distribution
        """
        self.device = 'cuda'
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
        obs_tensor = obs_tensor.to(self.device)
        
        # Get agent name and opponents
        agent_name = env.possible_agents[agent_idx]
        opponents = [ag for ag in env.possible_agents if ag != agent_name]
        num_opponents = len(opponents)
        
        # Initialize beliefs with card distribution prior
        # This represents P(hand | information)
        beliefs = torch.ones(1, num_opponents, self.card_types, device=self.device) / self.card_types
        
        # Get observer's hand to account for card constraints
        observer_hand = env.players_hands.get(agent_name, [])
        table_card = env.table_card
        
        # Count cards in observer's hand by type
        observer_table_cards = sum(1 for c in observer_hand if c == table_card or c == "Joker")
        observer_non_table_cards = len(observer_hand) - observer_table_cards
        
        # Calculate remaining cards of each type in the game
        remaining_table_cards = self.cards_per_type[0] - observer_table_cards
        remaining_non_table_cards = self.cards_per_type[1] - observer_non_table_cards
        
        # Extract full game history for all opponents
        game_history = self._extract_observable_history(env, opponents)
        
        # Process history to derive initial beliefs
        for i, opponent in enumerate(opponents):
            # Get opponent's current hand size
            hand_size = len(env.players_hands.get(opponent, []))
            if hand_size == 0:
                # If opponent has no cards, beliefs are meaningless
                # Set uniform distribution
                beliefs[0, i] = torch.ones(self.card_types, device=self.device) / self.card_types
                continue
                
            # Count total cards seen from this opponent (played or challenged)
            opponent_seen_cards = {0: 0, 1: 0}  # {table_cards: count, non_table_cards: count}
            
            # For each action in history, update beliefs using Bayes' rule
            history_entries = game_history.get(opponent, [])
            
            # Set initial prior based on card distribution
            prior = torch.tensor([remaining_table_cards, remaining_non_table_cards], 
                                device=self.device) / (remaining_table_cards + remaining_non_table_cards)
            
            # Keep track of total cards revealed by this opponent
            revealed_table_cards = 0
            revealed_non_table_cards = 0
            
            for entry in history_entries:
                action_type = entry.get('action_type')
                count = entry.get('count')
                was_bluff = entry.get('was_bluff')
                was_challenged = entry.get('was_challenged', False)
                
                if action_type == "Play" and count is not None:
                    if was_bluff is not None:  # Only if we know the truth value
                        # This is a play where the bluff status is known (was challenged)
                        if was_bluff:
                            # Cards played were not table cards
                            revealed_non_table_cards += count
                        else:
                            # Cards played were table cards
                            revealed_table_cards += count
                        
                    # Derive likelihood P(action | hand) using our probability model
                    likelihood = self._compute_action_probabilities(None, action_type, count, opponent, env)
                    
                    # Apply Bayes' rule: P(hand | action) ∝ P(action | hand) × P(hand)
                    posterior = prior * likelihood
                    posterior_sum = posterior.sum()
                    if posterior_sum > 0:
                        prior = posterior / posterior_sum
                    
                elif action_type == "Challenge":
                    # Challenges reveal information about the challenger's beliefs
                    likelihood = self._compute_action_probabilities(None, action_type, None, opponent, env)
                    
                    # Update beliefs
                    posterior = prior * likelihood
                    posterior_sum = posterior.sum()
                    if posterior_sum > 0:
                        prior = posterior / posterior_sum
            
            # Apply final physical constraints based on revealed cards
            # Calculate remaining cards after accounting for revealed cards
            adj_remaining_table = max(0, remaining_table_cards - revealed_table_cards)
            adj_remaining_non_table = max(0, remaining_non_table_cards - revealed_non_table_cards)
            
            # Calculate max possible cards in opponent's hand of each type
            max_table_in_hand = min(hand_size, adj_remaining_table)
            max_non_table_in_hand = min(hand_size, adj_remaining_non_table)
            
            # Normalize considering physically possible hands
            if max_table_in_hand + max_non_table_in_hand > 0:
                physical_constraint = torch.tensor([max_table_in_hand, max_non_table_in_hand], 
                                                device=self.device)
                physical_constraint = physical_constraint / (max_table_in_hand + max_non_table_in_hand)
                
                # Blend Bayesian posterior with physical constraints
                # Weight depends on how much information we have
                info_weight = min(len(history_entries) / 5.0, 0.8)
                final_belief = info_weight * prior + (1 - info_weight) * physical_constraint
            else:
                final_belief = prior
                
            # Normalize final belief
            belief_sum = final_belief.sum()
            if belief_sum > 0:
                beliefs[0, i] = final_belief / belief_sum
            else:
                # Fallback to uniform if we have a zero distribution
                beliefs[0, i] = torch.ones(self.card_types, device=self.device) / self.card_types
        
        # Apply correlation constraints - opponents' hands are not independent
        # because they draw from a shared pool of cards
        beliefs = self._apply_correlation_constraints(beliefs, remaining_table_cards, remaining_non_table_cards)
        
        # Apply physical constraints using observer's hand
        private_hand = torch.FloatTensor([observer_table_cards, observer_non_table_cards]).unsqueeze(0).to(self.device)
        beliefs = self.apply_physical_constraints(beliefs, private_hand)
        
        return beliefs


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
    
    def forward(self, obs, beliefs):
        """
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            beliefs: Belief state tensor [batch_size, belief_dim]
            
        Returns:
            value: Estimated counterfactual value [batch_size, 1]
            regrets: Counterfactual regret estimates [batch_size, action_dim]
        """
        beliefs_flat = beliefs.reshape(beliefs.size(0), -1)
        combined = torch.cat([obs, beliefs_flat], dim=-1)
        features = self.shared_network(combined)
        value = self.value_head(features)
        regrets = self.regret_head(features)
        return value, regrets
    
    def evaluate_public_state(self, public_obs, beliefs):
        """
        Evaluate the value of a public state without private information.
        Used for subgame solving at public state nodes.
        
        Args:
            public_obs: Public observation tensor [batch_size, public_dim]
            beliefs: Belief state tensor [batch_size, belief_dim]
            
        Returns:
            value: Estimated value [batch_size, 1]
            regrets: Estimated regrets [batch_size, action_dim]
        """
        # Create a dummy private observation (zeros)
        batch_size = public_obs.size(0)
        device = public_obs.device
        private_obs = torch.zeros(batch_size, self.private_dim, device=device)
        
        # Combine with public observation
        combined_obs = torch.cat([private_obs, public_obs], dim=1)
        
        # Use regular forward pass
        return self.forward(combined_obs, beliefs)


class RebelPolicyNetwork(nn.Module):
    """
    Policy network specialized for belief-based decision making.
    Updated to handle public/private belief separation.
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
        
        Args:
            x: Full observation tensor [batch_size, obs_dim]
            
        Returns:
            (public_obs, private_obs): Tuple of public and private observation tensors
        """
        # First two elements are the player's hand information (table cards, non-table cards)
        private_obs = x[:, :self.private_dim]
        
        # Remaining elements are public information
        public_obs = x[:, self.private_dim:]
        
        return public_obs, private_obs
    
    def forward(self, obs, beliefs=None, hidden_state=None):
        """
        Forward pass through the policy network.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            beliefs: Belief state tensor [batch_size, belief_dim] or None
            hidden_state: Not used, included for API compatibility
            
        Returns:
            action_probs: Action probabilities [batch_size, action_dim]
            value: State value estimate [batch_size, 1]
            search_policy_probs: Auxiliary search-derived action probabilities [batch_size, action_dim]
        """
        batch_size = obs.size(0)
        
        # Split observation into public and private parts
        public_obs, private_obs = self.split_observation(obs)
        
        # Process each part separately
        public_features = self.public_encoder(public_obs)
        private_features = self.private_encoder(private_obs)
        
        # Combine observation features
        obs_features = torch.cat([public_features, private_features], dim=1)
        
        if beliefs is not None:
            if beliefs.dim() > 2:
                beliefs_flat = beliefs.reshape(batch_size, -1)
            else:
                beliefs_flat = beliefs
            belief_features = self.belief_encoder(beliefs_flat)
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
        action_probs = F.softmax(action_logits, dim=1)
        
        # Value prediction
        value = self.value_head(joint_features)
        
        # Auxiliary search policy head
        search_policy_logits = self.search_policy_head(joint_features)
        search_policy_probs = F.softmax(search_policy_logits, dim=1)
        
        return action_probs, value, search_policy_probs
    
    def public_policy(self, public_obs, beliefs=None):
        """
        Compute policy based only on public information.
        Used for generating a blueprint strategy.
        
        Args:
            public_obs: Public observation tensor
            beliefs: Belief state tensor or None
            
        Returns:
            action_probs: Action probabilities
            value: State value estimate
            search_policy_probs: Auxiliary search policy
        """
        batch_size = public_obs.size(0)
        device = public_obs.device
        
        # Create dummy private observations (zeros)
        private_obs = torch.zeros(batch_size, self.private_dim, device=device)
        
        # Process public features
        public_features = self.public_encoder(public_obs)
        private_features = self.private_encoder(private_obs)
        
        # Combine observation features
        obs_features = torch.cat([public_features, private_features], dim=1)
        
        if beliefs is not None:
            if beliefs.dim() > 2:
                beliefs_flat = beliefs.reshape(batch_size, -1)
            else:
                beliefs_flat = beliefs
            belief_features = self.belief_encoder(beliefs_flat)
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
        action_probs = F.softmax(action_logits, dim=1)
        
        # Value prediction
        value = self.value_head(joint_features)
        
        # Auxiliary search policy head
        search_policy_logits = self.search_policy_head(joint_features)
        search_policy_probs = F.softmax(search_policy_logits, dim=1)
        
        return action_probs, value, search_policy_probs
    
    def act(self, observation, beliefs=None):
        """
        Choose an action based on observation and beliefs.
        
        Args:
            observation: Current observation tensor.
            beliefs: Current belief state.
            
        Returns:
            action: Selected action.
            action_prob: Probability of the selected action.
            state_value: Estimated state value.
        """
        with torch.no_grad():
            if not isinstance(observation, torch.Tensor):
                observation = torch.FloatTensor(observation).unsqueeze(0)
            action_probs, state_value, _ = self.forward(observation, beliefs)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
            action_prob = action_probs[0, action].item()
        return action, action_prob, state_value.item()


class ActionProbabilityModel(nn.Module):
    """
    Neural model to predict action probabilities based on game state features.
    Replaces hardcoded probability values with learned probabilities.
    """
    def __init__(self, input_dim=10, hidden_dim=64):
        super(ActionProbabilityModel, self).__init__()
        self.input_dim = input_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)  # Outputs [table_prob, non_table_prob]
        )
        
    def forward(self, features):
        """
        Predict action probabilities from state features.
        
        Args:
            features: Tensor of state features including:
                - action_type (play/challenge)
                - play_count (1/2/3)
                - hand_size
                - penalty_ratio
                - is_desperate (near elimination)
                - opponent_features (e.g., historical behavior)
                
        Returns:
            Tensor of [table_prob, non_table_prob]
        """
        logits = self.network(features)
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def extract_features(self, action_type, count, hand_size, penalty_ratio, 
                         opponent_id=None, opponent_memory=None):
        """
        Extract relevant features for probability prediction.
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
        
        # Opponent features
        if opponent_memory and opponent_id:
            opponent_summary = opponent_memory.get_summary(opponent_id)
            features.extend(opponent_summary)
        else:
            # Default opponent features if not available
            features.extend([0.5, 0.5, 0.5, 0.5, 0.0, 0.0])
            
        return torch.tensor(features, dtype=torch.float)
    
class ActionProbabilityDataCollector:
    """
    Collects data for training the action probability model.
    """
    def __init__(self):
        self.data = []
        
    def record_action(self, action_type, count, hand, table_card, was_bluff=None, 
                      hand_size=None, penalty_ratio=None, opponent_id=None, 
                      opponent_memory=None):
        """
        Record an action taken by a player.
        
        Args:
            action_type: Type of action ("Play" or "Challenge")
            count: Number of cards played (if applicable)
            hand: Player's hand
            table_card: Current table card
            was_bluff: Whether the play was a bluff (if known)
            hand_size: Size of player's hand
            penalty_ratio: Current penalty ratio
            opponent_id: Player ID
            opponent_memory: OpponentMemory instance
        """
        # Skip if we don't know if it was a bluff or not
        if action_type == "Play" and was_bluff is None:
            return
            
        # Count cards in hand
        if hand_size is None:
            hand_size = len(hand)
            
        table_card_count = sum(1 for c in hand if c == table_card or c == "Joker")
        non_table_card_count = hand_size - table_card_count
        
        # Extract features
        features = []
        is_play = 1.0 if action_type == "Play" else 0.0
        is_challenge = 1.0 if action_type == "Challenge" else 0.0
        features.extend([is_play, is_challenge])
        
        play_count = count if count is not None else 0
        features.append(play_count / 3.0)
        
        features.append(hand_size / 5.0)
        features.append(penalty_ratio if penalty_ratio is not None else 0.0)
        
        # Opponent features
        if opponent_memory and opponent_id:
            opponent_summary = opponent_memory.get_summary(opponent_id)
            features.extend(opponent_summary)
        else:
            features.extend([0.5, 0.5, 0.5, 0.5, 0.0, 0.0])
        
        # Create target based on actual card distribution in hand
        has_table_cards = table_card_count > 0
        has_non_table_cards = non_table_card_count > 0
        
        if action_type == "Play":
            if was_bluff:
                # A bluff means they played non-table cards as table cards
                target = [0.0, 1.0]  # [table_prob, non_table_prob]
            else:
                # Not a bluff means they played table cards truthfully
                target = [1.0, 0.0]  # [table_prob, non_table_prob]
        elif action_type == "Challenge":
            # For challenges, consider the player's own hand
            target = [1.0, 0.0] if has_table_cards else [0.0, 1.0]
        
        self.data.append({
            'features': features,
            'target': target,
            'meta': {
                'action_type': action_type,
                'count': count,
                'hand_size': hand_size,
                'table_card_count': table_card_count,
                'non_table_card_count': non_table_card_count,
                'was_bluff': was_bluff,
            }
        })
        
    def get_training_data(self):
        """Get collected data for training."""
        if not self.data:
            return None, None
            
        features = torch.tensor([d['features'] for d in self.data], dtype=torch.float)
        targets = torch.tensor([d['target'] for d in self.data], dtype=torch.float)
        return features, targets