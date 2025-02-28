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
    
    def infer_belief_from_game_state(self, observation, agent_idx, env):
        """
        Infer belief state using counterfactual reasoning from game state.
        
        Args:
            observation: Current observation
            agent_idx: Index of the agent
            env: Environment instance with full state
            
        Returns:
            Belief state representing counterfactual belief distribution
        """
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
        
        agent_name = env.possible_agents[agent_idx]
        opponents = [ag for ag in env.possible_agents if ag != agent_name]
        num_opponents = len(opponents)
        
        # Initialize uniform beliefs
        beliefs = torch.ones(1, num_opponents, self.card_types) / self.card_types
        
        # Extract observable history
        action_history = self._extract_observable_history(env, opponents)
        
        # For each opponent, compute counterfactual beliefs
        for i, opponent in enumerate(opponents):
            # Get prior belief (uniform or from memory if available)
            prior_belief = beliefs[0, i]
            
            # Traverse decision points and update beliefs using Bayes' rule
            for decision_point in action_history.get(opponent, []):
                action = decision_point['action']
                action_type = decision_point.get('action_type')
                count = decision_point.get('count')
                
                # Compute action probabilities for different possible hands
                action_probs = self._compute_action_probabilities(action, action_type, count, opponent, env)
                
                # Update beliefs using Bayes' rule
                prior_belief = self._update_beliefs_bayesian(prior_belief, action_probs)
            
            # Store updated beliefs
            beliefs[0, i] = prior_belief
        
        # Extract observer's hand information for physical constraints
        _, private_obs = self.split_observation(obs_tensor)
        
        # Apply correlation and physical constraints
        constrained_beliefs = self.model_correlations(beliefs, private_obs)
        
        return constrained_beliefs

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
        """
        table_card = env.table_card
        action_probs = torch.zeros(self.card_types)
        
        # Simplified version for illustration
        if action_type == "Play" and count is not None:
            # Probability of playing count cards if they have table_card/non-table cards
            if count == 1:
                # Probability of playing 1 card if they have table cards
                action_probs[0] = 0.7  # Higher if they have table cards
                # Probability of playing 1 card if they have non-table cards
                action_probs[1] = 0.3  # Lower if they only have non-table cards
            elif count == 2:
                # Probability of playing 2 cards if they have table cards
                action_probs[0] = 0.6  # Slightly lower for playing 2 table cards
                # Probability of playing 2 cards if they have non-table cards
                action_probs[1] = 0.2  # Even lower for playing 2 non-table cards
            else:  # count == 3
                # Probability of playing 3 cards if they have table cards
                action_probs[0] = 0.5  # Even lower for playing 3 table cards
                # Probability of playing 3 cards if they have non-table cards
                action_probs[1] = 0.1  # Much lower for playing 3 non-table cards
        elif action_type == "Challenge":
            # Probability of challenging if they have table_card/non-table cards
            action_probs[0] = 0.3  # Lower if they have table cards
            action_probs[1] = 0.7  # Higher if they have non-table cards
        
        # In a real implementation, these probabilities would come from:
        # - Policy network predictions
        # - Blueprint strategy if available
        # - Opponent modeling statistics
        
        return action_probs

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
        Optimized version of belief sampling that's much faster.
        """
        # For training, we can use a simpler approach that approximates full sampling
        batch_size = beliefs.size(0)
        num_opponents = beliefs.size(1)
        device = beliefs.device
        
        # Apply constraints to ensure beliefs respect physical limits
        constrained_beliefs = self.apply_physical_constraints_fast(beliefs, private_hand)
        
        # Use a faster sampling approach
        sampled_hands = torch.zeros(batch_size, num_samples, num_opponents, self.card_types, device=device)
        
        # Skip expensive sampling during training unless explicitly needed
        if hasattr(self, 'training') and self.training and torch.rand(1).item() > 0.1:
            # During training, 90% of the time just use expectations instead of full sampling
            # This gives a significant speed boost with minimal accuracy loss
            for s in range(num_samples):
                # Instead of sampling cards one by one, just use the expected values
                # Scale to match typical hand size (5 cards)
                expected_hand = constrained_beliefs * 5
                sampled_hands[:, s] = expected_hand
        else:
            # For inference or 10% of training, use proper sampling
            for s in range(num_samples):
                # Direct sampling from beliefs (much faster)
                for i in range(num_opponents):
                    probs = constrained_beliefs[:, i]
                    # Sample using multinomial (vectorized)
                    sample = torch.multinomial(
                        probs.reshape(batch_size, -1).clamp(min=1e-8),
                        num_samples=5,  # Sample 5 cards
                        replacement=True
                    )
                    
                    # Count card occurrences
                    for b in range(batch_size):
                        for card_idx in sample[b]:
                            sampled_hands[b, s, i, card_idx] += 1
        
        return sampled_hands
    
    def infer_belief_from_game_state(self, observation, agent_idx, env):
        """
        Infer belief state directly from game state for ground truth training.
        Now respects physical constraints.
        
        Args:
            observation: Current observation (can be numpy array, list, or tensor).
            agent_idx: Index of the agent.
            env: Environment instance with full state.
            
        Returns:
            Belief state representing ground truth probabilities for table/non-table cards.
        """
        # Convert observation to tensor if needed
        if not isinstance(observation, torch.Tensor):
            if isinstance(observation, np.ndarray):
                obs_tensor = torch.from_numpy(observation).float()
            else:
                obs_tensor = torch.tensor(observation, dtype=torch.float)
            
            # Add batch dimension if needed
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
        else:
            obs_tensor = observation
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
        
        agent_name = env.possible_agents[agent_idx]
        opponents = [ag for ag in env.possible_agents if ag != agent_name]
        num_opponents = len(opponents)
        
        # Start with uniform beliefs
        beliefs = torch.ones(1, num_opponents, self.card_types) / self.card_types
        
        # Extract observer's hand information
        _, private_obs = self.split_observation(obs_tensor)
        
        # Update based on observed plays
        for i, opponent in enumerate(opponents):
            history = env.public_opponent_histories.get(opponent, [])
            cards_remaining = len(env.players_hands.get(opponent, [])) / 5.0  # Normalized hand size
            
            # Start with an estimate based on observed plays
            for entry in history:
                if entry['action_type'] == "Play" and entry.get('was_bluff') is not None:
                    if entry['count'] is not None:
                        count = entry['count']
                        if entry['was_bluff'] is True:
                            # Was bluffing - decrease belief in table cards
                            beliefs[0, i, 0] *= 0.5  # Reduce probability of table cards
                        else:
                            # Was truthful - increase belief in table cards
                            beliefs[0, i, 0] *= 1.5  # Increase probability of table cards
            
            # Normalize to ensure valid probabilities
            beliefs[0, i] = beliefs[0, i] / (beliefs[0, i].sum() + 1e-10)
        
        # Apply correlation and physical constraints
        constrained_beliefs = self.model_correlations(beliefs, private_obs)
        
        return constrained_beliefs


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
