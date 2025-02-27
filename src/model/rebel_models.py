# src/model/rebel_models.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BeliefStateModel(nn.Module):
    """
    Models probability distributions over opponents' hands based on game history.
    Separates public and private belief states for proper ReBeL implementation.
    
    Public belief: Information available to all players (action history, visible cards)
    Private belief: Player-specific information and inferences
    """
    def __init__(self, input_dim, hidden_dim, deck_size, num_players, 
                 use_dropout=True, use_layer_norm=True, update_mode='multiplicative'):
        super(BeliefStateModel, self).__init__()
        self.deck_size = deck_size  # Total number of cards in deck
        self.num_players = num_players
        self.card_types = 2  # Simplified: (table_card, non_table_card)
        self.update_mode = update_mode  # Options: 'multiplicative' or 'weighted'
        
        # Determine dimensions for public and private parts of the observation
        # Public: opponent actions, table card, round info, etc.
        # Private: player's hand, specific observations
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
        
        # Belief update network – outputs logits for card type probabilities per opponent
        self.belief_update = nn.Linear(hidden_dim, (num_players - 1) * self.card_types)
        
        # Initialize with a uniform prior
        self.register_buffer('prior_belief', torch.ones(1, num_players - 1, self.card_types) / self.card_types)
        
        # If using weighted blending, add a learnable parameter alpha
        if self.update_mode == 'weighted':
            self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def split_observation(self, x):
        """
        Split observation into public and private components.
        
        Args:
            x: Full observation tensor [batch_size, obs_dim]
            
        Returns:
            (public_obs, private_obs): Tuple of public and private observation tensors
        """
        batch_size = x.size(0)
        
        # First two elements are the player's hand information (table cards, non-table cards)
        private_obs = x[:, :self.private_dim]
        
        # Remaining elements are public information
        public_obs = x[:, self.private_dim:]
        
        return public_obs, private_obs
    
    def get_public_belief_state(self, x, prev_beliefs=None):
        """
        Extract only the public belief state (independent of the player's private info).
        
        Args:
            x: Observation tensor [batch_size, obs_dim]
            prev_beliefs: Previous belief state or None
            
        Returns:
            Public belief state [batch_size, num_opponents, card_types]
        """
        batch_size = x.size(0)
        public_obs, _ = self.split_observation(x)
        
        # Process only public information
        public_features = self.public_encoder(public_obs)
        
        # Process directly to get belief update based only on public info
        update_logits = self.belief_update(public_features)
        update_logits = update_logits.view(batch_size, self.num_players - 1, self.card_types)
        belief_update = F.softmax(update_logits, dim=-1)
        
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
        
        # Renormalize to ensure a valid probability distribution
        updated_beliefs = updated_beliefs / (updated_beliefs.sum(dim=-1, keepdim=True) + 1e-10)
        return updated_beliefs
    
    def forward(self, x, prev_beliefs=None):
        """
        Full belief update using both public and private information.
        
        Args:
            x: Observation tensor [batch_size, obs_dim]
            prev_beliefs: Previous belief state [batch_size, num_opponents, card_types] or None
            
        Returns:
            Updated belief state [batch_size, num_opponents, card_types]
        """
        batch_size = x.size(0)
        public_obs, private_obs = self.split_observation(x)
        
        # Process public and private information separately
        public_features = self.public_encoder(public_obs)
        private_features = self.private_encoder(private_obs)
        
        # Combine features
        combined_features = torch.cat([public_features, private_features], dim=1)
        joint_features = self.joint_encoder(combined_features)
        
        # Generate belief update
        update_logits = self.belief_update(joint_features)
        update_logits = update_logits.view(batch_size, self.num_players - 1, self.card_types)
        belief_update = F.softmax(update_logits, dim=-1)
        
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
        
        # Renormalize to ensure a valid probability distribution
        updated_beliefs = updated_beliefs / (updated_beliefs.sum(dim=-1, keepdim=True) + 1e-10)
        return updated_beliefs
    
    def infer_belief_from_game_state(self, observation, agent_idx, env):
        """
        Infer belief state directly from game state for ground truth training.
        Creates target beliefs based on known information.
        
        Args:
            observation: Current observation.
            agent_idx: Index of the agent.
            env: Environment instance with full state.
            
        Returns:
            Belief state representing ground truth probabilities for table/non-table cards.
        """
        agent_name = env.possible_agents[agent_idx]
        opponents = [ag for ag in env.possible_agents if ag != agent_name]
        num_opponents = len(opponents)
        beliefs = torch.ones(1, num_opponents, self.card_types) / self.card_types
        
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