# src/model/recursive_search_agent.py
import torch
import numpy as np
from collections import defaultdict, namedtuple
from src.training.train_transformer import convert_memory_to_features
from src.env.liars_deck_env_utils_2 import decode_action
from src.env.liars_deck_env_utils import query_opponent_memory_full
class RecursiveSearchAgent:
    def __init__(self, policy_net, belief_model, value_net, env_creator, 
            device, search_depth=4, num_simulations=30, c_puct=1.0,
            agent_name=None, agent_index=None, blueprint=None,
            alpha=1.5, beta=0.5, gamma=2.0,
            strategy_transformer=None, event_encoder=None,
            response2idx=None, action2idx=None):
        """
        Agent that uses belief-based recursive search for decision making.
        Implements proper Counterfactual Regret Minimization (CFR) with
        public/private belief state separation.
        
        Args:
            policy_net: Policy network to generate prior probabilities
            belief_model: Model for tracking belief states
            value_net: Value network for evaluating belief states
            env_creator: Function that creates a copy of the environment for simulation
            device: Torch device to use
            search_depth: Maximum depth of recursive search
            num_simulations: Number of simulations per search
            c_puct: Exploration constant for PUCT algorithm
            agent_name: Name of the agent
            agent_index: Index of the agent in the game
            blueprint: Optional blueprint module for prior strategies
            alpha: DCFR positive regret discount parameter
            beta: DCFR negative regret discount parameter
            gamma: DCFR average strategy discount parameter
        """
        self.policy_net = policy_net
        self.belief_model = belief_model
        self.value_net = value_net
        self.env_creator = env_creator
        self.device = device
        self.search_depth = search_depth
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.name = agent_name
        self.agent_index = agent_index
        self.blueprint = blueprint
        self.strategy_transformer = strategy_transformer
        self.event_encoder = event_encoder
        self.response2idx = response2idx
        self.action2idx = action2idx
        # DCFR parameters
        self.alpha = alpha  # Positive regret discount (typically 1.5)
        self.beta = beta    # Negative regret discount (typically 0.5)
        self.gamma = gamma  # Average strategy discount (typically 2.0)
        
        # Persistent tree structure
        self.tree = {}  # Maps state_keys to TreeNode objects
        
        # Optimization data structures
        self.transposition_table = {}
        self.value_statistics = {}
        self.last_observation = None
        self.tree_reuse_count = 0
        self.tree_rebuild_count = 0
        
        # Belief state tracking
        self.current_beliefs = None
        self.current_public_beliefs = None
        self.action_history = []
        self.search_statistics = {}
        
        # CFR data structures
        self.iterations = defaultdict(int)
        self.cumulative_regrets = defaultdict(lambda: np.zeros(7))  # For 7 possible actions
        self.average_strategy = defaultdict(lambda: np.zeros(7))
        self.strategy_update_count = defaultdict(int)
        
        # Add opponent memory support
        
        # Define a TreeNode class for persistent tree structure
        self.TreeNode = namedtuple('TreeNode', ['state', 'player_id', 'children', 'parent', 'depth'])
        
        # Create a buffer for batched neural network evaluation
        self.nn_batch_size = 8  # Configurable
        self.nn_query_buffer = []
        self.nn_result_buffer = []

    def reset(self):
        """Reset agent state at the beginning of a new game."""
        self.current_beliefs = None
        self.current_public_beliefs = None
        self.action_history = []
        self.search_statistics = {}
        # Note: We don't reset cumulative_regrets or average_strategy as they persist across games

    def get_transformer_memory_embeddings(self, env):
        """
        Generate transformer-based memory embeddings for all opponents.
        
        Args:
            env: Current game environment
            
        Returns:
            Tuple of (list of embeddings, normalized flattened embeddings array)
        """
        # Skip if we don't have transformer components
        if not hasattr(self, 'strategy_transformer') or not self.strategy_transformer:
            return [], np.zeros(0, dtype=np.float32)
        
        embeddings_list = []
        opponents = [ag for ag in env.possible_agents if ag != env.agent_selection]
        strategy_dim = self.strategy_transformer.strategy_head.out_features
        
        # Process each opponent
        for opp in opponents:
            try:
                # Get opponent memory events 
                from src.env.liars_deck_env_utils import query_opponent_memory_full
                mem_summary = query_opponent_memory_full(env.agent_selection, opp)
                
                # Convert memory to feature format expected by EventEncoder
                from src.training.train_transformer import convert_memory_to_features
                features_list = convert_memory_to_features(mem_summary, self.response2idx, self.action2idx)
                
                if features_list:
                    # Create proper tensor for event encoder
                    feature_tensor = torch.tensor(features_list, dtype=torch.float32, device=self.device).unsqueeze(0)
                    
                    # Process through event encoder then transformer
                    with torch.no_grad():
                        projected = self.event_encoder(feature_tensor)
                        strategy_embedding, _ = self.strategy_transformer(projected)
                    
                    # Store embedding
                    embeddings_list.append(strategy_embedding.cpu().detach().numpy().flatten())
                else:
                    # Use zeros for empty memory
                    embeddings_list.append(np.zeros(strategy_dim, dtype=np.float32))
            except Exception as e:
                print(f"Error processing memory for {opp}: {e}")
                embeddings_list.append(np.zeros(strategy_dim, dtype=np.float32))
        
        # Concatenate all embeddings into a single array
        if embeddings_list:
            embeddings_arr = np.concatenate(embeddings_list, axis=0)
        else:
            # Create zeroed embeddings for each opponent if needed
            embeddings_arr = np.zeros(len(opponents) * strategy_dim, dtype=np.float32)
        
        # Normalize using min-max scaling for the one-dimensional array
        min_val = embeddings_arr.min()
        max_val = embeddings_arr.max()
        if (max_val - min_val) == 0:
            normalized_arr = embeddings_arr
        else:
            normalized_arr = (embeddings_arr - min_val) / (max_val - min_val)
        
        return embeddings_list, normalized_arr

    def _update_value_statistics(self, state_key, value):
        """Update running statistics for a state."""
        if state_key not in self.value_statistics:
            self.value_statistics[state_key] = [value, 0.0, 1]  # [mean, std, count]
        else:
            mean, std, count = self.value_statistics[state_key]
            new_count = count + 1
            delta = value - mean
            new_mean = mean + delta / new_count
            delta2 = value - new_mean
            new_std = np.sqrt((std ** 2 * count + delta * delta2) / new_count)
            self.value_statistics[state_key] = [new_mean, new_std, new_count]
    
    def set_training_mode(self, training=True):
        """
        Set agent to training or evaluation mode.
        In training mode, several optimizations are applied for speed.
        
        Args:
            training: Whether to enable training mode
        """
        self.policy_net.train(training)
        self.belief_model.train(training)
        self.value_net.train(training)
        
        # Set search parameters based on mode
        if training:
            # Use reduced parameters during training for speed
            self._original_search_depth = self.search_depth
            self._original_num_simulations = self.num_simulations
            self._original_c_puct = self.c_puct
            
            # Reduce parameters for faster training
            self.search_depth = min(3, self.search_depth)
            self.num_simulations = min(20, self.num_simulations)
            self.c_puct = 1.0
        else:
            # Restore original parameters for evaluation
            if hasattr(self, '_original_search_depth'):
                self.search_depth = self._original_search_depth
                self.num_simulations = self._original_num_simulations
                self.c_puct = self._original_c_puct

    def create_node_key(self, public_obs, beliefs):
        """
        Create a unique and efficient key for identifying game states in the transposition table.
        
        Args:
            public_obs: Public observation numpy array
            beliefs: Belief state tensor
            
        Returns:
            A hashable key that uniquely identifies this public belief state
        """
        # Convert to numpy if tensor
        if isinstance(beliefs, torch.Tensor):
            # Use a deterministic way to convert beliefs to bytes for hashing
            beliefs_np = beliefs.detach().cpu().numpy()
        else:
            beliefs_np = beliefs
            
        # Create a stable string representation of the observations
        obs_str = "_".join([f"{x:.5f}" for x in public_obs])
        
        # Create a stable hash of the beliefs - using only the most significant digits
        # to avoid floating point precision issues
        belief_str = "_".join([f"{x:.5f}" for x in beliefs_np.flatten()])
        
        # Combine and hash
        return hash(f"{obs_str}_{belief_str}")

    def check_early_termination(self, state_key, depth, reach_prob=1.0):
        """
        Enhanced check if simulation can terminate early based on value statistics.
        Much more aggressive termination to reduce simulation costs.
        
        Args:
            state_key: The key identifying the state in transposition tables
            depth: Current search depth
            reach_prob: Reach probability for this state
            
        Returns:
            (should_terminate, value): Boolean indicating if we can terminate early and estimated value
        """
        # Aggressive early termination based on depth
        # For deep nodes, frequently terminate early
        if depth >= 3:
            # For very deep nodes, terminate with 90% probability
            if np.random.random() < 0.9:
                # Use value from table if available, otherwise estimate as 0
                if state_key in self.value_statistics:
                    mean_value, _, _ = self.value_statistics[state_key]
                    return True, mean_value
                return True, 0.0
        
        # For medium-depth nodes, be somewhat aggressive
        elif depth == 2:
            # Terminate with 50% probability for depth 2
            if np.random.random() < 0.5:
                if state_key in self.value_statistics:
                    mean_value, _, _ = self.value_statistics[state_key]
                    return True, mean_value
                return True, 0.0
        
        # Original early termination logic with relaxed criteria
        if depth <= 1 or state_key not in self.value_statistics:
            return False, 0.0
            
        # Get accumulated statistics
        mean_value, std_value, count = self.value_statistics[state_key]
        
        # Relax required sample count - need fewer samples for reliable estimate
        if count < 5:  # Was 10
            return False, 0.0
            
        std_error = std_value / np.sqrt(count)
        
        # Relax confidence threshold
        confidence_threshold = 0.1  # Was 0.05 - doubled for faster termination
        
        # Scale threshold based on depth and reach probability
        adjusted_threshold = confidence_threshold * (1.0 + depth) / reach_prob
        
        if std_error < adjusted_threshold:
            return True, mean_value
            
        return False, 0.0

    def split_observation(self, observation):
        """
        Split observation into public and private components.
        
        Args:
            observation: Full observation (NumPy array)
            
        Returns:
            (public_obs, private_obs): Tuple of public and private observation tensors
        """
        # First two elements are the player's hand information (table cards, non-table cards)
        private_obs = observation[:2]
        
        # Remaining elements are public information
        public_obs = observation[2:]
        
        return public_obs, private_obs

    def update_beliefs(self, observation, action_mask=None):
        """
        Update both public and private belief states based on new observation,
        now with transformer-based memory and caching optimizations.
        """
        # Extract observation
        if isinstance(observation, dict):
            obs_data = observation[self.name]
        else:
            obs_data = observation

        # Check belief cache (optimization)
        if not hasattr(self, '_belief_cache'):
            self._belief_cache = {}
            self._belief_cache_hits = 0
            self._belief_cache_misses = 0
        
        # Create a cache key based on observation
        # Using string representation of numpy array with reduced precision for better hash collisions
        if isinstance(obs_data, np.ndarray):
            obs_key = hash(str(np.round(obs_data, 3)))
        else:
            obs_key = hash(str(obs_data))
        
        # Check if we have a cached belief for this observation
        if obs_key in self._belief_cache:
            self._belief_cache_hits += 1
            cached_data = self._belief_cache[obs_key]
            # Only reuse if less than 5 updates old
            if cached_data['age'] < 5:
                self.current_beliefs = cached_data['beliefs']
                self.current_public_beliefs = cached_data['public_beliefs']
                cached_data['age'] += 1
                return
        
        self._belief_cache_misses += 1
        
        # Convert to tensor
        if not isinstance(obs_data, torch.Tensor):
            self._last_obs_data = obs_data
            if hasattr(self, '_last_obs_tensor') and self._last_obs_tensor is not None and len(obs_data) == self._last_obs_tensor.size(1):
                self._last_obs_tensor[0].copy_(torch.tensor(obs_data, dtype=torch.float))
                obs_tensor = self._last_obs_tensor
            else:
                obs_tensor = torch.FloatTensor(obs_data).unsqueeze(0).to(self.device)
                self._last_obs_tensor = obs_tensor
        else:
            obs_tensor = obs_data.unsqueeze(0) if obs_data.dim() == 1 else obs_data
            obs_tensor = obs_tensor.to(self.device)

        # Extract private hand
        private_hand = obs_tensor[:, :2]

        with torch.no_grad():
            # Check if we need a full update
            if not hasattr(self, '_belief_update_counter'):
                self._belief_update_counter = 0
            self._belief_update_counter += 1

            # Reduce frequency of full updates (optimization)
            is_full_update = self._belief_update_counter % 4 == 0 or not self.policy_net.training

            # Create environment instance for counterfactual reasoning and transformer memory extraction
            current_env = self.env_creator()

            # Optimization: Cache transformer memory embeddings
            if not hasattr(self, '_cached_transformer_embeddings') or self._belief_update_counter % 5 == 0:
                # Only update transformer embeddings periodically
                embeddings_list, normalized_arr = self.get_transformer_memory_embeddings(current_env)
                self._cached_transformer_embeddings = (embeddings_list, normalized_arr)
            else:
                embeddings_list, normalized_arr = self._cached_transformer_embeddings

            if is_full_update:
                # Reconstruct game history for counterfactual reasoning
                if hasattr(self, '_game_history') and self._belief_update_counter % 8 != 0:
                    # Reuse game history reconstruction for efficiency
                    game_history = self._game_history
                else:
                    game_history = self._reconstruct_game_history(current_env)
                    self._game_history = game_history

                # Use counterfactual belief inference with game history
                if self.current_beliefs is None:
                    # For the first update, use the model directly with transformer features
                    self.current_beliefs = self.belief_model.infer_belief_from_game_state(
                        obs_tensor, self.agent_index, current_env, transformer_features=normalized_arr)
                else:
                    # For subsequent updates, use the Bayesian update with transformer features
                    self.current_beliefs = self._compute_counterfactual_beliefs(
                        obs_tensor, self.current_beliefs, game_history, current_env, 
                        transformer_features=normalized_arr)

                # Apply physical constraints
                self.current_beliefs = self.belief_model.apply_physical_constraints_fast(
                    self.current_beliefs, private_hand)
            else:
                # For non-full updates, use simple belief update
                if self.current_beliefs is None:
                    # First update must be full
                    self.current_beliefs = self.belief_model.infer_belief_from_game_state(
                        obs_tensor, self.agent_index, current_env, transformer_features=normalized_arr)
                    self.current_beliefs = self.belief_model.apply_physical_constraints_fast(
                        self.current_beliefs, private_hand)
                else:
                    # Simple forward pass for other updates
                    self.current_beliefs = self.belief_model(obs_tensor, self.current_beliefs)
                    self.current_beliefs = self.belief_model.apply_physical_constraints_fast(
                        self.current_beliefs, private_hand)

            # Also update public beliefs
            if self.current_public_beliefs is None:
                self.current_public_beliefs = self.belief_model.get_public_belief_state(obs_tensor)
            else:
                # Only do full public belief updates periodically
                if self._belief_update_counter % 3 == 0:
                    self.current_public_beliefs = self.belief_model.get_public_belief_state(
                        obs_tensor, self.current_public_beliefs)
                    
                    # Apply physical constraints to public beliefs as well
                    self.current_public_beliefs = self.belief_model.apply_physical_constraints_fast(
                        self.current_public_beliefs, private_hand)
        
        # Cache the beliefs for future reuse
        self._belief_cache[obs_key] = {
            'beliefs': self.current_beliefs,
            'public_beliefs': self.current_public_beliefs,
            'age': 0
        }
        
        # Manage cache size
        if len(self._belief_cache) > 1000:
            # Remove oldest entries
            self._belief_cache = {k: v for k, v in self._belief_cache.items() if v['age'] < 3}
            
    def compute_cfr_strategy(self, state_key, action_mask):
        """
        Compute a strategy according to the DCFR algorithm using discounted regrets.
        Optimized version with caching and simplified computation.

        Args:
            state_key: A unique identifier for the current state
            action_mask: Boolean mask of valid actions
            
        Returns:
            numpy array: A probability distribution over actions
        """
        # Check strategy cache first (optimization)
        if not hasattr(self, '_strategy_cache'):
            self._strategy_cache = {}
            self._strategy_cache_hits = 0
            self._strategy_cache_misses = 0
        
        # Create a cache key that includes action mask hash
        cache_key = (state_key, hash(tuple(action_mask)))
        
        if cache_key in self._strategy_cache:
            self._strategy_cache_hits += 1
            # If cache entry is new, use it directly
            cache_entry = self._strategy_cache[cache_key]
            if cache_entry['age'] < 5:
                cache_entry['age'] += 1
                return cache_entry['strategy'].copy()
        
        self._strategy_cache_misses += 1
        
        # Get cumulative regrets for this state
        regrets = self.cumulative_regrets[state_key]

        # Get iteration count for this state (increment first)
        self.iterations[state_key] += 1
        t = self.iterations[state_key]
        
        # Optimization: for low iteration counts, use simplified calculation
        if t <= 3:
            # For early iterations, just use regret matching with minimal computation
            positive_regrets = np.maximum(regrets, 0)
            masked_regrets = positive_regrets * action_mask
            regret_sum = np.sum(masked_regrets)
            
            if regret_sum <= 1e-8:
                valid_actions = np.where(action_mask)[0]
                strategy = np.zeros_like(action_mask, dtype=np.float32)
                if len(valid_actions) > 0:
                    strategy[valid_actions] = 1.0 / len(valid_actions)
                # Cache the result
                self._strategy_cache[cache_key] = {'strategy': strategy.copy(), 'age': 0}
                return strategy
            
            strategy = masked_regrets / regret_sum
            # Cache the result
            self._strategy_cache[cache_key] = {'strategy': strategy.copy(), 'age': 0}
            return strategy

        # Apply DCFR discounting - separate treatment for positive and negative regrets
        positive_regrets = np.maximum(regrets, 0)
        negative_regrets = np.minimum(regrets, 0)

        # Optimization: use precomputed discount factors for common iteration counts
        if t <= 100:
            # For common iteration counts, use lookup table
            alpha_discount = self._get_discount_factor(t, self.alpha)
            beta_discount = self._get_discount_factor(t, self.beta)
        else:
            # For high iteration counts, compute directly
            alpha_discount = (t ** self.alpha) / ((t+1) ** self.alpha)
            beta_discount = (t ** self.beta) / ((t+1) ** self.beta)
        
        # Apply discounts
        discounted_positive = positive_regrets * alpha_discount
        discounted_negative = negative_regrets * beta_discount
        
        # Combine discounted regrets
        discounted_regrets = discounted_positive + discounted_negative

        # Apply action mask and add epsilon for numerical stability
        epsilon = 1e-8  # Small constant to ensure numerical stability
        masked_regrets = np.maximum(discounted_regrets * action_mask, epsilon)
        regret_sum = np.sum(masked_regrets)

        # If sum is too small, use uniform random over valid actions
        if regret_sum <= epsilon:
            valid_actions = np.where(action_mask)[0]
            strategy = np.zeros_like(action_mask, dtype=np.float32)
            if len(valid_actions) > 0:  # Safety check
                strategy[valid_actions] = 1.0 / len(valid_actions)
            # Cache the result
            self._strategy_cache[cache_key] = {'strategy': strategy.copy(), 'age': 0}
            return strategy

        # Normalize by sum of positive regrets (regret matching)
        strategy = masked_regrets / regret_sum
        
        # Optimization: Only update average strategy periodically to save computation
        if t % 2 == 0:  # Only update every other iteration
            # Update average strategy using DCFR weighting
            self.strategy_update_count[state_key] += 1
            if self.strategy_update_count[state_key] > 1:
                weight = self._get_discount_factor(t, self.gamma)
                self.average_strategy[state_key] = (
                    self.average_strategy[state_key] * (1 - weight) +
                    strategy * weight
                )
            else:
                self.average_strategy[state_key] = strategy.copy()
        
        # Cache the result
        self._strategy_cache[cache_key] = {'strategy': strategy.copy(), 'age': 0}
        
        # Manage cache size - periodically clear old entries
        if len(self._strategy_cache) > 10000:
            # Keep only recent entries
            self._strategy_cache = {k: v for k, v in self._strategy_cache.items() if v['age'] < 3}
        
        return strategy

    def _get_discount_factor(self, t, exponent):
        """
        Helper method to get discount factors for CFR.
        Uses precomputed values for common iteration counts.
        
        Args:
            t: Current iteration count
            exponent: Discount exponent (alpha, beta, or gamma)
        
        Returns:
            float: Discount factor
        """
        # Initialize discount factor lookup table if not exists
        if not hasattr(self, '_discount_factors'):
            self._discount_factors = {}
            # Precompute common discount factors for various t values and exponents
            for exp in [self.alpha, self.beta, self.gamma]:
                self._discount_factors[exp] = {}
                for i in range(1, 101):
                    self._discount_factors[exp][i] = (i ** exp) / ((i+1) ** exp)
        
        # Return precomputed value if available
        if exponent in self._discount_factors and t in self._discount_factors[exponent]:
            return self._discount_factors[exponent][t]
        
        # Compute directly otherwise
        return (t ** exponent) / ((t+1) ** exponent)

    def update_average_strategy(self, state_key, current_strategy):
        """
        Update the average strategy for a state using DCFR discounting.
        
        Args:
            state_key: A unique identifier for the current state
            current_strategy: The current strategy for this state
        """
        # Increment counter for this state
        self.strategy_update_count[state_key] += 1
        count = self.strategy_update_count[state_key]
        
        # Get iteration count
        t = self.iterations[state_key]
        
        # Apply DCFR discounting - weight by t^gamma
        contribution_weight = t ** self.gamma
        
        # Simple running average update (safer)
        alpha = 1.0 / count
        
        # Update running average
        self.average_strategy[state_key] = (
            (1 - alpha) * self.average_strategy[state_key] + 
            alpha * current_strategy
        )
        
        # Ensure the result is non-negative
        self.average_strategy[state_key] = np.maximum(self.average_strategy[state_key], 0)

    def get_average_strategy(self, state_key, action_mask):
        """
        Get the average strategy for a state (normalized).
        
        Args:
            state_key: A unique identifier for the current state
            action_mask: Boolean mask of valid actions
            
        Returns:
            numpy array: A probability distribution over actions
        """
        strategy = self.average_strategy[state_key]
        
        # Ensure all values are non-negative (fix for the error)
        strategy = np.maximum(strategy, 0)
        
        # If state never seen, use uniform random over valid actions
        if np.sum(strategy) <= 0:
            valid_actions = np.where(action_mask)[0]
            strategy = np.zeros_like(action_mask, dtype=np.float32)
            strategy[valid_actions] = 1.0 / len(valid_actions)
            return strategy
        
        # Ensure the strategy is valid for current action mask
        masked_strategy = strategy * action_mask
        
        # If no valid actions in strategy, use uniform over valid actions
        if np.sum(masked_strategy) <= 0:
            valid_actions = np.where(action_mask)[0]
            masked_strategy = np.zeros_like(action_mask, dtype=np.float32)
            masked_strategy[valid_actions] = 1.0 / len(valid_actions)
            return masked_strategy
        
        # Normalize to ensure it's a valid probability distribution
        return masked_strategy / np.sum(masked_strategy)

    def _reconstruct_game_history(self, env):
        """
        Reconstruct the game history for counterfactual reasoning.
        
        Args:
            env: Current game environment
            
        Returns:
            Dictionary mapping opponents to their decision points
        """
        opponents = [ag for ag in env.possible_agents if ag != self.name]
        history = {}
        
        # For each opponent, extract their decision points
        for opponent in opponents:
            history[opponent] = []
            decisions = env.public_opponent_histories.get(opponent, [])
            
            for decision in decisions:
                action_type = decision.get('action_type')
                if action_type:
                    decision_point = {
                        'action_type': action_type,
                        'count': decision.get('count'),
                        'was_challenged': decision.get('was_challenged', False),
                        'was_bluff': decision.get('was_bluff')
                    }
                    history[opponent].append(decision_point)
        
        return history

    def _compute_counterfactual_beliefs(self, obs_tensor, current_beliefs, game_history, env, transformer_features=None):
        """
        Compute counterfactual beliefs based on game history.
        
        Args:
            obs_tensor: Current observation tensor
            current_beliefs: Current belief state
            game_history: Reconstructed game history
            env: Current game environment
            transformer_features: Optional transformer-based memory features
            
        Returns:
            Updated belief state using counterfactual reasoning
        """
        return self.belief_model.infer_belief_from_game_state(
            obs_tensor, self.agent_index, env, transformer_features=transformer_features)

    def mcts_search(self, observation, action_mask):
        """
        Perform Monte Carlo Tree Search with belief-based CFR.
        
        Args:
            observation: Current observation.
            action_mask: Mask of valid actions.
            
        Returns:
            Dictionary with search outcomes
        """
        # Convert observation appropriately
        if isinstance(observation, dict):
            observation = observation[self.name]
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Ensure beliefs are updated
        self.update_beliefs(observation, action_mask)
        
        # Split observation into public and private parts
        public_obs, private_obs = self.split_observation(observation)
        public_obs_tensor = torch.FloatTensor(public_obs).unsqueeze(0).to(self.device)
        
        # Get opponent ID from environment (first agent that's not us)
        current_env = self.env_creator()
        opponents = [ag for ag in current_env.possible_agents if ag != self.name]
        opponent_id = opponents[0] if opponents else None
        
        # Create a unique key for this PUBLIC state (key for CFR)
        public_state_key = self.create_node_key(np.array(public_obs), self.current_public_beliefs)
        
        # Clear batch buffers for this search
        self.nn_query_buffer = []
        self.nn_result_buffer = []
        
        # Get policy network's action priors
        with torch.no_grad():
            # Use the full policy (with private info) for priors
            priors, state_value, _ = self.policy_net(obs_tensor, self.current_beliefs)
            priors = priors.squeeze(0).cpu().numpy()
            
            # Also get public-only policy for comparison
            public_priors, public_value, _ = self.policy_net.public_policy(
                public_obs_tensor, self.current_public_beliefs)
            public_priors = public_priors.squeeze(0).cpu().numpy()
        
        # If we have a blueprint, query it with opponent identity
        blueprint_strategy = None
        blueprint_value = None
        if self.blueprint:
            # Query blueprint for prior strategy and value, including opponent_id
            blueprint_strategy, blueprint_value = self.blueprint.query(
                np.array(public_obs), 
                self.current_public_beliefs.cpu().numpy(), 
                action_mask,
                opponent_id  # Include opponent identity
            )
            # Use blueprint as prior (higher weight) combined with policy network
            blueprint_weight = 0.5  # Configurable
            combined_priors = blueprint_weight * blueprint_strategy + (1 - blueprint_weight) * priors
        else:
            # Use policy network normally if no blueprint
            combined_priors = priors
        
        # Apply action mask to priors
        masked_priors = combined_priors * action_mask
        if np.sum(masked_priors) > 0:
            masked_priors = masked_priors / np.sum(masked_priors)
        else:
            valid_actions = np.where(action_mask)[0]
            masked_priors = np.zeros_like(combined_priors)
            if len(valid_actions) > 0:  # Safety check
                masked_priors[valid_actions] = 1.0 / len(valid_actions)
        
        # Initialize search statistics: visit count (N), total value (W), and mean value (Q)
        N = {a: 0 for a in range(len(action_mask))}
        W = {a: 0.0 for a in range(len(action_mask))}
        Q = {a: 0.0 for a in range(len(action_mask))}
        
        # Track counterfactual values across all simulations
        cf_values = defaultdict(float)
        
        # Add virtual loss tracking to discourage parallel exploration of the same path
        virtual_loss = {a: 0.0 for a in range(len(action_mask))}
        
        # Run MCTS simulations
        from tqdm import tqdm
        for sim_idx in tqdm(range(self.num_simulations), desc="MCTS simulations", leave=False):
            sim_env = self.env_creator()  # Clone environment for simulation
            
            # Compute current CFR strategy using PUBLIC state key
            cfr_strategy = self.compute_cfr_strategy(public_state_key, action_mask)
            
            # Select action using adaptive exploration in the PUCT formula
            valid_actions = np.where(action_mask)[0]
            if not len(valid_actions):  # Safety check
                continue
                
            best_score = -float('inf')
            best_action = valid_actions[0]
            
            # Total visit count for normalization
            total_visits = sum(N.values()) or 1
            
            for action in valid_actions:
                visit_count = N[action]
                
                # Adaptive exploration factor that decreases with simulation count
                # and decreases with depth in the search tree
                adaptive_c = self.c_puct * (1.0 - (sim_idx / (2 * self.num_simulations)))
                adaptive_c = max(0.5, adaptive_c)  # Ensure minimum exploration
                
                # Progressive widening coefficient - controls exploration vs. exploitation
                pw_coeff = 0.5
                if total_visits == 0:
                    # First visit - use prior
                    exploration = adaptive_c * masked_priors[action]
                elif visit_count > 0:
                    # Use PUCT formula with progressive widening
                    # More visited nodes should be exploited more
                    exploitation = Q[action]
                    exploration = adaptive_c * cfr_strategy[action] * np.sqrt(np.log(total_visits) / (visit_count + 1e-5))
                    
                    # If we have visited this action more than expected by progressive widening,
                    # reduce exploration bonus
                    if visit_count > (total_visits ** pw_coeff):
                        exploration *= 0.5
                else:
                    # Unvisited node - high exploration
                    exploitation = 0
                    exploration = adaptive_c * cfr_strategy[action] * np.sqrt(np.log(total_visits) / 1e-5)
                
                # Apply virtual loss penalty to discourage siblings exploring same path
                temp_Q = Q[action] if N[action] > 0 else 0
                virtual_loss_penalty = (virtual_loss[action] * 0.1 / max(N[action], 1))
                score = temp_Q - virtual_loss_penalty + exploration
                
                if score > best_score:
                    best_score = score
                    best_action = action
            
            # Apply virtual loss for the selected action before simulation
            virtual_loss[best_action] += 1.0
            
            # Run simulation from the selected node
            sim_value, action_cf_values = self._simulate(
                sim_env, best_action, observation, public_obs, 
                self.current_beliefs, self.current_public_beliefs, 
                self.search_depth, reach_prob=1.0)
            
            # Remove virtual loss after simulation
            virtual_loss[best_action] -= 1.0
            
            # Update statistics for the selected action
            N[best_action] += 1
            W[best_action] += sim_value
            Q[best_action] = W[best_action] / N[best_action]
            
            # Update counterfactual values
            for a, value in action_cf_values.items():
                cf_values[a] += value / self.num_simulations
            
            # Update value statistics for this public state
            self._update_value_statistics(public_state_key, sim_value)
        
        # Compute overall value estimate as weighted average of Q-values
        value_sum = sum(N[a] * Q[a] for a in range(len(action_mask)))
        total_visits = sum(N.values())
        value_estimate = value_sum / max(total_visits, 1)
        
        # Compute counterfactual regrets based on final counterfactual values
        immediate_regrets = np.zeros(len(action_mask), dtype=np.float32)
        for a in range(len(action_mask)):
            if a in cf_values:
                immediate_regrets[a] = cf_values[a] - value_estimate
            elif action_mask[a]:  # Only compute regrets for valid actions
                # If no CF value, use Q-value if available, otherwise 0
                q_value = Q.get(a, 0)
                immediate_regrets[a] = q_value - value_estimate if N.get(a, 0) > 0 else 0
        
        # Update cumulative regrets (core CFR update step)
        self.cumulative_regrets[public_state_key] += immediate_regrets * action_mask
        
        # Compute current strategy using regret matching
        cfr_strategy = self.compute_cfr_strategy(public_state_key, action_mask)
        
        # Get the average strategy (this is what we use for actual play)
        average_strategy = self.get_average_strategy(public_state_key, action_mask)
        
        # Store search statistics for analysis and debugging
        self.search_statistics = {
            'N': N,
            'Q': Q,
            'masked_priors': masked_priors,
            'public_priors': public_priors,
            'value_estimate': value_estimate,
            'immediate_regrets': immediate_regrets,
            'cfr_strategy': cfr_strategy,
            'average_strategy': average_strategy,
            'counterfactual_values': cf_values,
            'blueprint_strategy': blueprint_strategy,
            'blueprint_value': blueprint_value
        }
        
        # Update the blueprint if we have one
        if self.blueprint and hasattr(self, 'average_strategy'):
            self.blueprint.update_from_search(
                np.array(public_obs),
                self.current_public_beliefs.cpu().numpy(),
                average_strategy,  # CFR average strategy
                value_estimate,
                immediate_regrets,
                visits=sum(N.values()),
                opponent_id=opponent_id
            )
        
        # Select action based on average strategy with a small amount of noise
        # to ensure exploration
        if any(np.isnan(average_strategy)):
            # Handle numerical issues - fall back to uniform random over valid actions
            valid_actions = np.where(action_mask)[0]
            selected_action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
        else:
            temperature = 0.1  # Controls randomness - lower = more deterministic
            if np.random.random() < 0.05:  # Small chance of pure exploration
                valid_actions = np.where(action_mask)[0]
                selected_action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
            else:
                # Add small noise to ensure exploration
                noisy_strategy = 0.95 * average_strategy + 0.05 * np.random.dirichlet(
                    np.ones(len(action_mask)) * 0.5)
                # Ensure valid by masking and renormalizing
                noisy_strategy = noisy_strategy * action_mask
                if np.sum(noisy_strategy) > 0:
                    noisy_strategy = noisy_strategy / np.sum(noisy_strategy)
                    # Sample according to the noisy strategy
                    selected_action = np.random.choice(len(action_mask), p=noisy_strategy)
                else:
                    # Fallback - shouldn't happen with proper action masks
                    valid_actions = np.where(action_mask)[0]
                    selected_action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
        
        return {
            'selected_action': selected_action,
            'search_policy': average_strategy,  # Using average strategy as the policy
            'value_estimate': value_estimate,
            'counterfactual_regrets': immediate_regrets,
            'cfr_strategy': cfr_strategy,
            'public_state_key': str(public_state_key),  # Include public state key for analysis
            'counterfactual_values': dict(cf_values),
            'blueprint_strategy': blueprint_strategy,
            'blueprint_value': blueprint_value
        }

    def _apply_progressive_pruning(self, strategy, valid_actions, depth):
        """
        Apply progressive pruning based on search depth and strategy values.
        Deeper in the tree, we focus on fewer but more promising actions.
        
        Args:
            strategy: The current probability distribution over actions.
            valid_actions: Array of indices representing valid actions.
            depth: Current remaining depth in the search.
            
        Returns:
            Pruned and renormalized strategy.
        """
        # Do not prune if only a few actions or shallow search depth
        if len(valid_actions) <= 2 or depth >= 3:
            return strategy

        # Adaptive threshold: becomes stricter as depth increases
        pruning_threshold = 0.05 * (1 + (3 - depth))
        
        # Select actions with probability above threshold
        significant_actions = [a for a in valid_actions if strategy[a] >= pruning_threshold]
        
        # Ensure at least two actions are preserved
        if len(significant_actions) < 2:
            sorted_actions = sorted(valid_actions, key=lambda a: strategy[a], reverse=True)
            significant_actions = sorted_actions[:2]
        
        # Create a mask for significant actions
        pruned_mask = np.zeros_like(strategy)
        pruned_mask[significant_actions] = 1
        
        # Apply mask and renormalize the strategy
        pruned_strategy = strategy * pruned_mask
        pruned_strategy = pruned_strategy / np.clip(np.sum(pruned_strategy), 1e-8, None)
        
        return pruned_strategy

    def manage_simulation_cache(self):
        """
        Manage the simulation cache to prevent memory issues.
        This function should be called periodically, such as after each game.
        """
        if not hasattr(self, '_simulation_cache'):
            self._simulation_cache = {}
            self._cache_hits = 0
            self._cache_misses = 0
            return
        
        # Print cache statistics
        total_lookups = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / max(1, total_lookups)) * 100
        
        print(f"Simulation cache stats:")
        print(f"  - Size: {len(self._simulation_cache)} entries")
        print(f"  - Hits: {self._cache_hits} ({hit_rate:.1f}%)")
        print(f"  - Misses: {self._cache_misses}")
        
        # Clear cache if it gets too large
        if len(self._simulation_cache) > 50000:
            print(f"  - Clearing simulation cache ({len(self._simulation_cache)} entries)")
            self._simulation_cache = {}
            self._cache_hits = 0
            self._cache_misses = 0
        
        # Optional: Keep only the most frequently accessed entries
        # This requires additional tracking of access counts
        
        return hit_rate

    def _simulate(self, env, action, observation, public_obs, beliefs, public_beliefs, 
              depth, reach_prob=1.0, parent_values=None):
        """
        Simulate taking an action and recursively evaluate the resulting state.
        This version adds neural network batching for depth=0 evaluations.
        """
        agent = self.name

        # Ensure belief tensors are on the correct device
        beliefs = beliefs.to(self.device)
        public_beliefs = public_beliefs.to(self.device)

        # Create a key for transposition table lookup
        state_key = self.create_node_key(np.array(public_obs), public_beliefs)
        
        # Create a cache key that includes depth and action for more specific caching
        cache_key = (state_key, depth, action)
        
        # Check simulation cache first (optimization #1)
        if not hasattr(self, '_simulation_cache'):
            self._simulation_cache = {}
            self._cache_hits = 0
            self._cache_misses = 0
        
        if cache_key in self._simulation_cache:
            self._cache_hits += 1
            return self._simulation_cache[cache_key]
        self._cache_misses += 1
        
        # Check for early termination based on value statistics
        if depth > 0:
            should_terminate, cached_value = self.check_early_termination(state_key, depth, reach_prob)
            if should_terminate:
                self._simulation_cache[cache_key] = (cached_value, {})
                return cached_value, {}
        
        # If already computed for a deeper node, use the transposition table
        if depth > 1 and state_key in self.transposition_table:
            cached_value = self.transposition_table[state_key]
            self._simulation_cache[cache_key] = (cached_value, {})
            return cached_value, {}
        
        # Optionally use sampled beliefs for deeper nodes for efficiency
        if depth > 1 and hasattr(self.belief_model, 'sample_consistent_beliefs'):
            private_hand = torch.FloatTensor(observation[:2]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _ = self.belief_model.sample_consistent_beliefs(beliefs, private_hand, num_samples=1)
        
        # Execute the chosen action in simulation
        env.step(action)
        reward = env.rewards[agent]
        done = env.terminations[agent]

        # Initialize counterfactual values dictionary
        cf_values = {}

        # Terminal state handling: if the simulation reaches a terminal state, return reward immediately.
        if done:
            self._simulation_cache[cache_key] = (reward, cf_values)
            return reward, cf_values

        # Depth limit reached – use the value network to estimate state value.
        if depth == 0:
            next_obs = env.observe(agent)
            if isinstance(next_obs, dict):
                next_obs = next_obs[agent]
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            action_mask = env.infos[agent]["action_mask"]
            
            # Neural Network Batching Optimization
            # Add to batch buffer instead of immediate evaluation
            if not hasattr(self, '_nn_batch'):
                self._nn_batch = {
                    'observations': [],
                    'beliefs': [],
                    'private_hands': [],
                    'action_masks': [],
                    'batch_ids': []  # To track which batch entry belongs to which call
                }
                self._current_batch_id = 0
                self._nn_results = {}  # Store results indexed by batch_id
                self._nn_batch_size = 8  # Configurable batch size
                
            # Generate a unique ID for this batch item
            batch_id = self._current_batch_id
            self._current_batch_id += 1
            
            # Add to batch
            self._nn_batch['observations'].append(next_obs_tensor)
            self._nn_batch['beliefs'].append(beliefs)
            self._nn_batch['private_hands'].append(torch.FloatTensor(next_obs[:2]).unsqueeze(0).to(self.device))
            self._nn_batch['action_masks'].append(action_mask)
            self._nn_batch['batch_ids'].append(batch_id)
            
            # Process batch if it reaches the threshold or if this is the root call
            if len(self._nn_batch['observations']) >= self._nn_batch_size or depth >= self.search_depth - 1:
                with torch.no_grad():
                    # Process beliefs in batch
                    batch_obs = torch.cat(self._nn_batch['observations'], dim=0)
                    batch_beliefs = torch.cat(self._nn_batch['beliefs'], dim=0)
                    batch_private_hands = torch.cat(self._nn_batch['private_hands'], dim=0)
                    
                    # Update beliefs in batch
                    next_beliefs_batch = self.belief_model(batch_obs, batch_beliefs)
                    
                    # Apply physical constraints in batch
                    next_beliefs_batch = self.belief_model.apply_physical_constraints_fast(
                        next_beliefs_batch, batch_private_hands)
                    
                    # Evaluate value network in batch
                    values_batch, regrets_batch, _ = self.value_net(batch_obs, next_beliefs_batch)
                    
                    # Store results for each batch item
                    for i, bid in enumerate(self._nn_batch['batch_ids']):
                        value = values_batch[i].item()
                        regrets = regrets_batch[i].cpu().numpy() * self._nn_batch['action_masks'][i]
                        
                        # Create counterfactual values
                        valid_actions = np.where(self._nn_batch['action_masks'][i])[0]
                        cf_values_dict = {}
                        for a in valid_actions:
                            cf_values_dict[a] = value + regrets[a]
                        
                        # Store result
                        self._nn_results[bid] = (value, cf_values_dict)
                    
                    # Clear batch
                    self._nn_batch = {
                        'observations': [],
                        'beliefs': [],
                        'private_hands': [],
                        'action_masks': [],
                        'batch_ids': []
                    }
            
            # Check if result is available
            if batch_id in self._nn_results:
                avg_value, cf_values = self._nn_results[batch_id]
                # Remove from results to free memory
                del self._nn_results[batch_id]
                
                self._simulation_cache[cache_key] = (avg_value, cf_values)
                return avg_value, cf_values
            else:
                # If result not available (uncommon), compute directly
                with torch.no_grad():
                    next_beliefs = self.belief_model(next_obs_tensor, beliefs)
                    private_hand = torch.FloatTensor(next_obs[:2]).unsqueeze(0).to(self.device)
                    next_beliefs = self.belief_model.apply_physical_constraints_fast(next_beliefs, private_hand)
                    value, regrets, _ = self.value_net(next_obs_tensor, next_beliefs)
                    avg_value = value.item()
                    regrets_np = regrets.squeeze(0).cpu().numpy() * action_mask
                    valid_actions = np.where(action_mask)[0]
                    cf_values = {}
                    for a in valid_actions:
                        cf_values[a] = avg_value + regrets_np[a]
                        
                self._simulation_cache[cache_key] = (avg_value, cf_values)
                return avg_value, cf_values

        # Continue recursive simulation normally.
        next_obs = env.observe(agent)
        if isinstance(next_obs, dict):
            next_obs = next_obs[agent]
        action_mask = env.infos[agent]["action_mask"]
        next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_public_obs, _ = self.split_observation(next_obs)
            if depth > 2:
                next_beliefs = self.belief_model(next_obs_tensor, beliefs)
                next_public_beliefs = self.belief_model.get_public_belief_state(next_obs_tensor, public_beliefs)
                private_hand = torch.FloatTensor(next_obs[:2]).unsqueeze(0).to(self.device)
                next_beliefs = self.belief_model.apply_physical_constraints_fast(next_beliefs, private_hand)
                next_public_beliefs = self.belief_model.apply_physical_constraints_fast(next_public_beliefs, private_hand)
            else:
                next_beliefs = beliefs.clone()
                next_public_beliefs = public_beliefs.clone()

            next_public_state_key = self.create_node_key(np.array(next_public_obs), next_public_beliefs)
            cfr_strategy = self.compute_cfr_strategy(next_public_state_key, action_mask)
            masked_strategy = cfr_strategy * action_mask
            if np.sum(masked_strategy) > 0:
                masked_strategy = masked_strategy / np.sum(masked_strategy)
            else:
                valid_actions = np.where(action_mask)[0]
                masked_strategy = np.zeros_like(action_mask, dtype=np.float32)
                if len(valid_actions) > 0:
                    masked_strategy[valid_actions] = 1.0 / len(valid_actions)
            valid_actions = np.where(action_mask)[0]
            masked_strategy = self._apply_progressive_pruning(masked_strategy, valid_actions, depth)

            cf_values = {}
            total_value = 0

            # Optimization #2: Simplified deep node evaluations
            # For shallow searches, evaluate all valid actions; for deeper searches, only evaluate top 3
            if depth <= 1 or len(valid_actions) <= 3:
                for a in valid_actions:
                    if masked_strategy[a] <= 0:
                        continue
                    action_reach = reach_prob * masked_strategy[a]
                    action_env = env.clone()
                    action_value, child_cf_values = self._simulate(
                        action_env, a, next_obs, next_public_obs,
                        next_beliefs, next_public_beliefs, depth - 1,
                        action_reach, parent_values=cf_values
                    )
                    cf_values[a] = action_value
                    total_value += masked_strategy[a] * action_value
            else:
                top_actions = sorted(
                    [a for a in valid_actions if masked_strategy[a] > 0],
                    key=lambda a: masked_strategy[a],
                    reverse=True
                )[:3]
                
                if not top_actions and valid_actions.size > 0:
                    top_actions = valid_actions[np.argsort(masked_strategy[valid_actions])[-3:]]
                
                for a in top_actions:
                    action_reach = reach_prob * masked_strategy[a]
                    action_env = env.clone()
                    action_value, child_cf_values = self._simulate(
                        action_env, a, next_obs, next_public_obs,
                        next_beliefs, next_public_beliefs, depth - 1,
                        action_reach, parent_values=cf_values
                    )
                    cf_values[a] = action_value
                    total_value += masked_strategy[a] * action_value

            result = (env.rewards[agent] + total_value, cf_values)
            self._simulation_cache[cache_key] = result
            
            if len(self._simulation_cache) > 1000000:
                self._simulation_cache = {}

        return env.rewards[agent] + total_value, cf_values


    def play_turn(self, observation, action_mask, table_card):
        """
        Interface method compatible with game environment.
        Now with improved search tree reuse and opponent memory updates.
        
        Args:
            observation: Current observation.
            action_mask: Mask of valid actions.
            table_card: Current table card.
            
        Returns:
            A dictionary containing search results.
        """
        # Check if we can reuse search tree from previous turn
        can_reuse_tree = False
        if hasattr(self, 'last_observation') and self.last_observation is not None:
            # Extract public observations from current and last observation
            last_public_obs, _ = self.split_observation(self.last_observation)
            current_public_obs, _ = self.split_observation(observation)
            
            # Check if only private information changed
            if np.array_equal(last_public_obs, current_public_obs):
                can_reuse_tree = True
                self.tree_reuse_count += 1
            else:
                # Clear transposition tables
                self.transposition_table = {}
                self.value_statistics = {}
                self.tree_rebuild_count += 1
        else:
            # Initialize from scratch
            self.transposition_table = {}
            self.value_statistics = {}
            self.tree_rebuild_count += 1
        
        # Store current observation for next turn
        self.last_observation = observation.copy() if isinstance(observation, np.ndarray) else observation
        
        # Update beliefs based on the latest observation
        self.update_beliefs(observation, action_mask)
        
        # Run MCTS search with CFR to obtain search outputs
        search_outcomes = self.mcts_search(observation, action_mask)
        
        # Split observation for logging
        public_obs, private_obs = self.split_observation(observation)
        
        # Record complete transition for later training
        self.action_history.append({
            'observation': observation,
            'public_observation': public_obs,
            'private_observation': private_obs,
            'action_mask': action_mask,
            'table_card': table_card,
            'selected_action': search_outcomes['selected_action'],
            'search_policy': search_outcomes['search_policy'],
            'value_estimate': search_outcomes['value_estimate'],
            'counterfactual_regrets': search_outcomes['counterfactual_regrets'],
            'cfr_strategy': search_outcomes['cfr_strategy'],
            'public_state_key': search_outcomes['public_state_key'],
            'tree_reuse': can_reuse_tree
        })
        
        # Update opponent memory
        current_env = self.env_creator()
        
        # Get information about the most recent action by opponents
        last_action_agent = current_env.last_action_agent
        last_action = current_env.last_action
        last_action_bluff = current_env.last_action_bluff
        
        if last_action_agent and last_action_agent != self.name:
            # Record the opponent's action in our memory
            action_type = f"Play_{last_action}" if last_action is not None else "None"
            card_count = len(current_env.players_hands.get(last_action_agent, []))
            penalty_count = current_env.penalties.get(last_action_agent, 0)
            
        
        # If we're about to make a challenge, record that information
        selected_action = search_outcomes['selected_action']
        action_type, _, count = decode_action(selected_action)
        
        # Add search efficiency statistics to the search results
        search_outcomes['tree_stats'] = {
            'reuse_count': self.tree_reuse_count,
            'rebuild_count': self.tree_rebuild_count,
            'transposition_hits': len(self.transposition_table)
        }
        
        return search_outcomes
