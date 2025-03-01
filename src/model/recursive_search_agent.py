# src/model/recursive_search_agent.py
import torch
import numpy as np
from collections import defaultdict

from src.env.liars_deck_env_utils_2 import decode_action

class RecursiveSearchAgent:
    def __init__(self, policy_net, belief_model, value_net, env_creator, 
             device, search_depth=4, num_simulations=30, c_puct=1.0,
             agent_name=None, agent_index=None, blueprint=None,
             alpha=1.5, beta=0.5, gamma=2.0):
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
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.transposition_table = {}
        self.value_statistics = {}
        self.last_observation = None
        # Belief state tracking
        self.current_beliefs = None
        self.current_public_beliefs = None
        self.action_history = []
        self.search_statistics = {}
        self.iterations = defaultdict(int)
        # Key additions for CFR
        self.cumulative_regrets = defaultdict(lambda: np.zeros(7))  # For 7 possible actions
        self.average_strategy = defaultdict(lambda: np.zeros(7))
        self.strategy_update_count = defaultdict(int)
        
        # State encoding function (for dictionary keys)
        self.public_state_to_key = lambda public_obs, public_beliefs: hash(str(public_obs.tolist()) + str(public_beliefs.detach().cpu().numpy().tolist()))
        
        # Add opponent memory support
        from src.model.rebel_memory import get_opponent_memory
        self.opponent_memory = get_opponent_memory(agent_name)

    def reset(self):
        """Reset agent state at the beginning of a new game."""
        self.current_beliefs = None
        self.current_public_beliefs = None
        self.action_history = []
        self.search_statistics = {}
        # Note: We don't reset cumulative_regrets or average_strategy as they persist across games

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
        now with counterfactual reasoning.
        """
        # Extract observation
        if isinstance(observation, dict):
            obs_data = observation[self.name]
        else:
            obs_data = observation
        
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
            
            is_full_update = self._belief_update_counter % 2 == 0 or not self.policy_net.training
            
            # Create environment instance for counterfactual reasoning
            current_env = self.env_creator()
            
            if is_full_update:
                # Reconstruct game history for counterfactual reasoning
                if hasattr(self, '_game_history'):
                    game_history = self._game_history
                else:
                    game_history = self._reconstruct_game_history(current_env)
                    self._game_history = game_history
                
                # Use counterfactual belief inference with game history
                if self.current_beliefs is None:
                    # For the first update, use the model directly
                    self.current_beliefs = self.belief_model.infer_belief_from_game_state(
                        obs_tensor, self.agent_index, current_env)
                else:
                    # For subsequent updates, use the Bayesian update
                    # with counterfactual reasoning
                    self.current_beliefs = self._compute_counterfactual_beliefs(
                        obs_tensor, self.current_beliefs, game_history, current_env)
                
                # Apply physical constraints
                self.current_beliefs = self.belief_model.apply_physical_constraints_fast(
                    self.current_beliefs, private_hand)
            
            # Also update public beliefs
            if self.current_public_beliefs is None:
                self.current_public_beliefs = self.belief_model.get_public_belief_state(obs_tensor)
            else:
                self.current_public_beliefs = self.belief_model.get_public_belief_state(
                    obs_tensor, self.current_public_beliefs)
            
            # Apply physical constraints to public beliefs as well
            self.current_public_beliefs = self.belief_model.apply_physical_constraints_fast(
                self.current_public_beliefs, private_hand)
        
        # Infer and record opponent actions
        current_env = self.env_creator()
        last_action_agent = current_env.last_action_agent
        last_action = current_env.last_action
        last_action_bluff = current_env.last_action_bluff
        
        if last_action_agent and last_action_agent != self.name:
            action_type = f"Play_{last_action}" if last_action is not None else "None"
            card_count = len(current_env.players_hands.get(last_action_agent, []))
            penalty_count = current_env.penalties.get(last_action_agent, 0)
            
            self.opponent_memory.update(
                opponent=last_action_agent,
                response=action_type,
                penalties=penalty_count,
                card_count=card_count
            )
            
            if last_action_bluff is not None and last_action is not None:
                self.opponent_memory.record_bluff(
                    opponent=last_action_agent,
                    was_bluff=last_action_bluff,
                    play_count=last_action
                )

    def compute_cfr_strategy(self, state_key, action_mask):
        """
        Compute a strategy according to the DCFR algorithm using discounted regrets.

        Args:
            state_key: A unique identifier for the current state
            action_mask: Boolean mask of valid actions
            
        Returns:
            numpy array: A probability distribution over actions
        """
        # Get cumulative regrets for this state
        regrets = self.cumulative_regrets[state_key]

        # Get iteration count for this state (increment first)
        self.iterations[state_key] += 1
        t = self.iterations[state_key]

        # Apply DCFR discounting
        positive_regrets = np.maximum(regrets, 0)
        negative_regrets = np.minimum(regrets, 0)

        # Discount positive and negative regrets differently
        discounted_positive = positive_regrets * (t ** self.alpha)
        discounted_negative = negative_regrets * (t ** self.beta)

        # Combine discounted regrets
        discounted_regrets = discounted_positive + discounted_negative

        # Apply action mask
        masked_regrets = discounted_regrets * action_mask
        regret_sum = masked_regrets.sum()

        # If sum is zero or state not seen before, use uniform random over valid actions
        if regret_sum <= 0:
            valid_actions = np.where(action_mask)[0]
            strategy = np.zeros_like(action_mask, dtype=np.float32)
            strategy[valid_actions] = 1.0 / len(valid_actions)
        else:
            # Normalize by sum of positive regrets (regret matching)
            strategy = masked_regrets / regret_sum

        return strategy

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

    def _compute_counterfactual_beliefs(self, obs_tensor, current_beliefs, game_history, env):
        """
        Compute counterfactual beliefs based on game history.
        
        Args:
            obs_tensor: Current observation tensor
            current_beliefs: Current belief state
            game_history: Reconstructed game history
            env: Current game environment
            
        Returns:
            Updated belief state using counterfactual reasoning
        """
        # This would call the belief model's counterfactual inference method
        return self.belief_model.infer_belief_from_game_state(
            obs_tensor, self.agent_index, env)

    def mcts_search(self, observation, action_mask):
        """
        Perform Monte Carlo Tree Search with blueprint priors, CFR, opponent memory, and virtual loss.
        
        Args:
            observation: Current observation.
            action_mask: Mask of valid actions.
            
        Returns:
            Dictionary with search outcomes:
                - selected_action: the chosen action (int)
                - search_policy: distribution over actions (np.array)
                - value_estimate: value from subgame solver (float)
                - counterfactual_regrets: vector of per-action regrets (np.array)
                - cfr_strategy: strategy based on regret matching (np.array)
                - public_state_key: string version of the public state key (for analysis)
                - counterfactual_values: per-action counterfactual values (dict)
                - blueprint_strategy: blueprint prior strategy if available (np.array or None)
                - blueprint_value: blueprint value if available (float or None)
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
        public_state_key = self.public_state_to_key(np.array(public_obs), self.current_public_beliefs)
        
        # Get policy network's action priors
        with torch.no_grad():
            # Use the full policy (with private info) for priors
            priors, _, _ = self.policy_net(obs_tensor)
            priors = priors.squeeze(0).cpu().numpy()
            
            # Also get public-only policy for comparison
            public_priors, _, _ = self.policy_net.public_policy(public_obs_tensor, self.current_public_beliefs)
            public_priors = public_priors.squeeze(0).cpu().numpy()
        
        # If we have a blueprint, query it with opponent identity
        if self.blueprint:
            # Query blueprint for prior strategy and value, including opponent_id
            blueprint_strategy, blueprint_value = self.blueprint.query(
                np.array(public_obs), 
                self.current_public_beliefs.cpu().numpy(), 
                action_mask,
                opponent_id  # Include opponent identity
            )
            # Use blueprint as prior (higher weight) combined with policy network
            blueprint_weight = 0.3
            combined_priors = blueprint_weight * blueprint_strategy + (1 - blueprint_weight) * priors
        else:
            # Use policy network normally if no blueprint
            combined_priors = priors
            blueprint_strategy = None
            blueprint_value = None
        
        # Apply action mask to priors
        masked_priors = combined_priors * action_mask
        if np.sum(masked_priors) > 0:
            masked_priors = masked_priors / np.sum(masked_priors)
        else:
            valid_actions = np.where(action_mask)[0]
            masked_priors = np.zeros_like(combined_priors)
            masked_priors[valid_actions] = 1.0 / len(valid_actions)
        
        # Initialize search statistics: visit count (N), total value (W), and mean value (Q)
        N = {a: 0 for a in range(len(action_mask))}
        W = {a: 0.0 for a in range(len(action_mask))}
        Q = {a: 0.0 for a in range(len(action_mask))}
        
        # Track counterfactual values across all simulations
        cf_values = {}
        
        # Add virtual loss tracking to discourage parallel exploration of the same path
        virtual_loss = {a: 0.0 for a in range(len(action_mask))}
        
        # Run MCTS simulations
        for _ in range(self.num_simulations):
            sim_env = self.env_creator()  # Clone environment for simulation
            
            # Compute current CFR strategy using PUBLIC state key
            cfr_strategy = self.compute_cfr_strategy(public_state_key, action_mask)
            
            # Select action using adaptive exploration in the PUCT formula
            valid_actions = np.where(action_mask)[0]
            best_score = -float('inf')
            best_action = valid_actions[0]
            
            for action in valid_actions:
                visit_count = N[action]
                total_visits = sum(N.values())
                
                # Adaptive exploration factor that decreases with depth
                adaptive_c = self.c_puct * (1.0 - 0.1 * len(self.action_history))
                adaptive_c = max(0.5, adaptive_c)  # Ensure minimum exploration
                
                # Progressive widening coefficient
                pw_coeff = 0.5
                if visit_count > (total_visits ** pw_coeff):
                    exploitation = Q[action]
                    exploration = adaptive_c * cfr_strategy[action] * np.sqrt(np.log(total_visits) / (visit_count + 1e-5))
                else:
                    exploitation = Q[action]
                    exploration = adaptive_c * cfr_strategy[action] * np.sqrt(np.log(total_visits + 1) / (visit_count + 1e-5))
                
                # Apply virtual loss penalty in the selection by adjusting the Q value temporarily
                temp_Q = Q[action] - (virtual_loss[action] * 0.1 / max(N[action], 1))
                score = temp_Q + exploration
                
                if score > best_score:
                    best_score = score
                    best_action = action
            
            # Apply virtual loss for the selected action before simulation
            virtual_loss[best_action] += 1.0
            
            # Run simulation from the selected node (with subgame solving)
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
                if a not in cf_values:
                    cf_values[a] = 0
                cf_values[a] += value / self.num_simulations
        
        # Compute overall value estimate as weighted average of Q-values
        value_estimate = sum(N[a] * Q[a] for a in range(len(action_mask))) / max(sum(N.values()), 1)
        
        # Compute counterfactual regrets based on final counterfactual values
        immediate_regrets = np.zeros(len(action_mask), dtype=np.float32)
        for a in range(len(action_mask)):
            if a in cf_values:
                immediate_regrets[a] = cf_values[a] - value_estimate
        
        # Update cumulative regrets (core CFR update step)
        self.cumulative_regrets[public_state_key] += immediate_regrets * action_mask
        
        # Compute current strategy using regret matching
        cfr_strategy = self.compute_cfr_strategy(public_state_key, action_mask)
        
        # Update average strategy
        self.update_average_strategy(public_state_key, cfr_strategy)
        
        # Get the average strategy (this is what we use for actual play)
        average_strategy = self.get_average_strategy(public_state_key, action_mask)
        
        # Store search statistics
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
        
        # Before returning, update the blueprint if we have one, including opponent identity
        if self.blueprint and hasattr(self, 'average_strategy'):
            self.blueprint.update_from_search(
                np.array(public_obs),
                self.current_public_beliefs.cpu().numpy(),
                average_strategy,  # CFR average strategy
                value_estimate,
                immediate_regrets,
                visits=sum(N.values()),
                opponent_id=opponent_id  # Include opponent identity
            )
        
        # For actual play, sample from the average strategy
        selected_action = np.random.choice(len(action_mask), p=average_strategy)
        
        return {
            'selected_action': selected_action,
            'search_policy': average_strategy,  # Using average strategy as the policy
            'value_estimate': value_estimate,
            'counterfactual_regrets': immediate_regrets,
            'cfr_strategy': cfr_strategy,
            'public_state_key': str(public_state_key),  # Include public state key for analysis
            'counterfactual_values': cf_values,
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

    def _simulate(self, env, action, observation, public_obs, beliefs, public_beliefs, depth, reach_prob=1.0, parent_values=None):
        """
        Simulate taking an action and recursively evaluate the resulting state with proper subgame solving.
        Optimized for performance while maintaining correlation-aware belief handling.

        Args:
            env: Cloned environment for simulation.
            action: Action to simulate.
            observation: Full observation.
            public_obs: Public part of the observation.
            beliefs: Full belief state.
            public_beliefs: Public belief state.
            depth: Remaining search depth.
            reach_prob: Current reach probability for this state.
            parent_values: Value estimates from parent subgame for boundary conditions.

        Returns:
            Tuple of (value, counterfactual_values) where:
                - value: Estimated value after taking the action (float).
                - counterfactual_values: Dict mapping actions to their counterfactual values.
        """
        agent = self.name
        original_agent_selection = env.agent_selection

        # Ensure belief tensors are on the correct device
        beliefs = beliefs.to(self.device)
        public_beliefs = public_beliefs.to(self.device)

        # Variance-based early termination (principled statistical confidence)
        if depth > 0 and hasattr(self, 'value_statistics'):
            state_key = self.public_state_to_key(np.array(public_obs), public_beliefs)
            if state_key in self.value_statistics:
                mean_value, std_value, count = self.value_statistics[state_key]
                std_error = std_value / np.sqrt(count)
                confidence_threshold = 0.05  # 95% confidence interval threshold
                if std_error < confidence_threshold and count >= 10:
                    return env.rewards[agent] + mean_value, {}

        # Sampling: perform expensive sampling only if depth > 1.
        if depth > 1:
            private_hand = torch.FloatTensor(observation[:2]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                if hasattr(self.belief_model, 'sample_consistent_beliefs'):
                    sampled_hands = self.belief_model.sample_consistent_beliefs(beliefs, private_hand, num_samples=1).squeeze(1)
                else:
                    sampled_hands = None
        else:
            sampled_hands = None

        # Execute the action in simulation
        env.step(action)
        reward = env.rewards[agent]
        done = env.terminations[agent]

        # Initialize counterfactual values dictionary
        cf_values = {}

        # Terminal state handling: if the game is done, return immediately.
        if done:
            return reward, cf_values

        # Depth limit reached - use value network to estimate the value and regrets.
        if depth == 0:
            next_obs = env.observe(agent)
            if isinstance(next_obs, dict):
                next_obs = next_obs[agent]
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            action_mask = env.infos[agent]["action_mask"]

            with torch.no_grad():
                next_beliefs = self.belief_model(next_obs_tensor, beliefs)
                private_hand = torch.FloatTensor(next_obs[:2]).unsqueeze(0).to(self.device)
                next_beliefs = self.belief_model.apply_physical_constraints_fast(next_beliefs, private_hand)
                value, regrets = self.value_net(next_obs_tensor, next_beliefs)
                avg_value = value.item()
                regrets_np = regrets.squeeze(0).cpu().numpy() * action_mask

                valid_actions = np.where(action_mask)[0]
                for a in valid_actions:
                    cf_values[a] = avg_value + regrets_np[a]
            return avg_value, cf_values

        # Nested Subgame Solving: if the active agent changed or the round ended.
        if env.agent_selection is None or env.agent_selection != original_agent_selection:
            next_obs = env.observe(agent)
            if isinstance(next_obs, dict):
                next_obs = next_obs[agent]
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)

            # Create a key for this subgame from the public observation and updated public belief state.
            next_public_obs, _ = self.split_observation(next_obs)
            updated_public_beliefs = self.belief_model.get_public_belief_state(next_obs_tensor)
            subgame_key = self.public_state_to_key(np.array(next_public_obs), updated_public_beliefs)

            # Check transposition table for a previously computed subgame value.
            if hasattr(self, 'transposition_table') and subgame_key in self.transposition_table:
                cached_value = self.transposition_table[subgame_key]
                return env.rewards[agent] + cached_value, {}

            with torch.no_grad():
                next_beliefs = self.belief_model(next_obs_tensor, beliefs)
                private_hand = torch.FloatTensor(next_obs[:2]).unsqueeze(0).to(self.device)
                next_beliefs = self.belief_model.apply_physical_constraints_fast(next_beliefs, private_hand)
                value, _ = self.value_net(next_obs_tensor, next_beliefs)
                subgame_value = value.item()

                # Safety constraint: use parent's estimate if subgame value is lower.
                if parent_values is not None and action in parent_values:
                    parent_value = parent_values[action]
                    if subgame_value < parent_value:
                        correction = parent_value - subgame_value
                        subgame_value = parent_value
                        if not hasattr(self, 'subgame_corrections'):
                            self.subgame_corrections = {}
                        self.subgame_corrections[subgame_key] = correction

                if not hasattr(self, 'transposition_table'):
                    self.transposition_table = {}
                self.transposition_table[subgame_key] = subgame_value

                # Update value statistics for potential future early terminations.
                self._update_value_statistics(subgame_key, subgame_value)

                return env.rewards[agent] + subgame_value, {}

        # Continue recursive simulation within the current subgame.
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

            next_public_state_key = self.public_state_to_key(np.array(next_public_obs), next_public_beliefs)
            cfr_strategy = self.compute_cfr_strategy(next_public_state_key, action_mask)

            # Initial masked strategy based on CFR strategy and action mask.
            masked_strategy = cfr_strategy * action_mask
            if np.sum(masked_strategy) > 0:
                masked_strategy = masked_strategy / np.sum(masked_strategy)
            else:
                valid_actions = np.where(action_mask)[0]
                masked_strategy = np.zeros_like(action_mask, dtype=np.float32)
                masked_strategy[valid_actions] = 1.0 / len(valid_actions)

            valid_actions = np.where(action_mask)[0]
            # Apply progressive pruning to focus on the most promising actions at deeper depths.
            masked_strategy = self._apply_progressive_pruning(masked_strategy, valid_actions, depth)

            action_cf_values = {}
            total_value = 0
            for a in valid_actions:
                if masked_strategy[a] <= 0:
                    continue
                action_reach = reach_prob * masked_strategy[a]
                if action_reach < 0.005:
                    continue
                action_env = env.clone()
                action_value, child_cf_values = self._simulate(
                    action_env, a, next_obs, next_public_obs,
                    next_beliefs, next_public_beliefs, depth - 1,
                    action_reach, parent_values=action_cf_values
                )
                # Update counterfactual values with results from the child simulation.
                cf_values[a] = action_value
                total_value += masked_strategy[a] * action_value

        return env.rewards[agent] + total_value, cf_values


    def play_turn(self, observation, action_mask, table_card):
        """
        Interface method compatible with the game environment.
        Now with improved search tree reuse between turns and
        enhanced opponent memory updates.
        
        Args:
            observation: Current observation.
            action_mask: Mask of valid actions.
            table_card: Current table card.
            
        Returns:
            A dictionary containing:
            - selected_action: Chosen action.
            - search_policy: Distribution over actions based on CFR average strategy.
            - value_estimate: Value estimate from the subgame solver.
            - counterfactual_regrets: Computed regrets for available actions.
            - cfr_strategy: Current CFR strategy based on regret matching.
        """
        # Check if we can reuse search tree from previous turn
        can_reuse_tree = False
        if hasattr(self, 'last_observation') and self.last_observation is not None and hasattr(self, 'transposition_table'):
            # Extract public observations from current and last observation
            last_public_obs = self.split_observation(self.last_observation)[0]
            current_public_obs = self.split_observation(observation)[0]
            
            # If only private information changed, we can reuse the tree
            if np.array_equal(last_public_obs[2:], current_public_obs[2:]):
                can_reuse_tree = True
                # Note: We keep the transposition_table and value_statistics
                # but will update beliefs with new private information
        
        if not can_reuse_tree:
            # If we can't reuse, clear the tables
            self.transposition_table = {}
            self.value_statistics = {}
        
        # Store current observation for next turn
        self.last_observation = observation
        
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
            'public_state_key': search_outcomes['public_state_key']
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
            
            self.opponent_memory.update(
                opponent=last_action_agent,
                response=action_type,
                penalties=penalty_count,
                card_count=card_count
            )
            
            # If we know whether it was a bluff, record that too
            if last_action_bluff is not None and last_action is not None:
                self.opponent_memory.record_bluff(
                    opponent=last_action_agent,
                    was_bluff=last_action_bluff,
                    play_count=last_action
                )
        
        # If we're about to make a challenge, record that information
        selected_action = search_outcomes['selected_action']
        action_type, _, count = decode_action(selected_action)
        
        if action_type == "Challenge" and last_action_agent:
            # We're challenging someone - update our memory with this decision
            self.opponent_memory.record_challenge_result(
                opponent=self.name,  # We're the challenger
                success=None,  # We don't know the outcome yet
                target=last_action_agent  # Who we're challenging
            )
        
        return search_outcomes
