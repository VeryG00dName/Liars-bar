# src/model/recursive_search_agent.py
import torch
import numpy as np
from collections import defaultdict

class RecursiveSearchAgent:
    def __init__(self, policy_net, belief_model, value_net, env_creator, 
                 device, search_depth=4, num_simulations=30, c_puct=1.0,
                 agent_name=None, agent_index=None, blueprint=None):
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
        # Belief state tracking
        self.current_beliefs = None
        self.current_public_beliefs = None
        self.action_history = []
        self.search_statistics = {}
        
        # Key additions for CFR
        self.cumulative_regrets = defaultdict(lambda: np.zeros(7))  # For 7 possible actions
        self.average_strategy = defaultdict(lambda: np.zeros(7))
        self.strategy_update_count = defaultdict(int)
        
        # State encoding function (for dictionary keys)
        self.public_state_to_key = lambda public_obs, public_beliefs: hash(str(public_obs.tolist()) + str(public_beliefs.cpu().numpy().tolist()))
    
    def reset(self):
        """Reset agent state at the beginning of a new game."""
        self.current_beliefs = None
        self.current_public_beliefs = None
        self.action_history = []
        self.search_statistics = {}
        # Note: We don't reset cumulative_regrets or average_strategy as they persist across games
    
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
        Optimized belief update with caching and faster processing.
        """
        if isinstance(observation, dict):
            obs_data = observation[self.name]
        else:
            obs_data = observation

        # Quick conversion to tensor
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

        private_hand = obs_tensor[:, :2]

        with torch.no_grad():
            if not hasattr(self, '_belief_update_counter'):
                self._belief_update_counter = 0
            self._belief_update_counter += 1

            is_full_update = self._belief_update_counter % 2 == 0 or not self.policy_net.training

            if is_full_update:
                if self.current_beliefs is None:
                    self.current_beliefs = self.belief_model(obs_tensor)
                else:
                    self.current_beliefs = self.belief_model(obs_tensor, self.current_beliefs)

                self.current_beliefs = self.belief_model.apply_physical_constraints_fast(
                    self.current_beliefs, private_hand)

            if self.current_public_beliefs is None:
                self.current_public_beliefs = self.belief_model.get_public_belief_state(obs_tensor)
            else:
                self.current_public_beliefs = self.belief_model.get_public_belief_state(
                    obs_tensor, self.current_public_beliefs)

            self.current_public_beliefs = self.belief_model.apply_physical_constraints_fast(
                self.current_public_beliefs, private_hand)
        
        # Extract private hand information for physical constraints
        # First two elements are the player's hand information (table cards, non-table cards)
        private_hand = obs_tensor[:, :2]
        
        with torch.no_grad():
            # Update full belief state (using both public and private info)
            if self.current_beliefs is None:
                self.current_beliefs = self.belief_model(obs_tensor)
            else:
                self.current_beliefs = self.belief_model(obs_tensor, self.current_beliefs)
            
            # Also update the public belief state (only uses public info)
            if self.current_public_beliefs is None:
                self.current_public_beliefs = self.belief_model.get_public_belief_state(obs_tensor)
            else:
                self.current_public_beliefs = self.belief_model.get_public_belief_state(
                    obs_tensor, self.current_public_beliefs)
            
            # Apply correlation and physical constraints
            self.current_beliefs = self.belief_model.model_correlations(
                self.current_beliefs, private_hand)
            self.current_public_beliefs = self.belief_model.model_correlations(
                self.current_public_beliefs, private_hand)
    
    def compute_cfr_strategy(self, state_key, action_mask):
        """
        Compute a strategy according to the CFR algorithm using cumulative regrets.
        
        Args:
            state_key: A unique identifier for the current state
            action_mask: Boolean mask of valid actions
            
        Returns:
            numpy array: A probability distribution over actions
        """
        # Get cumulative regrets for this state
        regrets = self.cumulative_regrets[state_key]
        
        # Only consider positive regrets (regret matching)
        positive_regrets = np.maximum(regrets, 0) * action_mask
        regret_sum = positive_regrets.sum()
        
        # If sum is zero or state not seen before, use uniform random over valid actions
        if regret_sum <= 0:
            valid_actions = np.where(action_mask)[0]
            strategy = np.zeros_like(action_mask, dtype=np.float32)
            strategy[valid_actions] = 1.0 / len(valid_actions)
        else:
            # Normalize by sum of positive regrets (regret matching)
            strategy = positive_regrets / regret_sum
        
        return strategy
    
    def update_average_strategy(self, state_key, current_strategy):
        """
        Update the average strategy for a state.
        
        Args:
            state_key: A unique identifier for the current state
            current_strategy: The current strategy for this state
        """
        # Increment counter for this state
        self.strategy_update_count[state_key] += 1
        
        # Update running average: weighted by count to properly compute average
        count = self.strategy_update_count[state_key]
        self.average_strategy[state_key] = (
            (count - 1) / count * self.average_strategy[state_key] + 
            (1 / count) * current_strategy
        )
    
    def get_average_strategy(self, state_key, action_mask):
        """
        Get the average strategy for a state.
        
        Args:
            state_key: A unique identifier for the current state
            action_mask: Boolean mask of valid actions
            
        Returns:
            numpy array: A probability distribution over actions
        """
        strategy = self.average_strategy[state_key]
        
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
    
    def mcts_search(self, observation, action_mask):
        """
        Perform Monte Carlo Tree Search with blueprint priors and CFR.
        
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
        
        # Create a unique key for this PUBLIC state (key for CFR)
        public_state_key = self.public_state_to_key(
            np.array(public_obs), self.current_public_beliefs)
        
        # Get policy priors - now use blueprint if available
        if self.blueprint:
            # Query blueprint for prior strategy and value
            blueprint_strategy, blueprint_value = self.blueprint.query(
                np.array(public_obs), 
                self.current_public_beliefs.cpu().numpy(), 
                action_mask
            )
            # Use blueprint as prior (higher weight) combined with policy network
            with torch.no_grad():
                # Get policy network priors as a fallback/complement
                priors, _, _ = self.policy_net(obs_tensor)
                priors = priors.squeeze(0).cpu().numpy()
                
                # Also get public-only policy for comparison
                public_priors, _, _ = self.policy_net.public_policy(public_obs_tensor, self.current_public_beliefs)
                public_priors = public_priors.squeeze(0).cpu().numpy()
            
            # Combine blueprint strategy with policy priors (weight blueprint more)
            blueprint_weight = 0.7
            combined_priors = blueprint_weight * blueprint_strategy + (1 - blueprint_weight) * priors
        else:
            # Use policy network normally if no blueprint
            with torch.no_grad():
                priors, _, _ = self.policy_net(obs_tensor)
                priors = priors.squeeze(0).cpu().numpy()
                
                public_priors, _, _ = self.policy_net.public_policy(public_obs_tensor, self.current_public_beliefs)
                public_priors = public_priors.squeeze(0).cpu().numpy()
            
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
        
        # Run MCTS simulations
        for _ in range(self.num_simulations):
            sim_env = self.env_creator()  # Clone environment for simulation
            
            # Compute current CFR strategy using PUBLIC state key
            cfr_strategy = self.compute_cfr_strategy(public_state_key, action_mask)
            
            # Select action using PUCT formula but with CFR strategy as prior
            valid_actions = np.where(action_mask)[0]
            best_score = -float('inf')
            best_action = valid_actions[0]
            
            for action in valid_actions:
                if N[action] > 0:
                    # Use PUCT formula with CFR strategy as prior
                    exploitation = Q[action]
                    exploration = self.c_puct * cfr_strategy[action] * np.sqrt(sum(N.values())) / (1 + N[action])
                    score = exploitation + exploration
                else:
                    score = self.c_puct * cfr_strategy[action] * np.sqrt(sum(N.values()) + 1e-5)
                
                if score > best_score:
                    best_score = score
                    best_action = action
            
            # Simulate taking the best_action recursively with subgame solving
            sim_value, action_cf_values = self._simulate(
                sim_env, best_action, observation, public_obs, 
                self.current_beliefs, self.current_public_beliefs, 
                self.search_depth, reach_prob=1.0)
            
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
        # Regret is difference between action's counterfactual value and expected value
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
        
        # Before returning, update the blueprint if available
        if self.blueprint and hasattr(self, 'average_strategy'):
            self.blueprint.update_from_search(
                np.array(public_obs),
                self.current_public_beliefs.cpu().numpy(),
                average_strategy,  # CFR average strategy
                value_estimate,
                immediate_regrets,
                visits=sum(N.values())
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
    
    def _simulate(self, env, action, observation, public_obs, beliefs, public_beliefs, depth, reach_prob=1.0, parent_values=None):
        """
        Simulate taking an action with correlated beliefs that respect physical constraints.
        
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

        # Sample consistent opponent hands (or belief states) that respect physical constraints.
        # Extract just the first two elements (e.g., table cards, non-table cards) from observation.
        private_hand = torch.FloatTensor(observation[:2]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            sampled_hands = self.belief_model.sample_consistent_beliefs(beliefs, private_hand, num_samples=1)
            # Use the first sample as the updated belief state.
            updated_beliefs = sampled_hands.squeeze(1)
        
        # Execute the action in simulation.
        env.step(action)
        reward = env.rewards[agent]
        done = env.terminations[agent]
        
        # Initialize counterfactual values.
        cf_values = {}
        
        # Terminal state handling.
        if done:
            return reward, cf_values
        
        # Boundary condition: depth limit reached.
        if depth == 0:
            next_obs = env.observe(agent)
            if isinstance(next_obs, dict):
                next_obs = next_obs[agent]
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            action_mask = env.infos[agent]["action_mask"]
            
            with torch.no_grad():
                # Update beliefs for the new state using the updated beliefs.
                next_beliefs = self.belief_model(next_obs_tensor, updated_beliefs)
                # Use the value network to obtain a value estimate and regrets.
                value, regrets = self.value_net(next_obs_tensor, next_beliefs)
                avg_value = value.item()
                regrets_np = regrets.squeeze(0).cpu().numpy() * action_mask
                
                valid_actions = np.where(action_mask)[0]
                for a in valid_actions:
                    cf_values[a] = avg_value + regrets_np[a]  # Counterfactual value = V + regret.
            
            return avg_value, cf_values
        
        # Subgame boundary: if round ended or agent changed.
        if env.agent_selection is None or env.agent_selection != original_agent_selection:
            next_obs = env.observe(agent)
            if isinstance(next_obs, dict):
                next_obs = next_obs[agent]
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                next_beliefs = self.belief_model(next_obs_tensor, updated_beliefs)
                value, _ = self.value_net(next_obs_tensor, next_beliefs)
                subgame_value = value.item()
                if parent_values is not None:
                    if action in parent_values and subgame_value < parent_values[action]:
                        subgame_value = parent_values[action]
            
            return reward + subgame_value, cf_values
        
        # Continue recursive simulation within the current subgame.
        next_obs = env.observe(agent)
        if isinstance(next_obs, dict):
            next_obs = next_obs[agent]
        action_mask = env.infos[agent]["action_mask"]
        next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Split next observation into public (and private) parts.
            next_public_obs, _ = self.split_observation(next_obs)
            next_public_tensor = torch.FloatTensor(next_public_obs).unsqueeze(0).to(self.device)
            
            # Update belief states for the new observation.
            next_beliefs = self.belief_model(next_obs_tensor, updated_beliefs)
            next_public_beliefs = self.belief_model.get_public_belief_state(next_obs_tensor, public_beliefs)
            
            # Create a unique key for the new PUBLIC state.
            next_public_state_key = self.public_state_to_key(np.array(next_public_obs), next_public_beliefs)
            # Retrieve the CFR strategy for this state.
            cfr_strategy = self.compute_cfr_strategy(next_public_state_key, action_mask)
            
            # Prepare for recursive simulation over valid actions.
            action_cf_values = {}
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) == 0:
                return reward, cf_values
            
            masked_strategy = cfr_strategy * action_mask
            if np.sum(masked_strategy) <= 0:
                masked_strategy = np.zeros_like(action_mask, dtype=np.float32)
                masked_strategy[valid_actions] = 1.0 / len(valid_actions)
            else:
                masked_strategy = masked_strategy / np.sum(masked_strategy)
            
            total_value = 0.0
            # For each valid action, simulate recursively.
            for a in valid_actions:
                action_reach = reach_prob * masked_strategy[a]
                if action_reach < 1e-6:
                    continue
                
                # Clone the environment to simulate the consequence of action 'a'.
                action_env = env.clone()
                # Recursively simulate the action.
                action_value, rec_cf_values = self._simulate(
                    action_env, a, next_obs, next_public_obs, 
                    next_beliefs, next_public_beliefs, depth - 1, 
                    action_reach, parent_values=action_cf_values
                )
                cf_values[a] = action_value
                total_value += masked_strategy[a] * action_value
        
        return reward + total_value, cf_values


    def play_turn(self, observation, action_mask, table_card):
        """
        Interface method compatible with the game environment.
        Runs recursive search to select an action and extracts additional outputs.
        
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
        
        return search_outcomes
