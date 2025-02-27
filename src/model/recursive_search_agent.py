# src/model/recursive_search_agent.py
import torch
import numpy as np

class RecursiveSearchAgent:
    def __init__(self, policy_net, belief_model, value_net, env_creator, 
                 device, search_depth=3, num_simulations=30, c_puct=1.0,
                 agent_name=None, agent_index=None):
        """
        Agent that uses belief-based recursive search for decision making.
        
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
        
        self.current_beliefs = None
        self.action_history = []
        self.search_statistics = {}
    
    def reset(self):
        """Reset agent state at the beginning of a new game."""
        self.current_beliefs = None
        self.action_history = []
        self.search_statistics = {}
    
    def update_beliefs(self, observation, action_mask=None):
        """
        Update belief states based on new observation.
        
        Args:
            observation: Current observation
            action_mask: Mask of valid actions
        """
        # If observation is a dict (from env.observe()), extract tensor for the current agent.
        if isinstance(observation, dict):
            obs_data = observation[self.name]
        else:
            obs_data = observation
        
        obs_tensor = torch.FloatTensor(obs_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.current_beliefs is None:
                self.current_beliefs = self.belief_model(obs_tensor)
            else:
                self.current_beliefs = self.belief_model(obs_tensor, self.current_beliefs)
    
    def mcts_search(self, observation, action_mask):
        """
        Perform Monte Carlo Tree Search with belief states.
        
        Args:
            observation: Current observation
            action_mask: Mask of valid actions
            
        Returns:
            Dictionary with search outcomes:
              - selected_action: the chosen action (int)
              - search_policy: distribution over actions (np.array)
              - value_estimate: value from subgame solver (float)
              - counterfactual_regrets: vector of per-action regrets (np.array)
        """
        # Convert observation appropriately
        if isinstance(observation, dict):
            observation = observation[self.name]
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Ensure beliefs are updated
        self.update_beliefs(observation, action_mask)
        
        # Get prior probabilities from policy network
        with torch.no_grad():
            priors, _, _ = self.policy_net(obs_tensor)
            priors = priors.squeeze(0).cpu().numpy()
        
        # Apply action mask
        masked_priors = priors * action_mask
        if np.sum(masked_priors) > 0:
            masked_priors = masked_priors / np.sum(masked_priors)
        else:
            valid_actions = np.where(action_mask)[0]
            masked_priors = np.zeros_like(priors)
            masked_priors[valid_actions] = 1.0 / len(valid_actions)
        
        # Initialize search statistics: visit count (N), total value (W), and mean value (Q)
        N = {a: 0 for a in range(len(action_mask))}
        W = {a: 0.0 for a in range(len(action_mask))}
        Q = {a: 0.0 for a in range(len(action_mask))}
        
        # Exploration parameter (adjusted dynamically)
        total_visits = 0
        c_puct = self.c_puct
        
        # Run MCTS simulations
        for _ in range(self.num_simulations):
            sim_env = self.env_creator()  # Clone environment for simulation
            # Select action using PUCT formula
            valid_actions = np.where(action_mask)[0]
            best_score = -float('inf')
            best_action = valid_actions[0]
            
            for action in valid_actions:
                if N[action] > 0:
                    exploitation = Q[action]
                    exploration = c_puct * masked_priors[action] * np.sqrt(sum(N.values())) / (1 + N[action])
                    score = exploitation + exploration
                else:
                    score = c_puct * masked_priors[action] * np.sqrt(sum(N.values()) + 1e-5)
                
                if score > best_score:
                    best_score = score
                    best_action = action
            
            # Simulate taking the best_action recursively
            sim_value = self._simulate(sim_env, best_action, observation, self.current_beliefs, self.search_depth)
            
            # Update statistics for the selected action
            N[best_action] += 1
            W[best_action] += sim_value
            Q[best_action] = W[best_action] / N[best_action]
            total_visits += 1
        
        # Compute search policy as normalized visit counts
        visit_array = np.array([N[a] for a in range(len(action_mask))], dtype=np.float32)
        if visit_array.sum() > 0:
            search_policy = visit_array / visit_array.sum()
        else:
            search_policy = masked_priors
        
        # Compute overall value estimate as weighted average of Q-values
        value_estimate = sum(N[a] * Q[a] for a in range(len(action_mask))) / (visit_array.sum() + 1e-10)
        
        # Compute counterfactual regrets: difference between each action's Q and the baseline value
        regrets = np.array([Q[a] - value_estimate for a in range(len(action_mask))], dtype=np.float32)
        
        # Store search statistics for later analysis/training
        self.search_statistics = {'N': N, 'Q': Q, 'masked_priors': masked_priors,
                                  'search_policy': search_policy, 'value_estimate': value_estimate,
                                  'counterfactual_regrets': regrets}
        
        # Action selection: using temperature-based sampling (for now, set temperature = 1.0)
        temperature = 1.0
        if temperature < 0.01:
            selected_action = max(N.items(), key=lambda x: x[1])[0]
        else:
            visit_counts = visit_array ** (1.0 / temperature)
            if visit_counts.sum() > 0:
                probs = visit_counts / visit_counts.sum()
                selected_action = np.random.choice(len(action_mask), p=probs)
            else:
                selected_action = np.argmax(masked_priors)
        
        return {
            'selected_action': selected_action,
            'search_policy': search_policy,
            'value_estimate': value_estimate,
            'counterfactual_regrets': regrets
        }
    
    def _simulate(self, env, action, observation, beliefs, depth):
        """
        Simulate taking an action and recursively evaluate the resulting state.
        
        Args:
            env: Cloned environment for simulation
            action: Action to simulate
            observation: Current observation
            beliefs: Current belief state
            depth: Remaining search depth
            
        Returns:
            Estimated value after taking the action (float)
        """
        agent = self.name
        original_agent_selection = env.agent_selection
        
        # Execute the action in simulation
        env.step(action)
        reward = env.rewards[agent]
        done = env.terminations[agent]
        
        # If terminal state or max depth reached, return immediate reward
        if done or depth == 0:
            return reward
        
        # If round ended, use the value network to estimate remaining value
        if env.agent_selection is None or env.agent_selection != original_agent_selection:
            next_obs = env.observe(agent)
            if isinstance(next_obs, dict):
                next_obs = next_obs[self.name]
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_beliefs = self.belief_model(next_obs_tensor, beliefs)
                value, _ = self.value_net(next_obs_tensor, next_beliefs)
            return reward + value.item()
        
        # Otherwise, get next observation, update beliefs, and recurse
        next_obs = env.observe(agent)
        if isinstance(next_obs, dict):
            next_obs = next_obs[self.name]
        action_mask = env.infos[agent]["action_mask"]
        next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_beliefs = self.belief_model(next_obs_tensor, beliefs)
            priors, _, _ = self.policy_net(next_obs_tensor)
            priors = priors.squeeze(0).cpu().numpy()
            masked_priors = priors * action_mask
            if np.sum(masked_priors) > 0:
                masked_priors = masked_priors / np.sum(masked_priors)
            else:
                valid_actions = np.where(action_mask)[0]
                masked_priors = np.zeros_like(priors)
                masked_priors[valid_actions] = 1.0 / len(valid_actions)
            next_action = np.random.choice(len(masked_priors), p=masked_priors)
            next_value = self._simulate(env, next_action, next_obs, next_beliefs, depth - 1)
        
        return reward + next_value

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
              - search_policy: Distribution over actions at the root.
              - value_estimate: Value estimate from the subgame solver.
              - counterfactual_regrets: Computed regrets for available actions.
        """
        # Update beliefs based on the latest observation
        self.update_beliefs(observation, action_mask)
        
        # Run MCTS search to obtain search outputs
        search_outcomes = self.mcts_search(observation, action_mask)
        
        # Record complete transition for later training
        self.action_history.append({
            'observation': observation,
            'action_mask': action_mask,
            'table_card': table_card,
            'selected_action': search_outcomes['selected_action'],
            'search_policy': search_outcomes['search_policy'],
            'value_estimate': search_outcomes['value_estimate'],
            'counterfactual_regrets': search_outcomes['counterfactual_regrets']
        })
        
        return search_outcomes
