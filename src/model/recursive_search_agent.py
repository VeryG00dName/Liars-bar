import torch
import numpy as np
import random
from src.env.liars_deck_env_utils_2 import decode_action, validate_claim
from src.model.memory import get_opponent_memory

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
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.current_beliefs is None:
                # Initialize beliefs
                self.current_beliefs = self.belief_model(obs_tensor)
            else:
                # Update beliefs with new observation
                self.current_beliefs = self.belief_model(obs_tensor, self.current_beliefs)
    
    def mcts_search(self, observation, action_mask):
        """
        Perform Monte Carlo Tree Search with belief states.
        
        Args:
            observation: Current observation
            action_mask: Mask of valid actions
            
        Returns:
            Best action according to search
        """
        # Convert to PyTorch tensors
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
        
        # Initialize search statistics
        N = {a: 0 for a in range(len(action_mask))}  # Visit count
        W = {a: 0.0 for a in range(len(action_mask))}  # Total value
        Q = {a: 0.0 for a in range(len(action_mask))}  # Mean value
        
        # Set exploration parameter - gradually reduce as confidence grows
        c_puct = self.c_puct * max(0.5, min(1.0, 10.0 / (sum(N.values()) + 10)))
        
        # Run simulations
        for _ in range(self.num_simulations):
            # Create a copy of the environment for simulation
            sim_env = self.env_creator()  # We use the clone method we added to the environment
            
            # Select action based on PUCT formula
            valid_actions = np.where(action_mask)[0]
            best_score = -float('inf')
            best_action = valid_actions[0]
            
            for action in valid_actions:
                # PUCT formula balances exploration and exploitation
                if N[action] > 0:
                    # Exploitation term: Q-value of the action
                    exploitation = Q[action]
                    # Exploration term: prior probability weighted by visit count
                    exploration = c_puct * masked_priors[action] * np.sqrt(sum(N.values())) / (1 + N[action])
                    score = exploitation + exploration
                else:
                    # For unvisited actions, prioritize by prior probability
                    score = c_puct * masked_priors[action] * np.sqrt(sum(N.values()) + 1e-5)
                
                if score > best_score:
                    best_score = score
                    best_action = action
            
            # Simulate the action and get value
            value = self._simulate(sim_env, best_action, observation, self.current_beliefs, self.search_depth)
            
            # Update statistics
            N[best_action] += 1
            W[best_action] += value
            Q[best_action] = W[best_action] / N[best_action]
        
        # Store search statistics for analysis
        self.search_statistics = {'N': N, 'Q': Q, 'masked_priors': masked_priors}
        
        # Temperature parameter for action selection
        # Use lower temperature (more greedy) as training progresses
        temperature = 1.0
        
        # Select best action based on visit count or Q-value
        if temperature < 0.01:
            # Greedy selection
            best_action = max(N.items(), key=lambda x: x[1])[0]
        else:
            # Temperature-based sampling
            visit_counts = np.array([N.get(a, 0) for a in range(len(action_mask))])
            visit_counts = visit_counts ** (1.0 / temperature)
            if visit_counts.sum() > 0:
                probs = visit_counts / visit_counts.sum()
                best_action = np.random.choice(len(action_mask), p=probs)
            else:
                # Fallback to prior if no visits
                best_action = np.argmax(masked_priors)
        
        return best_action
    
    def _simulate(self, env, action, observation, beliefs, depth):
        """
        Simulate taking an action and recursively evaluate resulting belief state.
        
        Args:
            env: Environment copy for simulation
            action: Action to simulate
            observation: Current observation
            beliefs: Current belief state
            depth: Remaining search depth
            
        Returns:
            Estimated value of the state after taking the action
        """
        # Get current agent and store original state
        agent = self.name
        original_agent_selection = env.agent_selection
        
        # Take the action in simulation environment
        env.step(action)
        
        # Get reward from this action
        reward = env.rewards[agent]
        done = env.terminations[agent]
        
        # If terminal state or max depth reached, return immediate reward
        if done or depth == 0:
            return reward
        
        # If round ended (indicated by env.agent_selection being None or different agent)
        if env.agent_selection is None or env.agent_selection != original_agent_selection:
            # For round end, just use value network to estimate remaining value
            # Since we can't effectively search past stochastic card drawing
            next_obs = env.observe(agent)
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                next_beliefs = self.belief_model(next_obs_tensor, beliefs)
                value = self.value_net(next_obs_tensor, next_beliefs).item()
            
            return reward + value
        
        # Otherwise, get next observation and update beliefs
        next_obs = env.observe(agent)
        action_mask = env.infos[agent]["action_mask"]
        
        next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Update beliefs based on new observation
            next_beliefs = self.belief_model(next_obs_tensor, beliefs)
            
            # Get prior probabilities from policy network
            priors, _, _ = self.policy_net(next_obs_tensor)
            priors = priors.squeeze(0).cpu().numpy()
            
            # Apply action mask
            masked_priors = priors * action_mask
            if np.sum(masked_priors) > 0:
                masked_priors = masked_priors / np.sum(masked_priors)
            else:
                valid_actions = np.where(action_mask)[0]
                masked_priors = np.zeros_like(priors)
                masked_priors[valid_actions] = 1.0 / len(valid_actions)
            
            # Choose next action based on priors (simplified simulation)
            next_action = np.random.choice(len(masked_priors), p=masked_priors)
            
            # Recursive simulation with reduced depth
            next_value = self._simulate(env, next_action, next_obs, next_beliefs, depth-1)
        
        return reward + next_value

    
    def play_turn(self, observation, action_mask, table_card):
        """
        Interface method compatible with the game environment.
        
        Args:
            observation: Current observation
            action_mask: Mask of valid actions
            table_card: Current table card
            
        Returns:
            Selected action
        """
        # Update beliefs
        self.update_beliefs(observation, action_mask)
        
        # Perform search to find best action
        action = self.mcts_search(observation, action_mask)
        
        # Record action for later analysis
        self.action_history.append({
            'observation': observation,
            'action': action,
            'table_card': table_card,
            'action_mask': action_mask
        })
        
        return action