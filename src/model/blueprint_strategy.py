# src/model/blueprint_strategy.py
import torch
import numpy as np
from collections import defaultdict
import os
import pickle

# Update the BlueprintStrategy class to incorporate opponent identity

class BlueprintStrategy:
    """
    Stores and provides access to a pre-computed strategy blueprint for the game.
    Now incorporates opponent identity for more specific strategies.
    """
    def __init__(self, policy_net=None, belief_model=None):
        """Initialize the blueprint strategy either empty or with networks."""
        self.policy_net = policy_net
        self.belief_model = belief_model
        
        # Storage for state-action mappings with opponent identity
        self.strategy_map = defaultdict(lambda: np.zeros(7))  # For 7 actions
        self.value_map = {}
        self.visit_counts = defaultdict(int)
        
        # CFR-related data
        self.average_strategy = defaultdict(lambda: np.zeros(7))
        self.cumulative_regrets = defaultdict(lambda: np.zeros(7))
    
    def state_to_key(self, public_obs, beliefs=None, opponent_id=None):
        """
        Convert public observation and belief state to a unique key for storage.
        Now incorporates opponent identity when available.
        
        Args:
            public_obs: Public observation vector
            beliefs: Current belief state (optional)
            opponent_id: The opponent's identity (optional)
        
        Returns:
            String key that uniquely identifies this public state
        """
        key_parts = [str(public_obs)]
        
        if beliefs is not None:
            if isinstance(beliefs, torch.Tensor):
                belief_arr = beliefs.cpu().numpy()
            else:
                belief_arr = beliefs
            key_parts.append(str(belief_arr))
        
        if opponent_id is not None:
            key_parts.append(str(opponent_id))
        
        return hash("_".join(key_parts))
    
    def update_strategy(self, public_obs, beliefs, strategy, value, visits=1, opponent_id=None):
        """
        Update the blueprint with a new strategy for a given state.
        Now incorporates opponent identity.
        
        Args:
            public_obs: Public observation vector
            beliefs: Current belief state
            strategy: Strategy (probability distribution over actions)
            value: Value estimate for this state
            visits: Number of visits to this state
            opponent_id: The opponent's identity (optional)
        """
        key = self.state_to_key(public_obs, beliefs, opponent_id)
        
        # Incremental update weighted by visits
        current_visits = self.visit_counts[key]
        total_visits = current_visits + visits
        
        if current_visits > 0:
            # Update with weighted average
            self.strategy_map[key] = (
                (current_visits / total_visits) * self.strategy_map[key] +
                (visits / total_visits) * strategy
            )
            self.value_map[key] = (
                (current_visits / total_visits) * self.value_map.get(key, 0) +
                (visits / total_visits) * value
            )
        else:
            # First visit
            self.strategy_map[key] = strategy
            self.value_map[key] = value
        
        self.visit_counts[key] = total_visits
    
    def query(self, public_obs, beliefs=None, action_mask=None, opponent_id=None):
        """
        Query the blueprint for a strategy in the given state.
        Now takes opponent identity into account.
        
        Args:
            public_obs: Public observation vector
            beliefs: Current belief state (optional)
            action_mask: Mask of valid actions (optional)
            opponent_id: The opponent's identity (optional)
        
        Returns:
            Tuple of (strategy, value)
        """
        # Try opponent-specific key first
        if opponent_id is not None:
            opponent_key = self.state_to_key(public_obs, beliefs, opponent_id)
            if opponent_key in self.strategy_map:
                strategy = self.strategy_map[opponent_key]
                value = self.value_map.get(opponent_key, 0.0)
                
                # Apply action mask if provided
                if action_mask is not None:
                    masked_strategy = strategy * action_mask
                    if np.sum(masked_strategy) > 0:
                        masked_strategy = masked_strategy / np.sum(masked_strategy)
                    else:
                        # Fallback to uniform over valid actions
                        valid_actions = np.where(action_mask)[0]
                        masked_strategy = np.zeros_like(strategy)
                        masked_strategy[valid_actions] = 1.0 / len(valid_actions)
                    return masked_strategy, value
                
                return strategy, value
        
        # Fall back to general state key without opponent identity
        generic_key = self.state_to_key(public_obs, beliefs)
        
        # If state exists in our map
        if generic_key in self.strategy_map:
            strategy = self.strategy_map[generic_key]
            value = self.value_map.get(generic_key, 0.0)
            
            # Apply action mask if provided
            if action_mask is not None:
                masked_strategy = strategy * action_mask
                if np.sum(masked_strategy) > 0:
                    masked_strategy = masked_strategy / np.sum(masked_strategy)
                else:
                    # Fallback to uniform over valid actions
                    valid_actions = np.where(action_mask)[0]
                    masked_strategy = np.zeros_like(strategy)
                    masked_strategy[valid_actions] = 1.0 / len(valid_actions)
                return masked_strategy, value
            
            return strategy, value
        
        # If we have neural networks but state not in map, use them
        elif self.policy_net is not None and beliefs is not None:
            # Convert to tensors
            device = next(self.policy_net.parameters()).device
            if not isinstance(public_obs, torch.Tensor):
                public_obs_tensor = torch.FloatTensor(public_obs).unsqueeze(0).to(device)
            else:
                public_obs_tensor = public_obs.to(device)
                
            if not isinstance(beliefs, torch.Tensor):
                beliefs_tensor = torch.FloatTensor(beliefs).unsqueeze(0).to(device)
            else:
                beliefs_tensor = beliefs.to(device)
            
            # Get public policy
            with torch.no_grad():
                probs, value, _ = self.policy_net.public_policy(public_obs_tensor, beliefs_tensor)
                strategy = probs.squeeze(0).cpu().numpy()
                value = value.item()
            
            # Apply action mask if provided
            if action_mask is not None:
                masked_strategy = strategy * action_mask
                if np.sum(masked_strategy) > 0:
                    masked_strategy = masked_strategy / np.sum(masked_strategy)
                else:
                    # Fallback to uniform over valid actions
                    valid_actions = np.where(action_mask)[0]
                    masked_strategy = np.zeros_like(strategy)
                    masked_strategy[valid_actions] = 1.0 / len(valid_actions)
                return masked_strategy, value
                
            return strategy, value
        
        # Fallback to uniform random if no data available
        else:
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]
                strategy = np.zeros(7)
                strategy[valid_actions] = 1.0 / len(valid_actions)
                return strategy, 0.0
            else:
                return np.ones(7) / 7, 0.0
    
    def update_from_search(self, public_obs, beliefs, cfr_strategy, value, regrets, visits=1, opponent_id=None):
        """
        Update blueprint from search results, including CFR information.
        Now incorporates opponent identity.
        
        Args:
            public_obs: Public observation
            beliefs: Belief state
            cfr_strategy: CFR strategy (average)
            value: Value estimate
            regrets: Counterfactual regrets
            visits: Visit count for weighting
            opponent_id: The opponent's identity (optional)
        """
        key = self.state_to_key(public_obs, beliefs, opponent_id)
        
        # Update strategy and value
        self.update_strategy(public_obs, beliefs, cfr_strategy, value, visits, opponent_id)
        
        # Update CFR data
        self.average_strategy[key] = cfr_strategy
        
        # Update cumulative regrets (weighted by visits)
        current_regrets = self.cumulative_regrets[key]
        self.cumulative_regrets[key] = current_regrets + regrets * visits
    
    def save(self, filepath):
        """Save the blueprint strategy to disk."""
        data = {
            'strategy_map': dict(self.strategy_map),
            'value_map': self.value_map,
            'visit_counts': dict(self.visit_counts),
            'average_strategy': dict(self.average_strategy),
            'cumulative_regrets': dict(self.cumulative_regrets)
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath, policy_net=None, belief_model=None):
        """Load a blueprint strategy from disk."""
        blueprint = cls(policy_net=policy_net, belief_model=belief_model)
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        blueprint.strategy_map = defaultdict(lambda: np.zeros(7), data['strategy_map'])
        blueprint.value_map = data['value_map']
        blueprint.visit_counts = defaultdict(int, data['visit_counts'])
        blueprint.average_strategy = defaultdict(lambda: np.zeros(7), data['average_strategy'])
        blueprint.cumulative_regrets = defaultdict(lambda: np.zeros(7), data['cumulative_regrets'])
        
        return blueprint