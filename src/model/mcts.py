# src/model/mcts.py
import numpy as np
import torch
from src import config

class PerfectMCTS:
    """
    Monte Carlo Tree Search with perfect information and action sequence recording.
    Optimized to find sequences where opponents get penalties and replay those exact sequences.
    """
    
    def __init__(self, env, training_agent, opponent_models, exploration_weight=1.0):
        """
        Initialize the Perfect MCTS with opponent penalty focused search.
        
        Args:
            env: The environment instance (will be cloned for simulation)
            training_agent: Name of the agent being trained (e.g., 'player_0')
            opponent_models: Dictionary mapping agent names to their model instances
            exploration_weight: Controls exploration vs exploitation in UCT formula
        """
        self.base_env = env
        self.training_agent = training_agent
        self.opponent_models = opponent_models
        self.exploration_weight = exploration_weight
        
        # Get opponent agent names
        self.opponent_agents = [agent for agent in env.possible_agents if agent != training_agent]
        
        # Performance optimization - cache already computed states
        self.cache = {}
        
        # Store the planned action sequence
        self.action_sequence = []  # List of (agent, action) tuples
        
    def _ucb_score(self, child_value, child_visits, parent_visits):
        """Calculate the UCB score for a node."""
        # Avoid division by zero
        if child_visits == 0:
            return float('inf')
            
        # UCB1 formula
        exploitation = child_value / child_visits
        exploration = self.exploration_weight * np.sqrt(2 * np.log(parent_visits) / child_visits)
        return exploitation + exploration
    
    def _select_opponent_action(self, env, agent):
        """Use the actual opponent model to select an action."""
        try:
            # Ensure we've observed the agent to generate infos
            env.observe(agent, new=True)
            
            # Get appropriate observation format for this opponent
            opponent_model = self.opponent_models[agent]
            
            # Check if agent exists in the environment observations
            if agent not in env.infos or "action_mask" not in env.infos[agent]:
                # Agent might be terminated or in a special state
                return 0  # Return a dummy action
                
            observation = env.observe(agent, new=True)[agent]
            action_mask = env.infos[agent]['action_mask']
            
            # Get action based on opponent type
            if hasattr(opponent_model, 'play_turn'):  # Hardcoded agent
                return opponent_model.play_turn(observation, action_mask, table_card=None)
            else:  # Historical model (neural network)
                # Format observation for historical model
                old_observation = env.observe(agent, new=False)[agent]
                
                # Historical models expect padded observation (similar structure to train_with_belief.py)
                obp_placeholder = np.zeros(2, dtype=np.float32)
                memory_placeholder = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
                final_obs = np.concatenate([old_observation, obp_placeholder, memory_placeholder], axis=0)
                
                # Convert to tensor
                observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device='cpu').unsqueeze(0)
                
                # Get action probabilities
                with torch.no_grad():
                    try:
                        probs, _, _ = opponent_model(observation_tensor, None)
                    except ValueError:
                        try:
                            probs, _ = opponent_model(observation_tensor, None)
                        except:
                            # Fallback to random valid action if model fails
                            valid_actions = [i for i, v in enumerate(action_mask) if v == 1]
                            return np.random.choice(valid_actions) if valid_actions else 0
                
                # Apply action mask
                probs = probs.squeeze().cpu().numpy()
                masked_probs = probs * action_mask
                
                # Normalize if needed
                if masked_probs.sum() > 0:
                    masked_probs = masked_probs / masked_probs.sum()
                else:
                    # If no valid actions according to mask, use uniform distribution over valid actions
                    valid_actions = [i for i, v in enumerate(action_mask) if v == 1]
                    if valid_actions:
                        masked_probs = np.zeros_like(probs)
                        masked_probs[valid_actions] = 1.0 / len(valid_actions)
                    else:
                        return 0  # No valid actions, return dummy action (will be ignored)
                
                # Sample from distribution
                action = np.random.choice(len(masked_probs), p=masked_probs)
                return action
        except Exception as e:
            # If anything goes wrong, return a dummy action
            print(f"Error in _select_opponent_action: {e}")
            return 0
    
    def _is_bluffing(self, env, agent):
        """
        Determine whether the agent's last play was a bluff by examining the actual cards played.
        
        Args:
            env: The environment
            agent: The agent to check
            
        Returns:
            bool: True if the agent was bluffing, False otherwise
        """
        # Get the played cards
        played_cards = env.last_played_cards.get(agent, [])
        if not played_cards:
            return False
            
        # Get the table card
        table_card = env.table_card
        
        # Check if all cards are table cards or jokers
        for card in played_cards:
            if card != table_card and card != "Joker":
                return True  # Found a non-table card - it's a bluff
                
        return False  # All cards were valid table cards or jokers
    
    def _should_skip_mcts(self, env):
        """
        Check if we should skip MCTS and use heuristic rules instead.
        Only challenge in specific scenarios:
        1. If opponent played their last card and it was a bluff
        2. If we're out of table cards and they played a non-table card (bluff)
        """
        # First, ensure we've observed the agent to generate the infos dictionary
        if env.agent_selection == self.training_agent:
            env.observe(self.training_agent, new=True)
            
        last_action_agent = env.last_action_agent
        
        # If no previous action, can't apply this heuristic
        if last_action_agent is None:
            return False, None
            
        # Get action mask for the training agent
        if self.training_agent not in env.infos or "action_mask" not in env.infos[self.training_agent]:
            return False, None
            
        action_mask = env.infos[self.training_agent]["action_mask"]
        
        # Check if challenge action is valid
        if action_mask[6] == 0:  # Challenge action is index 6
            return False, None
        
        # Only challenge if the opponent is actually bluffing
        if not self._is_bluffing(env, last_action_agent):
            return False, None
        
        # Now we know the opponent is bluffing, check our specific conditions
        
        # Check if we have any table cards
        hand = env.players_hands.get(self.training_agent, [])
        table_card = env.table_card
        have_table_cards = self._has_table_cards(hand, table_card)
        
        # Check if opponent has played their last card
        opponent_hand = env.players_hands.get(last_action_agent, [])
        if len(opponent_hand) == 0:
            # Opponent played their last card, and it was a bluff
            return True, 6
        
        # If we're out of table cards and they played a non-table card, challenge
        if not have_table_cards:
            return True, 6
        
        # Otherwise, let MCTS explore
        return False, None
        
    def _evaluate_state(self, env):
        """
        Evaluate the current state based on penalties.
        Returns 1.0 if any opponent has penalties, -1.0 if we have penalties.
        """
        # Get the current penalties
        our_penalties = env.penalties.get(self.training_agent, 0)
        opponent_penalties = {opp: env.penalties.get(opp, 0) for opp in self.opponent_agents}
        
        # Return 1.0 if any opponent has penalties
        if any(penalties > 0 for penalties in opponent_penalties.values()):
            return 1.0
        
        # Return -1.0 if we have penalties
        if our_penalties > 0:
            return -1.0
        
        # If the game is still going but we're in 1v1 situation, check hand sizes
        active_players = env._active_agents_in_round()
        if len(active_players) == 2 and self.training_agent in active_players:
            # Get opponent
            opponent = [p for p in active_players if p != self.training_agent][0]
            
            # If opponent has more cards than us, that's slightly good
            our_cards = len(env.players_hands.get(self.training_agent, []))
            opp_cards = len(env.players_hands.get(opponent, []))
            
            if our_cards < opp_cards:
                return 0.5  # Slight positive value
        
        # Otherwise return 0.0
        return 0.0
    
    def _has_table_cards(self, hand, table_card):
        """
        Check if the hand contains any table cards (matching table_card or Jokers).
        
        Args:
            hand: List of cards in the hand
            table_card: Current table card
            
        Returns:
            bool: True if the hand contains any table cards
        """
        return any(card == table_card or card == "Joker" for card in hand)
    
    def _count_table_cards(self, hand, table_card):
        """
        Count the number of table cards in the hand.
        
        Args:
            hand: List of cards in the hand
            table_card: Current table card
            
        Returns:
            int: Number of table cards in the hand
        """
        return sum(1 for card in hand if card == table_card or card == "Joker")
    
    def _count_non_table_cards(self, hand, table_card):
        """
        Count the number of non-table cards in the hand.
        
        Args:
            hand: List of cards in the hand
            table_card: Current table card
            
        Returns:
            int: Number of non-table cards in the hand
        """
        return sum(1 for card in hand if card != table_card and card != "Joker")
    
    def _simulate_and_record_sequence(self, env_state):
        """
        Run a simulation from the current state until finding a winning path.
        Records the entire action sequence (both our actions and opponent actions).
        
        Returns:
            action_sequence: List of (agent, action) tuples that lead to opponent penalty
            found_winning_path: Whether a winning path was found
        """
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)
        
        # Record the action sequence
        action_sequence = []
        found_winning_path = False
        
        # Maximum steps to prevent infinite loops
        max_steps = 100
        step_count = 0
        
        # Run simulation until we find a winning path or reach max steps
        while step_count < max_steps and sim_env.agent_selection is not None:
            step_count += 1
            current_agent = sim_env.agent_selection
            
            # Check if we already won
            if any(sim_env.penalties.get(opp, 0) > 0 for opp in self.opponent_agents):
                found_winning_path = True
                break
                
            # Check if we lost
            if sim_env.penalties.get(self.training_agent, 0) > 0:
                break
            
            # Select action based on agent type
            if current_agent == self.training_agent:
                # Use improved heuristic for our agent
                skip_mcts, heuristic_action = self._should_skip_mcts(sim_env)
                
                if skip_mcts and heuristic_action is not None:
                    action = heuristic_action
                else:
                    # Get valid actions
                    sim_env.observe(current_agent, new=True)
                    action_mask = sim_env.infos[current_agent]['action_mask']
                    valid_actions = [a for a, valid in enumerate(action_mask) if valid]
                    
                    if valid_actions:
                        # Get our hand and table card
                        hand = sim_env.players_hands.get(current_agent, [])
                        table_card = sim_env.table_card
                        
                        # Safety check - avoid bluffing when at high risk of elimination
                        our_penalties = sim_env.penalties.get(self.training_agent, 0)
                        penalty_threshold = sim_env.penalty_thresholds.get(self.training_agent, 3)
                        high_risk = (our_penalties >= penalty_threshold - 1)  # One penalty away from elimination
                        
                        # NEW PRIORITIZATION with safety check:
                        # 1. Challenge only if:
                        #    a. Opponent played their last card and it was a bluff, or
                        #    b. We're out of table cards and they played a non-table card
                        # 2. If at high risk, prioritize safe play (table cards)
                        # 3. Otherwise, prefer playing non-table cards with higher counts first
                        
                        if 6 in valid_actions and sim_env.last_action_agent is not None:
                            if self._is_bluffing(sim_env, sim_env.last_action_agent):
                                # Check our specific conditions for challenging
                                opponent_hand = sim_env.players_hands.get(sim_env.last_action_agent, [])
                                have_table_cards = self._has_table_cards(hand, table_card)
                                
                                # Challenge if opponent played their last card, or if we're out of table cards
                                if len(opponent_hand) == 0 or not have_table_cards:
                                    action = 6
                                else:
                                    # Don't challenge
                                    if high_risk:
                                        # Play it safe when at high risk
                                        table_card_actions = [a for a in [2, 1, 0] if a in valid_actions]
                                        if table_card_actions:
                                            action = table_card_actions[0]
                                        else:
                                            # If no table cards available, take least risky action
                                            non_table_actions = [a for a in [3, 4, 5] if a in valid_actions]
                                            if non_table_actions:
                                                action = non_table_actions[0]  # Choose non-table action with lowest count
                                            else:
                                                action = valid_actions[0]  # Fallback
                                    else:
                                        # Not high risk, prefer non-table cards with higher counts first
                                        non_table_actions = [a for a in [5, 4, 3] if a in valid_actions]
                                        table_card_actions = [a for a in [2, 1, 0] if a in valid_actions]
                                        
                                        if non_table_actions:
                                            action = non_table_actions[0]  # Choose non-table action with highest count
                                        elif table_card_actions:
                                            action = table_card_actions[0]  # Choose table card action with highest count
                                        else:
                                            action = valid_actions[0]  # Fallback
                            else:
                                # Don't challenge if opponent is not bluffing
                                if high_risk:
                                    # Play it safe when at high risk
                                    table_card_actions = [a for a in [2, 1, 0] if a in valid_actions]
                                    if table_card_actions:
                                        action = table_card_actions[0]
                                    else:
                                        # If no table cards available, take least risky action
                                        non_table_actions = [a for a in [3, 4, 5] if a in valid_actions]
                                        if non_table_actions:
                                            action = non_table_actions[0]  # Choose non-table action with lowest count
                                        else:
                                            action = valid_actions[0]  # Fallback
                                else:
                                    # Not high risk, prefer non-table cards with higher counts first
                                    non_table_actions = [a for a in [5, 4, 3] if a in valid_actions]
                                    table_card_actions = [a for a in [2, 1, 0] if a in valid_actions]
                                    
                                    if non_table_actions:
                                        action = non_table_actions[0]  # Choose non-table action with highest count
                                    elif table_card_actions:
                                        action = table_card_actions[0]  # Choose table card action with highest count
                                    else:
                                        action = valid_actions[0]  # Fallback
                        else:
                            # No option to challenge
                            if high_risk:
                                # Play it safe when at high risk
                                table_card_actions = [a for a in [2, 1, 0] if a in valid_actions]
                                if table_card_actions:
                                    action = table_card_actions[0]
                                else:
                                    # If no table cards available, take least risky action
                                    non_table_actions = [a for a in [3, 4, 5] if a in valid_actions]
                                    if non_table_actions:
                                        action = non_table_actions[0]  # Choose non-table action with lowest count
                                    else:
                                        action = valid_actions[0]  # Fallback
                            else:
                                # Not high risk, prefer non-table cards with higher counts first
                                non_table_actions = [a for a in [5, 4, 3] if a in valid_actions]
                                table_card_actions = [a for a in [2, 1, 0] if a in valid_actions]
                                
                                if non_table_actions:
                                    action = non_table_actions[0]  # Choose non-table action with highest count
                                elif table_card_actions:
                                    action = table_card_actions[0]  # Choose table card action with highest count
                                else:
                                    action = valid_actions[0]  # Fallback
                    else:
                        action = 0  # Fallback to dummy action
            else:
                # For opponent, use their model for simulation
                # This is only for simulation - the actual opponent action during gameplay
                # will be taken from the action_sequence
                action = self._select_opponent_action(sim_env, current_agent)
                
            # Record the action
            action_sequence.append((current_agent, action))
            
            # Execute the action
            sim_env.step(action)
            
            # Check if the action resulted in opponent penalty
            if any(sim_env.penalties.get(opp, 0) > 0 for opp in self.opponent_agents):
                found_winning_path = True
                break
                
            # Check if the action resulted in our penalty
            if sim_env.penalties.get(self.training_agent, 0) > 0:
                break
        
        # If we found a winning path or reached a winning state, return the sequence
        if found_winning_path:
            return action_sequence, True
        else:
            return action_sequence, False
    
    def get_next_opponent_action(self, agent):
        """
        Get the next action for the specified opponent agent from the recorded sequence.
        
        Args:
            agent: The opponent agent name
            
        Returns:
            action: The next action for this opponent, or None if no recorded action
        """
        # Look for the next action for this agent in the sequence
        for i, (seq_agent, action) in enumerate(self.action_sequence):
            if seq_agent == agent:
                # Found an action for this agent, remove it from the sequence
                self.action_sequence.pop(i)
                return action
        
        # If no action found in the sequence, return None
        return None