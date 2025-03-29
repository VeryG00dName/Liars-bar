# src/model/mcts.py
import numpy as np
import torch
from src import config

class PerfectMCTS:
    """
    Monte Carlo Tree Search with perfect information for Liar's Deck.
    
    This implementation is optimized for generating high-quality training trajectories
    by finding sequences where opponents get penalties. It uses full environment
    knowledge and opponent models to create perfect, replayable game traces.
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
        
        # Initialize beliefs for opponents (for training only)
        self.beliefs = self._initialize_beliefs()
        
        # Define opponent labels mapping
        self.opponent_labels = {
            'Classic': 0,
            'GreedyCardSpammer': 1,
            'RandomAgent': 2,
            'SelectiveTableConservativeChallenger': 3,
            'StrategicChallenger': 4,
            'TableFirstConservativeChallenger': 5,
            'TableNonTableAgent': 6,
            'Version_A_player_2': 7,
            'Version_C_player_0': 8,
            'Version_E_player_1': 9
        }
        
        # Debug flag for verbose logging
        self.debug = False
    
    def _log(self, message):
        """Log a message if debug is enabled."""
        if self.debug:
            print(f"MCTS DEBUG: {message}")
    
    def _initialize_beliefs(self):
        """
        Initialize belief vectors for each opponent.
        
        Returns:
            dict: Dictionary mapping opponent IDs to belief vectors
        """
        beliefs = {}
        for opponent in self.opponent_agents:
            # Start with uniform distribution
            belief = np.ones(10) / 10.0
            beliefs[opponent] = belief
        return beliefs
    
    def update_beliefs(self, opponent, true_label):
        """
        Update belief vector for a specific opponent.
        
        Args:
            opponent: Opponent agent ID
            true_label: True opponent type label (index or string)
        """
        # Convert string label to index if needed
        if isinstance(true_label, str):
            true_label = self.opponent_labels.get(true_label, 0)
            
        # Adjust beliefs based on true label
        belief = np.ones(10) * 0.6 / 9.0  # Equal prob for incorrect labels
        belief[true_label] = 0.4          # Higher prob for true label
        
        # Normalize to ensure sum = 1.0
        self.beliefs[opponent] = belief / belief.sum()
    
    def _select_opponent_action(self, env, agent):
        """
        Use the actual opponent model to select an action.
        
        Args:
            env: The environment instance
            agent: The opponent agent ID
            
        Returns:
            int: Selected action index for the opponent
        """
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
                return opponent_model.play_turn(observation, action_mask, table_card=env.table_card)
            else:  # Historical model (neural network)
                # Format observation for historical model
                old_observation = env.observe(agent, new=False)[agent]
                
                # Historical models expect padded observation
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
            self._log(f"Error in _select_opponent_action: {e}")
            return 0  # Return dummy action in case of error
    
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
    
    def _should_challenge_immediately(self, env):
        """
        Check if we should immediately challenge based on hardcoded rules.
        
        Immediate challenge is triggered in two cases:
        1. The opponent played exactly 3 non-table cards
        2. The opponent played all their remaining non-table cards
        
        Args:
            env: The environment
            
        Returns:
            bool: True if we should challenge immediately
        """
        # Only check if there's a last action and agent
        if env.last_action_agent is None or env.last_action is None:
            return False
            
        # Get the opponent who made the last action
        last_agent = env.last_action_agent
        
        # Skip if it's our own action
        if last_agent == self.training_agent:
            return False
        
        # Get played cards and current table card
        played_cards = env.last_played_cards.get(last_agent, [])
        if not played_cards:
            return False
            
        table_card = env.table_card
        
        # Rule 1: Check if the last play was exactly 3 non-table cards
        if len(played_cards) == 3:
            non_table_count = sum(1 for card in played_cards 
                              if card != table_card and card != "Joker")
            if non_table_count == 3:
                return True
            
        # Rule 2: Check if the opponent played all their remaining non-table cards
        hand_before_play = env.players_hands.get(last_agent, []).copy()
        if played_cards:
            # Add back the played cards to reconstruct hand before play
            for card in played_cards:
                hand_before_play.append(card)
            
            # Count non-table cards in the hand before play
            non_table_before = sum(1 for card in hand_before_play 
                                if card != table_card and card != "Joker")
            
            # Count non-table cards in current hand
            non_table_after = sum(1 for card in env.players_hands.get(last_agent, []) 
                               if card != table_card and card != "Joker")
            
            # If all non-table cards were played (and there were some before)
            if non_table_before > 0 and non_table_after == 0:
                return True
                
        return False
    
    def simulate_game(self, env_state, action):
        """
        Simulate a game starting from the given state and action until a penalty occurs or round ends.
        
        Args:
            env_state: The current environment state
            action: The action to simulate
            
        Returns:
            tuple: (reward, action_sequence, is_terminal)
                reward: 1 if opponent gets penalty, 0 if we get penalty
                action_sequence: List of (agent, action) pairs for replay
                is_terminal: Whether the game reached a terminal state
        """
        # Clone the environment for simulation
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)
        
        # Check if it's our turn
        if sim_env.agent_selection != self.training_agent:
            return 0, [], False
            
        # Take our action
        action_sequence = [(self.training_agent, action)]
        initial_penalties = {agent: sim_env.penalties.get(agent, 0) for agent in sim_env.possible_agents}
        
        sim_env.step(action)
        
        # Check if our action resulted in an opponent getting a penalty
        for opponent in self.opponent_agents:
            if sim_env.penalties.get(opponent, 0) > initial_penalties.get(opponent, 0):
                return 1, action_sequence, sim_env.terminations.get(opponent, False)
                
        # Check if we got a penalty
        if sim_env.penalties.get(self.training_agent, 0) > initial_penalties.get(self.training_agent, 0):
            return 0, action_sequence, sim_env.terminations.get(self.training_agent, False)
            
        # If the round ended or game ended, return current action sequence
        if sim_env.agent_selection is None:
            return 0, action_sequence, True
            
        # If it's our turn again immediately, the round ended
        if sim_env.agent_selection == self.training_agent:
            return 0, action_sequence, False
            
        # Simulate opponent actions until a penalty occurs or it's our turn again
        max_steps = 20  # Limit to prevent infinite loops
        step_count = 0
        
        while (sim_env.agent_selection != self.training_agent and 
               sim_env.agent_selection is not None and 
               step_count < max_steps):
            
            step_count += 1
            current_agent = sim_env.agent_selection
            
            # Get action for opponent
            opponent_action = self._select_opponent_action(sim_env, current_agent)
            action_sequence.append((current_agent, opponent_action))
            
            # Record penalties before step
            pre_step_penalties = {agent: sim_env.penalties.get(agent, 0) 
                                for agent in sim_env.possible_agents}
            
            # Take step
            sim_env.step(opponent_action)
            
            # Check if any penalties occurred
            for agent in sim_env.possible_agents:
                if sim_env.penalties.get(agent, 0) > pre_step_penalties.get(agent, 0):
                    # Our agent got a penalty - bad outcome
                    if agent == self.training_agent:
                        return 0, action_sequence, sim_env.terminations.get(agent, False)
                    # Opponent got a penalty - good outcome
                    else:
                        return 1, action_sequence, sim_env.terminations.get(agent, False)
                        
            # If it's our turn again, we're done with this simulation
            if sim_env.agent_selection == self.training_agent:
                break
        
        # If we reach max steps without resolution, return neutral outcome
        return 0, action_sequence, False
    
    def search_with_details(self, env_state):
        """
        Run search to find the best action and generate a complete action sequence.
        Returns full details including action sequence and beliefs.
        
        Args:
            env_state: Environment state to start search from
            
        Returns:
            dict: Dictionary containing action probabilities, best action, value, sequence, and beliefs
        """
        # Create a sim env to examine the current state
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)
        
        # Make sure it's our turn
        if sim_env.agent_selection != self.training_agent:
            self._log(f"Warning: search called when it's not our turn ({sim_env.agent_selection})")
            action_dim = sim_env.action_spaces[self.training_agent].n
            return {
                "action_probs": np.ones(action_dim) / action_dim,
                "best_action": None,
                "best_value": 0,
                "action_sequence": [],
                "beliefs": self.beliefs
            }
            
        # Get valid actions
        sim_env.observe(self.training_agent, new=True)
        if self.training_agent not in sim_env.infos or "action_mask" not in sim_env.infos[self.training_agent]:
            action_dim = sim_env.action_spaces[self.training_agent].n
            return {
                "action_probs": np.ones(action_dim) / action_dim,
                "best_action": None,
                "best_value": 0,
                "action_sequence": [],
                "beliefs": self.beliefs
            }
            
        action_mask = sim_env.infos[self.training_agent]['action_mask']
        valid_actions = [a for a, mask in enumerate(action_mask) if mask == 1]
        
        if not valid_actions:
            action_dim = sim_env.action_spaces[self.training_agent].n
            return {
                "action_probs": np.ones(action_dim) / action_dim,
                "best_action": None,
                "best_value": 0,
                "action_sequence": [],
                "beliefs": self.beliefs
            }
            
        # Check for immediate challenge rule
        challenge_action = 6  # Challenge action index
        if challenge_action in valid_actions and self._should_challenge_immediately(sim_env):
            reward, action_sequence, is_terminal = self.simulate_game(env_state, challenge_action)
            
            if reward > 0:  # If it's successful
                action_dim = sim_env.action_spaces[self.training_agent].n
                action_probs = np.zeros(action_dim)
                action_probs[challenge_action] = 1.0
                
                # Save the action sequence for future moves
                self.action_sequence = action_sequence
                
                return {
                    "action_probs": action_probs,
                    "best_action": challenge_action,
                    "best_value": reward,
                    "action_sequence": action_sequence,
                    "beliefs": self.beliefs
                }
        
        # 1. First try non-challenge actions
        non_challenge_actions = [a for a in valid_actions if a != 6]
        best_action = None
        best_value = -float('inf')
        best_sequence = []
        
        # Special case: playing table cards
        our_hand = sim_env.players_hands.get(self.training_agent, [])
        table_card_count = self._count_table_cards(our_hand, sim_env.table_card)
        
        # Try playing 3 table cards if we have them (Rule A)
        if table_card_count >= 3 and 2 in valid_actions:  # Action 2 = play 3 table cards
            reward, action_sequence, is_terminal = self.simulate_game(env_state, 2)
            
            # Only keep if an opponent challenges (and gets penalized)
            if reward > 0:
                best_action = 2
                best_value = reward
                best_sequence = action_sequence
        
        # Try playing 2 table cards if we have them (Rule B)
        if best_action is None and table_card_count >= 2 and 1 in valid_actions:  # Action 1 = play 2 table cards
            reward, action_sequence, is_terminal = self.simulate_game(env_state, 1)
            
            # Only keep if an opponent challenges (and gets penalized)
            if reward > 0:
                best_action = 1
                best_value = reward
                best_sequence = action_sequence
        
        # Try all other non-challenge actions
        for action in non_challenge_actions:
            if action == best_action:  # Skip if we already tried this action
                continue
                
            reward, action_sequence, is_terminal = self.simulate_game(env_state, action)
            
            if reward > best_value:
                best_action = action
                best_value = reward
                best_sequence = action_sequence
        
        # 2. If no good action found, try challenge
        if (best_action is None or best_value <= 0) and challenge_action in valid_actions:
            reward, action_sequence, is_terminal = self.simulate_game(env_state, challenge_action)
            
            if reward > best_value:
                best_action = challenge_action
                best_value = reward
                best_sequence = action_sequence
        
        # If still no good action, just pick first valid action
        if best_action is None and valid_actions:
            best_action = valid_actions[0]
            reward, action_sequence, is_terminal = self.simulate_game(env_state, best_action)
            best_value = 0
            best_sequence = action_sequence
        
        # Create one-hot probability distribution
        action_dim = sim_env.action_spaces[self.training_agent].n
        action_probs = np.zeros(action_dim)
        if best_action is not None:
            action_probs[best_action] = 1.0
        
        # Save the action sequence for future moves
        self.action_sequence = best_sequence
        
        return {
            "action_probs": action_probs,
            "best_action": best_action,
            "best_value": best_value,
            "action_sequence": best_sequence,"beliefs": self.beliefs
        }
    
    def search(self, env_state):
        """
        Simplified version of search that returns only the core elements needed
        for standard MCTS usage.
        
        Args:
            env_state: Environment state to start search from
            
        Returns:
            tuple: (action_probs, best_action, best_value)
        """
        # Call the detailed search method
        result = self.search_with_details(env_state)
        
        # Return just the core elements
        return result["action_probs"], result["best_action"], result["best_value"]
    
    def get_next_opponent_action(self, agent):
        """
        Get the next action for the specified opponent agent from the recorded sequence.
        
        Args:
            agent: The opponent agent name
            
        Returns:
            action: The next action for this opponent or None if not found
        """
        # Look for the next action for this agent in the sequence
        for i, (seq_agent, action) in enumerate(self.action_sequence):
            if seq_agent == agent:
                # Found an action for this agent, remove it from the sequence
                self.action_sequence.pop(i)
                self._log(f"Found action for {agent}: {action}")
                return action
        
        # If no action found, return None
        self._log(f"No pre-planned action found for {agent} in action sequence")
        return None
    
    def create_simulated_belief(self, opponent, true_label):
        """
        Create a simulated belief vector with higher probability on the true label.
        
        Args:
            opponent: Opponent agent ID
            true_label: The true opponent type (index or string)
            
        Returns:
            np.ndarray: Belief vector with shape (10,)
        """
        # Convert string label to index if needed
        if isinstance(true_label, str):
            true_label = self.opponent_labels.get(true_label, 0)
            
        # Create belief vector with higher probability on true label
        belief = np.ones(10) * 0.6 / 9.0
        belief[true_label] = 0.4
        
        # Normalize to ensure sum = 1.0
        return belief / belief.sum()
    
    def update_all_beliefs(self, opponent_labels):
        """
        Update beliefs for all opponents based on provided labels.
        
        Args:
            opponent_labels: Dictionary mapping opponent IDs to their true labels
        """
        for opponent, label in opponent_labels.items():
            if opponent in self.beliefs:
                self.update_beliefs(opponent, label)
                
    def simulate_round_end(self, env_state):
        """
        Simulate until the end of the current round to check outcomes.
        Useful for evaluating long-term consequences of actions.
        
        Args:
            env_state: Current environment state
            
        Returns:
            tuple: (reward, action_sequence, final_state)
                reward: Cumulative reward (positive if we win, negative if opponent wins)
                action_sequence: Complete action sequence for the round
                final_state: Final environment state
        """
        # Clone the environment for simulation
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)
        
        action_sequence = []
        total_reward = 0
        
        # Remember initial penalties to track changes
        initial_penalties = {agent: sim_env.penalties.get(agent, 0) 
                            for agent in sim_env.possible_agents}
        
        # Maximum number of steps to simulate (to prevent infinite loops)
        max_steps = 50
        step_count = 0
        
        # Simulate until round ends or max steps reached
        while step_count < max_steps:
            step_count += 1
            
            # If no agent to select, round has ended
            if sim_env.agent_selection is None:
                break
                
            current_agent = sim_env.agent_selection
            
            # Choose action based on agent
            if current_agent == self.training_agent:
                # For our agent, use first valid action for simple simulation
                sim_env.observe(current_agent, new=True)
                action_mask = sim_env.infos[current_agent]['action_mask']
                valid_actions = [a for a, mask in enumerate(action_mask) if mask == 1]
                
                if not valid_actions:
                    break
                    
                action = valid_actions[0]
            else:
                # For opponents, use their model
                action = self._select_opponent_action(sim_env, current_agent)
                
            # Record the action
            action_sequence.append((current_agent, action))
            
            # Take the step
            sim_env.step(action)
            
            # Check for penalties
            for agent in sim_env.possible_agents:
                penalty_diff = sim_env.penalties.get(agent, 0) - initial_penalties.get(agent, 0)
                
                if penalty_diff > 0:
                    # Reward us if opponent got penalty, punish if we got penalty
                    if agent == self.training_agent:
                        total_reward -= 1
                    else:
                        total_reward += 1
                        
                    # Update initial penalties for future penalty tracking
                    initial_penalties[agent] = sim_env.penalties.get(agent, 0)
            
            # If the round has ended, break
            if len(sim_env._active_agents_in_round()) <= 1:
                break
        
        return total_reward, action_sequence, sim_env.get_state()
    
    def evaluate_terminal_state(self, env_state):
        """
        Evaluate a terminal state to determine its value.
        
        Args:
            env_state: Environment state to evaluate
            
        Returns:
            float: Value of the state (1 if we win, -1 if we lose, 0 otherwise)
        """
        # Clone the environment to examine the state
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)
        
        # Check if game is over and who won
        if sim_env.winner:
            if sim_env.winner == self.training_agent:
                return 1.0  # We won
            else:
                return -1.0  # We lost
        
        # Check penalty counts
        our_penalty = sim_env.penalties.get(self.training_agent, 0)
        opponent_penalties = {opp: sim_env.penalties.get(opp, 0) for opp in self.opponent_agents}
        
        # Compare penalties
        if our_penalty == 0 and any(p > 0 for p in opponent_penalties.values()):
            return 0.5  # We're doing well, no penalties
        elif our_penalty > 0 and all(p == 0 for p in opponent_penalties.values()):
            return -0.5  # We have penalties but opponents don't
            
        # Default neutral value
        return 0.0