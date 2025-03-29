# src/model/mcts.py
import numpy as np
import torch
from src import config
from src.env.liars_deck_env_utils_2 import decode_action

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
        self.debug = True
    
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
            if true_label not in self.opponent_labels:
                raise ValueError(f"Unknown opponent label: {true_label}")
            true_label = self.opponent_labels[true_label]
            
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
        # Ensure we've observed the agent to generate infos
        env.observe(agent, new=True)
        
        # Get appropriate observation format for this opponent
        opponent_model = self.opponent_models[agent]
        
        # Check if agent exists in the environment observations
        if agent not in env.infos or "action_mask" not in env.infos[agent]:
            raise RuntimeError(f"Agent {agent} has no valid observation or action mask")
                
        observation = env.observe(agent, new=True)[agent]
        action_mask = env.infos[agent]['action_mask']
        
        # Verify action mask is valid
        if sum(action_mask) == 0:
            raise RuntimeError(f"Agent {agent} has no valid actions according to mask")
        
        # Get action based on opponent type
        if hasattr(opponent_model, 'play_turn'):  # Hardcoded agent
            action = opponent_model.play_turn(observation, action_mask, table_card=env.table_card)
            
            # Verify action is valid
            if action_mask[action] != 1:
                raise RuntimeError(f"Hardcoded agent {agent} returned invalid action {action}")
                
            return action
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
                    # Try with 3-return signature first
                    probs, _, _ = opponent_model(observation_tensor, None)
                except ValueError:
                    # Try with 2-return signature
                    probs, _ = opponent_model(observation_tensor, None)
                    
            # Apply action mask
            probs = probs.squeeze().cpu().numpy()
            masked_probs = probs * action_mask
            masked_probs_sum = masked_probs.sum()
            
            # Check if any probability mass remains after masking
            if masked_probs_sum == 0:
                raise RuntimeError(f"Model for {agent} produced no valid action probability mass")
                
            # Normalize
            masked_probs /= masked_probs_sum
            
            # Sample from distribution
            action = np.random.choice(len(masked_probs), p=masked_probs)
            return action
    
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
    
    def _should_challenge(self, env, is_simulation=False):
        """
        Determines whether to challenge the last play based on explicit rules
        about what constitutes a bluff.
        
        Args:
            env: The environment
            is_simulation: Whether this check is during simulation (affects logging)
            
        Returns:
            bool: True if a challenge is recommended, False otherwise
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
        
        # CRITICAL: First check if the opponent's move was ACTUALLY a bluff
        # This is a direct check of the cards played
        is_bluff = False
        for card in played_cards:
            if card != table_card and card != "Joker":
                is_bluff = True
                break
                
        # If not a bluff, we should NEVER challenge
        if not is_bluff:
            if not is_simulation:
                self._log(f"Last play by {last_agent} was not a bluff, will not challenge")
            return False
            
        # If it IS a bluff, we should challenge
        if not is_simulation:
            self._log(f"Detected bluff by {last_agent}, recommending challenge")
        return True
    
    def _simulate_single_opponent_action(self, sim_env, action_sequence):
        """
        Simulate a single opponent action and add it to the action sequence.
        This version does not silently recover from errors.
        
        Args:
            sim_env: Simulation environment.
            action_sequence: Current action sequence (list of (agent, action) tuples).
            
        Returns:
            True if the action was simulated; otherwise, an exception is raised.
        """
        current_agent = sim_env.agent_selection
        if current_agent is None or current_agent == self.training_agent:
            raise RuntimeError("Simulation error: Expected an opponent agent but got None or the training agent.")
        
        opponent_action = self._select_opponent_action(sim_env, current_agent)
        action_sequence.append((current_agent, opponent_action))
        sim_env.step(opponent_action)
        return True


    def simulate_game(self, env_state, action):
        """
        Recursively simulate the game starting from the given state and action until the round ends.
        
        After executing the given action, we simulate opponent moves until our training agent's turn is reached.
        If the training agent's hand size is neither 0 nor 5 (i.e. round has not ended), then we recursively
        simulate all valid moves from that state and choose the best outcome.
        
        Args:
            env_state: The current environment state.
            action: The action to simulate.
            
        Returns:
            tuple: (total_reward, full_action_sequence, is_terminal)
                total_reward: Outcome value at round end.
                full_action_sequence: List of (agent, action) tuples for replay.
                is_terminal: Boolean indicating whether the game terminated.
        """
        # Clone environment and set state.
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)
        
        if sim_env.agent_selection != self.training_agent:
            raise RuntimeError(f"simulate_game() expected training agent's turn but got {sim_env.agent_selection}.")
        
        # Start the action sequence with the provided action.
        action_sequence = [(self.training_agent, action)]
        action_type, card_category, count = decode_action(action)
        is_bluffing = False
        if action_type == "Play":
            if card_category == "non-table":
                is_bluffing = True
            elif card_category == "table":
                table_card = sim_env.table_card
                hand = sim_env.players_hands.get(self.training_agent, [])
                table_cards = sum(1 for card in hand if card == table_card or card == "Joker")
                is_bluffing = (count > table_cards)
        
        sim_env.step(action)
        self._log(f"After action {action}: training agent hand size = {len(sim_env.players_hands.get(self.training_agent, []))}")
        
        # Continue simulation until either:
        # 1. The game terminates (no active agent), or
        # 2. It's our turn AND our hand size is either 0 or 5 (i.e. round ended).
        max_steps = 100  # safeguard against infinite loops
        step = 0
        while sim_env.agent_selection is not None and sim_env.agent_selection != self.training_agent:
            step += 1
            if step > max_steps:
                raise RuntimeError("simulate_game() exceeded maximum simulation steps without round ending.")
            
            current_agent = sim_env.agent_selection
            self._log(f"Simulating opponent action for agent {current_agent}; "
                    f"training agent hand size = {len(sim_env.players_hands.get(self.training_agent, []))}, "
                    f"{current_agent} hand size = {len(sim_env.players_hands.get(current_agent, []))}")
            self._simulate_single_opponent_action(sim_env, action_sequence)
        
        # Now, either the game is over or it's our turn.
        training_hand_size = len(sim_env.players_hands.get(self.training_agent, []))
        if sim_env.agent_selection == self.training_agent and training_hand_size not in {0, 5}:
            self._log(f"Returned to training agent with hand size {training_hand_size} (round not ended): Recursing for next move.")
            sim_env.observe(self.training_agent, new=True)
            if self.training_agent not in sim_env.infos or "action_mask" not in sim_env.infos[self.training_agent]:
                raise RuntimeError(f"No valid action mask available for {self.training_agent}")
            action_mask = sim_env.infos[self.training_agent]['action_mask']
            valid_moves = [i for i, mask in enumerate(action_mask) if mask == 1]
            if not valid_moves:
                raise RuntimeError("No valid moves available on training agent's turn during recursion.")
            
            best_future_value = -float('inf')
            best_future_sequence = []
            # Recursively simulate each valid move from the current state.
            for move in valid_moves:
                future_value, future_sequence, is_terminal = self.simulate_game(sim_env.get_state(), move)
                self._log(f"Recursive move {move} yields value {future_value} with sequence length {len(future_sequence)}")
                if future_value > best_future_value:
                    best_future_value = future_value
                    best_future_sequence = future_sequence
            full_sequence = action_sequence + best_future_sequence
            total_value = best_future_value
            return total_value, full_sequence, (sim_env.agent_selection is None)
        else:
            # Simulation ends because either the game terminated or the round ended.
            if sim_env.agent_selection is None:
                self._log("Simulation ended because no active agent remains (terminal state).")
            elif training_hand_size in {0, 5}:
                self._log(f"Simulation completed: training agent hand size = {training_hand_size} (round ended).")
            reward = self.evaluate_terminal_state(sim_env.get_state())
            return reward, action_sequence, (sim_env.agent_selection is None)
    
    def search(self, env_state):
        """
        Enhanced search that evaluates each valid action from the current state by recursively simulating until the round ends.
        
        Raises an error if no valid action is found.
        
        Args:
            env_state: The environment state to start search from.
            
        Returns:
            tuple: (action_probs, best_action, best_value)
        """
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)
        
        if sim_env.agent_selection != self.training_agent:
            raise RuntimeError(f"Cannot simulate game when it's not our turn. Current agent: {sim_env.agent_selection}")
        
        sim_env.observe(self.training_agent, new=True)
        if self.training_agent not in sim_env.infos or "action_mask" not in sim_env.infos[self.training_agent]:
            raise RuntimeError(f"No valid action mask available for {self.training_agent}")
        
        action_mask = sim_env.infos[self.training_agent]['action_mask']
        valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
        
        if not valid_actions:
            raise RuntimeError(f"No valid actions available for {self.training_agent}")
        
        challenge_action = 6  # Challenge action index.
        if challenge_action in valid_actions and not self._should_challenge(sim_env):
            valid_actions.remove(challenge_action)
            self._log("Removed challenge action as opponent is not bluffing")
        
        if not valid_actions:
            raise RuntimeError("No valid actions available after filtering for the training agent.")
        
        safe_actions = []
        for action in valid_actions:
            action_type, card_category, count = decode_action(action)
            if action_type == "Play" and card_category == "table":
                table_card = sim_env.table_card
                hand = sim_env.players_hands.get(self.training_agent, [])
                table_cards = sum(1 for card in hand if card == table_card or card == "Joker")
                if count <= table_cards:
                    safe_actions.append(action)
        
        best_action = None
        best_value = -float('inf')
        best_sequence = []
        
        for action in valid_actions:
            value, action_sequence, is_terminal = self.simulate_game(env_state, action)
            action_type, card_category, count = decode_action(action)
            self._log(f"Action {action} ({action_type}, {card_category}, {count}): value={value}, seq_len={len(action_sequence)}")
            if value < -1.5:
                self._log(f"Rejecting action {action} with very negative value {value}")
                continue
            if action in safe_actions:
                value += 0.1  # Boost for safe actions.
            if value > best_value:
                best_action = action
                best_value = value
                best_sequence = action_sequence
                self._log(f"New best action: {action} with value {value}")
        
        if best_action is None:
            raise RuntimeError("No valid action found after evaluating all valid actions in MCTS search.")
        
        self.action_sequence = best_sequence
        self._log(f"Final selected action: {best_action} with value {best_value}")
        self._log(f"Action sequence ({len(best_sequence)} steps):")
        for i, (agent, seq_action) in enumerate(best_sequence):
            action_type, card_category, count = decode_action(seq_action)
            self._log(f"  {i+1}. {agent}: {action_type}, {card_category}, {count}")
        
        action_dim = sim_env.action_spaces[self.training_agent].n
        action_probs = np.zeros(action_dim)
        action_probs[best_action] = 1.0
        
        return action_probs, best_action, best_value
    
    def get_next_opponent_action(self, agent):
        """
        Get the next action for the specified opponent agent from the recorded sequence.
        
        Args:
            agent: The opponent agent name
            
        Returns:
            action: The next action for this opponent, or None if no action is found
        """
        # Look for the next action for this agent in the sequence
        for i, (seq_agent, action) in enumerate(self.action_sequence):
            if seq_agent == agent:
                # Found an action for this agent, remove it from the sequence
                self.action_sequence.pop(i)
                self._log(f"Found action for {agent}: {action}")
                return action
        
        # If no action found, return None instead of raising an error
        # This allows handling of new rounds where opponents might go first
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
            if true_label not in self.opponent_labels:
                raise ValueError(f"Unknown opponent label: {true_label}")
            true_label = self.opponent_labels[true_label]
            
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
        max_steps = 150
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
                    raise RuntimeError(f"No valid actions for {current_agent} during round simulation")
                    
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
        
        # Check if we hit the step limit
        if step_count >= max_steps:
            self._log(f"Warning: Round simulation reached max steps ({max_steps})")
            
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