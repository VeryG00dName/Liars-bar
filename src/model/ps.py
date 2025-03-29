# src/model/ps.py
import numpy as np
import torch
from src import config
from src.env.liars_deck_env_utils_2 import decode_action, select_cards_to_play, validate_claim

class PerfectSearch:
    """
    Perfect Search algorithm for Liar's Deck with exact opponent model knowledge.
    
    This implementation finds guaranteed winning paths by simulating the game tree
    with perfect information of all players' hands and exact opponent models.
    """
    
    def __init__(self, env, training_agent, opponent_models):
        """
        Initialize the Perfect Search algorithm.
        
        Args:
            env: The environment instance (will be cloned for simulation)
            training_agent: Name of the agent being trained (e.g., 'player_0')
            opponent_models: Dictionary mapping agent names to their model instances
        """
        self.base_env = env
        self.training_agent = training_agent
        self.opponent_models = opponent_models
        
        # Get opponent agent names
        self.opponent_agents = [agent for agent in env.possible_agents if agent != training_agent]
        
        # Store the complete action sequence for all agents (agent, action)
        self.action_sequence = []
        
        # Track the next position in the action sequence
        self.sequence_position = 0
        
        # Track if our agent has searched and found a winning path
        self.has_winning_path = False
        
        # Debug flag for verbose logging
        self.debug = False
    
    def _log(self, message):
        """Log a message if debug is enabled."""
        if self.debug:
            print(f"PS DEBUG: {message}")
    
    def _select_opponent_action(self, env, agent):
        """
        Use the exact opponent model to select an action.
        
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
            
            # Deterministically select the highest probability action
            action = np.argmax(masked_probs)
            return action
    
    def _should_challenge(self, env):
        """
        Determines whether to challenge the last play based on known game state.
        
        Args:
            env: The environment
            
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
        
        # Check if the opponent's move was ACTUALLY a bluff
        is_bluff = False
        for card in played_cards:
            if card != table_card and card != "Joker":
                is_bluff = True
                break
                
        # If it's a bluff, we should challenge
        return is_bluff
    
    def simulate_game(self, env_state, action):
        """
        Simulates a complete game starting from a state and initial action.
        Terminates early if our agent receives a penalty.
        
        Args:
            env_state: The current environment state
            action: The action to simulate
            
        Returns:
            tuple: (outcome_value, action_sequence, is_terminal)
        """
        # Clone environment and set state
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)
        
        if sim_env.agent_selection != self.training_agent:
            raise RuntimeError(f"Expected training agent's turn but got {sim_env.agent_selection}")
        
        # Start with our action
        action_sequence = [(self.training_agent, action)]
        
        # Get our starting penalty count to detect if we receive one
        starting_penalty = sim_env.penalties.get(self.training_agent, 0)
        
        # Take the action
        sim_env.step(action)
        
        # Check for penalty (immediate termination if we got one)
        if sim_env.penalties.get(self.training_agent, 0) > starting_penalty:
            return -100.0, action_sequence, False
        
        # Maximum steps to prevent infinite loops
        max_steps = 150
        
        # Continue simulation until it's our turn again or game ends
        for _ in range(max_steps):
            # If game is over, evaluate and return
            if sim_env.agent_selection is None:
                return self._evaluate_terminal_state(sim_env), action_sequence, True
            
            # If it's our turn again, but with a new hand (end of round)
            # we'll recursively search for the best action
            if sim_env.agent_selection == self.training_agent:
                hand_size = len(sim_env.players_hands.get(self.training_agent, []))
                
                # If we have a fresh hand (5 cards) or empty hand (0 cards), it's a new round
                if hand_size in (0, 5):
                    # Get valid actions for our turn
                    sim_env.observe(self.training_agent, new=True)
                    action_mask = sim_env.infos[self.training_agent].get('action_mask', [0] * 7)
                    valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                    
                    # Find the best action from this new state
                    best_val = -float('inf')
                    best_seq = []
                    best_is_terminal = False
                    
                    for next_action in valid_actions:
                        val, seq, is_terminal = self.simulate_game(sim_env.get_state(), next_action)
                        if val > best_val:
                            best_val = val
                            best_seq = seq
                            best_is_terminal = is_terminal
                    
                    # Return the combined sequence and value
                    return best_val, action_sequence + best_seq, best_is_terminal
                
                # If it's mid-round, continue playing our turn
                sim_env.observe(self.training_agent, new=True)
                action_mask = sim_env.infos[self.training_agent].get('action_mask', [0] * 7)
                valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                
                if not valid_actions:
                    return -1.0, action_sequence, False
                
                # Get our current penalty to check if we receive one
                current_penalty = sim_env.penalties.get(self.training_agent, 0)
                
                # If we can challenge successfully, do it
                if 6 in valid_actions and self._should_challenge(sim_env):
                    next_action = 6
                else:
                    # Otherwise play minimum valid cards
                    play_actions = [a for a in valid_actions if a < 6]
                    if play_actions:
                        # Sort by play count (lower is better)
                        next_action = min(play_actions, key=lambda a: (a % 3) + 1)
                    else:
                        next_action = valid_actions[0]
                
                # Take our action and add to sequence
                sim_env.step(next_action)
                action_sequence.append((self.training_agent, next_action))
                
                # Check if we got a penalty - immediately terminate if so
                if sim_env.penalties.get(self.training_agent, 0) > current_penalty:
                    return -100.0, action_sequence, False
                
            # If it's an opponent's turn, use their model
            else:
                current_agent = sim_env.agent_selection
                try:
                    # Get the opponent's action
                    opponent_action = self._select_opponent_action(sim_env, current_agent)
                    
                    # Take the action and record it
                    sim_env.step(opponent_action)
                    action_sequence.append((current_agent, opponent_action))
                    
                except Exception as e:
                    self._log(f"Error simulating opponent {current_agent}: {e}")
                    return -1.0, action_sequence, False
        
        # If we reach here, we hit the step limit
        return self._evaluate_state(sim_env), action_sequence, False
    
    def _evaluate_terminal_state(self, env):
        """
        Evaluate a terminal state to determine its value.
        
        Args:
            env: Environment in the terminal state
            
        Returns:
            float: Value of the state (positive if we win, negative if we lose)
        """
        # Check if game is over and who won
        if env.winner:
            if env.winner == self.training_agent:
                return 100.0  # We won (high value to prioritize winning paths)
            else:
                return -100.0  # We lost
        
        # Check penalty counts
        our_penalty = env.penalties.get(self.training_agent, 0)
        opponent_penalties = {opp: env.penalties.get(opp, 0) for opp in self.opponent_agents}
        
        # Check if we're in a better position (fewer penalties)
        max_opponent_penalty = max(opponent_penalties.values()) if opponent_penalties else 0
        if our_penalty < max_opponent_penalty:
            return 10.0
        elif our_penalty > max_opponent_penalty:
            return -10.0
        
        # Neutral value for unclear situations
        return 0.0
    
    def _evaluate_state(self, env):
        """
        Evaluates a non-terminal game state.
        
        Args:
            env: The environment to evaluate.
            
        Returns:
            float: Value of the state.
        """
        # If the game is over, use terminal state evaluation
        if env.agent_selection is None:
            return self._evaluate_terminal_state(env)
        
        # Evaluate current state based on various factors
        our_hand = env.players_hands.get(self.training_agent, [])
        our_penalty = env.penalties.get(self.training_agent, 0)
        
        # Calculate opponent penalties 
        opponent_penalties = {opp: env.penalties.get(opp, 0) for opp in self.opponent_agents}
        max_opponent_penalty = max(opponent_penalties.values()) if opponent_penalties else 0
        
        # Calculate hand sizes
        our_hand_size = len(our_hand)
        opponent_hand_sizes = {opp: len(env.players_hands.get(opp, [])) for opp in self.opponent_agents}
        
        # Calculate table cards in our hand
        table_card = env.table_card
        table_cards_count = sum(1 for card in our_hand if card == table_card or card == "Joker")
        
        # Penalty factor: prefer states where our penalty is low and opponents' are high
        penalty_factor = (max_opponent_penalty - our_penalty) * 5.0  # Higher weight on penalties
        
        # Hand size factor: prefer smaller hands
        hand_size_factor = (5.0 - our_hand_size) / 5.0  # Range: [0, 1]
        
        # Table cards factor: more table cards gives us more safe play options
        table_cards_factor = table_cards_count / max(1, our_hand_size)  # Range: [0, 1]
        
        # Weighted combination of factors
        score = penalty_factor + (hand_size_factor * 0.5) + (table_cards_factor * 0.5)
        
        return score
    
    def get_next_agent_action(self, agent):
        """
        Get the next action for any agent (including our agent) from the cached sequence.
        
        Args:
            agent: The agent name (can be training_agent or an opponent)
            
        Returns:
            action: The next action for this agent, or None if no action is found
        """
        # If we don't have a winning path yet, return None
        if not self.has_winning_path:
            return None
            
        # Check if we've reached the end of the sequence
        if self.sequence_position >= len(self.action_sequence):
            return None
            
        # Look for the next action for this agent starting from current position
        for i in range(self.sequence_position, len(self.action_sequence)):
            seq_agent, action = self.action_sequence[i]
            if seq_agent == agent:
                # Found an action for this agent
                self.sequence_position = i + 1  # Move past this action
                self._log(f"Found cached action for {agent}: {action} at position {i}")
                return action
        
        # If no action found, return None
        self._log(f"No cached action found for {agent} in remaining sequence")
        return None
    
    def search(self, env_state):
        """
        Searches for the best action by simulating complete games from the current state.
        
        Args:
            env_state: The environment state to start search from.
            
        Returns:
            tuple: (action_probs, best_action, best_value)
        """
        # Check if we already have a winning path
        if self.has_winning_path:
            # Get the next action for our agent from the cached sequence
            next_action = self.get_next_agent_action(self.training_agent)
            if next_action is not None:
                action_dim = 7  # Default action dimension
                action_probs = np.zeros(action_dim)
                action_probs[next_action] = 1.0
                return action_probs, next_action, 100.0
                
            # If we can't find an action, we need to reset and search again
            self._log("Cached sequence exhausted or invalid, performing new search")
            self.has_winning_path = False
            self.sequence_position = 0
        
        # Clone environment and set state
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)
        
        if sim_env.agent_selection != self.training_agent:
            raise RuntimeError(f"Cannot search when it's not our turn. Current agent: {sim_env.agent_selection}")
        
        # Get valid actions
        sim_env.observe(self.training_agent, new=True)
        if self.training_agent not in sim_env.infos or "action_mask" not in sim_env.infos[self.training_agent]:
            raise RuntimeError(f"No valid action mask available for {self.training_agent}")
        
        action_mask = sim_env.infos[self.training_agent]['action_mask']
        valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
        
        if not valid_actions:
            raise RuntimeError(f"No valid actions available for {self.training_agent}")
        
        # If we can challenge and should, prioritize that
        challenge_action = 6  # Challenge action index
        if challenge_action in valid_actions and self._should_challenge(sim_env):
            valid_actions = [challenge_action]  # Only consider challenging
            self._log("Prioritizing challenge action as opponent is bluffing")
        
        # Identify safe play actions (playing table cards we actually have)
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
        best_value = float('-inf')
        best_sequence = []
        best_terminal = False
        
        # First evaluate safe actions, then others
        # This ordering helps us find winning paths faster
        actions_to_try = safe_actions + [a for a in valid_actions if a not in safe_actions]
        
        for action in actions_to_try:
            value, sequence, is_terminal = self.simulate_game(env_state, action)
            
            action_type, card_category, count = decode_action(action)
            self._log(f"Action {action} ({action_type}, {card_category}, {count}): value={value}, terminal={is_terminal}")
            
            # Skip very negative values (e.g., from penalty paths)
            if value < -50:
                self._log(f"Skipping action {action} with very negative value {value}")
                continue
                
            # Prioritize terminal wins
            if is_terminal and value > 0:
                best_action = action
                best_value = value
                best_sequence = sequence
                best_terminal = True
                self._log(f"Found guaranteed win with action {action}")
                break
                
            # If no terminal win yet, track best action
            if not best_terminal and value > best_value:
                best_action = action
                best_value = value
                best_sequence = sequence
                best_terminal = is_terminal
                self._log(f"New best action: {action} with value {value}")
        
        if best_action is None:
            raise RuntimeError("No valid action found that doesn't lead to a penalty. No winning path exists.")
        
        # Store the best action sequence for future reference
        self.action_sequence = best_sequence
        self.sequence_position = 0  # Reset position to start of sequence
        
        # If we found a terminal winning path, mark it
        if best_terminal and best_value > 0:
            self.has_winning_path = True
            self._log(f"Found winning path with {len(best_sequence)} actions")
        
        # Build action probability vector
        action_dim = sim_env.action_spaces[self.training_agent].n
        action_probs = np.zeros(action_dim)
        action_probs[best_action] = 1.0
        
        return action_probs, best_action, best_value