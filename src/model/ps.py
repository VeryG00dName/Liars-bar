import numpy as np
from src.env.liars_deck_env_utils_2 import decode_action

class PerfectSearch:
    """
    Simplified Perfect Search algorithm for Liar's Deck.
    Focuses on finding paths where opponents get penalties.
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
        
        # Store the action sequence for the current round
        self.action_sequence = []
        
        # Track the next position in the action sequence
        self.sequence_position = 0
        
        # Track simulation statistics
        self.simulations_performed = 0
        
        # Cache for opponent actions to ensure consistency across runs and tests
        self.current_opponent_action_cache = {}
        
        # Debug flag for verbose logging
        self.debug = True
    
    def _log(self, message):
        """Log a message if debug is enabled."""
        if self.debug:
            print(f"PS DEBUG: {message}")
    
    def _select_opponent_action(self, env, agent):
        """
        Use the opponent model to select an action.
        
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
            import numpy as np
            from src import config
            obp_placeholder = np.zeros(2, dtype=np.float32)
            memory_placeholder = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
            final_obs = np.concatenate([old_observation, obp_placeholder, memory_placeholder], axis=0)
            
            # Convert to tensor
            import torch
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
    
    def simulate_round(self, env_state, action, opponent_action_cache=None):
        """
        Simulates until the end of the current round or until someone gets a penalty.
        Always continues until round change to ensure complete action sequences.
        Uses opponent_action_cache to ensure consistent opponent behavior.
        
        Args:
            env_state: The current environment state
            action: The action to simulate
            opponent_action_cache: Dictionary to cache opponent actions based on observations
            
        Returns:
            tuple: (outcome_value, action_sequence, is_terminal, is_new_round)
        """
        # Initialize opponent action cache if not provided
        if opponent_action_cache is None:
            opponent_action_cache = {}
            
        self.simulations_performed += 1
        
        # Clone environment and set state
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)
        
        # Record our starting state information
        starting_penalty = sim_env.penalties.get(self.training_agent, 0)
        starting_round = sim_env.round
        
        self._log(f"Starting simulation in round {starting_round} with action {action}")
        
        # Check for opportunity to challenge an opponent bluff
        should_challenge = False
        if action == 6:  # Challenge action
            last_agent = sim_env.last_action_agent
            if last_agent:
                played_cards = sim_env.last_played_cards.get(last_agent, [])
                table_card = sim_env.table_card
                is_bluff = any(card != table_card and card != "Joker" for card in played_cards)
                if is_bluff:
                    should_challenge = True
                    self._log(f"Found real bluff to challenge from {last_agent}")
        
        # Start with our action
        action_sequence = [(self.training_agent, action)]
        
        # Take the action
        sim_env.step(action)
        
        # Track if we found any opponent penalties
        found_opponent_penalty = False
        found_our_penalty = False
        
        # Get information after our action
        new_penalty = sim_env.penalties.get(self.training_agent, 0)
        penalty_increase = new_penalty - starting_penalty
        
        # If we got a penalty and it wasn't from a successful challenge, this is a very bad path
        if penalty_increase > 0 and not should_challenge:
            self._log(f"We got a penalty, but continuing simulation")
            penalty_value = -1000.0 * penalty_increase
            # If this was from a failed challenge, it's extra bad with high penalties
            if action == 6 and starting_penalty >= 2:
                penalty_value = -5000.0
            # Continue simulation but track that our penalty increased
            found_our_penalty = True
        
        # Check if we can stop here because the round changed already
        if sim_env.round > starting_round:
            success_value = 100.0 if not found_our_penalty else -1000.0
            self._log(f"Round changed after our action, returning {'positive' if not found_our_penalty else 'negative'} value")
            return success_value, action_sequence, False, True
        
        # Maximum steps to prevent infinite loops
        max_steps = 100
        step_count = 0
        
        # Continue simulation until game ends or round changes
        # IMPORTANT: We always continue until the round changes to ensure complete sequences
        while step_count < max_steps:
            step_count += 1
            
            # If game is over, evaluate and return
            if sim_env.agent_selection is None:
                winner = sim_env.winner
                if winner == self.training_agent:
                    self._log(f"Game ended after {step_count} steps - WE WIN!")
                    return 5000.0, action_sequence, True, False
                else:
                    self._log(f"Game ended after {step_count} steps - WE LOSE")
                    return -5000.0, action_sequence, True, False
            
            # Check if round changed
            if sim_env.round > starting_round:
                if found_our_penalty:
                    self._log(f"Round changed but we got a penalty earlier, returning negative value")
                    return -1000.0, action_sequence, False, True
                elif found_opponent_penalty:
                    self._log(f"Round changed and found opponent penalty, returning positive value")
                    return 2000.0, action_sequence, False, True
                else:
                    self._log(f"Round changed normally, returning neutral value")
                    return 100.0, action_sequence, False, True
            
            # If it's our turn
            if sim_env.agent_selection == self.training_agent:
                # Get valid actions
                sim_env.observe(self.training_agent, new=True)
                action_mask = sim_env.infos[self.training_agent].get('action_mask', [0] * 7)
                valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                
                if not valid_actions:
                    self._log("No valid actions available")
                    return -50.0, action_sequence, False, False
                
                # First check if we can challenge a bluff
                if 6 in valid_actions:
                    last_agent = sim_env.last_action_agent
                    if last_agent:
                        played_cards = sim_env.last_played_cards.get(last_agent, [])
                        table_card = sim_env.table_card
                        is_bluff = any(card != table_card and card != "Joker" for card in played_cards)
                        if is_bluff:
                            self._log(f"Found opponent bluff during simulation, challenging")
                            next_action = 6
                            # Take our action and add to sequence
                            old_penalty = sim_env.penalties.get(self.training_agent, 0)
                            sim_env.step(next_action)
                            action_sequence.append((self.training_agent, next_action))
                            
                            # Check opponent's penalty after challenge
                            new_opp_penalty = sim_env.penalties.get(last_agent, 0)
                            old_opp_penalty = sim_env.penalties.get(last_agent, 0) - 1  # Approx
                            if new_opp_penalty > old_opp_penalty:
                                found_opponent_penalty = True
                                self._log(f"Successful challenge - opponent penalty increased")
                            
                            # Check if round changed after challenge
                            if sim_env.round > starting_round:
                                if found_opponent_penalty:
                                    self._log(f"Round changed after successful challenge, returning very positive value")
                                    return 2000.0, action_sequence, False, True
                                else:
                                    self._log(f"Round changed after challenge, returning positive value")
                                    return 500.0, action_sequence, False, True
                            
                            # Also check if we got a penalty (failed challenge)
                            new_our_penalty = sim_env.penalties.get(self.training_agent, 0)
                            if new_our_penalty > old_penalty:
                                found_our_penalty = True
                                self._log(f"Failed challenge - our penalty increased")
                            
                            continue
                
                # If we have table cards and high penalties, play those
                hand = sim_env.players_hands.get(self.training_agent, [])
                table_card = sim_env.table_card
                table_cards = [c for c in hand if c == table_card or c == "Joker"]
                current_penalty = sim_env.penalties.get(self.training_agent, 0)
                
                # Smart decision logic - when at high penalties, prioritize table cards
                if current_penalty >= 2 and table_cards:
                    # Play 1 table card (action 0)
                    if 0 in valid_actions:
                        next_action = 0
                    elif 1 in valid_actions:
                        next_action = 1  # Play 2 table cards
                    elif 2 in valid_actions:
                        next_action = 2  # Play 3 table cards
                    else:
                        # Play the action with the lowest count
                        next_action = min(valid_actions, key=lambda a: (a % 3) + 1)
                    self._log(f"High penalties, playing table card: action {next_action}")
                
                # If we have few cards left, see if we should play table cards before challenging
                elif len(hand) <= 3 and found_opponent_penalty == False:
                    # Hold our table cards until right before challenging 
                    if 0 in valid_actions and len(table_cards) > 0:
                        next_action = 0  # Play 1 table card
                        self._log(f"Playing 1 table card before challenging")
                    # Play 1 non-table card
                    elif 3 in valid_actions and len(hand) > len(table_cards):
                        next_action = 3  # Play 1 non-table card
                        self._log(f"Playing 1 non-table card")
                    else:
                        # Play the action with the lowest count
                        play_actions = [a for a in valid_actions if a < 6]
                        if play_actions:
                            next_action = min(play_actions, key=lambda a: (a % 3) + 1)
                        else:
                            next_action = valid_actions[0]
                        self._log(f"Default play: action {next_action}")
                
                # Otherwise, try to play cards strategically
                else:
                    # If we have both table and non-table cards, alternate
                    if len(table_cards) > 0 and len(hand) > len(table_cards):
                        # Look at last action to decide
                        if action_sequence[-1][1] < 3:  # If last action was table card play
                            if 3 in valid_actions:
                                next_action = 3  # Play 1 non-table card
                                self._log(f"Alternating to non-table card")
                            else:
                                next_action = min(valid_actions, key=lambda a: (a % 3) + 1)
                        else:  # If last action was non-table card play
                            if 0 in valid_actions:
                                next_action = 0  # Play 1 table card
                                self._log(f"Alternating to table card")
                            else:
                                next_action = min(valid_actions, key=lambda a: (a % 3) + 1)
                    else:
                        # Play the action with the lowest count
                        play_actions = [a for a in valid_actions if a < 6]
                        if play_actions:
                            next_action = min(play_actions, key=lambda a: (a % 3) + 1)
                        else:
                            next_action = valid_actions[0]
                        self._log(f"Default play: action {next_action}")
                
                self._log(f"Our turn - selected action {next_action}")
                
                # Get the penalty before taking action
                old_penalty = sim_env.penalties.get(self.training_agent, 0)
                
                # Take our action and add to sequence
                sim_env.step(next_action)
                action_sequence.append((self.training_agent, next_action))
                
                # Check if round changed
                if sim_env.round > starting_round:
                    if found_our_penalty:
                        self._log(f"Round changed but we got a penalty earlier, returning negative value")
                        return -1000.0, action_sequence, False, True
                    elif found_opponent_penalty:
                        self._log(f"Round changed and found opponent penalty, returning positive value")
                        return 2000.0, action_sequence, False, True
                
                # Check if we got a penalty
                new_penalty = sim_env.penalties.get(self.training_agent, 0)
                if new_penalty > old_penalty:
                    found_our_penalty = True
                    self._log(f"We got a penalty, but continuing simulation to end of round")
            
            # If it's an opponent's turn
            else:
                current_agent = sim_env.agent_selection
                self._log(f"Opponent {current_agent}'s turn")
                
                try:
                    # Generate a hash key for the opponent's observation
                    # The key should reflect the state as the opponent sees it
                    sim_env.observe(current_agent, new=True)
                    
                    # Key components:
                    # 1. Opponent's hand (sorted)
                    hand = sorted(sim_env.players_hands.get(current_agent, []))
                    # 2. Table card
                    table_card = sim_env.table_card
                    # 3. Last action & agent
                    last_action = sim_env.last_action
                    last_agent = sim_env.last_action_agent
                    # 4. Cards played by last agent
                    cards_played = []
                    if last_agent:
                        cards_played = sorted(sim_env.last_played_cards.get(last_agent, []))
                    
                    # Create a consistent key for this observation
                    obs_key = (
                        current_agent,
                        tuple(hand),
                        table_card,
                        last_action,
                        last_agent,
                        tuple(cards_played)
                    )
                    
                    # Check if we've already determined an action for this observation
                    if obs_key in opponent_action_cache:
                        opponent_action = opponent_action_cache[obs_key]
                        self._log(f"Using cached action {opponent_action} for opponent {current_agent}")
                    else:
                        # Get the opponent's action
                        opponent_action = self._select_opponent_action(sim_env, current_agent)
                        # Cache it for future use
                        opponent_action_cache[obs_key] = opponent_action
                        self._log(f"Caching action {opponent_action} for opponent {current_agent}")
                    
                    # Get the opponent's penalty before action
                    old_penalty = sim_env.penalties.get(current_agent, 0)
                    
                    # Take the action and record it
                    sim_env.step(opponent_action)
                    action_sequence.append((current_agent, opponent_action))
                    
                    # Check if opponent got a penalty - this is good for us
                    new_penalty = sim_env.penalties.get(current_agent, 0)
                    if new_penalty > old_penalty:
                        found_opponent_penalty = True
                        self._log(f"Opponent got a penalty, but continuing to end of round")
                    
                    # Check if round changed
                    if sim_env.round > starting_round:
                        if found_our_penalty:
                            self._log(f"Round changed but we got a penalty earlier, returning negative value")
                            return -1000.0, action_sequence, False, True
                        elif found_opponent_penalty:
                            self._log(f"Round changed and found opponent penalty, returning positive value")
                            return 2000.0, action_sequence, False, True
                    
                except Exception as e:
                    self._log(f"Error with opponent action: {e}")
                    return -50.0, action_sequence, False, False
        
        # If we reach here, we hit the step limit
        self._log(f"Hit step limit ({max_steps}) - sequence length: {len(action_sequence)}")
        if found_our_penalty:
            return -1000.0, action_sequence, False, False
        elif found_opponent_penalty:
            return 1000.0, action_sequence, False, False
        else:
            return 10.0, action_sequence, False, False
    
    def _evaluate_state(self, env):
        """
        Evaluates a non-terminal game state, focusing on penalties.
        
        Args:
            env: The environment to evaluate.
            
        Returns:
            float: Value of the state.
        """
        # If the game is over, use terminal state evaluation
        if env.agent_selection is None:
            return self._evaluate_terminal_state(env)
        
        # Main evaluation is based on penalties
        our_penalty = env.penalties.get(self.training_agent, 0)
        opponent_penalties = [env.penalties.get(opp, 0) for opp in self.opponent_agents]
        max_opponent_penalty = max(opponent_penalties) if opponent_penalties else 0
        
        # Critical check: if we're eliminated, extremely bad
        if our_penalty >= 3:
            return -2000.0
        
        # Basic penalty-based score: positive if opponents have more penalties
        penalty_diff = max_opponent_penalty - our_penalty
        
        # Scale based on how close we are to elimination
        if our_penalty == 0:
            score = penalty_diff * 100
        elif our_penalty == 1:
            score = penalty_diff * 200
        else:  # our_penalty == 2
            score = penalty_diff * 500  # Much higher weight when at risk
        
        return score
    
    def _evaluate_terminal_state(self, env):
        """
        Evaluate a terminal state to determine its value.
        
        Args:
            env: Environment in the terminal state
            
        Returns:
            float: Value of the state (positive if we win, negative if we lose)
        """
        # Check if we won
        if env.winner == self.training_agent:
            return 2000.0  # We won - very high value
        else:
            return -2000.0  # We lost - very negative value
    
    def get_next_agent_action(self, agent):
        """
        Get the next action for any agent from the cached sequence.
        Validates the action against the current environment state.
        
        Args:
            agent: The agent name (can be training_agent or an opponent)
            
        Returns:
            action: The next action for this agent, or None if no action is found
        """
        # Check if we've reached the end of the sequence
        if self.sequence_position >= len(self.action_sequence):
            self._log(f"No more actions in cached sequence")
            return None
        
        # Look for the next action for this agent starting from current position
        for i in range(self.sequence_position, len(self.action_sequence)):
            seq_agent, action = self.action_sequence[i]
            if seq_agent == agent:
                # If it's our agent, validate the action is still valid
                if agent == self.training_agent:
                    # Check if the action is valid in the current environment
                    self.base_env.observe(agent, new=True)
                    action_mask = self.base_env.infos[agent].get('action_mask', [0] * 7)
                    if action_mask[action] != 1:
                        self._log(f"Cached action {action} for {agent} is no longer valid")
                        # Invalidate the entire sequence because our plan is broken
                        self.action_sequence = []
                        self.sequence_position = 0
                        return None
                    
                    # For challenge actions, verify the opponent is still bluffing
                    if action == 6:
                        last_agent = self.base_env.last_action_agent
                        if last_agent:
                            played_cards = self.base_env.last_played_cards.get(last_agent, [])
                            table_card = self.base_env.table_card
                            is_bluff = any(card != table_card and card != "Joker" for card in played_cards)
                            if not is_bluff:
                                self._log(f"Cached challenge is no longer valid - opponent not bluffing")
                                # Invalidate the sequence
                                self.action_sequence = []
                                self.sequence_position = 0
                                return None
                
                # Move past this action in the sequence
                self.sequence_position = i + 1
                self._log(f"Found cached action for {agent}: {action}")
                return action
        
        # If no action found, return None
        self._log(f"No cached action found for {agent}")
        return None
    
    def search(self, env_state):
        """
        Searches for the best action by simulating each valid action.
        Always simulates until round change for complete action sequences.
        
        Args:
            env_state: The environment state to start search from.
            
        Returns:
            tuple: (action_probs, best_action, best_value)
        """
        # Check if we already have a valid action sequence
        if self.sequence_position < len(self.action_sequence):
            # Get the next action for our agent from the cached sequence
            next_action = self.get_next_agent_action(self.training_agent)
            if next_action is not None:
                # Validate the action against the current mask
                sim_env = self.base_env.clone()
                sim_env.set_state(env_state)
                sim_env.observe(self.training_agent, new=True)
                action_mask = sim_env.infos[self.training_agent].get('action_mask', [0] * 7)
                
                if action_mask[next_action] == 1:
                    self._log(f"Using cached action: {next_action}")
                    action_dim = 7  # Default action dimension
                    action_probs = np.zeros(action_dim)
                    action_probs[next_action] = 1.0
                    return action_probs, next_action, 100.0
                else:
                    self._log(f"Cached action {next_action} is no longer valid")
        
        # Reset action sequence and position
        self.action_sequence = []
        self.sequence_position = 0
        self.simulations_performed = 0
        
        # Create opponent action cache to ensure consistent behavior
        opponent_action_cache = {}
        
        # Clone environment and set state
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)
        
        # Get valid actions
        sim_env.observe(self.training_agent, new=True)
        action_mask = sim_env.infos[self.training_agent].get('action_mask', [0] * 7)
        valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
        
        if not valid_actions:
            raise RuntimeError(f"No valid actions available for {self.training_agent}")
        
        # Get penalties and hand information for strategy
        current_penalty = sim_env.penalties.get(self.training_agent, 0)
        hand = sim_env.players_hands.get(self.training_agent, [])
        table_card = sim_env.table_card
        table_cards = [c for c in hand if c == table_card or c == "Joker"]
        
        # Try all valid actions and pick the best one
        best_action = None
        best_value = float('-inf')
        best_sequence = []
        
        # First, try challenging if available
        if 6 in valid_actions:
            last_agent = sim_env.last_action_agent
            if last_agent:
                played_cards = sim_env.last_played_cards.get(last_agent, [])
                is_bluff = any(card != table_card and card != "Joker" for card in played_cards)
                
                if is_bluff:
                    self._log("Found opponent bluff - trying challenge")
                    value, sequence, _, _ = self.simulate_round(env_state, 6, opponent_action_cache)
                    # Only use if value is positive (confirming this is good)
                    if value > 0:
                        self._log(f"Found successful challenge with value {value}")
                        self.action_sequence = sequence
                        self.sequence_position = 0
                        
                        # Store our cache in a class variable so test_all_opponents can use it
                        self.current_opponent_action_cache = opponent_action_cache
                        
                        return np.array([0, 0, 0, 0, 0, 0, 1.0]), 6, value
        
        # Second, always try playing table cards at high penalties
        if current_penalty >= 2 and 0 in valid_actions and len(table_cards) > 0:
            value, sequence, is_terminal, is_new_round = self.simulate_round(
                env_state, 0, opponent_action_cache
            )
            
            # If this seems good (positive value), use it immediately
            if value > 0:
                self._log(f"Found good table card play at high penalties")
                self.action_sequence = sequence
                self.sequence_position = 0
                # Store our cache in a class variable so test_all_opponents can use it
                self.current_opponent_action_cache = opponent_action_cache
                return np.array([1.0, 0, 0, 0, 0, 0, 0]), 0, value
            
            # Even if bad, this is likely our best option at high penalties
            best_action = 0
            best_value = value
            best_sequence = sequence
            self._log(f"Default to table card at high penalties: value={value}")
        
        # Try each action in priority order
        for action in valid_actions:
            
            # Simulate the action all the way to round change
            value, sequence, is_terminal, is_new_round = self.simulate_round(
                env_state, action, opponent_action_cache
            )
            self._log(f"Action {action}: value={value}, terminal={is_terminal}, new_round={is_new_round}, seq_len={len(sequence)}")
            
            # Always prefer actions that lead to winning
            if is_terminal and value > 0:
                best_action = action
                best_value = value
                best_sequence = sequence
                self._log(f"Found winning action: {action}")
                break
            
            # Highly prioritize actions where opponents get penalties
            if value > 1000:
                best_action = action
                best_value = value
                best_sequence = sequence
                self._log(f"Found action where opponent gets penalty: {action}")
                break
            
            # Update best action if better than current best
            if value > best_value:
                best_action = action
                best_value = value
                best_sequence = sequence
                self._log(f"New best action: {action} with value {value}")
        
        # If we couldn't find a good action, default to table cards if possible
        if (best_action is None or best_value < 0) and 0 in valid_actions and len(table_cards) > 0:
            self._log("No good actions found - defaulting to playing 1 table card")
            best_action = 0
            value, sequence, _, _ = self.simulate_round(env_state, best_action, opponent_action_cache)
            best_value = value
            best_sequence = sequence
        
        # If still nothing good, try one more fallback
        if (best_action is None or best_value < -1000):
            # Try action 3 (play 1 non-table card) if we have non-table cards
            if 3 in valid_actions and len(hand) > len(table_cards):
                self._log("Falling back to playing 1 non-table card")
                best_action = 3
                value, sequence, _, _ = self.simulate_round(env_state, best_action, opponent_action_cache)
                best_value = value
                best_sequence = sequence
            # Otherwise use the first valid action
            else:
                self._log("Using first valid action as fallback")
                best_action = valid_actions[0]
                value, sequence, _, _ = self.simulate_round(env_state, best_action, opponent_action_cache)
                best_value = value
                best_sequence = sequence
        
        # Store the best action sequence
        self.action_sequence = best_sequence
        self.sequence_position = 0
        
        # Store our cache in a class variable so test_all_opponents can use it
        self.current_opponent_action_cache = opponent_action_cache
        
        # Build action probability vector
        action_dim = 7
        action_probs = np.zeros(action_dim)
        action_probs[best_action] = 1.0
        
        self._log(f"Selected action {best_action} with value {best_value}")
        self._log(f"Stored sequence of length {len(best_sequence)}")
        
        return action_probs, best_action, best_value