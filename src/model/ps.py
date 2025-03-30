import numpy as np
from src.env.liars_deck_env_utils_2 import decode_action
import torch
from src import config
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
    
    def invalidate_plan(self):
        """Resets the cached action sequence because the game state has deviated."""
        if self.action_sequence: # Only log if there was a plan
            self._log("Plan invalidated due to deviation or new round.")
        self.action_sequence = []
        self.sequence_position = 0
        # Clear the opponent action cache tied to the old plan
        self.current_opponent_action_cache = {}
    
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
    
    def simulate_round(self, env_state, action, opponent_action_cache=None, depth=0, max_depth=20):
        """
        Recursively simulates possible action sequences from the initial action.
        Explores all valid actions on our turns up to max_depth.
        Uses opponent_action_cache to ensure consistent opponent behavior.
        
        Args:
            env_state: The current environment state
            action: The action to simulate
            opponent_action_cache: Dictionary to cache opponent actions based on observations
            depth: Current recursion depth
            max_depth: Maximum recursion depth for exploration
            
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
        starting_hand_size = len(sim_env.players_hands.get(self.training_agent, []))
        
        # Decode the action for better logging
        action_type, card_category, count = decode_action(action)
        self._log(f"[Depth {depth}] Simulating action {action} ({action_type}, {card_category}, {count})")
        
        # Start with our action
        action_sequence = [(self.training_agent, action)]

        # Get info before taking action
        pre_step_termination = sim_env.terminations.get(self.training_agent, False)
        pre_step_hand = sim_env.players_hands.get(self.training_agent, [])[:]  # Copy hand before action
        
        # Take the action
        sim_env.step(action)
        
        # Get info after the action for better debugging
        post_step_termination = sim_env.terminations.get(self.training_agent, False)
        post_step_penalty = sim_env.penalties.get(self.training_agent, 0)
        post_step_hand = sim_env.players_hands.get(self.training_agent, [])[:]  # Copy hand after action
        
        # Cards played in this action
        cards_played = [c for c in pre_step_hand if c not in post_step_hand]
        
        # Debug info about what happened
        self._log(f"[Depth {depth}] After action: Penalty={post_step_penalty}, Hand size={len(post_step_hand)}, Cards played={cards_played}")
        
        # Check if we got eliminated by our action
        if not pre_step_termination and post_step_termination:
            self._log(f"[Depth {depth}] Got eliminated by our own action! Very bad.")
            return -10000.0, action_sequence, True, False
        
        # Check if we got an immediate penalty
        if post_step_penalty > starting_penalty:
            self._log(f"[Depth {depth}] Got penalty right after our action.")
            if starting_penalty >= 2:
                # At risk of elimination
                return -5000.0, action_sequence, False, False
            else:
                return -1000.0, action_sequence, False, False
        
        # Check if game over after our action
        if sim_env.agent_selection is None:
            winner = sim_env.winner
            if winner == self.training_agent:
                self._log(f"[Depth {depth}] Game ended after our action - WE WIN!")
                return 5000.0, action_sequence, True, False
            else:
                self._log(f"[Depth {depth}] Game ended after our action - WE LOSE")
                return -5000.0, action_sequence, True, False
        
        # Check if we can stop here because the round changed already
        if sim_env.round > starting_round:
            # Even if round changed, we need to check final penalties
            final_penalty = sim_env.penalties.get(self.training_agent, 0)
            final_hand_size = len(sim_env.players_hands.get(self.training_agent, []))
            is_terminated = sim_env.terminations.get(self.training_agent, False)
            
            # Detect if we got eliminated during round change
            if is_terminated:
                self._log(f"[Depth {depth}] We got ELIMINATED during round change! Very negative value")
                return -10000.0, action_sequence, True, True
            
            # Detect penalty increases during round change
            if final_penalty > starting_penalty:
                self._log(f"[Depth {depth}] We got a penalty during round change, returning very negative value")
                return -5000.0, action_sequence, False, True
            
            # If our hand size changed significantly during round change (challenge), check our status
            if final_hand_size < starting_hand_size - 1:  # More than 1 card difference
                if final_penalty > starting_penalty:
                    self._log(f"[Depth {depth}] Lost cards AND penalty during round change - very bad")
                    return -5000.0, action_sequence, False, True
        
        # Track opponent penalties
        found_opponent_penalty = False
        
        # Maximum steps to prevent infinite loops - INCREASED for more thorough simulation
        max_steps = 20  # Increased from 10
        step_count = 0
        
        # Continue simulation until game ends or round changes
        while step_count < max_steps:
            step_count += 1
            
            # If game is over, evaluate and return
            if sim_env.agent_selection is None:
                winner = sim_env.winner
                if winner == self.training_agent:
                    self._log(f"[Depth {depth}] Game ended after {step_count} steps - WE WIN!")
                    return 5000.0, action_sequence, True, False
                else:
                    self._log(f"[Depth {depth}] Game ended after {step_count} steps - WE LOSE")
                    return -5000.0, action_sequence, True, False
            
            # Check if round changed
            if sim_env.round > starting_round:
                # Check our final status
                final_penalty = sim_env.penalties.get(self.training_agent, 0)
                final_hand_size = len(sim_env.players_hands.get(self.training_agent, []))
                is_terminated = sim_env.terminations.get(self.training_agent, False)
                
                # Detect if we got eliminated
                if is_terminated:
                    self._log(f"[Depth {depth}] We got ELIMINATED during round change! Very negative value")
                    return -10000.0, action_sequence, True, True
                
                # Detect penalty increases during round change
                if final_penalty > starting_penalty:
                    self._log(f"[Depth {depth}] We got a penalty during round change, returning very negative value")
                    return -5000.0, action_sequence, False, True
                
                # If found opponent penalty, this is excellent
                if found_opponent_penalty:
                    self._log(f"[Depth {depth}] Round changed and found opponent penalty, returning positive value")
                    return 2000.0, action_sequence, False, True
            
            # If it's our turn and we haven't exceeded max depth, EXPLORE ALL VALID ACTIONS RECURSIVELY
            if sim_env.agent_selection == self.training_agent and depth < max_depth:
                # Get valid actions
                sim_env.observe(self.training_agent, new=True)
                action_mask = sim_env.infos[self.training_agent].get('action_mask', [0] * 7)
                valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                
                if not valid_actions:
                    self._log(f"[Depth {depth}] No valid actions available")
                    return -50.0, action_sequence, False, False
                
                # RECURSIVE EXPLORATION - Try all valid actions and pick the best outcome
                best_value = float('-inf')
                best_sequence = None
                best_is_terminal = False
                best_is_new_round = False
                
                # Always check for challenge opportunities first
                if 6 in valid_actions:
                    last_agent = sim_env.last_action_agent
                    if last_agent:
                        played_cards = sim_env.last_played_cards.get(last_agent, [])
                        table_card = sim_env.table_card
                        is_bluff = any(card != table_card and card != "Joker" for card in played_cards)
                        if is_bluff:
                            self._log(f"[Depth {depth}] Found opponent bluff during simulation, trying challenge")
                            # Clone state for exploration
                            next_state = sim_env.get_state()
                            value, next_seq, is_terminal, is_new_round = self.simulate_round(
                                next_state, 6, opponent_action_cache, depth+1, max_depth
                            )
                            # If this outcome is good, use it immediately
                            if value > 0:
                                self._log(f"[Depth {depth}] Found successful challenge with value {value}")
                                return value, action_sequence + next_seq[1:], is_terminal, is_new_round
                            # Otherwise, still consider it with other actions
                            if value > best_value:
                                best_value = value
                                best_sequence = next_seq
                                best_is_terminal = is_terminal
                                best_is_new_round = is_new_round
                
                # Get current penalties and hand state
                current_penalty = sim_env.penalties.get(self.training_agent, 0)
                hand = sim_env.players_hands.get(self.training_agent, [])
                table_card = sim_env.table_card
                table_cards = [c for c in hand if c == table_card or c == "Joker"]
                
                # Prioritize actions - always try table cards first at high penalties
                prioritized_actions = []
                if current_penalty >= 2:
                    # First try table cards
                    for a in [0, 1, 2]:  # Table card plays (1, 2, 3 cards)
                        if a in valid_actions and (a % 3) + 1 <= len(table_cards):
                            prioritized_actions.append(a)
                    # Then other actions
                    for a in valid_actions:
                        if a not in prioritized_actions:
                            prioritized_actions.append(a)
                else:
                    # Prefer table cards, then low counts of non-table cards
                    for a in [0, 1, 2]:  # Table card plays
                        if a in valid_actions and (a % 3) + 1 <= len(table_cards):
                            prioritized_actions.append(a)
                    for a in [3]:  # Try 1 non-table card
                        if a in valid_actions:
                            prioritized_actions.append(a)
                    # Then other actions
                    for a in valid_actions:
                        if a not in prioritized_actions:
                            prioritized_actions.append(a)
                
                # Explore all actions and find the best outcome
                for next_action in prioritized_actions:
                    # Skip challenge if we already checked it
                    if next_action == 6 and best_sequence is not None:
                        continue
                        
                    self._log(f"[Depth {depth}] Exploring next action {next_action}")
                    # Clone state for exploration
                    next_state = sim_env.get_state()
                    value, next_seq, is_terminal, is_new_round = self.simulate_round(
                        next_state, next_action, opponent_action_cache, depth+1, max_depth
                    )
                    self._log(f"[Depth {depth}] Action {next_action} produced value {value}")
                    
                    # Early stopping: If we found an opponent getting a penalty, use it immediately
                    if value > 1000:
                        self._log(f"[Depth {depth}] Found path where opponent gets penalty, using immediately")
                        return value, action_sequence + next_seq[1:], is_terminal, is_new_round
                    
                    # Otherwise track best outcome
                    if value > best_value:
                        best_value = value
                        best_sequence = next_seq
                        best_is_terminal = is_terminal
                        best_is_new_round = is_new_round
                        self._log(f"[Depth {depth}] New best action: {next_action} with value {value}")
                
                # Return the best outcome found
                if best_sequence:
                    return best_value, action_sequence + best_sequence[1:], best_is_terminal, best_is_new_round
                else:
                    # No valid actions found - very unlikely but handle it
                    self._log(f"[Depth {depth}] No valid actions produced positive outcomes")
                    return -100.0, action_sequence, False, False
                    
            # If it's our turn but we've hit max depth, use a simple strategy
            elif sim_env.agent_selection == self.training_agent:
                # Get valid actions
                sim_env.observe(self.training_agent, new=True)
                action_mask = sim_env.infos[self.training_agent].get('action_mask', [0] * 7)
                valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                
                if not valid_actions:
                    self._log(f"[Depth {depth}] No valid actions available at max depth")
                    return -50.0, action_sequence, False, False
                
                # Simple strategy - use table cards at high penalties, otherwise 1 non-table to bait challenge
                hand = sim_env.players_hands.get(self.training_agent, [])
                table_card = sim_env.table_card
                table_cards = [c for c in hand if c == table_card or c == "Joker"]
                current_penalty = sim_env.penalties.get(self.training_agent, 0)
                
                # Select a reasonable action without recursion
                if 6 in valid_actions:  # Always consider challenging first
                    last_agent = sim_env.last_action_agent
                    if last_agent:
                        played_cards = sim_env.last_played_cards.get(last_agent, [])
                        is_bluff = any(card != table_card and card != "Joker" for card in played_cards)
                        if is_bluff:
                            next_action = 6  # Challenge
                            self._log(f"[Depth {depth}] At max depth - found bluff, challenging")
                elif current_penalty >= 2 and 0 in valid_actions and len(table_cards) > 0:
                    next_action = 0  # Play 1 table card
                    self._log(f"[Depth {depth}] At max depth - high penalties, playing 1 table card")
                elif 0 in valid_actions and len(table_cards) > 0:
                    next_action = 0  # Play 1 table card
                    self._log(f"[Depth {depth}] At max depth - playing 1 table card")
                elif current_penalty < 2 and 3 in valid_actions and len(hand) > len(table_cards):
                    next_action = 3  # Play 1 non-table card
                    self._log(f"[Depth {depth}] At max depth - playing 1 non-table card")
                else:
                    next_action = valid_actions[0]
                    self._log(f"[Depth {depth}] At max depth - using first valid action: {next_action}")
                
                # Take our action and add to sequence
                pre_step_termination = sim_env.terminations.get(self.training_agent, False)
                old_penalty = sim_env.penalties.get(self.training_agent, 0)
                sim_env.step(next_action)
                action_sequence.append((self.training_agent, next_action))
                
                # Check if we got eliminated
                post_step_termination = sim_env.terminations.get(self.training_agent, False)
                if not pre_step_termination and post_step_termination:
                    self._log(f"[Depth {depth}] Got eliminated at max depth! Very bad.")
                    return -10000.0, action_sequence, True, False
                
                # Check if round changed
                if sim_env.round > starting_round:
                    final_penalty = sim_env.penalties.get(self.training_agent, 0)
                    is_terminated = sim_env.terminations.get(self.training_agent, False)
                    
                    if final_penalty > old_penalty or is_terminated:
                        self._log(f"[Depth {depth}] Got penalty or eliminated at round change at max depth")
                        return -5000.0, action_sequence, is_terminated, True
                
                # Check if we got a penalty
                new_penalty = sim_env.penalties.get(self.training_agent, 0)
                if new_penalty > old_penalty:
                    self._log(f"[Depth {depth}] Got penalty at max depth, bad")
                    return -1000.0, action_sequence, False, False
            
            # If it's an opponent's turn
            else:
                current_agent = sim_env.agent_selection
                self._log(f"[Depth {depth}] Opponent {current_agent}'s turn")
                
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
                        self._log(f"[Depth {depth}] Using cached action {opponent_action} for opponent {current_agent}")
                    else:
                        # Get the opponent's action OR force challenge if likely
                        opponent_action = self._select_opponent_action(sim_env, current_agent)
                        
                        # Cache it for future use
                        opponent_action_cache[obs_key] = opponent_action
                        self._log(f"[Depth {depth}] Caching action {opponent_action} for opponent {current_agent}")
                    
                    # Decode for better logging
                    opp_action_type, opp_card_cat, opp_count = decode_action(opponent_action)
                    self._log(f"[Depth {depth}] Opponent action: {opponent_action} ({opp_action_type}, {opp_card_cat}, {opp_count})")
                    
                    # Get the opponent's penalty before action
                    old_penalty = sim_env.penalties.get(current_agent, 0)
                    
                    # Check if this is a challenge to our action
                    is_challenging_us = (opponent_action == 6 and last_agent == self.training_agent)
                    if is_challenging_us:
                        # Check if we were bluffing
                        played_cards = sim_env.last_played_cards.get(self.training_agent, [])
                        is_our_bluff = any(card != table_card and card != "Joker" for card in played_cards)
                        if is_our_bluff:
                            self._log(f"[Depth {depth}] Opponent is challenging our bluff! Very bad.")
                            # Take action anyway to record, but will return negative
                            sim_env.step(opponent_action)
                            action_sequence.append((current_agent, opponent_action))
                            # Extra negative if we're at high penalties
                            if starting_penalty >= 2:
                                return -10000.0, action_sequence, False, False  # Fatal at high penalties
                            return -5000.0, action_sequence, False, False
                    
                    # Take the action and record it
                    sim_env.step(opponent_action)
                    action_sequence.append((current_agent, opponent_action))
                    
                    # Check if game is over
                    if sim_env.agent_selection is None:
                        winner = sim_env.winner
                        if winner == self.training_agent:
                            self._log(f"[Depth {depth}] Game ended - WE WIN!")
                            return 5000.0, action_sequence, True, False
                        else:
                            self._log(f"[Depth {depth}] Game ended - WE LOSE")
                            return -5000.0, action_sequence, True, False
                    
                    # Check if opponent got a penalty - this is good for us
                    new_penalty = sim_env.penalties.get(current_agent, 0)
                    if new_penalty > old_penalty:
                        found_opponent_penalty = True
                        self._log(f"[Depth {depth}] Opponent got a penalty from action {opponent_action}, very good")
                        # Early return for good outcome
                        return 2000.0, action_sequence, False, False
                    
                    # Check if round changed
                    if sim_env.round > starting_round:
                        # Check our final status
                        final_penalty = sim_env.penalties.get(self.training_agent, 0)
                        is_terminated = sim_env.terminations.get(self.training_agent, False)
                        
                        if is_terminated:
                            self._log(f"[Depth {depth}] Got ELIMINATED during round change! Very negative")
                            return -10000.0, action_sequence, True, True
                        
                        if final_penalty > starting_penalty:
                            self._log(f"[Depth {depth}] Our penalty increased during round change, returning negative value")
                            return -5000.0, action_sequence, False, True
                        
                        if found_opponent_penalty:
                            self._log(f"[Depth {depth}] Round changed and found opponent penalty, returning positive value")
                            return 2000.0, action_sequence, False, True
                    
                except Exception as e:
                    self._log(f"[Depth {depth}] Error with opponent action: {e}")
                    return -50.0, action_sequence, False, False
        
        # If we reach here, we hit the step limit
        self._log(f"[Depth {depth}] Hit step limit ({max_steps}) - sequence length: {len(action_sequence)}")
        if found_opponent_penalty:
            return 1000.0, action_sequence, False, False
        else:
            return 10.0, action_sequence, False, False
    
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