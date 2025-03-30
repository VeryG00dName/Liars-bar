# src/model/ps.py
import numpy as np
import torch
from src import config
from src.env.liars_deck_env_utils_2 import decode_action, select_cards_to_play, validate_claim

class PerfectSearch:
    """
    Perfect Search algorithm for Liar's Deck with exact opponent model knowledge.
    
    This implementation finds guaranteed winning paths by simulating until the end
    of the current round (detected by the environment's round counter).
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
        
        # Debug flag for verbose logging
        self.debug = True
    
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
    
    def _is_bluffing(self, env, agent):
        """
        Checks if an agent's last action was a bluff (played non-table cards).
        
        Args:
            env: The environment
            agent: The agent to check
            
        Returns:
            bool: True if the agent was bluffing, False otherwise
        """
        # Check if there was a last action by this agent
        if env.last_action_agent != agent or env.last_action is None:
            return False
            
        # Get the played cards
        played_cards = env.last_played_cards.get(agent, [])
        if not played_cards:
            return False
            
        table_card = env.table_card
        
        # CRITICAL FIX: Be extremely precise about bluff detection
        # A card is a bluff if it is neither the table card nor a Joker
        for card in played_cards:
            if card != table_card and card != "Joker":
                self._log(f"Agent {agent} played a bluff card: {card} (table card: {table_card})")
                return True
                    
        # All cards match the table card or are Jokers, so it's not a bluff
        self._log(f"Agent {agent} played legitimate cards: {played_cards} (table card: {table_card})")
        return False
    
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
        
        # CRITICAL FIX: Be extremely precise about bluff detection
        is_bluff = False
        for card in played_cards:
            if card != table_card and card != "Joker":
                is_bluff = True
                break
        
        # Check our penalty status
        our_penalty = env.penalties.get(self.training_agent, 0)
        
        # CRITICAL FIX: If we have high penalties (2+), ONLY challenge when we're 100% certain
        if our_penalty >= 2:
            return is_bluff
        
        # If we have 1 penalty, still be conservative
        if our_penalty == 1:
            # Only consider additional challenge criteria if our confidence is high
            if is_bluff:
                return True
                
            # If not a confirmed bluff, only challenge in very suspicious cases
            opponent_played_count = len(played_cards)
            if opponent_played_count == 3:
                # Check how many of this card could reasonably exist
                # In a standard deck, there are 6 of each card type + 2 jokers
                our_hand = env.players_hands.get(self.training_agent, [])
                our_table_cards = sum(1 for card in our_hand if card == table_card or card == "Joker")
                
                # If they played 3 and we have 2 or more, that's 5+ already, which is suspicious
                if our_table_cards >= 2:
                    self._log(f"Opponent played {opponent_played_count} {table_card}s and we have {our_table_cards}. Suspicious!")
                    return True
            
            # Otherwise, don't challenge
            return False
        
        # If we have 0 penalties, we can be a bit more aggressive
        if is_bluff:
            return True
        
        # Even with 0 penalties, be somewhat careful
        # Consider the opponent's history if available
        if hasattr(env, 'bluff_counts') and hasattr(env, 'total_plays'):
            opponent_bluffs = env.bluff_counts.get(last_agent, 0)
            opponent_plays = env.total_plays.get(last_agent, 0)
            
            if opponent_plays > 3:  # Need enough history to make a judgment
                bluff_ratio = opponent_bluffs / opponent_plays
                # If more than 66% of their plays were bluffs, be more suspicious of large plays
                opponent_played_count = len(played_cards)
                if bluff_ratio > 0.66 and opponent_played_count >= 3:
                    self._log(f"Opponent has bluff ratio {bluff_ratio:.2f} and played {opponent_played_count} cards. Suspicious!")
                    return True
        
        # Final check: if they played 3+ cards and we have 3+ of the same, that's very suspicious
        opponent_played_count = len(played_cards)
        if opponent_played_count >= 3:
            our_hand = env.players_hands.get(self.training_agent, [])
            our_table_cards = sum(1 for card in our_hand if card == table_card or card == "Joker")
            if our_table_cards >= 3:
                self._log(f"Opponent played {opponent_played_count} {table_card}s and we have {our_table_cards}. Highly suspicious!")
                return True
        
        # Default to not challenging
        return False
    
    def _count_table_cards(self, env, agent):
        """
        Count the number of table cards (including jokers) in an agent's hand.
        
        Args:
            env: The environment
            agent: The agent name
            
        Returns:
            int: Number of table cards
        """
        hand = env.players_hands.get(agent, [])
        table_card = env.table_card
        return sum(1 for card in hand if card == table_card or card == "Joker")
    
    def simulate_round(self, env_state, action):
        """
        Simulates until the end of the current round or until a terminal state.
        Uses the environment's round counter to detect when a new round starts.
        
        Args:
            env_state: The current environment state
            action: The action to simulate
            
        Returns:
            tuple: (outcome_value, action_sequence, is_terminal, is_new_round)
                outcome_value: Evaluation of the resulting state
                action_sequence: List of (agent, action) tuples for the current round
                is_terminal: Whether the game ended
                is_new_round: Whether we reached a new round
        """
        self.simulations_performed += 1
        
        # Clone environment and set state
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)
        
        if sim_env.agent_selection != self.training_agent:
            raise RuntimeError(f"Expected training agent's turn but got {sim_env.agent_selection}")
        
        # Record our starting state information
        starting_penalty = sim_env.penalties.get(self.training_agent, 0)
        starting_round = sim_env.round
        
        self._log(f"Starting simulation in round {starting_round}")
        
        # CRITICAL FIX: If this is a challenge action, verify whether it would succeed
        # before we even try it, to ensure correct prediction
        action_type, _, _ = decode_action(action)
        if action_type == "Challenge":
            # Get the opponent who made the last action
            last_agent = sim_env.last_action_agent
            if last_agent is None:
                self._log("Invalid challenge - no last action agent")
                return -100.0, [(self.training_agent, action)], False, False
                
            # Get the cards they played
            played_cards = sim_env.last_played_cards.get(last_agent, [])
            if not played_cards:
                self._log("Invalid challenge - no cards played")
                return -100.0, [(self.training_agent, action)], False, False
                
            # Check if the opponent was actually bluffing
            table_card = sim_env.table_card
            was_bluff = any(card != table_card and card != "Joker" for card in played_cards)
            
            # CRITICAL: Predict the outcome of the challenge based on actual cards
            if not was_bluff:
                self._log(f"Predicted FAILED challenge against non-bluff with penalty {starting_penalty}")
                # This is a bad move that would lead to immediate loss if we're at 2 penalties
                if starting_penalty >= 2:
                    return -10000.0, [(self.training_agent, action)], False, False
                else:
                    return -1000.0, [(self.training_agent, action)], False, False
        
        # Start with our action
        action_sequence = [(self.training_agent, action)]
        
        # Debug: Print what cards we have before the action
        hand = sim_env.players_hands.get(self.training_agent, [])
        self._log(f"Initial hand: {hand}, table card: {sim_env.table_card}")
        
        action_type, card_category, count = decode_action(action)
        self._log(f"Taking action: {action} ({action_type}, {card_category}, {count})")
        
        # Take the action
        sim_env.step(action)
        
        # Debug: Print hand after our action
        new_hand = sim_env.players_hands.get(self.training_agent, [])
        new_penalty = sim_env.penalties.get(self.training_agent, 0)
        cards_played = sim_env.last_played_cards.get(self.training_agent, [])
        current_round = sim_env.round
        
        self._log(f"After our action - Hand: {new_hand}, Penalty: {new_penalty}, Cards played: {cards_played}, Round: {current_round}")
        
        # CRUCIAL FIX: Check if we're eliminated by this action
        if new_penalty >= 3:
            self._log(f"Our penalty increased to {new_penalty} - we're ELIMINATED! Returning extremely negative value")
            return -10000.0, action_sequence, True, False
        
        # Check for penalty increase - much more negative at high penalty levels
        if new_penalty > starting_penalty:
            if starting_penalty >= 2:
                self._log(f"Our penalty increased to {new_penalty} at critical level, terminating with extremely low value")
                return -5000.0, action_sequence, False, False
            elif starting_penalty >= 1:
                self._log(f"Our penalty increased to {new_penalty} at medium level, terminating with very low value")
                return -1000.0, action_sequence, False, False
            else:
                self._log(f"Our penalty increased to {new_penalty}, terminating with low value")
                return -500.0, action_sequence, False, False
        
        # BUGFIX: Don't immediately return if round changed after our action
        # Instead, check if our penalty increased first
        if current_round > starting_round:
            # Check if we were penalized in the process
            final_penalty = sim_env.penalties.get(self.training_agent, 0)
            if final_penalty > starting_penalty:
                self._log(f"Round advanced to {current_round} but we were penalized! Penalty: {final_penalty}")
                return -1000.0, action_sequence, False, True
            
            self._log(f"Round advanced from {starting_round} to {current_round} after our action")
            return self._evaluate_state(sim_env), action_sequence, False, True
        
        # Maximum steps to prevent infinite loops
        max_steps = 100
        step_count = 0
        
        # Flag to indicate if we should prioritize challenging in the future
        waiting_to_challenge = False
        
        # Continue simulation until game ends, round changes, or max steps reached
        while step_count < max_steps:
            step_count += 1
            
            # If game is over, evaluate and return
            if sim_env.agent_selection is None:
                self._log(f"Game ended after {step_count} steps")
                return self._evaluate_terminal_state(sim_env), action_sequence, True, False
            
            # BUGFIX: Don't immediately return if round changed
            # Instead, check if our penalty increased in the process
            current_round = sim_env.round
            if current_round > starting_round:
                # Check if we were penalized in the process
                final_penalty = sim_env.penalties.get(self.training_agent, 0)
                if final_penalty > starting_penalty:
                    self._log(f"Round advanced to {current_round} but we were penalized! Final penalty: {final_penalty}")
                    return -1000.0, action_sequence, False, True
                
                self._log(f"Round advanced from {starting_round} to {current_round} - ending simulation")
                return self._evaluate_state(sim_env), action_sequence, False, True
        
        # Maximum steps to prevent infinite loops
        max_steps = 100
        step_count = 0
        
        # Flag to indicate if we should prioritize challenging in the future
        waiting_to_challenge = False
        
        # Continue simulation until game ends, round changes, or max steps reached
        while step_count < max_steps:
            step_count += 1
            
            # If game is over, evaluate and return
            if sim_env.agent_selection is None:
                self._log(f"Game ended after {step_count} steps")
                return self._evaluate_terminal_state(sim_env), action_sequence, True, False
            
            # BUGFIX: Don't immediately return if round changed
            # Instead, check if our penalty increased in the process
            current_round = sim_env.round
            if current_round > starting_round:
                # Check if we were penalized in the process
                final_penalty = sim_env.penalties.get(self.training_agent, 0)
                if final_penalty > starting_penalty:
                    self._log(f"Round advanced to {current_round} but we were penalized! Final penalty: {final_penalty}")
                    return -100.0, action_sequence, False, True
                
                self._log(f"Round advanced from {starting_round} to {current_round} - ending simulation")
                return self._evaluate_state(sim_env), action_sequence, False, True
            
            # If it's our turn
            if sim_env.agent_selection == self.training_agent:
                # Get valid actions
                sim_env.observe(self.training_agent, new=True)
                action_mask = sim_env.infos[self.training_agent].get('action_mask', [0] * 7)
                valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                
                self._log(f"Our turn - Hand: {sim_env.players_hands.get(self.training_agent, [])}, Valid actions: {valid_actions}")
                
                if not valid_actions:
                    self._log("No valid actions available, terminating simulation")
                    return -50.0, action_sequence, False, False
                
                # Get our current penalty
                current_penalty = sim_env.penalties.get(self.training_agent, 0)
                
                # CRITICAL FIX: Check if we can challenge and should, but be extra careful about false challenges
                should_challenge = False
                if 6 in valid_actions:
                    # Use our updated _should_challenge method which considers penalties
                    should_challenge = self._should_challenge(sim_env)
                    
                    # Double-check that the opponent really is bluffing before challenging
                    last_agent = sim_env.last_action_agent
                    if last_agent:
                        played_cards = sim_env.last_played_cards.get(last_agent, [])
                        if played_cards:
                            table_card = sim_env.table_card
                            is_actual_bluff = any(card != table_card and card != "Joker" for card in played_cards)
                            
                            # If the original method indicated challenge but it's not actually a bluff, override
                            if should_challenge and not is_actual_bluff:
                                self._log(f"Overriding challenge decision - opponent is NOT actually bluffing")
                                should_challenge = False
                
                # Make the action decision
                if should_challenge:
                    next_action = 6
                    waiting_to_challenge = False
                    self._log(f"Selecting challenge action because opponent is bluffing")
                elif waiting_to_challenge:
                    # If waiting to challenge but can't yet, prefer table cards first when at high penalties
                    if current_penalty >= 2:
                        # With high penalties, prioritize safe table cards over waiting for challenge
                        table_actions = [a for a in valid_actions if a < 3]
                        if table_actions:
                            next_action = min(table_actions, key=lambda a: (a % 3) + 1)  # Play minimum count
                            self._log(f"High penalties, playing safe table cards: {next_action}")
                            waiting_to_challenge = False  # Stop waiting for challenge
                        else:
                            # If no table cards, play minimum non-table cards
                            play_actions = [a for a in valid_actions if a >= 3 and a <= 5]
                            if play_actions:
                                next_action = min(play_actions, key=lambda a: (a % 3) + 1)
                                self._log(f"High penalties but waiting to challenge, playing minimum cards: {next_action}")
                            else:
                                next_action = valid_actions[0]
                                self._log(f"High penalties, using fallback action: {next_action}")
                    else:
                        # Normal penalty situation, wait for challenge opportunity
                        play_actions = [a for a in valid_actions if a >= 3 and a <= 5]  # Non-table actions
                        if play_actions:
                            next_action = min(play_actions, key=lambda a: (a % 3) + 1)  # Play minimum count
                            self._log(f"Waiting to challenge, playing minimum non-table cards: {next_action}")
                        else:
                            # If we can't play non-table cards, play minimum table cards
                            play_actions = [a for a in valid_actions if a < 3]
                            if play_actions:
                                next_action = min(play_actions)
                                self._log(f"Waiting to challenge, but playing table cards: {next_action}")
                            else:
                                next_action = valid_actions[0]
                                self._log(f"Waiting to challenge, but had to use fallback action: {next_action}")
                else:
                    # CRITICAL FIX: Adopt a more conservative strategy when at high penalties
                    if current_penalty >= 2:
                        # With high penalties, play safe table cards or minimum cards
                        table_actions = [a for a in valid_actions if a < 3]
                        if table_actions:
                            next_action = min(table_actions, key=lambda a: (a % 3) + 1)  # Play minimum count
                            self._log(f"High penalties, playing safe table cards: {next_action}")
                        else:
                            # If no table cards, play minimum cards
                            play_actions = [a for a in valid_actions if a < 6]
                            if play_actions:
                                next_action = min(play_actions, key=lambda a: (a % 3) + 1)
                                self._log(f"High penalties, playing minimum cards: {next_action}")
                            else:
                                next_action = valid_actions[0]
                                self._log(f"High penalties, using fallback action: {next_action}")
                    else:
                        # Normal hand evaluation
                        hand = sim_env.players_hands.get(self.training_agent, [])
                        if len(hand) <= 3:
                            # Play non-table cards one at a time to set up for a challenge
                            if 3 in valid_actions:  # Play 1 non-table card
                                next_action = 3
                                waiting_to_challenge = True
                                self._log(f"Few cards left, playing 1 non-table card and setting up for challenge")
                            else:
                                # Otherwise play minimum valid cards
                                play_actions = [a for a in valid_actions if a < 6]
                                if play_actions:
                                    next_action = min(play_actions, key=lambda a: (a % 3) + 1)
                                    self._log(f"Few cards left, playing minimum count: {next_action}")
                                else:
                                    next_action = valid_actions[0]
                                    self._log(f"Few cards left, using fallback action: {next_action}")
                        else:
                            # With more cards, play 2-3 at a time to reduce hand size faster
                            if 4 in valid_actions:  # Play 2 non-table cards
                                next_action = 4
                                self._log(f"More cards left, playing 2 non-table cards")
                            elif 5 in valid_actions:  # Play 3 non-table cards
                                next_action = 5
                                self._log(f"More cards left, playing 3 non-table cards")
                            elif 3 in valid_actions:  # Play 1 non-table card
                                next_action = 3
                                waiting_to_challenge = True
                                self._log(f"More cards left, playing 1 non-table card and setting up for challenge")
                            else:
                                # Otherwise play table cards if available
                                table_actions = [a for a in valid_actions if a < 3]
                                if table_actions:
                                    next_action = max(table_actions)  # Play max table cards
                                    self._log(f"More cards left, playing max table cards: {next_action}")
                                else:
                                    next_action = valid_actions[0]
                                    self._log(f"More cards left, using fallback action: {next_action}")
                
                # Debug: Show action details
                action_type, card_category, count = decode_action(next_action)
                self._log(f"Selected action: {next_action} ({action_type}, {card_category}, {count})")
                
                # Get the round before taking action
                pre_round = sim_env.round
                
                # Take our action and add to sequence
                old_penalty = sim_env.penalties.get(self.training_agent, 0)
                sim_env.step(next_action)
                action_sequence.append((self.training_agent, next_action))
                
                # Debug: Show what happened after our action
                new_hand = sim_env.players_hands.get(self.training_agent, [])
                new_penalty = sim_env.penalties.get(self.training_agent, 0)
                cards_played = sim_env.last_played_cards.get(self.training_agent, [])
                post_round = sim_env.round
                
                self._log(f"After our action - Hand: {new_hand}, Penalty: {new_penalty}, Cards played: {cards_played}, Round: {post_round}")
                
                # Check if round changed
                if post_round > pre_round:
                    self._log(f"Round advanced from {pre_round} to {post_round} after our action")
                    return self._evaluate_state(sim_env), action_sequence, False, True
                
                # Check if we got a penalty - immediately terminate if so with scaled penalty
                if new_penalty > old_penalty:
                    penalty_value = -100.0
                    # Scale penalty worse if we're already at high penalties
                    if old_penalty >= 2:
                        penalty_value = -1000.0  # Much worse if already at 2 penalties
                    elif old_penalty >= 1:
                        penalty_value = -500.0   # Worse if already at 1 penalty
                    
                    self._log(f"Our penalty increased to {new_penalty}, terminating simulation with value {penalty_value}")
                    return penalty_value, action_sequence, False, False
            
            # If it's an opponent's turn, use their model
            else:
                current_agent = sim_env.agent_selection
                self._log(f"Opponent {current_agent}'s turn - Hand: {sim_env.players_hands.get(current_agent, [])}")
                
                try:
                    # Get the opponent's action
                    opponent_action = self._select_opponent_action(sim_env, current_agent)
                    
                    # Debug: Show opponent action details
                    action_type, card_category, count = decode_action(opponent_action)
                    self._log(f"Opponent action: {opponent_action} ({action_type}, {card_category}, {count})")
                    
                    # Get the round before taking action
                    pre_round = sim_env.round
                    
                    # Take the action and record it
                    sim_env.step(opponent_action)
                    action_sequence.append((current_agent, opponent_action))
                    
                    # Get the round after taking action
                    post_round = sim_env.round
                    
                    # Debug: Show what happened after opponent action
                    cards_played = sim_env.last_played_cards.get(current_agent, [])
                    is_bluffing = self._is_bluffing(sim_env, current_agent)
                    self._log(f"After opponent action - Cards played: {cards_played}, Bluffing: {is_bluffing}, Round: {post_round}")
                    
                    # Check if round changed
                    if post_round > pre_round:
                        self._log(f"Round advanced from {pre_round} to {post_round} after opponent action")
                        return self._evaluate_state(sim_env), action_sequence, False, True
                    
                    # Check if this opponent just bluffed - if so, we'll want to challenge next
                    if opponent_action < 6:  # If they played cards
                        waiting_to_challenge = is_bluffing
                        if is_bluffing:
                            self._log(f"Opponent {current_agent} is bluffing, setting up to challenge next turn")
                    
                except Exception as e:
                    self._log(f"Error simulating opponent {current_agent}: {e}")
                    return -50.0, action_sequence, False, False
        
        # If we reach here, we hit the step limit
        self._log(f"WARNING: Hit step limit ({max_steps}) during simulation")
        return self._evaluate_state(sim_env), action_sequence, False, False
    
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
                self._log(f"WE WON THE GAME, very high value (round {env.round})")
                return 1000.0  # We won (very high value to prioritize winning paths)
            else:
                self._log(f"WE LOST THE GAME, very low value (round {env.round})")
                return -1000.0  # We lost
        
        # Check penalty counts - CRITICAL FIX: Update penalty evaluation
        our_penalty = env.penalties.get(self.training_agent, 0)
        opponent_penalties = {opp: env.penalties.get(opp, 0) for opp in self.opponent_agents}
        
        # Add more nuance to penalty evaluation
        max_opponent_penalty = max(opponent_penalties.values()) if opponent_penalties else 0
        diff = max_opponent_penalty - our_penalty
        
        # If we're at max penalties, this is extremely bad
        if our_penalty >= 3:
            self._log(f"We have max penalties ({our_penalty}), game over, very negative value")
            return -1000.0
        
        # If we're at 2 penalties and any opponent has 3, we're in a very good position
        if our_penalty <= 2 and max_opponent_penalty >= 3:
            self._log(f"Opponents have max penalties ({max_opponent_penalty}), we have {our_penalty}, very positive value")
            return 500.0
        
        # Calculate penalty-based value with more weight when we're at higher penalty levels
        penalty_factor = diff * 100.0
        
        # If we have fewer cards left than opponents, that's good
        our_hand_size = len(env.players_hands.get(self.training_agent, []))
        opponent_hand_sizes = [len(env.players_hands.get(opp, [])) for opp in self.opponent_agents]
        min_opponent_hand_size = min(opponent_hand_sizes) if opponent_hand_sizes else 5
        
        # CRITICAL FIX: Add more weight to hand size difference when we're near the end
        hand_size_diff = min_opponent_hand_size - our_hand_size
        hand_factor = hand_size_diff * 20.0  # Increase weight
        
        # CRITICAL FIX: Empty hand is a strong positive
        if our_hand_size == 0:
            hand_factor += 100.0
            self._log(f"We have an empty hand, very positive value")
        
        # Combine factors
        total_value = penalty_factor + hand_factor
        
        # Add logging
        self._log(f"Terminal evaluation: penalty_factor={penalty_factor}, hand_factor={hand_factor}, total={total_value}")
        
        return total_value
    
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
        
        # CRITICAL FIX: Immediate elimination should be extremely negative
        if our_penalty >= 3:
            self._log(f"We are eliminated with {our_penalty} penalties! Returning extremely negative value.")
            return -10000.0  # Game over - we lost
        
        # Calculate opponent penalties 
        opponent_penalties = {opp: env.penalties.get(opp, 0) for opp in self.opponent_agents}
        max_opponent_penalty = max(opponent_penalties.values()) if opponent_penalties else 0
        
        # Calculate hand sizes
        our_hand_size = len(our_hand)
        opponent_hand_sizes = {opp: len(env.players_hands.get(opp, [])) for opp in self.opponent_agents}
        
        # Calculate table cards in our hand
        table_card = env.table_card
        table_cards_count = sum(1 for card in our_hand if card == table_card or card == "Joker")
        
        # CRITICAL FIX: Penalty factor should be negative when our penalty is higher!
        # It should also be scaled based on how close we are to elimination
        penalty_diff = max_opponent_penalty - our_penalty
        
        # CRITICAL FIX: Scale penalty factor exponentially based on our penalty count
        if our_penalty == 0:
            penalty_factor = penalty_diff * 50.0
        elif our_penalty == 1:
            penalty_factor = penalty_diff * 100.0
        else:  # our_penalty == 2
            penalty_factor = penalty_diff * 500.0  # Much higher weight when at risk of elimination
        
        # Hand size factor: prefer smaller hands
        hand_size_factor = (5.0 - our_hand_size) * 5.0
        
        # Table cards factor: more table cards gives us more safe play options
        table_cards_factor = table_cards_count * 2.0
        
        # Round factor: higher rounds are better (means we've survived longer)
        round_factor = env.round * 5.0
        
        # CRITICAL FIX: Check for last_challenge_success - if we were just successfully challenged, heavily penalize
        challenge_penalty = 0.0
        if hasattr(env, 'last_challenge_success') and env.last_challenge_success is True:
            # If the last challenge was successful AND we were the last_action_agent (i.e., we were caught bluffing)
            if env.last_action_agent == self.training_agent:
                challenge_penalty = -100.0
                self._log(f"Adding challenge penalty of {challenge_penalty} - we were caught bluffing")
        
        # Weighted combination of factors
        score = penalty_factor + hand_size_factor + table_cards_factor + round_factor + challenge_penalty
        
        self._log(f"State evaluation (round {env.round}): penalty_factor={penalty_factor}, hand_size_factor={hand_size_factor}, " +
                f"table_cards_factor={table_cards_factor}, round_factor={round_factor}, " +
                f"challenge_penalty={challenge_penalty}, total={score}")
        
        return score
    
    def get_next_agent_action(self, agent):
        """
        Get the next action for any agent from the cached sequence.
        
        Args:
            agent: The agent name (can be training_agent or an opponent)
            
        Returns:
            action: The next action for this agent, or None if no action is found
        """
        # Check if we've reached the end of the sequence
        if self.sequence_position >= len(self.action_sequence):
            return None
        
        # CRITICAL FIX: Clone environment to evaluate the current state
        sim_env = self.base_env.clone()
        
        # If this is for our training agent, ensure the action is still valid
        if agent == self.training_agent:
            # CRITICAL FIX: Observe to ensure we have the latest state and action mask
            sim_env.observe(agent, new=True)
            
            # Get the action mask to validate actions
            if agent in sim_env.infos and "action_mask" in sim_env.infos[agent]:
                action_mask = sim_env.infos[agent]['action_mask']
            else:
                self._log(f"No valid action mask available for {agent}, invalidating cache")
                # Invalidate cache if we can't get a valid action mask
                self.action_sequence = []
                self.sequence_position = 0
                return None
        
        # Look for the next action for this agent starting from current position
        for i in range(self.sequence_position, len(self.action_sequence)):
            seq_agent, action = self.action_sequence[i]
            if seq_agent == agent:
                # Found an action for this agent
                
                # Decode the action to check if it's a challenge
                action_type, _, _ = decode_action(action)
                
                # CRITICAL FIX: NEVER use cached challenge actions - always re-evaluate
                if action_type == "Challenge":
                    self._log(f"Found cached challenge action for {agent}, but re-evaluating for safety")
                    
                    # Check if challenging is actually valid right now
                    if self._should_challenge(sim_env):
                        # Double-check that it's actually a bluff by examining the cards
                        last_agent = sim_env.last_action_agent
                        played_cards = sim_env.last_played_cards.get(last_agent, [])
                        table_card = sim_env.table_card
                        
                        is_actual_bluff = False
                        for card in played_cards:
                            if card != table_card and card != "Joker":
                                is_actual_bluff = True
                                break
                        
                        if is_actual_bluff:
                            self._log(f"Confirmed opponent is actually bluffing, proceeding with challenge")
                            # Move past this action in the sequence and return it
                            self.sequence_position = i + 1
                            return action
                        else:
                            self._log(f"Opponent is NOT bluffing, invalidating cached challenge")
                            # Invalidate the entire cache as our challenge prediction was wrong
                            self.action_sequence = []
                            self.sequence_position = 0
                            return None
                    else:
                        self._log(f"Challenging no longer valid, invalidating cached challenge")
                        # Invalidate the entire cache
                        self.action_sequence = []
                        self.sequence_position = 0
                        return None
                
                # For other actions, validate against the current mask if it's our agent
                if agent == self.training_agent:
                    if action_mask[action] != 1:
                        self._log(f"Cached action {action} is no longer valid for {agent}, invalidating cache")
                        # Invalidate the entire cache if the action is invalid
                        self.action_sequence = []
                        self.sequence_position = 0
                        return None
                
                # Action is valid, move past it in the sequence
                self.sequence_position = i + 1
                self._log(f"Found cached action for {agent}: {action} at position {i}")
                return action
        
        # If no action found, return None
        self._log(f"No cached action found for {agent} in remaining sequence")
        return None
    
    def search(self, env_state):
        """
        Searches for the best action by simulating until the end of the current round.
        
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
                # CRITICAL FIX: Never use cached bluff actions when at high penalty count
                current_penalty = self.base_env.penalties.get(self.training_agent, 0)
                if current_penalty >= 2:
                    action_type, card_category, _ = decode_action(next_action)
                    if action_type == "Play" and card_category == "non-table":
                        self._log(f"Found cached bluff action {next_action} but we have high penalties ({current_penalty}), invalidating cache")
                        self.action_sequence = []
                        self.sequence_position = 0
                    else:
                        action_dim = 7  # Default action dimension
                        action_probs = np.zeros(action_dim)
                        action_probs[next_action] = 1.0
                        return action_probs, next_action, 100.0
                else:
                    action_dim = 7  # Default action dimension
                    action_probs = np.zeros(action_dim)
                    action_probs[next_action] = 1.0
                    return action_probs, next_action, 100.0
        
        # Reset action sequence and position since we're starting a new search
        self.action_sequence = []
        self.sequence_position = 0
        
        # Reset simulation counter for this search
        self.simulations_performed = 0
        
        # Clone environment and set state
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)
        
        # Log the starting round
        starting_round = sim_env.round
        self._log(f"Starting search in round {starting_round}")
        
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
        
        # Get our hand information
        hand = sim_env.players_hands.get(self.training_agent, [])
        hand_size = len(hand)
        table_card = sim_env.table_card
        
        # Check what table cards we have
        table_cards_count = sum(1 for card in hand if card == table_card or card == "Joker")
        
        # CRITICAL FIX: Use penalty count to determine how conservative to be
        current_penalty = sim_env.penalties.get(self.training_agent, 0)
        
        # Identify safe plays (playing real table cards)
        safe_actions = []
        for action in range(3):  # Actions 0, 1, 2 are table card plays
            if action in valid_actions:
                action_count = (action % 3) + 1  # Count of cards to play
                if action_count <= table_cards_count:
                    safe_actions.append(action)
        
        # CRITICAL FIX: At high penalties, ONLY consider safe plays and legitimate challenges
        if current_penalty >= 2:
            prioritized_actions = []
            
            # First, add any safe table card plays
            prioritized_actions.extend(safe_actions)
            
            # Second, check if we can legitimately challenge an opponent's bluff
            challenge_action = 6
            if challenge_action in valid_actions:
                last_agent = sim_env.last_action_agent
                if last_agent:
                    played_cards = sim_env.last_played_cards.get(last_agent, [])
                    is_bluff = any(card != table_card and card != "Joker" for card in played_cards)
                    if is_bluff:
                        self._log(f"Adding legitimate challenge to high-penalty actions")
                        prioritized_actions.append(challenge_action)
            
            # If no safe actions available, only then consider risky actions
            if not prioritized_actions:
                self._log(f"WARNING: No safe actions available with high penalty count ({current_penalty})")
                prioritized_actions = valid_actions
            
            actions_to_try = prioritized_actions
            self._log(f"High penalty count ({current_penalty}). Considering only safe actions: {actions_to_try}")
        else:
            # Normal prioritization for lower penalty counts
            # If we can challenge and should, prioritize that
            challenge_action = 6  # Challenge action index
            if challenge_action in valid_actions:
                # Check if the opponent who made the last action is bluffing
                last_agent = sim_env.last_action_agent
                if last_agent:
                    played_cards = sim_env.last_played_cards.get(last_agent, [])
                    is_bluff = any(card != table_card and card != "Joker" for card in played_cards)
                    if is_bluff:
                        self._log(f"Opponent is bluffing - prioritizing challenge action")
                        valid_actions = [challenge_action]  # Only consider challenging
                        actions_to_try = valid_actions
                        
                        # Simulate the challenge and return immediately if it's a good action
                        value, sequence, is_terminal, is_new_round = self.simulate_round(env_state, challenge_action)
                        if value > 0:
                            action_dim = sim_env.action_spaces[self.training_agent].n
                            action_probs = np.zeros(action_dim)
                            action_probs[challenge_action] = 1.0
                            self._log(f"Selected challenge action with confidence (value={value})")
                            return action_probs, challenge_action, value
            
            # Regular action prioritization
            prioritized_actions = []
            
            # First priority: safe plays with table cards
            prioritized_actions.extend(safe_actions)
            
            # Second priority: If we have many cards (>3), consider non-table plays
            # but only if we have more than 2 non-table cards
            non_table_count = hand_size - table_cards_count
            if hand_size > 3 and non_table_count > 2:
                # Try playing multiple non-table cards to reduce hand faster
                for a in [5, 4, 3]:  # Try 3, then 2, then 1 non-table cards
                    if a in valid_actions and a not in prioritized_actions:
                        prioritized_actions.append(a)
            
            # Third priority: Consider challenging if possible
            if 6 in valid_actions and 6 not in prioritized_actions:
                prioritized_actions.append(6)
                
            # Last priority: Any remaining valid plays
            for a in valid_actions:
                if a not in prioritized_actions:
                    prioritized_actions.append(a)
            
            # Make sure all valid actions are included
            actions_to_try = prioritized_actions
        
        self._log(f"Searching among {len(valid_actions)} actions in order: {actions_to_try}")
        
        best_action = None
        best_value = float('-inf')
        best_sequence = []
        best_terminal = False
        best_new_round = False
        
        # Track which actions we skipped due to risk
        skipped_actions = []
        
        # Try all actions in prioritized order
        for action in actions_to_try:
            # CRITICAL FIX: More aggressive filtering of risky bluffs at high penalties
            action_type, card_category, count = decode_action(action)
            if action_type == "Play" and card_category == "non-table":
                # Skip ALL non-table plays if we have high penalties
                if current_penalty >= 2:
                    self._log(f"SKIPPING risky bluff action {action} due to high penalty count ({current_penalty})")
                    skipped_actions.append(action)
                    continue
                # For medium penalties, only allow small bluffs
                elif current_penalty >= 1 and count > 1:
                    self._log(f"SKIPPING risky large bluff action {action} due to medium penalty count ({current_penalty})")
                    skipped_actions.append(action)
                    continue
            
            # Simulate the action
            value, sequence, is_terminal, is_new_round = self.simulate_round(env_state, action)
            
            self._log(f"Action {action} ({action_type}, {card_category}, {count}): value={value:.1f}, " +
                    f"terminal={is_terminal}, new_round={is_new_round}, seq_len={len(sequence)}")
            
            # Skip very negative values (e.g., from penalty paths)
            if value < -50:
                self._log(f"Skipping action {action} with very negative value {value:.1f}")
                skipped_actions.append(action)
                continue
                
            # Prioritize terminal wins
            if is_terminal and value > 0:
                best_action = action
                best_value = value
                best_sequence = sequence
                best_terminal = True
                best_new_round = is_new_round
                self._log(f"Found guaranteed win with action {action}")
                break
            
            # Prioritize actions that result in new rounds (completed round)
            elif is_new_round and not best_terminal and (not best_new_round or value > best_value):
                # CRITICAL FIX: Be extra careful with bluffs that complete rounds
                if action_type == "Play" and card_category == "non-table" and current_penalty >= 1:
                    self._log(f"Being cautious about bluff action {action} that completes a round")
                    # Penalize the value to be less favorable for bluffs
                    adjusted_value = value - (current_penalty * 300)
                    if adjusted_value > best_value:
                        best_action = action
                        best_value = adjusted_value  # Store the penalized value
                        best_sequence = sequence
                        best_terminal = is_terminal
                        best_new_round = True
                        self._log(f"Found action that completes round (with caution): {action}")
                else:
                    best_action = action
                    best_value = value
                    best_sequence = sequence
                    best_terminal = is_terminal
                    best_new_round = True
                    self._log(f"Found action that completes round: {action}")
                
            # If no terminal win or new round yet, track best action
            elif not best_terminal and not best_new_round and value > best_value:
                # CRITICAL FIX: Be extra careful with bluffs in normal evaluation
                if action_type == "Play" and card_category == "non-table" and current_penalty >= 1:
                    self._log(f"Being cautious about bluff action {action}")
                    # Penalize the value to be less favorable for bluffs
                    adjusted_value = value - (current_penalty * 300)
                    if adjusted_value > best_value:
                        best_action = action
                        best_value = adjusted_value  # Store the penalized value
                        best_sequence = sequence
                        best_terminal = is_terminal
                        best_new_round = is_new_round
                        self._log(f"New best action (with caution): {action}")
                else:
                    best_action = action
                    best_value = value
                    best_sequence = sequence
                    best_terminal = is_terminal
                    best_new_round = is_new_round
                    self._log(f"New best action: {action} with value {value:.1f}")
        
        if best_action is None:
            # Handle the case where all actions were skipped
            self._log("All regular actions were skipped, falling back to safest option")
            
            # First check if we have safe table card plays available
            if safe_actions:
                best_action = safe_actions[0]  # Take the safest table card play
                value, sequence, is_terminal, is_new_round = self.simulate_round(env_state, best_action)
                best_value = value
                best_sequence = sequence
                self._log(f"Selected safe table card play: {best_action} as least bad action")
            else:
                # If no safe table card plays, try to find the least risky action
                for action in valid_actions:
                    action_type, card_category, count = decode_action(action)
                    
                    # Challenge is safe if there's a legitimate bluff
                    if action_type == "Challenge":
                        last_agent = sim_env.last_action_agent
                        if last_agent:
                            played_cards = sim_env.last_played_cards.get(last_agent, [])
                            is_bluff = any(card != table_card and card != "Joker" for card in played_cards)
                            if is_bluff:
                                best_action = action
                                value, sequence, is_terminal, is_new_round = self.simulate_round(env_state, best_action)
                                best_value = value
                                best_sequence = sequence
                                self._log(f"Selected legitimate challenge as fallback")
                                break
                    
                    # Play smallest number of table cards if possible
                    if action_type == "Play" and card_category == "table":
                        if best_action is None or count < decode_action(best_action)[2]:
                            best_action = action
                    
                    # As a last resort, play the smallest non-table card play
                    if best_action is None and action_type == "Play" and card_category == "non-table":
                        if best_action is None or count < decode_action(best_action)[2]:
                            best_action = action
                
                if best_action is not None:
                    value, sequence, is_terminal, is_new_round = self.simulate_round(env_state, best_action)
                    best_value = value
                    best_sequence = sequence
                    self._log(f"Selected fallback action: {best_action}")
                else:
                    # This should never happen, but just in case
                    best_action = valid_actions[0]
                    value, sequence, is_terminal, is_new_round = self.simulate_round(env_state, best_action)
                    best_value = value
                    best_sequence = sequence
                    self._log(f"Selected absolute last resort action: {best_action}")
        
        # Store the best action sequence for future reference
        self.action_sequence = best_sequence
        self.sequence_position = 0  # Reset position to start of sequence
        
        self._log(f"Selected action {best_action} with value {best_value:.1f}, sequence length {len(best_sequence)}")
        self._log(f"Performed {self.simulations_performed} simulations")
        
        # Build action probability vector
        action_dim = sim_env.action_spaces[self.training_agent].n
        action_probs = np.zeros(action_dim)
        action_probs[best_action] = 1.0
        
        return action_probs, best_action, best_value