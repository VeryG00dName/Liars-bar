import numpy as np
from src.env.liars_deck_env_utils_2 import decode_action
import torch
from src import config

class PerfectSearch:
    """
    Simplified Perfect Search algorithm for Liar's Deck.
    Focuses on finding paths where opponents get penalties.
    Uses a single linear action sequence plan.
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

        # The *only* cache: the planned sequence of (agent, action) tuples
        self.action_sequence = []
        self.sequence_position = 0

        self.simulations_performed = 0
        self.debug = True  # Set default debug state

    def _log(self, message):
        """Log a message if debug is enabled."""
        if self.debug:
            print(f"PS DEBUG: {message}")

    def invalidate_plan(self):
        """Resets the cached action sequence because the game state has deviated."""
        if self.action_sequence:  # Only log if there was a plan
            self._log("Plan invalidated due to deviation, invalid action, or new round.")
        self.action_sequence = []
        self.sequence_position = 0

    def _select_opponent_action(self, env, agent, opponent_action_cache):
        """
        Use the opponent model to select an action.
        Uses the simulation-specific cache for consistency within one simulation branch.
        
        Args:
            env: The simulation environment instance.
            agent: The opponent agent ID.
            opponent_action_cache: Dictionary to cache opponent actions based on observations.
            
        Returns:
            int: Selected action index for the opponent.
        """
        # Ensure we've observed the agent to generate infos.
        env.observe(agent, new=True)

        # Generate a hash key for the opponent's observation.
        hand = sorted(env.players_hands.get(agent, []))
        table_card = env.table_card
        last_action = env.last_action
        last_agent = env.last_action_agent
        cards_played = []
        if last_agent:
            cards_played = sorted(env.last_played_cards.get(last_agent, []) or [])
        obs_key = (
            agent,
            tuple(hand),
            table_card,
            last_action,
            last_agent,
            tuple(cards_played)
        )

        # Check if we've already determined an action for this observation.
        if obs_key in opponent_action_cache:
            opponent_action = opponent_action_cache[obs_key]
            self._log(f"[Sim] Using cached action {opponent_action} for opponent {agent}")
            # Validate cached action against current mask in simulation.
            env.observe(agent, new=True)
            action_mask = env.infos[agent].get('action_mask', [0] * 7)
            if action_mask[opponent_action] != 1:
                self._log(f"[Sim] Cached action {opponent_action} for {agent} is no longer valid. Recalculating.")
            else:
                return opponent_action

        # Get appropriate observation format for this opponent.
        opponent_model = self.opponent_models[agent]
        observation = env.observe(agent, new=True)[agent]
        action_mask = env.infos[agent]['action_mask']
        if sum(action_mask) == 0:
            raise RuntimeError(f"Agent {agent} has no valid actions according to mask")
        
        # Get action based on opponent type.
        if hasattr(opponent_model, 'play_turn'):  # Hardcoded agent
            opponent_action = opponent_model.play_turn(observation, action_mask, table_card=env.table_card)
            if action_mask[opponent_action] != 1:
                raise RuntimeError(f"Hardcoded agent {agent} returned invalid action {opponent_action}")
        else:  # Historical model (neural network)
            old_observation = env.observe(agent, new=False)[agent]
            obp_placeholder = np.zeros(2, dtype=np.float32)
            memory_placeholder = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
            final_obs = np.concatenate([old_observation, obp_placeholder, memory_placeholder], axis=0)
            observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device='cpu').unsqueeze(0)
            with torch.no_grad():
                try:
                    probs, _, _ = opponent_model(observation_tensor, None)
                except ValueError:
                    probs, _ = opponent_model(observation_tensor, None)
            probs = probs.squeeze().cpu().numpy()
            masked_probs = probs * action_mask
            masked_probs_sum = masked_probs.sum()
            if masked_probs_sum == 0:
                raise RuntimeError(f"Model for {agent} produced no valid action probability mass")
            masked_probs /= masked_probs_sum
            opponent_action = np.argmax(masked_probs)

        # Ensure action is valid before caching.
        if action_mask[opponent_action] != 1:
            self._log(f"[Sim] ERROR: Model for {agent} returned invalid action {opponent_action}. Defaulting.")
            valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
            opponent_action = valid_actions[0]
        
        # Cache the determined action for this simulation state.
        opponent_action_cache[obs_key] = opponent_action
        self._log(f"[Sim] Caching action {opponent_action} for opponent {agent}")
        return opponent_action

    def simulate_round(self, env_state, action, opponent_action_cache=None, depth=0, max_depth=40):  # Increased max_depth
        """
        Recursively simulates possible action sequences from the initial action.
        Incorporates a bait-and-switch heuristic for a one-card bluff scenario and revises outcome checks.
        
        Args:
            env_state: The current environment state.
            action: The action to simulate.
            opponent_action_cache: Dictionary to cache opponent actions based on observations.
            depth: Current recursion depth.
            max_depth: Maximum recursion depth for exploration.
            
        Returns:
            tuple: (outcome_value, action_sequence, is_terminal, is_new_round)
        """
        if opponent_action_cache is None:
            opponent_action_cache = {}  # Cache specific to this simulation branch

        self.simulations_performed += 1
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)

        starting_penalty = sim_env.penalties.get(self.training_agent, 0)
        starting_round = sim_env.round
        starting_hand = sim_env.players_hands.get(self.training_agent, [])[:]  # Copy hand state
        has_table_card_at_start = any(c == sim_env.table_card or c == "Joker" for c in starting_hand)

        action_type, card_category, count = decode_action(action)
        self._log(f"[Depth {depth}] Simulating action {action} ({action_type}, {card_category}, {count}) for {self.training_agent}")

        # --- Heuristic Check: Are we simulating the specific bluff scenario? ---
        is_simulating_one_card_bluff = (action_type == "Play" and card_category == "non-table" and count == 1)
        # ---------------------------------------------------------------------------

        action_sequence = [(self.training_agent, action)]

        # --- Take Action and Check Immediate Outcome ---
        pre_step_termination = sim_env.terminations.get(self.training_agent, False)
        sim_env.step(action)
        post_step_termination = sim_env.terminations.get(self.training_agent, False)
        post_step_penalty = sim_env.penalties.get(self.training_agent, 0)

        # --- Order of Checks (Revised) ---
        # 1. Check if game ended immediately.
        if sim_env.agent_selection is None:
            winner = sim_env.winner
            if winner == self.training_agent:
                self._log(f"[Depth {depth}] Game ended after our action {action} - WE WIN!")
                return 5000.0, action_sequence, True, False
            else:
                if not pre_step_termination and post_step_termination:
                    self._log(f"[Depth {depth}] Game ended, WE LOST (terminated by action {action})")
                    return -10000.0, action_sequence, True, False
                else:
                    self._log(f"[Depth {depth}] Game ended after our action {action} - WE LOSE (opponent won)")
                    return -5000.0, action_sequence, True, False

        # 2. Check if our action caused our elimination.
        if not pre_step_termination and post_step_termination:
            self._log(f"[Depth {depth}] Got eliminated by our action {action}! Very bad.")
            return -10000.0, action_sequence, True, False

        # 3. Check if we incurred a penalty.
        if post_step_penalty > starting_penalty:
            self._log(f"[Depth {depth}] Got penalty right after our action {action}.")
            penalty_value = -5000.0 if starting_penalty >= 2 else -1000.0
            return penalty_value, action_sequence, False, False

        # 4. Check for an immediate round change.
        if sim_env.round > starting_round:
            final_penalty = sim_env.penalties.get(self.training_agent, 0)
            is_terminated = sim_env.terminations.get(self.training_agent, False)
            if is_terminated:
                self._log(f"[Depth {depth}] ELIMINATED during immediate round change after action {action}!")
                return -10000.0, action_sequence, True, True
            if final_penalty > starting_penalty:
                self._log(f"[Depth {depth}] Got penalty during immediate round change after action {action}")
                return -5000.0, action_sequence, False, True
            self._log(f"[Depth {depth}] Round changed immediately after action {action}, no penalty.")
            return 10.0, action_sequence, False, True

        # --- Main Simulation Loop ---
        max_steps = 20  # Maximum simulation steps
        step_count = 0
        found_opponent_penalty = False

        while step_count < max_steps:
            step_count += 1
            current_sim_agent = sim_env.agent_selection
            if current_sim_agent is None:
                winner = sim_env.winner
                win_value = 5000.0 if winner == self.training_agent else -5000.0
                self._log(f"[Depth {depth}] Game ended in loop step {step_count}. Winner: {winner}. Value: {win_value}")
                return win_value, action_sequence, True, False

            # Check for round change during simulation loop.
            if sim_env.round > starting_round:
                final_penalty = sim_env.penalties.get(self.training_agent, 0)
                is_terminated = sim_env.terminations.get(self.training_agent, False)
                if is_terminated:
                    self._log(f"[Depth {depth}] ELIMINATED during round change in loop!")
                    return -10000.0, action_sequence, True, True
                if final_penalty > starting_penalty:
                    self._log(f"[Depth {depth}] Got penalty during round change in loop.")
                    return -5000.0, action_sequence, False, True
                if found_opponent_penalty:
                    self._log(f"[Depth {depth}] Round changed after finding opponent penalty. Good outcome.")
                    return 2000.0, action_sequence, False, True
                self._log(f"[Depth {depth}] Round changed in loop, no specific outcome.")
                return 10.0, action_sequence, False, True

            # --- If it's our turn ---
            if current_sim_agent == self.training_agent:
                if depth < max_depth:
                    # Recursive Exploration: Get valid actions and prioritize them.
                    sim_env.observe(self.training_agent, new=True)
                    action_mask = sim_env.infos[self.training_agent].get('action_mask', [0] * 7)
                    valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                    if not valid_actions:
                        self._log(f"[Depth {depth}] No valid actions available")
                        return -50.0, action_sequence, False, False

                    # Prioritize actions based on penalties and hand state.
                    current_penalty = sim_env.penalties.get(self.training_agent, 0)
                    hand = sim_env.players_hands.get(self.training_agent, [])
                    table_card = sim_env.table_card
                    table_cards = [c for c in hand if c == table_card or c == "Joker"]
                    prioritized_actions = []
                    if current_penalty >= 2:
                        for a in [0, 1, 2]:
                            if a in valid_actions and (a % 3) + 1 <= len(table_cards):
                                prioritized_actions.append(a)
                        for a in valid_actions:
                            if a not in prioritized_actions:
                                prioritized_actions.append(a)
                    else:
                        for a in [0, 1, 2]:
                            if a in valid_actions and (a % 3) + 1 <= len(table_cards):
                                prioritized_actions.append(a)
                        for a in [3]:
                            if a in valid_actions:
                                prioritized_actions.append(a)
                        for a in valid_actions:
                            if a not in prioritized_actions:
                                prioritized_actions.append(a)

                    best_recursive_value = float('-inf')
                    best_recursive_sequence_suffix = []  # Part after current action
                    best_recursive_is_terminal = False
                    best_recursive_is_new_round = False

                    for next_action in prioritized_actions:
                        self._log(f"[Depth {depth}] Exploring recursive action {next_action}")
                        next_state = sim_env.get_state()
                        value, next_seq, is_terminal, is_new_round = self.simulate_round(
                            next_state, next_action, opponent_action_cache, depth+1, max_depth
                        )
                        self._log(f"[Depth {depth}] Recursive action {next_action} produced value {value}, seq_len={len(next_seq)}")
                        if value > best_recursive_value:
                            best_recursive_value = value
                            best_recursive_sequence_suffix = next_seq
                            best_recursive_is_terminal = is_terminal
                            best_recursive_is_new_round = is_new_round
                            self._log(f"[Depth {depth}] New best recursive path via action {next_action} with value {value}")

                    if best_recursive_sequence_suffix:
                        combined_sequence = action_sequence + best_recursive_sequence_suffix
                        self._log(f"[Depth {depth}] Returning best recursive result. Value: {best_recursive_value}")
                        return best_recursive_value, combined_sequence, best_recursive_is_terminal, best_recursive_is_new_round
                    else:
                        self._log(f"[Depth {depth}] No valid outcomes from recursive exploration.")
                        return -100.0, action_sequence, False, False
                else:
                    # Max Depth Reached: Use a simple heuristic action.
                    self._log(f"[Depth {depth}] Max depth reached. Selecting heuristic action.")
                    sim_env.observe(self.training_agent, new=True)
                    action_mask = sim_env.infos[self.training_agent].get('action_mask', [0] * 7)
                    valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                    if not valid_actions:
                        self._log(f"[Depth {depth}] No valid actions available at max depth")
                        return -50.0, action_sequence, False, False

                    hand = sim_env.players_hands.get(self.training_agent, [])
                    table_card = sim_env.table_card
                    table_cards = [c for c in hand if c == table_card or c == "Joker"]
                    current_penalty = sim_env.penalties.get(self.training_agent, 0)
                    if 6 in valid_actions:
                        last_agent = sim_env.last_action_agent
                        if last_agent:
                            played_cards = sim_env.last_played_cards.get(self.training_agent, [])
                            is_bluff = any(card != table_card and card != "Joker" for card in played_cards)
                            if is_bluff:
                                next_action = 6
                                self._log(f"[Depth {depth}] At max depth - found bluff, challenging")
                    elif current_penalty >= 2 and 0 in valid_actions and table_cards:
                        next_action = 0
                        self._log(f"[Depth {depth}] At max depth - high penalties, playing 1 table card")
                    elif 0 in valid_actions and table_cards:
                        next_action = 0
                        self._log(f"[Depth {depth}] At max depth - playing 1 table card")
                    elif current_penalty < 2 and 3 in valid_actions and len(hand) > len(table_cards):
                        next_action = 3
                        self._log(f"[Depth {depth}] At max depth - playing 1 non-table card")
                    else:
                        next_action = valid_actions[0]
                        self._log(f"[Depth {depth}] At max depth - using first valid action: {next_action}")

                    old_penalty = sim_env.penalties.get(self.training_agent, 0)
                    sim_env.step(next_action)
                    action_sequence.append((self.training_agent, next_action))
                    if not pre_step_termination and sim_env.terminations.get(self.training_agent, False):
                        self._log(f"[Depth {depth}] Got eliminated at max depth! Very bad.")
                        return -10000.0, action_sequence, True, False
                    if sim_env.round > starting_round:
                        final_penalty = sim_env.penalties.get(self.training_agent, 0)
                        is_terminated = sim_env.terminations.get(self.training_agent, False)
                        if final_penalty > old_penalty or is_terminated:
                            self._log(f"[Depth {depth}] Got penalty or eliminated at round change at max depth")
                            return -5000.0, action_sequence, is_terminated, True
                    new_penalty = sim_env.penalties.get(self.training_agent, 0)
                    if new_penalty > old_penalty:
                        self._log(f"[Depth {depth}] Got penalty at max depth, bad")
                        return -1000.0, action_sequence, False, False
                    final_val_at_max_depth = 0.0
                    if sim_env.winner == self.training_agent:
                        final_val_at_max_depth = 5000.0
                    elif sim_env.terminations.get(self.training_agent):
                        final_val_at_max_depth = -10000.0
                    elif sim_env.agent_selection is None:
                        final_val_at_max_depth = -5000.0
                    elif sim_env.penalties.get(self.training_agent, 0) > starting_penalty:
                        final_val_at_max_depth = -1000.0
                    elif sim_env.round > starting_round:
                        final_val_at_max_depth = 10.0
                    return final_val_at_max_depth, action_sequence, (final_val_at_max_depth in [5000.0, -10000.0]), (sim_env.round > starting_round)

            # --- If it's an opponent's turn ---
            else:
                self._log(f"[Depth {depth}] Opponent {current_sim_agent}'s turn")
                try:
                    opponent_action = self._select_opponent_action(sim_env, current_sim_agent, opponent_action_cache)
                    opp_action_type, _, _ = decode_action(opponent_action)
                    self._log(f"[Depth {depth}] Opponent action selected: {opponent_action} ({opp_action_type})")

                    # --- Bait-and-Switch Heuristic ---
                    if is_simulating_one_card_bluff and opponent_action == 6 and has_table_card_at_start:
                        self._log(f"[Depth {depth}] HEURISTIC TRIGGERED: Opponent challenging 1-card bluff, we have table card.")
                        action_sequence[0] = (self.training_agent, 0)  # Replace bluff with table card play
                        action_sequence.append((current_sim_agent, 6))
                        heuristic_win_value = 3000.0
                        self._log(f"[Depth {depth}] Heuristic outcome: Assuming opponent penalty. Value: {heuristic_win_value}")
                        return heuristic_win_value, action_sequence, False, True

                    # --- Normal Opponent Action Simulation ---
                    old_penalty = sim_env.penalties.get(current_sim_agent, 0)
                    if opponent_action == 6 and sim_env.last_action_agent == self.training_agent:
                        if getattr(sim_env, "last_action_bluff", False):
                            self._log(f"[Depth {depth}] Opponent is challenging our actual bluff! Very bad.")
                            sim_env.step(opponent_action)
                            action_sequence.append((current_sim_agent, opponent_action))
                            new_round_occurred = sim_env.round > starting_round
                            if sim_env.terminations.get(self.training_agent):
                                return -10000.0, action_sequence, True, new_round_occurred
                            return -5000.0, action_sequence, False, new_round_occurred

                    sim_env.step(opponent_action)
                    action_sequence.append((current_sim_agent, opponent_action))
                    new_penalty = sim_env.penalties.get(current_sim_agent, 0)
                    if new_penalty > old_penalty:
                        found_opponent_penalty = True
                        self._log(f"[Depth {depth}] Opponent got penalty from {opponent_action}. Good.")
                except Exception as e:
                    self._log(f"[Depth {depth}] Error simulating opponent {current_sim_agent}: {e}", exc_info=True)
                    return -50.0, action_sequence, False, False

        # End of loop: Hit step limit.
        self._log(f"[Depth {depth}] Hit step limit ({max_steps})")
        final_val_at_limit = 2000.0 if found_opponent_penalty else 10.0
        return final_val_at_limit, action_sequence, False, False

    def get_next_agent_action(self, agent_whose_turn_it_is):
        self._log(f"get_next_agent_action called for {agent_whose_turn_it_is}. Pos: {self.sequence_position}, SeqLen: {len(self.action_sequence)}")
        if not self.action_sequence or self.sequence_position >= len(self.action_sequence):
            self._log("--> Returning None (No plan/End of plan)")
            return None

        expected_agent, planned_action = self.action_sequence[self.sequence_position]
        self._log(f"--> Checking Agent: Current={agent_whose_turn_it_is}, Expected={expected_agent}")

        if agent_whose_turn_it_is != expected_agent:
            self._log("--> Agent MISMATCH. Invalidating plan.")
            self.invalidate_plan()
            self._log("--> Returning None (Agent Mismatch)")
            return None

        self._log(f"--> Agent OK. Checking Action {planned_action} Validity.")
        try:
            # Determine the current action mask based on agent type.
            if agent_whose_turn_it_is == self.training_agent:
                # Update observation for training agent
                self.base_env.observe(agent_whose_turn_it_is, new=True)
                current_action_mask = self.base_env.infos[agent_whose_turn_it_is].get('action_mask', [0] * 7)
                
                # Special handling for challenge action (assuming challenge is represented by 6)
                if planned_action == 6:
                    last_agent = self.base_env.last_action_agent
                    if last_agent:
                        played_cards = self.base_env.last_played_cards.get(last_agent, [])
                        table_card = self.base_env.table_card
                        is_bluff = any(card != table_card and card != "Joker" for card in played_cards)
                        if not is_bluff:
                            self._log("--> Challenge action invalid. Opponent not bluffing. Invalidating plan.")
                            self.invalidate_plan()
                            self._log("--> Returning None (Challenge Invalid)")
                            return None
            else:
                current_action_mask = self.base_env.infos[agent_whose_turn_it_is].get('action_mask', [0] * 7)

            # Validate the planned action against the current action mask.
            if current_action_mask[planned_action] != 1:
                self._log(f"--> Action {planned_action} INVALID. Invalidating plan.")
                self.invalidate_plan()
                self._log("--> Returning None (Action Invalid)")
                return None
            else:
                self._log(f"--> Plan OK! Using action {planned_action}. Advancing position to {self.sequence_position + 1}")
                self.sequence_position += 1
                self._log(f"--> Returning Action {planned_action}")
                return planned_action

        except Exception as e:
            self._log(f"Exception during action validation: {str(e)}")
            self.invalidate_plan()
            self._log("--> Returning None (Exception)")
            return None

    def search(self, env_state):
        """
        Searches for the best action by simulating each valid action.
        Stores the best linear sequence found.

        Args:
            env_state: The environment state to start search from.

        Returns:
            tuple: (action_probs, best_action, best_value)
        """
        # --- Check for existing valid plan first ---
        # (Removed the block that used get_next_agent_action within search,
        # as the main loop handles this now. Search should always generate a fresh plan)

        # --- Start Fresh Search ---
        self.invalidate_plan() # Ensure we start with no plan
        self.simulations_performed = 0

        # Setup simulation environment
        opponent_action_cache = {} # Cache for *this search run only*
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)

        # Get initial valid actions and state info
        sim_env.observe(self.training_agent, new=True)
        action_mask = sim_env.infos[self.training_agent].get('action_mask', [0] * 7)
        valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
        if not valid_actions:
            # This should ideally not happen if called in a valid state
            self._log(f"ERROR: No valid actions available for {self.training_agent} at search start.")
            raise RuntimeError(f"No valid actions available for {self.training_agent}")

        current_penalty = sim_env.penalties.get(self.training_agent, 0)
        hand = sim_env.players_hands.get(self.training_agent, [])
        table_card = sim_env.table_card
        table_cards = [c for c in hand if c == table_card or c == "Joker"]

        # Initialize tracking variables
        best_action = -1 # Use -1 to indicate no action selected yet
        best_value = float('-inf')
        best_sequence = []

        # --- Optional: Prioritized Check - Challenge Bluff ---
        # Check if challenging a potential bluff is a good immediate option
        if 6 in valid_actions:
            last_agent = sim_env.last_action_agent
            if last_agent:
                played_cards = sim_env.last_played_cards.get(last_agent, [])
                is_bluff = any(card != table_card and card != "Joker" for card in played_cards)
                if is_bluff:
                    self._log("Found opponent bluff - trying challenge as priority")
                    # Use a fresh opponent cache for this specific simulation branch
                    challenge_sim_cache = {}
                    value, sequence, _, _ = self.simulate_round(env_state, 6, challenge_sim_cache)
                    self._log(f"Priority Challenge Result: value={value}, seq_len={len(sequence)}")
                    # Only consider it if it's distinctly positive (successful challenge)
                    if value > 1000: # Threshold for successful challenge reward
                        best_action = 6
                        best_value = value
                        best_sequence = sequence
                        self._log(f"Prioritizing successful challenge action: {best_action} with value {value}")
                        # Don't necessarily return yet, let other actions compare

        # --- Main Loop: Simulate all valid root actions ---
        # (Consider creating a prioritized list of actions to simulate if needed)
        for action in valid_actions:
            # Skip challenge if we already evaluated it and it was the best so far
            if action == 6 and best_action == 6:
                 self._log(f"Skipping re-simulation of prioritized challenge action {action}")
                 continue

            # Use a fresh opponent cache for each root simulation branch
            action_sim_cache = {}
            value, sequence, is_terminal, is_new_round = self.simulate_round(
                env_state, action, action_sim_cache, depth=0 # Ensure depth starts at 0
            )
            self._log(f"Root Action {action}: value={value}, seq_len={len(sequence)}, terminal={is_terminal}, new_round={is_new_round}")

            # --- Update Best Logic ---
            # Priority 1: Winning action
            if is_terminal and value > 0: # Check if this action leads to immediate win
                 # Check if current winner is us
                 temp_env = self.base_env.clone()
                 temp_env.set_state(env_state)
                 temp_env.step(action) # Need to simulate step to check winner
                 if temp_env.winner == self.training_agent:
                    if value > best_value: # Only take if better than current best
                         best_action = action
                         best_value = value
                         best_sequence = sequence
                         self._log(f"Found winning action: {action} with value {value}. Prioritizing.")
                         # Consider breaking if win is guaranteed and highly valued? For now, let others compare.
                    continue # Move to next action after evaluating win condition

            # Priority 2: Opponent gets penalty (using value > 1000 as proxy)
            # Check if this is better than current best, allows comparison with wins/other penalties
            if value > 1000 and value > best_value:
                 best_action = action
                 best_value = value
                 best_sequence = sequence
                 self._log(f"Found action where opponent gets penalty: {action} with value {value}")
                 # Don't break here, a winning action might be found later

            # Priority 3: Any improvement over the current worst case
            elif value > best_value: # Checks if current action's value is better than the stored best_value
                 best_action = action
                 best_value = value
                 best_sequence = sequence
                 self._log(f"New best action (general update): {action} with value {value}")
            # --- End Update Best Logic ---

        # --- Handle Fallbacks (if no suitable action found after checking all valid actions) ---
        # Condition: No action selected OR the best value found is still very bad (e.g., <= -5000)
        if best_action == -1 or best_value <= -5000: # Adjust threshold as needed
            self._log(f"No suitable action found (best_value={best_value}). Trying fallbacks.")

            fallback_action_chosen = -1
            # Fallback 1: Play 1 table card if possible (often safest)
            if 0 in valid_actions and len(table_cards) > 0:
                 fallback_action_chosen = 0
                 self._log("Fallback: Defaulting to playing 1 table card.")
            # Fallback 2: Play 1 non-table card (risky bluff)
            elif 3 in valid_actions and len(hand) > len(table_cards):
                 fallback_action_chosen = 3
                 self._log("Fallback: Playing 1 non-table card.")
            # Fallback 3: Use the very first valid action found
            else:
                 fallback_action_chosen = valid_actions[0]
                 self._log(f"Fallback: Using first valid action ({fallback_action_chosen}).")

            # If a fallback was chosen and it wasn't the already selected 'best_action'
            if fallback_action_chosen != -1 and fallback_action_chosen != best_action:
                 # Re-simulate the chosen fallback action to get its sequence and potentially updated value
                 self._log(f"Re-simulating fallback action {fallback_action_chosen} to confirm.")
                 fallback_sim_cache = {}
                 value, sequence, _, _ = self.simulate_round(env_state, fallback_action_chosen, fallback_sim_cache, depth=0)
                 self._log(f"Fallback Action {fallback_action_chosen} Result: value={value}, seq_len={len(sequence)}")

                 # Update best_action etc. ONLY if we didn't have one before, or if somehow this fallback is better
                 # Generally, we use the fallback because the original best was too bad.
                 best_action = fallback_action_chosen
                 best_value = value # Use the value from the fallback simulation
                 best_sequence = sequence

        # --- Finalize and Return ---
        self.action_sequence = best_sequence
        self.sequence_position = 0

        action_dim = 7
        action_probs = np.zeros(action_dim)
        final_best_action_to_return = -1 # Variable to hold the action being returned

        if best_action != -1:
            # Ensure best_action is within bounds before setting prob
            if 0 <= best_action < action_dim:
                 action_probs[best_action] = 1.0
                 final_best_action_to_return = best_action
            else:
                 self._log(f"ERROR: best_action '{best_action}' is out of bounds. Defaulting.")
                 # Default to first valid action if something went wrong
                 final_best_action_to_return = valid_actions[0]
                 action_probs[final_best_action_to_return] = 1.0
        else:
            # This case should ideally be unreachable due to fallbacks
            self._log("CRITICAL ERROR: No best_action selected even after fallbacks!")
            final_best_action_to_return = valid_actions[0] # Last resort
            action_probs[final_best_action_to_return] = 1.0

        # *** THE ADDED DEBUGGING LOG ***
        self._log(f"FINAL RETURN: Action={final_best_action_to_return}, Value={best_value}, SeqLen={len(self.action_sequence)}")

        return action_probs, final_best_action_to_return, best_value