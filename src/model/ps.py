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
        self.debug = False # Set default debug state

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

    def simulate_round(self, env_state, action, opponent_action_cache=None, depth=0, max_depth=40): # Keep original max_depth
        """
        Recursively simulates possible action sequences from the initial action.
        Focuses on finding paths leading to penalties or game end.
        Continues simulation across rounds until a definitive outcome (penalty, game end) or limits are hit.

        Args:
            env_state: The current environment state.
            action: The action to simulate.
            opponent_action_cache: Dictionary to cache opponent actions based on observations.
            depth: Current recursion depth.
            max_depth: Maximum recursion depth for exploration.

        Returns:
            tuple: (outcome_value, action_sequence, is_terminal, is_new_round)
                   is_new_round indicates if the sequence ENDED with a round change.
        """
        if opponent_action_cache is None:
            opponent_action_cache = {}

        self.simulations_performed += 1

        # Clone environment and set state.
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)

        # Record initial state information
        starting_penalty_us = sim_env.penalties.get(self.training_agent, 0)
        initial_opponent_penalties = {opp: sim_env.penalties.get(opp, 0) for opp in self.opponent_agents}
        starting_round = sim_env.round
        starting_hand_size_us = len(sim_env.players_hands.get(self.training_agent, []))

        # Decode the action for logging
        action_type, card_category, count = decode_action(action)
        self._log(f"[Depth {depth}, Round {starting_round}] Simulating action {action} ({action_type}, {card_category}, {count})")

        # Start with our action.
        action_sequence = [(self.training_agent, action)]

        # Get info before taking action.
        pre_step_termination_us = sim_env.terminations.get(self.training_agent, False)
        pre_step_hand_us = sim_env.players_hands.get(self.training_agent, [])[:]

        # Take the action.
        sim_env.step(action)

        # Get info after the action.
        post_step_termination_us = sim_env.terminations.get(self.training_agent, False)
        post_step_penalty_us = sim_env.penalties.get(self.training_agent, 0)
        post_step_hand_us = sim_env.players_hands.get(self.training_agent, [])[:]
        cards_played = [c for c in pre_step_hand_us if c not in post_step_hand_us]

        self._log(f"[Depth {depth}, Round {sim_env.round}] After action: Penalty_Us={post_step_penalty_us}, Hand size={len(post_step_hand_us)}, Cards played={cards_played}")

        # --- Immediate Checks After Our Initial Action ---

        # Check if game over immediately
        if sim_env.agent_selection is None:
            winner = sim_env.winner
            value = 5000.0 if winner == self.training_agent else -5000.0
            self._log(f"[Depth {depth}] Game ended immediately after our action - Winner: {winner}")
            return value, action_sequence, True, False # is_terminal=True, is_new_round irrelevant

        # Check if we got eliminated by our action
        if not pre_step_termination_us and post_step_termination_us:
            self._log(f"[Depth {depth}] Got eliminated by our own action! Very bad.")
            return -10000.0, action_sequence, True, False # is_terminal=True

        # Check if we got an immediate penalty from our action (e.g., invalid play)
        if post_step_penalty_us > starting_penalty_us:
            self._log(f"[Depth {depth}] Got penalty immediately after our action.")
            penalty_value = -5000.0 if starting_penalty_us >= 2 else -1000.0
             # This action itself doesn't guarantee round end, so is_new_round=False
            return penalty_value, action_sequence, False, False

        # Check if the round changed *immediately* due to our action (e.g., forced challenge)
        # This needs careful checking *if* a penalty occurred during this immediate change
        if sim_env.round > starting_round:
            self._log(f"[Depth {depth}] Round changed *immediately* after our action (now Round {sim_env.round}).")
            # Re-evaluate penalties *after* the immediate round change logic in step()
            final_penalty_us_after_immediate_change = sim_env.penalties.get(self.training_agent, 0)
            is_terminated_us_after_immediate_change = sim_env.terminations.get(self.training_agent, False)

            if is_terminated_us_after_immediate_change:
                 self._log(f"[Depth {depth}] Eliminated during immediate round change.")
                 return -10000.0, action_sequence, True, True

            if final_penalty_us_after_immediate_change > starting_penalty_us:
                 self._log(f"[Depth {depth}] Got penalty during immediate round change.")
                 penalty_value = -5000.0 if starting_penalty_us >= 2 else -1000.0
                 return penalty_value, action_sequence, False, True

            opponent_penalized_at_immediate_change = any(
                 sim_env.penalties.get(opp, 0) > initial_opponent_penalties[opp] for opp in self.opponent_agents
            )
            if opponent_penalized_at_immediate_change:
                 self._log(f"[Depth {depth}] Opponent penalized during immediate round change.")
                 return 2000.0, action_sequence, False, True # Good outcome

            # If round changed immediately but no penalties detected (should be rare?)
            self._log(f"[Depth {depth}] Immediate round change, no penalties detected.")
            return 50.0, action_sequence, False, True # Neutral outcome

        # --- Simulation Loop ---
        max_steps = 50 # Increased limit, relying on penalty/game end primarily
        step_count = 0
        current_sim_round = sim_env.round # Track the round *within* the simulation loop

        while step_count < max_steps:
            step_count += 1

            # --- Check 1: Game Over ---
            if sim_env.agent_selection is None:
                winner = sim_env.winner
                value = 5000.0 if winner == self.training_agent else -5000.0
                self._log(f"[Depth {depth}, SimStep {step_count}] Game ended. Winner: {winner}")
                # is_new_round is False because game ended, round change isn't the defining feature
                return value, action_sequence, True, False

            # --- Check 2: Round Change Detection & Penalty Check ---
            # Detect if the round has advanced *since the last iteration* or initial state
            if sim_env.round > current_sim_round:
                self._log(f"[Depth {depth}, SimStep {step_count}] Detected round change (now Round {sim_env.round} from {current_sim_round}). Evaluating outcome AT change.")
                current_sim_round = sim_env.round # Update tracked round

                # Get state *at the moment the round change is detected*
                final_penalty_us = sim_env.penalties.get(self.training_agent, 0)
                is_terminated_us = sim_env.terminations.get(self.training_agent, False)

                # Check outcome priority: Our termination/penalty first
                if is_terminated_us:
                    self._log(f"[Depth {depth}] We got ELIMINATED during round change!")
                    return -10000.0, action_sequence, True, True # is_terminal=True, is_new_round=True

                if final_penalty_us > starting_penalty_us:
                    self._log(f"[Depth {depth}] We got penalty during round change (current={final_penalty_us}, start={starting_penalty_us}).")
                    penalty_value = -5000.0 if starting_penalty_us >= 2 else -1000.0
                    return penalty_value, action_sequence, False, True # is_terminal=False, is_new_round=True

                # Check if ANY OPPONENT got a penalty AT this round change
                opponent_penalized_at_change = False
                for opp in self.opponent_agents:
                     opp_penalty_now = sim_env.penalties.get(opp, 0)
                     if opp_penalty_now > initial_opponent_penalties[opp]:
                         self._log(f"[Depth {depth}] Opponent {opp} got penalty during round change (current={opp_penalty_now}, start={initial_opponent_penalties[opp]}).")
                         opponent_penalized_at_change = True
                         break # Found at least one

                if opponent_penalized_at_change:
                     # This is a primary positive outcome
                     return 2000.0, action_sequence, False, True # is_terminal=False, is_new_round=True

                # If round changed, but NO penalties detected for anyone vs *initial* state (very unlikely if env logic is correct)
                self._log(f"[Depth {depth}] WARNING: Round changed but no penalty detected vs initial state. Returning neutral.")
                return 50.0, action_sequence, False, True # is_terminal=False, is_new_round=True
                # If this log appears frequently, it indicates a potential issue in the environment's penalty application on round end.

            # --- Check 3: Whose Turn ---
            current_agent = sim_env.agent_selection

            # --- Our Turn ---
            if current_agent == self.training_agent:
                if depth >= max_depth:
                    # Max depth reached - Use heuristic, take one step, and return evaluation
                    self._log(f"[Depth {depth}, SimStep {step_count}] Max depth reached. Using heuristic.")
                    # ... (existing heuristic logic to choose next_action) ...
                    # (Ensure heuristic doesn't cause infinite loop if called repeatedly)
                    heuristic_action = valid_actions[0] # Simplistic fallback heuristic
                    if 6 in valid_actions: # Simple heuristic: challenge if possible
                        heuristic_action = 6
                    elif 0 in valid_actions: # Simple heuristic: play 1 table card if possible
                         heuristic_action = 0

                    self._log(f"[Depth {depth}] Heuristic action: {heuristic_action}")
                    sim_env.step(heuristic_action)
                    action_sequence.append((self.training_agent, heuristic_action))

                    # Evaluate state *after* this one heuristic step and *stop this branch*
                    penalty_after_heuristic = sim_env.penalties.get(self.training_agent, 0)
                    terminated_after_heuristic = sim_env.terminations.get(self.training_agent, False)
                    round_after_heuristic = sim_env.round

                    # Check for immediate negative consequences
                    if terminated_after_heuristic: return -10000.0, action_sequence, True, round_after_heuristic > current_sim_round # Use current_sim_round
                    if penalty_after_heuristic > starting_penalty_us: return -1000.0, action_sequence, False, round_after_heuristic > current_sim_round

                    # Check if opponent got penalized by our heuristic action (e.g. we challenged successfully)
                    opponent_penalized_by_heuristic = any(
                         sim_env.penalties.get(opp, 0) > initial_opponent_penalties[opp] for opp in self.opponent_agents
                    )
                    if opponent_penalized_by_heuristic: return 1500.0, action_sequence, False, round_after_heuristic > current_sim_round # Slightly less than normal path?

                    # Check if round changed due to heuristic
                    if round_after_heuristic > current_sim_round: return 50.0, action_sequence, False, True

                    # If nothing major happened, return neutral value indicating max depth stop
                    return 0.0, action_sequence, False, False

                else:
                    # --- Explore valid actions recursively (Standard Perfect Search step) ---
                    sim_env.observe(self.training_agent, new=True) # Get mask for current state
                    action_mask = sim_env.infos[self.training_agent].get('action_mask', [0] * 7)
                    valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                    if not valid_actions:
                        self._log(f"[Depth {depth}] No valid actions available for {self.training_agent}. Stuck.")
                        return -50.0, action_sequence, False, False # Stuck state

                    best_value = float('-inf')
                    best_sequence_continuation = None
                    best_is_terminal = False
                    best_is_new_round = False

                    # --- Explore each valid action by recursing ---
                    for next_action in valid_actions:
                        self._log(f"[Depth {depth}] Exploring recursive action {next_action} (will increment depth to {depth + 1})")
                        next_state = sim_env.get_state()
                        # Recursive call evaluates the consequences starting from *next_action*
                        value, next_seq_continuation, is_terminal, is_new_round = self.simulate_round(
                            next_state, next_action, opponent_action_cache.copy(), depth + 1, max_depth
                        )
                        self._log(f"[Depth {depth}] Recursive Action {next_action} returned: value={value}, term={is_terminal}, new_rnd={is_new_round}, seq_len={len(next_seq_continuation)}")

                        # Prioritize immediate wins or opponent penalties found recursively
                        if value >= 1500: # Found win or opponent penalty down this path
                             self._log(f"[Depth {depth}] Prioritizing good outcome (value={value}) from recursive action {next_action}")
                             # Combine current sequence with the successful continuation
                             return value, action_sequence + next_seq_continuation, is_terminal, is_new_round

                        # Update best outcome found so far
                        if value > best_value:
                            best_value = value
                            # The sequence returned includes the next_action we simulated
                            best_sequence_continuation = next_seq_continuation
                            best_is_terminal = is_terminal
                            best_is_new_round = is_new_round
                            self._log(f"[Depth {depth}] New best recursive path via action {next_action} with value {value}")

                    # After exploring all actions at this depth
                    if best_sequence_continuation:
                        # Return the best outcome found through recursion
                        # Combine our current sequence prefix with the best continuation found
                        return best_value, action_sequence + best_sequence_continuation, best_is_terminal, best_is_new_round
                    else:
                        # This case means all recursive paths led to extremely negative outcomes or errors
                        self._log(f"[Depth {depth}] No suitable recursive paths found. Returning very negative.")
                        # Return the most negative value encountered, or a default bad value
                        return best_value if best_value > float('-inf') else -1000.0, action_sequence, False, False


            # --- Opponent's Turn ---
            else:
                self._log(f"[Depth {depth}, SimStep {step_count}] Opponent {current_agent}'s turn (Round {current_sim_round})")
                try:
                    # Store opponent penalty before their action
                    opp_penalty_before = sim_env.penalties.get(current_agent, 0)
                    opponent_action = self._select_opponent_action(sim_env, current_agent, opponent_action_cache)
                    opp_action_type, _, _ = decode_action(opponent_action)
                    self._log(f"[Depth {depth}] Opponent {current_agent} selected action {opponent_action} ({opp_action_type})")

                    # --- Special Check: Opponent challenges US ---
                    # Check if they are challenging the action *we* took just before them
                    is_challenging_us = (opponent_action == 6 and sim_env.last_action_agent == self.training_agent)
                    if is_challenging_us:
                        # We need to know if *our* last action (which is being challenged) was a bluff
                        our_last_played_cards = sim_env.last_played_cards.get(self.training_agent, [])
                        table_card = sim_env.table_card
                        is_our_bluff = any(card != table_card and card != "Joker" for card in our_last_played_cards)

                        if is_our_bluff:
                            self._log(f"[Depth {depth}] Opponent {current_agent} is challenging our bluff! Very bad.")
                            # Simulate the challenge step
                            sim_env.step(opponent_action) # This applies penalty to us and ends round
                            action_sequence.append((current_agent, opponent_action))
                            # Return immediate high negative value. The round change will be detected next loop if needed, but penalty is key.
                            penalty_value = -10000.0 if starting_penalty_us >= 2 else -5000.0
                            # Challenge success/fail always ends round
                            return penalty_value, action_sequence, False, True # is_new_round=True

                    # --- Simulate opponent's action ---
                    sim_env.step(opponent_action)
                    action_sequence.append((current_agent, opponent_action))

                    # --- Check for Direct Opponent Penalty ---
                    # Check if the opponent's action *directly* resulted in a penalty for them (e.g., invalid action)
                    opp_penalty_after = sim_env.penalties.get(current_agent, 0)
                    if opp_penalty_after > opp_penalty_before:
                        self._log(f"[Depth {depth}] Opponent {current_agent} got penalty immediately from their own action {opponent_action}. Good outcome.")
                        # This is a positive outcome. Return now.
                        # The round may or may not end here, check at loop start handles it.
                        return 2000.0, action_sequence, False, False # is_new_round=False (for now)

                    # No need to check round change or game end here, it's handled comprehensively at the start of the loop.

                except Exception as e:
                    self._log(f"[Depth {depth}, SimStep {step_count}] Error during opponent {current_agent}'s turn: {e}")
                    import traceback
                    self._log(traceback.format_exc())
                    # Error during simulation, return negative value
                    return -50.0, action_sequence, False, False


        # --- Max steps reached ---
        # This should ideally not be the primary exit if env logic forces penalties, but acts as a failsafe
        self._log(f"[Depth {depth}] Hit step limit ({max_steps}) before penalty/game end. Evaluating final state.")
        final_penalty_us_at_limit = sim_env.penalties.get(self.training_agent, 0)
        opponent_penalized_at_limit = any(
            sim_env.penalties.get(opp, 0) > initial_opponent_penalties[opp] for opp in self.opponent_agents
        )

        if final_penalty_us_at_limit > starting_penalty_us:
             return -500.0, action_sequence, False, False # Penalized before max steps
        if opponent_penalized_at_limit:
            return 1000.0, action_sequence, False, False # Found opp penalty but hit step limit
        else:
            # Default slightly negative/neutral outcome if max steps hit without significant events
            self._log(f"[Depth {depth}] Max steps reached, no significant penalties detected. Neutral/Small negative return.")
            return -10.0, action_sequence, False, False # Indicate less desirable than a clean outcome

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