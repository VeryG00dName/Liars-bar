# src/model/ps.py

import numpy as np
from src.env.liars_deck_env_utils_2 import decode_action
import torch
from src import config
import traceback # Added for more detailed error logging if needed

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
        # Use new=True for consistency with how observations might be generated elsewhere
        env.observe(agent, new=True)

        # Generate a hash key for the opponent's observation.
        # Key should represent the state the opponent acts upon.
        hand = tuple(sorted(env.players_hands.get(agent, []))) # Use tuple for hashability
        table_card = env.table_card
        last_action_val = env.last_action # Value/Count of last action
        last_action_agent_name = env.last_action_agent # Agent who made last action
        # Include opponent penalties as part of their decision context? Maybe not for cache key simplicity yet.
        # Include opponent hand size? Maybe relevant.
        current_penalties = tuple(sorted(env.penalties.items())) # Ensure order doesn't matter

        # More robust key considering relevant state for opponent decisions
        obs_key = (
            agent,
            hand,
            table_card,
            last_action_val,
            last_action_agent_name,
            # env.round, # Round might influence strategy
            current_penalties # Penalties definitely influence
        )


        # Check if we've already determined an action for this observation IN THIS SIMULATION BRANCH.
        if obs_key in opponent_action_cache:
            opponent_action = opponent_action_cache[obs_key]
            self._log(f"[Sim Cache] Using cached action {opponent_action} for opponent {agent} based on key: {obs_key}")
            # Quick validation against current mask (should ideally match if state hasn't changed unexpectedly)
            action_mask = env.infos[agent].get('action_mask', [0] * 7)
            if sum(action_mask) > 0 and action_mask[opponent_action] != 1:
                 self._log(f"[Sim Cache WARNING] Cached action {opponent_action} for {agent} is INVALID in current mask {action_mask}. Recalculating.")
                 # Fall through to recalculate if invalid
            else:
                 return opponent_action

        # --- Calculate Action ---
        opponent_model = self.opponent_models[agent]
        observation = env.observe(agent, new=True)[agent] # Get 'newer' observation
        action_mask = env.infos[agent]['action_mask']

        if sum(action_mask) == 0:
            # This case should ideally be handled before calling _select_opponent_action,
            # but adding robustness here. Returning None might be better upstream.
            self._log(f"[Sim ERROR] Agent {agent} has no valid actions according to mask {action_mask}. Raising error.")
            raise RuntimeError(f"Agent {agent} has no valid actions in simulation.")

        opponent_action = -1 # Default invalid

        # Get action based on opponent type.
        if hasattr(opponent_model, 'play_turn'):  # Hardcoded agent interface
            try:
                opponent_action = opponent_model.play_turn(observation, action_mask, table_card=env.table_card)
                # Validate action immediately
                if action_mask[opponent_action] != 1:
                    self._log(f"[Sim ERROR] Hardcoded agent {agent} returned invalid action {opponent_action} for mask {action_mask}. Fixing.")
                    valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
                    opponent_action = valid_actions[0] # Simple fix: take first valid
            except Exception as e:
                 self._log(f"[Sim ERROR] Exception in hardcoded agent {agent}'s play_turn: {e}. Fixing.")
                 valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
                 opponent_action = valid_actions[0] # Simple fix

        else:  # Historical model (neural network) interface
            # Prepare observation for NN model (assuming it takes the older format + placeholders)
            old_observation = env.observe(agent, new=False)[agent]
            obp_placeholder = np.zeros(2, dtype=np.float32)
            memory_placeholder = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
            final_obs = np.concatenate([old_observation, obp_placeholder, memory_placeholder], axis=0)
            observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device='cpu').unsqueeze(0)

            with torch.no_grad():
                try:
                    # Adjust based on your actual model's output structure
                    model_output = opponent_model(observation_tensor, None)
                    if isinstance(model_output, tuple) and len(model_output) >= 1:
                        probs = model_output[0] # Assuming first element is probs/logits
                    else:
                        probs = model_output # Assuming model directly returns probs/logits

                    probs = probs.squeeze().cpu().numpy()
                    masked_probs = probs * action_mask
                    masked_probs_sum = masked_probs.sum()

                    if masked_probs_sum > 1e-6: # Use a tolerance for floating point sums
                        masked_probs /= masked_probs_sum
                        opponent_action = np.argmax(masked_probs)
                    else:
                        # If no probability mass on valid actions, choose first valid one
                        self._log(f"[Sim WARNING] Model for {agent} produced no valid probability mass on mask {action_mask}. Probs: {probs}. Fixing.")
                        valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
                        opponent_action = valid_actions[0]

                except Exception as e:
                    self._log(f"[Sim ERROR] Exception during NN model inference for {agent}: {e}. Fixing.")
                    valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
                    opponent_action = valid_actions[0] # Simple fix

        # Final validation before caching
        if action_mask[opponent_action] != 1:
            self._log(f"[Sim FATAL] Could not determine a valid action for {agent} even after fixes. Defaulting again.")
            valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
            opponent_action = valid_actions[0] # Absolute last resort

        # Cache the determined action for this simulation state.
        opponent_action_cache[obs_key] = opponent_action
        self._log(f"[Sim Cache] Caching action {opponent_action} for opponent {agent} with key: {obs_key}")
        return opponent_action

    def simulate_round(self, env_state, action, opponent_action_cache=None, depth=0, max_depth=40):
        """
        Recursively simulates possible action sequences from the initial action.
        Focuses on finding paths leading to penalties or game end.
        Continues simulation across rounds until a definitive outcome (penalty, game end) or limits are hit.

        Args:
            env_state: The current environment state.
            action: The action to simulate.
            opponent_action_cache: Dictionary to cache opponent actions based on observations for THIS branch.
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

        # Decode the action for logging
        action_type, card_category, count = decode_action(action)
        self._log(f"[Depth {depth}, Round {starting_round}] Simulating action {action} ({action_type}, {card_category}, {count}) for {self.training_agent}")

        # Start with our action.
        action_sequence = [(self.training_agent, action)]

        # Get info before taking action.
        pre_step_termination_us = sim_env.terminations.get(self.training_agent, False)

        # Take the action.
        try:
            sim_env.step(action)
        except Exception as e:
             self._log(f"[Depth {depth} ERROR] Exception during sim_env.step({action}): {e}")
             # Treat as highly negative outcome if our first action fails
             return -10000.0, action_sequence, True, False # Assume terminal failure

        # Get info after the action.
        post_step_termination_us = sim_env.terminations.get(self.training_agent, False)
        post_step_penalty_us = sim_env.penalties.get(self.training_agent, 0)
        post_step_round = sim_env.round

        # --- Immediate Checks After Our Initial Action ---
        if sim_env.agent_selection is None: # Game ended
            winner = sim_env.winner
            value = 5000.0 if winner == self.training_agent else -5000.0
            self._log(f"[Depth {depth}] Outcome: Game ended immediately after action. Winner: {winner}")
            return value, action_sequence, True, False

        if not pre_step_termination_us and post_step_termination_us: # We got eliminated
            self._log(f"[Depth {depth}] Outcome: Eliminated immediately by own action.")
            return -10000.0, action_sequence, True, False

        if post_step_penalty_us > starting_penalty_us: # We got penalized (invalid play?)
            self._log(f"[Depth {depth}] Outcome: Penalized immediately by own action.")
            penalty_value = -5000.0 if starting_penalty_us >= 2 else -1000.0
            # Check if this penalty also caused a round change (less likely for invalid play)
            return penalty_value, action_sequence, False, post_step_round > starting_round

        # Check if round changed *immediately* (e.g., forced challenge) and evaluate consequences
        if post_step_round > starting_round:
            self._log(f"[Depth {depth}] Round changed *immediately* (to {post_step_round}). Evaluating consequences.")
            # Penalties/termination might have occurred *during* the step's internal logic
            final_penalty_us = sim_env.penalties.get(self.training_agent, 0)
            is_terminated_us = sim_env.terminations.get(self.training_agent, False)

            if is_terminated_us:
                 self._log(f"[Depth {depth}] Outcome: Eliminated during immediate round change.")
                 return -10000.0, action_sequence, True, True
            if final_penalty_us > starting_penalty_us:
                 self._log(f"[Depth {depth}] Outcome: Penalized during immediate round change.")
                 penalty_value = -5000.0 if starting_penalty_us >= 2 else -1000.0
                 return penalty_value, action_sequence, False, True
            opponent_penalized = any(
                 sim_env.penalties.get(opp, 0) > initial_opponent_penalties[opp] for opp in self.opponent_agents
            )
            if opponent_penalized:
                 self._log(f"[Depth {depth}] Outcome: Opponent penalized during immediate round change.")
                 return 2000.0, action_sequence, False, True
            # If round changed but no penalties detected (should be rare)
            self._log(f"[Depth {depth}] Outcome: Immediate round change, no penalties detected (neutral).")
            return 50.0, action_sequence, False, True

        # --- Simulation Loop ---
        max_steps_in_sim = 50 # Safety limit for steps *within* this simulation branch
        step_count = 0
        # current_sim_round tracks round *within this simulation level*
        current_sim_round = sim_env.round

        while step_count < max_steps_in_sim:
            step_count += 1

            # --- Check 1: Game Over ---
            if sim_env.agent_selection is None:
                winner = sim_env.winner
                value = 5000.0 if winner == self.training_agent else -5000.0
                self._log(f"[Depth {depth}, SimStep {step_count}] Outcome: Game ended. Winner: {winner}")
                return value, action_sequence, True, False # is_new_round is irrelevant

            # --- Check 2: Round Change & Evaluate ---
            # Must check *before* taking the step for this turn
            if sim_env.round > current_sim_round:
                self._log(f"[Depth {depth}, SimStep {step_count}] Outcome: Round changed (to {sim_env.round}). Evaluating consequences.")
                # Update tracked round *before* evaluating consequences of the change
                current_sim_round = sim_env.round

                final_penalty_us = sim_env.penalties.get(self.training_agent, 0)
                is_terminated_us = sim_env.terminations.get(self.training_agent, False)

                if is_terminated_us: # Our termination is the worst outcome
                    self._log(f"[Depth {depth}] Outcome Detail: We were eliminated.")
                    return -10000.0, action_sequence, True, True
                if final_penalty_us > starting_penalty_us: # Our penalty is next worst
                    self._log(f"[Depth {depth}] Outcome Detail: We got penalized.")
                    penalty_value = -5000.0 if starting_penalty_us >= 2 else -1000.0
                    return penalty_value, action_sequence, False, True
                opponent_penalized = any( # Opponent penalty is best outcome
                     sim_env.penalties.get(opp, 0) > initial_opponent_penalties[opp] for opp in self.opponent_agents
                )
                if opponent_penalized:
                    self._log(f"[Depth {depth}] Outcome Detail: Opponent got penalized.")
                    return 2000.0, action_sequence, False, True
                # Fallback if round changed but somehow no penalties vs start state (shouldn't happen often)
                self._log(f"[Depth {depth}] Outcome Detail: Round changed, no penalties detected (neutral).")
                return 50.0, action_sequence, False, True

            # --- Check 3: Max Depth ---
            if depth >= max_depth:
                 self._log(f"[Depth {depth}, SimStep {step_count}] Outcome: Max depth reached. Evaluating final state heuristic.")
                 final_penalty_us_at_depth = sim_env.penalties.get(self.training_agent, 0)
                 opponent_penalized_at_depth = any(
                     sim_env.penalties.get(opp, 0) > initial_opponent_penalties[opp] for opp in self.opponent_agents
                 )
                 # Simple heuristic: prioritize opponent penalty > no penalty > our penalty
                 if final_penalty_us_at_depth > starting_penalty_us: return -500.0, action_sequence, False, False
                 if opponent_penalized_at_depth: return 1000.0, action_sequence, False, False
                 return 0.0, action_sequence, False, False # Neutral if nothing happened by max depth

            # --- Determine Current Agent and Act ---
            current_agent = sim_env.agent_selection

            # --- A) Our Turn (Recursive Step) ---
            if current_agent == self.training_agent:
                sim_env.observe(self.training_agent, new=True)
                action_mask = sim_env.infos[self.training_agent].get('action_mask', [0] * 7)
                valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]

                if not valid_actions:
                    self._log(f"[Depth {depth}, SimStep {step_count}] Outcome: Stuck (no valid actions).")
                    return -50.0, action_sequence, False, False

                best_value_recursive = float('-inf')
                best_sequence_continuation = None
                best_is_terminal_recursive = False
                best_is_new_round_recursive = False

                for next_action in valid_actions:
                    self._log(f"[Depth {depth}, SimStep {step_count}] Exploring recursive action {next_action} for {self.training_agent} (Depth {depth + 1})")
                    next_state = sim_env.get_state()
                    # Recursive call uses a COPY of the cache to avoid cross-branch contamination
                    value, next_seq_cont, is_term, is_new_rnd = self.simulate_round(
                        next_state, next_action, opponent_action_cache.copy(), depth + 1, max_depth
                    )
                    self._log(f"[Depth {depth}, SimStep {step_count}] Recursive Action {next_action} Result: V={value}, Term={is_term}, NewRnd={is_new_rnd}, Len={len(next_seq_cont)}")

                    # Prioritize clear wins or opponent penalties found recursively
                    if value >= 1500:
                        self._log(f"[Depth {depth}] Selecting prioritized path via action {next_action} (Value: {value})")
                        # This branch is good enough, return its result up the call stack
                        return value, action_sequence + next_seq_cont, is_term, is_new_rnd

                    if value > best_value_recursive:
                        best_value_recursive = value
                        best_sequence_continuation = next_seq_cont
                        best_is_terminal_recursive = is_term
                        best_is_new_round_recursive = is_new_rnd
                        self._log(f"[Depth {depth}] New best recursive path via action {next_action} (Value: {value})")

                # If no branches were prioritized, return the best one found
                if best_sequence_continuation:
                    self._log(f"[Depth {depth}] Returning best recursive result (Value: {best_value_recursive})")
                    return best_value_recursive, action_sequence + best_sequence_continuation, best_is_terminal_recursive, best_is_new_round_recursive
                else:
                    # Should only happen if all recursive calls failed or returned <= -infinity
                    self._log(f"[Depth {depth}] Outcome: No suitable recursive paths found (all failed?).")
                    return -1000.0, action_sequence, False, False # Default bad outcome

            # --- B) Opponent's Turn ---
            else:
                self._log(f"[Depth {depth}, SimStep {step_count}] Opponent {current_agent}'s turn.")
                try:
                    opp_penalty_before = sim_env.penalties.get(current_agent, 0)
                    # Use the action cache specific to this simulation branch
                    opponent_action = self._select_opponent_action(sim_env, current_agent, opponent_action_cache)
                    opp_action_type, _, _ = decode_action(opponent_action)
                    self._log(f"[Depth {depth}] Opponent {current_agent} selects action {opponent_action} ({opp_action_type})")

                    # Check if opponent is challenging *our* immediately preceding action
                    is_challenging_us = (opponent_action == 6 and sim_env.last_action_agent == self.training_agent)
                    if is_challenging_us:
                        our_last_played_cards = sim_env.last_played_cards.get(self.training_agent, [])
                        table_card = sim_env.table_card
                        is_our_bluff = any(card != table_card and card != "Joker" for card in our_last_played_cards)

                        if is_our_bluff:
                             self._log(f"[Depth {depth}] Outcome: Opponent challenges our bluff! Bad.")
                             sim_env.step(opponent_action) # Apply challenge (we get penalty, round ends)
                             action_sequence.append((current_agent, opponent_action))
                             penalty_value = -10000.0 if starting_penalty_us >= 2 else -5000.0
                             return penalty_value, action_sequence, False, True # is_new_round=True

                    # Simulate opponent's action
                    sim_env.step(opponent_action)
                    action_sequence.append((current_agent, opponent_action))

                    # Check if opponent's action *immediately* penalized them (invalid move)
                    opp_penalty_after = sim_env.penalties.get(current_agent, 0)
                    if opp_penalty_after > opp_penalty_before:
                        self._log(f"[Depth {depth}] Outcome: Opponent penalized self immediately. Good.")
                        return 2000.0, action_sequence, False, False # Good outcome, continue simulation

                    # Continue loop - game end/round change checked at top

                except RuntimeError as e:
                     # Catch specific case where opponent has no valid moves in sim
                     self._log(f"[Depth {depth}, SimStep {step_count}] Opponent {current_agent} RuntimeError: {e}. Treating as stuck.")
                     return -50.0, action_sequence, False, False # Stuck state
                except Exception as e:
                    self._log(f"[Depth {depth}, SimStep {step_count}] ERROR during opponent {current_agent}'s turn: {e}")
                    self._log(traceback.format_exc())
                    return -50.0, action_sequence, False, False # Error state


        # --- Max steps reached --- (End of loop)
        self._log(f"[Depth {depth}] Outcome: Max sim steps ({max_steps_in_sim}) reached. Evaluating final state heuristic.")
        final_penalty_us_at_limit = sim_env.penalties.get(self.training_agent, 0)
        opponent_penalized_at_limit = any(
            sim_env.penalties.get(opp, 0) > initial_opponent_penalties[opp] for opp in self.opponent_agents
        )
        if final_penalty_us_at_limit > starting_penalty_us: return -500.0, action_sequence, False, False
        if opponent_penalized_at_limit: return 1000.0, action_sequence, False, False
        return -10.0, action_sequence, False, False # Default if max steps hit uneventfully


    def get_next_agent_action(self, agent_whose_turn_it_is):
        """
        Checks if the current game state matches the expected state in the plan
        and returns the planned action if valid. Also advances plan position.
        """
        self._log(f"get_next_agent_action called for {agent_whose_turn_it_is}. Pos: {self.sequence_position}, SeqLen: {len(self.action_sequence)}")
        if not self.action_sequence or self.sequence_position >= len(self.action_sequence):
            self._log("--> Return: None (No plan or end of plan)")
            return None

        # Get expected state from plan
        expected_agent, planned_action = self.action_sequence[self.sequence_position]
        self._log(f"--> Plan Expects: Agent={expected_agent}, Action={planned_action}")

        # 1. Check Agent Match
        if agent_whose_turn_it_is != expected_agent:
            self._log(f"--> FAIL: Agent Mismatch (Current: {agent_whose_turn_it_is})")
            self.invalidate_plan()
            self._log("--> Return: None (Agent Mismatch)")
            return None

        # 2. Check Action Validity (using REAL environment state)
        self._log(f"--> Agent OK. Checking Action {planned_action} validity in REAL env.")
        try:
            # Get current action mask from the base environment
            self.base_env.observe(agent_whose_turn_it_is, new=True) # Ensure infos are up-to-date
            current_action_mask = self.base_env.infos[agent_whose_turn_it_is].get('action_mask')

            if current_action_mask is None:
                 self._log(f"--> FAIL: Could not get action mask for {agent_whose_turn_it_is} in real env.")
                 self.invalidate_plan()
                 self._log("--> Return: None (Mask Error)")
                 return None

            # Check if the planned action is valid NOW
            if current_action_mask[planned_action] != 1:
                self._log(f"--> FAIL: Planned action {planned_action} is INVALID now. Mask={current_action_mask}")
                self.invalidate_plan()
                self._log("--> Return: None (Action Invalid)")
                return None

            # If all checks pass:
            self._log(f"--> OK: Plan action {planned_action} is valid.")
            # Advance position *before* returning
            self.sequence_position += 1
            self._log(f"--> Advanced plan position to {self.sequence_position}")
            self._log(f"--> Return: Action {planned_action}")
            return planned_action

        except Exception as e:
            self._log(f"--> FAIL: Exception during action validation: {e}")
            self._log(traceback.format_exc())
            self.invalidate_plan()
            self._log("--> Return: None (Exception)")
            return None

    def search(self, env_state):
        """
        Searches for the best action by simulating each valid action.
        Stores the best linear sequence found (excluding the first action).

        Args:
            env_state: The environment state to start search from.

        Returns:
            tuple: (action_probs, best_action, best_value)
                   best_action is the action for the current agent to take NOW.
                   The plan stored in self.action_sequence starts from the NEXT agent's move.
        """
        self.invalidate_plan() # Start every search fresh
        self.simulations_performed = 0

        # Setup simulation environment
        # opponent_action_cache is created fresh inside simulate_round calls now
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)

        # Get initial valid actions and state info from the *real* env state passed in
        # Need to observe in the *cloned* env to get mask corresponding to env_state
        try:
             sim_env.observe(self.training_agent, new=True)
             action_mask = sim_env.infos[self.training_agent].get('action_mask')
             if action_mask is None: raise ValueError("Action mask not found in sim_env info")
        except Exception as e:
             self._log(f"ERROR: Failed to get initial action mask in search setup: {e}")
             # Cannot proceed without a mask
             raise RuntimeError(f"Failed to get action mask for {self.training_agent} at search start") from e

        valid_actions = [i for i, mask_val in enumerate(action_mask) if mask_val == 1]
        if not valid_actions:
            self._log(f"ERROR: No valid actions available for {self.training_agent} at search start.")
            raise RuntimeError(f"No valid actions available for {self.training_agent}")

        # Get relevant state info for potential heuristics
        current_penalty = sim_env.penalties.get(self.training_agent, 0)
        hand = sim_env.players_hands.get(self.training_agent, [])
        table_card = sim_env.table_card
        table_cards_in_hand = [c for c in hand if c == table_card or c == "Joker"]

        # Initialize tracking variables
        best_action_found = -1
        best_value_found = float('-inf')
        best_sequence_found = [] # Stores the full sequence initially

        # --- Optional: Prioritized Check - Challenge Bluff ---
        if 6 in valid_actions: # Is challenging a valid action now?
            last_agent = sim_env.last_action_agent
            # Check if there was a last action and it was by an opponent
            if last_agent and last_agent != self.training_agent:
                played_cards = sim_env.last_played_cards.get(last_agent, [])
                if played_cards: # Ensure cards were actually played
                    is_bluff = any(card != table_card and card != "Joker" for card in played_cards)
                    if is_bluff:
                        self._log("Heuristic Check: Opponent's last play appears to be a bluff. Evaluating challenge first.")
                        # Use a fresh opponent cache for this specific simulation branch
                        challenge_sim_cache = {}
                        value, sequence, is_term, is_new_rnd = self.simulate_round(
                            env_state, 6, challenge_sim_cache, depth=0
                        )
                        self._log(f"Priority Challenge Result: V={value}, Term={is_term}, NewRnd={is_new_rnd}, Len={len(sequence)}")
                        # If challenge is successful (high positive value)
                        if value >= 1500: # Threshold for successful challenge
                             best_action_found = 6
                             best_value_found = value
                             best_sequence_found = sequence
                             self._log(f"Prioritizing successful challenge action: {best_action_found} with value {value}")
                             # Continue search to see if other actions are even better

        # --- Main Loop: Simulate all valid root actions ---
        for action in valid_actions:
            # Skip re-simulating challenge if it was already chosen as the best by the priority check
            if action == 6 and best_action_found == 6:
                self._log(f"Skipping re-simulation of prioritized challenge action {action}")
                continue

            # Use a fresh opponent cache for each root simulation branch
            action_sim_cache = {}
            self._log(f"--- Simulating Root Action: {action} ---")
            value, sequence, is_terminal, is_new_round = self.simulate_round(
                env_state, action, action_sim_cache, depth=0
            )
            self._log(f"--- Root Action {action} Result: V={value}, Term={is_terminal}, NewRnd={is_new_round}, Len={len(sequence)} ---")

            # --- Update Best Logic ---
            # Compare 'value' with 'best_value_found'
            if value > best_value_found:
                 self._log(f"New best action found: {action} (Value: {value} > Current Best: {best_value_found})")
                 best_value_found = value
                 best_action_found = action
                 best_sequence_found = sequence # Store the full sequence from this simulation

                 # Optional: Add a check for immediate win to potentially break early
                 if is_terminal and value > 0:
                     temp_env_check = self.base_env.clone()
                     temp_env_check.set_state(env_state)
                     temp_env_check.step(action)
                     if temp_env_check.winner == self.training_agent:
                          self._log(f"Action {action} leads to immediate win. Stopping search.")
                          break # Found a winning move, no need to check others

            # --- End Update Best Logic ---

        # --- Handle Fallbacks (if no suitable action found) ---
        if best_action_found == -1 or best_value_found <= -5000: # Threshold for "unacceptable outcome"
            self._log(f"No suitable action found (best_value={best_value_found}). Trying fallbacks.")

            fallback_action_chosen = -1
            # Define fallback priorities
            fallback_options = []
            if 0 in valid_actions and len(table_cards_in_hand) >= 1: fallback_options.append(0) # Play 1 table card
            if 3 in valid_actions and len(hand) > len(table_cards_in_hand): fallback_options.append(3) # Play 1 non-table card (bluff)
            if 6 in valid_actions: fallback_options.append(6) # Challenge if possible
            # Add other play actions if available
            for i in [1, 2, 4, 5]:
                 if i in valid_actions: fallback_options.append(i)

            if fallback_options:
                 fallback_action_chosen = fallback_options[0] # Take the highest priority fallback
                 self._log(f"Selected fallback action: {fallback_action_chosen}")
            else:
                 # This should be impossible if valid_actions was not empty initially
                 self._log("CRITICAL ERROR: No valid actions and no fallback options!")
                 fallback_action_chosen = 0 # Absolute failsafe, likely indicates error

            # If fallback is different from current best (or if no best was found)
            if fallback_action_chosen != best_action_found:
                 self._log(f"Re-simulating fallback action {fallback_action_chosen} to get sequence/value.")
                 fallback_sim_cache = {}
                 value, sequence, _, _ = self.simulate_round(env_state, fallback_action_chosen, fallback_sim_cache, depth=0)
                 self._log(f"Fallback Action {fallback_action_chosen} Result: V={value}, Len={len(sequence)}")
                 # Update based on fallback simulation
                 best_action_found = fallback_action_chosen
                 best_value_found = value
                 best_sequence_found = sequence

        # --- Finalize Plan and Return ---
        action_to_return = best_action_found # This is the action for player_0 to take NOW

        # Store the rest of the sequence (from the next step onwards)
        if best_sequence_found and len(best_sequence_found) > 1:
            self.action_sequence = best_sequence_found[1:] # Store from step 1
            self._log(f"Storing plan sequence starting from opponent (Len: {len(self.action_sequence)}): {self.action_sequence[:5]}...")
        else:
            self.action_sequence = [] # No plan if sequence is too short or empty
            self._log("Storing empty plan sequence.")
        self.sequence_position = 0 # Reset position relative to the *stored* sequence

        # Prepare return values
        action_dim = 7
        action_probs = np.zeros(action_dim)
        if 0 <= action_to_return < action_dim:
            action_probs[action_to_return] = 1.0
        else:
             # This indicates an error somewhere above if action_to_return is invalid
             self._log(f"ERROR: Final action_to_return '{action_to_return}' is invalid. Defaulting.")
             # Find *some* valid action to return
             final_valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
             action_to_return = final_valid_actions[0] if final_valid_actions else 0
             action_probs[action_to_return] = 1.0

        self._log(f"--- Search Complete ---")
        self._log(f"FINAL RETURN: Action={action_to_return}, Value={best_value_found}, Stored Plan Len={len(self.action_sequence)}")

        return action_probs, action_to_return, best_value_found