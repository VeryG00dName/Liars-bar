# src/model/ps.py

import numpy as np
from src.env.liars_deck_env_utils_2 import decode_action
import torch
from src import config
import traceback # Added for more detailed error logging if needed

class PerfectSearch:
    """
    Perfect Search algorithm for Liar's Deck.
    Prioritizes actions leading to opponent penalties and stops search early if found.
    """

    def __init__(self, env, training_agent, opponent_models):
        self.base_env = env
        self.training_agent = training_agent
        self.opponent_models = opponent_models
        self.opponent_agents = [agent for agent in env.possible_agents if agent != training_agent]
        self.action_sequence = []
        self.sequence_position = 0
        self.simulations_performed = 0
        self.debug = False
        # --- Define threshold for "opponent penalized" outcome ---
        self.OPPONENT_PENALTY_THRESHOLD = 1500.0 # Value >= this means an opponent likely got penalized

    def _log(self, message):
        if self.debug:
            print(f"PS DEBUG: {message}")

    def invalidate_plan(self):
        if self.action_sequence:
            self._log("Plan invalidated due to deviation, invalid action, or new round.")
        self.action_sequence = []
        self.sequence_position = 0

    def _select_opponent_action(self, env, agent, opponent_action_cache):
        # Ensure we've observed the agent to generate infos.
        env.observe(agent, new=True)

        # Generate a hash key for the opponent's observation.
        hand = tuple(sorted(env.players_hands.get(agent, [])))
        table_card = env.table_card
        last_action_val = env.last_action
        last_action_agent_name = env.last_action_agent
        current_penalties = tuple(sorted(env.penalties.items()))

        obs_key = (
            agent, hand, table_card, last_action_val,
            last_action_agent_name, current_penalties
        )

        if obs_key in opponent_action_cache:
            opponent_action = opponent_action_cache[obs_key]
            self._log(f"[Sim Cache] Using cached action {opponent_action} for opponent {agent} based on key: {obs_key}")
            action_mask = env.infos[agent].get('action_mask', [0] * 7)
            if sum(action_mask) > 0 and action_mask[opponent_action] != 1:
                 self._log(f"[Sim Cache WARNING] Cached action {opponent_action} for {agent} is INVALID in current mask {action_mask}. Recalculating.")
            else:
                 return opponent_action

        # --- Calculate Action ---
        opponent_model = self.opponent_models[agent]
        observation = env.observe(agent, new=True)[agent]
        action_mask = env.infos[agent]['action_mask']

        if sum(action_mask) == 0:
            self._log(f"[Sim ERROR] Agent {agent} has no valid actions according to mask {action_mask}. Raising error.")
            raise RuntimeError(f"Agent {agent} has no valid actions in simulation.")

        opponent_action = -1

        if hasattr(opponent_model, 'play_turn'):  # Hardcoded agent interface
            try:
                opponent_action = opponent_model.play_turn(observation, action_mask, table_card=env.table_card)
                if action_mask[opponent_action] != 1:
                    self._log(f"[Sim ERROR] Hardcoded agent {agent} returned invalid action {opponent_action} for mask {action_mask}. Fixing.")
                    valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
                    opponent_action = valid_actions[0]
            except Exception as e:
                 self._log(f"[Sim ERROR] Exception in hardcoded agent {agent}'s play_turn: {e}. Fixing.")
                 valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
                 opponent_action = valid_actions[0]

        else:  # Historical model (neural network) interface
            old_observation = env.observe(agent, new=False)[agent]
            obp_placeholder = np.zeros(2, dtype=np.float32)
            memory_placeholder = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
            final_obs = np.concatenate([old_observation, obp_placeholder, memory_placeholder], axis=0)
            observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device='cpu').unsqueeze(0)

            with torch.no_grad():
                try:
                    model_output = opponent_model(observation_tensor, None)
                    if isinstance(model_output, tuple) and len(model_output) >= 1:
                        probs = model_output[0]
                    else:
                        probs = model_output

                    probs = probs.squeeze().cpu().numpy()
                    masked_probs = probs * action_mask
                    masked_probs_sum = masked_probs.sum()

                    if masked_probs_sum > 1e-6:
                        masked_probs /= masked_probs_sum
                        opponent_action = np.argmax(masked_probs)
                    else:
                        self._log(f"[Sim WARNING] Model for {agent} produced no valid probability mass on mask {action_mask}. Probs: {probs}. Fixing.")
                        valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
                        opponent_action = valid_actions[0]

                except Exception as e:
                    self._log(f"[Sim ERROR] Exception during NN model inference for {agent}: {e}. Fixing.")
                    valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
                    opponent_action = valid_actions[0]

        if action_mask[opponent_action] != 1:
            self._log(f"[Sim FATAL] Could not determine a valid action for {agent} even after fixes. Defaulting again.")
            valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
            opponent_action = valid_actions[0]

        opponent_action_cache[obs_key] = opponent_action
        self._log(f"[Sim Cache] Caching action {opponent_action} for opponent {agent} with key: {obs_key}")
        return opponent_action

    def simulate_round(self, env_state, action, opponent_action_cache=None, depth=0, max_depth=40):
        """
        Recursively simulates possible action sequences from the initial action.
        Includes heuristic early exit if opponent challenges a bluff but table cards were possible.
        Stops simulation early if opponent penalties are detected or game ends.
        """
        if opponent_action_cache is None:
            opponent_action_cache = {}

        self.simulations_performed += 1
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)
        starting_penalty_us = sim_env.penalties.get(self.training_agent, 0)
        initial_opponent_penalties = {opp: sim_env.penalties.get(opp, 0) for opp in self.opponent_agents}
        starting_round = sim_env.round
        action_type, card_category, count = decode_action(action)
        self._log(f"[Depth {depth}, Round {starting_round}] Simulating action {action} ({action_type}, {card_category}, {count}) for {self.training_agent}")
        action_sequence = [(self.training_agent, action)] # Store initial action
        pre_step_termination_us = sim_env.terminations.get(self.training_agent, False)

        try:
            sim_env.step(action) # Apply our initial action
        except Exception as e:
             self._log(f"[Depth {depth} ERROR] Exception during initial sim_env.step({action}): {e}")
             return -10000.0, action_sequence, True, False

        post_step_termination_us = sim_env.terminations.get(self.training_agent, False)
        post_step_penalty_us = sim_env.penalties.get(self.training_agent, 0)
        post_step_round = sim_env.round

        # --- Immediate Checks After Our Initial Action ---
        if sim_env.agent_selection is None:
            winner = sim_env.winner
            value = 5000.0 if winner == self.training_agent else -5000.0
            self._log(f"[Depth {depth}] Outcome: Game ended immediately after action. Winner: {winner}")
            return value, action_sequence, True, False
        if not pre_step_termination_us and post_step_termination_us:
            self._log(f"[Depth {depth}] Outcome: Eliminated immediately by own action.")
            return -10000.0, action_sequence, True, False
        if post_step_penalty_us > starting_penalty_us:
            self._log(f"[Depth {depth}] Outcome: Penalized immediately by own action.")
            penalty_value = -5000.0 if starting_penalty_us >= 2 else -1000.0
            return penalty_value, action_sequence, False, post_step_round > starting_round
        if post_step_round > starting_round:
            self._log(f"[Depth {depth}] Round changed *immediately* (to {post_step_round}). Evaluating consequences.")
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
                 # Return the defined positive value for opponent penalty
                 return self.OPPONENT_PENALTY_THRESHOLD, action_sequence, False, True
            self._log(f"[Depth {depth}] Outcome: Immediate round change, no penalties detected (neutral).")
            return 50.0, action_sequence, False, True

        # --- Simulation Loop ---
        max_steps_in_sim = 50
        step_count = 0
        current_sim_round = sim_env.round

        while step_count < max_steps_in_sim:
            step_count += 1

            # --- Check 1: Game Over ---
            if sim_env.agent_selection is None:
                winner = sim_env.winner
                value = 5000.0 if winner == self.training_agent else -5000.0
                self._log(f"[Depth {depth}, SimStep {step_count}] Outcome: Game ended. Winner: {winner}")
                return value, action_sequence, True, False

            # --- Check 2: Round Change & Evaluate ---
            if sim_env.round > current_sim_round:
                self._log(f"[Depth {depth}, SimStep {step_count}] Outcome: Round changed (to {sim_env.round}). Evaluating consequences.")
                current_sim_round = sim_env.round
                final_penalty_us = sim_env.penalties.get(self.training_agent, 0)
                is_terminated_us = sim_env.terminations.get(self.training_agent, False)
                if is_terminated_us:
                    self._log(f"[Depth {depth}] Outcome Detail: We were eliminated.")
                    return -10000.0, action_sequence, True, True
                if final_penalty_us > starting_penalty_us:
                    self._log(f"[Depth {depth}] Outcome Detail: We got penalized.")
                    penalty_value = -5000.0 if starting_penalty_us >= 2 else -1000.0
                    return penalty_value, action_sequence, False, True
                opponent_penalized = any(
                     sim_env.penalties.get(opp, 0) > initial_opponent_penalties[opp] for opp in self.opponent_agents
                )
                if opponent_penalized:
                    self._log(f"[Depth {depth}] Outcome Detail: Opponent got penalized.")
                    # Return the defined positive value for opponent penalty
                    return self.OPPONENT_PENALTY_THRESHOLD, action_sequence, False, True
                self._log(f"[Depth {depth}] Outcome Detail: Round changed, no penalties detected (neutral).")
                return 50.0, action_sequence, False, True

            # --- Check 3: Max Depth ---
            if depth >= max_depth:
                 self._log(f"[Depth {depth}, SimStep {step_count}] Outcome: Max depth reached. Evaluating final state heuristic.")
                 final_penalty_us_at_depth = sim_env.penalties.get(self.training_agent, 0)
                 opponent_penalized_at_depth = any(
                     sim_env.penalties.get(opp, 0) > initial_opponent_penalties[opp] for opp in self.opponent_agents
                 )
                 if final_penalty_us_at_depth > starting_penalty_us: return -500.0, action_sequence, False, False
                 # Return value slightly less than OPPONENT_PENALTY_THRESHOLD for opponent penalty at max depth
                 if opponent_penalized_at_depth: return self.OPPONENT_PENALTY_THRESHOLD - 500.0, action_sequence, False, False
                 return 0.0, action_sequence, False, False

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
                simulation_order = [3, 5, 4, 0, 2, 1]
                actions_to_simulate = [act for act in simulation_order if act in valid_actions]
                for next_action in actions_to_simulate:
                    self._log(f"[Depth {depth}, SimStep {step_count}] Exploring recursive action {next_action} for {self.training_agent} (Depth {depth + 1})")
                    next_state = sim_env.get_state()
                    value, next_seq_cont, is_term, is_new_rnd = self.simulate_round(
                        next_state, next_action, opponent_action_cache.copy(), depth + 1, max_depth
                    )
                    self._log(f"[Depth {depth}, SimStep {step_count}] Recursive Action {next_action} Result: V={value}, Term={is_term}, NewRnd={is_new_rnd}, Len={len(next_seq_cont)}")

                    # Use the defined threshold for prioritizing opponent penalties
                    if value >= self.OPPONENT_PENALTY_THRESHOLD:
                        self._log(f"[Depth {depth}] Selecting prioritized path via action {next_action} (Value: {value})")
                        return value, action_sequence + next_seq_cont, is_term, is_new_rnd

                    if value > best_value_recursive:
                        best_value_recursive = value
                        best_sequence_continuation = next_seq_cont
                        best_is_terminal_recursive = is_term
                        best_is_new_round_recursive = is_new_rnd
                        self._log(f"[Depth {depth}] New best recursive path via action {next_action} (Value: {value})")

                if best_sequence_continuation:
                    self._log(f"[Depth {depth}] Returning best recursive result (Value: {best_value_recursive})")
                    return best_value_recursive, action_sequence + best_sequence_continuation, best_is_terminal_recursive, best_is_new_round_recursive
                else:
                    self._log(f"[Depth {depth}] Outcome: No suitable recursive paths found (all failed?).")
                    return -1000.0, action_sequence, False, False

            # --- B) Opponent's Turn ---
            else:
                self._log(f"[Depth {depth}, SimStep {step_count}] Opponent {current_agent}'s turn.")
                try:
                    opp_penalty_before = sim_env.penalties.get(current_agent, 0)
                    opponent_action = self._select_opponent_action(sim_env, current_agent, opponent_action_cache)
                    opp_action_type, _, _ = decode_action(opponent_action)
                    self._log(f"[Depth {depth}] Opponent {current_agent} selects action {opponent_action} ({opp_action_type})")

                    is_challenging_us = (opponent_action == 6 and sim_env.last_action_agent == self.training_agent)

                    # --- MODIFIED CHALLENGE HANDLING ---
                    if is_challenging_us:
                        our_last_played_cards = sim_env.last_played_cards.get(self.training_agent, []) # Cards from initial action
                        table_card = sim_env.table_card
                        is_our_bluff = any(card != table_card and card != "Joker" for card in our_last_played_cards)

                        if is_our_bluff:
                            # --- START HEURISTIC IMPLEMENTATION ---
                            initial_action_type, _, initial_count = decode_action(action) # Decode the *first* action of this sim

                            if initial_action_type == 'Play' and action >= 3: # Was it a Play Non-Table action?
                                # Check initial hand state passed into this simulation call
                                initial_hand = env_state['players_hands'].get(self.training_agent, [])
                                table_cards_in_initial_hand = [c for c in initial_hand if c == table_card or c == "Joker"]
                                num_table_cards_available = len(table_cards_in_initial_hand)

                                # Check if the required count for the *initial* action could have been met with table cards
                                if num_table_cards_available >= initial_count:
                                    # HEURISTIC APPLIES: Pretend we played table cards
                                    self._log(f"[Depth {depth}] HEURISTIC: Opponent challenges bluff, but swapping last play to table cards (had {num_table_cards_available} available for initial count {initial_count}).")
                                    # Calculate the equivalent table action based on the count of the *initial* action
                                    equivalent_table_action = action - 3

                                    # --- CORRECTED LOGIC ---
                                    # Modify the *last* action in the sequence, which was our bluff that is now being challenged.
                                    # Ensure the last action was indeed ours (should be guaranteed by is_challenging_us check)
                                    if action_sequence and action_sequence[-1][0] == self.training_agent:
                                        original_action_being_swapped = action_sequence[-1][1]
                                        action_sequence[-1] = (self.training_agent, equivalent_table_action)
                                        self._log(f"[Depth {depth}] HEURISTIC: Swapped our last action {original_action_being_swapped} to equivalent table action {equivalent_table_action}.")
                                    else:
                                        # Log a warning if something unexpected happened
                                        self._log(f"[Depth {depth}] HEURISTIC WARNING: Could not find our last action in sequence to swap. Sequence: {action_sequence}")
                                        # Fallback: Treat as a failed bluff (might be safer than proceeding with potentially incorrect sequence)
                                        penalty_value = -10000.0 if starting_penalty_us >= 2 else -5000.0
                                        action_sequence.append((current_agent, opponent_action)) # Add challenge before returning
                                        return penalty_value, action_sequence, False, True
                                    # --- END CORRECTED LOGIC ---

                                    # Append the opponent's challenge action *after* the swap
                                    action_sequence.append((current_agent, opponent_action))
                                    # Return POSITIVE value (opponent penalized because our swapped play is now valid)
                                    return self.OPPONENT_PENALTY_THRESHOLD, action_sequence, False, True
                                else:
                                     # HEURISTIC DOES NOT APPLY: Standard bad outcome
                                     self._log(f"[Depth {depth}] Outcome: Opponent challenges bluff! Bad. (Only had {num_table_cards_available} table cards for initial count {initial_count})")
                                     penalty_value = -10000.0 if starting_penalty_us >= 2 else -5000.0
                                     action_sequence.append((current_agent, opponent_action)) # Add challenge before returning
                                     return penalty_value, action_sequence, False, True # Returns penalty, indicates round potentially ends
                            else:
                                # Safety check: Bluff detected but initial action wasn't Play Non-Table or count mismatch
                                self._log(f"[Depth {depth}] Outcome: Opponent challenges bluff, initial action mismatch or invalid count? Applying standard penalty.")
                                penalty_value = -10000.0 if starting_penalty_us >= 2 else -5000.0
                                action_sequence.append((current_agent, opponent_action))
                                return penalty_value, action_sequence, False, True # Returns penalty, indicates round potentially ends
                        # else: Opponent challenged our VALID play. Let sim_env.step handle it below.

                    # --- Apply opponent action if not handled by heuristic/bluff return ---
                    # This part is reached only if it wasn't our bluff being challenged, or if it was our valid play being challenged.
                    sim_env.step(opponent_action)
                    action_sequence.append((current_agent, opponent_action))

                    # Check if opponent's action immediately penalized them (e.g., invalid challenge, invalid play)
                    opp_penalty_after = sim_env.penalties.get(current_agent, 0)
                    if opp_penalty_after > opp_penalty_before:
                        self._log(f"[Depth {depth}] Outcome: Opponent penalized self immediately. Good.")
                        # Use the defined positive value for opponent penalty
                        return self.OPPONENT_PENALTY_THRESHOLD, action_sequence, False, False


                except RuntimeError as e:
                     self._log(f"[Depth {depth}, SimStep {step_count}] Opponent {current_agent} RuntimeError: {e}. Treating as stuck.")
                     return -50.0, action_sequence, False, False
                except Exception as e:
                    self._log(f"[Depth {depth}, SimStep {step_count}] ERROR during opponent {current_agent}'s turn: {e}")
                    self._log(traceback.format_exc())
                    return -50.0, action_sequence, False, False

        # --- Max steps reached ---
        self._log(f"[Depth {depth}] Outcome: Max sim steps ({max_steps_in_sim}) reached. Evaluating final state heuristic.")
        final_penalty_us_at_limit = sim_env.penalties.get(self.training_agent, 0)
        opponent_penalized_at_limit = any(
            sim_env.penalties.get(opp, 0) > initial_opponent_penalties[opp] for opp in self.opponent_agents
        )
        if final_penalty_us_at_limit > starting_penalty_us: return -500.0, action_sequence, False, False
        if opponent_penalized_at_limit: return self.OPPONENT_PENALTY_THRESHOLD - 500.0, action_sequence, False, False
        return -10.0, action_sequence, False, False


    def get_next_agent_action(self, agent_whose_turn_it_is):
        """
        Checks if the current game state matches the expected state in the plan
        and returns the planned action if valid. Also advances plan position.
        REMOVED the special challenge validation check.
        """
        self._log(f"get_next_agent_action called for {agent_whose_turn_it_is}. Pos: {self.sequence_position}, SeqLen: {len(self.action_sequence)}")
        if not self.action_sequence or self.sequence_position >= len(self.action_sequence):
            self._log("--> Return: None (No plan or end of plan)")
            return None

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
            self.base_env.observe(agent_whose_turn_it_is, new=True)
            current_action_mask = self.base_env.infos[agent_whose_turn_it_is].get('action_mask')

            if current_action_mask is None:
                 self._log(f"--> FAIL: Could not get action mask for {agent_whose_turn_it_is} in real env.")
                 self.invalidate_plan()
                 self._log("--> Return: None (Mask Error)")
                 return None

            if current_action_mask[planned_action] != 1:
                self._log(f"--> FAIL: Planned action {planned_action} is INVALID now. Mask={current_action_mask}")
                self.invalidate_plan()
                self._log("--> Return: None (Action Invalid)")
                return None

            # --- Special check for planned challenge removed ---

            # If all checks pass:
            self._log(f"--> OK: Plan action {planned_action} is valid.")
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
        Searches for the best action using a specific simulation order [3, 5, 4, 0, 2, 1].
        Stops early if an action leading to an opponent penalty is found.
        Correctly returns the modified action if the heuristic was applied.
        Includes priority check for challenging apparent bluffs.
        """
        self.invalidate_plan()
        self.simulations_performed = 0

        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)

        try:
             sim_env.observe(self.training_agent, new=True)
             action_mask = sim_env.infos[self.training_agent].get('action_mask')
             if action_mask is None: raise ValueError("Action mask not found in sim_env info")
        except Exception as e:
             self._log(f"ERROR: Failed to get initial action mask in search setup: {e}")
             raise RuntimeError(f"Failed to get action mask for {self.training_agent} at search start") from e

        valid_actions = {i for i, mask_val in enumerate(action_mask) if mask_val == 1}
        if not valid_actions:
            self._log(f"ERROR: No valid actions available for {self.training_agent} at search start.")
            raise RuntimeError(f"No valid actions available for {self.training_agent}")

        hand = sim_env.players_hands.get(self.training_agent, [])
        table_card = sim_env.table_card

        best_action_found = -1
        best_value_found = float('-inf')
        best_sequence_found = []
        found_opponent_penalty_action = False

        # --- Optional: Prioritized Check - Challenge Bluff ---
        if 6 in valid_actions:
            # ... (Priority check logic remains the same) ...
            last_agent = sim_env.last_action_agent
            if last_agent and last_agent != self.training_agent:
                played_cards = sim_env.last_played_cards.get(last_agent, [])
                if played_cards:
                    is_bluff = any(card != table_card and card != "Joker" for card in played_cards)
                    if is_bluff:
                        self._log("Heuristic Check: Opponent's last play appears to be a bluff. Evaluating challenge first.")
                        challenge_sim_cache = {}
                        value, sequence, is_term, is_new_rnd = self.simulate_round(
                            env_state, 6, challenge_sim_cache, depth=0
                        )
                        self._log(f"Priority Challenge Result: V={value}, Term={is_term}, NewRnd={is_new_rnd}, Len={len(sequence)}")
                        if value >= self.OPPONENT_PENALTY_THRESHOLD:
                             best_action_found = 6
                             best_value_found = value
                             best_sequence_found = sequence
                             found_opponent_penalty_action = True
                             self._log(f"Prioritizing successful challenge action: {best_action_found} with value {value}. Stopping search.")
                        elif value > best_value_found:
                             best_action_found = 6
                             best_value_found = value
                             best_sequence_found = sequence

        # --- Define Simulation Order and Filter Valid Actions ---
        simulation_order = [3, 5, 4, 0, 2, 1]
        actions_to_simulate = [act for act in simulation_order if act in valid_actions]
        if 6 in valid_actions and best_action_found != 6:
            actions_to_simulate.append(6)

        self._log(f"Simulation Order (filtered): {actions_to_simulate}")

        # --- Main Loop: Simulate actions in specified order ---
        if not found_opponent_penalty_action:
            for action in actions_to_simulate: # 'action' is the initial action being simulated
                if action == 6 and best_action_found == 6:
                    self._log(f"Skipping re-simulation of challenge action {action} (already evaluated)")
                    continue

                action_sim_cache = {}
                self._log(f"--- Simulating Action (Order): {action} ---")
                value, sequence, is_terminal, is_new_round = self.simulate_round(
                    env_state, action, action_sim_cache, depth=0
                )
                self._log(f"--- Action {action} Result: V={value}, Term={is_terminal}, NewRnd={is_new_round}, Len={len(sequence)} ---")

                # --- Update Best Logic (with early exit AND heuristic check) ---
                if value >= self.OPPONENT_PENALTY_THRESHOLD:
                    self._log(f"Opponent Penalty Found! Action {action} simulation resulted in Value: {value}. Checking for heuristic swap...")

                    # --- FIX: Check if heuristic modified the action ---
                    returned_action = sequence[0][1] if sequence else action # Get the first action from the returned sequence
                    if returned_action != action:
                        self._log(f"HEURISTIC SWAP DETECTED: Initial action {action} was swapped to {returned_action} in simulation.")
                        # Use the swapped action as the best action
                        best_action_to_use = returned_action
                    else:
                        # No swap occurred, use the original action
                        best_action_to_use = action
                    # --------------------------------------------------

                    best_value_found = value
                    best_action_found = best_action_to_use # Store the potentially swapped action
                    best_sequence_found = sequence
                    found_opponent_penalty_action = True
                    self._log(f"Stopping search. Best action set to {best_action_found}.")
                    break # Exit loop

                elif value > best_value_found:
                    self._log(f"New best action found: {action} (Value: {value} > Current Best: {best_value_found})")
                    best_value_found = value
                    best_action_found = action # Store the original action if no early exit yet
                    best_sequence_found = sequence

                if is_terminal and value >= 5000:
                     self._log(f"Action {action} leads to immediate win (Value: {value}). Stopping search.")
                     if value > best_value_found:
                          best_value_found = value
                          best_action_found = action
                          best_sequence_found = sequence
                     found_opponent_penalty_action = True
                     break

        # --- Handle Fallbacks ---
        if best_action_found == -1 or (not found_opponent_penalty_action and best_value_found <= -5000):
            # ... (Fallback logic remains the same) ...
            self._log(f"No opponent penalty action found and best value ({best_value_found}) is poor. Trying fallbacks.")
            fallback_action_chosen = -1
            ordered_fallback_options = actions_to_simulate
            if not ordered_fallback_options:
                 ordered_fallback_options = sorted(list(valid_actions))
            if ordered_fallback_options:
                 fallback_action_chosen = ordered_fallback_options[0]
                 self._log(f"Selected fallback action based on order: {fallback_action_chosen}")
            else:
                 self._log("CRITICAL ERROR: No valid actions and no fallback options!")
                 fallback_action_chosen = 0

            if fallback_action_chosen != best_action_found or best_action_found == -1:
                 self._log(f"Re-simulating fallback action {fallback_action_chosen} to get sequence/value.")
                 fallback_sim_cache = {}
                 value, sequence, _, _ = self.simulate_round(env_state, fallback_action_chosen, fallback_sim_cache, depth=0)
                 self._log(f"Fallback Action {fallback_action_chosen} Result: V={value}, Len={len(sequence)}")

                 # --- FIX: Check heuristic swap on fallback too ---
                 returned_action_fallback = sequence[0][1] if sequence else fallback_action_chosen
                 action_to_use_fallback = returned_action_fallback if returned_action_fallback != fallback_action_chosen else fallback_action_chosen
                 if returned_action_fallback != fallback_action_chosen:
                     self._log(f"HEURISTIC SWAP ON FALLBACK: Initial fallback {fallback_action_chosen} swapped to {returned_action_fallback}.")
                 # -------------------------------------------------

                 if best_action_found == -1 or value > best_value_found:
                      best_action_found = action_to_use_fallback # Use potentially swapped action
                      best_value_found = value
                      best_sequence_found = sequence


        # --- Finalize Plan and Return ---
        action_to_return = best_action_found

        # Store the rest of the sequence (from the next step onwards)
        if best_sequence_found and len(best_sequence_found) > 1:
            # Check if the first step action matches action_to_return, log if not (debug)
            if best_sequence_found[0][1] != action_to_return:
                 self._log(f"WARNING: Mismatch between action_to_return ({action_to_return}) and first step of best_sequence ({best_sequence_found[0]}). Using action_to_return.")
            self.action_sequence = best_sequence_found[1:]
            self._log(f"Storing plan sequence starting from opponent (Len: {len(self.action_sequence)}): {self.action_sequence[:5]}...")
        else:
            self.action_sequence = []
            self._log("Storing empty plan sequence.")
        self.sequence_position = 0

        # Prepare return values
        action_dim = 7
        action_probs = np.zeros(action_dim)
        if 0 <= action_to_return < action_dim:
            action_probs[action_to_return] = 1.0
        else:
             self._log(f"ERROR: Final action_to_return '{action_to_return}' is invalid. Defaulting.")
             final_valid_actions_list = sorted(list(valid_actions))
             action_to_return = final_valid_actions_list[0] if final_valid_actions_list else 0
             action_probs[action_to_return] = 1.0

        self._log(f"--- Search Complete ---")
        self._log(f"FINAL RETURN: Action={action_to_return}, Value={best_value_found}, Stored Plan Len={len(self.action_sequence)}")

        return action_probs, action_to_return, best_value_found