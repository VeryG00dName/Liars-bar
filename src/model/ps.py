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

    def simulate_round(self, env_state, action, opponent_action_cache=None, depth=0, max_depth=40, p=0.2):
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
            p: Probability (default 20%) to immediately accept a negative outcome if our penalty count is <2.

        Returns:
            tuple: (outcome_value, action_sequence, is_terminal, is_new_round, p_triggered)
                   p_triggered is True if the branch was accepted via the p probability.
        """
        import random
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

        if sim_env.agent_selection is None:
            winner = sim_env.winner
            value = 5000.0 if winner == self.training_agent else -5000.0
            self._log(f"[Depth {depth}] Game ended immediately after our action - Winner: {winner}")
            return value, action_sequence, True, False, False

        if not pre_step_termination_us and post_step_termination_us:
            self._log(f"[Depth {depth}] Got eliminated by our own action! Very bad.")
            return -10000.0, action_sequence, True, False, False

        if post_step_penalty_us > starting_penalty_us:
            self._log(f"[Depth {depth}] Got penalty immediately after our action.")
            penalty_value = -5000.0 if starting_penalty_us >= 2 else -1000.0
            return penalty_value, action_sequence, False, False, False

        if sim_env.round > starting_round:
            self._log(f"[Depth {depth}] Round changed immediately after our action (now Round {sim_env.round}).")
            final_penalty_us_after = sim_env.penalties.get(self.training_agent, 0)
            is_terminated_after = sim_env.terminations.get(self.training_agent, False)

            if is_terminated_after:
                self._log(f"[Depth {depth}] Eliminated during immediate round change.")
                return -10000.0, action_sequence, True, True, False

            if final_penalty_us_after > starting_penalty_us:
                self._log(f"[Depth {depth}] Got penalty during immediate round change.")
                penalty_value = -5000.0 if starting_penalty_us >= 2 else -1000.0
                return penalty_value, action_sequence, False, True, False

            opponent_penalized = any(
                sim_env.penalties.get(opp, 0) > initial_opponent_penalties[opp] for opp in self.opponent_agents
            )
            if opponent_penalized:
                self._log(f"[Depth {depth}] Opponent penalized during immediate round change.")
                return 2000.0, action_sequence, False, True, False

        # --- Simulation Loop ---
        max_steps = 50
        step_count = 0
        current_sim_round = sim_env.round

        while step_count < max_steps:
            step_count += 1

            if sim_env.agent_selection is None:
                winner = sim_env.winner
                value = 5000.0 if winner == self.training_agent else -5000.0
                self._log(f"[Depth {depth}, SimStep {step_count}] Game ended. Winner: {winner}")
                return value, action_sequence, True, False, False

            if sim_env.round > current_sim_round:
                self._log(f"[Depth {depth}, SimStep {step_count}] Detected round change (now Round {sim_env.round} from {current_sim_round}).")
                current_sim_round = sim_env.round
                final_penalty_us = sim_env.penalties.get(self.training_agent, 0)
                is_terminated_us = sim_env.terminations.get(self.training_agent, False)

                if is_terminated_us:
                    self._log(f"[Depth {depth}] Eliminated during round change!")
                    return -10000.0, action_sequence, True, True, False

                if final_penalty_us > starting_penalty_us:
                    self._log(f"[Depth {depth}] Got penalty during round change (current={final_penalty_us}, start={starting_penalty_us}).")
                    penalty_value = -5000.0 if starting_penalty_us >= 2 else -1000.0
                    return penalty_value, action_sequence, False, True, False

                opponent_penalized = False
                for opp in self.opponent_agents:
                    if sim_env.penalties.get(opp, 0) > initial_opponent_penalties[opp]:
                        self._log(f"[Depth {depth}] Opponent {opp} got penalty during round change.")
                        opponent_penalized = True
                        break

                if opponent_penalized:
                    return 2000.0, action_sequence, False, True, False

                self._log(f"[Depth {depth}] Round changed but no penalty detected. Returning neutral.")
                return 50.0, action_sequence, False, True, False

            current_agent = sim_env.agent_selection

            if current_agent == self.training_agent:
                sim_env.observe(self.training_agent, new=True)
                action_mask = sim_env.infos[self.training_agent].get('action_mask', [0] * 7)
                valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
                if not valid_actions:
                    self._log(f"[Depth {depth}] No valid actions available for {self.training_agent}. Stuck.")
                    return -50.0, action_sequence, False, False, False

                # Reorder valid actions as: 5,4,3,2,1,0,6
                custom_order = [5, 4, 3, 2, 1, 0, 6]
                valid_actions = [act for act in custom_order if act in valid_actions]

                best_value = float('-inf')
                best_sequence_continuation = None
                best_is_terminal = False
                best_is_new_round = False

                for next_action in valid_actions:
                    self._log(f"[Depth {depth}] Exploring recursive action {next_action} (depth {depth+1})")
                    next_state = sim_env.get_state()
                    value, next_seq, is_terminal, is_new_round, p_triggered = self.simulate_round(
                        next_state, next_action, opponent_action_cache.copy(), depth + 1, max_depth, p
                    )
                    self._log(f"[Depth {depth}] Action {next_action} returned: value={value}, term={is_terminal}, new_round={is_new_round}, p_triggered={p_triggered}, seq_len={len(next_seq)}")

                    # If a branch is accepted via p, immediately use it.
                    if p_triggered:
                        self._log(f"[Depth {depth}] Accepting branch from action {next_action} via p trigger (p_triggered=True).")
                        return value, action_sequence + next_seq, is_terminal, is_new_round, True

                    if value >= 1500:
                        self._log(f"[Depth {depth}] Prioritizing good outcome from action {next_action} with value {value}.")
                        return value, action_sequence + next_seq, is_terminal, is_new_round, False

                    if value > best_value:
                        best_value = value
                        best_sequence_continuation = next_seq
                        best_is_terminal = is_terminal
                        best_is_new_round = is_new_round
                        self._log(f"[Depth {depth}] New best path via action {next_action} with value {value}")

                if best_sequence_continuation:
                    return best_value, action_sequence + best_sequence_continuation, best_is_terminal, best_is_new_round, False
                else:
                    self._log(f"[Depth {depth}] No suitable recursive paths found. Returning very negative.")
                    return best_value if best_value > float('-inf') else -1000.0, action_sequence, False, False, False

            else:
                self._log(f"[Depth {depth}, SimStep {step_count}] Opponent {current_agent}'s turn (Round {current_sim_round})")
                try:
                    opp_penalty_before = sim_env.penalties.get(current_agent, 0)
                    opponent_action = self._select_opponent_action(sim_env, current_agent, opponent_action_cache)
                    opp_action_type, _, _ = decode_action(opponent_action)
                    self._log(f"[Depth {depth}] Opponent {current_agent} selected action {opponent_action} ({opp_action_type})")

                    is_challenging_us = (opponent_action == 6 and sim_env.last_action_agent == self.training_agent)
                    if is_challenging_us:
                        our_last_played = sim_env.last_played_cards.get(self.training_agent, [])
                        table_card = sim_env.table_card
                        is_our_bluff = any(card != table_card and card != "Joker" for card in our_last_played)
                        if is_our_bluff:
                            self._log(f"[Depth {depth}] Opponent {current_agent} challenges our bluff! Very bad.")
                            sim_env.step(opponent_action)
                            action_sequence.append((current_agent, opponent_action))
                            penalty_value = -10000.0 if starting_penalty_us >= 2 else -5000.0
                            return penalty_value, action_sequence, False, True, False

                    sim_env.step(opponent_action)
                    action_sequence.append((current_agent, opponent_action))
                    opp_penalty_after = sim_env.penalties.get(current_agent, 0)
                    if opp_penalty_after > opp_penalty_before:
                        self._log(f"[Depth {depth}] Opponent {current_agent} got a penalty from action {opponent_action}.")
                        return 2000.0, action_sequence, False, False, False
                except Exception as e:
                    self._log(f"[Depth {depth}, SimStep {step_count}] Error during opponent {current_agent}'s turn: {e}")
                    import traceback
                    self._log(traceback.format_exc())
                    return -50.0, action_sequence, False, False, False

        self._log(f"[Depth {depth}] Hit step limit ({max_steps}) before penalty/game end.")
        final_penalty = sim_env.penalties.get(self.training_agent, 0)
        opponent_penalized = any(
            sim_env.penalties.get(opp, 0) > initial_opponent_penalties[opp] for opp in self.opponent_agents
        )

        if final_penalty > starting_penalty_us:
            return -500.0, action_sequence, False, False, False
        if opponent_penalized:
            return 1000.0, action_sequence, False, False, False
        else:
            self._log(f"[Depth {depth}] Max steps reached without significant events. Neutral outcome.")
            return -10.0, action_sequence, False, False, False

    def search(self, env_state):
        """
        Searches for the best action by simulating each valid action.
        Stores the best linear sequence found.

        Args:
            env_state: The environment state to start search from.

        Returns:
            tuple: (action_probs, best_action, best_value)
        """
        self.invalidate_plan()
        self.simulations_performed = 0

        opponent_action_cache = {}
        sim_env = self.base_env.clone()
        sim_env.set_state(env_state)

        sim_env.observe(self.training_agent, new=True)
        action_mask = sim_env.infos[self.training_agent].get('action_mask', [0] * 7)
        valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
        if not valid_actions:
            self._log(f"ERROR: No valid actions available for {self.training_agent} at search start.")
            raise RuntimeError(f"No valid actions available for {self.training_agent}")

        current_penalty = sim_env.penalties.get(self.training_agent, 0)
        hand = sim_env.players_hands.get(self.training_agent, [])
        table_card = sim_env.table_card
        table_cards = [c for c in hand if c == table_card or c == "Joker"]

        best_action = -1
        best_value = float('-inf')
        best_sequence = None

        # Prioritized check for a challenge option.
        if 6 in valid_actions:
            last_agent = sim_env.last_action_agent
            if last_agent:
                played_cards = sim_env.last_played_cards.get(last_agent, [])
                is_bluff = any(card != table_card and card != "Joker" for card in played_cards)
                if is_bluff:
                    self._log("Found opponent bluff - trying challenge as priority")
                    challenge_cache = {}
                    value, sequence, _, _, p_triggered = self.simulate_round(env_state, 6, challenge_cache)
                    self._log(f"Priority Challenge: value={value}, seq_len={len(sequence)}, p_triggered={p_triggered}")
                    if value > 1000:
                        best_action = 6
                        best_value = value
                        best_sequence = sequence
                        self._log(f"Prioritizing challenge action {best_action} with value {value}")
                        # If this branch was p triggered, we immediately use it.
                        if p_triggered:
                            self.action_sequence = best_sequence
                            self.sequence_position = 0
                            action_dim = 7
                            action_probs = [0.0] * action_dim
                            action_probs[6] = 1.0
                            return action_probs, 6, best_value

        for action in valid_actions:
            if action == 6 and best_action == 6:
                self._log(f"Skipping re-simulation of prioritized challenge action {action}")
                continue

            branch_cache = {}
            value, sequence, is_terminal, is_new_round, p_triggered = self.simulate_round(
                env_state, action, branch_cache, depth=0, p=0.2
            )
            self._log(f"Root Action {action}: value={value}, seq_len={len(sequence)}, terminal={is_terminal}, new_round={is_new_round}, p_triggered={p_triggered}")

            # If a branch was accepted via p, immediately use it.
            if p_triggered:
                self._log(f"Action {action} was accepted via p trigger. Using this branch immediately.")
                best_action = action
                best_value = value
                best_sequence = sequence
                break

            if is_terminal and value > 0:
                temp_env = self.base_env.clone()
                temp_env.set_state(env_state)
                temp_env.step(action)
                if temp_env.winner == self.training_agent:
                    if value > best_value:
                        best_action = action
                        best_value = value
                        best_sequence = sequence
                        self._log(f"Found winning action {action} with value {value}.")
                    continue

            if value > 1000 and value > best_value:
                best_action = action
                best_value = value
                best_sequence = sequence
                self._log(f"Action {action} gives opponent penalty with value {value}.")
            elif value > best_value:
                best_action = action
                best_value = value
                best_sequence = sequence
                self._log(f"New best action: {action} with value {value}.")

        if best_action == -1 or best_value <= -5000:
            self._log(f"No suitable action found (best_value={best_value}). Trying fallbacks.")
            fallback_action = -1
            if 0 in valid_actions and len(table_cards) > 0:
                fallback_action = 0
                self._log("Fallback: playing one table card.")
            elif 3 in valid_actions and len(hand) > len(table_cards):
                fallback_action = 3
                self._log("Fallback: playing one non-table card.")
            else:
                fallback_action = valid_actions[0]
                self._log(f"Fallback: using first valid action ({fallback_action}).")

            self._log(f"Re-simulating fallback action {fallback_action}.")
            fallback_cache = {}
            value, sequence, _, _, _ = self.simulate_round(env_state, fallback_action, fallback_cache, depth=0)
            self._log(f"Fallback Action {fallback_action} returned value={value}, seq_len={len(sequence)}")
            best_action = fallback_action
            best_value = value
            best_sequence = sequence

        self.action_sequence = best_sequence
        self.sequence_position = 0

        action_dim = 7
        action_probs = [0.0] * action_dim
        final_best_action = best_action if 0 <= best_action < action_dim else valid_actions[0]
        action_probs[final_best_action] = 1.0

        self._log(f"FINAL RETURN: Action={final_best_action}, Value={best_value}, SeqLen={len(self.action_sequence)}")
        return action_probs, final_best_action, best_value