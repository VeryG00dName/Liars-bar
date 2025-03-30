#!/usr/bin/env python3
# test_ps_simulation_accuracy.py - Tests PerfectSearch simulation accuracy

import unittest
import copy
import numpy as np
from unittest import mock
import logging

# Environment and Utilities
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.env.liars_deck_env_utils_2 import decode_action, TABLE_CARD_MAP
from src import config

# Agent Models
from src.model.ps import PerfectSearch
from src.model.hard_coded_agents import (
    RandomAgent,
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    # Add others if needed for specific scenarios
)

# Helper function to create a detailed environment state dictionary
def create_initial_state(num_players=3,
                         round_num=1,
                         current_agent='player_0',
                         hands=None,
                         penalties=None,
                         table_card='King',
                         last_action=None,
                         last_action_agent=None,
                         last_played_cards=None,
                         last_action_bluff=None,
                         terminations=None,
                         round_eliminated=None,
                         seed=42):
    """Creates a state dictionary for env.set_state(), filling defaults."""

    base_agents = [f'player_{i}' for i in range(num_players)]

    state = {
        'possible_agents': base_agents,
        'agents': base_agents[:], # Current active agents list
        'agent_selection': current_agent,
        'round': round_num,
        'table_card': table_card,
        'table_card_idx': TABLE_CARD_MAP.get(table_card, 0),
        'last_action': last_action,
        'last_action_agent': last_action_agent,
        'last_action_bluff': last_action_bluff,
        'winner': None,
        'deck': [], # Assume deck is dealt out or irrelevant for state setting
        'random_seed': seed,

        # Per-agent state (provide defaults if None)
        'players_hands': hands if hands is not None else {ag: [] for ag in base_agents},
        'penalties': penalties if penalties is not None else {ag: 0 for ag in base_agents},
        'penalty_thresholds': {ag: 3 for ag in base_agents},
        'last_played_cards': last_played_cards if last_played_cards is not None else {ag: [] for ag in base_agents},
        'terminations': terminations if terminations is not None else {ag: False for ag in base_agents},
        'round_eliminated': round_eliminated if round_eliminated is not None else {ag: False for ag in base_agents},

        # Default other fields often included in get_state
        'truncations': {ag: False for ag in base_agents},
        'infos': {ag: {'action_mask': [1]*7} for ag in base_agents}, # Will be updated by env logic
        '_cumulative_rewards': {ag: 0.0 for ag in base_agents},
        'rewards': {ag: 0 for ag in base_agents},
        'pending_bluff': None,
        'last_agent_action': {ag: None for ag in base_agents},
        'consecutive_action_count': {ag: 0 for ag in base_agents},
        'successful_bluffs': {ag: 0 for ag in base_agents},
        'failed_bluffs': {ag: 0 for ag in base_agents},
        'successful_challenges': {ag: 0 for ag in base_agents},
        'failed_challenges': {ag: 0 for ag in base_agents},
        'bluff_counts': {ag: 0 for ag in base_agents},
        'total_plays': {ag: 0 for ag in base_agents},
        'public_opponent_histories': {ag: [] for ag in base_agents},
        'private_opponent_histories': {ag: [] for ag in base_agents},
        'num_players': num_players,
    }
     # Ensure agents list only contains non-terminated agents for agent selector setup
    state['agents'] = [ag for ag in base_agents if not state['terminations'].get(ag, False)]

    return state


# Helper function to compare two environment state dictionaries
def compare_env_states(state_dict1, state_dict2, check_rewards=False):
    """
    Compares two state dictionaries from env.get_state().
    Returns True if relevant fields match, False otherwise, prints differences.
    """
    diffs = []
    fields_to_check = [
        'agent_selection', 'round', 'table_card', 'last_action',
        'last_action_agent', 'last_action_bluff', 'winner',
        'penalties', 'terminations', 'round_eliminated'
    ]
    if check_rewards:
        fields_to_check.extend(['rewards', '_cumulative_rewards'])

    # Simple field comparison
    for field in fields_to_check:
        val1 = state_dict1.get(field)
        val2 = state_dict2.get(field)
        # Handle dict comparison specifically for penalties/terminations etc.
        if isinstance(val1, dict) and isinstance(val2, dict):
            if val1 != val2:
                 diffs.append(f"Mismatch '{field}': Sim={val1}, Ref={val2}")
        elif val1 != val2:
            diffs.append(f"Mismatch '{field}': Sim='{val1}', Ref='{val2}'")


    # Compare player hands (order doesn't matter)
    hands1 = state_dict1.get('players_hands', {})
    hands2 = state_dict2.get('players_hands', {})
    if sorted(hands1.keys()) != sorted(hands2.keys()):
         diffs.append(f"Mismatch 'players_hands' keys: Sim={sorted(hands1.keys())}, Ref={sorted(hands2.keys())}")
    else:
        for agent in hands1:
            h1 = sorted(hands1.get(agent, []))
            h2 = sorted(hands2.get(agent, []))
            if h1 != h2:
                 diffs.append(f"Mismatch '{agent}' hand: Sim={h1}, Ref={h2}")

    # Compare last played cards (order doesn't matter within list)
    played1 = state_dict1.get('last_played_cards', {})
    played2 = state_dict2.get('last_played_cards', {})
    if sorted(played1.keys()) != sorted(played2.keys()):
         diffs.append(f"Mismatch 'last_played_cards' keys: Sim={sorted(played1.keys())}, Ref={sorted(played2.keys())}")
    else:
        for agent in played1:
            # Ensure value is list before sorting
            p1_val = played1.get(agent, [])
            p2_val = played2.get(agent, [])
            p1 = sorted(p1_val) if isinstance(p1_val, list) else p1_val
            p2 = sorted(p2_val) if isinstance(p2_val, list) else p2_val
            if p1 != p2:
                 diffs.append(f"Mismatch '{agent}' last_played: Sim={p1}, Ref={p2}")

    if diffs:
        print("\nState Comparison Failed:")
        for d in diffs:
            print(f"  - {d}")
        return False
    return True


class TestPerfectSearchSimulationAccuracy(unittest.TestCase):

    def setUp(self):
        """Set up a base environment and opponent models."""
        self.num_players = 3
        self.training_agent = 'player_0'
        self.opponent_agents = ['player_1', 'player_2']

        # Use real, simple, deterministic opponents
        self.opponent_model_instances = {
            'player_1': GreedyCardSpammer(agent_name='player_1'),
            'player_2': TableFirstConservativeChallenger(agent_name='player_2')
            # Add more complex ones if needed, ensure they can be seeded
        }

        # Use a high logging level to suppress output during tests
        self.log_level = logging.CRITICAL # Use CRITICAL to silence logs

        # We need a base_env primarily for cloning in PS, but tests will create fresh envs
        self.base_env = LiarsDeckEnv(num_players=self.num_players, log_level=self.log_level)

        self.ps = PerfectSearch(
            env=self.base_env,
            training_agent=self.training_agent,
            opponent_models=self.opponent_model_instances
        )
        # Disable PS debug logging during tests unless needed
        self.ps.debug = False # Set to True to see PS internal logs for debugging a specific test

    # --- Test _select_opponent_action Accuracy ---

    def test_select_opponent_action_matches_real_hardcoded(self):
        """Verify _select_opponent_action gets same action as real hardcoded model."""
        opponent_id = 'player_1' # GreedyCardSpammer
        state_dict = create_initial_state(
            num_players=self.num_players,
            current_agent=opponent_id,
            table_card='King',
            hands={'player_0': ['Ace', 'Ace'],
                   'player_1': ['King', 'King', 'Queen'], # Has 2 Kings (Table), 1 Queen (Non)
                   'player_2': ['Joker']},
            last_action_agent='player_0',
            last_action=1,
            last_played_cards={'player_0': ['Ace']}
        )

        # Setup Sim Env
        sim_env = LiarsDeckEnv(num_players=self.num_players, log_level=self.log_level)
        sim_env.reset(seed=state_dict['random_seed'])
        sim_env.set_state(copy.deepcopy(state_dict))
        sim_env.observe(opponent_id, new=True) # Ensure infos are generated

        # Get action from Real Model directly
        real_model = self.opponent_model_instances[opponent_id]
        obs_dict = sim_env.observe(opponent_id, new=True)
        observation = obs_dict[opponent_id]
        action_mask = sim_env.infos[opponent_id]['action_mask']

        # --- Debug Prints --- (Optional: uncomment to see details)
        # print(f"\nDEBUG (test_select_opponent_action_matches_real_hardcoded):")
        # print(f"  Agent: {opponent_id}")
        # print(f"  Hand: {sim_env.players_hands.get(opponent_id)}")
        # print(f"  Table Card: {sim_env.table_card}")
        # print(f"  Action Mask Passed to Real Model: {action_mask}")
        # --- End Debug Prints ---

        real_action = real_model.play_turn(observation, action_mask, sim_env.table_card)
        # print(f"  Real Model Returned: {real_action}")

        # Get action from PS simulation helper
        opponent_action_cache = {}
        # Create a fresh sim_env state for the PS call to ensure no side effects
        sim_env_for_ps = LiarsDeckEnv(num_players=self.num_players, log_level=self.log_level)
        sim_env_for_ps.reset(seed=state_dict['random_seed'])
        sim_env_for_ps.set_state(copy.deepcopy(state_dict))

        sim_action = self.ps._select_opponent_action(sim_env_for_ps, opponent_id, opponent_action_cache)
        # print(f"  Sim Helper Returned: {sim_action}")

        # Assertions
        self.assertEqual(action_mask[0], 1, "Mask should allow playing 1 King")
        self.assertEqual(action_mask[1], 1, "Mask should allow playing 2 Kings")
        self.assertEqual(action_mask[3], 1, "Mask should allow playing 1 Queen")

        # ****** CORRECTED ASSERTION BASED ON AGENT LOGIC ******
        self.assertEqual(real_action, 3, f"Real model expected action 3 (Play 1 Queen - non-table first), got {real_action}")
        # ******************************************************

        self.assertEqual(sim_action, real_action, "Simulated action must match real model action")
        self.assertTrue(len(opponent_action_cache) > 0, "Cache should have been populated")
        self.assertTrue(any(key[0] == opponent_id for key in opponent_action_cache.keys()), f"Cache key should include agent {opponent_id}")


    def test_select_opponent_action_caching_works_correctly(self):
        """Verify caching returns the same action and avoids re-computation."""
        opponent_id = 'player_2' # TableFirstConservativeChallenger
        state_dict = create_initial_state(
            num_players=self.num_players,
            current_agent=opponent_id,
            table_card='Queen',
            hands={'player_0': ['Ace'],
                   'player_1': ['King'],
                   'player_2': ['Queen', 'Queen', 'King']}, # Has 2 Queens (Table)
            last_action_agent='player_1',
            last_action=1,
            last_played_cards={'player_1': ['King']}
        )
        sim_env = LiarsDeckEnv(num_players=self.num_players, log_level=self.log_level)
        sim_env.reset(seed=state_dict['random_seed'])
        sim_env.set_state(copy.deepcopy(state_dict))
        sim_env.observe(opponent_id, new=True)

        opponent_action_cache = {}

        # Mock the *real* model's play_turn to track calls
        real_model = self.opponent_model_instances[opponent_id]
        with mock.patch.object(real_model, 'play_turn', wraps=real_model.play_turn) as mocked_play_turn:
            # First call (cache miss) - TableFirst should play 2 Queens (Action 1)
            action1 = self.ps._select_opponent_action(sim_env, opponent_id, opponent_action_cache)
            self.assertEqual(action1, 1, "TableFirst should play 2 Queens (Action 1)")
            mocked_play_turn.assert_called_once()
            self.assertTrue(len(opponent_action_cache) > 0)
            # Get the key used (best effort)
            try:
                cache_key = list(opponent_action_cache.keys())[0]
                key_found = True
            except Exception:
                key_found = False

            # Reset mock for second call
            mocked_play_turn.reset_mock()

            # Second call (cache hit)
            action2 = self.ps._select_opponent_action(sim_env, opponent_id, opponent_action_cache)
            mocked_play_turn.assert_not_called()
            self.assertEqual(action1, action2, "Cached action should match original action")

            # Third call with slightly modified state (cache miss)
            sim_env.players_hands[opponent_id].append('Joker') # Change hand state
            # Re-observe to update internal state representations if necessary
            sim_env.observe(opponent_id, new=True)
            action3 = self.ps._select_opponent_action(sim_env, opponent_id, opponent_action_cache)
            mocked_play_turn.assert_called_once() # Should be called again
            # TableFirst with 2 Queens + Joker should play 3 Table cards (Action 2)
            self.assertEqual(action3, 2, "With added Joker, should now play 3 table cards (Action 2)")


    def test_select_opponent_action_cache_validation(self):
        """Verify cached action is recalculated if it becomes invalid."""
        opponent_id = 'player_1' # GreedyCardSpammer
        state_dict = create_initial_state(
            num_players=self.num_players,
            current_agent=opponent_id,
            table_card='King',
             hands={'player_0': ['Ace'],
                   'player_1': ['King', 'King', 'Queen'], # Initially plays 1 Queen (Action 3)
                   'player_2': ['Joker']},
             last_action=1, last_action_agent='player_0'
        )
        sim_env = LiarsDeckEnv(num_players=self.num_players, log_level=self.log_level)
        sim_env.reset(seed=state_dict['random_seed'])
        sim_env.set_state(copy.deepcopy(state_dict))
        sim_env.observe(opponent_id, new=True)

        opponent_action_cache = {}
        real_model = self.opponent_model_instances[opponent_id]
        key_found = False # Flag to track if we could get the cache key

        with mock.patch.object(real_model, 'play_turn', wraps=real_model.play_turn) as mocked_play_turn:
            # First call - caches action 3 (Play 1 Queen) based on agent logic
            action1 = self.ps._select_opponent_action(sim_env, opponent_id, opponent_action_cache)

            # ****** CORRECTED ASSERTION BASED ON AGENT LOGIC ******
            self.assertEqual(action1, 3, "Initial action should be 3 (Play 1 Queen)")
            # ******************************************************

            mocked_play_turn.assert_called_once()
            self.assertTrue(len(opponent_action_cache) > 0)
            try:
                 cached_key = list(opponent_action_cache.keys())[0]
                 key_found = True
            except Exception:
                 key_found = False
                 print("Warning: Could not retrieve cache key for deeper validation.")


            # Modify state: remove the Queen, making action 3 invalid
            sim_env.players_hands[opponent_id].remove('Queen') # Remove the non-table card
            sim_env.observe(opponent_id, new=True) # Regenerate info/mask
            current_mask = sim_env.infos[opponent_id]['action_mask']
            self.assertEqual(current_mask[3], 0, "Action 3 should now be invalid")
            self.assertEqual(current_mask[1], 1, "Action 1 (Play 2 Kings) should still be valid")

            mocked_play_turn.reset_mock()

            # Second call - cached action 3 is invalid. _select_opponent_action should detect this.
            # Because the cached action (3) is invalid according to the *current* mask,
            # it must recalculate. With no non-table cards, GreedyCardSpammer logic falls
            # back to playing table cards. It checks action 2 (invalid), then action 1 (valid).
            action2 = self.ps._select_opponent_action(sim_env, opponent_id, opponent_action_cache)

            # Should have called the model again because cached action was invalid
            mocked_play_turn.assert_called_once()
            # Should now choose action 1 (Play 2 Kings)
            self.assertEqual(action2, 1, "After removing Queen, should play 2 Kings (action 1)")

            # Verify the cache was updated for the *new* state, and the old state's value wasn't overwritten inappropriately
            if key_found:
                 # The value associated with the original state's key should still be 3 (if key is stable)
                 # We mostly care that the *new* state led to the *new* action being cached.
                 self.assertNotEqual(opponent_action_cache.get(cached_key), 1, "Original cache entry value should remain 3, not updated to 1")

            # Check that the *new* action (1) is now cached under *some* key
            self.assertIn(1, opponent_action_cache.values())


    # --- Test simulate_round Accuracy ---

    def run_comparison(self, initial_state_dict, action_sequence):
        """
        Helper to run simulation and reference steps and compare final states.

        Args:
            initial_state_dict (dict): Starting state for both environments.
            action_sequence (list): List of (agent_id, action_idx) tuples.
                                     Use None for opponent actions to use model.
        """
        seed = initial_state_dict['random_seed']
        print_output = False # Set to True to see step-by-step output for debugging

        # --- Reference Environment ---
        ref_env = LiarsDeckEnv(num_players=self.num_players, log_level=self.log_level)
        ref_env.reset(seed=seed)
        ref_env.set_state(copy.deepcopy(initial_state_dict))
        if print_output: print(f"\n--- REF ENV START (Seed: {seed}) ---")
        if print_output: ref_env.render()

        ref_steps = 0
        for agent_id, intended_action in action_sequence:
            current_agent_ref = ref_env.agent_selection
            if current_agent_ref is None:
                if print_output: print("REF ENV: Game ended early.")
                break
            # Ensure the sequence matches the environment's agent selection
            if agent_id != current_agent_ref:
                 if print_output: print(f"REF ENV: Agent mismatch detected. Sequence expected {agent_id}, Env has {current_agent_ref}. Stopping comparison.")
                 break # Stop comparison if sequence diverges from reality

            # Get action mask and observation for reference model if needed
            ref_env.observe(current_agent_ref, new=True)
            action_mask_ref = ref_env.infos[current_agent_ref]['action_mask']
            if sum(action_mask_ref) == 0:
                if print_output: print(f"REF ENV: Agent {current_agent_ref} has no valid actions. Skipping.")
                break

            if intended_action is None: # Use opponent model
                model = self.opponent_model_instances[current_agent_ref]
                obs_dict_ref = ref_env.observe(current_agent_ref, new=True)
                obs_ref = obs_dict_ref[current_agent_ref]
                if hasattr(model, 'play_turn'):
                    ref_action = model.play_turn(obs_ref, action_mask_ref, ref_env.table_card)
                else:
                    raise NotImplementedError("NN model logic not implemented in test runner")
            else: # Use predefined action (usually for PS agent)
                ref_action = intended_action

            if action_mask_ref[ref_action] == 0:
                 if print_output: print(f"WARNING: REF ENV action {ref_action} by {current_agent_ref} is INVALID according to mask {action_mask_ref}. This indicates a potential issue in test design or env logic.")
                 # Allow the step to proceed to check env handles invalid action penalty

            action_type, card_cat, count = decode_action(ref_action)
            if print_output: print(f"REF STEP {ref_steps}: {current_agent_ref} takes action {ref_action} ({action_type}, {card_cat}, {count})")
            ref_env.step(ref_action)
            ref_steps += 1
            if print_output: ref_env.render() # Show state after step
            if all(ref_env.terminations.values()) or ref_env.winner:
                 if print_output: print("REF ENV: Game ended.")
                 break

        ref_final_state = ref_env.get_state()
        if print_output: print("--- REF ENV FINAL ---")
        if print_output: ref_env.render()


        # --- Simulation Environment ---
        # Replicate the sequence step-by-step using PS's opponent selection logic
        sim_env = LiarsDeckEnv(num_players=self.num_players, log_level=self.log_level)
        sim_env.reset(seed=seed) # Use same seed
        sim_env.set_state(copy.deepcopy(initial_state_dict))
        # PS instance just for _select_opponent_action
        current_ps = PerfectSearch(
            env=sim_env,
            training_agent=self.training_agent,
            opponent_models=self.opponent_model_instances
        )
        current_ps.debug = False


        if print_output: print(f"\n--- SIM ENV START (Seed: {seed}) ---")
        if print_output: sim_env.render()

        opponent_action_cache_sim = {}
        sim_steps = 0
        for agent_id, intended_action in action_sequence:
            current_agent_sim = sim_env.agent_selection
            if current_agent_sim is None:
                 if print_output: print("SIM ENV: Game ended early.")
                 break
            # Ensure sequence matches simulation agent selection
            if agent_id != current_agent_sim:
                 if print_output: print(f"SIM ENV: Agent mismatch detected. Sequence expected {agent_id}, Env has {current_agent_sim}. Stopping comparison.")
                 break # Stop comparison if sequence diverges

            sim_env.observe(current_agent_sim, new=True) # Ensure infos/mask are ready
            action_mask_sim = sim_env.infos[current_agent_sim]['action_mask']
            if sum(action_mask_sim) == 0:
                 if print_output: print(f"SIM ENV: Agent {current_agent_sim} has no valid actions. Skipping.")
                 break

            if current_agent_sim == self.training_agent:
                sim_action = intended_action
                if sim_action is None:
                    raise ValueError("Intended action for training agent cannot be None in sequence")
            else: # Opponent's turn, use _select_opponent_action
                try:
                    sim_action = current_ps._select_opponent_action(
                        sim_env, current_agent_sim, opponent_action_cache_sim
                    )
                except Exception as e:
                    print(f"SIM ENV: Error selecting opponent action for {current_agent_sim}: {e}")
                    raise

            if action_mask_sim[sim_action] == 0:
                 if print_output: print(f"WARNING: SIM ENV action {sim_action} by {current_agent_sim} is INVALID according to mask {action_mask_sim}. Indicates issue in PS._select_opponent_action or env logic.")
                 # Allow step to proceed

            action_type, card_cat, count = decode_action(sim_action)
            if print_output: print(f"SIM STEP {sim_steps}: {current_agent_sim} takes action {sim_action} ({action_type}, {card_cat}, {count})")
            sim_env.step(sim_action)
            sim_steps += 1
            if print_output: sim_env.render() # Show state after step
            if all(sim_env.terminations.values()) or sim_env.winner:
                 if print_output: print("SIM ENV: Game ended.")
                 break

        sim_final_state = sim_env.get_state()
        if print_output: print("--- SIM ENV FINAL ---")
        if print_output: sim_env.render()

        # --- Comparison ---
        # Check if the number of steps taken matches (indicates no early termination mismatch)
        self.assertEqual(sim_steps, ref_steps, f"Simulation took {sim_steps} steps, Reference took {ref_steps} steps.")
        # Compare final states
        self.assertTrue(
            compare_env_states(sim_final_state, ref_final_state),
            "Final states of simulation and reference environments must match."
        )


    def test_simulation_simple_play_sequence(self):
        """Scenario: Simple sequence of 'Play' actions."""
        initial_state = create_initial_state(
            num_players=3,
            current_agent='player_0',
            table_card='King',
            hands={'player_0': ['King', 'Ace', 'Ace'], # Plays 1 King (Action 0)
                   'player_1': ['Queen', 'Queen', 'Joker'], # Greedy plays 1 Queen (Action 3)
                   'player_2': ['King', 'King', 'Ace']} # TableFirst plays 2 Kings (Action 1)
        )
        # Player 0 (PS) plays 1 King, Player 1 (Greedy) plays 1 Queen, Player 2 (TableFirst) plays 2 Kings
        action_seq = [
            ('player_0', 0),  # PS plays 1 King
            ('player_1', None), # Greedy plays 1 Queen (Action 3)
            ('player_2', None), # TableFirst plays 2 Kings (Action 1)
        ]
        self.run_comparison(initial_state, action_seq)

    def test_simulation_challenge_success(self):
        """Scenario: PS challenges a successful bluff."""
        initial_state = create_initial_state(
            num_players=3,
            current_agent='player_0',
            table_card='Ace',
            hands={'player_0': ['King', 'King', 'King'], # Holds no Aces, will challenge
                   'player_1': ['Queen', 'Queen', 'Joker'], # Bluffs 1 Ace (plays Queen - action 3)
                   'player_2': ['Ace', 'Ace', 'Joker']},
            last_action_agent='player_1',
            last_action=1, # Player 1 claims 1 Ace
            last_played_cards={'player_1': ['Queen']}, # Actual bluff
            last_action_bluff=True,
            penalties={'player_0': 1, 'player_1': 0, 'player_2': 0}
        )
        action_seq = [
            ('player_0', 6), # PS Challenges P1 -> P1 gets penalty, round ends
        ]
        self.run_comparison(initial_state, action_seq)

    def test_simulation_challenge_fail(self):
        """Scenario: PS challenges a valid play."""
        initial_state = create_initial_state(
            num_players=3,
            current_agent='player_0',
            table_card='King',
            hands={'player_0': ['Queen', 'Queen', 'Queen'], # Will challenge
                   'player_1': ['King', 'Ace', 'Joker'], # Plays valid 1 King (Action 0)
                   'player_2': ['King', 'King', 'Ace']},
            last_action_agent='player_1',
            last_action=1, # Player 1 claims 1 King
            last_played_cards={'player_1': ['King']}, # Valid play
            last_action_bluff=False
        )
        action_seq = [
            ('player_0', 6), # PS Challenges P1 -> P0 gets penalty, round ends
        ]
        self.run_comparison(initial_state, action_seq)

    def test_simulation_round_end_elimination(self):
        """Scenario: A player gets eliminated mid-round by challenge."""
        initial_state = create_initial_state(
            num_players=3,
            current_agent='player_0',
            table_card='King',
            hands={'player_0': ['Ace', 'Ace', 'Ace'], # Challenges P2
                   'player_1': ['Queen', 'Joker'],
                   'player_2': ['Queen', 'Queen']},   # Bluffs 1 King (Action 3), gets eliminated
            penalties={'player_0': 0, 'player_1': 0, 'player_2': 2}, # P2 close to termination
            last_action_agent='player_2',
            last_action=1, # P2 claims 1 King
            last_played_cards={'player_2': ['Queen']}, # Bluff
            last_action_bluff=True
        )
        action_seq = [
            ('player_0', 6), # P0 challenges P2 -> P2 gets penalty -> P2 terminated -> Round ends
        ]
        self.run_comparison(initial_state, action_seq)


if __name__ == '__main__':
    unittest.main()