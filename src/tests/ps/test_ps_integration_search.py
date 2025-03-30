#!/usr/bin/env python3
# test_ps_integration_search.py - Integration Tests for PerfectSearch.search

import unittest
import copy
import numpy as np
import logging

# Environment and Utilities
# Assuming these files are in the structure src/env/ and src/
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.env.liars_deck_env_utils_2 import decode_action, TABLE_CARD_MAP
from src import config # Assuming config might be needed indirectly

# Agent Models
from src.model.ps import PerfectSearch
from src.model.hard_coded_agents import (
    RandomAgent, # Good for unpredictable but simple behavior
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
)

# Helper function from the previous test file
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
    """Creates a state dictionary for env.set_state(), filling defaults
       and ensuring all agents are keys in per-agent dicts."""

    base_agents = [f'player_{i}' for i in range(num_players)]

    # --- Create Default Dictionaries ---
    default_hands = {ag: [] for ag in base_agents}
    default_penalties = {ag: 0 for ag in base_agents}
    default_last_played = {ag: [] for ag in base_agents}
    default_terminations = {ag: False for ag in base_agents}
    default_round_eliminated = {ag: False for ag in base_agents}
    default_infos = {ag: {'action_mask': [1]*7} for ag in base_agents} # Default mask

    # --- Merge provided args with defaults ---
    final_hands = default_hands.copy()
    if hands: final_hands.update(hands)

    final_penalties = default_penalties.copy()
    if penalties: final_penalties.update(penalties)

    final_last_played = default_last_played.copy()
    if last_played_cards: final_last_played.update(last_played_cards)

    final_terminations = default_terminations.copy()
    if terminations: final_terminations.update(terminations)

    final_round_eliminated = default_round_eliminated.copy()
    if round_eliminated: final_round_eliminated.update(round_eliminated)

    # Base state structure
    state = {
        'possible_agents': base_agents,
        'agents': [ag for ag in base_agents if not final_terminations.get(ag, False)], # Active agents list based on final terminations
        'agent_selection': current_agent,
        'round': round_num,
        'table_card': table_card,
        'table_card_idx': TABLE_CARD_MAP.get(table_card, 0),
        'last_action': last_action,
        'last_action_agent': last_action_agent,
        'last_action_bluff': last_action_bluff,
        'winner': None,
        'deck': [],
        'random_seed': seed,
        'num_players': num_players,

        # Use the merged dictionaries
        'players_hands': final_hands,
        'penalties': final_penalties,
        'penalty_thresholds': {ag: 3 for ag in base_agents}, # Assuming fixed threshold
        'last_played_cards': final_last_played,
        'terminations': final_terminations,
        'round_eliminated': final_round_eliminated,

        # Default other fields
        'truncations': {ag: False for ag in base_agents},
        'infos': default_infos, # Start with default, env logic updates mask
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
    }
    # Recalculate active agents list one last time based on final terminations
    state['agents'] = [ag for ag in base_agents if not state['terminations'].get(ag, False)]

    # Make sure current_agent is valid
    if current_agent not in state['agents'] and state['agents']:
         # If the specified agent isn't active, default to the first active one
         state['agent_selection'] = state['agents'][0]
    elif not state['agents']:
        state['agent_selection'] = None # No active agents

    return state


class TestPerfectSearchIntegration(unittest.TestCase):

    def setUp(self):
        """Set up base environment, opponents, and PS instance."""
        self.num_players = 3
        self.training_agent = 'player_0'
        self.opponent_agents = ['player_1', 'player_2']

        # Use real, predictable hardcoded opponents
        self.opponent_models = {
            'player_1': GreedyCardSpammer(agent_name='player_1'),
            'player_2': TableFirstConservativeChallenger(agent_name='player_2')
        }

        # Set logging level high to avoid clutter during tests
        self.log_level = logging.CRITICAL
        self.base_env = LiarsDeckEnv(num_players=self.num_players, log_level=self.log_level)

        # Create the PS instance that will be tested
        # Note: Tests will often create their own env and pass its state to search
        self.ps = PerfectSearch(
            env=self.base_env, # Base env primarily for cloning
            training_agent=self.training_agent,
            opponent_models=self.opponent_models
        )
        self.ps.debug = False # Disable PS debug logs unless debugging a specific test


    def test_search_finds_immediate_win(self):
        """Scenario: PS agent can play their last card matching table card."""
        seed = 101
        state_dict = create_initial_state(
            num_players=self.num_players,
            current_agent=self.training_agent,
            table_card='Ace',
            hands={self.training_agent: ['Ace'], # Only one card left, matching table
                   'player_1': ['King', 'Queen'],
                   'player_2': ['King', 'Joker']},
            seed=seed
        )
        test_env = LiarsDeckEnv(num_players=self.num_players, log_level=self.log_level)
        test_env.reset(seed=seed)
        test_env.set_state(copy.deepcopy(state_dict))

        # Instantiate PS specific to this test's env setup if needed, or use self.ps
        current_ps = PerfectSearch(test_env, self.training_agent, self.opponent_models)
        current_ps.debug = False # Ensure debug is off

        action_probs, best_action, best_value = current_ps.search(test_env.get_state())

        self.assertEqual(best_action, 0, "Should play the last Ace (action 0)")
        # Winning should result in a very high value
        self.assertGreater(best_value, 4000, "Winning value should be very high")


    def test_search_finds_necessary_challenge_opponent_bluff(self):
        """Scenario: Opponent bluffs, PS has high penalties and must challenge."""
        seed = 102
        state_dict = create_initial_state(
            num_players=self.num_players,
            current_agent=self.training_agent,
            table_card='King',
            hands={self.training_agent: ['Ace', 'Ace', 'Ace'], # No Kings, cannot play safely
                   'player_1': ['Queen', 'Queen', 'Joker'], # P1 will bluff 1 King (plays Queen)
                   'player_2': ['King', 'King']},
            penalties={self.training_agent: 2, 'player_1': 0, 'player_2': 0}, # PS has high penalty
            # Simulate P1 having just bluffed (claimed 1 King, played Queen)
            last_action_agent='player_1',
            last_action=1, # Claimed 1
            last_played_cards={'player_1': ['Queen']}, # The actual bluff
            last_action_bluff=True,
            seed=seed
        )
        test_env = LiarsDeckEnv(num_players=self.num_players, log_level=self.log_level)
        test_env.reset(seed=seed)
        test_env.set_state(copy.deepcopy(state_dict))

        current_ps = PerfectSearch(test_env, self.training_agent, self.opponent_models)
        current_ps.debug = False

        action_probs, best_action, best_value = current_ps.search(test_env.get_state())

        self.assertEqual(best_action, 6, "Should challenge the opponent's bluff (action 6)")
        # Successful challenge value should be positive and significant
        self.assertGreater(best_value, 1000, "Successful challenge value should be high positive")


    def test_search_avoids_bad_challenge_opponent_valid(self):
        """Scenario: Opponent makes a valid play, PS should not challenge."""
        seed = 103
        state_dict = create_initial_state(
            num_players=self.num_players,
            current_agent=self.training_agent,
            table_card='Queen',
            hands={self.training_agent: ['King', 'King', 'Ace'], # Has cards, but cannot play Queen
                   'player_1': ['Queen', 'Joker', 'Ace'], # P1 plays 1 Queen (valid)
                   'player_2': ['King', 'Queen']},
            penalties={self.training_agent: 2, 'player_1': 0, 'player_2': 0}, # PS high penalty
            # Simulate P1 having just played a valid Queen
            last_action_agent='player_1',
            last_action=1, # Claimed 1
            last_played_cards={'player_1': ['Queen']}, # The valid play
            last_action_bluff=False,
            seed=seed
        )
        test_env = LiarsDeckEnv(num_players=self.num_players, log_level=self.log_level)
        test_env.reset(seed=seed)
        test_env.set_state(copy.deepcopy(state_dict))

        current_ps = PerfectSearch(test_env, self.training_agent, self.opponent_models)
        current_ps.debug = False

        action_probs, best_action, best_value = current_ps.search(test_env.get_state())

        # PS has no Queens. Challenging P1 (valid) is bad.
        # Blufﬁng 1 King (action 3) is possible.
        # The simulation should show challenging (action 6) leads to a penalty (-ve value)
        # and bluffing (action 3) might be neutral or slightly positive depending on simulation outcome.
        self.assertNotEqual(best_action, 6, "Should NOT challenge the opponent's valid play")
        # Likely chooses to bluff action 3 (Play 1 Non-Table) as the least bad option
        self.assertEqual(best_action, 3, "Should likely bluff (action 3) as best alternative")
        # Value should not be extremely negative (like a failed challenge penalty)
        self.assertGreater(best_value, -4000, "Value should be better than penalty for failed challenge")


    def test_search_makes_safe_play_high_penalty(self):
        """Scenario: PS has high penalty and holds table cards, should play safely."""
        seed = 104
        state_dict = create_initial_state(
            num_players=self.num_players,
            current_agent=self.training_agent,
            table_card='King',
            hands={self.training_agent: ['King', 'King', 'Ace'], # Has 2 Kings (table), 1 Ace (non)
                   'player_1': ['Queen', 'Queen'],
                   'player_2': ['Joker', 'Ace']},
            penalties={self.training_agent: 2, 'player_1': 0, 'player_2': 1}, # PS very high penalty
            last_action_agent='player_2', # Assume some previous action
            last_action=1,
            last_played_cards={'player_2': ['Ace']},
            last_action_bluff=False,
            seed=seed
        )
        test_env = LiarsDeckEnv(num_players=self.num_players, log_level=self.log_level)
        test_env.reset(seed=seed)
        test_env.set_state(copy.deepcopy(state_dict))

        current_ps = PerfectSearch(test_env, self.training_agent, self.opponent_models)
        current_ps.debug = False

        action_probs, best_action, best_value = current_ps.search(test_env.get_state())

        # Valid actions: Play 1 King (0), Play 2 Kings (1), Play 1 Ace (3), Challenge (6)
        # With penalty=2, challenging is very risky if opponent might be valid.
        # Blufﬁng (playing Ace as King) is also risky.
        # Safest is playing the actual Kings. Playing 2 (action 1) gets rid of more cards.
        # Playing 1 (action 0) is also safe.
        self.assertIn(best_action, [0, 1], "Should play table cards (action 0 or 1) safely")
        self.assertGreater(best_value, -100, "Value for safe play should not be highly negative")


    def test_search_bluffs_when_safe_and_necessary(self):
        """Scenario: Low penalty, no table cards, bluffing is the best option."""
        seed = 105
        state_dict = create_initial_state(
            num_players=self.num_players,
            current_agent=self.training_agent,
            table_card='Ace', # Table card is Ace
            hands={self.training_agent: ['King', 'King', 'Queen'], # No Aces held
                   'player_1': ['Ace', 'Joker'],
                   'player_2': ['Queen', 'King']},
            penalties={self.training_agent: 0, 'player_1': 1, 'player_2': 0}, # PS low penalty
            # Assume P2 just played something valid
            last_action_agent='player_2',
            last_action=1,
            last_played_cards={'player_2': ['King']},
            last_action_bluff=False,
            seed=seed
        )
        test_env = LiarsDeckEnv(num_players=self.num_players, log_level=self.log_level)
        test_env.reset(seed=seed)
        test_env.set_state(copy.deepcopy(state_dict))

        current_ps = PerfectSearch(test_env, self.training_agent, self.opponent_models)
        current_ps.debug = False # Set True for detailed simulation logs if needed

        action_probs, best_action, best_value = current_ps.search(test_env.get_state())

        # Valid actions: Play 1 King/Queen as Ace (3), Play 2 K/Q as Ace (4), Play 3 K/Q as Ace (5), Challenge (6)
        # PS has no Aces. Challenging P2's valid play is bad.
        # Simulation should explore bluffing. Playing 1 non-table card (action 3) is a common bluff.
        self.assertIn(best_action, [3, 4, 5], "Should choose to bluff (action 3, 4, or 5)")
        # The value might be slightly positive or neutral if the simulation doesn't immediately see a failure
        self.assertGreater(best_value, -1000, "Value for potential bluff should be better than failed challenge")

    def test_search_handles_forced_challenge_scenario_correctly(self):
        """ Scenario: Only PS and one opponent remain, opponent plays last card.
            PS should be forced to challenge (or win if opponent bluffed).
            This tests if the simulation correctly handles the env's forced challenge rule.
        """
        seed = 106
        opponent_id = 'player_1'
        state_dict = create_initial_state(
            num_players=3, # Start with 3, but one is terminated
            current_agent=self.training_agent,
            table_card='King',
            hands={
                self.training_agent: ['Ace', 'Ace'], # PS holds non-table cards
                opponent_id: ['Queen'], # Opponent has one card left, will play it
                'player_2': [] # Player 2 is already out
            },
            terminations={'player_2': True}, # Player 2 already eliminated
             # Simulate opponent P1 having just played their last card (Queen) as a King bluff
            last_action_agent=opponent_id,
            last_action=1, # Claimed 1 King
            last_played_cards={opponent_id: ['Queen']},
            last_action_bluff=True,
            seed=seed
        )
        # Manually update 'agents' list for the current env state
        state_dict['agents'] = [self.training_agent, opponent_id]

        test_env = LiarsDeckEnv(num_players=self.num_players, log_level=self.log_level)
        test_env.reset(seed=seed)
        test_env.set_state(copy.deepcopy(state_dict))

        # Set agent selection correctly after state set
        test_env.agent_selection = self.training_agent
        test_env._agent_selector.reinit(test_env.agents) # Reinit selector with active agents
        # Make sure the selector points to the training agent
        while test_env.agent_selection != self.training_agent:
             test_env.agent_selection = test_env._agent_selector.next()


        current_ps = PerfectSearch(test_env, self.training_agent, self.opponent_models)
        # current_ps.debug = True # Enable for debugging this complex scenario

        # In this state, PS's only valid action should be Challenge (6), forced by env rules.
        # The search simulates this challenge. Since P1 bluffed, the challenge succeeds.
        action_probs, best_action, best_value = current_ps.search(test_env.get_state())

        # current_ps.debug = False # Disable after debugging

        self.assertEqual(best_action, 6, "Should be forced to challenge (action 6)")
        self.assertGreater(best_value, 1000, "Value should reflect successful forced challenge")


if __name__ == '__main__':
    unittest.main()