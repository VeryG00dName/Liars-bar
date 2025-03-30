#!/usr/bin/env python3
# test_ps_unit.py - Unit Tests for PerfectSearch class

import unittest
from unittest import mock
import numpy as np
import torch # Needed if NN models are involved, even if mocked

# Modules to test
from src.model.ps import PerfectSearch
# Assuming config might be needed for NN input shapes if not mocked fully
from src import config


# --- Mock Objects ---

class MockEnv:
    """A simplified mock of the LiarsDeckEnv for unit testing PS."""
    def __init__(self, num_players=3):
        self.num_players = num_players
        self.possible_agents = [f'player_{i}' for i in range(num_players)]
        self.agent_selection = 'player_0' # Default starting agent
        self.players_hands = {agent: [] for agent in self.possible_agents}
        self.penalties = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.round_eliminated = {agent: False for agent in self.possible_agents}
        self.table_card = "King"
        self.last_action = None
        self.last_action_agent = None
        self.last_played_cards = {agent: [] for agent in self.possible_agents}
        self.round = 1
        self.winner = None
        self.infos = {agent: {'action_mask': [1]*7} for agent in self.possible_agents} # Default: all valid
        self.num_players = num_players # Explicitly add if needed by tested code

    def clone(self):
        cloned = MockEnv(self.num_players)
        cloned.possible_agents = self.possible_agents[:]
        cloned.agent_selection = self.agent_selection
        cloned.players_hands = {k: v[:] for k, v in self.players_hands.items()}
        cloned.penalties = self.penalties.copy()
        cloned.terminations = self.terminations.copy()
        cloned.round_eliminated = self.round_eliminated.copy()
        cloned.table_card = self.table_card
        cloned.last_action = self.last_action
        cloned.last_action_agent = self.last_action_agent
        cloned.last_played_cards = {k: v[:] for k, v in self.last_played_cards.items()}
        cloned.round = self.round
        cloned.winner = self.winner
        cloned.infos = {k: v.copy() for k, v in self.infos.items()}
        return cloned

    def set_state(self, state_dict):
        for key, value in state_dict.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    setattr(self, key, value.copy())
                elif isinstance(value, list):
                     setattr(self, key, value[:])
                else:
                    setattr(self, key, value)
        for agent in self.possible_agents:
            if agent not in self.infos:
                self.infos[agent] = {'action_mask': [1]*7}
            elif 'action_mask' not in self.infos[agent]:
                 self.infos[agent]['action_mask'] = [1]*7

    def get_state(self):
        # Ensure all keys potentially accessed by PS are present
        state = {
            'possible_agents': self.possible_agents,
            'agent_selection': self.agent_selection,
            'players_hands': self.players_hands,
            'penalties': self.penalties,
            'terminations': self.terminations,
            'round_eliminated': self.round_eliminated,
            'table_card': self.table_card,
            'last_action': self.last_action,
            'last_action_agent': self.last_action_agent,
            'last_played_cards': self.last_played_cards,
            'round': self.round,
            'winner': self.winner,
            'infos': self.infos,
            'num_players': self.num_players,
            # Add defaults for other keys potentially needed by full state logic
            'deck': [],
            'table_card_idx': TABLE_CARD_MAP.get(self.table_card, 0),
            'penalty_thresholds': {a: 3 for a in self.possible_agents},
            '_cumulative_rewards': {a: 0 for a in self.possible_agents},
            'pending_bluff': None,
            'last_agent_action': {a: None for a in self.possible_agents},
            'consecutive_action_count': {a: 0 for a in self.possible_agents},
            'successful_bluffs': {a: 0 for a in self.possible_agents},
            'failed_bluffs': {a: 0 for a in self.possible_agents},
            'successful_challenges': {a: 0 for a in self.possible_agents},
            'failed_challenges': {a: 0 for a in self.possible_agents},
            'bluff_counts': {a: 0 for a in self.possible_agents},
            'total_plays': {a: 0 for a in self.possible_agents},
            'public_opponent_histories': {a: [] for a in self.possible_agents},
            'private_opponent_histories': {a: [] for a in self.possible_agents},
            'random_seed': 0 # Or capture actual seed state if necessary
        }
        # Ensure infos sub-dict is complete for all agents
        for agent in self.possible_agents:
            if agent not in state['infos']:
                state['infos'][agent] = {'action_mask': [1]*7}
            elif 'action_mask' not in state['infos'][agent]:
                state['infos'][agent]['action_mask'] = [1]*7
        return state


    def observe(self, agent, new=False, newer=False):
        if agent not in self.infos:
             self.infos[agent] = {'action_mask': [1]*7}
        elif 'action_mask' not in self.infos[agent]:
             self.infos[agent]['action_mask'] = [1]*7

        # Provide a dummy observation consistent with NN input expectation
        # obs_dim = old_obs_len + obp (2) + memory (STRATEGY_DIM * (num_players-1))
        # Assume old_obs_len = 10 for mocking purposes
        old_obs_len = 10
        # Use config.STRATEGY_DIM if available, otherwise use a placeholder like 4
        strategy_dim = getattr(config, 'STRATEGY_DIM', 4)
        obs_dim = old_obs_len + 2 + strategy_dim * (self.num_players - 1)
        dummy_obs = np.zeros(obs_dim, dtype=np.float32)

        # Return the observation format expected by the calling code
        # Need to return {agent: obs} dict format
        return {agent: dummy_obs}

    def step(self, action):
        current_agent_index = self.possible_agents.index(self.agent_selection)
        acting_agent = self.agent_selection
        next_agent_index = (current_agent_index + 1) % self.num_players
        self.agent_selection = self.possible_agents[next_agent_index]
        self.last_action = action
        self.last_action_agent = acting_agent

# Need TABLE_CARD_MAP for get_state
TABLE_CARD_MAP = {"King": 0, "Queen": 1, "Ace": 2}

# --- Test Class ---

class TestPerfectSearchUnit(unittest.TestCase):

    def setUp(self):
        """Set up mocks and PS instance for each test."""
        self.mock_env = MockEnv(num_players=3)
        self.training_agent = 'player_0'
        self.opponent_agents = ['player_1', 'player_2']

        self.mock_opponent1 = mock.Mock()
        self.mock_opponent1.play_turn = mock.Mock(return_value=0) # Hardcoded model

        self.mock_opponent2 = mock.Mock(spec=torch.nn.Module) # NN model
        mock_probs = torch.tensor([[0.1, 0.5, 0.1, 0.1, 0.1, 0.0, 0.1]])
        # **FIX #2**: Return 3 values to satisfy the try block in _select_opponent_action
        self.mock_opponent2.return_value = (mock_probs, None, None)

        self.opponent_models = {
            'player_1': self.mock_opponent1,
            'player_2': self.mock_opponent2
        }

        self.ps = PerfectSearch(
            env=self.mock_env,
            training_agent=self.training_agent,
            opponent_models=self.opponent_models
        )
        # Disable debug logging during tests unless specifically needed
        self.ps.debug = False


    # --- Test __init__ ---
    def test_init_stores_arguments(self):
        self.assertIs(self.ps.base_env, self.mock_env)
        self.assertEqual(self.ps.training_agent, self.training_agent)
        self.assertIs(self.ps.opponent_models, self.opponent_models)

    def test_init_identifies_opponents(self):
        self.assertListEqual(sorted(self.ps.opponent_agents), sorted(self.opponent_agents))

    def test_init_plan_state(self):
        self.assertListEqual(self.ps.action_sequence, [])
        self.assertEqual(self.ps.sequence_position, 0)

    # --- Test Plan Management ---
    def test_invalidate_plan_resets_sequence(self):
        self.ps.action_sequence = [('player_0', 1), ('player_1', 0)]
        self.ps.sequence_position = 1
        self.ps.invalidate_plan()
        self.assertListEqual(self.ps.action_sequence, [])
        self.assertEqual(self.ps.sequence_position, 0)

    def test_invalidate_plan_on_empty_sequence(self):
        self.ps.action_sequence = []
        self.ps.sequence_position = 0
        self.ps.invalidate_plan()
        self.assertListEqual(self.ps.action_sequence, [])
        self.assertEqual(self.ps.sequence_position, 0)

    def test_get_next_action_empty_plan(self):
        self.ps.action_sequence = []
        self.ps.sequence_position = 0
        action = self.ps.get_next_agent_action('player_0')
        self.assertIsNone(action)

    def test_get_next_action_end_of_plan(self):
        self.ps.action_sequence = [('player_0', 1)]
        self.ps.sequence_position = 1
        action = self.ps.get_next_agent_action('player_0')
        self.assertIsNone(action)

    @mock.patch.object(PerfectSearch, 'invalidate_plan')
    def test_get_next_action_agent_mismatch(self, mock_invalidate):
        self.ps.action_sequence = [('player_1', 0)]
        self.ps.sequence_position = 0
        action = self.ps.get_next_agent_action('player_0')
        self.assertIsNone(action)
        mock_invalidate.assert_called_once()

    def test_get_next_action_success(self):
        agent = self.training_agent
        planned_action = 1
        self.ps.action_sequence = [(agent, planned_action), ('player_1', 2)]
        self.ps.sequence_position = 0
        self.mock_env.infos[agent]['action_mask'] = [0, 1, 0, 0, 0, 0, 0] # Action 1 is valid
        action = self.ps.get_next_agent_action(agent)
        self.assertEqual(action, planned_action)
        self.assertEqual(self.ps.sequence_position, 1) # Position advanced

    @mock.patch.object(PerfectSearch, 'invalidate_plan')
    def test_get_next_action_action_invalid(self, mock_invalidate):
        agent = self.training_agent
        planned_action = 1
        self.ps.action_sequence = [(agent, planned_action)]
        self.ps.sequence_position = 0
        self.mock_env.infos[agent]['action_mask'] = [1, 0, 0, 0, 0, 0, 0] # Action 1 is NOT valid
        action = self.ps.get_next_agent_action(agent)
        self.assertIsNone(action)
        mock_invalidate.assert_called_once()

    @mock.patch.object(PerfectSearch, 'invalidate_plan')
    def test_get_next_action_handles_exception(self, mock_invalidate):
        agent = self.training_agent
        planned_action = 1
        self.ps.action_sequence = [(agent, planned_action)]
        self.ps.sequence_position = 0
        # Mock observe to raise an exception during validity check
        self.mock_env.observe = mock.Mock(side_effect=Exception("Mock Observe Error"))
        action = self.ps.get_next_agent_action(agent)
        self.assertIsNone(action)
        mock_invalidate.assert_called_once()


    # --- Test Search Logic (Mocking simulate_round) ---

    @mock.patch.object(PerfectSearch, 'simulate_round')
    @mock.patch.object(PerfectSearch, 'invalidate_plan')
    def test_search_chooses_highest_value(self, mock_invalidate, mock_simulate):
        """Test search selects the action with the highest simulation value."""
        agent = self.training_agent
        sim_results = {
            0: (10.0, [(agent, 0)], False, False),
            1: (100.0, [(agent, 1)], False, False),
            2: (50.0, [(agent, 2)], False, False),
        }
        # **FIX #1**: side_effect signature matches arguments passed by search
        mock_simulate.side_effect = lambda state, action, cache, depth: sim_results[action]
        self.mock_env.infos[agent]['action_mask'] = [1, 1, 1, 0, 0, 0, 0] # Actions 0, 1, 2 are valid

        action_probs, best_action, best_value = self.ps.search(self.mock_env.get_state())

        mock_invalidate.assert_called_once() # Search should invalidate previous plan
        self.assertEqual(best_action, 1)
        self.assertEqual(best_value, 100.0)
        np.testing.assert_array_equal(action_probs, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(self.ps.action_sequence, [(agent, 1)])
        self.assertEqual(self.ps.sequence_position, 0)

    @mock.patch.object(PerfectSearch, 'simulate_round')
    @mock.patch.object(PerfectSearch, 'invalidate_plan')
    def test_search_prioritizes_opponent_penalty(self, mock_invalidate, mock_simulate):
        """Test search prioritizes actions causing opponent penalties (value > 1000)."""
        agent = self.training_agent
        sim_results = {
            0: (50.0, [(agent, 0)], False, False),       # Normal action
            1: (2000.0, [(agent, 1)], False, False),     # High value -> Opponent penalty
            6: (100.0, [(agent, 6)], False, False)       # Challenge action
        }
        # **FIX #1**: side_effect signature matches arguments passed by search
        mock_simulate.side_effect = lambda state, action, cache, depth: sim_results[action]
        self.mock_env.infos[agent]['action_mask'] = [1, 1, 0, 0, 0, 0, 1] # Actions 0, 1, 6 are valid

        action_probs, best_action, best_value = self.ps.search(self.mock_env.get_state())

        self.assertEqual(best_action, 1) # Should choose the high value action
        self.assertEqual(best_value, 2000.0)
        self.assertEqual(self.ps.action_sequence, [(agent, 1)])

    @mock.patch('src.model.ps.PerfectSearch.simulate_round') # Patching the method directly
    @mock.patch.object(PerfectSearch, 'invalidate_plan')
    def test_search_uses_fallback_on_negative_values(self, mock_invalidate, mock_simulate):
        """Test search uses fallback logic when all simulations yield poor values."""
        agent = self.training_agent
        sim_results_bad = {
            3: (-5000.0, [(agent, 3)], False, False), # Action 3 is bad
            6: (-10000.0, [(agent, 6)], True, False), # Action 6 is worse
        }
        # Result for the fallback action (action 0 in this case)
        sim_result_fallback = (10.0, [(agent, 0)], False, False)

        # **FIX #1**: side_effect signature matches arguments passed by search (state, action, cache, depth)
        def side_effect_func(state, action, cache, depth):
            if action == 0: # Fallback action simulation
                return sim_result_fallback
            elif action in sim_results_bad: # Original bad actions
                 return sim_results_bad[action]
            else: # Default for any other unexpected action tested
                 return (-99999.0, [(agent, action)], False, False)

        mock_simulate.side_effect = side_effect_func
        # Setup env for fallback: agent has 'King' (table card), 'Queen'
        self.mock_env.players_hands[agent] = ['King', 'Queen']
        self.mock_env.table_card = 'King'
        # Valid actions: Play 1 Table (0), Play 1 Non-Table (3), Challenge (6)
        self.mock_env.infos[agent]['action_mask'] = [1, 0, 0, 1, 0, 0, 1]

        action_probs, best_action, best_value = self.ps.search(self.mock_env.get_state())

        # Assertions
        self.assertEqual(best_action, 0) # Should select fallback action 0
        self.assertEqual(best_value, 10.0) # Should have the fallback value
        self.assertEqual(self.ps.action_sequence, [(agent, 0)]) # Sequence from fallback
        # Check simulate was called for all originally valid actions PLUS the fallback
        called_actions = sorted([call.args[1] for call in mock_simulate.call_args_list])
        self.assertListEqual(called_actions, [0, 3, 6]) # Called for 0, 3, 6
        self.assertEqual(mock_simulate.call_count, 3)


    @mock.patch.object(PerfectSearch, 'simulate_round')
    @mock.patch.object(PerfectSearch, 'invalidate_plan')
    def test_search_returns_correct_format(self, mock_invalidate, mock_simulate):
        """Verify the return format of the search method."""
        agent = self.training_agent
        sim_results = { 0: (10.0, [(agent, 0)], False, False) }
        # **FIX #1**: side_effect signature matches arguments passed by search
        mock_simulate.side_effect = lambda state, action, cache, depth: sim_results[action]
        self.mock_env.infos[agent]['action_mask'] = [1, 0, 0, 0, 0, 0, 0] # Only action 0 is valid

        action_probs, best_action, best_value = self.ps.search(self.mock_env.get_state())

        self.assertIsInstance(action_probs, np.ndarray)
        self.assertIsInstance(best_action, int)
        self.assertIsInstance(best_value, float)
        self.assertEqual(action_probs.shape, (7,))
        self.assertEqual(np.sum(action_probs), 1.0)
        self.assertEqual(action_probs[best_action], 1.0)
        self.assertEqual(best_action, 0)
        self.assertEqual(best_value, 10.0)

    @mock.patch.object(PerfectSearch, 'simulate_round')
    def test_search_handles_no_valid_actions_error(self, mock_simulate):
        """Test search raises error if no actions are valid (shouldn't happen in theory)."""
        agent = self.training_agent
        self.mock_env.infos[agent]['action_mask'] = [0] * 7 # No valid actions
        with self.assertRaisesRegex(RuntimeError, f"No valid actions available for {agent}"):
            self.ps.search(self.mock_env.get_state())

    @mock.patch.object(PerfectSearch, 'simulate_round')
    def test_search_uses_fresh_cache_per_root_simulation(self, mock_simulate):
        """Verify simulate_round gets a fresh opponent cache for each root action."""
        agent = self.training_agent
        sim_results = {
            0: (10.0, [(agent, 0)], False, False),
            1: (20.0, [(agent, 1)], False, False),
        }
        caches_received_at_depth_0 = []
        # **FIX #1**: side_effect signature matches arguments passed by search
        def side_effect_func(state, action, cache, depth):
            if depth == 0: # Only capture cache for root calls
                 caches_received_at_depth_0.append(cache.copy())
            return sim_results.get(action, (-999.0, [], False, False))

        mock_simulate.side_effect = side_effect_func
        self.mock_env.infos[agent]['action_mask'] = [1, 1, 0, 0, 0, 0, 0] # Actions 0, 1 are valid

        self.ps.search(self.mock_env.get_state())

        # Check how many times simulate was called with depth=0
        root_calls = [call for call in mock_simulate.call_args_list if call.kwargs.get('depth') == 0]

        self.assertEqual(len(root_calls), 2) # Should have called for action 0 and 1 at depth 0
        self.assertEqual(len(caches_received_at_depth_0), 2) # Should have captured 2 caches
        # Check that the captured caches were empty (as expected for root calls)
        for cache in caches_received_at_depth_0:
            self.assertDictEqual(cache, {})


    # --- Test _select_opponent_action (Basic checks, more in accuracy tests) ---

    @mock.patch('src.model.ps.torch.no_grad') # Mock torch context manager
    def test_select_opponent_action_calls_hardcoded_model(self, mock_no_grad):
        """Verify _select_opponent_action calls play_turn for hardcoded models."""
        opponent_agent = 'player_1'
        opponent_action_cache = {}
        self.mock_env.agent_selection = opponent_agent
        expected_mask = [0, 0, 1, 0, 0, 0, 0]
        expected_table_card = self.mock_env.table_card
        # Need to call observe to potentially populate infos/get obs
        _ = self.mock_env.observe(opponent_agent, new=True)
        expected_observation = self.mock_env.observe(opponent_agent, new=True)[opponent_agent] # Get the dummy obs

        self.mock_env.infos[opponent_agent]['action_mask'] = expected_mask
        self.mock_opponent1.play_turn.return_value = 2 # Action 2 is valid according to mask

        selected_action = self.ps._select_opponent_action(self.mock_env, opponent_agent, opponent_action_cache)

        # Assert play_turn was called correctly
        self.mock_opponent1.play_turn.assert_called_once()
        call_args, call_kwargs = self.mock_opponent1.play_turn.call_args
        np.testing.assert_array_equal(call_args[0], expected_observation)
        np.testing.assert_array_equal(call_args[1], expected_mask)
        self.assertEqual(call_kwargs.get('table_card'), expected_table_card)

        self.assertEqual(selected_action, 2)
        # Check cache was populated
        self.assertTrue(len(opponent_action_cache) > 0)
        # Simple check if *something* was cached, key generation tested elsewhere
        cached_action = list(opponent_action_cache.values())[0]
        self.assertEqual(cached_action, 2)


    @mock.patch('src.model.ps.torch.no_grad')
    @mock.patch('src.model.ps.np.argmax')
    @mock.patch('src.model.ps.torch.tensor')
    def test_select_opponent_action_calls_nn_model(self, mock_tensor, mock_argmax, mock_no_grad):
        """Verify _select_opponent_action calls forward pass for NN models."""
        opponent_agent = 'player_2'
        opponent_action_cache = {}
        self.mock_env.agent_selection = opponent_agent
        # Make action 1 valid
        self.mock_env.infos[opponent_agent]['action_mask'] = [0, 1, 0, 0, 0, 0, 0]

        # Setup mocks for tensor creation and model call
        mock_tensor_instance = mock.Mock()
        mock_tensor_unsqueezed = mock.Mock(name="unsqueezed_tensor")
        mock_tensor_instance.unsqueeze.return_value = mock_tensor_unsqueezed
        mock_tensor.return_value = mock_tensor_instance

        mock_argmax.return_value = 1 # Ensure argmax returns the valid action 1

        selected_action = self.ps._select_opponent_action(self.mock_env, opponent_agent, opponent_action_cache)

        # Assertions
        mock_tensor.assert_called() # Tensor creation attempted
        mock_tensor_instance.unsqueeze.assert_called_once_with(0)
        # **FIX #2**: Check it was called exactly once now
        self.mock_opponent2.assert_called_once_with(mock_tensor_unsqueezed, None)
        mock_argmax.assert_called_once() # Argmax should be called on the probabilities
        self.assertEqual(selected_action, 1) # Action returned by argmax
        # Check cache
        self.assertTrue(len(opponent_action_cache) > 0)
        cached_action = list(opponent_action_cache.values())[0]
        self.assertEqual(cached_action, 1)

    def test_select_opponent_action_cache_hit(self):
        """Test that caching prevents recalculating opponent action."""
        opponent_agent = 'player_1' # Use hardcoded model for simplicity
        opponent_action_cache = {}
        self.mock_env.agent_selection = opponent_agent
        self.mock_env.infos[opponent_agent]['action_mask'] = [1, 0, 0, 0, 0, 0, 0] # Action 0 valid
        self.mock_opponent1.play_turn.return_value = 0

        # First call - should call play_turn
        action1 = self.ps._select_opponent_action(self.mock_env, opponent_agent, opponent_action_cache)
        self.assertEqual(action1, 0)
        self.mock_opponent1.play_turn.assert_called_once()

        # Reset mock for the next call check
        self.mock_opponent1.play_turn.reset_mock()

        # Second call - should hit cache, NOT call play_turn
        action2 = self.ps._select_opponent_action(self.mock_env, opponent_agent, opponent_action_cache)
        self.assertEqual(action2, 0)
        self.mock_opponent1.play_turn.assert_not_called()


if __name__ == '__main__':
    unittest.main()