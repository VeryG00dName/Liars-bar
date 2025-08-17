# src/env/liars_deck_env_core.py
import logging
import random

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector
import copy
from src import config

# Example utility imports - adjust as needed for your project structure
from src.env.liars_deck_env_utils_2 import (
    create_deck,
    encode_hand,
    decode_action,
    simulate_russian_roulette,
    TABLE_CARD_MAP
)
from src.env.liars_deck_env_utils import (
    apply_action,
    get_observations,
    get_new_observations,
    get_newer_observations,
    get_newest_observations,
    get_newerest_observations
)


class LiarsDeckEnv(AECEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_players=3,
        render_mode=None,
        log_level=logging.WARNING,
        scoring_params=None
    ):
        super().__init__()
        self.render_mode = render_mode
        self.num_players = num_players
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agents = []

        # Default scoring parameters
        default_scoring_params = config.DEFAULT_SCORING_PARAMS

        # Merge provided scoring_params with defaults
        if scoring_params is None:
            scoring_params = default_scoring_params
        else:
            for k, v in default_scoring_params.items():
                if k not in scoring_params:
                    scoring_params[k] = v

        self.scoring_params = scoring_params

        # Configure logger
        self.logger = logging.getLogger('LiarsDeckEnv')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(log_level)

        # Example hand vector calculation
        example_hand = ["King", "Queen", "Ace", "Joker", "King"]
        example_hand_vector = encode_hand(example_hand, "King")
        hand_vector_length = len(example_hand_vector)  # e.g., 2

        # Opponent feature dimension: action_type (3) + count (1) = 4 per opponent
        opponent_feature_dim = 4 * (self.num_players - 1)

        # obs_dim = hand_vector (2) + last_action (1) + active_players (num_players) + opponent_features
        obs_dim = hand_vector_length + 1 + self.num_players + opponent_feature_dim

        self.observation_spaces = {
            agent: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: spaces.Discrete(7) for agent in self.possible_agents
        }

        # Initialize game state
        self.deck = []
        self.penalties = {}
        self.penalty_thresholds = {}
        self.last_action = None
        self.last_played_cards = {agent: [] for agent in self.possible_agents}
        self.last_action_agent = None
        self.last_action_bluff = None
        self.must_force = False
        
        # Tracking results and statistics
        self.successful_bluffs = {}
        self.failed_bluffs = {}
        self.successful_challenges = {}
        self.failed_challenges = {}

        # Opponent tracking
        self.public_opponent_histories = {agent: [] for agent in self.possible_agents}
        self.private_opponent_histories = {agent: [] for agent in self.possible_agents}
        self.bluff_counts = {agent: 0 for agent in self.possible_agents}
        self.total_plays = {agent: 0 for agent in self.possible_agents}

        self.terminations = {}
        self.round_eliminated = {}
        self.truncations = {}
        self.infos = {}
        self._cumulative_rewards = {}
        self.players_hands = {}
        self.winner = None
        self._agent_selector = None
        self.rewards = {}
        self.round = 0
        self.game_history = []
        self.global_step = 0
        # Initialize tracking for newer observations
        self.last_challenge_success = None  # Track outcome of the last challenge
        
        self.last_agent_action = {agent: None for agent in self.possible_agents}
        self.consecutive_action_count = {agent: 0 for agent in self.possible_agents}

        # Initialize table_card
        self.table_card = random.choice(["King", "Queen", "Ace"])

        self.pending_bluff = None

        self.reset()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def observe(self, agent, new=False, newer=False, newest=False, newerest=False):
        """
        Generates the observation for the agent.
        If new=True, it uses the new observation construction.
        If newer=True, it uses the newer observation that focuses on action history.
        In addition to the observation vector, it attaches an "action_mask" in the infos.
        """
        if  newerest:
            obs_dict = get_newerest_observations(self, agent_specific=agent)
        elif newest:
            obs_dict = get_newest_observations(self, agent_specific=agent)
        elif newer:
            obs_dict = get_newer_observations(self, agent_specific=agent)
        elif new:
            obs_dict = get_new_observations(self, agent_specific=agent)
        else:
            obs_dict = get_observations(self, agent_specific=agent)
        
        mask = self._compute_action_mask(agent)
        self.infos[agent]["action_mask"] = mask
        return obs_dict

    def _compute_action_mask(self, agent):
        """
        Returns a list of length 7 (action_space = Discrete(7)),
        where each entry is 1 if the action is valid and 0 otherwise.
        """
        # If the agent is terminated or round-eliminated, no actions are valid.
        if self.terminations.get(agent, False) or self.round_eliminated.get(agent, False):
            return [0]*7
        if self.must_force:
            mask= [0]*7
            mask[6] = 1
            return mask
        # Start with all actions valid.
        mask = [1]*7

        # Challenge action is action 6.
        # It is invalid if there is no claim (i.e. no last action agent or no played cards).
        if not self.last_action_agent or not self.last_played_cards.get(self.last_action_agent, []):
            mask[6] = 0

        # Get the agent's hand and count card types.
        current_hand = self.players_hands.get(agent, [])
        table_card = self.table_card
        # Cards that count toward playing table cards (either matching table_card or Joker).
        table_cards = [c for c in current_hand if c == table_card or c == "Joker"]
        # Cards that are not table cards.
        non_table_cards = [c for c in current_hand if c not in table_cards]

        # Actions 0-2: "Play" actions for table cards with count = (i % 3) + 1.
        # Actions 3-5: "Play" actions for non-table cards with count = (i % 3) + 1.
        for i in range(6):
            play_count = (i % 3) + 1
            if i < 3:
                if len(table_cards) < play_count:
                    mask[i] = 0
            else:
                if len(non_table_cards) < play_count:
                    mask[i] = 0

        return mask

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is None:
            seed = np.random.randint(0,1000000)
        np.random.seed(seed)
        self.np_random = np.random.default_rng(seed)
        random.seed(seed)
        self.deck = create_deck(self.np_random)
        self.penalties = {agent: 0 for agent in self.possible_agents}
        self.penalty_thresholds = {agent: simulate_russian_roulette(self.np_random) for agent in self.possible_agents}
        self.successful_bluffs = {agent: 0 for agent in self.possible_agents}
        self.failed_bluffs = {agent: 0 for agent in self.possible_agents}
        self.successful_challenges = {agent: 0 for agent in self.possible_agents}
        self.failed_challenges = {agent: 0 for agent in self.possible_agents}

        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.round_eliminated = {agent: False for agent in self.possible_agents}

        self.last_action = None
        self.last_played_cards = {agent: [] for agent in self.possible_agents}
        self.last_action_agent = None
        self.last_action_bluff = None
        self.must_force = False
        self.winner = None
        self.round = 0
        self.game_history = []
        self.global_step = 0
        
        self.opponent_histories = {agent: [] for agent in self.possible_agents}
        self.bluff_counts = {agent: 0 for agent in self.possible_agents}
        self.total_plays = {agent: 0 for agent in self.possible_agents}

        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)

        self.rewards = {agent: 0 for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

        # Reset tracking for newer observations
        self.last_challenge_success = None
        
        # Reset table_card at the start of each episode
        self.table_card = self.np_random.choice(["King", "Queen", "Ace"])
        self.logger.debug(f"Resetting environment. Initial table_card: {self.table_card}")

        # Reset the last_agent_action and consecutive_action_count
        for agent in self.possible_agents:
            self.last_agent_action[agent] = None
            self.consecutive_action_count[agent] = 0

        # Reset pending bluff (if any) ---
        self.pending_bluff = None

        self.start_new_round()
        return get_observations(self), self.infos

    def start_new_round(self):
        # Check for any pending bluff before starting a new round ---
        if self.pending_bluff is not None:
            bluffing_agent = self.pending_bluff["agent"]
            # For any agent who never challenged the bluff in this round, apply penalty.
            for ag in self.pending_bluff["unchallenged_agents"]:
                penalty = self.scoring_params['unchallenged_bluff_penalty'] * self.pending_bluff["num_cards"]
                self.rewards[ag] += penalty
                self.logger.debug(
                    f"Agent {ag} penalized at round end for not challenging bluff from {bluffing_agent}"
                )
            # Reward the bluffing agent for a successful, unchallenged bluff.
            self.rewards[bluffing_agent] += self.scoring_params['successful_bluff_reward']
            self.logger.info(
                f"Bluff by {bluffing_agent} was successful; rewarded {self.scoring_params['successful_bluff_reward']}"
            )
            # Clear pending bluff since it is now resolved.
            self.pending_bluff = None

        for agent in self.possible_agents:
            self.round_eliminated[agent] = False

        eligible_agents = [ag for ag in self.possible_agents if not self.terminations[ag]]
        if not eligible_agents:
            self.logger.info("No eligible agents remain for a new round. Game ends.")
            self._check_game_end()
            return

        # ### CHANGED: Reset consecutive action tracking for the new round
        for agent in self.possible_agents:
            self.last_agent_action[agent] = None
            self.consecutive_action_count[agent] = 0
        random.shuffle(eligible_agents)
        self.agents = eligible_agents
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next() if self.agents else None

        self.last_action = None
        self.last_played_cards = {agent: [] for agent in self.possible_agents}
        self.last_action_agent = None
        self.last_action_bluff = None
        self.round += 1
        self.table_card = self.np_random.choice(["King", "Queen", "Ace"])
        self.table_card_idx = TABLE_CARD_MAP[self.table_card]

        hand_size = 5
        self.players_hands = {}
        self.deck = create_deck(self.np_random)
        for agent in self.agents:
            self.players_hands[agent] = self.deck[:hand_size]
            self.deck = self.deck[hand_size:]

        self.infos = {agent: {} for agent in self.possible_agents}

        self.logger.info("Starting a new round.")
        self.logger.debug(f"New Table Card: {self.table_card}")
        for agent in self.agents:
            self.logger.debug(f"{agent}'s new hand: {self.players_hands[agent]}")

    def step(self, action):
        agent = self.agent_selection
        self.global_step += 1
        # Decode the action early for use in our new bluff logic.
        action_type, _, count = decode_action(action)

        # If there is a pending bluff and the acting agent is not the bluffing agent,
        # then if the agent does not challenge, penalize them immediately.
        if self.pending_bluff is not None:
            bluffing_agent = self.pending_bluff["agent"]
            if agent != bluffing_agent and action_type != "Challenge":
                penalty = self.scoring_params['unchallenged_bluff_penalty'] * self.pending_bluff["num_cards"]
                self.rewards[agent] += penalty
                self.logger.debug(
                    f"Agent {agent} penalized for not challenging bluff from {bluffing_agent} (action: {action_type})."
                )
                # Remove this agent from the pending list so they are not penalized repeatedly.
                self.pending_bluff["unchallenged_agents"].discard(agent)

        # Check for consecutive identical discrete actions
        if action == self.last_agent_action[agent]:
            self.consecutive_action_count[agent] += 1
        else:
            self.consecutive_action_count[agent] = 1
            self.last_agent_action[agent] = action

        # Apply penalty if the same discrete action is repeated.
        # Optionally, only penalize repeating "Play" actions (actions 0..5).
        if self.consecutive_action_count[agent] > 1 and action_type == "Play":
            self.rewards[agent] += self.scoring_params.get('consecutive_action_penalty', -1)
            self.logger.debug(
                f"Penalty applied to {agent} for repeating the same action {action}. "
                f"Consecutive count: {self.consecutive_action_count[agent]}"
            )

        # Apply the main logic for the environment step.
        apply_action(self, agent, action)
        self._cumulative_rewards[agent] = self.rewards[agent]
        self.logger.debug(f"Action applied by {agent}: {action}")
        self.logger.debug(f"Rewards after action: {self.rewards}")
        self.logger.debug(f"Terminations after action: {self.terminations}")

        # If a challenge action is taken, clear any pending bluff.
        if action_type == "Challenge":
            self.pending_bluff = None

        # If a bluff is played (a Play action that is not truthful), then set a pending bluff.
        if action_type == "Play" and self.last_action_bluff:
            # Create a pending bluff record where all other agents are expected to challenge.
            self.pending_bluff = {
            "agent": agent,
            "unchallenged_agents": set(self.possible_agents) - {agent},
            "num_cards": count  # Store bluff size
            }
            self.logger.debug(
                f"Pending bluff set by {agent}. Agents expected to challenge: {self.pending_bluff['unchallenged_agents']}"
            )

        self._advance_to_next_agent()
        self._check_round_end()
        self._check_game_end()

        if self.render_mode == 'human':
            self.render()

    def _check_round_end(self):
        active_agents = self._active_agents_in_round()
        self.logger.debug(f"Active agents in round: {active_agents}")
        if len(active_agents) <= 1:
            self.logger.debug("Round ending as only one or no active agents remain.")
            if len(active_agents) == 0:
                self.logger.info("No active agents remain. Ending the game.")
                self._check_game_end()
                return

            # Finalize any unmarked bluff status for the last agent
            for agent in self.possible_agents:
                for entry in reversed(self.opponent_histories[agent]):
                    if entry['action_type'] == "Play" and entry['was_bluff'] is None:
                        entry['was_bluff'] = False
                        self.total_plays[agent] += 1
                        self.logger.debug(f"Finalizing bluff status for {agent}: was_bluff=False")
                        break

            self.start_new_round()

    def _check_game_end(self):
        eligible_agents = [ag for ag in self.possible_agents if not self.terminations[ag]]
        self.logger.debug(f"Eligible agents after round: {eligible_agents}")
        if len(eligible_agents) == 1:
            winner = eligible_agents[0]
            self._declare_game_winner(winner)

    def _declare_game_winner(self, winner):
        self.rewards[winner] += self.scoring_params['game_win_bonus']
        self.infos[winner]['winner'] = True
        self.winner = winner

        for ag in self.possible_agents:
            if ag != winner:
                self.rewards[ag] += self.scoring_params['game_lose_penalty']

        for ag in self.possible_agents:
            self.terminations[ag] = True
        self.agent_selection = None
        self.logger.info(f"Game winner declared: {winner}")

    def _active_agents_in_round(self):
        return [
            ag for ag in self.possible_agents
            if not self.terminations[ag] and not self.round_eliminated[ag]
        ]

    def _advance_to_next_agent(self):
        active_round_agents = self._active_agents_in_round()
        self.logger.debug(f"Advancing to next agent. Active round agents: {active_round_agents}")
        if active_round_agents:
            try:
                self.agent_selection = self._agent_selector.next()
                self.logger.debug(f"Next agent selected: {self.agent_selection}")
                while (
                    self.terminations.get(self.agent_selection, False)
                    or self.round_eliminated.get(self.agent_selection, False)
                    or self.truncations.get(self.agent_selection, False)
                ):
                    self.logger.debug(f"Skipping agent {self.agent_selection} due to termination or truncation.")
                    self.agent_selection = self._agent_selector.next()
                    self.logger.debug(f"Next agent selected: {self.agent_selection}")
            except StopIteration:
                self.agent_selection = None
                self.logger.debug("No more agents to select. Ending episode.")
        else:
            self.agent_selection = None
            self.logger.debug("No active agents remaining. Ending episode.")

    def update_scoring_params(self, new_params):
        """
        Updates the environment's scoring parameters by merging in new values.
        This method allows on-the-fly tuning of rewards/penalties without restarting the environment.
        
        Args:
            new_params (dict): Dictionary of scoring parameter updates.
                            Only keys provided in new_params will be updated.
        """
        self.scoring_params.update(new_params)

    def clone(self):
        """
        Creates a deep copy of the current environment state for simulation purposes.
        Correctly handles agent_selector state.

        Returns:
            LiarsDeckEnv: A new environment with the identical state.
        """
        cloned_env = LiarsDeckEnv(
            num_players=self.num_players,
            render_mode=None,  # Simulations don't need rendering
            log_level=logging.WARNING,  # Quiet logging for simulations
            scoring_params=copy.deepcopy(self.scoring_params) # Deep copy params
        )

        # Copy core game state
        cloned_env.deck = self.deck.copy()
        cloned_env.table_card = self.table_card
        cloned_env.table_card_idx = self.table_card_idx

        # Deep copy players' hands (list copy is sufficient here)
        cloned_env.players_hands = {agent: hand.copy() for agent, hand in self.players_hands.items()}

        # Copy agent state
        cloned_env.agents = self.agents.copy() # Use the current list of active agents
        original_agent_selection = self.agent_selection # Store original selection

        # --- Corrected Agent Selector Handling ---
        # 1. Create a new selector based on the cloned 'agents' list
        cloned_env._agent_selector = agent_selector(cloned_env.agents)
        # 2. Set the cloned env's agent_selection initially based on the new selector's start
        if cloned_env.agents:
             cloned_env.agent_selection = cloned_env._agent_selector.reset() # Reset gets the first agent
        else:
             cloned_env.agent_selection = None
        # 3. Iterate the cloned selector until it matches the original agent selection
        if original_agent_selection is not None and cloned_env.agents:
             # Check if the original agent is still in the active list for the clone
             if original_agent_selection in cloned_env.agents:
                 # Cycle through the cloned selector until it lands on the original agent
                 # Important: Check against selected_agent attribute after calling next()
                 while cloned_env._agent_selector.selected_agent != original_agent_selection:
                     cloned_env.agent_selection = cloned_env._agent_selector.next()
                 # Ensure the cloned_env's agent_selection is correctly set after loop
                 cloned_env.agent_selection = cloned_env._agent_selector.selected_agent
             else:
                 # If the original agent is no longer active (e.g., terminated),
                 # the cloned selector starting from the beginning is the correct state.
                 pass
        # --- End Corrected Agent Selector Handling ---

        # Use deepcopy for potentially complex/nested or critical dictionaries
        cloned_env.penalties = copy.deepcopy(self.penalties)
        cloned_env.penalty_thresholds = copy.deepcopy(self.penalty_thresholds)
        cloned_env.last_action = self.last_action
        cloned_env.last_action_agent = self.last_action_agent
        cloned_env.last_action_bluff = self.last_action_bluff
        cloned_env.must_force = self.must_force
        cloned_env.round = self.round

        cloned_env.last_played_cards = {
            agent: cards.copy() if cards else []
            for agent, cards in self.last_played_cards.items()
        }

        cloned_env.terminations = copy.deepcopy(self.terminations)
        cloned_env.truncations = copy.deepcopy(self.truncations)
        cloned_env.round_eliminated = copy.deepcopy(self.round_eliminated)
        cloned_env.winner = self.winner
        
        cloned_env.game_history = copy.deepcopy(self.game_history)
        cloned_env.global_step = self.global_step
        
        cloned_env.rewards = copy.deepcopy(self.rewards)
        cloned_env._cumulative_rewards = copy.deepcopy(self._cumulative_rewards)
        cloned_env.infos = copy.deepcopy(self.infos)

        cloned_env.pending_bluff = copy.deepcopy(self.pending_bluff) if self.pending_bluff else None

        cloned_env.last_agent_action = copy.deepcopy(self.last_agent_action)
        cloned_env.consecutive_action_count = copy.deepcopy(self.consecutive_action_count)

        # Statistics
        cloned_env.successful_bluffs = copy.deepcopy(self.successful_bluffs)
        cloned_env.failed_bluffs = copy.deepcopy(self.failed_bluffs)
        cloned_env.successful_challenges = copy.deepcopy(self.successful_challenges)
        cloned_env.failed_challenges = copy.deepcopy(self.failed_challenges)
        cloned_env.bluff_counts = copy.deepcopy(self.bluff_counts)
        cloned_env.total_plays = copy.deepcopy(self.total_plays)

        # Opponent histories
        cloned_env.public_opponent_histories = copy.deepcopy(self.public_opponent_histories)
        cloned_env.private_opponent_histories = copy.deepcopy(self.private_opponent_histories)

        # Copy RNG state
        cloned_env.np_random = np.random.default_rng()
        try:
             cloned_env.np_random.bit_generator.state = self.np_random.bit_generator.state
        except Exception as e:
             cloned_env.logger.warning(f"Could not copy RNG state during clone: {e}. Using original seed.")
             # Fallback? Or assume seeding in test setup is enough.

        return cloned_env

    def set_state(self, state_dict):
        """
        Sets the environment state from a state dictionary.
        Correctly handles agent_selector state.

        Args:
            state_dict (dict): Dictionary containing the full environment state.
        """
        # Set core attributes from state_dict using deepcopy for mutable types
        self.deck = copy.deepcopy(state_dict.get('deck', []))
        self.table_card = state_dict['table_card']
        self.table_card_idx = state_dict['table_card_idx']
        self.players_hands = copy.deepcopy(state_dict['players_hands'])
        self.possible_agents = copy.deepcopy(state_dict['possible_agents']) # Restore possible agents
        self.num_players = len(self.possible_agents) # Update num_players based on state
        self.agents = copy.deepcopy(state_dict.get('agents', self.possible_agents[:])) # Use state's agent list
        agent_selection_from_state = state_dict['agent_selection']

        # --- Corrected Agent Selector Handling ---
        # 1. Recreate the selector based on the 'agents' list from the state_dict
        self._agent_selector = agent_selector(self.agents)
        # 2. Reset the selector to get the first agent in the list
        if self.agents:
            self.agent_selection = self._agent_selector.reset()
        else:
            self.agent_selection = None # No active agents
        # 3. Iterate the new selector until it matches the agent selection from the state
        if agent_selection_from_state is not None and self.agents:
            if agent_selection_from_state in self.agents:
                # Cycle through the new selector until it lands on the desired agent
                while self._agent_selector.selected_agent != agent_selection_from_state:
                    self.agent_selection = self._agent_selector.next()
                # Ensure self.agent_selection is set correctly after the loop
                self.agent_selection = self._agent_selector.selected_agent
            else:
                # If the agent from state is not in the active list, the selector
                # starting from the beginning is correct. self.agent_selection is already set.
                pass
        elif agent_selection_from_state is None:
             # If the state indicated no agent was selected, ensure it's None
             self.agent_selection = None
        # --- End Corrected Agent Selector Handling ---

        self.penalties = copy.deepcopy(state_dict['penalties'])
        self.penalty_thresholds = copy.deepcopy(state_dict['penalty_thresholds'])
        self.last_action = state_dict['last_action']
        self.last_action_agent = state_dict['last_action_agent']
        self.last_action_bluff = state_dict['last_action_bluff']
        self.must_force = state_dict.get('must_force', False)
        self.last_played_cards = copy.deepcopy(state_dict['last_played_cards'])
        self.terminations = copy.deepcopy(state_dict['terminations'])
        self.truncations = copy.deepcopy(state_dict['truncations'])
        self.round_eliminated = copy.deepcopy(state_dict['round_eliminated'])
        self.winner = state_dict['winner']
        self.round = state_dict['round']
        self.game_history = copy.deepcopy(state_dict.get('game_history', []))
        self.global_step = state_dict.get('global_step', 0)
        self.rewards = copy.deepcopy(state_dict['rewards'])
        self._cumulative_rewards = copy.deepcopy(state_dict['_cumulative_rewards'])
        self.infos = copy.deepcopy(state_dict['infos'])
        self.pending_bluff = copy.deepcopy(state_dict['pending_bluff'])
        self.last_agent_action = copy.deepcopy(state_dict['last_agent_action'])
        self.consecutive_action_count = copy.deepcopy(state_dict['consecutive_action_count'])

        # Statistics
        self.successful_bluffs = copy.deepcopy(state_dict.get('successful_bluffs', {a: 0 for a in self.possible_agents}))
        self.failed_bluffs = copy.deepcopy(state_dict.get('failed_bluffs', {a: 0 for a in self.possible_agents}))
        self.successful_challenges = copy.deepcopy(state_dict.get('successful_challenges', {a: 0 for a in self.possible_agents}))
        self.failed_challenges = copy.deepcopy(state_dict.get('failed_challenges', {a: 0 for a in self.possible_agents}))
        self.bluff_counts = copy.deepcopy(state_dict.get('bluff_counts', {a: 0 for a in self.possible_agents}))
        self.total_plays = copy.deepcopy(state_dict.get('total_plays', {a: 0 for a in self.possible_agents}))

        # Histories
        self.public_opponent_histories = copy.deepcopy(state_dict.get('public_opponent_histories', {a: [] for a in self.possible_agents}))
        self.private_opponent_histories = copy.deepcopy(state_dict.get('private_opponent_histories', {a: [] for a in self.possible_agents}))

        # Restore RNG state
        rng_state = state_dict.get('random_seed')
        if rng_state is not None:
            try:
                self.np_random = np.random.default_rng()
                if isinstance(rng_state, dict):
                    self.np_random.bit_generator.state = rng_state
                elif isinstance(rng_state, int):
                    self.np_random = np.random.default_rng(rng_state) # Re-seed
                # Add handling for other potential RNG state representations if needed
            except Exception as e:
                 self.logger.warning(f"Could not restore RNG state during set_state: {e}")
                 seed_val = rng_state if isinstance(rng_state, int) else 42
                 self.np_random = np.random.default_rng(seed_val) # Fallback
        else:
             self.np_random = np.random.default_rng(42) # Default if not provided

        # Ensure basic info structure exists for all agents after loading state
        for agent in self.possible_agents:
            if agent not in self.infos:
                self.infos[agent] = {}

    def get_state(self):
        """
        Returns a dictionary containing the full environment state.
        Includes 'possible_agents' key.

        Returns:
            dict: Dictionary containing the complete environment state.
        """
        state = {
            # --- ADD THIS LINE ---
            'possible_agents': self.possible_agents.copy(),
            # ---------------------

            # Core game state
            'deck': self.deck.copy(),
            'table_card': self.table_card,
            'table_card_idx': self.table_card_idx,

            # Players' hands
            'players_hands': {agent: hand.copy() for agent, hand in self.players_hands.items()},

            # Agent state
            'agents': self.agents.copy(), # Current active agents
            'agent_selection': self.agent_selection,

            # Game tracking state
            'penalties': self.penalties.copy(),
            'penalty_thresholds': self.penalty_thresholds.copy(),
            'last_action': self.last_action,
            'last_action_agent': self.last_action_agent,
            'last_action_bluff': self.last_action_bluff,
            'must_force': self.must_force,
            'last_played_cards': {a: c.copy() if c else [] for a, c in self.last_played_cards.items()},

            # Game status flags
            'terminations': self.terminations.copy(),
            'truncations': self.truncations.copy(),
            'round_eliminated': self.round_eliminated.copy(),
            'winner': self.winner,
            'round': self.round,
            'game_history': self.game_history.copy(),
            'global_step': self.global_step,
            'num_players': self.num_players, # Include num_players

            # Rewards and other info
            'rewards': self.rewards.copy(),
            '_cumulative_rewards': self._cumulative_rewards.copy(),
            'infos': copy.deepcopy(self.infos), # Use deepcopy for safety

            # Bluff-related state
            'pending_bluff': copy.deepcopy(self.pending_bluff) if self.pending_bluff else None,

            # Action tracking
            'last_agent_action': self.last_agent_action.copy(),
            'consecutive_action_count': self.consecutive_action_count.copy(),

            # Statistics
            'successful_bluffs': self.successful_bluffs.copy(),
            'failed_bluffs': self.failed_bluffs.copy(),
            'successful_challenges': self.successful_challenges.copy(),
            'failed_challenges': self.failed_challenges.copy(),
            'bluff_counts': self.bluff_counts.copy(),
            'total_plays': self.total_plays.copy(),

            # Opponent histories
            'public_opponent_histories': copy.deepcopy(self.public_opponent_histories),
            'private_opponent_histories': copy.deepcopy(self.private_opponent_histories),

            # Random seed state (important for reproducibility)
            'random_seed': copy.deepcopy(self.np_random.bit_generator.state) # Store full RNG state
        }
        return state

    def render(self, mode='human'):
        if mode not in ['human', 'player']:
            raise ValueError("Invalid render mode. Supported modes are 'human' and 'player'.")

        if mode == 'human':
            print("\n=== Current Game State ===")
            print(f"Round: {self.round}")
            print(f"Table Card: {self.table_card}")
            print("Players' Hands:")
            for agent in self.possible_agents:
                phand = self.players_hands.get(agent, [])
                print(f"  {agent}: {phand}")

            if self.last_action is not None:
                print(f"Last Action: Played {self.last_action} card(s) by {self.last_action_agent}")

            if self.winner:
                print(f"Winner Declared: {self.winner}")
            print("Active Players:")
            for agent in self.possible_agents:
                if self.terminations.get(agent, False):
                    status = "Game-Terminated"
                elif self.round_eliminated.get(agent, False):
                    status = "Round-Eliminated"
                else:
                    status = "Active"
                print(f"  {agent}: {status}")

            print("Penalties and Penalty Thresholds:")
            for agent in self.possible_agents:
                penalty = self.penalties.get(agent, 0)
                threshold = self.penalty_thresholds.get(agent, 3)
                print(f"  {agent}: Penalty = {penalty}, Threshold = {threshold}")

            print("==========================\n")

        elif mode == 'player':
            print("\n=== Your Turn ===")
            current_agent = self.agent_selection
            print(f"Table Card: {self.table_card}")
            print("Your Hand:")
            print(f"  {current_agent}: {self.players_hands.get(current_agent, [])}")

            print("Opponent Hands:")  # Display how many cards each opponent has
            for agent in self.possible_agents:
                if agent != current_agent:
                    print(f"  {agent}: {len(self.players_hands.get(agent, []))}")

            if self.last_action is not None:
                print(f"Last Action: Played {self.last_action} card(s) by {self.last_action_agent}")

            print("Active Players:")
            for agent in self.possible_agents:
                if self.terminations.get(agent, False):
                    status = "Game-Terminated"
                elif self.round_eliminated.get(agent, False):
                    status = "Round-Eliminated"
                else:
                    status = "Active"
                print(f"  {agent}: {status}")

            print("Penalties:")
            for agent in self.possible_agents:
                penalty = self.penalties.get(agent, 0)
                print(f"  {agent}: Penalty = {penalty}")

            print("==========================\n")

    def close(self):
        pass
