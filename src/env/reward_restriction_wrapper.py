import copy
from pettingzoo.utils.env import AECEnv

class RewardRestrictionWrapper(AECEnv):
    """
    A wrapper around LiarsDeckEnv that:
      1) Enforces that the agent must play all non-table cards before playing any table cards.
      2) Once they start playing table cards, they cannot go back to non-table.
      3) They must play as many cards as possible (up to 3) of whichever type they are currently playing.
         - If they have >= 3 of that type, they must play exactly 3.
         - If they have fewer than 3, they must play all of them.
      4) Penalizes agents for taking action 6.
    """

    def __init__(self, env):
        super().__init__()
        self.env = env
        # Copy PettingZoo attributes
        self.metadata = copy.deepcopy(env.metadata)
        self.possible_agents = env.possible_agents
        self.agents = env.agents
        self.observation_spaces = env.observation_spaces
        self.action_spaces = env.action_spaces
        self.violators = {agent: False for agent in self.agents}  # Track who violated
        # Track how many non-table vs table cards each agent has played so far
        # to enforce "non-table first, then table"
        self.non_table_played = {agent: 0 for agent in self.agents}
        self.table_played = {agent: 0 for agent in self.agents}

        # Keep track of the agent's hand *before* step to identify
        # how many cards of each type they had for "play maximum" logic.
        self.prev_hands = {agent: [] for agent in self.agents}

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents

        # Reset trackers
        for agent in self.agents:
            self.non_table_played[agent] = 0
            self.table_played[agent] = 0
            # Copy the starting hand from the underlying environment
            self.prev_hands[agent] = self.env.players_hands[agent][:]  # shallow copy of the list

        return obs, infos

    def observe(self, agent):
        return self.env.observe(agent)

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)
    
    @property
    def num_players(self):
        # Example pass-through
        return self.env.num_players

    @property
    def private_opponent_histories(self):
        """
        Pass-through property to match the underlying environment.
        """
        return self.env.private_opponent_histories
    
    @property
    def total_plays(self):
        return self.env.total_plays
    
    @property
    def bluff_counts(self):
        return self.env.bluff_counts

    @property
    def agent_selection(self):
        return self.env.agent_selection

    @agent_selection.setter
    def agent_selection(self, value):
        self.env.agent_selection = value

    @property
    def terminations(self):
        return self.env.terminations

    @property
    def truncations(self):
        return self.env.truncations

    @property
    def rewards(self):
        # Copy original rewards
        rewards = self.env.rewards.copy()

        # Ensure all violators receive only negative rewards
        for agent in self.agents:
            if self.violators[agent]:
                rewards[agent] = min(rewards[agent], -1.0)  # Ensure negative rewards

        return rewards

    @property
    def infos(self):
        return self.env.infos

    def step(self, action):
        agent = self.env.agent_selection

        # Store old hand before stepping
        old_hand = self.prev_hands[agent][:]  # copy
        old_non_table = sum(1 for c in old_hand if c != self.env.table_card)
        old_table = sum(1 for c in old_hand if c == self.env.table_card)

        # Step the base environment
        self.env.step(action)

        # If agent is done, no post-processing needed
        if self.terminations[agent] or self.truncations[agent]:
            return

        # Identify which cards the agent *actually* played
        played_cards = self.env.last_played_cards.get(agent, [])
        if len(played_cards) == 0:
            # Possibly the action was "Challenge" or "Pass", etc.
            # We won't penalize them if they didn't actually attempt to play cards.
            # But if you want to enforce playing cards, you could set reward to 0 here.
            self._update_prev_hand(agent)
            return

        table_card = self.env.table_card
        n_played_non_table = sum(1 for c in played_cards if c != table_card)
        n_played_table = sum(1 for c in played_cards if c == table_card)

        # Check if they're trying to go back to non-table after table
        backtrack_violation = False
        if n_played_non_table > 0 and self.table_played[agent] > 0:
            backtrack_violation = True

        # Check "must play maximum possible" logic
        max_play_violation = False
        if n_played_non_table > 0:
            # The agent is playing non-table cards
            required_to_play = min(old_non_table, 3)
            if n_played_non_table != required_to_play:
                max_play_violation = True
        else:
            # The agent is playing table cards
            required_to_play = min(old_table, 3)
            if n_played_table != required_to_play:
                max_play_violation = True

        # Initialize flags for violations
        any_violation = backtrack_violation or max_play_violation
        action_violation = False

        # Check for action 6 violation
        if action == 6:
            action_violation = True
            any_violation = True  # Ensure reward is penalized

        # Apply penalties based on violations
        if any_violation:
            # Initialize the reward if not already penalized
            self.violators[agent] = True  # Mark agent as a violator
            self.env.rewards[agent] = self.env.rewards.get(agent, 0.0) - 1.0
            if backtrack_violation or max_play_violation:
                self.env.infos[agent]["pattern_violation"] = True
            if action_violation:
                self.env.infos[agent]["action_violation"] = True
        else:
            # No violation => update the counters
            if n_played_non_table > 0:
                self.non_table_played[agent] += n_played_non_table
            else:
                self.table_played[agent] += n_played_table
            self.env.infos[agent]["pattern_violation"] = False
            self.env.infos[agent]["action_violation"] = False

        # Update self.prev_hands for next time
        self._update_prev_hand(agent)

    def _update_prev_hand(self, agent):
        """
        Store the new hand for the agent so we can properly
        check how many cards they had on the next turn.
        """
        if agent in self.env.players_hands:
            self.prev_hands[agent] = self.env.players_hands[agent][:]

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()
