import copy
from pettingzoo.utils.env import AECEnv

class RewardRestrictionWrapper2(AECEnv):
    """
    A wrapper around LiarsDeckEnv that enforces a specific strategy:
    1) Play as many table cards as possible (up to 3) each turn until none remain.
    2) Then play exactly 1 non-table card per turn.
    3) If only one non-table card is left, challenge (action 6) instead of playing it.
    Penalizes agents for violating this strategy.
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

        # Track phase and must_challenge for each agent
        self.phase = {agent: 'table' for agent in self.agents}
        self.must_challenge = {agent: False for agent in self.agents}

        # Track previous hand to compute played cards
        self.prev_hands = {agent: [] for agent in self.agents}

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents

        # Reset trackers
        for agent in self.agents:
            self.phase[agent] = 'table'
            self.must_challenge[agent] = False
            self.prev_hands[agent] = self.env.players_hands[agent][:]  # shallow copy

        return obs, infos

    def observe(self, agent):
        return self.env.observe(agent)

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)
    
    @property
    def num_players(self):
        return self.env.num_players

    @property
    def private_opponent_histories(self):
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
        return self.env.rewards

    @property
    def infos(self):
        return self.env.infos

    def step(self, action):
        agent = self.env.agent_selection

        # Store previous hand
        prev_hand = self.prev_hands[agent][:]
        table_card = self.env.table_card
        prev_table = sum(1 for c in prev_hand if c == table_card or c == 'Joker')
        prev_non_table = sum(1 for c in prev_hand if c != table_card and c != 'Joker')

        # Step the base environment
        self.env.step(action)

        # If agent is terminated/truncated, skip checks
        if self.env.terminations.get(agent, False) or self.env.truncations.get(agent, False):
            self._update_prev_hand(agent)
            return

        # Get current hand and played cards
        current_hand = self.env.players_hands.get(agent, [])
        played_cards = self.env.last_played_cards.get(agent, [])
        played_table = sum(1 for c in played_cards if c == table_card or c == 'Joker')
        played_non_table = sum(1 for c in played_cards if c != table_card and c != 'Joker')
        current_table = sum(1 for c in current_hand if c == table_card or c == 'Joker')
        current_non_table = sum(1 for c in current_hand if c != table_card and c != 'Joker')

        violation = False
        reason = ""

        # Check if must_challenge was required from previous step
        if self.must_challenge.get(agent, False):
            if action != 6:
                violation = True
                reason = "Must challenge when one non-table card remains but did not."
            self.must_challenge[agent] = False  # Reset regardless of action

        # Check phase-based violations
        current_phase = self.phase[agent]
        if current_phase == 'table':
            expected_table = min(3, prev_table)
            if played_non_table > 0:
                violation = True
                reason = "Played non-table cards during table phase."
            elif played_table != expected_table:
                violation = True
                reason = f"Played {played_table} table cards (expected {expected_table})."
        elif current_phase == 'non-table':
            if played_table > 0:
                violation = True
                reason = "Played table cards during non-table phase."
            elif played_non_table != 1:
                violation = True
                reason = f"Played {played_non_table} non-table cards (expected 1)."

        # Check if challenged when not required
        if action == 6 and not self.must_challenge.get(agent, False):
            violation = True
            reason = "Challenged when not required."

        # Update phase based on current table cards
        if current_phase == 'table' and current_table == 0:
            self.phase[agent] = 'non-table'

        # Update must_challenge for next step
        self.must_challenge[agent] = (self.phase[agent] == 'non-table' and current_non_table == 1)

        # Apply penalties if any violation
        if violation:
            self.env.rewards[agent] = self.env.rewards.get(agent, 0.0) - 1.0
            self.env.infos[agent]["pattern_violation"] = True
            self.env.infos[agent]["violation_reason"] = reason
        else:
            self.env.infos[agent]["pattern_violation"] = False

        # Update previous hand for next step
        self._update_prev_hand(agent)

    def _update_prev_hand(self, agent):
        if agent in self.env.players_hands:
            self.prev_hands[agent] = self.env.players_hands[agent][:]

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()