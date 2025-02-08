# src/env/liars_deck_env_utils.py

import numpy as np
from src.env.liars_deck_env_utils_2 import decode_action, select_cards_to_play, validate_claim

# --- Existing functions ---

def record_action_history(env, agent, action_type, card_category, count, was_challenged=False):
    entry = {
        'action_type': action_type,
        'count': count,
        'was_challenged': was_challenged
    }

    # Keep only basic action tracking
    env.public_opponent_histories[agent].append(entry)
    
    # Maintain history length
    H = 10
    if len(env.public_opponent_histories[agent]) > H:
        env.public_opponent_histories[agent].pop(0)

def apply_challenge(env, challenger_agent, claimant_agent):
    """
    Applies the challenge logic between two agents.
    
    Args:
        env (LiarsDeckEnv): The environment instance.
        challenger_agent (str): The agent initiating the challenge.
        claimant_agent (str): The agent being challenged.
    """
    claimed_cards = env.last_played_cards.get(claimant_agent, [])
    env.logger.debug(f"Applying challenge: {challenger_agent} vs {claimant_agent}, claimed_cards={claimed_cards}")

    # Find the last play entry in the histories
    def find_last_play_entry(hist):
        for entry in reversed(hist):
            if entry['action_type'] == "Play":
                return entry
        return None

    private_last_play = find_last_play_entry(env.private_opponent_histories.get(claimant_agent, []))
    public_last_play = find_last_play_entry(env.public_opponent_histories.get(claimant_agent, []))

    if not claimed_cards:
        # Claimant played no cards; invalid scenario
        env.penalties[claimant_agent] += 1
        env.failed_bluffs[claimant_agent] += 1
        env.successful_challenges[challenger_agent] += 1
        env.rewards[challenger_agent] += env.scoring_params['forced_challenge_success_challenger_reward']
        env.rewards[claimant_agent] += env.scoring_params['forced_challenge_success_claimant_penalty']
        env.logger.info(f"{challenger_agent} challenged {claimant_agent} but no cards were played.")

        if env.penalties[claimant_agent] >= env.penalty_thresholds[claimant_agent]:
            env.terminations[claimant_agent] = True
            env.rewards[claimant_agent] += env.scoring_params['termination_penalty']
            env.logger.info(f"{claimant_agent} has been terminated due to excessive penalties.")

        # Reveal was_bluff (True) in the public history (claimant lied by not playing cards)
        if public_last_play:
            public_last_play['was_bluff'] = True
            env.logger.debug(f"Updated public history for {claimant_agent}: was_bluff=True")

        # --- MEMORY UPDATE: Record that claimant did not play any cards ---
        from src.model.memory import get_opponent_memory
        for observer in env.possible_agents:
            if observer != claimant_agent:
                get_opponent_memory(observer).update(
                    opponent=claimant_agent,
                    response="NoPlay",
                    triggering_action="Challenge",
                    penalties=env.penalties.get(claimant_agent, 0),
                    card_count=len(env.players_hands.get(claimant_agent, []))
                )
        env.start_new_round()
        return

    is_valid = validate_claim(claimed_cards, env.table_card)
    if is_valid:
        # Challenger was wrong
        env.penalties[challenger_agent] += 1
        env.failed_challenges[challenger_agent] += 1
        env.rewards[challenger_agent] += env.scoring_params['challenge_fail_challenger_penalty']
        env.rewards[claimant_agent] += env.scoring_params['challenge_fail_claimant_reward']
        env.logger.info(f"{challenger_agent} failed to challenge {claimant_agent}'s valid play.")

        if env.penalties[challenger_agent] >= env.penalty_thresholds[challenger_agent]:
            env.terminations[challenger_agent] = True
            env.rewards[challenger_agent] += env.scoring_params['termination_penalty']
            env.logger.info(f"{challenger_agent} has been terminated due to excessive penalties.")

        if public_last_play:
            public_last_play['was_bluff'] = False
            env.logger.debug(f"Updated public history for {claimant_agent}: was_bluff=False")
        outcome = "Truthful"
    else:
        # Claimant was bluffing
        env.penalties[claimant_agent] += 1
        env.failed_bluffs[claimant_agent] += 1
        env.successful_challenges[challenger_agent] += 1
        env.rewards[challenger_agent] += env.scoring_params['challenge_success_challenger_reward']
        env.rewards[claimant_agent] += env.scoring_params['challenge_success_claimant_penalty']
        env.logger.info(f"{challenger_agent} successfully challenged {claimant_agent}'s bluff.")

        if env.penalties[claimant_agent] >= env.penalty_thresholds[claimant_agent]:
            env.terminations[claimant_agent] = True
            env.rewards[claimant_agent] += env.scoring_params['termination_penalty']
            env.logger.info(f"{claimant_agent} has been terminated due to excessive penalties.")

        if public_last_play:
            public_last_play['was_bluff'] = True
            env.logger.debug(f"Updated public history for {claimant_agent}: was_bluff=True")
        outcome = "Bluff"

    # --- MEMORY UPDATE: Record the outcome of the challenge for claimant ---
    from src.model.memory import get_opponent_memory
    for observer in env.possible_agents:
        if observer != claimant_agent:
            get_opponent_memory(observer).update(
                opponent=claimant_agent,
                response=outcome,
                triggering_action="Challenge",
                penalties=env.penalties.get(claimant_agent, 0),
                card_count=len(env.players_hands.get(claimant_agent, []))
            )

    env.start_new_round()
    eligible_agents = [ag for ag in env.possible_agents if not env.terminations[ag]]
    if len(eligible_agents) == 1:
        winner = eligible_agents[0]
        env._declare_game_winner(winner)

def apply_action(env, agent, action):
    """
    Applies the given action by the agent to the environment.
    
    Args:
        env (LiarsDeckEnv): The environment instance.
        agent (str): The agent performing the action.
        action (int): The encoded action to perform.
    """
    info = {}
    action_type, card_category, count = decode_action(action)
    env.logger.debug(f"Decoded action: {action_type}, {card_category}, {count}")
    env.current_action_type = action_type
    current_hand = env.players_hands.get(agent, [])

    if action_type == "Play":
        selected_cards = select_cards_to_play(current_hand, card_category, count, env.table_card, env.np_random)
        if selected_cards:
            for card in selected_cards:
                current_hand.remove(card)
            env.last_played_cards[agent] = selected_cards
            env.last_action = len(selected_cards)  # Store the count of cards played
            env.last_action_agent = agent

            # Determine if this play is a bluff
            env.last_action_bluff = not all(c == env.table_card or c == "Joker" for c in selected_cards)

            # Track total plays and bluffs
            env.total_plays[agent] += 1
            if env.last_action_bluff:
                env.bluff_counts[agent] += 1

            # Calculate reward based on the number of cards played
            num_cards_played = len(selected_cards)
            play_reward = env.scoring_params.get('play_reward_per_card', 1.0) * num_cards_played
            env.rewards[agent] += play_reward
            env.logger.debug(f"{agent} played {num_cards_played} card(s). Reward increased by {play_reward}.")

            # Record in public history
            record_action_history(env, agent, "Play", card_category, count, was_challenged=False)

            # ----------------- NEW CODE ADDED BELOW -----------------
            # Record in private history with 'was_bluff'
            private_entry = {
                'action_type': "Play",
                'count': count,
                'was_bluff': env.last_action_bluff
            }
            env.private_opponent_histories[agent].append(private_entry)

            # Maintain history length (e.g., last 10 entries)
            H = 10
            if len(env.private_opponent_histories[agent]) > H:
                env.private_opponent_histories[agent].pop(0)
            # ----------------- END OF NEW CODE ----------------------

            # --- MEMORY UPDATE: After a play action, update all other agents’ memory about this play ---
            from src.model.memory import get_opponent_memory
            for observer in env.possible_agents:
                if observer != agent:
                    # Record that 'agent' played (and whether it was a bluff)
                    get_opponent_memory(observer).update(
                        opponent=agent,
                        response="Play_Bluff" if env.last_action_bluff else "Play_Truthful",
                        triggering_action="Play_" + str(num_cards_played),
                        penalties=env.penalties.get(agent, 0),
                        card_count=len(env.players_hands.get(agent, []))
                    )

            if not current_hand:
                env.logger.debug(f"{agent} emptied their hand. Adding hand emptying bonus.")

                # Add the hand emptying bonus
                hand_empty_bonus = env.scoring_params.get('hand_empty_bonus', 5)
                env.rewards[agent] += hand_empty_bonus

                env.logger.info(f"{agent} received a bonus of {hand_empty_bonus} for emptying their hand.")

                active_agents = env._active_agents_in_round()
                env.logger.debug(f"Active agents after {agent} emptied hand: {active_agents}")

                if len(active_agents) == 2:
                    claimant_agent = agent
                    challenger_agent = [ag for ag in active_agents if ag != claimant_agent][0]
                    env.logger.info(f"Forced challenge triggered by {challenger_agent} against {claimant_agent}")
                    apply_challenge(env, challenger_agent, claimant_agent)
                    if not env.terminations.get(claimant_agent, False):
                        env.round_eliminated[claimant_agent] = True
                        env.logger.debug(f"{claimant_agent} round eliminated after forced challenge resolution.")
                else:
                    env.round_eliminated[agent] = True
                    env.logger.debug(f"{agent} round eliminated (no forced challenge triggered).")
        else:
            env.penalties[agent] += 1
            info["penalty"] = "Invalid Play (No cards selected)"
            env.rewards[agent] += env.scoring_params['invalid_play_penalty']
            env.logger.debug(f"Invalid Play by {agent}: Penalty={env.penalties[agent]}, Reward={env.rewards[agent]}")

    elif action_type == "Challenge":
        record_action_history(env, agent, "Challenge", card_category=None, count=None, was_challenged=True)

        if env.last_action_agent is not None and env.last_played_cards.get(env.last_action_agent, []):
            challenger = agent
            claimant = env.last_action_agent
            env.logger.info(f"{challenger} initiated a challenge against {claimant}")
            apply_challenge(env, challenger, claimant)
        else:
            env.penalties[agent] += 1
            info["penalty"] = "Invalid Challenge (No claim available)"
            env.rewards[agent] += env.scoring_params['invalid_challenge_penalty']
            env.logger.debug(f"Invalid Challenge by {agent}: Penalty={env.penalties[agent]}, Reward={env.rewards[agent]}")
            env._check_round_end()
            env._check_game_end()

    else:
        # Invalid action handling
        env.penalties[agent] += 1
        info["penalty"] = "Invalid action"
        env.rewards[agent] += env.scoring_params['invalid_play_penalty']
        env.logger.debug(f"Invalid Action by {agent}: Penalty={env.penalties[agent]}, Reward={env.rewards[agent]}")

    # Check for termination due to penalties
    if env.penalties[agent] >= env.penalty_thresholds[agent]:
        env.terminations[agent] = True
        env.rewards[agent] += env.scoring_params['termination_penalty']
        env.logger.info(f"{agent} has been terminated due to excessive penalties.")
        env.logger.debug(f"Rewards after termination: {env.rewards}")

    env.infos[agent] = info

def get_opponent_features(env, observing_agent):
    """
    Extracts opponent features for the observing agent.
    
    Args:
        env (LiarsDeckEnv): The environment instance.
        observing_agent (str): The agent observing opponents.
    
    Returns:
        list: A list of opponent feature vectors.
    """
    opponents = [ag for ag in env.possible_agents if ag != observing_agent]
    features = []
    for opp in opponents:
        history = env.public_opponent_histories.get(opp, [])
        last_action = history[-1] if history else None
        
        # Action Type: No-Action=0, Play=1, Challenge=2
        atype_onehot = [0.0, 0.0, 0.0]
        count_val = 0.0
        
        if last_action:
            if last_action['action_type'] == "Play":
                atype_onehot[1] = 1.0
            elif last_action['action_type'] == "Challenge":
                atype_onehot[2] = 1.0
            raw_count = last_action.get('count', 0)
            count_val = float(raw_count if raw_count is not None else 0) / 5.0
        else:
            # No previous action - set No-Action flag
            atype_onehot[0] = 1.0
        
        features.extend(atype_onehot + [count_val])
    
    return features

def get_observations(env, agent_specific=None):
    """
    Generates observations for all agents or a specific agent.
    
    Args:
        env (LiarsDeckEnv): The environment instance.
        agent_specific (str, optional): Specific agent to generate observation for.
    
    Returns:
        dict: A dictionary of observations keyed by agent names.
    """
    observations = {}
    last_action_val = np.array([env.last_action if env.last_action is not None else 0], dtype=np.float32)

    active_players_vector = np.array([
        len(env.players_hands.get(ag, [])) / 5.0
        for ag in env.possible_agents
    ], dtype=np.float32)

    agents_to_observe = [agent_specific] if agent_specific else env.agents
    for agent in agents_to_observe:
        if env.terminations.get(agent, False):
            observations[agent] = np.zeros(env.observation_spaces[agent].shape, dtype=np.float32)
            env.logger.debug(f"{agent} is terminated. Providing zeroed observation.")
            continue

        current_hand = env.players_hands.get(agent, [])
        from src.env.liars_deck_env_utils_2 import encode_hand
        hand_vector = encode_hand(current_hand, env.table_card).astype(np.float32)

        opponent_features = get_opponent_features(env, agent)
        opponent_features = np.array(opponent_features, dtype=np.float32)

        flattened_obs = np.concatenate([
            hand_vector,
            last_action_val,
            active_players_vector,
            opponent_features
        ], dtype=np.float32)

        observations[agent] = flattened_obs
        env.logger.debug(f"Observation for {agent}: Shape={flattened_obs.shape}, Data={flattened_obs}")

    return observations

# New helper function to query persistent opponent memory.
def query_opponent_memory(observer, opponent):
    """
    Returns the persistent summary vector for a given opponent as seen by the observer.
    
    Args:
        observer (str): The observing agent's identifier.
        opponent (str): The opponent's identifier.
    
    Returns:
        np.ndarray: The summary vector from the observer's persistent memory.
                    If no events are recorded, returns a zero vector.
    """
    from src.model.memory import get_opponent_memory
    return get_opponent_memory(observer).get_summary(opponent)

def query_opponent_memory_full(observer, opponent):
    """
    Returns the full memory (all events) for a given opponent as seen by the observer.
    
    Args:
        observer (str): The observing agent's identifier.
        opponent (str): The opponent's identifier.
    
    Returns:
        list: The full list of recorded events from the observer's persistent memory for the opponent.
              If no events are recorded, returns an empty list.
    """
    from src.model.memory import get_opponent_memory
    return get_opponent_memory(observer).get_full_memory(opponent)