# test_liars_deck_env.py

import logging
from src.env.liars_deck_env_core import LiarsDeckEnv

def setup_logging():
    """
    Sets up logging to display information about the environment interactions.
    """
    logger = logging.getLogger('TestLiarsDeckEnv')
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to see all logs
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(ch)
    return logger

def run_controlled_test_episode(env, logger, episode_num, predefined_actions=None):
    """
    Runs a single controlled test episode with predefined actions for agents and logs rewards after each move.
    
    Args:
        env (LiarsDeckEnv): The environment instance.
        logger (logging.Logger): Logger for outputting information.
        episode_num (int): Current episode number.
        predefined_actions (dict, optional): A dictionary mapping agents to a list of actions.
    """
    logger.info(f"=== Starting Controlled Test Episode {episode_num} ===")
    observations, infos = env.reset()
    agents = env.agents
    step_num = 0

    # Initialize action counters for each agent
    action_counters = {agent: 0 for agent in agents}

    while env.agent_selection is not None:
        agent = env.agent_selection

        # Retrieve the last observation and reward for the current agent
        observation, reward, done, truncated, info = env.last()

        if done or truncated:
            # Agent is done; no action needed
            action = None
            logger.info(f"Episode {episode_num}, Step {step_num}: Agent '{agent}' is done.")
        else:
            # Select predefined action if available
            if predefined_actions and agent in predefined_actions:
                agent_actions = predefined_actions[agent]
                if action_counters[agent] < len(agent_actions):
                    action = agent_actions[action_counters[agent]]
                    action_counters[agent] += 1
                    logger.info(f"Episode {episode_num}, Step {step_num}: Agent '{agent}' takes predefined action '{action}'.")
                else:
                    # If no predefined action left, select random
                    action = env.action_space(agent).sample()
                    logger.info(f"Episode {episode_num}, Step {step_num}: Agent '{agent}' takes random action '{action}'.")
            else:
                # For testing purposes, select a random valid action
                action = env.action_space(agent).sample()
                logger.info(f"Episode {episode_num}, Step {step_num}: Agent '{agent}' takes action '{action}'.")

        # Execute the action in the environment
        env.step(action)

        # After env.step(), either another agent is selected or the game ended
        if env.agent_selection is not None:
            # Game continues, log reward for the action-taking agent
            new_observation, new_reward, new_done, new_truncated, new_info = env.last()
            logger.info(f"Episode {episode_num}, Step {step_num}: Agent '{agent}' received reward {new_reward}.")
        else:
            # Game has ended
            if env.winner:
                logger.info(f"Episode {episode_num}, Step {step_num}: Winner is '{env.winner}'.")
            else:
                logger.info(f"Episode {episode_num}, Step {step_num}: No winner declared.")

            # Log final rewards for all agents
            logger.info(f"Episode {episode_num}, Step {step_num}: Final rewards: {env.rewards}")
            logger.info(f"Episode {episode_num}, Step {step_num}: Game has ended.")

        step_num += 1

    logger.info(f"=== Ending Controlled Test Episode {episode_num} ===\n")

def run_random_test_episode(env, logger, episode_num):
    """
    Runs a single random test episode where agents take random actions and logs rewards after each move.
    
    Args:
        env (LiarsDeckEnv): The environment instance.
        logger (logging.Logger): Logger for outputting information.
        episode_num (int): Current episode number.
    """
    logger.info(f"=== Starting Random Test Episode {episode_num} ===")
    observations, infos = env.reset()
    agents = env.agents
    step_num = 0

    while env.agent_selection is not None:
        agent = env.agent_selection
        observation, reward, done, truncated, info = env.last()

        if done or truncated:
            # Agent is done; no action needed
            action = None
            logger.info(f"Episode {episode_num}, Step {step_num}: Agent '{agent}' is done.")
        else:
            # Select a random valid action
            action = env.action_space(agent).sample()
            logger.info(f"Episode {episode_num}, Step {step_num}: Agent '{agent}' takes action '{action}'.")

        # Execute the action in the environment
        env.step(action)

        if env.agent_selection is not None:
            new_observation, new_reward, new_done, new_truncated, new_info = env.last()
            logger.info(f"Episode {episode_num}, Step {step_num}: Agent '{agent}' received reward {new_reward}.")
        else:
            # Game has ended
            if env.winner:
                logger.info(f"Episode {episode_num}, Step {step_num}: Winner is '{env.winner}'.")
            else:
                logger.info(f"Episode {episode_num}, Step {step_num}: No winner declared.")

            # Log final rewards for all agents
            logger.info(f"Episode {episode_num}, Step {step_num}: Final rewards: {env.rewards}")
            logger.info(f"Episode {episode_num}, Step {step_num}: Game has ended.")

        step_num += 1

    logger.info(f"=== Ending Random Test Episode {episode_num} ===\n")

def main():
    """
    Main function to run multiple controlled and random test episodes on the LiarsDeckEnv environment.
    """
    logger = setup_logging()

    # Initialize the environment with human render mode
    num_players = 3
    env = LiarsDeckEnv(num_players=num_players, render_mode='human')

    # Define the number of test episodes
    num_test_episodes = 2

    # Define predefined actions for controlled testing (example actions)
    predefined_actions = {
        'player_0': [0, 1, 2],
        'player_1': [3, 4, 5],
        'player_2': [6, 0, 1]
    }

    logger.info("Starting Controlled LiarsDeckEnv Testing...\n")

    for episode in range(1, num_test_episodes + 1):
        run_controlled_test_episode(env, logger, episode, predefined_actions=predefined_actions)

    # Optionally, run random test episodes
    num_random_test_episodes = 2
    logger.info("Starting Random LiarsDeckEnv Testing...\n")

    for episode in range(1, num_random_test_episodes + 1):
        run_random_test_episode(env, logger, episode)

    env.close()
    logger.info("Testing completed successfully.")

if __name__ == "__main__":
    main()
