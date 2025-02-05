#!/usr/bin/env python
"""
Script: run_train_eval_loop.py

This script runs training until a target total episode count is reached,
then runs an evaluation tournament. After evaluation, it checks whether the best agent
came from the newest checkpoint. If not, the newest checkpoint is deleted before continuing training.
"""

import os
import re
import logging
import time
import torch

from src import config
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.env.reward_restriction_wrapper_2 import RewardRestrictionWrapper2
from src.training.train_vs_hardcoded import train_agents
from src.evaluation.evaluate_tournament import (
    initialize_players,
    run_group_swiss_tournament,
)

# Configure global logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TrainEvalLoop")


def create_training_env():
    """
    Creates and returns the training environment.
    Uses the reward restriction wrapper if configured.
    """
    base_env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=config.RENDER_MODE)
    if config.USE_WRAPPER:
        env = RewardRestrictionWrapper2(base_env)
    else:
        env = base_env
    return env


def create_evaluation_env():
    """
    Creates and returns an evaluation environment.
    """
    env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=None)
    # Reset the environment to initialize agents and spaces.
    obs, infos = env.reset()
    agents = env.agents
    config.set_derived_config(
        env.observation_spaces[agents[0]], env.action_spaces[agents[0]], config.NUM_PLAYERS - 1
    )
    return env


def run_training_phase(target_total_episodes):
    """
    Runs a training phase until the total episode count reaches target_total_episodes.
    The train_agents routine loads the checkpoint and resumes training until the target is reached.
    """
    logger.info(f"Starting training phase until a total of {target_total_episodes} episodes...")
    env = create_training_env()
    device = torch.device(config.DEVICE)
    training_results = train_agents(
        env=env,
        device=device,
        num_episodes=target_total_episodes,
        load_checkpoint=True,
        log_tensorboard=False  # Disable TensorBoard logging if desired
    )
    logger.info("Training phase complete.")
    return training_results


def run_evaluation_phase(num_games_per_match=5, num_rounds=3):
    """
    Runs the Swiss tournament evaluation and returns the players dict.
    """
    logger.info("Starting evaluation phase...")
    device = torch.device(config.DEVICE)
    env_eval = create_evaluation_env()
    # Load players from checkpoint files.
    players = initialize_players(config.CHECKPOINT_DIR, device)
    if len(players) < 3:
        raise ValueError("Need at least 3 individual players for the tournament.")
    
    # Run the group Swiss tournament.
    run_group_swiss_tournament(
        env=env_eval,
        device=device,
        players=players,
        num_games_per_match=num_games_per_match,
        NUM_ROUNDS=num_rounds
    )
    logger.info("Evaluation phase complete.")
    return players


def parse_episode_number(filename):
    """
    Extracts the episode number from a checkpoint filename.
    Expected format: checkpoint_episode_x.pth where x is an integer.
    Returns the integer episode number, or None if parsing fails.
    """
    match = re.search(r"checkpoint_episode_(\d+)\.pth", filename)
    if match:
        return int(match.group(1))
    return None


def get_newest_checkpoint_episode():
    """
    Returns the highest episode number among the checkpoint files in config.CHECKPOINT_DIR.
    """
    checkpoint_files = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.endswith(".pth")]
    newest_episode = -1
    for fname in checkpoint_files:
        ep = parse_episode_number(fname)
        if ep is not None and ep > newest_episode:
            newest_episode = ep
    return newest_episode if newest_episode >= 0 else None


def delete_newest_checkpoint():
    """
    Finds the checkpoint file with the highest episode number in config.CHECKPOINT_DIR and deletes it.
    """
    checkpoint_files = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.endswith(".pth")]
    newest_file = None
    highest_episode = -1

    for fname in checkpoint_files:
        episode = parse_episode_number(fname)
        if episode is not None and episode > highest_episode:
            highest_episode = episode
            newest_file = fname

    if newest_file is None:
        logger.info("No valid checkpoint files found to delete.")
        return

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, newest_file)
    try:
        os.remove(checkpoint_path)
        logger.info(f"Deleted newest checkpoint: {newest_file} (episode {highest_episode})")
    except Exception as e:
        logger.error(f"Failed to delete checkpoint {newest_file}: {e}")


def main():
    """
    Main loop: run training until target_total_episodes, then evaluation,
    then decide whether to delete the newest checkpoint. Increase the target
    for the next training phase and continue.
    """

    iteration = 1
    training_increment = 5000
    newest_checkpoint = get_newest_checkpoint_episode()
    target_total_episodes = newest_checkpoint + training_increment

    while True:
        logger.info(f"\n{'='*40}\nIteration {iteration}: Starting training-evaluation cycle.\n{'='*40}")
        # 1. Run training phase until target_total_episodes.
        run_training_phase(target_total_episodes=target_total_episodes)
        
        # Pause briefly to ensure checkpoint files are flushed.
        time.sleep(2)
        
        # 2. Run evaluation phase.
        players = run_evaluation_phase(num_games_per_match=5, num_rounds=3)
        
        # 3. Get the newest checkpoint episode.
        newest_episode = get_newest_checkpoint_episode()
        if newest_episode is None:
            logger.warning("No checkpoints found. Skipping deletion step.")
        else:
            logger.info(f"Newest checkpoint episode is: {newest_episode}")
        
        # 4. Determine the best agent's checkpoint episode from evaluation.
        sorted_players = sorted(players.items(), key=lambda x: x[1]['score'], reverse=True)
        best_player_id, best_player_data = sorted_players[0]
        best_cp_filename = best_player_id.split("_player_")[0]  # Extract the checkpoint prefix
        best_episode = parse_episode_number(best_cp_filename)
        if best_episode is None:
            logger.warning(f"Could not parse episode number from best agent checkpoint: {best_cp_filename}")
        else:
            logger.info(f"Best agent's checkpoint episode is: {best_episode}")
        
        # 5. If the best agent's checkpoint is not the newest checkpoint, delete the newest checkpoint.
        if newest_episode is not None and best_episode is not None:
            if best_episode != newest_episode:
                logger.info("Best agent is NOT from the newest checkpoint. Deleting the newest checkpoint.")
                delete_newest_checkpoint()
            else:
                logger.info("Best agent is from the newest checkpoint. No deletion will occur.")
                target_total_episodes += training_increment
        iteration += 1
        logger.info("Starting next training phase...\n")


if __name__ == "__main__":
    main()
