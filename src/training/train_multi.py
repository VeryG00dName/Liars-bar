# src/training/train_multi.py

import logging
import time
import os
import random
import numpy as np
import pygame
import torch
from torch.utils.tensorboard import SummaryWriter

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.env.reward_restriction_wrapper import RewardRestrictionWrapper
from src.model.models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor
from src.model.memory import RolloutMemory
from src.training.train_utils import compute_gae
from src.training.train_extras import set_seed
from src.training.train_multi_utils import save_multi_checkpoint, load_multi_checkpoint
from src.training.train import train
from src import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

torch.backends.cudnn.benchmark = True

def configure_logger():
    logger = logging.getLogger('TrainMulti')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

def init_pygame_ui():
    pygame.init()
    screen = pygame.display.set_mode((300, 100))
    pygame.display.set_caption("Training Control")
    button_rect = pygame.Rect(75, 25, 150, 50)
    return screen, button_rect

def draw_button(screen, button_rect, paused):
    screen.fill((30, 30, 30))
    color = (200, 50, 50) if not paused else (50, 200, 50)
    pygame.draw.rect(screen, color, button_rect)
    font = pygame.font.SysFont("Arial", 24)
    text = "Pause" if not paused else "Resume"
    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=button_rect.center)
    screen.blit(text_surface, text_rect)
    pygame.display.flip()

def create_player_pool(total_players, device):
    """
    Create a global pool of players keyed by pool agent names.
    """
    player_pool = {}
    for p in range(total_players):
        key = f"player_{p}"
        policy_net = PolicyNetwork(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            use_lstm=True,
            use_dropout=True,
            use_layer_norm=True
        ).to(device)
        value_net = ValueNetwork(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            use_dropout=True,
            use_layer_norm=True
        ).to(device)
        optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE)
        optimizer_value = torch.optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE)
        memory = RolloutMemory([key])
        player_pool[key] = {
            'policy_net': policy_net,
            'value_net': value_net,
            'optimizer_policy': optimizer_policy,
            'optimizer_value': optimizer_value,
            'memory': memory,
            'entropy_coef': config.INIT_ENTROPY_COEF
        }
    return player_pool

def initialize_obp(device):
    obp_model = OpponentBehaviorPredictor(
        input_dim=config.OPPONENT_INPUT_DIM,
        hidden_dim=config.OPPONENT_HIDDEN_DIM,
        output_dim=2
    ).to(device)
    obp_optimizer = torch.optim.Adam(obp_model.parameters(), lr=config.OPPONENT_LEARNING_RATE)
    return obp_model, obp_optimizer

def main():
    set_seed()
    logger = configure_logger()
    device = torch.device(config.DEVICE)

    # --- Settings ---
    TOTAL_PLAYERS = 9
    GROUP_SIZE = 3
    NUM_BATCH_EPISODES = 50000  # Episodes per group per block.
    NUM_TOTAL_BATCHES = 4
    MIX_INTERVAL = 1  # Groups will be re-mixed every MIX_INTERVAL batches

    temp_env = LiarsDeckEnv(num_players=GROUP_SIZE, render_mode=config.RENDER_MODE)
    agents = temp_env.agents
    config.set_derived_config(
        temp_env.observation_spaces[agents[0]],
        temp_env.action_spaces[agents[0]],
        GROUP_SIZE - 1
    )

    logger.info(f"Derived config: INPUT_DIM={config.INPUT_DIM}, OUTPUT_DIM={config.OUTPUT_DIM}")

    player_pool = create_player_pool(TOTAL_PLAYERS, device)
    obp_model, obp_optimizer = initialize_obp(device)
    obp_memory = []

    writer = SummaryWriter(log_dir=config.MULTI_LOG_DIR)

    # Load checkpoints, including OBP model and optimizer
    start_batch, loaded_entropy_coefs, obp_state, obp_optimizer_state = load_multi_checkpoint(
        player_pool, config.CHECKPOINT_DIR, group_size=GROUP_SIZE
    )

    if loaded_entropy_coefs is not None:
        for agent, coef in loaded_entropy_coefs.items():
            player_pool[agent]['entropy_coef'] = coef

    if obp_state is not None:
        obp_model.load_state_dict(obp_state)
    if obp_optimizer_state is not None:
        obp_optimizer.load_state_dict(obp_optimizer_state)

    screen, button_rect = init_pygame_ui()
    paused = False

    logger.info("Starting multi-group training...")
    all_player_ids = list(player_pool.keys())
    groups = [all_player_ids[i:i + GROUP_SIZE] for i in range(0, len(all_player_ids), GROUP_SIZE)]

    block_episode_offset = 0

    for batch_id in range(start_batch, NUM_TOTAL_BATCHES + 1):
        logger.info(f"\n--- Batch {batch_id} ---")

        for g_idx, group in enumerate(groups):
            logger.info(f"Training group {g_idx}: {group}")

            while paused:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        writer.close()
                        return
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if button_rect.collidepoint(event.pos):
                            paused = not paused
                            draw_button(screen, button_rect, paused)

            env = LiarsDeckEnv(num_players=len(group), render_mode=config.RENDER_MODE)
            agent_map = {env_agent: group[idx] for idx, env_agent in enumerate(env.agents)}

            group_agents = {pool_agent: player_pool[pool_agent] for pool_agent in group}

            train(
                agents_dict=group_agents,
                env=env,
                device=device,
                num_episodes=NUM_BATCH_EPISODES,
                episode_offset=block_episode_offset,
                log_tensorboard=True,
                logger=logger,
                writer=writer,
                agent_mapping=agent_map,
                obp_model=obp_model,
                obp_optimizer=obp_optimizer
            )

        if batch_id % MIX_INTERVAL == 0 and batch_id < NUM_TOTAL_BATCHES:
            random.shuffle(all_player_ids)
            groups = [all_player_ids[i:i + GROUP_SIZE] for i in range(0, len(all_player_ids), GROUP_SIZE)]

        block_episode_offset += NUM_BATCH_EPISODES

        # Save checkpoints, including OBP model and optimizer
        save_multi_checkpoint(
            player_pool=player_pool,
            obp_model=obp_model,
            obp_optimizer=obp_optimizer,
            batch=batch_id,
            checkpoint_dir=config.CHECKPOINT_DIR,
            group_size=GROUP_SIZE
        )
        logger.info(f"Saved multi-checkpoint for batch {batch_id}.")

    writer.close()
    logger.info("Finished multi-group training.")

if __name__ == "__main__":
    main()