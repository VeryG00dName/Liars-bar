# src/training/train_utils.py

import random
import torch
import os
import re
from torch.utils.tensorboard import SummaryWriter
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src import config

AGENT_NAME_MAPPING = {
    'player_0': 'player_0',
    'player_1': 'player_1',
    'player_2': 'player_2'
}

def compute_gae(rewards, dones, values, next_values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for step in reversed(range(len(rewards))):
        if step < len(values) - 1:
            next_val = next_values[step] if step < len(next_values) else 0
            delta = rewards[step] + gamma * next_val * (1 - dones[step]) - values[step]
        else:
            delta = rewards[step] - values[step] if step < len(rewards) else 0
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns

def save_checkpoint(policy_nets, value_nets, optimizers_policy, optimizers_value, obp_model, obp_optimizer, episode, checkpoint_dir=config.CHECKPOINT_DIR, checkpoint_filename=None):
    """
    Saves the current state of the training process, including models and optimizers.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    if checkpoint_filename is None:
        checkpoint_filename = f"checkpoint_episode_{episode}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    checkpoint = {
        'episode': episode,
        'policy_nets': {agent: net.state_dict() for agent, net in policy_nets.items()},
        'value_nets': {agent: net.state_dict() for agent, net in value_nets.items()},
        'optimizers_policy': {agent: opt.state_dict() for agent, opt in optimizers_policy.items()},
        'optimizers_value': {agent: opt.state_dict() for agent, opt in optimizers_value.items()},
        'obp_model': obp_model.state_dict(),
        'obp_optimizer': obp_optimizer.state_dict(),
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint_if_available(policy_nets, value_nets, optimizers_policy, optimizers_value, obp_model, obp_optimizer, checkpoint_dir=config.CHECKPOINT_DIR):
    """
    Loads the latest checkpoint if available and restores the state of models and optimizers.
    """
    if not os.path.isdir(checkpoint_dir):
        logging.info("Checkpoint directory does not exist. Starting training from scratch.")
        entropy_coefs = {agent: config.INIT_ENTROPY_COEF for agent in policy_nets.keys()}
        return 1, entropy_coefs

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_episode_") and f.endswith(".pth")]
    if not checkpoint_files:
        logging.info("No checkpoint files found. Starting training from scratch.")
        entropy_coefs = {agent: config.INIT_ENTROPY_COEF for agent in policy_nets.keys()}
        return 1, entropy_coefs

    episode_numbers = []
    for file in checkpoint_files:
        match = re.search(r'checkpoint_episode_(\d+)\.pth', file)
        if match:
            episode_numbers.append(int(match.group(1)))

    if not episode_numbers:
        logging.info("No valid checkpoint files found. Starting training from scratch.")
        entropy_coefs = {agent: config.INIT_ENTROPY_COEF for agent in policy_nets.keys()}
        return 1, entropy_coefs

    latest_episode = max(episode_numbers)
    latest_checkpoint = os.path.join(checkpoint_dir, f"checkpoint_episode_{latest_episode}.pth")
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')

    for agent, net in policy_nets.items():
        mapped_agent = AGENT_NAME_MAPPING.get(agent, agent)
        if mapped_agent in checkpoint['policy_nets']:
            try:
                missing_keys, unexpected_keys = net.load_state_dict(checkpoint['policy_nets'][mapped_agent], strict=False)
                if missing_keys:
                    logging.warning(f"Policy Network for {agent} is missing keys: {missing_keys}")
                if unexpected_keys:
                    logging.warning(f"Policy Network for {agent} has unexpected keys: {unexpected_keys}")
                net.to(next(net.parameters()).device)
                logging.info(f"Loaded Policy Network for {agent} from episode {latest_episode} with strict=False.")
            except RuntimeError as e:
                logging.error(f"Error loading Policy Network for {agent}: {e}")
                logging.info(f"Skipping loading Policy Network for {agent}. Initializing randomly.")
        else:
            logging.warning(f"Policy Network for {agent} not found in checkpoint. Skipping.")

    for agent, net in value_nets.items():
        mapped_agent = AGENT_NAME_MAPPING.get(agent, agent)
        if mapped_agent in checkpoint['value_nets']:
            try:
                missing_keys, unexpected_keys = net.load_state_dict(checkpoint['value_nets'][mapped_agent], strict=False)
                if missing_keys:
                    logging.warning(f"Value Network for {agent} is missing keys: {missing_keys}")
                if unexpected_keys:
                    logging.warning(f"Value Network for {agent} has unexpected keys: {unexpected_keys}")
                net.to(next(net.parameters()).device)
                logging.info(f"Loaded Value Network for {agent} from episode {latest_episode} with strict=False.")
            except RuntimeError as e:
                logging.error(f"Error loading Value Network for {agent}: {e}")
                logging.info(f"Skipping loading Value Network for {agent}. Initializing randomly.")
        else:
            logging.warning(f"Value Network for {agent} not found in checkpoint. Skipping.")

    for agent, opt in optimizers_policy.items():
        mapped_agent = AGENT_NAME_MAPPING.get(agent, agent)
        if mapped_agent in checkpoint['optimizers_policy']:
            try:
                opt.load_state_dict(checkpoint['optimizers_policy'][mapped_agent])
                logging.info(f"Loaded Policy Optimizer for {agent} from episode {latest_episode}.")
            except RuntimeError as e:
                logging.error(f"Error loading Policy Optimizer for {agent}: {e}")
                logging.info(f"Skipping loading Policy Optimizer for {agent}.")
        else:
            logging.warning(f"Policy Optimizer for {agent} not found in checkpoint. Skipping.")

    for agent, opt in optimizers_value.items():
        mapped_agent = AGENT_NAME_MAPPING.get(agent, agent)
        if mapped_agent in checkpoint['optimizers_value']:
            try:
                opt.load_state_dict(checkpoint['optimizers_value'][mapped_agent])
                logging.info(f"Loaded Value Optimizer for {agent} from episode {latest_episode}.")
            except RuntimeError as e:
                logging.error(f"Error loading Value Optimizer for {agent}: {e}")
                logging.info(f"Skipping loading Value Optimizer for {agent}.")
        else:
            logging.warning(f"Value Optimizer for {agent} not found in checkpoint. Skipping.")

    if 'obp_model' in checkpoint and 'obp_optimizer' in checkpoint:
        try:
            obp_model.load_state_dict(checkpoint['obp_model'])
            obp_model.to(next(obp_model.parameters()).device)
            obp_optimizer.load_state_dict(checkpoint['obp_optimizer'])
            logging.info(f"Loaded Opponent Behavior Predictor model and optimizer from episode {latest_episode}.")
        except RuntimeError as e:
            logging.error(f"Error loading Opponent Behavior Predictor: {e}")
            logging.info("Skipping loading Opponent Behavior Predictor. Initializing randomly.")
    else:
        logging.warning("Opponent Behavior Predictor model and/or optimizer not found in checkpoint. Skipping.")

    if 'entropy_coefs' in checkpoint:
        logging.info("Ignoring entropy coefficients found in checkpoint.")

    entropy_coefs = {agent: config.INIT_ENTROPY_COEF for agent in policy_nets.keys()}
    logging.info(f"Resuming training from episode {latest_episode + 1}.")
    return latest_episode + 1, entropy_coefs

def get_tensorboard_writer(log_dir=config.TENSORBOARD_RUNS_DIR):
    return SummaryWriter(log_dir=log_dir)

def train_obp(obp_model, obp_optimizer, obp_memory, device, logger):
    """
    Train OBP on training samples.
    Expect each sample to be a triplet: (features, memory_embedding, label)
    """
    if len(obp_memory) <= 100:
        logger.info(f"Insufficient OBP training samples ({len(obp_memory)}). Skipping training.")
        return None, None

    all_features = [f for (f, m, l) in obp_memory]
    all_memories = [m for (f, m, l) in obp_memory]
    all_labels = [l for (f, m, l) in obp_memory]

    features_tensor = torch.tensor(np.array(all_features, dtype=np.float32)).to(device)
    memories_tensor = torch.tensor(np.array(all_memories, dtype=np.float32)).to(device)
    labels_tensor = torch.tensor(np.array(all_labels, dtype=np.int64)).to(device)

    perm = torch.randperm(features_tensor.size(0))
    features_tensor = features_tensor[perm]
    memories_tensor = memories_tensor[perm]
    labels_tensor = labels_tensor[perm]

    obp_batch_size = 64
    obp_epochs = 5

    dataset = TensorDataset(features_tensor, memories_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=obp_batch_size, shuffle=True)

    obp_model.train()
    total_loss_obp = 0.0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(obp_epochs):
        epoch_loss = 0.0
        for batch_features, batch_memories, batch_labels in dataloader:
            obp_optimizer.zero_grad()
            logits = obp_model(batch_features, batch_memories)
            loss_obp = criterion(logits, batch_labels)
            loss_obp.backward()
            obp_optimizer.step()
            epoch_loss += loss_obp.item()

            predictions = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)
        avg_epoch_loss = epoch_loss / len(dataloader)
        total_loss_obp += avg_epoch_loss
    avg_loss_obp = total_loss_obp / obp_epochs
    accuracy = correct / total if total > 0 else 0.0

    logger.debug(f"OBP trained on {len(obp_memory)} samples, Avg Loss: {avg_loss_obp:.4f}, Accuracy: {accuracy * 100:.2f}%")

    return avg_loss_obp, accuracy

def configure_logger():
    """
    Configures and returns a logger for training.
    """
    logger = logging.getLogger('Train')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

def load_specific_historical_models(players_dir, device):
    """
    Loads specific historical player models based on predefined versions.
    - player_1 from Version_E
    - player_0 from Version_B
    - player_0 from Version_A
    """
    required_models = {
        "Version_E": "player_1",
        "Version_B": "player_0",
        "Version_A": "player_0",
    }

    historical_models = []
    logger = logging.getLogger("Train.Historical")

    for version, player_name in required_models.items():
        version_path = os.path.join(players_dir, version)
        if os.path.isdir(version_path):
            checkpoint_files = [f for f in os.listdir(version_path) if f.endswith(".pth")]
            for checkpoint_file in checkpoint_files:
                checkpoint_path = os.path.join(version_path, checkpoint_file)
                try:
                    from src.eval.evaluate_utils import load_combined_checkpoint
                    # Import the helper function to determine the hidden dimension
                    from src.eval.evaluate_utils import get_hidden_dim_from_state_dict
                    checkpoint = load_combined_checkpoint(checkpoint_path, device)
                    policy_nets = checkpoint['policy_nets']

                    if player_name in policy_nets:
                        policy_state_dict = policy_nets[player_name]
                        # Get the input dimension from the existing state dict
                        actual_input_dim = policy_state_dict['fc1.weight'].shape[1]
                        # Use the helper to determine the hidden dimension
                        actual_hidden_dim = get_hidden_dim_from_state_dict(policy_state_dict, layer_prefix='fc1')
                        from src.model.new_models import PolicyNetwork
                        hist_policy = PolicyNetwork(
                            input_dim=actual_input_dim,
                            hidden_dim=actual_hidden_dim,
                            output_dim=config.OUTPUT_DIM,
                            use_lstm=True,
                            use_dropout=True,
                            use_layer_norm=True,
                            use_aux_classifier=False,
                            num_opponent_classes=config.NUM_OPPONENT_CLASSES
                        ).to(device)
                        hist_policy.load_state_dict(policy_state_dict, strict=False)
                        hist_policy.eval()
                        hist_policy.is_historical = True
                        identifier = f"{version}_{player_name}"
                        historical_models.append((hist_policy, identifier))
                        logger.debug(f"Loaded {player_name} from {version} ({checkpoint_path})")
                        break
                except Exception as e:
                    logger.error(f"Error loading {checkpoint_path}: {str(e)}")
        else:
            logger.warning(f"Version {version} not found in {players_dir}")

    return historical_models

def select_injected_bot(agent, injected_bots, win_stats, match_stats):
    """
    Selects an injected bot for the given agent based on win rates.
    Weight is computed as 1 - win_rate (default win_rate = 0.5 if no data).
    """
    weights = []
    for candidate in injected_bots:
        bot_type, bot_data = candidate
        if bot_type == "historical":
            opponent_key = bot_data[1]
        else:
            opponent_key = bot_data.__name__
        
        matches = match_stats[agent].get(opponent_key, 0)
        wins = win_stats[agent].get(opponent_key, 0)
        if isinstance(wins, int):
            win_rate = (wins / matches) if matches > 0 else 0.5
        else:
            win_rate = (sum(wins) / matches) if matches > 0 else 0.5
        
        # Ensure win_rate is in [0, 1] so that weight = 1 - win_rate is non-negative.
        win_rate = min(max(win_rate, 0), 1)
        weight = 1 - win_rate
        weights.append(weight)
    
    total_weight = sum(weights)
    if total_weight == 0:
        return random.choice(injected_bots)
    
    normalized = [w / total_weight for w in weights]
    
    # The normalized list should be non-negative and sum to 1.
    index = np.random.choice(len(injected_bots), p=normalized)
    return injected_bots[index]
