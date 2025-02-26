import logging
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
# Suppress PyTorch warnings.
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")
warnings.filterwarnings("ignore", category=FutureWarning)
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # For cosine similarity and loss functions.
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque  # For moving average win rate tracking

# Environment & model imports
from src.env.reward_restriction_wrapper_2 import RewardRestrictionWrapper2
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.new_models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor, StrategyTransformer
from src.model.memory import RolloutMemory
from src.env.reward_restriction_wrapper import RewardRestrictionWrapper
from src import config

# Import our hard-coded agent classes
from src.model.hard_coded_agents import (
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    StrategicChallenger,
    SelectiveTableConservativeChallenger,
    RandomAgent,
    TableNonTableAgent,
    Classic
)

# Import query_opponent_memory for opponent memory integration
from src.env.liars_deck_env_utils import query_opponent_memory_full

torch.backends.cudnn.benchmark = False

# Imports from our refactored files
from src.training.train_utils import (
    compute_gae,
    save_checkpoint,
    load_checkpoint_if_available,
    get_tensorboard_writer,
    train_obp,
    load_specific_historical_models,
    select_injected_bot,
    configure_logger
)
from src.training.train_extras import (
    set_seed,
    extract_obp_training_data,
    run_obp_inference,
    convert_memory_to_features
)

# ---- Import EventEncoder (used to project raw opponent memory features) ----
from src.training.train_transformer import EventEncoder

# ---------------------------
# Define a mapping from hard-coded agent class names to integer labels.
# (Make sure config.NUM_OPPONENT_CLASSES equals number of hard-coded classes + number of historical models.)
HARD_CODED_LABELS = {
    "GreedyCardSpammer": 0,
    "StrategicChallenger": 1,
    "TableNonTableAgent": 2,
    "Classic": 3,
    "TableFirstConservativeChallenger": 4,
    "SelectiveTableConservativeChallenger": 5,
}
# The historical models will be assigned distinct labels.
historical_label_mapping = {}

# ---------------------------
# Instantiate the device.
# ---------------------------
device = torch.device(config.DEVICE)

# ---------------------------
# Instantiate the Strategy Transformer.
# ---------------------------
strategy_transformer = StrategyTransformer(
    num_tokens=config.STRATEGY_NUM_TOKENS,
    token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM,
    nhead=config.STRATEGY_NHEAD,
    num_layers=config.STRATEGY_NUM_LAYERS,
    strategy_dim=config.STRATEGY_DIM,
    num_classes=config.STRATEGY_NUM_CLASSES,  # Classification head removed below.
    dropout=config.STRATEGY_DROPOUT,
    use_cls_token=True
).to(device)

# ---------------------------
# Load the transformer checkpoint (and related mappings).
# ---------------------------
transformer_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth")
if os.path.exists(transformer_checkpoint_path):
    checkpoint = torch.load(transformer_checkpoint_path, map_location=device)
    strategy_transformer.load_state_dict(checkpoint["transformer_state_dict"], strict=False)
    print(f"Loaded transformer from {transformer_checkpoint_path}")
    if "response2idx" in checkpoint and "action2idx" in checkpoint:
        response2idx = checkpoint["response2idx"]
        action2idx = checkpoint["action2idx"]
        print("Loaded response and action mappings from checkpoint.")
    else:
        raise ValueError("Checkpoint is missing response2idx and/or action2idx.")
    if "label_mapping" in checkpoint:
        label_mapping = checkpoint["label_mapping"]
        label2idx = label_mapping["label2idx"]
        idx2label = label_mapping["idx2label"]
        print("Loaded label mapping from checkpoint.")
    event_encoder = EventEncoder(
        response_vocab_size=len(response2idx),
        action_vocab_size=len(action2idx),
        token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM
    ).to(device)
    event_encoder.load_state_dict(checkpoint["event_encoder_state_dict"])
else:
    raise FileNotFoundError(f"Transformer checkpoint not found at {transformer_checkpoint_path}")

# Override the transformer's token embedding and remove its classification head.
strategy_transformer.token_embedding = nn.Identity()
strategy_transformer.classification_head = None
strategy_transformer.eval()

historical_models = load_specific_historical_models(config.HISTORICAL_MODEL_DIR, device)
print(f"Loaded {len(historical_models)} historical PPO models: {', '.join([id for _, id in historical_models])}")

# Build a mapping from historical model identifier to unique label.
for idx, (_, identifier) in enumerate(historical_models):
    historical_label_mapping[identifier] = len(HARD_CODED_LABELS) + idx

# ---------------------------
# Main training loop.
# ---------------------------
def train_agents(env, device, num_episodes=1000, load_checkpoint=True, load_directory=None, log_tensorboard=True):
    set_seed(config.SEED)
    obs, infos = env.reset(seed=config.SEED)
    agents = env.agents
    assert len(agents) == config.NUM_PLAYERS, f"Expected {config.NUM_PLAYERS} agents, but got {len(agents)} agents."
    num_opponents = config.NUM_PLAYERS - 1
    config.set_derived_config(env.observation_spaces[agents[0]], env.action_spaces[agents[0]], num_opponents)

    # Initialize moving average win tracking:
    # For each agent (the learning agents) and each opponent type, we keep a deque (window=100) of binary win outcomes.
    win_history = {agent: {} for agent in agents}

    # New dictionary to count games played over a moving window of 100 episodes.
    games_played_counter = {agent: {} for agent in agents}

    # Initialize networks, optimizers, and memories for each agent.
    policy_nets = {}
    value_nets = {}
    optimizers_policy = {}
    optimizers_value = {}
    memories = {}

    for agent in agents:
        policy_net = PolicyNetwork(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            use_lstm=True,
            use_dropout=True,
            use_layer_norm=True,
            use_aux_classifier=True,
            num_opponent_classes=config.NUM_OPPONENT_CLASSES
        ).to(device)
        value_net = ValueNetwork(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            use_dropout=True,
            use_layer_norm=True
        ).to(device)
        policy_nets[agent] = policy_net
        value_nets[agent] = value_net
        optimizers_policy[agent] = optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE)
        optimizers_value[agent] = optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE)
        memories[agent] = RolloutMemory([agent])

    # Initialize Opponent Behavior Predictor (OBP)
    obp_model = OpponentBehaviorPredictor(
        input_dim=config.OPPONENT_INPUT_DIM, 
        hidden_dim=config.OPPONENT_HIDDEN_DIM, 
        output_dim=2,
        memory_dim=config.STRATEGY_DIM
    ).to(device)
    obp_optimizer = optim.Adam(obp_model.parameters(), lr=config.OPPONENT_LEARNING_RATE)
    obp_memory = []

    obp_model.eval()  # Disable dropout, batch norm randomness
    example_observation = torch.randn(1, config.OPPONENT_INPUT_DIM).to(device)
    example_memory_embedding = torch.randn(1, config.STRATEGY_DIM).to(device)

    obp_model = torch.jit.trace(obp_model, (example_observation, example_memory_embedding))
    obp_model.train(True)

    logger = logging.getLogger('Train')
    writer = get_tensorboard_writer(log_dir=config.TENSORBOARD_RUNS_DIR) if log_tensorboard else None
    checkpoint_dir = load_directory if load_directory is not None else config.CHECKPOINT_DIR

    if load_checkpoint:
        checkpoint_data = load_checkpoint_if_available(
            policy_nets,
            value_nets,
            optimizers_policy,
            optimizers_value,
            obp_model,
            obp_optimizer,
            checkpoint_dir=checkpoint_dir
        )
        if checkpoint_data is not None:
            start_episode, _ = checkpoint_data
            logger.info(f"Loaded checkpoint from episode {start_episode}")
        else:
            start_episode = 1
    else:
        start_episode = 1

    static_entropy_coef = config.INIT_ENTROPY_COEF
    last_log_time = time.time()
    steps_since_log = 0
    episodes_since_log = 0

    invalid_action_counts_periodic = {agent: 0 for agent in agents}
    action_counts_periodic = {agent: {action: 0 for action in range(config.OUTPUT_DIM)} for agent in agents}
    recent_rewards = {agent: [] for agent in agents}
    original_agent_order = list(env.agents)

    # Combine hardcoded agents and historical models for injected bots.
    hardcoded_agent_classes = [GreedyCardSpammer, StrategicChallenger, TableNonTableAgent, Classic,
                               TableFirstConservativeChallenger, SelectiveTableConservativeChallenger]
    injected_bots = []
    for cls in hardcoded_agent_classes:
        injected_bots.append(("hardcoded", cls))
    for hist_model, identifier in historical_models:
        injected_bots.append(("historical", (hist_model, identifier)))

    # Variables to hold the current injected bot.
    current_injected_agent_id = None
    current_injected_agent_instance = None
    current_injected_bot_type = None  # "hardcoded" or "historical"
    current_injected_bot_identifier = None  # For historical bots, track identifier

    global_step = 0
    tracked_agent = None  
    tracked_agent_last_embeddings = {}

    for episode in range(start_episode, num_episodes + 1):
        env_seed = config.SEED + episode
        obs, infos = env.reset(seed=env_seed)
        agents = env.agents
        pending_rewards = {agent: 0.0 for agent in agents}

        # Every 5 episodes, choose a new injected bot for one randomly selected agent.
        if (episode - start_episode) % 5 == 0:
            # Choose a learning agent at random.
            current_injected_agent_id = random.choice(agents)
            tracked_agent = current_injected_agent_id
            # Instead of a uniform random selection, select based on win rates.
            selected_bot = select_injected_bot(current_injected_agent_id, injected_bots, win_history, games_played_counter)
            current_injected_bot_type = selected_bot[0]
            if current_injected_bot_type == "hardcoded":
                bot_class = selected_bot[1]
                if bot_class == StrategicChallenger:
                    current_injected_agent_instance = bot_class(
                        agent_name=current_injected_agent_id, 
                        num_players=config.NUM_PLAYERS, 
                        agent_index=agents.index(current_injected_agent_id)
                    )
                else:
                    current_injected_agent_instance = bot_class(agent_name=current_injected_agent_id)
                current_injected_bot_identifier = bot_class.__name__
            else:
                current_injected_agent_instance, current_injected_bot_identifier = selected_bot[1]
            #logger.info(f"Episode {episode}: Injecting {current_injected_bot_type} bot ({current_injected_bot_identifier}) for agent {current_injected_agent_id}")

        episode_rewards = {agent: 0 for agent in agents}
        steps_in_episode = 0

        while env.agent_selection is not None:
            steps_in_episode += 1
            global_step += 1
            agent = env.agent_selection

            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue

            observation_dict = env.observe(agent)
            observation = observation_dict[agent]
            action_mask = env.infos[agent]['action_mask']

            # Integrate OBP memory (or use zeros for the injected bot).
            if agent == current_injected_agent_id:
                obp_probs = run_obp_inference(
                    obp_model, observation, device, env.num_players,
                    memory_embeddings=[torch.zeros(1, config.STRATEGY_DIM, device=device)
                                       for _ in range(env.num_players - 1)]
                )
            else:
                embeddings_list = []
                for opp in env.possible_agents:
                    if opp != agent:
                        mem_summary = query_opponent_memory_full(agent, opp)
                        features_list = convert_memory_to_features(mem_summary, response2idx, action2idx)
                        if features_list:
                            feature_tensor = torch.tensor(features_list, dtype=torch.float32, device=device).unsqueeze(0)
                            with torch.no_grad():
                                projected = event_encoder(feature_tensor)
                                strategy_embedding, _ = strategy_transformer(projected)
                        else:
                            strategy_embedding = None
                        if strategy_embedding is not None:
                            embeddings_list.append(strategy_embedding.cpu().detach().numpy().flatten())
                            if tracked_agent is not None and opp == tracked_agent:
                                if agent in tracked_agent_last_embeddings:
                                    prev_emb = tracked_agent_last_embeddings[agent]
                                    similarity = F.cosine_similarity(strategy_embedding, prev_emb, dim=1).item()
                                    if writer is not None:
                                        writer.add_scalar(f"MemorySimilarity/{agent}/{tracked_agent}", similarity, global_step)
                                tracked_agent_last_embeddings[agent] = strategy_embedding.detach()
                        else:
                            embeddings_list.append(np.zeros(config.STRATEGY_DIM, dtype=np.float32))
                if embeddings_list:
                    embeddings_arr = np.concatenate(embeddings_list, axis=0)
                    min_val = embeddings_arr.min()
                    max_val = embeddings_arr.max()
                    normalized_arr = embeddings_arr if (max_val - min_val)==0 else (embeddings_arr - min_val) / (max_val - min_val)
                else:
                    normalized_arr = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
                num_opponents = len(env.possible_agents) - 1
                segment_size = config.STRATEGY_DIM
                normalized_segments = []
                for i in range(num_opponents):
                    seg = normalized_arr[i * segment_size:(i + 1) * segment_size]
                    normalized_segments.append(torch.tensor(seg, dtype=torch.float32, device=device).unsqueeze(0))
                obp_memory_embeddings = normalized_segments
                transformer_features = normalized_arr
                obp_probs = run_obp_inference(obp_model, observation, device, env.num_players,
                                              memory_embeddings=obp_memory_embeddings)

            # Build final observation based on the agent type.
            if agent == current_injected_agent_id:
                base_obs = observation
                obp_arr = np.array(obp_probs, dtype=np.float32)
                # For historical models, use their expected input dim; for hardcoded, use config.INPUT_DIM.
                if current_injected_bot_type == "historical":
                    expected_input_dim = current_injected_agent_instance.fc1.weight.shape[1]
                else:
                    expected_input_dim = config.INPUT_DIM
                current_dim = base_obs.shape[0] + obp_arr.shape[0]
                missing_dim = expected_input_dim - current_dim
                if missing_dim > 0:
                    mem_features = np.zeros(missing_dim, dtype=np.float32)
                    final_obs = np.concatenate([base_obs, obp_arr, mem_features], axis=0)
                else:
                    final_obs = np.concatenate([base_obs, obp_arr], axis=0)
            else:
                final_obs = np.concatenate([observation, np.array(obp_probs, dtype=np.float32), transformer_features], axis=0)

            # Decide action.
            if agent == current_injected_agent_id:
                if current_injected_bot_type == "hardcoded":
                    action = current_injected_agent_instance.play_turn(observation, action_mask, table_card=None)
                    log_prob_value = 0.0
                else:
                    observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        probs, _, _ = current_injected_agent_instance(observation_tensor, None)
                    probs = torch.clamp(probs, 1e-8, 1.0).squeeze(0)
                    mask_t = torch.tensor(action_mask, dtype=torch.float32, device=device)
                    masked_probs = probs * mask_t
                    if masked_probs.sum() == 0:
                        valid_indices = torch.nonzero(mask_t, as_tuple=True)[0]
                        masked_probs[valid_indices] = 1.0 / valid_indices.numel() if len(valid_indices) > 0 else torch.ones_like(probs) / probs.size(0)
                    else:
                        masked_probs /= masked_probs.sum()
                    m = Categorical(masked_probs)
                    action = m.sample().item()
                    log_prob_value = m.log_prob(torch.tensor(action, device=device)).item()
                # Do not store transitions for injected bots.
            else:
                observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
                probs, _, _ = policy_nets[agent](observation_tensor, None)
                probs = torch.clamp(probs, 1e-8, 1.0).squeeze(0)
                mask_t = torch.tensor(action_mask, dtype=torch.float32, device=device)
                masked_probs = probs * mask_t
                if masked_probs.sum() == 0:
                    valid_indices = torch.nonzero(mask_t, as_tuple=True)[0]
                    masked_probs[valid_indices] = 1.0 / valid_indices.numel() if len(valid_indices) > 0 else torch.ones_like(probs) / probs.size(0)
                else:
                    masked_probs /= masked_probs.sum()
                m = Categorical(masked_probs)
                action = m.sample().item()
                log_prob_value = m.log_prob(torch.tensor(action, device=device)).item()

            action_counts_periodic[agent][action] += 1
            env.step(action)
            
            step_rewards = env.rewards.copy()
            env.rewards = {agent: 0 for agent in env.possible_agents}
            # Update pending rewards.
            for ag in agents:
                if ag != agent:
                    pending_rewards[ag] += step_rewards[ag]
                else:
                    reward = step_rewards[agent] + pending_rewards[agent]
                    pending_rewards[agent] = 0
                    if agent != current_injected_agent_id:
                        memories[agent].store_transition(
                            agent=agent,
                            state=final_obs,
                            action=action,
                            log_prob=log_prob_value,
                            reward=reward,
                            is_terminal=env.terminations[agent] or env.truncations[agent],
                            state_value=( 
                                value_nets[agent](torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)).item()
                                if agent != current_injected_agent_id else 0.0
                            ),
                            action_mask=action_mask
                        )
                    episode_rewards[ag] += reward

        # --- Update moving average win tracking after the episode using env.winner ---
        winners = env.winner
        if not isinstance(winners, list):
            winners = [winners]

        if current_injected_agent_id is not None:
            if current_injected_bot_type == "historical":
                opponent_key = current_injected_bot_identifier
            else:
                opponent_key = current_injected_agent_instance.__class__.__name__
            for agent in agents:
                if agent == current_injected_agent_id:
                    continue
                # Initialize the deque if it doesn't exist yet.
                if opponent_key not in win_history[agent]:
                    win_history[agent][opponent_key] = deque(maxlen=100)
                # Determine win (1) or loss (0) for this episode.
                win = 1 if (agent in winners and current_injected_agent_id not in winners) else 0
                win_history[agent][opponent_key].append(win)
                # Also update games played counter if needed.
                games_played_counter[agent].setdefault(opponent_key, 0)
                games_played_counter[agent][opponent_key] += 1

        for agent in agents:
            recent_rewards[agent].append(episode_rewards[agent])
            if len(recent_rewards[agent]) > 100:
                recent_rewards[agent].pop(0)
        avg_rewards = {agent: np.mean(recent_rewards[agent]) if recent_rewards[agent] else 0.0 for agent in agents}

        # Compute GAE for agents being trained.
        for agent in agents:
            if agent == current_injected_agent_id:
                continue
            memory = memories[agent]
            rewards_agent = memory.rewards[agent]
            dones_agent = memory.is_terminals[agent]
            values_agent = memory.state_values[agent]
            next_values_agent = values_agent[1:] + [0]
            mean_reward = np.mean(rewards_agent)
            std_reward = np.std(rewards_agent) + 1e-5
            normalized_rewards = (np.array(rewards_agent) - mean_reward) / std_reward
            advantages, returns_ = compute_gae(
                rewards=normalized_rewards,
                dones=dones_agent,
                values=values_agent,
                next_values=next_values_agent,
                gamma=config.GAMMA,
                lam=config.GAE_LAMBDA,
            )
            memory.advantages[agent] = advantages
            memory.returns[agent] = returns_
            
        episode_obp_data = extract_obp_training_data(env)
        obp_memory.extend(episode_obp_data)
        
        if episode % config.UPDATE_STEPS == 0:
            for agent in agents:
                if agent == current_injected_agent_id:
                    continue
                memory = memories[agent]
                if not memory.states[agent]:
                    continue
                states = torch.from_numpy(np.array(memory.states[agent], dtype=np.float32)).to(device)
                actions_ = torch.from_numpy(np.array(memory.actions[agent], dtype=np.int64)).to(device)
                old_log_probs = torch.from_numpy(np.array(memory.log_probs[agent], dtype=np.float32)).to(device)
                returns_ = torch.from_numpy(np.array(memory.returns[agent], dtype=np.float32)).to(device)
                advantages_ = torch.from_numpy(np.array(memory.advantages[agent], dtype=np.float32)).to(device)
                action_masks_ = torch.from_numpy(np.array(memory.action_masks[agent], dtype=np.float32)).to(device)
                # Safely normalize advantages.
                adv_std = advantages_.std()
                if adv_std < 1e-5:
                    normalized_advantages = advantages_
                else:
                    normalized_advantages = (advantages_ - advantages_.mean()) / (adv_std + 1e-5)

                kl_divs = []
                policy_grad_norms = []
                value_grad_norms = []
                policy_losses = []
                value_losses = []
                entropies = []
                classification_losses = []

                for _ in range(config.K_EPOCHS):
                    probs, _, opponent_logits = policy_nets[agent](states, None)
                    probs = torch.clamp(probs, 1e-8, 1.0)
                    masked_probs = probs * action_masks_
                    row_sums = masked_probs.sum(dim=-1, keepdim=True)
                    masked_probs = torch.where(
                        row_sums > 0,
                        masked_probs / row_sums,
                        torch.ones_like(masked_probs) / masked_probs.shape[1]
                    )
                    m = Categorical(masked_probs)
                    new_log_probs = m.log_prob(actions_)
                    entropy = m.entropy().mean()
                    kl_div = torch.mean(old_log_probs - new_log_probs)
                    kl_divs.append(kl_div.item())
                    ratios = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratios * normalized_advantages
                    surr2 = torch.clamp(ratios, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP) * normalized_advantages
                    policy_loss = -torch.min(surr1, surr2).mean() - static_entropy_coef * entropy
                    state_values = value_nets[agent](states).squeeze()
                    value_loss = nn.MSELoss()(state_values, returns_)
    
                    if opponent_logits is not None:
                        if current_injected_bot_type == "historical":
                            target_label = historical_label_mapping[current_injected_bot_identifier]
                        else:
                            label_name = current_injected_agent_instance.__class__.__name__
                            target_label = HARD_CODED_LABELS.get(label_name, 0)
                        target_labels = torch.full((opponent_logits.size(0),), target_label, dtype=torch.long, device=device)
                        classification_loss = F.cross_entropy(opponent_logits, target_labels)
                        classification_losses.append(classification_loss.item())
                        predicted_labels = opponent_logits.argmax(dim=1)
                        accuracy = (predicted_labels == target_labels).float().mean().item()
                        if writer is not None:
                            writer.add_scalar(f"Accuracy/Classification/{agent}", accuracy, episode)
                            if current_injected_bot_type == "historical":
                                opp_key = current_injected_bot_identifier
                            else:
                                opp_key = current_injected_agent_instance.__class__.__name__
                            writer.add_scalar(f"Accuracy/Classification/{agent}_vs_{opp_key}", accuracy, episode)
                        total_loss = policy_loss + 0.5 * value_loss + config.AUX_LOSS_WEIGHT * classification_loss
                    else:
                        total_loss = policy_loss + 0.5 * value_loss
    
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropies.append(entropy.item())
                    total_loss.backward()
    
                    p_grad_norm = sum(param.grad.data.norm(2).item() ** 2 for param in policy_nets[agent].parameters() if param.grad is not None) ** 0.5
                    policy_grad_norms.append(p_grad_norm)
                    v_grad_norm = sum(param.grad.data.norm(2).item() ** 2 for param in value_nets[agent].parameters() if param.grad is not None) ** 0.5
                    value_grad_norms.append(v_grad_norm)
    
                    torch.nn.utils.clip_grad_norm_(policy_nets[agent].parameters(), max_norm=config.MAX_NORM)
                    torch.nn.utils.clip_grad_norm_(value_nets[agent].parameters(), max_norm=config.MAX_NORM)
                    optimizers_policy[agent].step()
                    optimizers_value[agent].step()
    
                if writer is not None:
                    writer.add_scalar(f"Loss/Policy/{agent}", np.mean(policy_losses), episode)
                    writer.add_scalar(f"Loss/Value/{agent}", np.mean(value_losses), episode)
                    writer.add_scalar(f"Entropy/{agent}", np.mean(entropies), episode)
                    writer.add_scalar(f"Entropy_Coef/{agent}", static_entropy_coef, episode)
                    writer.add_scalar(f"KL_Divergence/{agent}", np.mean(kl_divs), episode)
                    writer.add_scalar(f"Gradient_Norms/Policy/{agent}", np.mean(policy_grad_norms), episode)
                    writer.add_scalar(f"Gradient_Norms/Value/{agent}", np.mean(value_grad_norms), episode)
                    writer.add_scalar(f"Loss/Classification/{agent}", np.mean(classification_losses) if classification_losses else 0.0, episode)
    
            for agent in agents:
                if agent != current_injected_agent_id:
                    memories[agent].reset()
    
        if len(obp_memory) > 100:
            avg_loss_obp, accuracy = train_obp(obp_model, obp_optimizer, obp_memory, device, logger)
            if avg_loss_obp is not None and accuracy is not None and writer is not None:
                writer.add_scalar("OBP/Loss", avg_loss_obp, episode)
                writer.add_scalar("OBP/Accuracy", accuracy, episode)
            obp_memory = []
    
        if episode % config.CHECKPOINT_INTERVAL == 0 and load_checkpoint:
            save_checkpoint(
                policy_nets,
                value_nets,
                optimizers_policy,
                optimizers_value,
                obp_model,
                obp_optimizer,
                episode,
                checkpoint_dir=checkpoint_dir
            )
            logger.info(f"Saved global checkpoint at episode {episode}.")
    
        steps_since_log += steps_in_episode
        episodes_since_log += 1
    
        if episode % config.LOG_INTERVAL == 0:
            avg_rewards_str = ", ".join([f"{agent}: {avg_rewards.get(agent, 0.0):.2f}" for agent in original_agent_order])
            avg_steps_per_episode = steps_since_log / episodes_since_log
            elapsed_time = time.time() - last_log_time
            steps_per_second = steps_since_log / elapsed_time if elapsed_time > 0 else 0.0
            logger.info(
                f"Episode {episode}\tAverage Rewards: [{avg_rewards_str}]\t"
                f"Avg Steps/Ep: {avg_steps_per_episode:.2f}\t"
                f"Time since last log: {elapsed_time:.2f} seconds\t"
                f"Steps/s: {steps_per_second:.2f}"
            )
            for agent in agents:
                if agent == current_injected_agent_id:
                    continue
                # Compute overall moving average win rate over the last 100 episodes for this agent.
                all_outcomes = []
                for outcomes in win_history[agent].values():
                    all_outcomes.extend(outcomes)
                overall_rate = (sum(all_outcomes)/len(all_outcomes)*100) if all_outcomes else 0
                if writer is not None:
                    writer.add_scalar(f"WinRate/{agent}_Overall", overall_rate, episode)
                for opp_key, outcomes in win_history[agent].items():
                    rate = (sum(outcomes)/len(outcomes)*100) if outcomes else 0
                    if writer is not None:
                        writer.add_scalar(f"WinRate/{agent}_vs_{opp_key}", rate, episode)
    
            if writer is not None:
                for agent, reward in avg_rewards.items():
                    writer.add_scalar(f"Average Reward/{agent}", reward, episode)
                for agent in agents:
                    for action in range(config.OUTPUT_DIM):
                        writer.add_scalar(
                            f"Action Counts/{agent}/Action_{action}",
                            action_counts_periodic[agent][action],
                            episode
                        )
            for agent in agents:
                invalid_action_counts_periodic[agent] = 0
                for action in range(config.OUTPUT_DIM):
                    action_counts_periodic[agent][action] = 0
            last_log_time = time.time()
            steps_since_log = 0
            episodes_since_log = 0
            for agent in games_played_counter:
                for opp_key, count in games_played_counter[agent].items():
                    if writer is not None:
                        writer.add_scalar(f"games_played/{agent}_vs_{opp_key}", count, episode)
            games_played_counter = {agent: {} for agent in agents}
            if episode % config.CULL_INTERVAL == 0:
                average_rewards = {agent: (sum(recent_rewards[agent]) / len(recent_rewards[agent]) if recent_rewards[agent] else 0.0) for agent in agents}
                lowest_agent = min(average_rewards, key=average_rewards.get)
                lowest_score = average_rewards[lowest_agent]
                logger.info(f"Culling Agent '{lowest_agent}' with average reward {lowest_score:.2f}.")
                if lowest_agent != current_injected_agent_id:
                    policy_nets[lowest_agent] = PolicyNetwork(
                        input_dim=config.INPUT_DIM,
                        hidden_dim=config.HIDDEN_DIM,
                        output_dim=config.OUTPUT_DIM,
                        use_lstm=True,
                        use_dropout=True,
                        use_layer_norm=True,
                        use_aux_classifier=True,
                        num_opponent_classes=config.NUM_OPPONENT_CLASSES
                    ).to(device)
                    value_nets[lowest_agent] = ValueNetwork(
                        input_dim=config.INPUT_DIM,
                        hidden_dim=config.HIDDEN_DIM,
                        use_dropout=True,
                        use_layer_norm=True
                    ).to(device)
                    optimizers_policy[lowest_agent] = optim.Adam(policy_nets[lowest_agent].parameters(), lr=config.LEARNING_RATE)
                    optimizers_value[lowest_agent] = optim.Adam(value_nets[lowest_agent].parameters(), lr=config.LEARNING_RATE)
                    memories[lowest_agent] = RolloutMemory([lowest_agent])
                    recent_rewards[lowest_agent] = []
    
    if writer is not None:
        writer.close()
    
    trained_agents = {}
    for agent in agents:
        trained_agents[agent] = {
            'policy_net': policy_nets[agent],
            'value_net': value_nets[agent],
            'obp_model': obp_model
        }
    
    return {
        'agents': trained_agents,
        'optimizers_policy': optimizers_policy,
        'optimizers_value': optimizers_value,
        'obp_optimizer': obp_optimizer
    }

def main():
    """
    Trains agents against a variety of opponents, including hardcoded bots and historical models.
    Uses reinforcement learning with opponent behavior prediction and strategy embeddings.
    Logs training progress and saves checkpoints.
    """
    set_seed()
    device = torch.device(config.DEVICE)
    if config.USE_WRAPPER: 
        base_env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=config.RENDER_MODE)
        env = RewardRestrictionWrapper2(base_env)
    else:
        env = LiarsDeckEnv(num_players=config.NUM_PLAYERS, render_mode=config.RENDER_MODE)
    logger = configure_logger()
    logger.info("Starting training process...")

    training_results = train_agents(
        env=env,
        device=device,
        num_episodes=config.NUM_EPISODES,
        load_checkpoint=True,
        log_tensorboard=True
    )
    if training_results is None:
        logger.error("Training results are None. Exiting.")
        return

    trained_agents = training_results['agents']
    optimizers_policy = training_results['optimizers_policy']
    optimizers_value = training_results['optimizers_value']
    obp_optimizer = training_results['obp_optimizer']
    any_agent = next(iter(trained_agents))
    save_checkpoint(
        {a: trained_agents[a]['policy_net'] for a in trained_agents if trained_agents[a]['policy_net'] is not None},
        {a: trained_agents[a]['value_net'] for a in trained_agents if trained_agents[a]['value_net'] is not None},
        optimizers_policy,
        optimizers_value,
        trained_agents[any_agent]['obp_model'],
        obp_optimizer,
        config.NUM_EPISODES,
        checkpoint_dir=config.CHECKPOINT_DIR
    )
    logger.info("Saved final checkpoint after training.")

if __name__ == "__main__":
    main()
