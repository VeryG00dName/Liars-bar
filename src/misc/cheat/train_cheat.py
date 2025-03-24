# src/training/train_cheat.py
import logging
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
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
from collections import defaultdict, deque

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# Environment & model imports
from src.env.liars_deck_env_core import LiarsDeckEnv
# Use the new model from other_models.
from src.model.other_models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor, StrategyTransformer
from src.model.memory import RolloutMemory
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

# For querying memory
from src.env.liars_deck_env_utils import query_opponent_memory_full

torch.backends.cudnn.benchmark = False

# PPO and logging utilities
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
    convert_memory_to_features,
    visualize_strategy_embeddings
)

# Strategy Transformer and event encoder
from src.training.train_transformer import EventEncoder

class ConsoleLogger:
    def __init__(self):
        self.last_messages = {}  # Track last message per agent
        self.repeat_counts = {}  # Track repeat count per agent

    def log(self, agent, message):
        """Efficient logging that updates the last line if the message is the same for the same agent."""
        if agent in self.last_messages and self.last_messages[agent] == message:
            self.repeat_counts[agent] += 1
            sys.stdout.write(f"\r{message} (x{self.repeat_counts[agent]})")  # Overwrites last line
            sys.stdout.flush()
        else:
            if agent in self.last_messages:
                sys.stdout.write("\n")  # Newline before printing new message for that agent
            sys.stdout.write(message)
            sys.stdout.flush()
            self.last_messages[agent] = message
            self.repeat_counts[agent] = 1

def map_expert_inputs(raw_index):
    if raw_index <= 6:
        return raw_index
    elif 7 <= raw_index <= 22:
        return 7
    elif 23 <= raw_index <= 38:
        return 8
    elif 39 <= raw_index <= 41:
        return 9
    else:
        raise ValueError(f"Raw expert index {raw_index} out of expected range.")
    
HARD_CODED_LABELS = {
    "GreedyCardSpammer": 2,
    "StrategicChallenger": 4,
    "TableNonTableAgent": 6,
    "Classic": 0,
    "TableFirstConservativeChallenger": 5,
    "SelectiveTableConservativeChallenger": 1,
    "RandomAgent": 3
}
historical_label_mapping = {}

device = torch.device(config.DEVICE)

# Initialize the strategy transformer
strategy_transformer = StrategyTransformer(
    num_tokens=config.STRATEGY_NUM_TOKENS,
    token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM,
    nhead=config.STRATEGY_NHEAD,
    num_layers=config.STRATEGY_NUM_LAYERS,
    strategy_dim=config.STRATEGY_DIM,
    num_classes=config.STRATEGY_NUM_CLASSES,
    dropout=config.STRATEGY_DROPOUT,
    use_cls_token=True
).to(device)

transformer_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth")
if os.path.exists(transformer_checkpoint_path):
    checkpoint = torch.load(transformer_checkpoint_path, map_location=device, weights_only=False)
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

# Replace the token embedding with identity.
strategy_transformer.token_embedding = nn.Identity()
# We still keep the classification head in the model (for external expert selection), but it will no longer be used for expert selection.
transformer_classification_head = strategy_transformer.classification_head
transformer_classification_head.eval()
strategy_transformer.eval()

historical_models = load_specific_historical_models(config.HISTORICAL_MODEL_DIR, device)
print(f"Loaded {len(historical_models)} historical PPO models: {', '.join([id for _, id in historical_models])}")

for idx, (_, identifier) in enumerate(historical_models):
    historical_label_mapping[identifier] = len(HARD_CODED_LABELS) + idx

def train_agents(env, device, num_episodes=1000, load_checkpoint=True, load_directory=None, log_tensorboard=True):
    set_seed(config.SEED)
    obs, infos = env.reset(seed=config.SEED)
    agents = env.agents
    assert len(agents) == config.NUM_PLAYERS, f"Expected {config.NUM_PLAYERS} agents, but got {len(agents)} agents."
    num_opponents = config.NUM_PLAYERS - 1
    config.set_derived_config(env.observation_spaces[agents[0]], env.action_spaces[agents[0]], num_opponents)
    
    # Build list of possible injected bots.
    hardcoded_agent_classes = [
        GreedyCardSpammer,
        TableFirstConservativeChallenger,
        StrategicChallenger,
        SelectiveTableConservativeChallenger,
        TableNonTableAgent,
        RandomAgent,
        Classic
    ]
    injected_bots = []
    for cls in hardcoded_agent_classes:
        injected_bots.append(("hardcoded", cls))
    for hist_model, identifier in historical_models:
        injected_bots.append(("historical", (hist_model, identifier)))
    
    # Track wins and games played.
    win_history = {agent: {} for agent in agents}
    games_played_counter = {agent: {} for agent in agents}
    
    # ------------------ NEW: Initialize Transformer Classification Accuracy Tracking ------------------
    transformer_accuracy_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    # ------------------------------------------------------------------------------------------------------
    console_logger = ConsoleLogger()
    # Setup policy/value networks using the new mixture-of-experts model.
    policy_nets = {}
    value_nets = {}
    optimizers_policy = {}
    optimizers_value = {}
    memories = {}
    for agent in agents:
        # ---------------- New: Initialize policy network (no gating network load) ----------------
        # Note: For learning agents, the final observation will be 16 dims (14 from env + 2 from OBP).
        policy_net = PolicyNetwork(
            input_dim=16,  # learning agents we use only 16.
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            num_experts=len(injected_bots)+1,  # one expert per injected bot, plus one for default
            use_lstm=True,
            use_dropout=True,
            use_layer_norm=True,
        ).to(device)
        value_net = ValueNetwork(
            input_dim=16,
            hidden_dim=config.HIDDEN_DIM,
            use_dropout=True,
            use_layer_norm=True
        ).to(device)
        policy_nets[agent] = policy_net
        value_nets[agent] = value_net
        optimizers_policy[agent] = optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE)
        optimizers_value[agent] = optim.Adam(value_net.parameters(), lr=config.LEARNING_RATE)
        # IMPORTANT: We now assume that RolloutMemory stores a new key "expert_indices"
        memories[agent] = RolloutMemory([agent])
    
    obp_model = OpponentBehaviorPredictor(
        input_dim=config.OPPONENT_INPUT_DIM,
        hidden_dim=config.OPPONENT_HIDDEN_DIM,
        output_dim=2,
        memory_dim=config.STRATEGY_DIM
    ).to(device)
    obp_optimizer = optim.Adam(obp_model.parameters(), lr=config.OPPONENT_LEARNING_RATE)
    obp_memory = []
    
    obp_model.eval()
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
    
    action_counts_periodic = {agent: {action: 0 for action in range(config.OUTPUT_DIM)} for agent in agents}
    recent_rewards = {agent: [] for agent in agents}
    original_agent_order = list(env.agents)
    
    current_injected_agent_id = None
    current_injected_agent_instance = None
    current_injected_bot_type = None
    current_injected_bot_identifier = None
    
    global_step = 0
    tracked_agent = None
    
    strategy_embeddings_by_agent_opponent = {}
    
    for episode in range(start_episode, num_episodes + 1):
        env_seed = config.SEED + episode
        obs, infos = env.reset(seed=env_seed)
        agents = env.agents
        pending_rewards = {agent: 0.0 for agent in agents}
    
        # Switch injected bot every 5 episodes.
        if (episode - start_episode) % 5 == 0:
            from src.model.memory import PERSISTENT_OPPONENT_MEMORIES
            PERSISTENT_OPPONENT_MEMORIES.clear()
            current_injected_agent_id = random.choice(agents)
            tracked_agent = current_injected_agent_id
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
                current_injected_bot_identifier = current_injected_agent_instance.__class__.__name__
            else:
                current_injected_agent_instance, current_injected_bot_identifier = selected_bot[1]
    
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
            observation = observation_dict[agent]  # Assume 14 dims from env.
            action_mask = env.infos[agent]['action_mask']
    
            # --- OBP Memory Gathering for injected agent ---
            if agent == current_injected_agent_id:
                # Historical/injected agent: use original processing (14+2+memory).
                if current_injected_bot_type == "hardcoded":
                    obp_probs = []
                else:
                    embeddings_list = []
                    for opp in env.possible_agents:
                        if opp != agent:
                            memory_full = query_opponent_memory_full(agent, opp)
                            # Pad memory_full to length 200 if needed.
                            if len(memory_full) < 200:
                                pad_event = {"response": "", "triggering_action": "", "penalties": 0, "card_count": 0}
                                memory_full = memory_full + [pad_event] * (200 - len(memory_full))
                            features_list = convert_memory_to_features(memory_full, response2idx, action2idx)
                            if features_list:
                                feature_tensor = torch.tensor(features_list, dtype=torch.float32, device=device).unsqueeze(0)
                                with torch.no_grad():
                                    projected = event_encoder(feature_tensor)
                                    strategy_embedding, _ = strategy_transformer(projected)
                            else:
                                strategy_embedding = None
                            if strategy_embedding is not None:
                                embeddings_list.append(strategy_embedding.cpu().detach().numpy().flatten())
                            else:
                                embeddings_list.append(np.zeros(config.STRATEGY_DIM, dtype=np.float32))
                    if embeddings_list:
                        embeddings_arr = np.concatenate(embeddings_list, axis=0)
                        norm_val = np.linalg.norm(embeddings_arr, ord=2)
                        normalized_arr = embeddings_arr if norm_val == 0 else embeddings_arr / norm_val
                    else:
                        normalized_arr = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
                    num_opponents = len(env.possible_agents) - 1
                    segment_size = config.STRATEGY_DIM
                    normalized_segments = []
                    for i in range(num_opponents):
                        seg = normalized_arr[i * segment_size:(i + 1) * segment_size]
                        normalized_segments.append(torch.tensor(seg, dtype=torch.float32, device=device).unsqueeze(0))
                    obp_memory_embeddings = normalized_segments
                    obp_probs = run_obp_inference(obp_model, observation, device, env.num_players,
                                                  memory_embeddings=obp_memory_embeddings)
                # Construct final observation for historical agent (14+2+memory embedding)
                if current_injected_bot_type == "hardcoded":
                    final_obs = observation
                else:
                    base_obs = observation
                    obp_arr = np.array(obp_probs, dtype=np.float32)
                    expected_input_dim = current_injected_agent_instance.fc1.weight.shape[1]
                    current_dim = base_obs.shape[0] + obp_arr.shape[0]
                    missing_dim = expected_input_dim - current_dim
                    if missing_dim > 0:
                        mem_features = np.zeros(missing_dim, dtype=np.float32)
                        final_obs = np.concatenate([base_obs, obp_arr, mem_features], axis=0)
                    else:
                        final_obs = np.concatenate([base_obs, obp_arr], axis=0)
            else:
                # ---------- Learning Agent Processing ----------
                # Compute OBP with embeddings (same as historical branch)
                embeddings_list = []
                for opp in env.possible_agents:
                    if opp != agent:
                        memory_full = query_opponent_memory_full(agent, opp)
                        # Pad memory_full to length 200 if needed.
                        if len(memory_full) < 200:
                            pad_event = {"response": "", "triggering_action": "", "penalties": 0, "card_count": 0}
                            memory_full = memory_full + [pad_event] * (200 - len(memory_full))
                        features_list = convert_memory_to_features(memory_full, response2idx, action2idx)
                        if features_list:
                            feature_tensor = torch.tensor(features_list, dtype=torch.float32, device=device).unsqueeze(0)
                            with torch.no_grad():
                                projected = event_encoder(feature_tensor)
                                strategy_embedding, _ = strategy_transformer(projected)
                        else:
                            strategy_embedding = None
                        if strategy_embedding is not None:
                            embeddings_list.append(strategy_embedding.cpu().detach().numpy().flatten())
                        else:
                            embeddings_list.append(np.zeros(config.STRATEGY_DIM, dtype=np.float32))
                if embeddings_list:
                    embeddings_arr = np.concatenate(embeddings_list, axis=0)
                    norm_val = np.linalg.norm(embeddings_arr, ord=2)
                    normalized_arr = embeddings_arr if norm_val == 0 else embeddings_arr / norm_val
                else:
                    normalized_arr = np.zeros(config.STRATEGY_DIM * (env.num_players - 1), dtype=np.float32)
                num_opponents = len(env.possible_agents) - 1
                segment_size = config.STRATEGY_DIM
                normalized_segments = []
                for i in range(num_opponents):
                    seg = normalized_arr[i * segment_size:(i + 1) * segment_size]
                    normalized_segments.append(torch.tensor(seg, dtype=torch.float32, device=device).unsqueeze(0))
                obp_memory_embeddings = normalized_segments
                obp_probs = run_obp_inference(obp_model, observation, device, env.num_players,
                                              memory_embeddings=obp_memory_embeddings)
                
                # Compute final_obs: only the 14-dim env observation and 2-dim OBP output.
                obp_arr = np.array(obp_probs, dtype=np.float32)  # 2 dims
                final_obs = np.concatenate([observation, obp_arr], axis=0)  # 14 + 2 = 16 dims
    
                # Instead of computing an expert input and classifying it, we directly select the opponent.
                # If an injected opponent exists, use its label.
                if current_injected_agent_id is not None and current_injected_agent_id != agent:
                    selected_opp = current_injected_agent_id
                    if current_injected_bot_type == "hardcoded":
                        expert_inputs = HARD_CODED_LABELS[current_injected_agent_instance.__class__.__name__]
                    else:
                        expert_inputs = historical_label_mapping[current_injected_bot_identifier]
                else:
                    expert_inputs = 10  # default expert index if no injected opponent is available
    
            # ---------- Action Selection ----------
            if agent == current_injected_agent_id:
                # Historical/injected agent uses its own policy.
                if current_injected_bot_type == "hardcoded":
                    action = current_injected_agent_instance.play_turn(observation, action_mask, table_card=None)
                    log_prob_value = 0.0
                else:
                    observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        probs, _ , _= current_injected_agent_instance(observation_tensor, None)
                    probs = torch.clamp(probs, 1e-8, 1.0).squeeze(0)
                    mask_t = torch.tensor(action_mask, dtype=torch.float32, device=device)
                    masked_probs = probs * mask_t
                    if masked_probs.sum() == 0:
                        valid_indices = torch.nonzero(mask_t, as_tuple=True)[0]
                        if len(valid_indices) > 0:
                            masked_probs[valid_indices] = 1.0 / valid_indices.numel()
                        else:
                            masked_probs = torch.ones_like(probs) / probs.size(0)
                    else:
                        masked_probs /= masked_probs.sum()
                    m = Categorical(masked_probs)
                    action = m.sample().item()
                    log_prob_value = m.log_prob(torch.tensor(action, device=device)).item()
            else:
                # Learning agent uses the injected opponent's label (saved as expert_inputs) for the transition.
                observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
                probs, _ = policy_nets[agent](observation_tensor, expert_inputs, None)
                probs = torch.clamp(probs, 1e-8, 1.0).squeeze(0)
                mask_t = torch.tensor(action_mask, dtype=torch.float32, device=device)
                masked_probs = probs * mask_t
                if masked_probs.sum() == 0:
                    valid_indices = torch.nonzero(mask_t, as_tuple=True)[0]
                    if len(valid_indices) > 0:
                        masked_probs[valid_indices] = 1.0 / valid_indices.numel()
                    else:
                        masked_probs = torch.ones_like(probs) / probs.size(0)
                else:
                    masked_probs /= masked_probs.sum()
                m = Categorical(masked_probs)
                action = m.sample().item()
                log_prob_value = m.log_prob(torch.tensor(action, device=device)).item()
    
            # ---------- Environment Step ----------
            action_counts_periodic[agent][action] += 1
            env.step(action)
            
            step_rewards = env.rewards.copy()
            env.rewards = {agent: 0 for agent in env.possible_agents}
            for ag in agents:
                if ag != agent:
                    pending_rewards[ag] += step_rewards[ag]
                else:
                    reward = step_rewards[agent] + pending_rewards[agent]
                    pending_rewards[agent] = 0
                    # --- Save the expert_inputs with the transition ---
                    if agent != current_injected_agent_id:
                        memories[agent].store_transition(
                            agent=agent,
                            state=final_obs,                # 16 dims: 14 (env) + 2 (OBP)
                            expert_input=expert_inputs,      # Save the expert index used during rollout
                            action=action,
                            log_prob=log_prob_value,
                            reward=reward,
                            is_terminal=env.terminations[agent] or env.truncations[agent],
                            state_value=(value_nets[agent](torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)).item()
                                         if agent != current_injected_agent_id else 0.0),
                            action_mask=action_mask
                        )
                    episode_rewards[ag] += reward
    
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
                if opponent_key not in win_history[agent]:
                    win_history[agent][opponent_key] = deque(maxlen=100)
                win = 1 if (agent in winners and current_injected_agent_id not in winners) else 0
                win_history[agent][opponent_key].append(win)
                games_played_counter[agent].setdefault(opponent_key, 0)
                games_played_counter[agent][opponent_key] += 1
    
        for agent in agents:
            recent_rewards[agent].append(episode_rewards[agent])
            if len(recent_rewards[agent]) > 100:
                recent_rewards[agent].pop(0)
        avg_rewards = {agent: np.mean(recent_rewards[agent]) if recent_rewards[agent] else 0.0 for agent in agents}
    
        # --- Compute Advantages using GAE ---
        for agent in memories:
            if memories[agent].states[agent]:
                rewards_agent = memories[agent].rewards[agent]
                dones_agent = memories[agent].is_terminals[agent]
                values_agent = memories[agent].state_values[agent]
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
                memories[agent].advantages[agent] = advantages
                memories[agent].returns[agent] = returns_
    
        # --- Extract OBP training data ---
        episode_obp_data = extract_obp_training_data(env)
        if episode_obp_data:
            obp_memory.extend(episode_obp_data)
    
        # --- PPO Update Loop: every UPDATE_STEPS episodes ---
        if episode % config.UPDATE_STEPS == 0:
            for agent in agents:
                memory = memories[agent]
                states = torch.tensor(np.array(memory.states[agent], dtype=np.float32), device=device)
                actions_ = torch.tensor(np.array(memory.actions[agent], dtype=np.int64), device=device)
                old_log_probs = torch.tensor(np.array(memory.log_probs[agent], dtype=np.float32), device=device)
                returns_ = torch.tensor(np.array(memory.returns[agent], dtype=np.float32), device=device)
                advantages_ = torch.tensor(np.array(memory.advantages[agent], dtype=np.float32), device=device)
                action_masks_ = torch.tensor(np.array(memory.action_masks[agent], dtype=np.float32), device=device)
                
                if advantages_.numel() == 0:
                    logger.debug(f"Skipping update for {agent} at episode {episode} because advantages are empty.")
                    continue

                adv_std = advantages_.std()
                if adv_std < 1e-5:
                    normalized_advantages = advantages_
                else:
                    normalized_advantages = (advantages_ - advantages_.mean()) / (adv_std + 1e-5)
                
                kl_divs = []
                policy_losses = []
                value_losses = []
                entropies = []
    
                # --- Use stored expert indices ---
                # Retrieve the stored expert indices for this agent from memory.
                # We use the mode (most common expert index) over the batch.
                stored_expert_indices = np.array(memory.expert_inputs[agent], dtype=np.int64)
                expert_inputs = int(np.bincount(stored_expert_indices).argmax())
    
                for _ in range(config.K_EPOCHS):
                    probs, _ = policy_nets[agent](states, expert_inputs, None)
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
    
                    total_loss = policy_loss + 0.5 * value_loss
                    optimizers_policy[agent].zero_grad()
                    optimizers_value[agent].zero_grad()
    
                    total_loss.backward(retain_graph=True)
                    optimizers_policy[agent].step()
                    optimizers_value[agent].step()
    
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropies.append(entropy.item())
    
                if writer is not None:
                    writer.add_scalar(f"Loss/Policy/{agent}", np.mean(policy_losses), episode)
                    writer.add_scalar(f"Loss/Value/{agent}", np.mean(value_losses), episode)
                    writer.add_scalar(f"Entropy/{agent}", np.mean(entropies), episode)
                    writer.add_scalar(f"Entropy_Coef/{agent}", static_entropy_coef, episode)
                    writer.add_scalar(f"KL_Divergence/{agent}", np.mean(kl_divs), episode)
    
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
            avg_rewards_str = ", ".join([f"{agent}: {np.mean(recent_rewards[agent]) if recent_rewards[agent] else 0.0:.2f}" for agent in original_agent_order])
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
            # ------------------ NEW: Log Transformer Classification Accuracy ------------------
            if writer is not None:
                for opp_identifier, counts in transformer_accuracy_counts.items():
                    if counts["total"] > 0:
                        accuracy_percentage = counts["correct"] / counts["total"] * 100
                        writer.add_scalar(f"Transformer_Classification_Accuracy/{opp_identifier}", accuracy_percentage, episode)
            transformer_accuracy_counts.clear()
            # -----------------------------------------------------------------------------------
    
            last_log_time = time.time()
            steps_since_log = 0
            episodes_since_log = 0
            for agent in games_played_counter:
                for opp_key, count in games_played_counter[agent].items():
                    if writer is not None:
                        writer.add_scalar(f"games_played/{agent}_vs_{opp_key}", count, episode)
            games_played_counter = {agent: {} for agent in agents}
    
            if episode % 1000 == 0 and writer is not None and len(strategy_embeddings_by_agent_opponent) > 1:
                all_agents = list(agents)
                all_opponents = list(set([k[1] for k in strategy_embeddings_by_agent_opponent.keys()]))
                reference_embeddings = np.array(list(strategy_embeddings_by_agent_opponent.values()))
                logger.info(f"Generating strategy embedding visualizations for episode {episode}")
                visualize_strategy_embeddings(writer, strategy_embeddings_by_agent_opponent, all_agents, all_opponents, episode, method='pca', reference_embeddings=reference_embeddings)
                visualize_strategy_embeddings(writer, strategy_embeddings_by_agent_opponent, all_agents, all_opponents, episode, method='tsne', reference_embeddings=reference_embeddings)
                strategy_embeddings_by_agent_opponent.clear()
    
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
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)

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
