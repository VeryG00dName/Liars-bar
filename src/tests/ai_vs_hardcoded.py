# src/tests/ai_vs_hardcoded.py

import os
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
import torch
import numpy as np
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.new_models import PolicyNetwork, OpponentBehaviorPredictor
from src import config
import random

from src.model.hard_coded_agents import (
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    StrategicChallenger,
    RandomAgent,
    TableNonTableAgent,
    Classic
)

from src.evaluation.evaluate import run_obp_inference
from src.evaluation.evaluate_utils import (
    adapt_observation_for_version,
    get_hidden_dim_from_state_dict
)

logging.basicConfig(level=logging.INFO)  # Set to DEBUG for detailed logs
logger = logging.getLogger("AgentBattleground")

# --- Global transformer/event encoder variables for transformer‐based memory integration ---
global_strategy_transformer = None
global_event_encoder = None
global_response2idx = None
global_action2idx = None

# --- New helper: Convert memory events into 4D continuous features ---
def convert_memory_to_features(memory, response_mapping, action_mapping):
    """
    Convert the opponent memory (a list of events) to a list of 4-dimensional feature vectors.
    Each event must be a dictionary with keys: "response", "triggering_action", "penalties", and "card_count".
    """
    features = []
    for event in memory:
        if not isinstance(event, dict):
            raise ValueError(f"Memory event is not a dictionary: {event}.")
        resp = event.get("response", "")
        act = event.get("triggering_action", "")
        penalties = float(event.get("penalties", 0))
        card_count = float(event.get("card_count", 0))
        # Convert categorical features to numbers using provided mappings.
        resp_val = float(response_mapping.get(resp, 0))
        act_val = float(action_mapping.get(act, 0))
        features.append([resp_val, act_val, penalties, card_count])
    return features

class AgentBattlegroundGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Agent Battleground")
        self.root.geometry("800x600")
        
        self.loaded_models = {}
        self.hardcoded_agents = {
            "GreedySpammer": GreedyCardSpammer,
            "TableFirst": TableFirstConservativeChallenger,
            "Strategic": lambda name: StrategicChallenger(name, 3, 2),  # Pass agent_index=2
            "Conservative": lambda name: TableFirstConservativeChallenger(name),
            "TableNonTableAgent": TableNonTableAgent,
            "Classic": Classic,
            "Random": RandomAgent
        }
        
        self.create_file_drop_zone()
        self.create_model_info_panel()
        self.create_ai_selection()
        self.create_control_buttons()
        self.create_results_display()

        self.current_env = None  # Will be set when a match is run

    def get_hidden_dim_from_state_dict(self, state_dict, layer_prefix='fc1'):
        """Extracts hidden dimension from model weights using imported utility."""
        return get_hidden_dim_from_state_dict(state_dict, layer_prefix)

    def create_file_drop_zone(self):
        frame = ttk.LabelFrame(self.root, text="Model Files", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        self.file_list = tk.Listbox(frame, height=3, selectmode=tk.SINGLE)
        self.file_list.pack(fill=tk.X)
        drop_label = ttk.Label(frame, text="Drag and drop .pth files here")
        drop_label.pack(pady=5)
        for widget in [frame, self.file_list, drop_label]:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", self.on_file_drop)

    def create_model_info_panel(self):
        frame = ttk.LabelFrame(self.root, text="Model Info", padding=10)
        frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
        self.info_text = tk.Text(frame, wrap=tk.WORD, state=tk.DISABLED, height=4)
        self.info_text.pack(fill=tk.BOTH, expand=True)

    def create_ai_selection(self):
        frame = ttk.LabelFrame(self.root, text="AI Agents Selection", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        self.agent_selectors = {}
        for i in range(2):
            ttk.Label(frame, text=f"AI Agent {i+1}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            self.agent_selectors[i] = ttk.Combobox(frame, state="readonly", width=50)
            self.agent_selectors[i].grid(row=i, column=1, sticky=tk.EW, padx=5, pady=2)
        frame.columnconfigure(1, weight=1)

    def create_control_buttons(self):
        frame = ttk.Frame(self.root)
        frame.pack(pady=10)
        ttk.Button(frame, text="Refresh Agents", command=self.update_agent_selectors).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Start Battleground", command=self.start_battleground).pack(side=tk.LEFT, padx=5)

    def create_results_display(self):
        frame = ttk.LabelFrame(self.root, text="Results", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.results_text = tk.Text(frame, wrap=tk.WORD, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def on_file_drop(self, event):
        file_path = event.data.strip()
        if os.name == 'nt':
            file_path = file_path.replace("{", "").replace("}", "").strip('"')
        else:
            if file_path.startswith("file://"):
                file_path = file_path[7:]
        file_path = os.path.normpath(file_path)
        if not file_path.endswith(".pth"):
            self.show_info("Only .pth files are supported")
            return
        if file_path in self.loaded_models:
            self.show_info("Model already loaded")
            return
        try:
            self.load_model(file_path)
            self.file_list.insert(tk.END, os.path.basename(file_path))
            self.update_agent_selectors()
            self.show_info(f"Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            self.show_info(f"Error: {str(e)}")

    def load_model(self, file_path):
        checkpoint = torch.load(file_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid checkpoint format")
        required_keys = ["policy_nets", "obp_model"]
        if any(k not in checkpoint for k in required_keys):
            raise ValueError("Missing required keys in checkpoint")
        
        # Determine the observation version based on policy net's input dimension.
        any_policy = next(iter(checkpoint["policy_nets"].values()))
        input_dim = any_policy['fc1.weight'].shape[1]
        
        if input_dim == 18:
            obs_version = 1  # OBS_VERSION_1
        elif input_dim in (16, 24, 26):
            obs_version = 2  # OBS_VERSION_2
        else:
            raise ValueError(f"Unknown input_dim {input_dim} for model {file_path}")
        
        # Check if the model uses memory by looking for "fc4.weight"
        uses_memory = ("fc4.weight" in any_policy)
        
        self.loaded_models[file_path] = {
            "policy_nets": checkpoint["policy_nets"],
            "obp_model": checkpoint["obp_model"],
            "obs_version": obs_version,
            "input_dim": input_dim,
            "uses_memory": uses_memory
        }

    def show_info(self, message):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, message)
        self.info_text.config(state=tk.DISABLED)

    def update_agent_selectors(self, event=None):
        agent_options = []
        for file_path, data in self.loaded_models.items():
            folder_name = os.path.basename(os.path.dirname(file_path))
            for agent_name in data["policy_nets"].keys():
                display_text = f"{folder_name} - {os.path.basename(file_path)} - {agent_name}"
                agent_options.append(display_text)
        for i in range(2):
            self.agent_selectors[i]["values"] = agent_options
            if agent_options:
                self.agent_selectors[i].current(0)
            self.agent_selectors[i].state(["!disabled"])

    def load_selected_agents(self):
        """Loads the selected AI agents from the selectors."""
        ai_agents = {}
        try:
            for i in range(2):
                selection = self.agent_selectors[i].get()
                if not selection:
                    raise ValueError(f"Select AI Agent {i+1}")
                parts = selection.split(" - ")
                if len(parts) != 3:
                    raise ValueError("Invalid agent format")
                folder_name, file_name, agent_name = parts
                file_path_candidates = [p for p in self.loaded_models.keys() if os.path.basename(p) == file_name]
                if not file_path_candidates:
                    raise ValueError(f"File for {file_name} not found among loaded models.")
                file_path = file_path_candidates[0]
                model_data = self.loaded_models[file_path]
                ai_agents[f"player_{i}"] = {
                    "policy_net": model_data["policy_nets"][agent_name],
                    "obp_model": model_data["obp_model"],
                    "obs_version": model_data["obs_version"],
                    "input_dim": model_data["input_dim"],
                    "uses_memory": model_data["uses_memory"]
                }
            return ai_agents
        except Exception as e:
            self.show_info(f"Error loading selected agents: {str(e)}")
            return None

    def start_battleground(self):
        try:
            ai_agents = self.load_selected_agents()
            if not ai_agents:
                return

            results = {}
            for hc_name, hc_class in self.hardcoded_agents.items():
                wins = [0, 0, 0]  # [AI1 Wins, AI2 Wins, Hardcoded Wins]
                for _ in range(20):
                    winner = self.run_match(ai_agents, hc_class(hc_name))
                    if winner == "player_0":
                        wins[0] += 1
                    elif winner == "player_1":
                        wins[1] += 1
                    elif winner == "hardcoded_agent":
                        wins[2] += 1
                    else:
                        logger.warning(f"Unknown winner identifier: {winner}")
                results[hc_name] = wins

            self.display_results(results)

        except Exception as e:
            self.show_info(f"Error: {str(e)}")

    def run_match(self, ai_agents, hardcoded_agent):
        env = LiarsDeckEnv(num_players=3, render_mode=None)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_env = env  # Save current environment for memory queries
        
        # Initialize AI agents
        policy_nets = {}
        obp_models = {}
        obs_versions = {}
        input_dims = {}
        uses_memories = {}
        for agent_id, agent_data in ai_agents.items():
            obs_versions[agent_id] = agent_data["obs_version"]
            input_dims[agent_id] = agent_data["input_dim"]
            uses_memories[agent_id] = agent_data["uses_memory"]
            
            policy_net = PolicyNetwork(
                input_dim=agent_data["input_dim"],
                hidden_dim=self.get_hidden_dim_from_state_dict(agent_data["policy_net"]),
                output_dim=env.action_spaces[agent_id].n,
                use_lstm=True,
                use_layer_norm=True
            )
            policy_net.load_state_dict(agent_data["policy_net"])
            policy_net.to(device).eval()
            policy_nets[agent_id] = policy_net

            obp_model_state = agent_data["obp_model"]
            if obp_model_state:
                obp_input_dim = 5 if agent_data["obs_version"] == 1 else 4
                obp_model = OpponentBehaviorPredictor(
                    input_dim=obp_input_dim,
                    hidden_dim=config.OPPONENT_HIDDEN_DIM,
                    output_dim=2
                )
                obp_model.load_state_dict(obp_model_state)
                obp_model.to(device).eval()
                obp_models[agent_id] = obp_model
            else:
                obp_models[agent_id] = None

        env.reset()
        while env.agent_selection is not None:
            current_agent = env.agent_selection
            obs, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                env.step(None)
                continue

            if current_agent in policy_nets:
                action = self.choose_action(
                    current_agent,
                    policy_nets[current_agent],
                    obp_models[current_agent],
                    obs[current_agent],
                    info['action_mask'],
                    device,
                    env.num_players,
                    obs_versions[current_agent],
                    input_dims[current_agent],
                    uses_memories[current_agent]
                )
            else:
                action = hardcoded_agent.play_turn(
                    obs[current_agent],
                    info['action_mask'],
                    env.table_card
                )
                if not isinstance(action, int):
                    logger.error(f"Hardcoded agent {hardcoded_agent.__class__.__name__} returned non-integer action: {action}")
                    raise ValueError(f"Hardcoded agent {hardcoded_agent.__class__.__name__} returned non-integer action: {action}")
            
            env.step(action)
        
        max_reward = max(env.rewards.values())
        winners = [agent for agent, reward in env.rewards.items() if reward == max_reward]
        winner = winners[0]
        hardcoded_agent_id = f"player_{env.num_players - 1}"  # 'player_2'
        
        if winner in ai_agents:
            return winner
        elif winner == hardcoded_agent_id:
            return "hardcoded_agent"
        else:
            return "unknown_agent"

    def choose_action(self, agent_id, policy_net, obp_model, observation, action_mask, device, num_players, obs_version, input_dim, uses_memory):
        """Selects an action using OBP inference and, if applicable, transformer‐based memory integration."""
        converted_obs = adapt_observation_for_version(observation, num_players, obs_version)
        logging.debug(f"Converted observation (length {len(converted_obs)}): {converted_obs}")

        obp_probs = run_obp_inference(obp_model, converted_obs, device, num_players, obs_version)
        logging.debug(f"OBP probabilities: {obp_probs}")

        if obs_version == 2 and uses_memory:
            required_mem_dim = input_dim - (len(converted_obs) + len(obp_probs))
            if required_mem_dim == config.STRATEGY_DIM * (num_players - 1):
                from src.env.liars_deck_env_utils import query_opponent_memory_full
                global global_response2idx, global_action2idx
                transformer_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth")
                if global_response2idx is None or global_action2idx is None:
                    if os.path.exists(transformer_checkpoint_path):
                        ckpt = torch.load(transformer_checkpoint_path, map_location=device)
                        global_response2idx = ckpt.get("response2idx", {})
                        global_action2idx = ckpt.get("action2idx", {})
                    else:
                        global_response2idx = {}
                        global_action2idx = {}
                mem_features_list = []
                for opp in self.current_env.possible_agents:
                    if opp != agent_id:
                        from src.env.liars_deck_env_utils import query_opponent_memory_full
                        mem_summary = query_opponent_memory_full(agent_id, opp)
                        features_list = convert_memory_to_features(mem_summary, global_response2idx, global_action2idx)
                        if features_list:
                            # Create a float tensor of continuous features.
                            feature_tensor = torch.tensor(features_list, dtype=torch.float, device=device).unsqueeze(0)
                            global global_event_encoder, global_strategy_transformer
                            if global_event_encoder is None:
                                from src.training.train_transformer import EventEncoder
                                global_event_encoder = EventEncoder(
                                    response_vocab_size=len(global_response2idx),
                                    action_vocab_size=len(global_action2idx),
                                    token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM
                                ).to(device)
                                if os.path.exists(transformer_checkpoint_path):
                                    ckpt = torch.load(transformer_checkpoint_path, map_location=device)
                                    global_event_encoder.load_state_dict(ckpt["event_encoder_state_dict"])
                                    global_event_encoder.eval()
                            if global_strategy_transformer is None:
                                from src.model.new_models import StrategyTransformer
                                global_strategy_transformer = StrategyTransformer(
                                    num_tokens=config.STRATEGY_NUM_TOKENS,
                                    token_embedding_dim=config.STRATEGY_TOKEN_EMBEDDING_DIM,
                                    nhead=config.STRATEGY_NHEAD,
                                    num_layers=config.STRATEGY_NUM_LAYERS,
                                    strategy_dim=config.STRATEGY_DIM,
                                    num_classes=config.STRATEGY_NUM_CLASSES,
                                    dropout=config.STRATEGY_DROPOUT,
                                    use_cls_token=True
                                ).to(device)
                                if os.path.exists(transformer_checkpoint_path):
                                    ckpt = torch.load(transformer_checkpoint_path, map_location=device)
                                    global_strategy_transformer.load_state_dict(ckpt["transformer_state_dict"], strict=False)
                                # Override the token embedding to bypass index lookup.
                                global_strategy_transformer.token_embedding = torch.nn.Identity()
                                global_strategy_transformer.classification_head = None
                                global_strategy_transformer.eval()
                            with torch.no_grad():
                                # First, project the continuous features.
                                projected = global_event_encoder(feature_tensor)
                                embedding, _ = global_strategy_transformer(projected)
                            mem_features_list.append(embedding.cpu().numpy().flatten())
                        else:
                            mem_features_list.append(np.zeros(config.STRATEGY_DIM, dtype=np.float32))
                if mem_features_list:
                    mem_features = np.concatenate(mem_features_list, axis=0)
                else:
                    mem_features = np.zeros(config.STRATEGY_DIM * (num_players - 1), dtype=np.float32)
            else:
                from src.env.liars_deck_env_utils import query_opponent_memory
                mem_features_list = []
                for opp in self.current_env.possible_agents:
                    if opp != agent_id:
                        mem_summary = query_opponent_memory(agent_id, opp)
                        mem_features_list.append(mem_summary)
                if mem_features_list:
                    mem_features = np.concatenate(mem_features_list, axis=0)
                else:
                    mem_features = np.array([], dtype=np.float32)
            current_mem_dim = mem_features.shape[0]
            if current_mem_dim < required_mem_dim:
                pad = np.zeros(required_mem_dim - current_mem_dim, dtype=np.float32)
                mem_features = np.concatenate([mem_features, pad], axis=0)
            elif current_mem_dim > required_mem_dim:
                mem_features = mem_features[:required_mem_dim]
            final_obs = np.concatenate([converted_obs, np.array(obp_probs, dtype=np.float32), mem_features], axis=0)
        else:
            final_obs = np.concatenate([converted_obs, np.array(obp_probs, dtype=np.float32)], axis=0)
        logging.debug(f"Final observation (length {len(final_obs)}): {final_obs}")
        
        expected_dim = input_dim
        actual_dim = final_obs.shape[0]
        logging.debug(f"Expected dim: {expected_dim}, Actual dim: {actual_dim}")
        assert actual_dim == expected_dim, f"Expected observation dimension {expected_dim}, got {actual_dim}"
        
        observation_tensor = torch.tensor(final_obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs, _ = policy_net(observation_tensor, None)
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32).to(device)
        masked_probs = action_probs * mask_tensor
        if masked_probs.sum() == 0:
            masked_probs = mask_tensor / mask_tensor.sum()
        else:
            masked_probs /= masked_probs.sum()
        m = torch.distributions.Categorical(masked_probs)
        action = m.sample().item()
        logging.debug(f"Action probabilities: {masked_probs.cpu().numpy()}")
        logging.debug(f"Selected action: {action}")
        if action_mask[action] == 6:
            logging.debug("Challenge Action Selected")
        return action

    def display_results(self, results):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        header = "Hardcoded Agent | AI1 Wins | AI2 Wins | Hardcoded Wins | AI1 Win Rate | AI2 Win Rate | Hardcoded Win Rate\n"
        self.results_text.insert(tk.END, header)
        self.results_text.insert(tk.END, "-"*100 + "\n")
        
        for hc_name, wins in results.items():
            ai1_wins, ai2_wins, hc_wins = wins
            total = ai1_wins + ai2_wins + hc_wins
            rate1 = ai1_wins / total if total > 0 else 0.0
            rate2 = ai2_wins / total if total > 0 else 0.0
            rate_hc = hc_wins / total if total > 0 else 0.0
            line = (f"{hc_name:20} | {ai1_wins:^9} | {ai2_wins:^9} | {hc_wins:^15} | "
                    f"{rate1:12.2%} | {rate2:12.2%} | {rate_hc:18.2%}\n")
            self.results_text.insert(tk.END, line)
            
        self.results_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = AgentBattlegroundGUI(root)
    root.mainloop()
