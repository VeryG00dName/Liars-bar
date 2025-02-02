# src/misc/ai_vs_hardcoded.py

import os
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
import torch
import numpy as np
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.models import PolicyNetwork, OpponentBehaviorPredictor
from src import config
import random

from src.model.hard_coded_agents import GreedyCardSpammer, TableFirstConservativeChallenger, StrategicChallenger, RandomAgent

from src.evaluation.evaluate import run_obp_inference
from src.evaluation.evaluate_utils import (
    adapt_observation_for_version,
    get_hidden_dim_from_state_dict
)

logging.basicConfig(level=logging.INFO)  # Set to DEBUG for detailed logs
logger = logging.getLogger("AgentBattleground")

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
            "Random": RandomAgent
        }
        
        self.create_file_drop_zone()
        self.create_model_info_panel()
        self.create_ai_selection()
        self.create_control_buttons()
        self.create_results_display()

        self.current_env = None

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
        
        # Determine the observation version based on policy network's input dimension
        any_policy = next(iter(checkpoint["policy_nets"].values()))
        input_dim = any_policy['fc1.weight'].shape[1]
        
        if input_dim == 18:
            obs_version = 1  # OBS_VERSION_1
        elif input_dim == 16:
            obs_version = 2  # OBS_VERSION_2
        else:
            raise ValueError(f"Unknown input_dim {input_dim} for model {file_path}")
        
        self.loaded_models[file_path] = {
            "policy_nets": checkpoint["policy_nets"],
            "obp_model": checkpoint["obp_model"],
            "obs_version": obs_version,
            "input_dim": input_dim
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
                    "input_dim": model_data["input_dim"]
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
                for _ in range(10):
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
        
        # Initialize AI agents
        policy_nets = {}
        obp_models = {}
        obs_versions = {}
        input_dims = {}
        for agent_id, agent_data in ai_agents.items():
            obs_version = agent_data["obs_version"]
            input_dim = agent_data["input_dim"]
            obs_versions[agent_id] = obs_version
            input_dims[agent_id] = input_dim
            
            policy_net = PolicyNetwork(
                input_dim=input_dim,
                hidden_dim=self.get_hidden_dim_from_state_dict(agent_data["policy_net"]),
                output_dim=env.action_spaces[agent_id].n,
                use_lstm=True,
                use_layer_norm=True
            )
            policy_net.load_state_dict(agent_data["policy_net"])
            policy_net.to(device).eval()
            policy_nets[agent_id] = policy_net

            # Initialize OBP model
            obp_model_state = agent_data["obp_model"]
            if obp_model_state:
                obp_input_dim = 5 if obs_version == 1 else 4
                obp_model = OpponentBehaviorPredictor(
                    input_dim=obp_input_dim,
                    hidden_dim=config.OPPONENT_HIDDEN_DIM,
                    output_dim=2
                )
                obp_model.load_state_dict(obp_model_state)
                obp_model.to(device).eval()
                obp_models[agent_id] = obp_model
            else:
                obp_models[agent_id] = None  # Handle cases where OBP is not present

        env.reset()
        while env.agent_selection is not None:
            current_agent = env.agent_selection
            obs, reward, termination, truncation, info = env.last()
            
            if termination or truncation:
                env.step(None)
                continue

            if current_agent in policy_nets:
                # AI Agent's turn
                obs_version = obs_versions[current_agent]
                input_dim = input_dims[current_agent]
                action = self.choose_action(
                    policy_nets[current_agent],
                    obp_models[current_agent],
                    obs[current_agent],
                    info['action_mask'],
                    device,
                    env.num_players,
                    obs_version,
                    input_dim
                )
            else:
                # Hardcoded Agent's turn
                action = hardcoded_agent.play_turn(
                    obs[current_agent],
                    info['action_mask'],
                    env.table_card
                )
                # Ensure 'action' is an integer
                if not isinstance(action, int):
                    logger.error(f"Hardcoded agent {hardcoded_agent.__class__.__name__} returned non-integer action: {action}")
                    raise ValueError(f"Hardcoded agent {hardcoded_agent.__class__.__name__} returned non-integer action: {action}")
            
            env.step(action)
        
        # Determine winner
        max_reward = max(env.rewards.values())
        winners = [agent for agent, reward in env.rewards.items() if reward == max_reward]
        # Assuming single winner for simplicity
        winner = winners[0]
        
        # Determine hardcoded agent's identifier
        hardcoded_agent_id = f"player_{env.num_players - 1}"  # 'player_2'
        
        if winner in ai_agents:
            return winner  # 'player_0' or 'player_1'
        elif winner == hardcoded_agent_id:
            return "hardcoded_agent"
        else:
            return "unknown_agent"

    def choose_action(self, policy_net, obp_model, observation, action_mask, device, num_players, obs_version, input_dim):
        """Full action selection with OBP integration using imported utilities."""
        # Adapt observation based on agent version
        converted_obs = adapt_observation_for_version(
            observation, 
            num_players,
            obs_version
        )
        logging.debug(f"Converted observation (length {len(converted_obs)}): {converted_obs}")

        # Run OBP inference
        obp_probs = run_obp_inference(
            obp_model,
            converted_obs,
            device,
            num_players,
            obs_version
        )
        logging.debug(f"OBP probabilities: {obp_probs}")

        # Append OBP predictions to observation
        final_obs = np.concatenate([converted_obs, np.array(obp_probs, dtype=np.float32)], axis=0)
        logging.debug(f"Final observation (length {len(final_obs)}): {final_obs}")
        
        # Determine expected input dimension
        # Since input_dim already includes the OBP probabilities, no need to add num_opponents
        expected_dim = input_dim  # Corrected
        actual_dim = final_obs.shape[0]
        logging.debug(f"Expected dim: {expected_dim}, Actual dim: {actual_dim}")
        assert actual_dim == expected_dim, f"Expected observation dimension {expected_dim}, got {actual_dim}"

        observation_tensor = torch.tensor(final_obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Get action probabilities
        with torch.no_grad():
            action_probs, _ = policy_net(observation_tensor)
        
        # Apply action mask
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32).to(device)
        masked_probs = action_probs * mask_tensor
        
        # Handle zero-probability cases
        if masked_probs.sum() == 0:
            # Fallback to uniform distribution over valid actions
            masked_probs = mask_tensor / mask_tensor.sum()
        else:
            # Normalize valid actions
            masked_probs /= masked_probs.sum()
        
        # Sample action
        m = torch.distributions.Categorical(masked_probs)
        action = m.sample().item()
        
        # Debugging: Log action probabilities
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
