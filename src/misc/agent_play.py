# src/misc/agent_play.py

import logging
import tkinter as tk
from tkinter import ttk
from tkinterdnd2 import TkinterDnD, DND_FILES
import torch
import numpy as np
import os

from src.env.liars_deck_env_core import LiarsDeckEnv

# First try to import the new models.
try:
    from src.model.new_models import PolicyNetwork, OpponentBehaviorPredictor
except ImportError:
    from src.model.models import PolicyNetwork, OpponentBehaviorPredictor

# Explicitly import the legacy models so we can use them when needed.
import importlib
legacy_module = importlib.import_module("src.model.models")
legacy_PolicyNetwork = legacy_module.PolicyNetwork
legacy_OpponentBehaviorPredictor = legacy_module.OpponentBehaviorPredictor

# Import evaluation utilities for observation conversion and hidden dim extraction.
from src.eval.evaluate_utils import adapt_observation_for_version, get_hidden_dim_from_state_dict
from src.eval.evaluate import run_obp_inference  # New import for OBP inference
from src import config


class AgentManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Liars Deck Agent Manager")
        self.root.geometry("1000x800")
        
        self.loaded_models = {}
        self.selected_agents = {}
        
        self.create_file_drop_zone()
        self.create_model_info_panel()
        self.create_player_selection()
        self.create_control_buttons()
        
        # Debug state
        self.drop_counter = 0
        logging.debug("Application initialized")

    def create_file_drop_zone(self):
        frame = ttk.LabelFrame(self.root, text="Model Files", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.file_list = tk.Listbox(frame, height=4, selectmode=tk.SINGLE)
        self.file_list.pack(fill=tk.X)
        
        drop_label = ttk.Label(frame, text="Drag and drop .pth files here")
        drop_label.pack(pady=5)

        # Register drop target on multiple elements.
        for widget in [frame, self.file_list, drop_label]:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", self.on_file_drop)
        
        logging.debug("Created file drop zone")

    def create_model_info_panel(self):
        frame = ttk.LabelFrame(self.root, text="Model Contents", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.info_text = tk.Text(frame, wrap=tk.WORD, state=tk.DISABLED)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        logging.debug("Created info panel")

    def create_player_selection(self):
        frame = ttk.LabelFrame(self.root, text="Game Configuration", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Configure grid layout.
        frame.columnconfigure(1, weight=1)
        
        ttk.Label(frame, text="Number of Players:").grid(row=0, column=0, sticky=tk.W)
        self.player_count = ttk.Combobox(frame, values=[3, 4], state="readonly")
        self.player_count.current(0)
        self.player_count.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        
        self.agent_selectors = {}
        for i in range(4):
            ttk.Label(frame, text=f"Player {i+1}:").grid(row=i+1, column=0, sticky=tk.W, pady=2)
            self.agent_selectors[i] = ttk.Combobox(frame, state="readonly", width=50)
            self.agent_selectors[i].grid(row=i+1, column=1, sticky=tk.EW, padx=5, pady=2)
        
        for i in range(4):
            frame.rowconfigure(i+1, weight=1)
        
        self.player_count.bind("<<ComboboxSelected>>", self.update_agent_selectors)
        logging.debug("Created player selection")

    def create_control_buttons(self):
        frame = ttk.Frame(self.root)
        frame.pack(pady=10)
        ttk.Button(frame, text="Refresh Agents", command=self.update_agent_selectors).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Start Game", command=self.start_game).pack(side=tk.LEFT, padx=5)
        logging.debug("Created control buttons")

    def on_file_drop(self, event):
        try:
            self.drop_counter += 1
            logging.debug(f"\nDrop event #{self.drop_counter} received")
            logging.debug(f"Raw event data: {repr(event.data)}")
            file_path = event.data.strip()

            if os.name == 'nt':
                file_path = file_path.replace("{", "").replace("}", "")
                if file_path.startswith('"') and file_path.endswith('"'):
                    file_path = file_path[1:-1]
            else:
                if file_path.startswith("file://"):
                    file_path = file_path[7:]
            file_path = os.path.normpath(file_path)
            logging.debug(f"Processed file path: {file_path}")

            if not file_path.lower().endswith(".pth"):
                logging.warning(f"Rejected non-pth file: {file_path}")
                self.show_info("Only .pth files are supported")
                return

            if not os.path.exists(file_path):
                logging.error(f"File not found: {file_path}")
                self.show_info(f"File not found: {os.path.basename(file_path)}")
                return

            if file_path in self.loaded_models:
                logging.debug(f"File already loaded: {file_path}")
                self.show_info(f"Already loaded: {os.path.basename(file_path)}")
                return

            self.load_model(file_path)
            self.file_list.insert(tk.END, os.path.basename(file_path))
            self.update_agent_selectors()
            self.show_info(f"Loaded: {os.path.basename(file_path)}")
            logging.info(f"Successfully loaded: {file_path}")

        except Exception as e:
            logging.exception("Error in on_file_drop")
            self.show_info(f"Drop error: {str(e)}")

    def load_model(self, file_path):
        try:
            logging.debug(f"Attempting to load model: {file_path}")
            checkpoint = torch.load(file_path, map_location="cpu")
            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint is not a dictionary")
            required_keys = ["policy_nets", "obp_model"]
            missing_keys = [k for k in required_keys if k not in checkpoint]
            if missing_keys:
                raise ValueError(f"Missing required keys: {missing_keys}")
            logging.debug(f"Model contains agents: {list(checkpoint['policy_nets'].keys())}")
            self.loaded_models[file_path] = {
                "policy_nets": checkpoint["policy_nets"],
                "obp_model": checkpoint["obp_model"]
            }
            logging.info(f"Model loaded successfully: {file_path}")
        except Exception as e:
            logging.error(f"Failed to load model {file_path}: {str(e)}")
            self.show_info(f"Error loading {os.path.basename(file_path)}:\n{str(e)}")
            raise

    def show_info(self, message):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, message)
        self.info_text.config(state=tk.DISABLED)
        logging.debug(f"Info panel updated: {message}")

    def update_agent_selectors(self, event=None):
        try:
            logging.debug("Updating agent selectors")
            agent_options = []
            for file_path, data in self.loaded_models.items():
                folder_name = os.path.basename(os.path.dirname(file_path))
                for agent_name in data["policy_nets"].keys():
                    display_text = f"{folder_name} - {os.path.basename(file_path)} - {agent_name}"
                    agent_options.append(display_text)
            num_players = int(self.player_count.get())
            logging.debug(f"Updating for {num_players} players with {len(agent_options)} options")
            for i in range(4):
                if i < num_players:
                    self.agent_selectors[i]["values"] = agent_options
                    self.agent_selectors[i].state(["!disabled"])
                    logging.debug(f"Updated selector {i} with {len(agent_options)} options")
                else:
                    self.agent_selectors[i].set("")
                    self.agent_selectors[i].state(["disabled"])
        except Exception as e:
            logging.exception("Error in update_agent_selectors")
            self.show_info(f"Selection error: {str(e)}")

    def start_game(self):
        logging.debug("Attempting to start game")
        try:
            num_players = int(self.player_count.get())
            agents = {}
            for i in range(num_players):
                selection = self.agent_selectors[i].get()
                if not selection:
                    raise ValueError(f"No agent selected for Player {i+1}")
                parts = selection.split(" - ")
                if len(parts) != 3:
                    raise ValueError(f"Invalid selection format: {selection}")
                file_name = parts[1]
                agent_name = parts[2]
                file_path_candidates = [p for p in self.loaded_models.keys() if os.path.basename(p) == file_name]
                if not file_path_candidates:
                    raise ValueError(f"File for {file_name} not found among loaded models.")
                file_path = file_path_candidates[0]
                agents[f"player_{i}"] = {
                    "policy_net": self.loaded_models[file_path]["policy_nets"][agent_name],
                    "obp_model": self.loaded_models[file_path]["obp_model"]
                }
            logging.info(f"Starting game with {num_players} players")
            self.root.after(100, lambda: self.play_game(agents, num_players))
        except Exception as e:
            logging.exception("Game start failed")
            self.show_info(f"Start error: {str(e)}")

    def play_game(self, agents, num_players):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = LiarsDeckEnv(num_players=num_players, render_mode="human")
        
        # Dictionaries to store the reconstructed models and observation properties.
        policy_nets = {}
        obp_models = {}
        obs_versions = {}    # Maps agent_id to observation version (1 or 2)
        input_dims = {}      # Maps agent_id to expected input dimension of the policy network
        uses_memories = {}   # Maps agent_id to a boolean indicating whether the policy uses memory
        
        for agent_id, data in agents.items():
            # Determine observation version and input dimension from the policy checkpoint.
            policy_state = data["policy_net"]
            actual_input_dim = policy_state["fc1.weight"].shape[1]
            if actual_input_dim == 18:
                obs_version = 1
            elif actual_input_dim in (16, 24):
                obs_version = 2
            else:
                raise ValueError(f"Unknown input dimension {actual_input_dim} for {agent_id}")
            output_dim = env.action_spaces[agent_id].n
            try:
                hidden_dim = get_hidden_dim_from_state_dict(policy_state)
            except ValueError as e:
                logging.error(f"Error loading {agent_id}: {str(e)}")
                self.show_info(f"Invalid model for {agent_id}: {str(e)}")
                return

            # Reconstruct the policy network – use the new model if the checkpoint has extra keys;
            # otherwise, use the legacy model.
            if "fc4.weight" in policy_state:
                policy_net = PolicyNetwork(
                    input_dim=actual_input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    use_lstm=True,
                    use_dropout=True,
                    use_layer_norm=True
                )
            else:
                policy_net = legacy_PolicyNetwork(
                    input_dim=actual_input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    use_lstm=True,
                    use_dropout=True,
                    use_layer_norm=True
                )
            policy_net.load_state_dict(policy_state)
            policy_net.to(device)
            policy_net.eval()
            policy_nets[agent_id] = policy_net
            obs_versions[agent_id] = obs_version
            input_dims[agent_id] = actual_input_dim
            uses_memories[agent_id] = ("fc4.weight" in policy_state)

            # Reconstruct OBP model if provided.
            obp_model = None
            if data["obp_model"]:
                obp_input_dim = 5 if obs_version == 1 else 4
                obp_hidden_dim = get_hidden_dim_from_state_dict(data["obp_model"], layer_prefix='fc1')
                if "fc3.weight" in data["obp_model"]:
                    obp_model = OpponentBehaviorPredictor(
                        input_dim=obp_input_dim,
                        hidden_dim=obp_hidden_dim,
                        output_dim=2
                    )
                else:
                    obp_model = legacy_OpponentBehaviorPredictor(
                        input_dim=obp_input_dim,
                        hidden_dim=obp_hidden_dim,
                        output_dim=2
                    )
                obp_model.load_state_dict(data["obp_model"])
                obp_model.to(device)
                obp_model.eval()
            obp_models[agent_id] = obp_model

        env.reset()
        while env.agent_selection is not None:
            current_agent = env.agent_selection
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                env.step(None)
                continue
            # Get the raw observation (assumed to be a numpy array)
            observation_array = obs[current_agent]
            # Adapt the observation to the version expected by the agent.
            converted_obs = adapt_observation_for_version(observation_array, num_players, obs_versions[current_agent])
            action_mask = info['action_mask']
            # Pass the additional parameters (input dimension and memory flag) to choose_action.
            action = self.choose_action(
                policy_nets[current_agent],
                obp_models[current_agent],
                converted_obs,
                device,
                num_players,
                action_mask,
                obs_versions[current_agent],
                input_dims[current_agent],
                uses_memories[current_agent]
            )
            logging.info(f"{current_agent} chose action {action}")
            env.step(action)
        env.close()

    def choose_action(self, policy_net, obp_model, converted_obs, device, num_players, action_mask, obs_version, input_dim, uses_memory):
        """
        Chooses an action with OBP integration and memory padding if needed.
        """
        # Run OBP inference to get opponent behavior probabilities.
        obp_probs = run_obp_inference(
            obp_model,
            converted_obs,
            device,
            num_players,
            obs_version
        )
        # If the model uses memory (and obs_version is 2), pad with zeros for memory features.
        if obs_version == 2 and uses_memory:
            required_mem_dim = input_dim - (len(converted_obs) + len(obp_probs))
            if required_mem_dim > 0:
                mem_features = np.zeros(required_mem_dim, dtype=np.float32)
            else:
                mem_features = np.array([], dtype=np.float32)
            final_obs = np.concatenate([converted_obs, np.array(obp_probs, dtype=np.float32), mem_features], axis=0)
        else:
            final_obs = np.concatenate([converted_obs, np.array(obp_probs, dtype=np.float32)], axis=0)

        expected_dim = input_dim  # Expected input dimension for the policy network
        if len(final_obs) != expected_dim:
            raise ValueError(f"Expected observation dimension {expected_dim}, got {len(final_obs)}")
        observation_tensor = torch.tensor(final_obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs, _ = policy_net(observation_tensor)
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32).to(device)
        masked_probs = action_probs * mask_tensor
        if masked_probs.sum() == 0:
            masked_probs = mask_tensor / mask_tensor.sum()
        else:
            masked_probs /= masked_probs.sum()
        m = torch.distributions.Categorical(masked_probs)
        action = m.sample().item()
        return action

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root = TkinterDnD.Tk()
    app = AgentManagerApp(root)
    root.mainloop()
