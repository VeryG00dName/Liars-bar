# src/misc/play_vs_ai.py

import os
import logging
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
import torch
import numpy as np
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.new_models import PolicyNetwork, OpponentBehaviorPredictor
from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PlayVsAI")

class PlayVsAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Play vs AI")
        self.root.geometry("800x500")
        
        self.loaded_models = {}
        self.selected_agents = {}
        
        self.create_file_drop_zone()
        self.create_model_info_panel()
        self.create_ai_selection()
        self.create_control_buttons()
        
        self.game_window = None
        self.current_env = None

    def get_hidden_dim_from_state_dict(self, state_dict, layer_prefix='fc1'):
        """Extracts hidden dimension from model weights"""
        weight_key = f"{layer_prefix}.weight"
        if weight_key in state_dict:
            return state_dict[weight_key].shape[0]
        for key in state_dict.keys():
            if key.endswith('.weight') and ('fc' in key or 'layer' in key):
                return state_dict[key].shape[0]
        raise ValueError(f"Cannot determine hidden_dim from state_dict for layer prefix '{layer_prefix}'")

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
        ttk.Button(frame, text="Start Game", command=self.start_game).pack(side=tk.LEFT, padx=5)

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
        self.loaded_models[file_path] = {
            "policy_nets": checkpoint["policy_nets"],
            "obp_model": checkpoint["obp_model"]
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
            self.agent_selectors[i].state(["!disabled"])

    def start_game(self):
        try:
            agents = {}
            for i in range(2):
                selection = self.agent_selectors[i].get()
                if not selection:
                    raise ValueError(f"Select AI Agent {i+1}")
                parts = selection.split(" - ")
                if len(parts) != 3:
                    raise ValueError("Invalid agent format")
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
            self.root.after(100, lambda: self.play_game(agents))
        except Exception as e:
            self.show_info(f"Error: {str(e)}")

    def show_game_result(self):
        # Get final rewards
        rewards = self.current_env.rewards
        max_reward = max(rewards.values())
        winners = [agent for agent, reward in rewards.items() if reward == max_reward]

        # Create result message
        result_text = "Game Results:\n"
        for agent, reward in rewards.items():
            result_text += f"{agent}: {reward}\n"
        
        result_text += "\nWinner(s):\n" + "\n".join(winners)

        # Show popup
        messagebox.showinfo("Game Over", result_text)
        self.current_env = None

    def play_game(self, agents):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_players = 3  # Assuming player vs 2 AI agents
        self.current_env = LiarsDeckEnv(num_players=num_players, render_mode="player")
        
        # Initialize networks
        policy_nets = {}
        obp_models = {}
        for agent_id, agent_data in agents.items():
            # Policy Network
            policy_state = agent_data["policy_net"]
            try:
                hidden_dim = self.get_hidden_dim_from_state_dict(policy_state)
            except ValueError as e:
                self.show_info(f"Invalid policy model for {agent_id}: {str(e)}")
                return
            
            # Assuming config.INPUT_DIM is set to observation_space.shape[0] + num_opponents
            # For num_players=3, num_opponents=2
            observation_space = self.current_env.observation_spaces[agent_id]
            num_opponents = num_players - 1
            input_dim = observation_space.shape[0] + num_opponents  # Append OBP probs
            output_dim = self.current_env.action_spaces[agent_id].n
            
            policy_net = PolicyNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                use_lstm=True,
                use_layer_norm=True
            )
            policy_net.load_state_dict(policy_state)
            policy_net.to(device).eval()
            policy_nets[agent_id] = policy_net
            
            # Opponent Behavior Predictor (OBP) Model
            obp_model_state = agent_data["obp_model"]
            if obp_model_state:
                obp_input_dim = config.OPPONENT_INPUT_DIM  # Should be 4
                obp_hidden_dim = config.OPPONENT_HIDDEN_DIM  # Ensure this aligns with your model architecture
                obp_model = OpponentBehaviorPredictor(
                    input_dim=obp_input_dim,
                    hidden_dim=obp_hidden_dim,
                    output_dim=2
                )
                obp_model.load_state_dict(obp_model_state)
                obp_model.to(device).eval()
                obp_models[agent_id] = obp_model
            else:
                obp_models[agent_id] = None  # Handle cases where OBP is not present

        # Game loop
        self.current_env.reset()
        while self.current_env.agent_selection is not None:
            current_agent = self.current_env.agent_selection
            obs, reward, termination, truncation, info = self.current_env.last()
            
            if termination or truncation:
                self.current_env.step(None)
                continue

            if current_agent in policy_nets:
                # AI Agent's turn
                observation = obs[current_agent]
                action_mask = info['action_mask']
                action = self.choose_action(
                    policy_nets[current_agent],
                    obp_models[current_agent],
                    observation,
                    device,
                    num_players=num_players,
                    action_mask=action_mask
                )
            else:
                # Human player's turn
                self.current_env.render('player')
                action = self.get_human_action()
            
            self.current_env.step(action)
        self.show_game_result()
        self.current_env.close()

    def choose_action(self, policy_net, obp_model, observation, device, num_players, action_mask):
        """Full action selection with OBP integration"""
        num_opponents = num_players - 1
        opp_feature_dim = config.OPPONENT_INPUT_DIM  # Should be 4
        
        # Extract opponent features from observation
        # The opponent features start after hand_vector_length + 1 + num_players
        hand_vector_length = 2
        last_action_val_length = 1
        active_players_length = num_players

        opp_features_start = hand_vector_length + last_action_val_length + active_players_length
        opp_features_end = opp_features_start + opp_feature_dim * num_opponents

        opponent_features = observation[opp_features_start:opp_features_end]
        opponent_features = opponent_features.reshape(num_opponents, opp_feature_dim)

        # Run OBP inference
        obp_probs = []
        for opp_feat in opponent_features:
            opp_feat_tensor = torch.tensor(opp_feat, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if obp_model:
                    logits = obp_model(opp_feat_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    bluff_prob = probs[0, 1].item()  # Probability of "bluff" class
                else:
                    bluff_prob = 0.0  # Default if OBP is not available
            obp_probs.append(bluff_prob)

        # Append OBP predictions to observation
        final_obs = np.concatenate([observation, np.array(obp_probs, dtype=np.float32)], axis=0)
        
        # Optional: Add assertion to verify input dimensions
        expected_dim = config.INPUT_DIM
        actual_dim = final_obs.shape[0]
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
        if action == 6:
            print("Challenge")
        return action

    def get_human_action(self):
        action_window = tk.Toplevel(self.root)
        action_window.title("Your Turn")

        # This variable will store the selected action (0-6)
        action_var = tk.IntVar(value=-1)

        def select_action(action_value):
            action_var.set(action_value)
            action_window.destroy()

        # Define the actions and their labels
        actions = [
            (0, "Play 1 Table Card (Action 0)"),
            (1, "Play 2 Table Cards (Action 1)"),
            (2, "Play 3 Table Cards (Action 2)"),
            (3, "Play 1 Non-Table Card (Action 3)"),
            (4, "Play 2 Non-Table Cards (Action 4)"),
            (5, "Play 3 Non-Table Cards (Action 5)"),
            (6, "Challenge (Action 6)")
        ]

        # Create a button for each action
        for action_value, label in actions:
            btn = ttk.Button(action_window, text=label,
                            command=lambda val=action_value: select_action(val))
            btn.pack(padx=10, pady=5, fill=tk.X)

        # Wait for the user to select an action
        action_window.wait_window()

        # Return the action that was selected
        return action_var.get()

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = PlayVsAIGUI(root)
    root.mainloop()
