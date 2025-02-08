#!/usr/bin/env python
# src/tests/data_gen.py

import os
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
import torch
import numpy as np
import pickle

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.new_models import PolicyNetwork, OpponentBehaviorPredictor
from src import config

from src.model.hard_coded_agents import (
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    StrategicChallenger,
    RandomAgent,
    TableNonTableAgent
)

from src.evaluation.evaluate import run_obp_inference
from src.evaluation.evaluate_utils import (
    adapt_observation_for_version,
    get_hidden_dim_from_state_dict
)

# Import the opponent memory query helper from memory.py
from src.model.memory import get_opponent_memory

logging.basicConfig(level=logging.INFO)  # Set to DEBUG for detailed logs
logger = logging.getLogger("AgentBattleground")


class AgentBattlegroundGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Agent Battleground")
        self.root.geometry("900x700")
        
        # Dictionary to hold loaded PPO models.
        self.loaded_models = {}
        # Hardcoded bots remain available.
        self.hardcoded_agents = {
            "GreedySpammer": GreedyCardSpammer,
            "TableFirst": TableFirstConservativeChallenger,
            "Strategic": lambda name: StrategicChallenger(name, 3, 2),
            "Conservative": lambda name: TableFirstConservativeChallenger(name),
            "TableNonTableAgent": TableNonTableAgent,
            "Random": RandomAgent
        }
        
        # This list will accumulate training examples.
        # Each element is a tuple: (memory_segment, label)
        self.training_data = []

        self.create_file_drop_zone()
        self.create_model_info_panel()
        self.create_agent_selection_panels()
        self.create_control_buttons()
        self.create_results_display()

    def get_hidden_dim_from_state_dict(self, state_dict, layer_prefix='fc1'):
        """Extracts hidden dimension from model weights using the imported utility."""
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

    def create_agent_selection_panels(self):
        # Panel for selecting the Main PPO Agent.
        main_frame = ttk.LabelFrame(self.root, text="Main PPO Agent", padding=10)
        main_frame.pack(fill=tk.X, padx=10, pady=5)
        self.main_agent_selector = ttk.Combobox(main_frame, state="readonly", width=70)
        self.main_agent_selector.pack(fill=tk.X, padx=5, pady=5)
        
        # Panel for selecting Opponent PPO Agents.
        opp_frame = ttk.LabelFrame(self.root, text="Opponent PPO Agents (Select one or more)", padding=10)
        opp_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        self.opp_agent_list = tk.Listbox(opp_frame, selectmode=tk.MULTIPLE, height=5)
        self.opp_agent_list.pack(fill=tk.BOTH, padx=5, pady=5, expand=True)

    def create_control_buttons(self):
        frame = ttk.Frame(self.root)
        frame.pack(pady=10)
        ttk.Button(frame, text="Refresh Agents", command=self.update_agent_selectors).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Start Battleground with All Agents", command=self.start_battleground_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Save Training Data", command=self.save_training_data).pack(side=tk.LEFT, padx=5)

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
        
        # Determine the observation version based on the policy network's input dimension.
        any_policy = next(iter(checkpoint["policy_nets"].values()))
        input_dim = any_policy['fc1.weight'].shape[1]
        if input_dim == 18:
            obs_version = 1  # OBS_VERSION_1
        elif input_dim in (16, 24):
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
        # Build a list of display strings for each loaded PPO agent.
        agent_options = []
        for file_path, data in self.loaded_models.items():
            folder_name = os.path.basename(os.path.dirname(file_path))
            for agent_name in data["policy_nets"].keys():
                display_text = f"{folder_name} - {os.path.basename(file_path)} - {agent_name}"
                agent_options.append(display_text)
        # Update Main PPO Agent combobox.
        self.main_agent_selector['values'] = agent_options
        if agent_options:
            self.main_agent_selector.current(0)
        # Update Opponent PPO Agents listbox.
        self.opp_agent_list.delete(0, tk.END)
        for option in agent_options:
            self.opp_agent_list.insert(tk.END, option)

    def load_agent_from_string(self, display_text):
        """
        Given a display string in the format "folder - file - agent",
        return a dictionary with keys:
          - policy_net
          - obp_model
          - obs_version
          - input_dim
          - uses_memory
          - label (the agent name)
        """
        parts = display_text.split(" - ")
        if len(parts) != 3:
            raise ValueError("Invalid agent format")
        folder_name, file_name, agent_name = parts
        file_path_candidates = [p for p in self.loaded_models.keys() if os.path.basename(p) == file_name]
        if not file_path_candidates:
            raise ValueError(f"File for {file_name} not found among loaded models.")
        file_path = file_path_candidates[0]
        model_data = self.loaded_models[file_path]
        return {
            "policy_net": model_data["policy_nets"][agent_name],
            "obp_model": model_data["obp_model"],
            "obs_version": model_data["obs_version"],
            "input_dim": model_data["input_dim"],
            "uses_memory": model_data["uses_memory"],
            "label": agent_name
        }

    def start_battleground_all(self):
        """
        Run matches with all selected PPO agents (including the main agent)
        and all available hardcoded bots. Matches are played in a single game
        with total players = (# PPO agents selected + main agent) + (# hardcoded bots).
        Training data is extracted from all agents, and win counts are recorded.
        Matches are repeated until every agent has accumulated at least a target
        number of training segments.
        """
        try:
            # Load selected PPO agents from the Opponent list.
            selected_indices = self.opp_agent_list.curselection()
            ppo_agents = []
            for idx in selected_indices:
                ppo_agents.append(self.load_agent_from_string(self.opp_agent_list.get(idx)))
            # Also load the Main PPO Agent.
            main_selection = self.main_agent_selector.get()
            if not main_selection:
                self.show_info("Select a Main PPO Agent")
                return
            main_agent = self.load_agent_from_string(main_selection)
            # Ensure the main agent is included (avoid duplicates).
            if main_agent not in ppo_agents:
                ppo_agents.insert(0, main_agent)
            
            # Get all hardcoded agents.
            hardcoded_agents = []
            for label, agent_class in self.hardcoded_agents.items():
                # Create an instance by calling the class with its label.
                agent_instance = agent_class(label)
                hardcoded_agents.append({
                    "type": "hardcoded",
                    "agent_instance": agent_instance,
                    "label": label
                })
            
            # Build a combined dictionary of agents.
            # Assign player IDs in order: first all PPO agents, then all hardcoded bots.
            all_agents = {}
            player_id = 0
            for agent in ppo_agents:
                agent["type"] = "ppo"
                all_agents[f"player_{player_id}"] = agent
                player_id += 1
            for agent in hardcoded_agents:
                all_agents[f"player_{player_id}"] = agent
                player_id += 1
            
            total_players = len(all_agents)
            target_segments = 20  # Target training segments per agent.
            
            # Initialize win counts for each agent label.
            wins = {}
            for pid, agent in all_agents.items():
                wins[agent["label"]] = 0
            
            # Helper: count segments per label.
            def segments_count(label):
                return sum(1 for seg, l in self.training_data if l == label)
            
            all_labels = [agent["label"] for agent in all_agents.values()]
            match_count = 0
            while any(segments_count(lbl) < target_segments for lbl in all_labels):
                match_count += 1
                winner = self.run_match(all_agents)
                # Update win counts for each winner.
                if isinstance(winner, list):
                    for w in winner:
                        wins[w] += 1
                else:
                    wins[winner] += 1
                if match_count % 10 == 0:
                    logger.info(f"After {match_count} matches, training segments: " +
                                ", ".join(f"{lbl}: {segments_count(lbl)}" for lbl in all_labels))
            self.display_results(wins)
            self.show_info(f"Battleground complete. Generated {len(self.training_data)} training examples from {match_count} matches.")
        except Exception as e:
            self.show_info(f"Error: {str(e)}")

    def run_match(self, all_agents):
        """
        Runs a single match with a combined dictionary of agents.
        Agents with type "ppo" use their policy network and OBP model,
        while agents with type "hardcoded" use their play_turn() method.
        After the match, training examples are extracted from each agent's
        opponent memory (using get_full_memory) and segmented.
        Returns the label(s) of the winning agent(s).
        """
        total_players = len(all_agents)
        env = LiarsDeckEnv(num_players=total_players, render_mode=None)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # For PPO agents, initialize their networks.
        policy_nets = {}
        obp_models = {}
        obs_versions = {}
        input_dims = {}
        uses_memories = {}
        for pid, agent in all_agents.items():
            if agent.get("type") == "ppo":
                obs_versions[pid] = agent["obs_version"]
                input_dims[pid] = agent["input_dim"]
                uses_memories[pid] = agent["uses_memory"]
                policy_net = PolicyNetwork(
                    input_dim=agent["input_dim"],
                    hidden_dim=self.get_hidden_dim_from_state_dict(agent["policy_net"]),
                    output_dim=env.action_spaces[pid].n,
                    use_lstm=True,
                    use_layer_norm=True
                )
                policy_net.load_state_dict(agent["policy_net"])
                policy_net.to(device).eval()
                policy_nets[pid] = policy_net

                obp_model_state = agent["obp_model"]
                if obp_model_state:
                    obp_input_dim = 5 if agent["obs_version"] == 1 else 4
                    obp_model = OpponentBehaviorPredictor(
                        input_dim=obp_input_dim,
                        hidden_dim=config.OPPONENT_HIDDEN_DIM,
                        output_dim=2
                    )
                    obp_model.load_state_dict(obp_model_state)
                    obp_model.to(device).eval()
                    obp_models[pid] = obp_model
                else:
                    obp_models[pid] = None
        
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
                # Hardcoded agent.
                hardcoded_agent = all_agents[current_agent]["agent_instance"]
                action = hardcoded_agent.play_turn(
                    obs[current_agent],
                    info['action_mask'],
                    env.table_card
                )
                if not isinstance(action, int):
                    logger.error(f"Hardcoded agent {all_agents[current_agent]['label']} returned non-integer action: {action}")
                    raise ValueError(f"Hardcoded agent {all_agents[current_agent]['label']} returned non-integer action: {action}")
            env.step(action)
        
        # Determine winner(s) based on rewards.
        max_reward = max(env.rewards.values())
        winners = [pid for pid, r in env.rewards.items() if r == max_reward]
        winner_labels = [all_agents[pid]["label"] for pid in winners]
        
        # Extract training data from each agent's opponent memory.
        for pid in env.agents:
            label = all_agents[pid]["label"]
            memory_obj = get_opponent_memory(pid)
            full_memory = []
            for opp in list(memory_obj.memory.keys()):
                full_memory.extend(memory_obj.get_full_memory(opp))
            if len(full_memory) >= 5:
                seg_length = len(full_memory) // 5
                for i in range(5):
                    segment = full_memory[i * seg_length: (i + 1) * seg_length] if i < 4 else full_memory[i * seg_length:]
                    if segment:
                        self.training_data.append((segment, label))
            else:
                if full_memory:
                    self.training_data.append((full_memory, label))
            memory_obj.memory.clear()
            memory_obj.aggregates.clear()
        return winner_labels if len(winner_labels) > 1 else winner_labels[0]

    def choose_action(self, agent_id, policy_net, obp_model, observation, action_mask, device, num_players, obs_version, input_dim, uses_memory):
        """Selects an action using the policy network (with OBP integration)."""
        converted_obs = adapt_observation_for_version(observation, num_players, obs_version)
        logging.debug(f"Converted observation (length {len(converted_obs)}): {converted_obs}")

        obp_probs = run_obp_inference(obp_model, converted_obs, device, num_players, obs_version)
        logging.debug(f"OBP probabilities: {obp_probs}")

        if obs_version == 2 and uses_memory:
            required_mem_dim = input_dim - (len(converted_obs) + len(obp_probs))
            mem_features = np.zeros(required_mem_dim, dtype=np.float32) if required_mem_dim > 0 else np.array([], dtype=np.float32)
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
            action_probs, _ = policy_net(observation_tensor)
        
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32).to(device)
        masked_probs = action_probs * mask_tensor
        
        # Fix: Use .item() on the sum to get a Python number.
        if masked_probs.sum().item() == 0:
            masked_probs = mask_tensor / mask_tensor.sum().item()
        else:
            masked_probs /= masked_probs.sum()
        
        m = torch.distributions.Categorical(masked_probs)
        action = m.sample().item()
        logging.debug(f"Action probabilities: {masked_probs.cpu().numpy()}")
        logging.debug(f"Selected action: {action}")
        if action_mask[action] == 6:
            logging.debug("Challenge Action Selected")
        return action

    def save_training_data(self):
        """Append the accumulated training data to file instead of overwriting it."""
        file_path = os.path.join(os.getcwd(), "opponent_training_data.pkl")
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    existing_data = pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading existing training data: {e}")
                existing_data = []
        else:
            existing_data = []
        combined_data = existing_data + self.training_data
        try:
            with open(file_path, "wb") as f:
                pickle.dump(combined_data, f)
            self.show_info(f"Training data saved to {file_path} (appended {len(self.training_data)} new samples)")
            self.training_data.clear()
        except Exception as e:
            self.show_info(f"Error saving training data: {str(e)}")

    def display_results(self, wins):
        """Display win counts per agent label."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        header = "Agent Label                | Wins\n"
        self.results_text.insert(tk.END, header)
        self.results_text.insert(tk.END, "-" * 40 + "\n")
        for label, count in wins.items():
            line = f"{label:25} | {count:^5}\n"
            self.results_text.insert(tk.END, line)
        self.results_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = AgentBattlegroundGUI(root)
    root.mainloop()
