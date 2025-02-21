# src/tests/ai_vs_hardcoded.py

import os
import logging
import tkinter as tk
from tkinter import ttk
from tkinterdnd2 import TkinterDnD, DND_FILES
import threading
import torch
import numpy as np
from src.env.liars_deck_env_core import LiarsDeckEnv
from src import config

# Import hardcoded agents.
from src.model.hard_coded_agents import (
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    StrategicChallenger,
    RandomAgent,
    TableNonTableAgent,
    Classic
)

# Import evaluation helpers and conversion utilities.
from src.evaluation.evaluate_utils import (
    get_hidden_dim_from_state_dict,
    evaluate_agents
)

# Import the unified ModelFactory API.
from src.model.model_factory import ModelFactory

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s: %(message)s")
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
        resp_val = float(response_mapping.get(resp, 0))
        act_val = float(action_mapping.get(act, 0))
        features.append([resp_val, act_val, penalties, card_count])
    return features

class AgentBattlegroundGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Agent Battleground")
        self.root.geometry("1000x650")  # Wider and taller window
        
        self.loaded_models = {}
        self.hardcoded_agents = {
            "GreedySpammer": GreedyCardSpammer,
            "TableFirst": TableFirstConservativeChallenger,
            "Strategic": lambda name: StrategicChallenger(name, 3, 2),
            "Conservative": lambda name: TableFirstConservativeChallenger(name),
            "TableNonTableAgent": TableNonTableAgent,
            "Classic": Classic,
            "Random": RandomAgent
        }
        
        self.create_file_drop_zone()
        self.create_model_info_panel()
        self.create_ai_selection()
        self.create_control_buttons()
        self.create_progress_bar()  # New progress bar
        self.create_results_display()

        self.current_env = None  # Will be set when a match is run

    def get_hidden_dim_from_state_dict(self, state_dict, layer_prefix='fc1'):
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
        ttk.Button(frame, text="Start Battleground", command=self.start_battleground_thread).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame, text="Rounds:").pack(side=tk.LEFT, padx=5)
        self.rounds_var = tk.StringVar(value="20")
        self.rounds_spinbox = ttk.Spinbox(frame, from_=1, to=1000, textvariable=self.rounds_var, width=5)
        self.rounds_spinbox.pack(side=tk.LEFT, padx=5)

    def create_progress_bar(self):
        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(frame, text="Progress:").pack(side=tk.LEFT, padx=5)
        self.progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate")
        self.progress.pack(fill=tk.X, padx=5, pady=5)

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
        
        any_policy = next(iter(checkpoint["policy_nets"].values()))
        input_dim = any_policy['fc1.weight'].shape[1]
        if input_dim == 18:
            obs_version = 1
        elif input_dim in (16, 24, 26):
            obs_version = 2
        else:
            raise ValueError(f"Unknown input_dim {input_dim} for model {file_path}")
        
        uses_memory = True
        
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
                key = f"player_{i}"
                ai_agents[key] = {
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

    def start_battleground_thread(self):
        """Run start_battleground in a separate thread to keep UI responsive."""
        thread = threading.Thread(target=self.start_battleground)
        thread.daemon = True
        thread.start()

    def start_battleground(self):
        try:
            ai_agents = self.load_selected_agents()
            if not ai_agents:
                return

            rounds = int(self.rounds_var.get())
            results = {}
            total_matches = rounds * len(self.hardcoded_agents)
            self.progress['value'] = 0
            self.progress['maximum'] = total_matches
            progress_counter = 0

            for hc_name, hc_class in self.hardcoded_agents.items():
                wins = [0, 0, 0]  # [AI1 Wins, AI2 Wins, Hardcoded Wins]
                for _ in range(rounds):
                    winner = self.run_match(ai_agents, hc_class(hc_name))
                    if winner == "player_0":
                        wins[0] += 1
                    elif winner == "player_1":
                        wins[1] += 1
                    elif winner == "hardcoded_agent":
                        wins[2] += 1
                    else:
                        logger.warning(f"Unknown winner identifier: {winner}")
                    progress_counter += 1
                    self.progress['value'] = progress_counter
                    self.root.update_idletasks()
                results[hc_name] = wins

            self.display_results(results)

        except Exception as e:
            self.show_info(f"Error: {str(e)}")

    def run_match(self, ai_agents, hardcoded_agent):
        """
        Constructs a players dictionary for a 3-player match (2 AI and 1 hardcoded),
        then runs one episode using evaluate_agents and determines the winner.
        """
        env = LiarsDeckEnv(num_players=3, render_mode=None)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_env = env

        players_in_this_game = {}

        # AI Agents (player_0 and player_1)
        for key in ["player_0", "player_1"]:
            agent_data = ai_agents[key]
            hidden_dim = get_hidden_dim_from_state_dict(agent_data["policy_net"], "fc1")
            num_opponents = env.num_players - 1
            # Use the full input_dim from checkpoint so fc1 dimensions match.
            obs_dim = agent_data["input_dim"]
            policy_net = ModelFactory.create_policy_network(
                input_dim=obs_dim,
                hidden_dim=hidden_dim,
                output_dim=env.action_spaces[key].n
            )
            policy_net.load_state_dict(agent_data["policy_net"], strict=False)
            policy_net.to(device).eval()

            obp_state = agent_data["obp_model"]
            if obp_state is not None:
                obp_hidden_dim = get_hidden_dim_from_state_dict(obp_state, "fc1")
                obp_input_dim = obp_state["fc1.weight"].shape[1]
                if obp_input_dim == config.OPPONENT_INPUT_DIM + config.STRATEGY_DIM:
                    obp_model = ModelFactory.create_obp(
                        use_transformer_memory=True,
                        input_dim=config.OPPONENT_INPUT_DIM,
                        hidden_dim=obp_hidden_dim,
                        output_dim=2
                    )
                elif obp_input_dim == config.OPPONENT_INPUT_DIM:
                    obp_model = ModelFactory.create_obp(
                        use_transformer_memory=False,
                        input_dim=config.OPPONENT_INPUT_DIM,
                        hidden_dim=obp_hidden_dim,
                        output_dim=2
                    )
                else:
                    raise ValueError(f"Unexpected OBP input dimension: {obp_input_dim}")
                obp_model = ModelFactory.load_obp_state_dict(obp_model, obp_state)
                obp_model.to(device).eval()
            else:
                obp_model = None

            players_in_this_game[key] = {
                "policy_net": policy_net,  # use our wrapper here
                "obp_model": obp_model,
                "obs_version": agent_data["obs_version"],
                "rating": None,
                "uses_memory": agent_data["uses_memory"]
            }

        # Hardcoded agent as player_2.
        players_in_this_game["player_2"] = {
            "hardcoded_bot": True,
            "agent": hardcoded_agent,
            "obs_version": 2,
            "rating": None,
            "uses_memory": False
        }

        cumulative_wins, _, _, _, _ = evaluate_agents(env, device, players_in_this_game, episodes=1)

        winner = max(cumulative_wins, key=cumulative_wins.get)
        if winner in ["player_0", "player_1"]:
            return winner
        elif winner == "player_2":
            return "hardcoded_agent"
        else:
            return "unknown_agent"


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
