# src/tests/data_gen.py

from collections import Counter
import os
import logging
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import traceback
from tkinterdnd2 import TkinterDnD, DND_FILES
import torch
import torch.nn.functional as F
import numpy as np
import random
import pickle

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.new_models import PolicyNetwork, OpponentBehaviorPredictor
from src import config

from src.model.hard_coded_agents import (
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    SelectiveTableConservativeChallenger,
    StrategicChallenger,
    RandomAgent,
    TableNonTableAgent,
    Classic
)

from src.eval.evaluate_utils import (
    adapt_observation_for_version,
    get_hidden_dim_from_state_dict,
    get_opponent_memory_embedding,
    run_obp_inference,
    run_obp_inference_tournament
)

# Import ModelFactory for OBP creation.
from src.model.model_factory import ModelFactory

# Import the opponent memory query helper from memory.py
from src.model.memory import get_opponent_memory

# New parameters for early culling:
viable_segment_threshold = 5
game_threshold = 50

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentBattleground")


class AgentBattlegroundGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Agent Battleground")
        self.root.geometry("900x750")
        
        self.loaded_models = {}
        self.hardcoded_agents = {
            # We still keep all definitions here in case you want them later,
            # but we'll only use "Conservative" in the battleground loop.
            "GreedySpammer": GreedyCardSpammer,
            "TableFirst": TableFirstConservativeChallenger,
            "Strategic": lambda name: StrategicChallenger(name, 3, 2),
            "Conservative": lambda name: SelectiveTableConservativeChallenger(name),
            "TableNonTableAgent": TableNonTableAgent,
            "Classic": Classic,
            "Random": RandomAgent
        }
        
        self.training_data = []
        self.strategy_transformer = None
        self.current_env = None
        self.games_since_last_collection = {}
        self.target_segments = 500  # Default value; also set via parameters box
        
        # New: maintain a set of culled agents (by label)
        self.culled_agents = set()

        self.create_file_drop_zone()
        self.create_model_info_panel()
        self.create_ai_selection()
        self.create_match_options()
        self.create_parameters_box()
        self.create_progress_and_culling_controls()
        self.create_control_buttons()
        self.create_results_display()
        self.create_summary_button()

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
        frame = ttk.LabelFrame(self.root, text="PPO Agents Selection", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        self.agent_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, width=50)
        self.agent_listbox.pack(fill=tk.X, padx=5, pady=5)

    def create_match_options(self):
        frame = ttk.LabelFrame(self.root, text="Match Options", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        self.include_hardcoded = tk.BooleanVar(value=True)
        self.include_ppo = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Include Matches vs. Hardcoded Opponents", variable=self.include_hardcoded).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(frame, text="Include Matches among PPO Agents", variable=self.include_ppo).pack(anchor=tk.W, pady=2)

    def create_parameters_box(self):
        frame = ttk.LabelFrame(self.root, text="Parameters", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        target_label = ttk.Label(frame, text="Target Segments per Agent:")
        target_label.pack(side=tk.LEFT, padx=5)
        self.target_segments_var = tk.StringVar(value="500")
        target_entry = ttk.Entry(frame, textvariable=self.target_segments_var, width=10)
        target_entry.pack(side=tk.LEFT)
        target_entry.bind("<FocusOut>", self.update_target_segments)

    def update_target_segments(self, event):
        try:
            self.target_segments = int(self.target_segments_var.get())
        except ValueError:
            self.show_info("Invalid target segments value, using default of 500.")
            self.target_segments = 500

    def create_progress_and_culling_controls(self):
        # New frame for progress bar, dropdown to select an agent, and a button to cull the agent.
        frame = ttk.LabelFrame(self.root, text="Agent Progress & Culling", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(frame, text="Select Agent:").pack(side=tk.LEFT, padx=5)
        self.agent_combobox = ttk.Combobox(frame, state="readonly")
        self.agent_combobox.pack(side=tk.LEFT, padx=5)
        self.cull_button = ttk.Button(frame, text="Cull Agent", command=self.cull_selected_agent)
        self.cull_button.pack(side=tk.LEFT, padx=5)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.update_agent_combobox()

    def update_agent_combobox(self):
        # Only include agents that are not culled.
        agents = self.load_selected_agents()
        if agents:
            labels = [data["label"] for data in agents.values() if data["label"] not in self.culled_agents]
            self.agent_combobox["values"] = labels
            if labels:
                self.agent_combobox.current(0)
                self.update_progress_bar()
            else:
                self.agent_combobox.set("")
                self.progress_var.set(0)
        else:
            self.agent_combobox["values"] = []

    def update_progress_bar(self):
        selected = self.agent_combobox.get()
        if not selected:
            self.progress_var.set(0)
            return
        current_count = sum(1 for seg, lbl in self.training_data if lbl == selected)
        percentage = (current_count / self.target_segments) * 100
        self.progress_var.set(min(percentage, 100))

    def cull_selected_agent(self):
        # Instead of deleting training samples, flag the agent as culled.
        selected = self.agent_combobox.get()
        if not selected:
            self.show_info("No agent selected for culling.")
            return
        self.culled_agents.add(selected)
        self.show_info(f"Agent {selected} has been culled and will no longer generate new samples.")
        self.update_agent_combobox()

    def create_control_buttons(self):
        frame = ttk.Frame(self.root)
        frame.pack(pady=10)
        ttk.Button(frame, text="Refresh Agents", command=self.refresh_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Start Battleground", command=self.start_battleground_async).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Save Training Data", command=self.save_training_data).pack(side=tk.LEFT, padx=5)

    def refresh_all(self):
        self.update_agent_selectors()
        self.update_agent_combobox()

    def create_summary_button(self):
        frame = ttk.Frame(self.root)
        frame.pack(pady=5)
        ttk.Button(frame, text="Print Data Summary", command=self.print_data_summary).pack(side=tk.LEFT, padx=5)

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
            self.update_agent_combobox()
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
        self.agent_listbox.delete(0, tk.END)
        for option in agent_options:
            self.agent_listbox.insert(tk.END, option)

    def print_data_summary(self):
        total_samples = len(self.training_data)
        label_counts = Counter(label for _, label in self.training_data)
        lengths = [len(seg) for seg, _ in self.training_data]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        min_length = min(lengths) if lengths else 0
        max_length = max(lengths) if lengths else 0
        summary = (
            "----- Training Data Summary -----\n"
            f"Total training samples: {total_samples}\n"
            "Label distribution:\n"
        )
        for label, count in label_counts.items():
            summary += f"  {label}: {count} samples\n"
        summary += f"Average sequence length: {avg_length:.2f}\n"
        summary += f"Minimum sequence length: {min_length}\n"
        summary += f"Maximum sequence length: {max_length}\n"
        summary += "First 5 samples:\n"
        for i, (seg, label) in enumerate(self.training_data[:5]):
            summary += f"  Sample {i}: {seg} -> {label}\n"
        summary += "---------------------------------\n"
        print(summary)
        self.show_info(summary)

    def load_selected_agents(self):
        selections = self.agent_listbox.curselection()
        if not selections:
            self.show_info("Please select at least one PPO agent.")
            return None
        ai_agents = {}
        all_options = self.agent_listbox.get(0, tk.END)
        for i, idx in enumerate(selections):
            selection = all_options[idx]
            parts = selection.split(" - ")
            if len(parts) != 3:
                self.show_info(f"Invalid agent format: '{selection}'")
                continue
            folder_name, file_name, agent_name = parts
            file_path_candidates = [p for p in self.loaded_models.keys() if os.path.basename(p) == file_name]
            if not file_path_candidates:
                self.show_info(f"File for {file_name} not found among loaded models.")
                continue
            file_path = file_path_candidates[0]
            model_data = self.loaded_models[file_path]
            if agent_name not in model_data["policy_nets"]:
                available_keys = list(model_data["policy_nets"].keys())
                self.show_info(f"Agent '{agent_name}' not found in model {file_name}. Available keys: {available_keys}. Skipping.")
                continue
            policy_net = model_data["policy_nets"][agent_name]
            key = f"player_{i}"
            ai_agents[key] = {
                "policy_net": policy_net,
                "obp_model": model_data["obp_model"],
                "obs_version": model_data["obs_version"],
                "input_dim": model_data["input_dim"],
                "uses_memory": model_data["uses_memory"],
                "label": f"{file_name}_{agent_name}"
            }
        if not ai_agents:
            self.show_info("No valid PPO agents selected.")
            return None
        return ai_agents

    def start_battleground_async(self):
        import threading
        thread = threading.Thread(target=self.start_battleground)
        thread.start()

    def start_battleground(self):
        try:
            try:
                target_segments_val = int(self.target_segments_var.get())
            except ValueError:
                self.show_info("Invalid target segments value, using default of 500.")
                target_segments_val = 500
            self.target_segments = target_segments_val

            overall_results = {}

            # --- Hardcoded Agents Matches ---
            # We'll ONLY generate data vs. "Conservative" (SelectiveTableConservativeChallenger).
            if self.include_hardcoded.get():
                hardcoded_results = {}
                ai_agents_all = self.load_selected_agents()
                if not ai_agents_all:
                    return
                ai_agents = {}
                for i, key in enumerate(ai_agents_all.keys()):
                    # We'll just pick up to 2 PPO agents
                    if i < 2:
                        ai_agents[key] = ai_agents_all[key]
                if len(ai_agents) < 2:
                    self.show_info("Need at least two PPO agents for matches vs. hardcoded opponents.")
                    return

                # Filter out everything but "Conservative"
                for hc_name, hc_class in self.hardcoded_agents.items():
                    if hc_name != "Strategic":
                        continue  # Skip all other hardcoded bots

                    wins = [0, 0, 0]
                    match_count = 0
                    while True:
                        current_segments = sum(1 for seg, label in self.training_data if label == hc_name)
                        if current_segments >= target_segments_val:
                            break
                        match_count += 1
                        if match_count >= game_threshold and current_segments < viable_segment_threshold:
                            logger.info(f"[Hardcoded:{hc_name}] Early culled: Only {current_segments} segments in {match_count} matches.")
                            self.training_data = [(seg, label) for seg, label in self.training_data if label != hc_name]
                            break
                        winner = self.run_match(ai_agents, hardcoded_agent=hc_class(hc_name), hardcoded_label=hc_name)
                        if winner == list(ai_agents.keys())[0]:
                            wins[0] += 1
                        elif winner == list(ai_agents.keys())[1]:
                            wins[1] += 1
                        elif winner == "hardcoded_agent":
                            wins[2] += 1
                        logger.info(
                            f"[Hardcoded:{hc_name}] After {match_count} matches, training segments: "
                            f"{sum(1 for seg, label in self.training_data if label == hc_name)}"
                        )
                    hardcoded_results[hc_name] = wins
                overall_results["hardcoded"] = hardcoded_results

            # --- PPO-Only Matches: We can keep this or remove it, as desired.
            if self.include_ppo.get():
                all_agents = self.load_selected_agents()
                if not all_agents:
                    return

                def sample_count(agent_data):
                    label = agent_data['label']
                    if label in self.culled_agents:
                        return float('inf')
                    return sum(1 for seg, lbl in self.training_data if lbl == label)

                overall_match_count = 0

                # We'll keep the logic to do PPO vs PPO matches if you still want them.
                while True:
                    available_agents = {
                        k: v for k, v in all_agents.items()
                        if v['label'] not in self.culled_agents and
                        sum(1 for seg, lbl in self.training_data if lbl == v['label']) < target_segments_val
                    }
                    if len(available_agents) < 3:
                        self.show_info("Not enough non-culled PPO agents remain for further matches.")
                        break

                    selected_keys = sorted(available_agents.keys(), key=lambda k: sample_count(available_agents[k]))[:3]

                    new_subset = {}
                    mapping = {}
                    for i, key in enumerate(selected_keys):
                        new_key = f"player_{i}"
                        new_subset[new_key] = available_agents[key]
                        mapping[new_key] = available_agents[key]['label']

                    local_game_counts = { new_key: 0 for new_key in new_subset }
                    match_count = 0

                    while all(
                        sum(1 for seg, lbl in self.training_data if lbl == mapping[new_key]) < target_segments_val
                        for new_key in new_subset
                    ):
                        match_count += 1
                        overall_match_count += 1
                        to_remove = []
                        for new_key in list(new_subset.keys()):
                            local_game_counts[new_key] += 1
                            curr_segments = sum(1 for seg, lbl in self.training_data if lbl == mapping[new_key])
                            if local_game_counts[new_key] >= game_threshold and curr_segments < viable_segment_threshold:
                                logger.info(
                                    f"[PPO] Early culling {mapping[new_key]}: "
                                    f"Only {curr_segments} segments in {local_game_counts[new_key]} matches."
                                )
                                to_remove.append(new_key)
                        for rem in to_remove:
                            del new_subset[rem]
                        if len(new_subset) < 3:
                            logger.info("PPO-only match: Not enough agents remain in current combination after early culling.")
                            break
                        try:
                            winner = self.run_match(new_subset, hardcoded_agent=None, hardcoded_label=None)
                        except Exception as e:
                            logger.error(f"Error running match for selected agents: {e}")
                            break
                        logger.info("[PPO] After {} matches in current combination, training segments: {}".format(
                            match_count,
                            ", ".join(
                                f"{mapping[new_key]}: "
                                f"{sum(1 for seg, lbl in self.training_data if lbl == mapping[new_key])}"
                                for new_key in new_subset
                            )
                        ))
                overall_results["ppo"] = {"matches": overall_match_count}
            
            self.display_results(overall_results)
            total_examples = len(self.training_data)
            self.show_info(f"Battleground complete. Generated {total_examples} training examples.")
            self.update_progress_bar()
        except Exception as e:
            self.show_info(f"Error: {str(e)}")

    def run_match(self, ai_agents, hardcoded_agent=None, hardcoded_label=None):
        try:
            env = LiarsDeckEnv(num_players=3, render_mode=None)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.current_env = env

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
                policy_net.load_state_dict(agent_data["policy_net"], strict=False)
                policy_net.to(device).eval()
                policy_nets[agent_id] = policy_net

                obp_model_state = agent_data["obp_model"]
                if obp_model_state:
                    obp_input_dim = obp_model_state["fc1.weight"].shape[1]
                    obp_hidden_dim = self.get_hidden_dim_from_state_dict(obp_model_state, "fc1")
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
                    if hardcoded_agent is None:
                        raise ValueError("No hardcoded agent provided for non-PPO-controlled player.")
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

            for agent in env.agents:
                if agent not in self.games_since_last_collection:
                    self.games_since_last_collection[agent] = 0
                self.games_since_last_collection[agent] += 1

                memory_obj = get_opponent_memory(agent)
                for opp, events in memory_obj.memory.items():
                    events_list = list(events)
                    if events_list and (len(events_list) >= 50 or self.games_since_last_collection[agent] >= 5):
                        if opp in ai_agents:
                            opp_label = ai_agents[opp]['label']
                        else:
                            opp_label = hardcoded_label
                        self.training_data.append((events_list, opp_label))
                        events.clear()
                        memory_obj.aggregates[opp] = {
                            'early_total': 0,
                            'late_total': 0,
                            'early_challenge_count': 0,
                            'late_challenge_count': 0,
                            'early_three_card_trigger_count': 0,
                            'late_three_card_trigger_count': 0
                        }
                self.games_since_last_collection[agent] = 0

            if winner in ai_agents:
                return winner
            else:
                return "unknown_agent"

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.show_info(f"Error in run_match: {e}\nTraceback:\n{tb}")
            raise

    def choose_action(self, agent, policy_net, obp_model, observation, action_mask, device, num_players, obs_version, input_dim, uses_memory):
        base_obs = adapt_observation_for_version(observation, num_players, obs_version)
        
        from src import config
        transformer_features = np.zeros(config.STRATEGY_DIM * (num_players - 1), dtype=np.float32)
        obp_memory_embeddings = None

        if obs_version == 2 and uses_memory:
            opponents = [opp for opp in self.current_env.possible_agents if opp != agent]
            emb_list = []
            for opp in opponents:
                emb_tensor = get_opponent_memory_embedding(agent, opp, device)
                emb_arr = emb_tensor.squeeze(0).cpu().numpy()
                emb_list.append(emb_arr)
            if emb_list:
                emb_concat = np.concatenate(emb_list, axis=0)
                min_val = emb_concat.min()
                max_val = emb_concat.max()
                logger.debug(f"Memory embedding min: {min_val}, max: {max_val}")
                if max_val - min_val == 0:
                    normalized_emb = emb_concat
                else:
                    normalized_emb = (emb_concat - min_val) / (max_val - min_val)
                segment_size = config.STRATEGY_DIM
                obp_memory_embeddings = []
                for i in range(len(opponents)):
                    seg = normalized_emb[i * segment_size:(i + 1) * segment_size]
                    obp_memory_embeddings.append(torch.tensor(seg, dtype=torch.float32, device=device).unsqueeze(0))
                transformer_features = normalized_emb
            else:
                obp_memory_embeddings = [torch.zeros(1, config.STRATEGY_DIM, device=device) for _ in range(num_players - 1)]
                transformer_features = np.zeros(config.STRATEGY_DIM * (num_players - 1), dtype=np.float32)
            
            obp_probs = run_obp_inference(
                obp_model,
                base_obs,
                device,
                num_players,
                obs_version,
                agent,
                self.current_env,
                memory_embeddings=obp_memory_embeddings
            )
            
            final_obs = np.concatenate([
                np.array(base_obs, dtype=np.float32),
                np.array(obp_probs, dtype=np.float32),
                transformer_features
            ], axis=0)
        else:
            obp_probs = run_obp_inference(
                obp_model,
                base_obs,
                device,
                num_players,
                obs_version,
                agent,
                self.current_env
            )
            final_obs = np.concatenate([
                np.array(base_obs, dtype=np.float32),
                np.array(obp_probs, dtype=np.float32)
            ], axis=0)
        
        observation_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            probs, _, _ = policy_net(observation_tensor, None)
            probs = torch.clamp(probs, 1e-8, 1.0).squeeze(0)
        
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=device)
        masked_probs = probs * mask_tensor
        if masked_probs.sum() == 0:
            valid_indices = torch.nonzero(mask_tensor, as_tuple=True)[0]
            if len(valid_indices) > 0:
                masked_probs[valid_indices] = 1.0 / valid_indices.numel()
            else:
                masked_probs = torch.ones_like(probs) / probs.size(0)
        else:
            masked_probs /= masked_probs.sum()
        
        m = torch.distributions.Categorical(masked_probs)
        action = m.sample().item()
        logger.debug(f"Final observation (length {len(final_obs)}): {final_obs}")
        logger.debug(f"Action probabilities: {masked_probs.cpu().numpy()}")
        logger.debug(f"Selected action: {action}")
        
        return action

    def save_training_data(self):
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
            self.update_progress_bar()
        except Exception as e:
            self.show_info(f"Error saving training data: {str(e)}")

    def display_results(self, results):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        output = ""
        if "hardcoded" in results:
            output += "Matches vs. Hardcoded Opponents:\n"
            header = ("Hardcoded Agent   | PPO1 Wins | PPO2 Wins | Hardcoded Wins | "
                      "PPO1 Win Rate | PPO2 Win Rate | Hardcoded Win Rate\n")
            output += header
            output += "-" * 100 + "\n"
            for hc_name, wins in results["hardcoded"].items():
                ppo1_wins, ppo2_wins, hc_wins = wins
                total = ppo1_wins + ppo2_wins + hc_wins
                rate1 = ppo1_wins / total if total > 0 else 0.0
                rate2 = ppo2_wins / total if total > 0 else 0.0
                rate_hc = hc_wins / total if total > 0 else 0.0
                output += (f"{hc_name:18} | {ppo1_wins:^9} | {ppo2_wins:^9} | {hc_wins:^14} | "
                           f"{rate1:12.2%} | {rate2:12.2%} | {rate_hc:16.2%}\n")
            output += "\n"
        if "ppo" in results:
            output += "Matches among PPO Agents:\n"
            header = "PPO Agent             | Matches\n"
            output += header
            output += "-" * 50 + "\n"
            total_matches = results["ppo"]["matches"]
            output += f"Total Matches: {total_matches}\n"
        self.results_text.insert(tk.END, output)
        self.results_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = AgentBattlegroundGUI(root)
    root.mainloop()
