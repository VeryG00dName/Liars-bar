# src/tests/data_gen.py

import os
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
import torch
import numpy as np
import random
import pickle

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.new_models import PolicyNetwork, OpponentBehaviorPredictor
from src import config

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

# Import the opponent memory query helper from memory.py
from src.model.memory import get_opponent_memory

logging.basicConfig(level=logging.INFO)  # Set to DEBUG for detailed logs
logger = logging.getLogger("AgentBattleground")


class AgentBattlegroundGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Agent Battleground")
        self.root.geometry("900x700")
        
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
        
        # This list will accumulate training examples.
        # Each element is a tuple: (memory_segment, label)
        self.training_data = []
        # Will hold the transformer instance once loaded.
        self.strategy_transformer = None
        # Will store the current environment for memory queries.
        self.current_env = None

        self.create_file_drop_zone()
        self.create_model_info_panel()
        self.create_ai_selection()
        self.create_match_options()
        self.create_control_buttons()
        self.create_results_display()

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
        frame = ttk.LabelFrame(self.root, text="PPO Agents Selection", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        self.agent_selectors = {}
        # Now support 3 PPO agent selectors.
        for i in range(3):
            ttk.Label(frame, text=f"PPO Agent {i+1}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            self.agent_selectors[i] = ttk.Combobox(frame, state="readonly", width=50)
            self.agent_selectors[i].grid(row=i, column=1, sticky=tk.EW, padx=5, pady=2)
        frame.columnconfigure(1, weight=1)

    def create_match_options(self):
        # Add checkbuttons to choose which match types to run.
        frame = ttk.LabelFrame(self.root, text="Match Options", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        self.include_hardcoded = tk.BooleanVar(value=True)
        self.include_ppo = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Include Matches vs. Hardcoded Opponents", variable=self.include_hardcoded).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(frame, text="Include Matches among PPO Agents", variable=self.include_ppo).pack(anchor=tk.W, pady=2)

    def create_control_buttons(self):
        frame = ttk.Frame(self.root)
        frame.pack(pady=10)
        ttk.Button(frame, text="Refresh Agents", command=self.update_agent_selectors).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Start Battleground", command=self.start_battleground_async).pack(side=tk.LEFT, padx=5)
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
        required_keys = ["policy_nets"]
        if any(k not in checkpoint for k in required_keys):
            raise ValueError("Missing required keys in checkpoint")
        
        # Determine the observation version based on policy network's input dimension
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
            "obp_model": checkpoint.get("obp_model", None),  # Handle missing OBP
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
        for i in range(3):
            self.agent_selectors[i]["values"] = agent_options
            if agent_options:
                # Select first option if nothing is selected
                self.agent_selectors[i].current(0)
            self.agent_selectors[i].state(["!disabled"])

    def load_selected_agents(self, num_agents):
        """Loads the selected PPO agents.
        
        num_agents should be 2 (for matches vs. hardcoded opponents) or 3 (for PPO vs. PPO matches).
        """
        ai_agents = {}
        try:
            for i in range(num_agents):
                selection = self.agent_selectors[i].get()
                if not selection:
                    raise ValueError(f"Select PPO Agent {i+1}")
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
                    "uses_memory": model_data["uses_memory"],
                    "label": f"{file_name}_{agent_name}"  # Label includes the file name for differentiation.
                }
            return ai_agents
        except Exception as e:
            self.show_info(f"Error loading selected agents: {str(e)}")
            return None

    def start_battleground_async(self):
        # Run the battleground in a separate thread so the GUI remains responsive.
        import threading
        thread = threading.Thread(target=self.start_battleground)
        thread.start()

    def start_battleground(self):
        try:
            overall_results = {}
            target_segments = 20  # We want to generate at least 20 training segments per opponent label.
            
            # -------------------------------
            # Matches vs. Hardcoded Opponents
            # -------------------------------
            if self.include_hardcoded.get():
                hardcoded_results = {}
                # Use 2 PPO agents for these matches.
                ai_agents = self.load_selected_agents(num_agents=2)
                if not ai_agents:
                    return
                for hc_name, hc_class in self.hardcoded_agents.items():
                    wins = [0, 0, 0]  # [PPO1 Wins, PPO2 Wins, Hardcoded Wins]
                    current_segments = sum(1 for seg, label in self.training_data if label == hc_name)
                    match_count = 0
                    while current_segments < target_segments:
                        match_count += 1
                        winner = self.run_match(ai_agents, hardcoded_agent=hc_class(hc_name), hardcoded_label=hc_name)
                        if winner == "player_0":
                            wins[0] += 1
                        elif winner == "player_1":
                            wins[1] += 1
                        elif winner == "hardcoded_agent":
                            wins[2] += 1
                        current_segments = sum(1 for seg, label in self.training_data if label == hc_name)
                        logger.info(f"[Hardcoded:{hc_name}] After {match_count} matches, training segments: {current_segments}")
                    hardcoded_results[hc_name] = wins
                overall_results["hardcoded"] = hardcoded_results

            # -------------------------------
            # Matches among PPO Agents
            # -------------------------------
            if self.include_ppo.get():
                ppo_results = {}  # keys: PPO agent label, value: win count
                # Use 3 PPO agents for these matches.
                ai_agents = self.load_selected_agents(num_agents=3)
                if not ai_agents:
                    return
                # Initialize win counts for each PPO agent.
                for agent in ai_agents.values():
                    ppo_results[agent['label']] = 0
                match_count = 0
                # Continue until each PPO agent has at least target_segments training examples.
                while any(
                    sum(1 for seg, label in self.training_data if label == agent_label) < target_segments
                    for agent_label in ppo_results.keys()
                ):
                    match_count += 1
                    # In PPO vs. PPO matches, no hardcoded opponent is used.
                    winner = self.run_match(ai_agents, hardcoded_agent=None, hardcoded_label=None)
                    if winner in ai_agents:
                        ppo_results[ai_agents[winner]['label']] += 1
                    logger.info(f"[PPO] After {match_count} matches, training segments: " +
                                ", ".join(f"{lbl}: {sum(1 for seg, label in self.training_data if label == lbl)}" for lbl in ppo_results))
                overall_results["ppo"] = {"wins": ppo_results, "matches": match_count}
            
            self.display_results(overall_results)
            total_examples = len(self.training_data)
            self.show_info(f"Battleground complete. Generated {total_examples} training examples.")
        except Exception as e:
            self.show_info(f"Error: {str(e)}")

    def run_match(self, ai_agents, hardcoded_agent=None, hardcoded_label=None):
        """
        Runs a single match with either:
          - Two PPO agents (from ai_agents) and one hardcoded agent (if provided), or
          - Three PPO agents if hardcoded_agent is None.
        After the match, training examples are extracted from each agent's opponent memory.
        The full memory is split into up to 5 segments.
        """
        env = LiarsDeckEnv(num_players=3, render_mode=None)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Set the current environment so that memory queries work in choose_action.
        self.current_env = env
        
        # Initialize policy networks and OBP models for PPO agents.
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
                # For matches with a hardcoded agent.
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
        
        # Determine winner.
        max_reward = max(env.rewards.values())
        winners = [agent for agent, reward in env.rewards.items() if reward == max_reward]
        winner = winners[0]  # Assuming a single winner.

        # Generate training data from opponent memories.
        for agent in env.agents:
            if agent in ai_agents:
                label = ai_agents[agent]['label']
            else:
                label = hardcoded_label
            memory_obj = get_opponent_memory(agent)
            full_memory = list(memory_obj.memory)
            # Always split the memory into 5 segments (using np.array_split)
            if full_memory:
                segments = np.array_split(full_memory, 5)
                for seg in segments:
                    seg_list = seg.tolist() if hasattr(seg, 'tolist') else list(seg)
                    if seg_list:
                        self.training_data.append((seg_list, label))
            memory_obj.memory.clear()
            memory_obj.aggregates.clear()

        hardcoded_agent_id = f"player_{env.num_players - 1}"  # e.g. 'player_2'
        if winner in ai_agents:
            return winner
        elif winner == hardcoded_agent_id:
            return "hardcoded_agent"
        else:
            return "unknown_agent"

    def choose_action(self, agent_id, policy_net, obp_model, observation, action_mask, device, num_players, obs_version, input_dim, uses_memory):
        """Selects an action using OBP integration and, if applicable, transformer‐based memory integration."""
        converted_obs = adapt_observation_for_version(observation, num_players, obs_version)
        logging.debug(f"Converted observation (length {len(converted_obs)}): {converted_obs}")

        # Run OBP inference.
        obp_probs = run_obp_inference(obp_model, converted_obs, device, num_players, obs_version)
        logging.debug(f"OBP probabilities: {obp_probs}")

        if obs_version == 2 and uses_memory:
            required_mem_dim = input_dim - (len(converted_obs) + len(obp_probs))
            if required_mem_dim == config.STRATEGY_DIM * (num_players - 1):
                from src.env.liars_deck_env_utils import query_opponent_memory_full

                class Vocabulary:
                    def __init__(self, max_size):
                        self.token2idx = {"<PAD>": 0, "<UNK>": 1}
                        self.idx2token = {0: "<PAD>", 1: "<UNK>"}
                        self.max_size = max_size
                    def encode(self, token):
                        if token in self.token2idx:
                            return self.token2idx[token]
                        else:
                            if len(self.token2idx) < self.max_size:
                                idx = len(self.token2idx)
                                self.token2idx[token] = idx
                                self.idx2token[idx] = token
                                return idx
                            else:
                                return self.token2idx["<UNK>"]

                def convert_memory_to_tokens(memory, vocab):
                    tokens = []
                    for event in memory:
                        if isinstance(event, dict):
                            sorted_items = sorted(event.items())
                            token_str = "_".join(f"{k}-{v}" for k, v in sorted_items)
                        else:
                            token_str = str(event)
                        tokens.append(vocab.encode(token_str))
                    return tokens

                vocab_inst = Vocabulary(max_size=config.STRATEGY_NUM_TOKENS)
                mem_features_list = []
                for opp in self.current_env.possible_agents:
                    if opp != agent_id:
                        mem_summary = query_opponent_memory_full(agent_id, opp)
                        token_seq = convert_memory_to_tokens(mem_summary, vocab_inst)
                        token_tensor = torch.tensor(token_seq, dtype=torch.long, device=device).unsqueeze(0)
                        # Use the stored transformer (load once)
                        if self.strategy_transformer is None:
                            from src.model.new_models import StrategyTransformer
                            self.strategy_transformer = StrategyTransformer(
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
                                state_dict = torch.load(transformer_checkpoint_path, map_location=device)
                                self.strategy_transformer.load_state_dict(state_dict)
                                logging.info(f"Loaded transformer from '{transformer_checkpoint_path}'.")
                            else:
                                logging.warning("Transformer checkpoint not found, using randomly initialized transformer.")
                            self.strategy_transformer.classification_head = None
                            self.strategy_transformer.eval()
                        with torch.no_grad():
                            embedding, _ = self.strategy_transformer(token_tensor)
                        mem_features_list.append(embedding.cpu().numpy().flatten())
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
            action_probs, _ = policy_net(observation_tensor)
        
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

    def save_training_data(self):
        """Append the accumulated training data to the file instead of overwriting it."""
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

    def display_results(self, results):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        output = ""
        # Display hardcoded match results, if available.
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
        # Display PPO vs. PPO match results, if available.
        if "ppo" in results:
            output += "Matches among PPO Agents:\n"
            header = "PPO Agent             | Wins | Win Rate\n"
            output += header
            output += "-" * 50 + "\n"
            total_matches = results["ppo"]["matches"]
            for agent_label, wins in results["ppo"]["wins"].items():
                win_rate = wins / total_matches if total_matches > 0 else 0.0
                output += f"{agent_label:22} | {wins:^4} | {win_rate:8.2%}\n"
        self.results_text.insert(tk.END, output)
        self.results_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = AgentBattlegroundGUI(root)
    root.mainloop()
