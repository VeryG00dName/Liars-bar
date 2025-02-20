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

# New parameters for early culling (for non-GUI parts remain unchanged):
viable_segment_threshold = 5  # Agent must generate at least this many segments
game_threshold = 50            # Within this many games, otherwise cull

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
        # Initialize a counter for games played per agent since last memory collection.
        self.games_since_last_collection = {}

        self.create_file_drop_zone()
        self.create_model_info_panel()
        self.create_ai_selection()  # now a multi-select listbox
        self.create_match_options()
        self.create_parameters_box()  # <-- New parameters input box for target_segments
        self.create_control_buttons()
        self.create_results_display()
        # Add a button to print a summary of training data.
        self.create_summary_button()

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
        # Use a Listbox with multiple selection.
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
        """Creates a GUI box to set parameters, such as target_segments per agent."""
        frame = ttk.LabelFrame(self.root, text="Parameters", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        target_label = ttk.Label(frame, text="Target Segments per Agent:")
        target_label.pack(side=tk.LEFT, padx=5)
        # Use a StringVar to hold the value (default "500")
        self.target_segments_var = tk.StringVar(value="500")
        target_entry = ttk.Entry(frame, textvariable=self.target_segments_var, width=10)
        target_entry.pack(side=tk.LEFT)

    def create_control_buttons(self):
        frame = ttk.Frame(self.root)
        frame.pack(pady=10)
        ttk.Button(frame, text="Refresh Agents", command=self.update_agent_selectors).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Start Battleground", command=self.start_battleground_async).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Save Training Data", command=self.save_training_data).pack(side=tk.LEFT, padx=5)

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
        self.agent_listbox.delete(0, tk.END)
        for option in agent_options:
            self.agent_listbox.insert(tk.END, option)

    def print_data_summary(self):
        """Print a summary of self.training_data for debugging."""
        total_samples = len(self.training_data)
        label_counts = Counter(label for _, label in self.training_data)
        lengths = [len(seg) for seg, _ in self.training_data]
        avg_length = sum(lengths)/len(lengths) if lengths else 0
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
        # Print first 5 samples for inspection.
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
        # Use the order of selection to assign player IDs
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
            # Remap key to match environment (player_0, player_1, etc.)
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
            # Get the target_segments value from the GUI. Use 500 as default if conversion fails.
            try:
                target_segments_val = int(self.target_segments_var.get())
            except ValueError:
                self.show_info("Invalid target segments value, using default of 500.")
                target_segments_val = 500

            overall_results = {}
            # -------------------------------
            # Matches vs. Hardcoded Opponents
            # -------------------------------
            if self.include_hardcoded.get():
                hardcoded_results = {}
                ai_agents_all = self.load_selected_agents()
                if not ai_agents_all:
                    return
                # Choose first two agents.
                ai_agents = {}
                for i, key in enumerate(ai_agents_all.keys()):
                    if i < 2:
                        ai_agents[key] = ai_agents_all[key]
                if len(ai_agents) < 2:
                    self.show_info("Need at least two PPO agents for matches vs. hardcoded opponents.")
                    return
                for hc_name, hc_class in self.hardcoded_agents.items():
                    wins = [0, 0, 0]  # [PPO1 Wins, PPO2 Wins, Hardcoded Wins]
                    match_count = 0
                    while True:
                        current_segments = sum(1 for seg, label in self.training_data if label == hc_name)
                        if current_segments >= target_segments_val:
                            break
                        match_count += 1
                        if match_count >= game_threshold and current_segments < viable_segment_threshold:
                            logger.info(f"[Hardcoded:{hc_name}] Early culled: Only {current_segments} segments in {match_count} matches.")
                            self.training_data = [ (seg, label) for seg, label in self.training_data if label != hc_name ]
                            break
                        winner = self.run_match(ai_agents, hardcoded_agent=hc_class(hc_name), hardcoded_label=hc_name)
                        if winner == list(ai_agents.keys())[0]:
                            wins[0] += 1
                        elif winner == list(ai_agents.keys())[1]:
                            wins[1] += 1
                        elif winner == "hardcoded_agent":
                            wins[2] += 1
                        logger.info(f"[Hardcoded:{hc_name}] After {match_count} matches, training segments: {sum(1 for seg, label in self.training_data if label == hc_name)}")
                    hardcoded_results[hc_name] = wins
                overall_results["hardcoded"] = hardcoded_results

            # -------------------------------
            # Matches among PPO Agents
            # -------------------------------
            if self.include_ppo.get():
                all_agents = self.load_selected_agents()
                if not all_agents:
                    return
                from itertools import combinations
                selected_keys = list(all_agents.keys())
                if len(selected_keys) > 3:
                    agent_combos = list(combinations(selected_keys, 3))
                else:
                    agent_combos = [tuple(selected_keys)]
                ppo_results = { all_agents[k]['label']: 0 for k in selected_keys }
                overall_match_count = 0
                for combo in agent_combos:
                    new_subset = {}
                    mapping = {}
                    for i, key in enumerate(combo):
                        new_key = f"player_{i}"
                        new_subset[new_key] = all_agents[key]
                        mapping[new_key] = all_agents[key]['label']
                    match_count = 0
                    local_game_counts = { new_key: 0 for new_key in new_subset }
                    while any(
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
                                logger.info(f"[PPO] Early culling {mapping[new_key]}: Only {curr_segments} segments in {local_game_counts[new_key]} matches.")
                                to_remove.append(new_key)
                        for rem in to_remove:
                            del new_subset[rem]
                        if not new_subset:
                            logger.info(f"[PPO] All agents in combination {combo} culled early.")
                            break
                        try:
                            winner = self.run_match(new_subset, hardcoded_agent=None, hardcoded_label=None)
                        except Exception as e:
                            logger.error(f"Error running match for combination {combo}: {e}")
                            break
                        if winner in new_subset:
                            orig_label = mapping[winner]
                            ppo_results[orig_label] += 1
                        logger.info("[PPO] Combo {}: After {} matches, training segments: {}".format(
                            combo,
                            match_count,
                            ", ".join(f"{mapping[new_key]}: {sum(1 for seg, lbl in self.training_data if lbl == mapping[new_key])}" 
                                      for new_key in new_subset)
                        ))
                    logger.info(f"Finished combination {combo} after {match_count} matches.")
                overall_results["ppo"] = {"wins": ppo_results, "matches": overall_match_count}
            
            self.display_results(overall_results)
            total_examples = len(self.training_data)
            self.show_info(f"Battleground complete. Generated {total_examples} training examples.")
        except Exception as e:
            self.show_info(f"Error: {str(e)}")

    def run_match(self, ai_agents, hardcoded_agent=None, hardcoded_label=None):
        try:
            env = LiarsDeckEnv(num_players=3, render_mode=None)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.current_env = env

            # Set up policy and OBP models for each PPO-controlled agent.
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

            # --- Updated Memory Collection: Process Each Opponent Separately ---
            for agent in env.agents:
                # (We no longer label by the current agent; instead we label each sample by the opponent's identity.)
                # Update the game counter for this agent.
                if agent not in self.games_since_last_collection:
                    self.games_since_last_collection[agent] = 0
                self.games_since_last_collection[agent] += 1

                memory_obj = get_opponent_memory(agent)
                # Process each opponent's memory individually.
                for opp, events in memory_obj.memory.items():
                    events_list = list(events)
                    # Check if the events for this opponent meet our threshold.
                    if events_list and (len(events_list) >= 50 or self.games_since_last_collection[agent] >= 5):
                        # Determine the opponent's label:
                        if opp in ai_agents:
                            opp_label = ai_agents[opp]['label']
                        else:
                            opp_label = hardcoded_label
                        # Append a training sample labeled with the opponent's identity.
                        self.training_data.append((events_list, opp_label))
                        # Clear the memory for this opponent.
                        events.clear()
                        # Reset the aggregates for this opponent.
                        memory_obj.aggregates[opp] = {
                            'early_total': 0,
                            'late_total': 0,
                            'early_challenge_count': 0,
                            'late_challenge_count': 0,
                            'early_three_card_trigger_count': 0,
                            'late_three_card_trigger_count': 0
                        }
                # After processing all opponents for this agent, reset its game counter.
                self.games_since_last_collection[agent] = 0

            hardcoded_agent_id = f"player_{env.num_players - 1}"
            if winner in ai_agents:
                return winner
            elif winner == hardcoded_agent_id:
                return "hardcoded_agent"
            else:
                return "unknown_agent"

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.show_info(f"Error in run_match: {e}\nTraceback:\n{tb}")
            raise

    def choose_action(self, agent_id, policy_net, obp_model, observation, action_mask, device, num_players, obs_version, input_dim, uses_memory):
        logging.debug(f"Converted observation (length {len(adapt_observation_for_version(observation, num_players, obs_version))}): {adapt_observation_for_version(observation, num_players, obs_version)}")
        obp_probs = run_obp_inference(obp_model, adapt_observation_for_version(observation, num_players, obs_version), device, num_players, obs_version)
        logging.debug(f"OBP probabilities: {obp_probs}")
        if obs_version == 2 and uses_memory:
            required_mem_dim = input_dim - (len(adapt_observation_for_version(observation, num_players, obs_version)) + len(obp_probs))
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
            final_obs = np.concatenate([adapt_observation_for_version(observation, num_players, obs_version), np.array(obp_probs, dtype=np.float32), mem_features], axis=0)
        else:
            final_obs = np.concatenate([adapt_observation_for_version(observation, num_players, obs_version), np.array(obp_probs, dtype=np.float32)], axis=0)
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
