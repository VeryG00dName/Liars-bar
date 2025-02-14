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

# Import memory utilities and opponent memory query functions.
from src.model.memory import get_opponent_memory
from src.env.liars_deck_env_utils import query_opponent_memory, query_opponent_memory_full

# Import Rich components for a nicer console display.
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PlayVsAI")

# Global transformer variable (for transformer‐based memory integration)
global_strategy_transformer = None


class PlayVsAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Play vs AI")
        self.root.geometry("800x500")
        
        # Bind the window close event to ensure proper cleanup.
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.loaded_models = {}
        self.selected_agents = {}
        
        # Track moves made by opponents since your last turn.
        self.moves_since_last_turn = []
        
        # Create a persistent Rich Console and Live display.
        self.console = Console()
        self.live = None
        
        self.create_file_drop_zone()
        self.create_model_info_panel()
        self.create_ai_selection()
        self.create_control_buttons()
        
        # Create a persistent action window for human input (this window stays visible).
        self.create_action_window()
        
        self.game_window = None
        self.current_env = None

    def on_closing(self):
        """Callback when the main window is closed. Stop the Live display and then close the window."""
        if self.live is not None:
            self.live.stop()
            self.live = None
        self.root.destroy()

    def get_hidden_dim_from_state_dict(self, state_dict, layer_prefix='fc1'):
        """Extracts hidden dimension from model weights."""
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

    def create_action_window(self):
        """Creates a persistent Toplevel window for human actions (this window stays visible)."""
        self.action_window = tk.Toplevel(self.root)
        self.action_window.title("Your Turn")
        self.action_var = tk.IntVar(value=-1)
        # Prevent the user from closing the window manually.
        self.action_window.protocol("WM_DELETE_WINDOW", lambda: None)
        self.action_buttons = []
        actions = [
            (0, "Play 1 Table Card (Action 0)"),
            (1, "Play 2 Table Cards (Action 1)"),
            (2, "Play 3 Table Cards (Action 2)"),
            (3, "Play 1 Non-Table Card (Action 3)"),
            (4, "Play 2 Non-Table Cards (Action 4)"),
            (5, "Play 3 Non-Table Cards (Action 5)"),
            (6, "Challenge (Action 6)")
        ]
        for action_value, label in actions:
            btn = ttk.Button(self.action_window, text=label,
                             command=lambda val=action_value: self.select_action(val))
            btn.pack(padx=10, pady=5, fill=tk.X)
            self.action_buttons.append(btn)
        # The action window remains visible.

    def select_action(self, action_value):
        """Callback when a human player selects an action."""
        self.action_var.set(action_value)

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
                    "policy_state": self.loaded_models[file_path]["policy_nets"][agent_name],
                    "obp_model_state": self.loaded_models[file_path]["obp_model"]
                }
            self.root.after(100, lambda: self.play_game(agents))
        except Exception as e:
            self.show_info(f"Error: {str(e)}")

    def show_game_result(self):
        rewards = self.current_env.rewards
        max_reward = max(rewards.values())
        winners = [agent for agent, reward in rewards.items() if reward == max_reward]
        result_text = "Game Results:\n"
        for agent, reward in rewards.items():
            result_text += f"{agent}: {reward}\n"
        result_text += "\nWinner(s):\n" + "\n".join(winners)
        messagebox.showinfo("Game Over", result_text)
        self.current_env = None

    def track_move(self, agent, action):
        """Record an AI move made since your last turn."""
        if action in range(0, 6):
            if action <= 2:
                num_cards = action + 1
            else:
                num_cards = action - 2
            move_str = f"{agent} played {num_cards} card{'s' if num_cards > 1 else ''}"
        elif action == 6:
            move_str = f"{agent} challenged"
        else:
            move_str = f"{agent} performed action {action}"
        self.moves_since_last_turn.append(move_str)

    def render_player_mode(self, env):
        """Update the persistent Rich display (Live) for player mode."""
        header_panel = Panel("[bold green]Your Turn[/bold green]", expand=False)
        table_card_panel = Panel(
            f"Table Card: [bold yellow]{env.table_card}[/bold yellow]",
            title="Table Card",
            border_style="bright_blue"
        )
        current_player = env.agent_selection
        your_hand = env.players_hands.get(current_player, [])
        your_hand_panel = Panel(
            f"[bold white]{your_hand}[/bold white]",
            title=f"Your Hand ({current_player})",
            border_style="green"
        )
        opponent_table = Table(title="Opponent Hands", expand=True)
        opponent_table.add_column("Opponent", style="cyan", no_wrap=True)
        opponent_table.add_column("Cards Left", justify="center", style="magenta")
        for agent in env.possible_agents:
            if agent != current_player:
                hand = env.players_hands.get(agent, [])
                opponent_table.add_row(agent, f"{len(hand)} cards")
        moves_text = "\n".join(self.moves_since_last_turn) if self.moves_since_last_turn else "No moves since your last turn."
        moves_panel = Panel(moves_text, title="Moves Since Your Last Turn", border_style="blue")
        active_table = Table(title="Active Players", expand=True)
        active_table.add_column("Agent", style="cyan", no_wrap=True)
        active_table.add_column("Status", style="bold")
        for agent in env.possible_agents:
            if env.terminations.get(agent, False):
                status = "[red]Game-Terminated[/red]"
            elif env.round_eliminated.get(agent, False):
                status = "[yellow]Round-Eliminated[/yellow]"
            else:
                status = "[green]Active[/green]"
            active_table.add_row(agent, status)
        penalties_table = Table(title="Penalties", expand=True)
        penalties_table.add_column("Agent", style="cyan", no_wrap=True)
        penalties_table.add_column("Penalty", style="bold")
        for agent in env.possible_agents:
            penalty = env.penalties.get(agent, 0)
            penalties_table.add_row(agent, str(penalty))

        layout = Layout()
        layout.split(
            Layout(header_panel, name="header", size=3),
            Layout(name="body", ratio=1)
        )
        layout["body"].split_row(
            Layout(table_card_panel, name="left", size=30),
            Layout(name="center", ratio=2),
            Layout(name="right", size=30)
        )
        layout["center"].split(
            Layout(your_hand_panel, name="your_hand", size=7),
            Layout(opponent_table, name="opponent_hands", size=7),
            Layout(moves_panel, name="moves", size=5)
        )
        layout["right"].split(
            Layout(active_table, name="active_players", size=10),
            Layout(penalties_table, name="penalties", size=10)
        )

        if self.live is None:
            self.live = Live(layout, console=self.console, refresh_per_second=4)
            self.live.start()
        else:
            self.live.update(layout)

    def get_human_action(self, action_mask):
        """Update button states in the persistent action window and wait for the user to select an action."""
        for idx, btn in enumerate(self.action_buttons):
            state = tk.NORMAL if action_mask[idx] != 0 else tk.DISABLED
            btn.config(state=state)
        self.action_var.set(-1)
        self.action_window.lift()
        self.action_window.wait_variable(self.action_var)
        return self.action_var.get()

    def choose_action(self, agent_id, policy_net, obp_model, observation, device, num_players, action_mask, uses_memory=False):
        """Choose an action using the provided policy network (with optional memory integration)."""
        global global_strategy_transformer
        num_opponents = num_players - 1
        opp_feature_dim = config.OPPONENT_INPUT_DIM

        hand_vector_length = 2
        last_action_val_length = 1
        active_players_length = num_players
        opp_features_start = hand_vector_length + last_action_val_length + active_players_length
        opp_features_end = opp_features_start + opp_feature_dim * num_opponents

        opponent_features = observation[opp_features_start:opp_features_end]
        opponent_features = opponent_features.reshape(num_opponents, opp_feature_dim)

        obp_probs = []
        for idx, opp_feat in enumerate(opponent_features):
            opp_feat_tensor = torch.tensor(opp_feat, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if obp_model:
                    if uses_memory:
                        opp_id = self.current_env.possible_agents[idx]
                        mem_summary = query_opponent_memory_full(agent_id, opp_id)
                        if mem_summary:
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
                            token_seq = convert_memory_to_tokens(mem_summary, vocab_inst)
                            if token_seq:
                                token_tensor = torch.tensor(token_seq, dtype=torch.long, device=device).unsqueeze(0)
                            else:
                                token_tensor = None
                        else:
                            token_tensor = None
                        if token_tensor is not None:
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
                                transformer_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth")
                                if os.path.exists(transformer_checkpoint_path):
                                    state_dict = torch.load(transformer_checkpoint_path, map_location=device)
                                    if "transformer_state_dict" in state_dict:
                                        state_dict = state_dict["transformer_state_dict"]
                                    global_strategy_transformer.load_state_dict(state_dict, strict=False)
                                    logger.info(f"Loaded transformer from '{transformer_checkpoint_path}'.")
                                else:
                                    logger.warning("Transformer checkpoint not found, using randomly initialized transformer.")
                                global_strategy_transformer.classification_head = None
                                global_strategy_transformer.eval()
                            with torch.no_grad():
                                memory_embedding, _ = global_strategy_transformer(token_tensor)
                        else:
                            memory_embedding = torch.zeros(1, config.STRATEGY_DIM, device=device)
                    else:
                        memory_embedding = None
                    if memory_embedding is not None:
                        logits = obp_model(opp_feat_tensor, memory_embedding)
                    else:
                        logits = obp_model(opp_feat_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    bluff_prob = probs[0, 1].item()
                else:
                    bluff_prob = 0.0
            obp_probs.append(bluff_prob)
        
        if uses_memory:
            expected_dim = policy_net.fc1.in_features
            non_memory_dim = len(observation) + len(obp_probs)
            required_mem_dim = expected_dim - non_memory_dim

            if required_mem_dim == config.STRATEGY_DIM * (num_players - 1):
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
                            transformer_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "transformer_classifier.pth")
                            if os.path.exists(transformer_checkpoint_path):
                                state_dict = torch.load(transformer_checkpoint_path, map_location=device)
                                if "transformer_state_dict" in state_dict:
                                    state_dict = state_dict["transformer_state_dict"]
                                global_strategy_transformer.load_state_dict(state_dict, strict=False)
                                logger.info(f"Loaded transformer from '{transformer_checkpoint_path}'.")
                            else:
                                logger.warning("Transformer checkpoint not found, using randomly initialized transformer.")
                            global_strategy_transformer.classification_head = None
                            global_strategy_transformer.eval()
                        with torch.no_grad():
                            embedding, _ = global_strategy_transformer(token_tensor)
                        mem_features_list.append(embedding.cpu().numpy().flatten())
                if mem_features_list:
                    mem_features = np.concatenate(mem_features_list, axis=0)
                else:
                    mem_features = np.zeros(config.STRATEGY_DIM * (num_players - 1), dtype=np.float32)
            else:
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
            
            final_obs = np.concatenate([observation, np.array(obp_probs, dtype=np.float32), mem_features], axis=0)
        else:
            final_obs = np.concatenate([observation, np.array(obp_probs, dtype=np.float32)], axis=0)

        expected_dim = policy_net.fc1.in_features
        actual_dim = final_obs.shape[0]
        assert actual_dim == expected_dim, f"Expected observation dimension {expected_dim}, got {actual_dim}"
        
        observation_tensor = torch.tensor(final_obs, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = policy_net(observation_tensor)
            if isinstance(output, tuple):
                action_probs = output[0]
            else:
                action_probs = output
        
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32).to(device)
        masked_probs = action_probs * mask_tensor
        if masked_probs.sum() == 0:
            masked_probs = mask_tensor / mask_tensor.sum()
        else:
            masked_probs /= masked_probs.sum()
        
        m = torch.distributions.Categorical(masked_probs)
        action = m.sample().item()
        
        # Debug logging is commented out.
        # logger.debug(f"Action probabilities: {masked_probs.cpu().numpy()}")
        # logger.debug(f"Selected action: {action}")
        return action

    def play_game(self, agents):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_players = 3  # Assuming human player + 2 AI agents
        self.current_env = LiarsDeckEnv(num_players=num_players, render_mode="player")
        
        policy_nets = {}
        obp_models = {}
        agent_memory_usage = {}
        for agent_id, agent_data in agents.items():
            policy_state = agent_data["policy_state"]
            uses_memory = ("fc4.weight" in policy_state)
            agent_memory_usage[agent_id] = uses_memory

            expected_input_dim = policy_state['fc1.weight'].shape[1]
            hidden_dim = self.get_hidden_dim_from_state_dict(policy_state)
            output_dim = self.current_env.action_spaces[agent_id].n

            if "fc_classifier.weight" in policy_state:
                use_aux_classifier = True
                num_opponent_classes = config.NUM_OPPONENT_CLASSES
            else:
                use_aux_classifier = False
                num_opponent_classes = None

            policy_net = PolicyNetwork(
                input_dim=expected_input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                use_lstm=True,
                use_layer_norm=True,
                use_aux_classifier=use_aux_classifier,
                num_opponent_classes=num_opponent_classes
            )
            policy_net.load_state_dict(policy_state)
            policy_net.to(device).eval()
            policy_nets[agent_id] = policy_net
            
            obp_model_state = agent_data["obp_model_state"]
            if obp_model_state:
                fc1_weight = obp_model_state.get("fc1.weight")
                if fc1_weight is None:
                    raise ValueError("OBP checkpoint missing fc1.weight")
                if fc1_weight.shape[1] == config.OPPONENT_INPUT_DIM + config.STRATEGY_DIM:
                    obp_input_dim = config.OPPONENT_INPUT_DIM
                    obp_hidden_dim = config.OPPONENT_HIDDEN_DIM
                    obp_model = OpponentBehaviorPredictor(
                        input_dim=obp_input_dim,
                        hidden_dim=obp_hidden_dim,
                        output_dim=2,
                        memory_dim=config.STRATEGY_DIM
                    )
                elif fc1_weight.shape[1] == config.OPPONENT_INPUT_DIM:
                    obp_input_dim = config.OPPONENT_INPUT_DIM
                    obp_hidden_dim = config.OPPONENT_HIDDEN_DIM
                    obp_model = OpponentBehaviorPredictor(
                        input_dim=obp_input_dim,
                        hidden_dim=obp_hidden_dim,
                        output_dim=2
                    )
                else:
                    raise ValueError(f"Unexpected OBP input dimension: {fc1_weight.shape[1]}")
                obp_model.load_state_dict(obp_model_state)
                obp_model.to(device).eval()
                obp_models[agent_id] = obp_model
            else:
                obp_models[agent_id] = None
        
        self.current_env.reset()
        while self.current_env.agent_selection is not None:
            current_agent = self.current_env.agent_selection
            obs, reward, termination, truncation, info = self.current_env.last()
            
            if termination or truncation:
                self.current_env.step(None)
                continue

            if current_agent in policy_nets:
                observation = obs[current_agent]
                action_mask = info['action_mask']
                action = self.choose_action(
                    agent_id=current_agent,
                    policy_net=policy_nets[current_agent],
                    obp_model=obp_models[current_agent],
                    observation=observation,
                    device=device,
                    num_players=num_players,
                    action_mask=action_mask,
                    uses_memory=agent_memory_usage[current_agent]
                )
                self.track_move(current_agent, action)
                # Non-transformer logging is commented out.
            else:
                self.render_player_mode(self.current_env)
                self.moves_since_last_turn = []
                action = self.get_human_action(info['action_mask'])
            
            self.current_env.step(action)
        
        self.show_game_result()
        if self.current_env is not None:
            self.current_env.close()
            self.current_env = None
        if self.live is not None:
            self.live.stop()
            self.live = None


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = PlayVsAIGUI(root)
    root.mainloop()
