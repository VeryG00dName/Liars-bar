# src/misc/play_vs_ai.py
import os
import logging
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import copy
import torch

# Silence TensorFlow oneDNN notices if TensorFlow gets imported indirectly.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.agents.autoregressive_ppo_agent import PPOAutoregressiveAgent
from src import config
from src.training import train_extras

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PlayVsAI")


class PlayVsAIGUI:
    """Tkinter-based UI for playing against autoregressive PPO agents."""

    CHECKPOINT_RELATIVE_PATH = os.path.join("checkpoints", "test24", "gen_5", "final.pth")

    def __init__(self, root):
        self.root = root
        self.root.title("Play vs AI")
        self.root.geometry("800x500")

        # Bind the window close event to ensure proper cleanup.
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_checkpoint = None
        self.loaded_state_dict = None
        self.ai_agents = {}
        self.human_agent_id = None

        # Track moves made by opponents since your last turn.
        self.moves_since_last_turn = []

        # Create a persistent Rich Console and Live display.
        self.console = Console()
        self.live = None

        # UI components
        self.checkpoint_status_var = tk.StringVar(value="Loading checkpoint...")
        self.info_text = None

        self.create_checkpoint_panel()
        self.create_model_info_panel()
        self.create_control_buttons()

        # Create a persistent action window for human input (this window stays visible).
        self.create_action_window()

        self.game_window = None
        self.current_env = None

        # Load the default checkpoint immediately
        self.load_checkpoint()

    def on_closing(self):
        """Callback when the main window is closed. Stop the Live display and then close the window."""
        if self.live is not None:
            self.live.stop()
            self.live = None
        self.root.destroy()

    def create_checkpoint_panel(self):
        frame = ttk.LabelFrame(self.root, text="Checkpoint", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        checkpoint_display = "checkpoints\\test24\\gen_5\\final.pth"
        ttk.Label(frame, text=f"Using checkpoint: {checkpoint_display}").pack(anchor=tk.W)
        ttk.Label(frame, textvariable=self.checkpoint_status_var).pack(anchor=tk.W, pady=(6, 0))

    def create_model_info_panel(self):
        frame = ttk.LabelFrame(self.root, text="Status", padding=10)
        frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
        self.info_text = tk.Text(frame, wrap=tk.WORD, state=tk.DISABLED, height=4)
        self.info_text.pack(fill=tk.BOTH, expand=True)

    def create_control_buttons(self):
        frame = ttk.Frame(self.root)
        frame.pack(pady=10)
        ttk.Button(frame, text="Start 4-Player Game", command=self.start_game).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame, text="(3 AI opponents)").pack(side=tk.LEFT, padx=5)

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

    def show_info(self, message):
        if self.info_text is None:
            return
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, message)
        self.info_text.config(state=tk.DISABLED)

    def load_checkpoint(self):
        relative_path = self.CHECKPOINT_RELATIVE_PATH
        absolute_path = os.path.join(config.BASE_DIR, relative_path)
        try:
            checkpoint = torch.load(absolute_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            if not isinstance(state_dict, dict):
                raise ValueError("Checkpoint missing model_state_dict.")

            # Copy tensors to avoid accidental mutation between agents.
            cloned_state = {}
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    cloned_state[key] = value.clone().detach()
                else:
                    cloned_state[key] = copy.deepcopy(value)
            self.loaded_state_dict = cloned_state
            self.loaded_checkpoint = checkpoint if isinstance(checkpoint, dict) else {"model_state_dict": self.loaded_state_dict}

            self.checkpoint_status_var.set("Checkpoint loaded: PPO autoregressive model ready.")
            self.show_info("Checkpoint loaded successfully.")
        except FileNotFoundError:
            self.loaded_checkpoint = None
            self.loaded_state_dict = None
            self.checkpoint_status_var.set("Checkpoint not found.")
            self.show_info(f"Checkpoint not found at {absolute_path}.")
            logger.error("Checkpoint not found at %s", absolute_path)
        except Exception as exc:
            self.loaded_checkpoint = None
            self.loaded_state_dict = None
            self.checkpoint_status_var.set("Failed to load checkpoint.")
            self.show_info(f"Error loading checkpoint: {exc}")
            logger.exception("Failed to load checkpoint from %s", absolute_path)

    def start_game(self):
        if self.loaded_checkpoint is None:
            self.show_info("Cannot start game: checkpoint not loaded.")
            return
        self.root.after(100, self.play_game)

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
            border_style="bright_blue",
        )
        current_player = env.agent_selection
        your_hand = env.players_hands.get(current_player, [])
        your_hand_panel = Panel(
            f"[bold white]{your_hand}[/bold white]",
            title=f"Your Hand ({current_player})",
            border_style="green",
        )
        opponent_table = Table(title="Opponent Hands", expand=True)
        opponent_table.add_column("Opponent", style="cyan", no_wrap=True)
        opponent_table.add_column("Cards Left", justify="center", style="magenta")
        for agent in env.possible_agents:
            if agent != current_player:
                hand = env.players_hands.get(agent, [])
                opponent_table.add_row(agent, f"{len(hand)} cards")
        moves_text = (
            "\n".join(self.moves_since_last_turn)
            if self.moves_since_last_turn
            else "No moves since your last turn."
        )
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
            Layout(name="body", ratio=1),
        )
        layout["body"].split_row(
            Layout(table_card_panel, name="left", size=30),
            Layout(name="center", ratio=2),
            Layout(name="right", size=30),
        )
        layout["center"].split(
            Layout(your_hand_panel, name="your_hand", size=7),
            Layout(opponent_table, name="opponent_hands", size=7),
            Layout(moves_panel, name="moves", size=5),
        )
        layout["right"].split(
            Layout(active_table, name="active_players", size=10),
            Layout(penalties_table, name="penalties", size=10),
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

    def play_game(self):
        num_players = 4
        self.current_env = LiarsDeckEnv(num_players=num_players, render_mode="player")
        self.moves_since_last_turn = []

        self.human_agent_id = self.current_env.possible_agents[-1]
        ai_agent_ids = [agent for agent in self.current_env.possible_agents if agent != self.human_agent_id]

        if self.loaded_state_dict is None:
            self.show_info("No model state available in checkpoint.")
            return

        # Build AI agents from the checkpoint using the fused PPO state dict.
        self.ai_agents = {}
        for agent_id in ai_agent_ids:
            agent = PPOAutoregressiveAgent(device=self.device, player_id=agent_id)
            checkpoint_payload = {"policy_nets": {"agent_model": copy.deepcopy(self.loaded_state_dict)}}
            agent.load_models_from_checkpoint(checkpoint_payload, "agent_model")
            self.ai_agents[agent_id] = agent

        try:
            self.current_env.reset()
            while self.current_env.agent_selection is not None:
                current_agent = self.current_env.agent_selection
                obs, reward, termination, truncation, info = self.current_env.last()

                if termination or truncation:
                    self.current_env.step(None)
                    continue

                observation = obs[current_agent]

                if current_agent in self.ai_agents:
                    action = self.ai_agents[current_agent].get_action(
                        self.current_env,
                        current_agent,
                        observation=observation,
                        info=info,
                    )
                    self.track_move(current_agent, action)
                else:
                    self.render_player_mode(self.current_env)
                    self.moves_since_last_turn = []
                    action = self.get_human_action(info['action_mask'])

                self.current_env.step(action)

            self.show_game_result()
        except Exception as exc:
            self.show_info(f"Error during game: {exc}")
            logger.exception("Error during game loop")
        finally:
            if self.current_env is not None:
                self.current_env.close()
                self.current_env = None
            for agent in self.ai_agents.values():
                agent.reset()
            if self.live is not None:
                self.live.stop()
                self.live = None


if __name__ == "__main__":
    root = tk.Tk()
    app = PlayVsAIGUI(root)
    root.mainloop()
