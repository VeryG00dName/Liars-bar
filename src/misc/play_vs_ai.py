# src/misc/play_vs_ai.py
import os
import logging
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import copy
from typing import Dict, Optional
from datetime import datetime
import torch

# Silence TensorFlow oneDNN notices if TensorFlow gets imported indirectly.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.env.liars_deck_env_core import LiarsDeckEnv
from src.agents.autoregressive_ppo_agent import PPOAutoregressiveAgent
from src import config

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.rule import Rule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PlayVsAI")


class PlayVsAIGUI:
    """Tkinter-based UI for playing against autoregressive PPO agents."""

    CHECKPOINT_RELATIVE_PATH = os.path.join("checkpoints", "test68", "gen_16", "final.pth")

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
        self.ai_win_probs: Dict[str, Optional[float]] = {}
        
        self.moves_since_last_turn = []   # now a list of dicts
        self.move_order_counter = 1

        # AI delay (ms) controlled from UI
        self.ai_delay_ms_var = tk.IntVar(value=3000)
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

        # Challenge banner state
        self.challenge_banner_until = None
        self.challenge_banner_text = ""
        self.challenge_banner_color = "green"
        self.challenge_banner_by = None

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
        checkpoint_display = "checkpoints\\test8\\gen_16\\final.pth"
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

        # --- AI delay control ---
        delay_frame = ttk.Frame(self.root)
        delay_frame.pack(pady=(0,10))
        ttk.Label(delay_frame, text="AI move delay (ms):").pack(side=tk.LEFT, padx=(0,6))
        delay_spin = ttk.Spinbox(delay_frame, from_=0, to=5000, increment=250,
                                textvariable=self.ai_delay_ms_var, width=7)
        delay_spin.pack(side=tk.LEFT)
        ttk.Label(delay_frame, text="(0–5000; try 3000–5000)").pack(side=tk.LEFT, padx=(6,0))

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

    def _do_ai_move(self, agent_id):
        """Execute the AI move after a scheduled delay."""
        env = self.current_env
        if env is None:
            return

        # If turn changed (e.g., game ended or someone else acted), just continue
        if env.agent_selection != agent_id:
            self.root.after(0, self.step_game)
            return

        try:
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                env.step(None)
                self.root.after(0, self.step_game)
                return

            observation = obs[agent_id]
            action = self.ai_agents[agent_id].get_action(
                env, agent_id, observation=observation, info=info
            )
            expert_info = self.ai_agents[agent_id].get_last_expert_info()
            self.ai_win_probs[agent_id] = getattr(expert_info, "win_probability", None)
            self.track_move(agent_id, action)

            # If AI challenges and your last play was the claim, show a 5s banner.
            try:
                if int(action) == 6 and env.last_action_agent == self.human_agent_id:
                    last_human_action = env.last_agent_action.get(self.human_agent_id)
                    if last_human_action in (0, 1, 2):
                        banner_color = "green"
                    elif last_human_action in (3, 4, 5):
                        banner_color = "red"
                    else:
                        banner_color = "green"

                    from datetime import datetime, timedelta
                    self.challenge_banner_until = datetime.now() + timedelta(seconds=5)
                    self.challenge_banner_color = banner_color
                    self.challenge_banner_by = agent_id
                    # Minimal banner: just a colored line (no text)
                    self.challenge_banner_text = ""
            except Exception:
                pass

            # Apply the move and continue
            env.step(action)
            if self._check_human_elimination_and_finish():
                return
            self.root.after(0, self.step_game)
        except Exception as exc:
            self.show_info(f"Error during AI move: {exc}")
            logger.exception("Error during AI move")

    def _check_human_elimination_and_finish(self):
        """If the human is eliminated (round or game), finish immediately."""
        env = self.current_env
        if env is None:
            return False
        me = self.human_agent_id

        human_out = env.terminations.get(me, False)


        # Also stop if all agents terminated (game ended)
        everyone_out = all(env.terminations.get(a, False) or env.truncations.get(a, False)
                        for a in env.possible_agents)

        if human_out or everyone_out:
            self.finish_game()
            return True
        return False

    def start_game(self):
        if self.loaded_checkpoint is None:
            self.show_info("Cannot start game: checkpoint not loaded.")
            return

        num_players = 4
        self.current_env = LiarsDeckEnv(num_players=num_players, render_mode="player")
        self.moves_since_last_turn = []
        self.move_order_counter = 1
        self.ai_win_probs = {}

        # You are always player_0
        self.human_agent_id = "player_0" if "player_0" in self.current_env.possible_agents else self.current_env.possible_agents[0]

        ai_agent_ids = [a for a in self.current_env.possible_agents if a != self.human_agent_id]

        if self.loaded_state_dict is None:
            self.show_info("No model state available in checkpoint.")
            return

        # Build AI agents
        self.ai_agents = {}
        for agent_id in ai_agent_ids:
            agent = PPOAutoregressiveAgent(device=self.device, player_id=agent_id)
            payload = {"policy_nets": {"agent_model": copy.deepcopy(self.loaded_state_dict)}}
            agent.load_models_from_checkpoint(payload, "agent_model")
            self.ai_agents[agent_id] = agent
            self.ai_win_probs[agent_id] = None

        try:
            self.current_env.reset()
        except Exception as exc:
            self.show_info(f"Error during game reset: {exc}")
            logger.exception("Error during game reset")
            return

        # Kick off the scheduler
        self.root.after(0, self.step_game)


    def step_game(self):
        env = self.current_env
        if env is None or env.agent_selection is None:
            self.finish_game()
            return

        current_agent = env.agent_selection
        obs, reward, termination, truncation, info = env.last()

        if termination or truncation:
            env.step(None)
            if self._check_human_elimination_and_finish():
                return
            self.root.after(0, self.step_game)
            return
        
        # --- AI turn: render first, then wait, then act ---
        if current_agent in self.ai_agents:
            # Show board (and effectively “thinking” delay)
            self.render_player_mode(env)

            delay = max(0, int(self.ai_delay_ms_var.get()))
            self.root.after(delay, lambda a=current_agent: self._do_ai_move(a))
            return

        # --- Human turn ---
        self.render_player_mode(env)
        # Reset moves log at start of your turn
        self.moves_since_last_turn = []
        self.move_order_counter = 1

        try:
            action = self.get_human_action(info['action_mask'])
            env.step(action)
            if self._check_human_elimination_and_finish():
                return
            self.track_move(self.human_agent_id, action)
            self.root.after(0, self.step_game)
        except Exception as exc:
            self.show_info(f"Error during human move: {exc}")
            logger.exception("Error during human move")


    def finish_game(self):
        if self.current_env is not None:
            try:
                self.show_game_result()
            except Exception:
                pass

        # Cleanup
        if self.current_env is not None:
            self.current_env.close()
            self.current_env = None
        for agent in self.ai_agents.values():
            try:
                agent.reset()
            except Exception:
                pass
        if self.live is not None:
            self.live.stop()
            self.live = None


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
        """Record an AI move made since your last turn (ordered & timestamped)."""
        # Normalize (handles np.int64, etc.)
        try:
            a = int(action)
        except Exception:
            a = action  # fall back; will hit the "performed action" branch
            
        # Map action -> description
        if a in (0, 1, 2):
            num_cards = a + 1            # 0,1,2 -> 1,2,3 table cards
            text = f"played {num_cards} card{'s' if num_cards > 1 else ''}"
        elif a in (3, 4, 5):
            num_cards = a - 2            # 3,4,5 -> 1,2,3 non-table cards
            text = f"played {num_cards} card{'s' if num_cards > 1 else ''}"
        elif a == 6:
            text = "challenged"
        else:
            text = f"performed action {a}"

        self.moves_since_last_turn.append({
            "idx": self.move_order_counter,
            "time": datetime.now().strftime("%H:%M:%S"),
            "agent": agent,
            "text": text,
        })
        self.move_order_counter += 1

    def render_player_mode(self, env):
        """Update the persistent Rich display (Live) for player mode."""
        current_agent = env.agent_selection
        # If it's your turn
        if current_agent == self.human_agent_id:
            header_text = "[bold green]Your Turn[/bold green]"
            border_color = "green"
        else:
            header_text = f"[bold red]Waiting for {current_agent}...[/bold red]"
            border_color = "red"

        header_panel = Panel(header_text, expand=False, border_style=border_color)

        table_card_panel = Panel(
            f"Table Card: [bold yellow]{env.table_card}[/bold yellow]",
            title="Table Card",
            border_style="bright_blue",
        )

        table_card_panel = Panel(
            f"Table Card: [bold yellow]{env.table_card}[/bold yellow]",
            title="Table Card",
            border_style="bright_blue",
        )

        # Always show *your* hand, not the current agent's.
        your_hand = env.players_hands.get(self.human_agent_id, [])
        your_hand_panel = Panel(
            f"[bold white]{your_hand}[/bold white]",
            title=f"Your Hand ({self.human_agent_id})",
            border_style="green",
        )

        # Opponents table
        opponent_table = Table(title="Opponent Hands", expand=True)
        opponent_table.add_column("Opponent", style="cyan", no_wrap=True)
        opponent_table.add_column("Cards Left", justify="center", style="magenta")
        opponent_table.add_column("Win Prob", justify="center", style="green")
        for agent in env.possible_agents:
            if agent != self.human_agent_id:
                hand = env.players_hands.get(agent, [])
                win_prob = self.ai_win_probs.get(agent)
                win_prob_str = f"{win_prob * 100:.1f}%" if win_prob is not None else "N/A"
                opponent_table.add_row(agent, f"{len(hand)} cards", win_prob_str)

        # Moves since last turn
        if self.moves_since_last_turn:
            moves_lines = [
                f"{m['idx']}) {m['time']}  {m['agent']} {m['text']}"
                for m in self.moves_since_last_turn
            ]
            moves_text = "\n".join(moves_lines)
        else:
            moves_text = "No moves since your last turn."
        moves_panel = Panel(moves_text, title="Moves Since Your Last Turn", border_style="blue")

        # Active players
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

        # Penalties
        penalties_table = Table(title="Penalties", expand=True)
        penalties_table.add_column("Agent", style="cyan", no_wrap=True)
        penalties_table.add_column("Penalty", style="bold")
        for agent in env.possible_agents:
            penalty = env.penalties.get(agent, 0)
            penalties_table.add_row(agent, str(penalty))

        # Layout
        layout = Layout()

        # Build optional challenge banner (single line using Rule)
        from datetime import datetime
        banner_active = self.challenge_banner_until is not None and datetime.now() <= self.challenge_banner_until
        if banner_active:
            layout.split(
                Layout(header_panel, name="header", size=3),
                Layout(name="banner", size=1),
                Layout(name="body", ratio=1),
            )
        else:
            # Clear expired banner state
            self.challenge_banner_until = None
            self.challenge_banner_text = ""
            self.challenge_banner_by = None
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

        # Fill banner content if present
        if banner_active:
            # If no explicit text provided, use a minimal caption.
            banner_text = self.challenge_banner_text or "CHALLENGED"
            layout["banner"].update(Rule(banner_text, style=self.challenge_banner_color))

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

    def on_closing(self):
        """Stop Live rendering, close env, and destroy root cleanly."""
        try:
            if self.live is not None:
                self.live.stop()
                self.live = None
            if self.current_env is not None:
                self.current_env.close()
                self.current_env = None
        except Exception:
            pass
        finally:
            # Force quit after destroying root (helps ensure exit)
            self.root.quit()
            self.root.destroy()
            os._exit(0)  # hard exit to terminate background threads


if __name__ == "__main__":
    import signal
    import sys

    root = tk.Tk()
    app = PlayVsAIGUI(root)

    # Properly handle Ctrl+C (KeyboardInterrupt)
    def handle_sigint(signum, frame):
        print("\nExiting gracefully...")
        app.on_closing()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        handle_sigint(None, None)
        sys.exit(0)
