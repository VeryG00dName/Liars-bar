#!/usr/bin/env python
import os
import logging
import torch
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Import your environment, agents, evaluation helpers, and model factory.
# Make sure these imports match your project structure.
from src.env.liars_deck_env_core import LiarsDeckEnv
from src import config
from src.model.hard_coded_agents import (
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    StrategicChallenger,
    RandomAgent,
    TableNonTableAgent,
    Classic
)
from src.evaluation.evaluate_utils import (
    get_hidden_dim_from_state_dict,
    evaluate_agents
)
from src.model.model_factory import ModelFactory
from src.training.train_vs_everyone import load_specific_historical_models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("AgentBattleground")

# --- Helper function (unchanged) ---
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

# --- Custom QListWidget for drag-and-drop ---
class DropListWidget(QtWidgets.QListWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                self.main_window.on_file_drop(file_path)
            event.acceptProposedAction()
        else:
            event.ignore()

# --- Worker thread to run the battleground matches ---
class BattlegroundWorker(QThread):
    progress_signal = pyqtSignal(int)
    results_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, ai_agents, historical_models, hardcoded_agents, rounds, parent=None):
        super().__init__(parent)
        self.ai_agents = ai_agents
        self.historical_models = historical_models
        self.hardcoded_agents = hardcoded_agents
        self.rounds = rounds

    def run(self):
        try:
            combined_opponents = {}
            # Add hardcoded agents.
            for name, cls in self.hardcoded_agents.items():
                combined_opponents[name] = ("hardcoded", cls)
            # Then add historical models.
            for identifier, hist_model in self.historical_models.items():
                combined_opponents[identifier] = ("historical", hist_model)

            total_matches = self.rounds * len(combined_opponents)
            progress_counter = 0
            results = {}

            for opp_name, (opp_type, opp_obj) in combined_opponents.items():
                wins = [0, 0, 0]  # [AI1 Wins, AI2 Wins, Opponent Wins]
                for _ in range(self.rounds):
                    winner = self.run_match(opp_type, opp_obj, opp_name)
                    if winner == "player_0":
                        wins[0] += 1
                    elif winner == "player_1":
                        wins[1] += 1
                    elif winner == "opponent":
                        wins[2] += 1
                    progress_counter += 1
                    self.progress_signal.emit(progress_counter)
                results[opp_name] = wins

            self.results_signal.emit(results)
        except Exception as e:
            self.error_signal.emit(str(e))
    
    def run_match(self, opponent_type, opponent_obj, opponent_name):
        env = LiarsDeckEnv(num_players=3, render_mode=None)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        players_in_this_game = {}

        # AI Agents (player_0 and player_1)
        for key in ["player_0", "player_1"]:
            agent_data = self.ai_agents[key]
            hidden_dim = get_hidden_dim_from_state_dict(agent_data["policy_net"], "fc1")
            obs_dim = agent_data["input_dim"]
            policy_net = ModelFactory.create_policy_network(
                input_dim=obs_dim,
                hidden_dim=hidden_dim,
                output_dim=env.action_spaces[key].n,
                use_aux_classifier=True,
                num_opponent_classes=config.NUM_OPPONENT_CLASSES
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
                "policy_net": policy_net,
                "obp_model": obp_model,
                "obs_version": agent_data["obs_version"],
                "rating": None,
                "uses_memory": agent_data["uses_memory"]
            }

        # Opponent as player_2.
        if opponent_type == "hardcoded":
            opponent_instance = opponent_obj(opponent_name)
            players_in_this_game["player_2"] = {
                "hardcoded_bot": True,
                "agent": opponent_instance,
                "obs_version": 2,
                "rating": None,
                "uses_memory": False
            }
        elif opponent_type == "historical":
            hist_state_dict = opponent_obj.state_dict()
            hidden_dim = get_hidden_dim_from_state_dict(hist_state_dict, "fc1")
            obs_dim = hist_state_dict["fc1.weight"].shape[1]
            policy_net = ModelFactory.create_policy_network(
                input_dim=obs_dim,
                hidden_dim=hidden_dim,
                output_dim=env.action_spaces["player_2"].n,
                use_aux_classifier=True,
                num_opponent_classes=config.NUM_OPPONENT_CLASSES
            )
            policy_net.load_state_dict(hist_state_dict, strict=False)
            policy_net.to(device).eval()

            obp_model = ModelFactory.create_obp(
                use_transformer_memory=True,
                input_dim=config.OPPONENT_INPUT_DIM,
                hidden_dim=config.OPPONENT_HIDDEN_DIM,
                output_dim=2
            )
            obp_model.to(device).eval()

            players_in_this_game["player_2"] = {
                "policy_net": policy_net,
                "obp_model": obp_model,
                "obs_version": 2,
                "rating": None,
                "uses_memory": True
            }
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        cumulative_wins, _, _, _, _ = evaluate_agents(env, device, players_in_this_game, episodes=1)
        winner = max(cumulative_wins, key=cumulative_wins.get)
        if winner in ["player_0", "player_1"]:
            return winner
        elif winner == "player_2":
            return "opponent"
        else:
            return "unknown"

# --- Main GUI class using PyQt with a Discord-like style ---
class AgentBattlegroundGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Agent Battleground")
        self.resize(1000, 700)

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

        # Load historical models (assumed to be PPO models in evaluation mode).
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.historical_models = {}
        hist_models_list = load_specific_historical_models(config.HISTORICAL_MODEL_DIR, device)
        for model, identifier in hist_models_list:
            self.historical_models[identifier] = model

        self.initUI()

    def initUI(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # --- Model Files Group ---
        model_files_group = QtWidgets.QGroupBox("Model Files")
        model_files_layout = QtWidgets.QVBoxLayout(model_files_group)
        self.file_list = DropListWidget(self)
        self.file_list.setMinimumHeight(60)
        model_files_layout.addWidget(self.file_list)
        drop_label = QtWidgets.QLabel("Drag and drop .pth files here")
        model_files_layout.addWidget(drop_label)
        main_layout.addWidget(model_files_group)

        # --- Model Info Group ---
        model_info_group = QtWidgets.QGroupBox("Model Info")
        model_info_layout = QtWidgets.QVBoxLayout(model_info_group)
        self.info_text = QtWidgets.QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFixedHeight(80)
        model_info_layout.addWidget(self.info_text)
        main_layout.addWidget(model_info_group)

        # --- AI Agents Selection Group ---
        ai_selection_group = QtWidgets.QGroupBox("AI Agents Selection")
        ai_selection_layout = QtWidgets.QGridLayout(ai_selection_group)
        self.agent_selectors = {}
        for i in range(2):
            label = QtWidgets.QLabel(f"AI Agent {i+1}:")
            ai_selection_layout.addWidget(label, i, 0)
            combo = QtWidgets.QComboBox()
            combo.setEditable(False)
            ai_selection_layout.addWidget(combo, i, 1)
            self.agent_selectors[i] = combo
        main_layout.addWidget(ai_selection_group)

        # --- Control Buttons Layout ---
        control_layout = QtWidgets.QHBoxLayout()
        refresh_button = QtWidgets.QPushButton("Refresh Agents")
        refresh_button.clicked.connect(self.update_agent_selectors)
        control_layout.addWidget(refresh_button)
        start_button = QtWidgets.QPushButton("Start Battleground")
        start_button.clicked.connect(self.start_battleground)
        control_layout.addWidget(start_button)
        rounds_label = QtWidgets.QLabel("Rounds:")
        control_layout.addWidget(rounds_label)
        self.rounds_spinbox = QtWidgets.QSpinBox()
        self.rounds_spinbox.setMinimum(1)
        self.rounds_spinbox.setMaximum(1000)
        self.rounds_spinbox.setValue(20)
        control_layout.addWidget(self.rounds_spinbox)
        main_layout.addLayout(control_layout)

        # --- Progress Bar ---
        progress_layout = QtWidgets.QHBoxLayout()
        progress_label = QtWidgets.QLabel("Progress:")
        progress_layout.addWidget(progress_label)
        self.progress_bar = QtWidgets.QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        main_layout.addLayout(progress_layout)

        # --- Results Group ---
        results_group = QtWidgets.QGroupBox("Results")
        results_layout = QtWidgets.QVBoxLayout(results_group)
        self.results_text = QtWidgets.QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(300)
        self.results_text.setSizePolicy(QtWidgets.QSizePolicy.Expanding, 
                                QtWidgets.QSizePolicy.Expanding)
        results_layout.addWidget(self.results_text)
        main_layout.addWidget(results_group)

    def on_file_drop(self, file_path):
        file_path = file_path.strip()
        if not file_path.endswith(".pth"):
            self.show_info("Only .pth files are supported")
            return
        if file_path in self.loaded_models:
            self.show_info("Model already loaded")
            return
        try:
            self.load_model(file_path)
            self.file_list.addItem(os.path.basename(file_path))
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
        self.info_text.setPlainText(message)

    def update_agent_selectors(self):
        agent_options = []
        for file_path, data in self.loaded_models.items():
            folder_name = os.path.basename(os.path.dirname(file_path))
            for agent_name in data["policy_nets"].keys():
                display_text = f"{folder_name} - {os.path.basename(file_path)} - {agent_name}"
                agent_options.append(display_text)
        for i in range(2):
            self.agent_selectors[i].clear()
            self.agent_selectors[i].addItems(agent_options)
            if agent_options:
                self.agent_selectors[i].setCurrentIndex(0)

    def load_selected_agents(self):
        """Loads the selected AI agents from the selectors."""
        ai_agents = {}
        try:
            for i in range(2):
                selection = self.agent_selectors[i].currentText()
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

    def start_battleground(self):
        ai_agents = self.load_selected_agents()
        if not ai_agents:
            return
        rounds = self.rounds_spinbox.value()
        total_matches = rounds * (len(self.hardcoded_agents) + len(self.historical_models))
        self.progress_bar.setMaximum(total_matches)
        self.progress_bar.setValue(0)
        self.results_text.clear()
        self.worker = BattlegroundWorker(ai_agents, self.historical_models, self.hardcoded_agents, rounds)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.results_signal.connect(self.display_results)
        self.worker.error_signal.connect(lambda msg: self.show_info(f"Error: {msg}"))
        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    # --- Here is the HTML table approach for displaying results ---
    def display_results(self, results):
        # Build an HTML table with some inline styling
        html = """
        <table style="border: 1px solid #7289da; border-collapse: collapse; width: 100%;">
          <thead>
            <tr style="background-color: #4f545c;">
              <th style="border: 1px solid #7289da; padding: 8px;">Opponent Name</th>
              <th style="border: 1px solid #7289da; padding: 8px;">AI1 Wins</th>
              <th style="border: 1px solid #7289da; padding: 8px;">AI2 Wins</th>
              <th style="border: 1px solid #7289da; padding: 8px;">Opponent Wins</th>
              <th style="border: 1px solid #7289da; padding: 8px;">AI1 Win Rate</th>
              <th style="border: 1px solid #7289da; padding: 8px;">AI2 Win Rate</th>
              <th style="border: 1px solid #7289da; padding: 8px;">Opponent Win Rate</th>
            </tr>
          </thead>
          <tbody>
        """

        for opp_name, wins in results.items():
            ai1_wins, ai2_wins, opp_wins = wins
            total = ai1_wins + ai2_wins + opp_wins
            rate1 = ai1_wins / total if total > 0 else 0.0
            rate2 = ai2_wins / total if total > 0 else 0.0
            rate_opp = opp_wins / total if total > 0 else 0.0

            row = f"""
            <tr>
              <td style="border: 1px solid #7289da; padding: 6px;">{opp_name}</td>
              <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{ai1_wins}</td>
              <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{ai2_wins}</td>
              <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{opp_wins}</td>
              <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{rate1:.2%}</td>
              <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{rate2:.2%}</td>
              <td style="border: 1px solid #7289da; padding: 6px; text-align: center;">{rate_opp:.2%}</td>
            </tr>
            """
            html += row

        html += """
          </tbody>
        </table>
        """
        self.results_text.setHtml(html)

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    # Apply a dark, Discord-like style
    dark_stylesheet = """
    QWidget {
        background-color: #2f3136;
        color: #dcddde;
        font-family: "Helvetica", "Arial", sans-serif;
    }
    QGroupBox {
        border: 1px solid #202225;
        border-radius: 4px;
        margin-top: 1ex;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 3px;
        color: #fff;
    }
    QPushButton {
        background-color: #7289da;
        border: none;
        border-radius: 4px;
        padding: 5px 10px;
        color: #fff;
    }
    QPushButton:hover {
        background-color: #5b6eae;
    }
    QLineEdit, QComboBox, QSpinBox, QTextEdit, QListWidget {
        background-color: #36393f;
        border: 1px solid #202225;
        border-radius: 4px;
        padding: 4px;
    }
    QProgressBar {
        background-color: #36393f;
        border: 1px solid #202225;
        border-radius: 4px;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #7289da;
        border-radius: 4px;
    }
    """
    app.setStyleSheet(dark_stylesheet)

    window = AgentBattlegroundGUI()
    window.show()
    sys.exit(app.exec_())
