# src/evaluation/eval_gui.py
import itertools
import os
from typing import Any, Dict, Optional, Type
from PyQt5 import QtCore
from src.agents.base_agent import BaseAgent
from src.agents.hardcoded_agent_wrapper import HardcodedAgentWrapper
from src.agents.agent_factory import AgentFactory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal
# Import your environment, agents, evaluation helpers, and model factory.
from src.env.liars_deck_env_core import LiarsDeckEnv
from src import config
from src.model.hard_coded_agents import (
    GreedyCardSpammer,
    TableFirstConservativeChallenger,
    StrategicChallenger,
    SelectiveTableConservativeChallenger,
    RandomAgent,
    TableNonTableAgent,
    Classic
)
from src.eval.evaluate_utils import evaluate_agents

from src.misc.cheat.ai_vs_hardcoded_cheat import LABELS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("AgentBattleground")
logger.propagate = True

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
    expert_signal = pyqtSignal(dict)  # New signal for expert activations
    error_signal = pyqtSignal(str)

    def __init__(self,
                 selected_ai_agent_configs: Dict[str, Dict],
                 historical_model_configs: Dict[str, Dict[str, str]],
                 hardcoded_agent_classes: Dict[str, type],
                 rounds: int,
                 device: torch.device,
                 num_players: int = 3,
                 two_player: Optional[str] = None,
                 cheat: bool = False,
                 onev2: bool = False,
                 team_mode: bool = False, # RENAMED from duo
                 clear_memory: bool = False,
                 parent=None):
        super().__init__(parent)
        self.selected_ai_agent_configs = selected_ai_agent_configs
        self.historical_model_configs = historical_model_configs
        self.hardcoded_agent_classes = hardcoded_agent_classes
        self.rounds = rounds
        self.device = device
        self.num_players = num_players
        self.two_player = two_player
        self.cheat = cheat
        self.onev2 = onev2
        self.team_mode = team_mode # RENAMED
        self.expert_activations = {}
        self.clear_memory = clear_memory
        self.agent_factory = AgentFactory(self.device)

    def run(self):
        # --- Prepare Opponent Pool ---
        opponent_pool = {} # Maps opponent_name -> (type, config)

        for name, cls in self.hardcoded_agent_classes.items():
            opponent_pool[name] = ("hardcoded", self.agent_factory.create_hardcoded_agent_config(cls, name))

        for identifier, hist_config in self.historical_model_configs.items():
            opponent_pool[identifier] = ("historical_ai", {
                'path': hist_config['path'],
                'key': hist_config['key'],
                'id_prefix': identifier
            })

        progress_counter = 0
        total_matches_estimate = self.rounds * len(opponent_pool)
        if self.team_mode:
            num_opponents_in_team = self.num_players - 1
            num_combinations = len(list(itertools.combinations(opponent_pool.items(), num_opponents_in_team)))
            total_matches_estimate = self.rounds * num_combinations

        results = {}
        self.expert_activations = {}

        try:
            # --- Main Loop (Team, OneV2, Standard) ---
            if self.team_mode:
                num_opponents_in_team = self.num_players - 1
                opponent_teams = list(itertools.combinations(opponent_pool.items(), num_opponents_in_team))
                logger.info(f"Running 1 AI vs Team Mode ({self.num_players} players) with {len(opponent_teams)} opponent teams.")

                for team_tuple in opponent_teams:
                    opponent_display_name = "+".join([name for name, _ in team_tuple])
                    opponent_configs = tuple(config for _, (_, config) in team_tuple)
                    opponent_types = tuple(type for _, (type, _) in team_tuple)

                    cheat_indices = [LABELS.get(name) for name, _ in team_tuple]
                    current_cheat_index = tuple(cheat_indices) if self.cheat and all(idx is not None for idx in cheat_indices) else None

                    cumulative_wins, expert_acts, player_id_map = self.run_match(
                        opponent_configs=opponent_configs,
                        opponent_types=opponent_types,
                        opponent_display_name=opponent_display_name,
                        episodes=self.rounds,
                        progress_callback=lambda ep: self.progress_signal.emit(progress_counter + ep),
                        cheat_expert_index=current_cheat_index,
                        mode='team', # Pass mode
                        clear_memory=self.clear_memory
                    )

                    results[opponent_display_name] = self._format_wins(cumulative_wins, player_id_map)
                    self.expert_activations[opponent_display_name] = expert_acts
                    progress_counter += self.rounds
                    self.progress_signal.emit(progress_counter)

            elif self.onev2:
                 logger.info(f"Running 1v2 Mode against {len(opponent_pool)} opponents.")
                 for opp_name, (opp_type, opp_config) in opponent_pool.items():
                     opponent_display_name = f"{opp_name}(x2)"
                     cheat_idx = LABELS.get(opp_name)
                     current_cheat_index = (cheat_idx, cheat_idx) if self.cheat and cheat_idx is not None else None

                     cumulative_wins, expert_acts, player_id_map = self.run_match(
                         opponent_configs=(opp_config, opp_config),
                         opponent_types=(opp_type, opp_type),
                         opponent_display_name=opponent_display_name,
                         episodes=self.rounds,
                         progress_callback=lambda ep: self.progress_signal.emit(progress_counter + ep),
                         cheat_expert_index=current_cheat_index,
                         mode='onev2',
                         clear_memory=self.clear_memory
                     )
                     results[opponent_display_name] = self._format_wins(cumulative_wins, player_id_map)
                     self.expert_activations[opponent_display_name] = expert_acts
                     progress_counter += self.rounds
                     self.progress_signal.emit(progress_counter)

            else: # Standard mode
                 logger.info(f"Running Standard Mode ({self.num_players} players) against {len(opponent_pool)} opponents.")
                 for opp_name, (opp_type, opp_config) in opponent_pool.items():
                     cheat_idx = LABELS.get(opp_name)
                     current_cheat_index = cheat_idx if self.cheat and cheat_idx is not None else None

                     cumulative_wins, expert_acts, player_id_map = self.run_match(
                         opponent_configs=(opp_config,),
                         opponent_types=(opp_type,),
                         opponent_display_name=opp_name,
                         episodes=self.rounds,
                         progress_callback=lambda ep: self.progress_signal.emit(progress_counter + ep),
                         cheat_expert_index=current_cheat_index,
                         mode='standard',
                         clear_memory=self.clear_memory
                     )
                     results[opp_name] = self._format_wins(cumulative_wins, player_id_map)
                     self.expert_activations[opp_name] = expert_acts
                     progress_counter += self.rounds
                     self.progress_signal.emit(progress_counter)

            logger.info("Battleground run finished.")
            self.expert_signal.emit(self.expert_activations)
            self.results_signal.emit(results)

        except Exception as e:
            logger.error(f"Error during battleground run: {e}", exc_info=True)
            self.error_signal.emit(str(e))

    def _format_wins(self, cumulative_wins: Dict[str, int], player_id_map: Dict[str, str]) -> list:
        """Formats wins into [p0_wins, p1_wins, p2_wins, ...] using the player_id -> env_id map."""
        formatted = [0] * self.num_players
        env_id_to_wins = {env_id: cumulative_wins.get(pid, 0) for pid, env_id in player_id_map.items()}

        for i in range(self.num_players):
            env_id = f"player_{i}"
            formatted[i] = env_id_to_wins.get(env_id, 0)

        return formatted

    def run_match(self, opponent_configs: tuple, opponent_types: tuple, opponent_display_name: str, episodes: int, mode:str, progress_callback=None, cheat_expert_index=None,clear_memory=False):
        """Runs a single match configuration using the AgentFactory."""
        env = LiarsDeckEnv(num_players=self.num_players, render_mode=None)
        players_for_eval: Dict[str, BaseAgent] = {} # Maps env_id -> Agent object

        try:
            # --- Instantiate AI Agents ---
            num_ai_agents = 1 if mode in ['team', 'onev2'] else (self.num_players - 1)
            for i in range(num_ai_agents):
                env_id = f"player_{i}"
                if env_id not in self.selected_ai_agent_configs:
                     raise ValueError(f"Configuration missing for required AI agent {env_id} in {mode} mode.")
                ai_config = self.selected_ai_agent_configs[env_id]
                agent = self.agent_factory.create_agent_from_checkpoint(
                    checkpoint_path=ai_config['path'],
                    player_id_prefix=ai_config['id_prefix'],
                    agent_key=ai_config['key']
                )
                players_for_eval[env_id] = agent

            # --- Instantiate Opponent(s) ---
            num_opponents = len(opponent_configs)
            for i in range(num_opponents):
                 env_id = f"player_{num_ai_agents + i}"
                 opp_config = opponent_configs[i]
                 opp_type = opponent_types[i]

                 if opp_type == "hardcoded":
                     hc_class = opp_config['class']
                     hc_name = opp_config['name']
                     player_id = f"Hardcoded_{hc_name}"
                     try:
                          agent_index = int(env_id.split('_')[-1])
                          hc_instance = hc_class(hc_name, self.num_players, agent_index)
                          logger.debug(f"Instantiated {hc_name} with name, num_players={self.num_players}, agent_index={agent_index}")
                     except TypeError:
                          hc_instance = hc_class(hc_name)
                          logger.debug(f"Instantiated {hc_name} with just name.")
                     except Exception as e:
                          logger.error(f"Failed to instantiate hardcoded agent {hc_name}: {e}", exc_info=True)
                          raise ValueError(f"Cannot instantiate hardcoded agent {hc_name}") from e
                     agent = HardcodedAgentWrapper(hc_instance, self.device, player_id)

                 elif opp_type == "historical_ai":
                     agent = self.agent_factory.create_agent_from_checkpoint(
                         checkpoint_path=opp_config['path'],
                         player_id_prefix=opp_config['id_prefix'],
                         agent_key=opp_config['key']
                     )
                 else:
                     raise ValueError(f"Unsupported opponent type: {opp_type}")

                 players_for_eval[env_id] = agent

            # --- Run Evaluation ---
            logger.info(f"Starting match vs {opponent_display_name}. Mode: {mode}. Agents: { {env_id: agent.get_player_id() for env_id, agent in players_for_eval.items()} }")
            cumulative_wins, _, _, _, _, expert_activations, player_id_map = evaluate_agents(
                env,
                self.device,
                players_for_eval,
                episodes=episodes,
                two_player=self.two_player,
                track_experts=True,
                progress_callback=progress_callback,
                cheat_expert_index=cheat_expert_index,
                clear_memory=clear_memory,
            )

            return cumulative_wins, expert_activations, player_id_map

        except Exception as e:
             logger.error(f"Error running match against {opponent_display_name}: {e}", exc_info=True)
             return {}, {}, {}

# --- Main GUI class using PyQt with a Discord-like style ---
class AgentBattlegroundGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Agent Battleground")
        self.resize(1000, 750)
        self.loaded_model_paths: Dict[str, Dict[str, Any]] = {}
        self.hardcoded_agents: Dict[str, Type] = {
             "Classic": Classic, "GreedyCardSpammer": GreedyCardSpammer, "RandomAgent": RandomAgent,
             "SelectiveTableConservativeChallenger": SelectiveTableConservativeChallenger,
             "StrategicChallenger": StrategicChallenger, "TableFirstConservativeChallenger": TableFirstConservativeChallenger,
             "TableNonTableAgent": TableNonTableAgent
        }
        self.historical_model_configs: Dict[str, Dict[str, str]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.load_historical_models()
        self.previous_results: Optional[Dict[str, list]] = None
        self.current_results: Optional[Dict[str, list]] = None
        self.expert_activations: Optional[Dict[str, Dict[str, Any]]] = None
        self.worker: Optional[BattlegroundWorker] = None
        self.initUI()


    def initUI(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        model_files_group = QtWidgets.QGroupBox("Load AI Model Files (.pth)")
        model_files_layout = QtWidgets.QVBoxLayout(model_files_group)
        self.file_list = DropListWidget(self); self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        model_files_layout.addWidget(self.file_list)
        drop_label = QtWidgets.QLabel("Drag and drop model files here."); drop_label.setAlignment(QtCore.Qt.AlignCenter); drop_label.setStyleSheet("color: #aaa;")
        model_files_layout.addWidget(drop_label)
        main_layout.addWidget(model_files_group)

        model_info_group = QtWidgets.QGroupBox("Status / Info")
        model_info_layout = QtWidgets.QVBoxLayout(model_info_group)
        self.info_text = QtWidgets.QTextEdit(); self.info_text.setReadOnly(True); self.info_text.setFixedHeight(80); self.info_text.setPlaceholderText("Status messages...")
        model_info_layout.addWidget(self.info_text)
        main_layout.addWidget(model_info_group)

        ai_selection_group = QtWidgets.QGroupBox("Select AI Agents for Battle")
        ai_selection_layout = QtWidgets.QGridLayout(ai_selection_group)
        self.agent_selectors: Dict[int, QtWidgets.QComboBox] = {}
        for i in range(3): # Max 3 AI agents for 4p standard
            label = QtWidgets.QLabel(f"AI Agent {i+1}:"); ai_selection_layout.addWidget(label, i, 0)
            combo = QtWidgets.QComboBox(); combo.setEditable(False); combo.setMinimumWidth(300); combo.addItem("No models loaded"); combo.setEnabled(False)
            ai_selection_layout.addWidget(combo, i, 1); self.agent_selectors[i] = combo
        main_layout.addWidget(ai_selection_group)

        # --- Layout: Controls & Options ---
        control_layout = QtWidgets.QHBoxLayout()
        refresh_button = QtWidgets.QPushButton("Refresh Agents"); refresh_button.setToolTip("Update agent selection dropdowns."); refresh_button.clicked.connect(self.update_agent_selectors)
        control_layout.addWidget(refresh_button)
        control_layout.addStretch(1)
        rounds_label = QtWidgets.QLabel("Rounds:"); control_layout.addWidget(rounds_label)
        self.rounds_spinbox = QtWidgets.QSpinBox(); self.rounds_spinbox.setMinimum(1); self.rounds_spinbox.setMaximum(10000); self.rounds_spinbox.setValue(20); self.rounds_spinbox.setFixedWidth(80)
        control_layout.addWidget(self.rounds_spinbox)
        control_layout.addStretch(1)

        # --- Mode Selection ---
        mode_group = QtWidgets.QGroupBox("Mode")
        mode_layout = QtWidgets.QHBoxLayout(mode_group)
        self.standard_mode_radio = QtWidgets.QRadioButton("Standard")
        self.onev2_mode_radio = QtWidgets.QRadioButton("1 AI vs 2 Opponents")
        self.team_mode_radio = QtWidgets.QRadioButton("1 AI vs Team") # RENAMED
        self.team_mode_radio.setToolTip("1 AI vs a team of unique opponents (2 in 3p, 3 in 4p).") # TOOLTIP
        self.standard_mode_radio.setChecked(True)
        self.standard_mode_radio.toggled.connect(self._update_agent_selector_states)
        self.onev2_mode_radio.toggled.connect(self._update_agent_selector_states)
        self.team_mode_radio.toggled.connect(self._update_agent_selector_states)
        mode_layout.addWidget(self.standard_mode_radio)
        mode_layout.addWidget(self.onev2_mode_radio)
        mode_layout.addWidget(self.team_mode_radio)
        control_layout.addWidget(mode_group)

        # --- Other Options ---
        options_group = QtWidgets.QGroupBox("Options")
        options_layout = QtWidgets.QHBoxLayout(options_group)
        self.four_player_checkbox = QtWidgets.QCheckBox("4 Player Mode")
        self.four_player_checkbox.toggled.connect(self._update_agent_selector_states)
        options_layout.addWidget(self.four_player_checkbox)
        self.disable_historical_checkbox = QtWidgets.QCheckBox("Disable Historical")
        options_layout.addWidget(self.disable_historical_checkbox)
        self.two_player_checkbox = QtWidgets.QCheckBox("2 Player Only"); self.two_player_checkbox.setToolTip("Eliminate one player at the start of a 3-player game.")
        options_layout.addWidget(self.two_player_checkbox)
        self.combine_results_checkbox = QtWidgets.QCheckBox("Combine Results")
        self.combine_results_checkbox.setToolTip("Combine AI wins (Standard) or Opponent wins (1vTeam).")
        self.combine_results_checkbox.stateChanged.connect(self.update_results_display)
        options_layout.addWidget(self.combine_results_checkbox)
        self.cheat_checkbox = QtWidgets.QCheckBox("Cheat (Expert Index)"); self.cheat_checkbox.setToolTip("Provide opponent type index via LABELS.")
        options_layout.addWidget(self.cheat_checkbox)
        self.clear_memory_checkbox = QtWidgets.QCheckBox("Clear Memory")
        self.clear_memory_checkbox.setToolTip("If enabled, clears belief memory and sequence history after each game.")
        options_layout.addWidget(self.clear_memory_checkbox)
        control_layout.addWidget(options_group)

        control_layout.addStretch(2)
        self.start_button = QtWidgets.QPushButton(" Start Battleground "); self.start_button.setStyleSheet("padding: 8px 15px; font-weight: bold;")
        self.start_button.clicked.connect(self.start_battleground)
        control_layout.addWidget(self.start_button)
        main_layout.addLayout(control_layout)

        # --- Progress Bar & Results ---
        progress_layout = QtWidgets.QHBoxLayout(); progress_label = QtWidgets.QLabel("Progress:"); progress_layout.addWidget(progress_label)
        self.progress_bar = QtWidgets.QProgressBar(); self.progress_bar.setTextVisible(True); progress_layout.addWidget(self.progress_bar)
        main_layout.addLayout(progress_layout)
        results_analysis_group = QtWidgets.QGroupBox("Results / Analysis")
        results_analysis_layout = QtWidgets.QHBoxLayout(results_analysis_group)
        self.results_text = QtWidgets.QTextEdit(); self.results_text.setReadOnly(True); self.results_text.setMinimumHeight(300); self.results_text.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding); self.results_text.setPlaceholderText("Results...")
        results_analysis_layout.addWidget(self.results_text, 4)
        analysis_button_layout = QtWidgets.QVBoxLayout(); analysis_button_layout.setAlignment(QtCore.Qt.AlignTop)
        self.compare_button = QtWidgets.QPushButton("Compare Results"); self.compare_button.setToolTip("Compare current vs previous results."); self.compare_button.clicked.connect(self.compare_results)
        analysis_button_layout.addWidget(self.compare_button)
        self.expert_button = QtWidgets.QPushButton("Show Expert Usage"); self.expert_button.setToolTip("Analyze MoE/Gating/Belief activations."); self.expert_button.clicked.connect(self.show_expert_usage)
        analysis_button_layout.addWidget(self.expert_button)
        analysis_button_layout.addStretch(1)
        results_analysis_layout.addLayout(analysis_button_layout, 1)
        main_layout.addWidget(results_analysis_group, 1)

        self._update_agent_selector_states()

    def load_historical_models(self):
        self.historical_model_configs.clear()
        hist_dir = config.HISTORICAL_MODEL_DIR
        required_models = {"Version_A": "player_2", "Version_C": "player_0", "Version_E": "player_1"}
        logger.info(f"Attempting to load specific historical model configs from base: {hist_dir}")
        found_count = 0
        for version_dir_name, player_key in required_models.items():
            version_path = os.path.join(hist_dir, version_dir_name)
            identifier = f"{version_dir_name}_{player_key}"
            if os.path.isdir(version_path):
                checkpoint_file = next((f for f in os.listdir(version_path) if f.endswith(".pth")), None)
                if checkpoint_file:
                    full_path = os.path.join(version_path, checkpoint_file)
                    self.historical_model_configs[identifier] = {'path': full_path, 'key': player_key}
                    found_count += 1
                    logger.info(f"Found config: {identifier} at {full_path}")
                else:
                    logger.warning(f"No .pth found in {version_path} for {identifier}")
            else:
                logger.warning(f"Directory not found: {version_path} for {identifier}")
        logger.info(f"Found configs for {found_count} specific historical models.")

    def _update_agent_selector_states(self):
        is_standard_mode = self.standard_mode_radio.isChecked()
        is_team_mode = self.team_mode_radio.isChecked()
        is_4p_mode = self.four_player_checkbox.isChecked()
        models_loaded = self.agent_selectors[0].count() > 1
    
        self.agent_selectors[0].setEnabled(models_loaded)
        self.agent_selectors[1].setEnabled(is_standard_mode and models_loaded)
        self.agent_selectors[2].setEnabled(is_standard_mode and is_4p_mode and models_loaded)
    
        # 1v2 mode is only for 3-player games
        self.onev2_mode_radio.setEnabled(not is_4p_mode)
        if is_4p_mode and self.onev2_mode_radio.isChecked():
            self.standard_mode_radio.setChecked(True)
            
        self.combine_results_checkbox.setEnabled(is_standard_mode or is_team_mode)
        if not (is_standard_mode or is_team_mode):
            self.combine_results_checkbox.setChecked(False)

    def on_file_drop(self, file_path):
        file_path = file_path.strip()
        if not file_path.endswith(".pth"):
            self.show_info("Only .pth files are supported")
            return

        base_display_name = f"{os.path.basename(os.path.dirname(file_path))}/{os.path.basename(file_path)}"

        try:
            ckpt = torch.load(file_path, map_location='cpu', weights_only=False)
            policy_keys = []
            if 'policy_nets' in ckpt:
                 policy_keys = list(ckpt['policy_nets'].keys())
            elif 'model' in ckpt:
                 policy_keys = ['player_0']
            else:
                 self.show_info(f"Checkpoint '{base_display_name}' has no recognizable policy keys ('policy_nets' or 'model').")
                 return

            if not policy_keys:
                 self.show_info(f"No player keys found within policy networks in '{base_display_name}'.")
                 return

            added_count = 0
            for key in sorted(policy_keys):
                display_name_with_key = f"{base_display_name} [{key}]"
                if display_name_with_key in self.loaded_model_paths:
                    continue
                self.loaded_model_paths[display_name_with_key] = {'path': file_path, 'key': key, 'display_name': base_display_name}
                self.file_list.addItem(display_name_with_key)
                added_count += 1

            if added_count > 0:
                self.update_agent_selectors()
                self.show_info(f"Added {added_count} player(s) from: {base_display_name}")
            else:
                 self.show_info(f"All players from '{base_display_name}' were already added.")

        except Exception as e:
            logger.error(f"Error inspecting checkpoint {file_path}: {e}", exc_info=True)
            self.show_info(f"Error reading checkpoint: {os.path.basename(file_path)}")

    def get_selected_ai_configs(self) -> Dict[str, Dict]:
        selected_configs = {}
        is_4p = self.four_player_checkbox.isChecked()
        is_standard = self.standard_mode_radio.isChecked()

        if self.onev2_mode_radio.isChecked() or self.team_mode_radio.isChecked():
            num_ai_needed = 1
        elif is_standard and is_4p:
            num_ai_needed = 3
        elif is_standard and not is_4p:
            num_ai_needed = 2
        else:
            num_ai_needed = 2

        for i in range(num_ai_needed):
             env_id = f"player_{i}"
             selector = self.agent_selectors[i]
             selected_index = selector.currentIndex()
             if selected_index <= 0:
                  raise ValueError(f"Please select a model for AI Agent {i+1}.")

             item_data = selector.itemData(selected_index)
             if not item_data or 'path' not in item_data or 'key' not in item_data:
                  display_name_with_key = selector.itemText(selected_index)
                  raise ValueError(f"Missing data for selected agent: {display_name_with_key}. Please reload models.")

             path = item_data['path']
             agent_key = item_data['key']
             display_name = item_data['display_name']
             prefix = display_name.replace("/", "_").replace("\\", "_").replace(".pth", "")
             selected_configs[env_id] = {
                 'path': path, 'key': agent_key, 'id_prefix': prefix
             }
             selected_configs[env_id]['id'] = f"{prefix}_{agent_key}"

        return selected_configs

    def show_info(self, message):
        self.info_text.setPlainText(message)

    def update_agent_selectors(self):
        agent_options = sorted(list(self.loaded_model_paths.keys()))

        for i in range(3):
            selector = self.agent_selectors[i]
            current_index = selector.currentIndex()
            current_data = selector.itemData(current_index) if current_index > 0 else None

            selector.clear()
            if not agent_options:
                 selector.addItem("No models loaded")
                 selector.setEnabled(False)
            else:
                 selector.addItem("Select Model...")
                 for display_name_with_key in agent_options:
                      item_data = self.loaded_model_paths[display_name_with_key]
                      selector.addItem(display_name_with_key, userData=item_data)

                 restored = False
                 if current_data:
                      for idx in range(1, selector.count()):
                           if selector.itemData(idx) == current_data:
                                selector.setCurrentIndex(idx)
                                restored = True
                                break
                 if not restored:
                      selector.setCurrentIndex(0)

        self._update_agent_selector_states()

    def start_battleground(self):
        if self.worker is not None and self.worker.isRunning():
            self.show_info("A battleground run is already in progress.")
            return

        try:
            selected_ai_configs = self.get_selected_ai_configs()
        except ValueError as e:
            self.show_info(f"Selection Error: {e}")
            return
        
        historical_configs = {} if self.disable_historical_checkbox.isChecked() else self.historical_model_configs
        two_player_param = "player_2" if self.two_player_checkbox.isChecked() else None # Usually P2 is last in 3p
        cheat_flag = self.cheat_checkbox.isChecked()
        clear_memory_flag = self.clear_memory_checkbox.isChecked()
        onev2_enabled = self.onev2_mode_radio.isChecked()
        team_mode_enabled = self.team_mode_radio.isChecked()
        num_players = 4 if self.four_player_checkbox.isChecked() else 3
        rounds = self.rounds_spinbox.value()

        num_hardcoded = len(self.hardcoded_agents)
        num_historical = len(historical_configs)
        num_opponents_total = num_hardcoded + num_historical
        if num_opponents_total == 0:
             self.show_info("No opponents (hardcoded or historical) found to run against.")
             return

        if team_mode_enabled:
            num_opp_in_team = num_players - 1
            if num_opponents_total < num_opp_in_team:
                self.show_info(f"Not enough opponents ({num_opponents_total}) to form teams of {num_opp_in_team}.")
                return
            num_combinations = len(list(itertools.combinations(range(num_opponents_total), num_opp_in_team)))
            total_matches = rounds * num_combinations
        else:
            total_matches = rounds * num_opponents_total

        self.progress_bar.setMaximum(max(1, total_matches))
        self.progress_bar.setValue(0)

        if self.results_text.toPlainText().strip():
             self.previous_results = self.current_results
        self.results_text.clear()
        self.expert_activations = None

        self.start_button.setEnabled(False)
        mode_str = "1vTeam" if team_mode_enabled else "1v2" if onev2_enabled else "Standard"
        self.show_info(f"Starting battleground ({mode_str} mode, {num_players} players)...")

        self.worker = BattlegroundWorker(
            selected_ai_agent_configs=selected_ai_configs,
            historical_model_configs=historical_configs,
            hardcoded_agent_classes=self.hardcoded_agents,
            rounds=rounds,
            device=self.device,
            num_players=num_players,
            two_player=two_player_param,
            cheat=cheat_flag,
            onev2=onev2_enabled,
            team_mode=team_mode_enabled,
            clear_memory=clear_memory_flag
        )
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.results_signal.connect(self.on_results_received)
        self.worker.expert_signal.connect(self.store_expert_activations)
        self.worker.error_signal.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def store_expert_activations(self, expert_activations):
        self.expert_activations = expert_activations
        logger.info(f"Received expert activations for {len(expert_activations)} opponents")

    def on_results_received(self, results):
         self.display_results(results)
         self.show_info("Battleground finished.")

    def on_worker_error(self, error_message):
         self.show_info(f"Worker Error: {error_message}")
         logger.error(f"Worker thread reported error: {error_message}")
         self.start_button.setEnabled(True)

    def on_worker_finished(self):
         self.start_button.setEnabled(True)
         if not self.current_results:
              self.show_info("Battleground worker finished.")

    def display_results(self, results: Dict[str, list]):
        self.current_results = results
        combine = self.combine_results_checkbox.isChecked()
        is_onev2 = self.onev2_mode_radio.isChecked()
        is_team_mode = self.team_mode_radio.isChecked()

        html = self._generate_results_html(results, combine, is_onev2, is_team_mode)
        self.results_text.setHtml(html)

    def _generate_results_html(self, results, combine, is_onev2, is_team_mode):
        """Generates the results HTML table based on mode and combine flag."""
        html = """<table style="border: 1px solid #7289da; border-collapse: collapse; width: 100%;"><thead><tr style="background-color: #4f545c;">"""
        num_players = 4 if self.four_player_checkbox.isChecked() else 3
        num_ai_agents_standard = num_players - 1
        num_opponents_team = num_players - 1
        num_data_cols = 0 # To help with summary row layout

        # --- Table Header Logic ---
        if is_onev2 or is_team_mode:
            html += """<th style="border: 1px solid #7289da; padding: 8px;">Opponent Team</th>
                       <th style="border: 1px solid #7289da; padding: 8px;">AI Wins (P0)</th>"""
            if combine:
                html += """<th style="border: 1px solid #7289da; padding: 8px;">Opponent Wins</th>
                           <th style="border: 1px solid #7289da; padding: 8px;">AI Win Rate</th>
                           <th style="border: 1px solid #7289da; padding: 8px;">Opponent Win Rate</th>"""
                num_data_cols = 3
            else:
                for i in range(num_opponents_team):
                    html += f"""<th style="border: 1px solid #7289da; padding: 8px;">Opp{i+1} Wins (P{i+1})</th>"""
                html += """<th style="border: 1px solid #7289da; padding: 8px;">AI Win Rate</th>"""
                for i in range(num_opponents_team):
                    html += f"""<th style="border: 1px solid #7289da; padding: 8px;">Opp{i+1} Win Rate</th>"""
                num_data_cols = 1 + num_opponents_team + (num_opponents_team + 1)
        else: # Standard Mode
            html += """<th style="border: 1px solid #7289da; padding: 8px;">Opponent Name</th>"""
            if combine:
                 html += """<th style="border: 1px solid #7289da; padding: 8px;">AI Wins</th>
                            <th style="border: 1px solid #7289da; padding: 8px;">Opponent Wins</th>
                            <th style="border: 1px solid #7289da; padding: 8px;">AI Win Rate</th>
                            <th style="border: 1px solid #7289da; padding: 8px;">Opponent Win Rate</th>"""
                 num_data_cols = 4
            else:
                for i in range(num_ai_agents_standard):
                    html += f"""<th style="border: 1px solid #7289da; padding: 8px;">AI{i+1} Wins (P{i})</th>"""
                html += f"""<th style="border: 1px solid #7289da; padding: 8px;">Opponent Wins (P{num_ai_agents_standard})</th>"""
                for i in range(num_ai_agents_standard):
                    html += f"""<th style="border: 1px solid #7289da; padding: 8px;">AI{i+1} Win Rate</th>"""
                html += f"""<th style="border: 1px solid #7289da; padding: 8px;">Opponent Win Rate</th>"""
                num_data_cols = num_ai_agents_standard + 1 + (num_ai_agents_standard + 1)

        html += """<th style="border: 1px solid #7289da; padding: 8px;">Result vs AI</th></tr></thead><tbody>"""

        # Initialize aggregators for summary rows
        total_ai_wins_overall = 0
        total_opp_wins_overall = 0
        total_games_overall = 0
        min_ai_rate = 1.0
        min_opp_name = None

        for opp_display_name, wins_list in results.items():
            if len(wins_list) < num_players: wins_list.extend([0] * (num_players - len(wins_list)))
            p_wins = wins_list

            # --- Per-Match Calculations ---
            if is_onev2 or is_team_mode:
                ai_wins = p_wins[0]
                opp_wins_combined = sum(p_wins[1:])
                total_games_match = ai_wins + opp_wins_combined
                current_ai_wins_in_match = ai_wins
                current_opp_wins_in_match = opp_wins_combined
            else: # Standard Mode
                ai_wins_combined = sum(p_wins[:num_ai_agents_standard])
                opp_wins = p_wins[num_ai_agents_standard]
                total_games_match = ai_wins_combined + opp_wins
                current_ai_wins_in_match = ai_wins_combined
                current_opp_wins_in_match = opp_wins
            
            # --- Update Overall Aggregators ---
            total_games_overall += total_games_match
            total_ai_wins_overall += current_ai_wins_in_match
            total_opp_wins_overall += current_opp_wins_in_match
            
            ai_rate = current_ai_wins_in_match / total_games_match if total_games_match > 0 else 0.0
            
            # Track min rate
            if total_games_match > 0 and ai_rate < min_ai_rate:
                min_ai_rate = ai_rate
                min_opp_name = opp_display_name

            result_str = "Win" if ai_rate > 0.5 else "Loss" if ai_rate < 0.5 else "Draw"
            html += f"""<tr><td style="border:1px solid #7289da;padding:6px;">{opp_display_name}</td>"""

            # --- Generate HTML for table row ---
            if is_onev2 or is_team_mode:
                html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{ai_wins}</td>"""
                if combine:
                    opp_rate = 1.0 - ai_rate if total_games_match > 0 else 0.0
                    html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{opp_wins_combined}</td><td style="border:1px solid #7289da;padding:6px;text-align:center;">{ai_rate:.2%}</td><td style="border:1px solid #7289da;padding:6px;text-align:center;">{opp_rate:.2%}</td>"""
                else:
                    for i in range(num_opponents_team):
                        html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{p_wins[i+1]}</td>"""
                    html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{ai_rate:.2%}</td>"""
                    for i in range(num_opponents_team):
                        opp_rate = p_wins[i+1] / total_games_match if total_games_match > 0 else 0.0
                        html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{opp_rate:.2%}</td>"""
            else: # Standard
                if combine:
                    opp_rate = 1.0 - ai_rate if total_games_match > 0 else 0.0
                    html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{ai_wins_combined}</td><td style="border:1px solid #7289da;padding:6px;text-align:center;">{opp_wins}</td><td style="border:1px solid #7289da;padding:6px;text-align:center;">{ai_rate:.2%}</td><td style="border:1px solid #7289da;padding:6px;text-align:center;">{opp_rate:.2%}</td>"""
                else:
                    for i in range(num_ai_agents_standard):
                        html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{p_wins[i]}</td>"""
                    html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{opp_wins}</td>"""
                    for i in range(num_ai_agents_standard):
                         win_rate = p_wins[i] / total_games_match if total_games_match > 0 else 0.0
                         html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{win_rate:.2%}</td>"""
                    opp_rate = opp_wins / total_games_match if total_games_match > 0 else 0.0
                    html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{opp_rate:.2%}</td>"""

            html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{result_str}</td></tr>"""
            
        # --- Overall Summary Row ---
        overall_ai_rate = total_ai_wins_overall / total_games_overall if total_games_overall > 0 else 0.0
        overall_opp_rate = total_opp_wins_overall / total_games_overall if total_games_overall > 0 else 0.0
        overall_result = "Win" if overall_ai_rate > 0.5 else "Loss" if overall_ai_rate < 0.5 else "Draw"
        
        html += f"""<tr style="background-color:#2f3136;font-weight:bold;">
                    <td style="border:1px solid #7289da;padding:6px;">Overall</td>"""

        # Add cells based on whether data is combined
        if combine:
             html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{total_ai_wins_overall}</td>
                         <td style="border:1px solid #7289da;padding:6px;text-align:center;">{total_opp_wins_overall}</td>
                         <td style="border:1px solid #7289da;padding:6px;text-align:center;">{overall_ai_rate:.2%}</td>
                         <td style="border:1px solid #7289da;padding:6px;text-align:center;">{overall_opp_rate:.2%}</td>"""
        else: # Uncombined view needs placeholders
            # The number of empty cells is the number of data columns minus the one cell we use for the summary
            colspan = num_data_cols
            summary_text = f"{total_ai_wins_overall} (AI) vs {total_opp_wins_overall} (Opp) | Overall AI Win Rate: {overall_ai_rate:.2%}"
            html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;" colspan="{colspan}">{summary_text}</td>"""

        html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{overall_result}</td></tr>"""


        # --- Min Win Rate Row ---
        if min_opp_name is not None:
             min_result_str = "Win" if min_ai_rate > 0.5 else "Loss" if min_ai_rate < 0.5 else "Draw"
             # Span all columns except the first and last two
             colspan = num_data_cols
             
             html += f"""<tr style="background-color:#202225;color:#aaa;">
                         <td style="border:1px solid #7289da;padding:6px;">Lowest AI Rate vs:</td>
                         <td style="border:1px solid #7289da;padding:6px;" colspan="{colspan-1}">{min_opp_name}</td>
                         <td style="border:1px solid #7289da;padding:6px;text-align:center;">{min_ai_rate:.2%}</td>
                         <td style="border:1px solid #7289da;padding:6px;text-align:center;">{min_result_str}</td></tr>"""

        html += "</tbody></table>"
        return html
        
    def update_results_display(self):
        if self.current_results:
            self.display_results(self.current_results)
        
    def compare_results(self):
        if self.previous_results is None or self.current_results is None:
            QtWidgets.QMessageBox.information(self, "Comparison", "No previous results to compare. Run at least two battles.")
            self.previous_results = self.current_results
            return

        LABEL_SHORT_NAMES = {"GreedyCardSpammer": "GCS", "StrategicChallenger": "SC", "TableNonTableAgent": "TNTA", "Classic": "CL", "TableFirstConservativeChallenger": "TFCC", "SelectiveTableConservativeChallenger": "STCC", "RandomAgent": "RA", "Version_E_player_1": "VE", "Version_C_player_0": "VC", "Version_A_player_2": "VA"}

        def make_acronym(name):
            parts = name.split("+")
            short_parts = [LABEL_SHORT_NAMES.get(part, part[:4].upper()) for part in parts]
            return "+".join(short_parts)

        opp_names = list(self.current_results.keys())
        display_names = [make_acronym(name) for name in opp_names]
        ai_prev_rates, opp_prev_rates, ai_curr_rates, opp_curr_rates = [], [], [], []

        is_team_or_1v2 = self.onev2_mode_radio.isChecked() or self.team_mode_radio.isChecked()
        is_4p_mode = self.four_player_checkbox.isChecked()
        num_ai_agents = 1 if is_team_or_1v2 else 3 if is_4p_mode else 2
        
        for opp in opp_names:
            prev = self.previous_results.get(opp, [0]*4)
            curr = self.current_results.get(opp, [0]*4)

            prev_ai_wins = sum(prev[:num_ai_agents])
            curr_ai_wins = sum(curr[:num_ai_agents])
            prev_opp_wins = sum(prev[num_ai_agents:])
            curr_opp_wins = sum(curr[num_ai_agents:])

            prev_total = prev_ai_wins + prev_opp_wins
            curr_total = curr_ai_wins + curr_opp_wins

            ai_prev_rates.append(prev_ai_wins / prev_total if prev_total > 0 else 0)
            opp_prev_rates.append(prev_opp_wins / prev_total if prev_total > 0 else 0)
            ai_curr_rates.append(curr_ai_wins / curr_total if curr_total > 0 else 0)
            opp_curr_rates.append(curr_opp_wins / curr_total if curr_total > 0 else 0)

        x, width = np.arange(len(opp_names)), 0.35

        for rates1, rates2, title_part in [(ai_prev_rates, ai_curr_rates, "AI"), (opp_prev_rates, opp_curr_rates, "Opponent")]:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(x - width/2, rates1, width, label=f'Previous {title_part} Win Rate')
            ax.bar(x + width/2, rates2, width, label=f'Current {title_part} Win Rate')
            ax.set_xticks(x)
            ax.set_xticklabels(display_names, rotation=45)
            ax.set_ylabel("Win Rate")
            ax.set_title(f"{title_part} Win Rate Comparison")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            plt.show()

    def show_expert_usage(self):
        """Display expert activation information, adapting for all game modes."""
        if not self.expert_activations:
            QtWidgets.QMessageBox.information(self, "Expert Usage", "No expert data available.")
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Expert/Belief Activation Analysis")
        dialog.setMinimumSize(900, 700)
        layout = QtWidgets.QVBoxLayout(dialog)
        tab_widget = QtWidgets.QTabWidget()

        # --- Identify AI Player IDs ---
        ai_player_ids = []
        try:
            configs = self.get_selected_ai_configs()
            # Sort by player number ('player_0', 'player_1', etc.) to ensure consistent tab order
            sorted_configs = sorted(configs.items(), key=lambda item: int(item[0].split('_')[-1]))
            ai_player_ids = [cfg['id'] for _, cfg in sorted_configs]
        except ValueError:
            logger.warning("Could not get AI configs for expert analysis. Falling back to data keys.")
        
        if not ai_player_ids and self.expert_activations:
            # Fallback: Get AI IDs from the first recorded match data
            first_match_data = next(iter(self.expert_activations.values()), {})
            ai_player_ids = sorted(list(first_match_data.keys()))
        
        if not ai_player_ids:
            QtWidgets.QMessageBox.warning(self, "Expert Usage", "Could not identify AI player IDs.")
            return

        is_team_mode = self.team_mode_radio.isChecked()
        num_players = 4 if self.four_player_checkbox.isChecked() else 3
        num_opponents_in_team = num_players - 1

        for idx, player_id in enumerate(ai_player_ids):
            player_tab = QtWidgets.QWidget()
            player_layout = QtWidgets.QVBoxLayout(player_tab)

            # --- Determine Data Type (Belief vs. MoE/Gating) ---
            first_match_data = next(iter(self.expert_activations.values()), {})
            player_data_for_first_match = first_match_data.get(player_id, {})
            steps_data = player_data_for_first_match.get('steps', []) # Default to an empty list

            is_belief_agent_data = False # Default to False
            if steps_data: # Check if the steps list is not empty
                first_step_info = steps_data[0]
                is_belief_agent_data = isinstance(first_step_info, dict) and \
                                       all(k.startswith(('player_', 'Hardcoded_', 'Version_')) for k in first_step_info.keys())

            # --- Build Table Header ---
            html = f"<h2>{'Belief Peak' if is_belief_agent_data else 'Expert/Gate'} Activations for AI Agent {idx+1} ({player_id})</h2>"
            html += """<table style="border: 1px solid #7289da; border-collapse: collapse; width: 100%;">
                       <thead><tr style='background-color: #4f545c;'><th>Opponent Match</th>"""
            if is_belief_agent_data and is_team_mode:
                 for i in range(num_opponents_in_team):
                     html += f"<th>Opp{i+1} Peak (Rate)</th>"
            else:
                html += "<th>Most Used</th><th>Rate</th>"
            html += "<th>Total Steps</th></tr></thead><tbody>"

            # Data for plotting
            plot_match_names = []
            plot_data_lists = [[] for _ in range(num_opponents_in_team)] # Create lists for each potential opponent

            for match_name, match_data in self.expert_activations.items():
                player_expert_step_data = match_data.get(player_id, {}).get('steps')
                if not player_expert_step_data:
                    continue

                html += f"""<tr><td style="border:1px solid #7289da;padding:6px;">{match_name}</td>"""
                plot_match_names.append(match_name)
                total_steps_in_match = len(player_expert_step_data)

                if is_belief_agent_data:
                     # Aggregate belief peaks for each opponent encountered in the match
                     agg_peaks_per_opponent = defaultdict(lambda: defaultdict(int))
                     for step_info in player_expert_step_data:
                          if isinstance(step_info, dict):
                               for opp_id, peak_info in step_info.items():
                                    if peak_info and 'expert_index' in peak_info:
                                         peak_idx_str = str(peak_info['expert_index'])
                                         agg_peaks_per_opponent[opp_id][peak_idx_str] += 1
                     
                     if is_team_mode:
                         opp_ids = sorted(list(agg_peaks_per_opponent.keys()))
                         for i in range(num_opponents_in_team):
                             if i < len(opp_ids):
                                 opp_peaks = agg_peaks_per_opponent.get(opp_ids[i], {})
                                 opp_total = sum(opp_peaks.values())
                                 if opp_total > 0:
                                     peak_expert, peak_count = max(opp_peaks.items(), key=lambda item: item[1])
                                     peak_rate = peak_count / opp_total
                                     html += f"""<td style="border:1px solid #7289da;padding:6px;">T{peak_expert} ({peak_rate:.1%})</td>"""
                                     plot_data_lists[i].append((peak_expert, peak_rate))
                                 else:
                                     html += """<td style="border:1px solid #7289da;padding:6px;">N/A (0.0%)</td>"""
                                     plot_data_lists[i].append(("N/A", 0.0))
                             else: # Not enough opponents found in data for this slot
                                 html += """<td style="border:1px solid #7289da;padding:6px;">N/A</td>"""
                                 plot_data_lists[i].append(("N/A", 0.0))
                     else: # Belief agent, but not team mode (e.g., standard)
                         all_peaks_agg = defaultdict(int)
                         for opp_peaks in agg_peaks_per_opponent.values():
                              for expert_idx, count in opp_peaks.items(): all_peaks_agg[expert_idx] += count
                         total_agg_activations = sum(all_peaks_agg.values())
                         if total_agg_activations > 0:
                              peak_expert, peak_count = max(all_peaks_agg.items(), key=lambda i: i[1])
                              peak_rate = peak_count / total_agg_activations
                              html += f"""<td style="border:1px solid #7289da;padding:6px;">Peak T{peak_expert}</td><td style="border:1px solid #7289da;padding:6px;">{peak_rate:.1%}</td>"""
                              plot_data_lists[0].append((peak_expert, peak_rate))
                         else:
                              html += """<td>N/A</td><td>0.0%</td>"""
                              plot_data_lists[0].append(("N/A", 0.0))
                
                else: # Standard MoE/Gating Agent
                     expert_counts = defaultdict(int)
                     for step_info in player_expert_step_data:
                          if isinstance(step_info, dict) and 'expert_index' in step_info:
                               expert_idx_str = str(step_info['expert_index'])
                               expert_counts[expert_idx_str] += 1
                     total_activations = sum(expert_counts.values())
                     if total_activations > 0:
                          most_used_expert, max_count = max(expert_counts.items(), key=lambda i: i[1])
                          activation_rate = max_count / total_activations
                          html += f"""<td style="border:1px solid #7289da;padding:6px;">E/G {most_used_expert}</td><td style="border:1px solid #7289da;padding:6px;">{activation_rate:.1%}</td>"""
                          plot_data_lists[0].append((most_used_expert, activation_rate))
                     else:
                          html += """<td>N/A</td><td>0.0%</td>"""
                          plot_data_lists[0].append(("N/A", 0.0))

                html += f"""<td style="border:1px solid #7289da;padding:6px;">{total_steps_in_match}</td></tr>"""

            html += "</tbody></table>"
            text_browser = QtWidgets.QTextBrowser()
            text_browser.setHtml(html)
            player_layout.addWidget(text_browser)

            # --- Plotting ---
            if plot_match_names:
                 num_matches = len(plot_match_names)
                 x = np.arange(num_matches)
                 figure = plt.figure(figsize=(max(8, num_matches * 0.8), 6))
                 ax = figure.add_subplot(111)

                 if is_belief_agent_data and is_team_mode:
                     total_bars = num_opponents_in_team
                     bar_width = 0.8 / total_bars
                     offsets = np.linspace(-bar_width * (total_bars - 1) / 2, bar_width * (total_bars - 1) / 2, total_bars)

                     for i in range(total_bars):
                         plot_data = plot_data_lists[i]
                         experts = [p[0] for p in plot_data]
                         rates = [p[1] for p in plot_data]
                         bars = ax.bar(x + offsets[i], rates, bar_width, label=f'Opp {i+1} Peak Rate')
                         for bar, expert in zip(bars, experts):
                              if expert != "N/A":
                                   ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01, f"T{expert}", ha='center', va='bottom', fontsize=8)
                     ax.set_title(f'Opponent Belief Peak Analysis (AI {idx+1}: {player_id})')
                     ax.set_ylabel('Rate of Peak Belief Type')
                 else: # Single bar plot for MoE or non-team Belief
                     plot_data = plot_data_lists[0]
                     experts = [p[0] for p in plot_data]
                     rates = [p[1] for p in plot_data]
                     bars = ax.bar(x, rates, width=0.6, label='Dominant Rate')
                     for bar, expert in zip(bars, experts):
                          if expert != "N/A":
                               ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01, f"E/T{expert}", ha='center', va='bottom', fontsize=8)
                     ax.set_title(f'Dominant Activation Analysis (AI {idx+1}: {player_id})')
                     ax.set_ylabel('Activation Rate')

                 ax.set_xticks(x)
                 ax.set_xticklabels(plot_match_names, rotation=45, ha='right', fontsize=9)
                 ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                 ax.set_ylim(0, 1.05)
                 ax.grid(axis='y', linestyle='--', alpha=0.7)
                 figure.tight_layout(rect=[0, 0, 0.85, 1])
                 canvas = FigureCanvasQTAgg(figure)
                 player_layout.addWidget(canvas)

            tab_widget.addTab(player_tab, f"AI Agent {idx+1}")

        layout.addWidget(tab_widget)
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        dialog.setLayout(layout)
        dialog.exec_()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dark_stylesheet = """
    QWidget { background-color: #2f3136; color: #dcddde; font-family: "Helvetica", "Arial", sans-serif; }
    QGroupBox { border: 1px solid #202225; border-radius: 4px; margin-top: 1ex; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; color: #fff; }
    QPushButton { background-color: #7289da; border: none; border-radius: 4px; padding: 5px 10px; color: #fff; }
    QPushButton:hover { background-color: #5b6eae; }
    QLineEdit, QComboBox, QSpinBox, QTextEdit, QListWidget { background-color: #36393f; border: 1px solid #202225; border-radius: 4px; padding: 4px; }
    QProgressBar { background-color: #36393f; border: 1px solid #202225; border-radius: 4px; text-align: center; }
    QProgressBar::chunk { background-color: #7289da; border-radius: 4px; }
    """
    app.setStyleSheet(dark_stylesheet)
    window = AgentBattlegroundGUI()
    window.show()
    sys.exit(app.exec_())
    