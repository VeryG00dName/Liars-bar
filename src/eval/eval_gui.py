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
torch.backends.cudnn.benchmark = True

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
    
    # Now include extra parameters "onev2" and "duo"
    def __init__(self,
                 selected_ai_agent_configs: Dict[str, Dict],
                 # MODIFIED: Expects {identifier: {'path': ..., 'key': ...}}
                 historical_model_configs: Dict[str, Dict[str, str]],
                 hardcoded_agent_classes: Dict[str, type],
                 rounds: int,
                 device: torch.device,
                 two_player: Optional[str] = None,
                 cheat: bool = False,
                 onev2: bool = False,
                 duo: bool = False,
                 parent=None):
        super().__init__(parent)
        self.selected_ai_agent_configs = selected_ai_agent_configs
        self.historical_model_configs = historical_model_configs # Store the dict
        self.hardcoded_agent_classes = hardcoded_agent_classes
        # ... rest of __init__ ...
        self.rounds = rounds
        self.device = device
        self.two_player = two_player
        self.cheat = cheat
        self.onev2 = onev2
        self.duo = duo
        self.expert_activations = {}
        self.agent_factory = AgentFactory(self.device)
    
    def run(self):
        # --- Prepare Opponent Pool ---
        opponent_pool = {} # Maps opponent_name -> (type, config)

        # Add hardcoded agent configs
        for name, cls in self.hardcoded_agent_classes.items():
            opponent_pool[name] = ("hardcoded", self.agent_factory.create_hardcoded_agent_config(cls, name))

        # Add historical models using the new config structure
        # MODIFIED: Iterate through the historical config dict
        for identifier, hist_config in self.historical_model_configs.items():
            # Config already contains path and key
            opponent_pool[identifier] = ("historical_ai", {
                'path': hist_config['path'],
                'key': hist_config['key'],
                'id_prefix': identifier # Use the unique ID as prefix
            })

        # ... (progress calculation, results/expert dict init) ...
        progress_counter = 0
        total_matches_estimate = self.rounds * len(opponent_pool)
        if self.duo: total_matches_estimate = self.rounds * (len(opponent_pool) * (len(opponent_pool) - 1) // 2) if len(opponent_pool)>1 else 0

        results = {} # Stores {display_name: [p0, p1, p2 wins]}
        self.expert_activations = {} # Stores {display_name: {player_id: {expert_idx: count}}}

        try:
            # --- Main Loop (Duo, OneV2, Standard) ---
            if self.duo:
                # ... (duo loop setup) ...
                duo_pairs = list(itertools.combinations(opponent_pool.items(), 2))
                logger.info(f"Running Duo Mode with {len(duo_pairs)} opponent pairs.")
                for ((opp_name1, (opp_type1, opp_config1)), (opp_name2, (opp_type2, opp_config2))) in duo_pairs:
                    opponent_display_name = f"{opp_name1}+{opp_name2}"
                    # Pass tuples of configs and types
                    opponent_configs = (opp_config1, opp_config2)
                    opponent_types = (opp_type1, opp_type2)

                    cheat_idx1 = LABELS.get(opp_name1)
                    cheat_idx2 = LABELS.get(opp_name2)
                    current_cheat_index = (cheat_idx1, cheat_idx2) if self.cheat and cheat_idx1 is not None and cheat_idx2 is not None else None

                    # Run the match
                    cumulative_wins, expert_acts, player_id_map = self.run_match( # Get player_id_map
                        opponent_configs=opponent_configs,
                        opponent_types=opponent_types,
                        opponent_display_name=opponent_display_name,
                        episodes=self.rounds,
                        progress_callback=lambda ep: self.progress_signal.emit(progress_counter + ep),
                        cheat_expert_index=current_cheat_index,
                        mode='duo' # Pass mode
                    )

                    # Format results using the map
                    results[opponent_display_name] = self._format_wins(cumulative_wins, player_id_map)
                    self.expert_activations[opponent_display_name] = expert_acts
                    progress_counter += self.rounds
                    self.progress_signal.emit(progress_counter) # Update progress bar

            elif self.onev2:
                 # ... (onev2 loop setup) ...
                 logger.info(f"Running 1v2 Mode against {len(opponent_pool)} opponents.")
                 for opp_name, (opp_type, opp_config) in opponent_pool.items():
                     opponent_display_name = f"{opp_name}(x2)"
                     cheat_idx = LABELS.get(opp_name)
                     current_cheat_index = (cheat_idx, cheat_idx) if self.cheat and cheat_idx is not None else None

                     cumulative_wins, expert_acts, player_id_map = self.run_match(
                         opponent_configs=(opp_config, opp_config), # Duplicate config
                         opponent_types=(opp_type, opp_type),       # Duplicate type
                         opponent_display_name=opponent_display_name,
                         episodes=self.rounds,
                         progress_callback=lambda ep: self.progress_signal.emit(progress_counter + ep),
                         cheat_expert_index=current_cheat_index,
                         mode='onev2' # Pass mode
                     )
                     results[opponent_display_name] = self._format_wins(cumulative_wins, player_id_map)
                     self.expert_activations[opponent_display_name] = expert_acts
                     progress_counter += self.rounds
                     self.progress_signal.emit(progress_counter)

            else: # Standard mode
                 # ... (standard loop setup) ...
                 logger.info(f"Running Standard Mode against {len(opponent_pool)} opponents.")
                 for opp_name, (opp_type, opp_config) in opponent_pool.items():
                     cheat_idx = LABELS.get(opp_name)
                     current_cheat_index = cheat_idx if self.cheat and cheat_idx is not None else None

                     cumulative_wins, expert_acts, player_id_map = self.run_match(
                         opponent_configs=(opp_config,), # Single opponent config
                         opponent_types=(opp_type,),     # Single opponent type
                         opponent_display_name=opp_name,
                         episodes=self.rounds,
                         progress_callback=lambda ep: self.progress_signal.emit(progress_counter + ep),
                         cheat_expert_index=current_cheat_index,
                         mode='standard' # Pass mode
                     )
                     results[opp_name] = self._format_wins(cumulative_wins, player_id_map)
                     self.expert_activations[opp_name] = expert_acts
                     progress_counter += self.rounds
                     self.progress_signal.emit(progress_counter)


            # Emit final signals
            logger.info("Battleground run finished.")
            self.expert_signal.emit(self.expert_activations)
            self.results_signal.emit(results)

        except Exception as e:
            logger.error(f"Error during battleground run: {e}", exc_info=True)
            self.error_signal.emit(str(e))

    def _format_wins(self, cumulative_wins: Dict[str, int], player_id_map: Dict[str, str]) -> list:
        """Formats wins into [p0_wins, p1_wins, p2_wins] using the player_id -> env_id map."""
        formatted = [0] * 3 # Assuming 3 players
        env_id_to_wins = {env_id: cumulative_wins.get(pid, 0) for pid, env_id in player_id_map.items()}

        for i in range(3):
            env_id = f"player_{i}"
            formatted[i] = env_id_to_wins.get(env_id, 0)

        return formatted

    def run_match(self, opponent_configs: tuple, opponent_types: tuple, opponent_display_name: str, episodes: int, mode:str, progress_callback=None, cheat_expert_index=None):
        """Runs a single match configuration using the AgentFactory."""
        env = LiarsDeckEnv(num_players=3, render_mode=None)
        players_for_eval: Dict[str, BaseAgent] = {} # Maps env_id -> Agent object

        try:
            # --- Instantiate AI Agents ---
            num_ai_agents = 1 if mode in ['duo', 'onev2'] else 2
            for i in range(num_ai_agents):
                env_id = f"player_{i}"
                # Need to handle case where config might be missing (e.g., user didn't select agent 2 in standard mode)
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
                     # Instantiate hardcoded agent here using the config
                     hc_class = opp_config['class']
                     hc_name = opp_config['name']
                     player_id = f"Hardcoded_{hc_name}" # Unique ID
                     # --- MODIFIED: Instantiate hardcoded agent with context ---
                     try:
                          # Determine agent index from env_id
                          agent_index = int(env_id.split('_')[-1])
                          # Try instantiating with context (name, num_players, agent_index)
                          hc_instance = hc_class(hc_name, 3, agent_index)
                          logger.debug(f"Instantiated {hc_name} with name, num_players=3, agent_index={agent_index}")
                     except TypeError:
                          # Fallback to just name if the constructor doesn't accept context
                          hc_instance = hc_class(hc_name)
                          logger.debug(f"Instantiated {hc_name} with just name.")
                     except Exception as e:
                          logger.error(f"Failed to instantiate hardcoded agent {hc_name}: {e}", exc_info=True)
                          raise ValueError(f"Cannot instantiate hardcoded agent {hc_name}") from e

                     # Wrap the instantiated agent
                     agent = HardcodedAgentWrapper(hc_instance, self.device, player_id)
                     # --- End Modification ---

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
            # MODIFIED: Get player_id_map from return value
            cumulative_wins, _, _, _, _, expert_activations, player_id_map = evaluate_agents(
                env,
                self.device,
                players_for_eval,
                episodes=episodes,
                two_player=self.two_player,
                track_experts=True,
                progress_callback=progress_callback,
                cheat_expert_index=cheat_expert_index
            )

            # Return wins keyed by unique player_id, expert activations, AND the map
            return cumulative_wins, expert_activations, player_id_map

        except Exception as e:
             logger.error(f"Error running match against {opponent_display_name}: {e}", exc_info=True)
             # Return empty results and map on error
             return {}, {}, {}

# --- Main GUI class using PyQt with a Discord-like style ---
class AgentBattlegroundGUI(QtWidgets.QMainWindow):
    def __init__(self):
        # ... (Initialization remains the same as previous version) ...
        super().__init__()
        self.setWindowTitle("Agent Battleground")
        self.resize(1000, 750)
        self.loaded_model_paths: Dict[str, Dict[str, Any]] = {}
        self.hardcoded_agents: Dict[str, Type] = { # Hardcoded agents ...
             "Classic": Classic, "GreedyCardSpammer": GreedyCardSpammer, "RandomAgent": RandomAgent,
             "SelectiveTableConservativeChallenger": SelectiveTableConservativeChallenger,
             "StrategicChallenger": StrategicChallenger, "TableFirstConservativeChallenger": TableFirstConservativeChallenger,
             "TableNonTableAgent": TableNonTableAgent
        }
        self.historical_model_configs: Dict[str, Dict[str, str]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        # Load specific historical configs (as refined previously)
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
                    found_count += 1; logger.info(f"Found config: {identifier} at {full_path}")
                else: logger.warning(f"No .pth found in {version_path} for {identifier}")
            else: logger.warning(f"Directory not found: {version_path} for {identifier}")
        logger.info(f"Found configs for {found_count} specific historical models.")
        # State tracking
        self.previous_results: Optional[Dict[str, list]] = None
        self.current_results: Optional[Dict[str, list]] = None
        self.expert_activations: Optional[Dict[str, Dict[str, Any]]] = None # Changed type hint slightly
        self.worker: Optional[BattlegroundWorker] = None
        self.initUI()


    def initUI(self):
        # ... (Setup central widget, main layout) ...
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # ... (Model Files Group, Model Info Group, AI Agents Selection Group remain the same) ...
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
        for i in range(2):
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
        self.standard_mode_radio = QtWidgets.QRadioButton("Standard (2v1)")
        self.onev2_mode_radio = QtWidgets.QRadioButton("1 AI vs 2 Opponents")
        self.duo_mode_radio = QtWidgets.QRadioButton("1 AI vs Duo Opponents")
        self.standard_mode_radio.setChecked(True)
        # Connect radio buttons AFTER they are all defined
        self.standard_mode_radio.toggled.connect(self._update_agent_selector_states)
        self.onev2_mode_radio.toggled.connect(self._update_agent_selector_states)
        self.duo_mode_radio.toggled.connect(self._update_agent_selector_states) # Connect duo mode too
        mode_layout.addWidget(self.standard_mode_radio)
        mode_layout.addWidget(self.onev2_mode_radio)
        mode_layout.addWidget(self.duo_mode_radio)
        control_layout.addWidget(mode_group)

        # --- Other Options ---
        options_group = QtWidgets.QGroupBox("Options")
        options_layout = QtWidgets.QHBoxLayout(options_group)
        self.two_player_checkbox = QtWidgets.QCheckBox("2 Player Only"); self.two_player_checkbox.setToolTip("Eliminate Player 2 at start.")
        options_layout.addWidget(self.two_player_checkbox)
        # --- MODIFIED: Checkbox Renamed and Connected ---
        self.combine_results_checkbox = QtWidgets.QCheckBox("Combine Results")
        self.combine_results_checkbox.setToolTip("Combine AI wins (Standard) or Opponent wins (Duo/1v2).") # Updated tooltip
        self.combine_results_checkbox.stateChanged.connect(self.update_results_display)
        options_layout.addWidget(self.combine_results_checkbox)
        # --- End Modification ---
        self.cheat_checkbox = QtWidgets.QCheckBox("Cheat (Expert Index)"); self.cheat_checkbox.setToolTip("Provide opponent type index via LABELS.")
        options_layout.addWidget(self.cheat_checkbox)
        control_layout.addWidget(options_group)

        control_layout.addStretch(2)
        self.start_button = QtWidgets.QPushButton(" Start Battleground "); self.start_button.setStyleSheet("padding: 8px 15px; font-weight: bold;")
        self.start_button.clicked.connect(self.start_battleground)
        control_layout.addWidget(self.start_button)
        main_layout.addLayout(control_layout)

        # ... (Progress Bar, Results Group remain the same visually) ...
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


        # Call explicitly after all relevant widgets are created
        self._update_agent_selector_states()

    def _update_agent_selector_states(self):
        is_standard_mode = self.standard_mode_radio.isChecked()
        is_duo_mode = self.duo_mode_radio.isChecked()
        if 1 in self.agent_selectors:
             models_loaded = self.agent_selectors[1].count() > 1
             self.agent_selectors[1].setEnabled(is_standard_mode and models_loaded)
        # --- CORRECTED: Checkbox enable logic ---
        self.combine_results_checkbox.setEnabled(is_standard_mode or is_duo_mode)
        if not (is_standard_mode or is_duo_mode): self.combine_results_checkbox.setChecked(False)
        # --- End Correction ---

    def on_file_drop(self, file_path):
        """Handles dropped files, inspects checkpoints for players, and adds them."""
        file_path = file_path.strip()
        if not file_path.endswith(".pth"):
            self.show_info("Only .pth files are supported")
            return

        # Use a base display name from the path
        base_display_name = f"{os.path.basename(os.path.dirname(file_path))}/{os.path.basename(file_path)}"

        try:
            # Load checkpoint just to inspect keys
            ckpt = torch.load(file_path, map_location='cpu', weights_only=False)
            policy_keys = []
            if 'policy_nets' in ckpt:
                 policy_keys = list(ckpt['policy_nets'].keys())
            elif 'model' in ckpt:
                 # If single 'model' key, treat it as player_0 for selection purposes
                 policy_keys = ['player_0'] # Represent the single model as 'player_0'
            else:
                 self.show_info(f"Checkpoint '{base_display_name}' has no recognizable policy keys ('policy_nets' or 'model').")
                 return

            if not policy_keys:
                 self.show_info(f"No player keys found within policy networks in '{base_display_name}'.")
                 return

            added_count = 0
            for key in sorted(policy_keys): # Sort keys for consistent order
                # Create a unique display name including the player key
                display_name_with_key = f"{base_display_name} [{key}]"

                if display_name_with_key in self.loaded_model_paths:
                    # Silently skip if this specific player from this file is already added
                    continue

                # Store path and the specific key for this player entry
                self.loaded_model_paths[display_name_with_key] = {'path': file_path, 'key': key, 'display_name': base_display_name}
                self.file_list.addItem(display_name_with_key) # Add the specific player entry to the list
                added_count += 1

            if added_count > 0:
                self.update_agent_selectors() # Update dropdowns
                self.show_info(f"Added {added_count} player(s) from: {base_display_name}")
            else:
                 self.show_info(f"All players from '{base_display_name}' were already added.")


        except Exception as e:
            logger.error(f"Error inspecting checkpoint {file_path}: {e}", exc_info=True)
            self.show_info(f"Error reading checkpoint: {os.path.basename(file_path)}")

    def get_selected_ai_configs(self) -> Dict[str, Dict]:
        """Gets the configuration for the AI agents selected in the dropdowns."""
        selected_configs = {}
        # --- CORRECTED: Check radio buttons for mode ---
        num_ai_needed = 1 if self.onev2_mode_radio.isChecked() or self.duo_mode_radio.isChecked() else 2

        for i in range(num_ai_needed):
             env_id = f"player_{i}"
             selector = self.agent_selectors[i]
             # --- MODIFIED: Get selection text AND associated data ---
             selected_index = selector.currentIndex()
             if selected_index <= 0: # Index 0 is placeholder "Select..." or "No models..."
                  # Check if this agent is actually needed for the current mode
                  if i == 0 or (i == 1 and self.standard_mode_radio.isChecked()):
                       raise ValueError(f"Please select a model for AI Agent {i+1}.")
                  else:
                       continue # Skip agent 2 if not in standard mode

             # Retrieve data stored with the item
             item_data = selector.itemData(selected_index)
             if not item_data or 'path' not in item_data or 'key' not in item_data:
                  # Fallback if data wasn't stored correctly (shouldn't happen with new update_selectors)
                  display_name_with_key = selector.itemText(selected_index)
                  raise ValueError(f"Missing data for selected agent: {display_name_with_key}. Please reload models.")

             path = item_data['path']
             agent_key = item_data['key'] # Key like 'player_0'
             display_name = item_data['display_name'] # Original display name without key

             prefix = display_name.replace("/", "_").replace("\\", "_").replace(".pth", "")
             selected_configs[env_id] = {
                 'path': path,
                 'key': agent_key, # Key *within* the checkpoint
                 'id_prefix': prefix
             }
             selected_configs[env_id]['id'] = f"{prefix}_{agent_key}" # Unique ID

        return selected_configs

    def show_info(self, message):
        self.info_text.setPlainText(message)

    def update_agent_selectors(self):
        """Populates agent selectors with 'Model File [player_key]' entries."""
        # Get sorted list of display names with keys (e.g., "dir/model.pth [player_0]")
        agent_options = sorted(list(self.loaded_model_paths.keys()))

        for i in range(2): # Update both selectors
            selector = self.agent_selectors[i]
            # Store current selection's data to restore it if possible
            current_index = selector.currentIndex()
            current_data = selector.itemData(current_index) if current_index > 0 else None

            selector.clear()
            if not agent_options:
                 selector.addItem("No models loaded")
                 selector.setEnabled(False)
            else:
                 selector.addItem("Select Model...") # Placeholder at index 0
                 # Add each specific player as an item
                 for display_name_with_key in agent_options:
                      item_data = self.loaded_model_paths[display_name_with_key]
                      # Store path and key as data associated with the item
                      selector.addItem(display_name_with_key, userData=item_data)

                 # Re-enable based on mode and if models exist
                 is_standard_mode = self.standard_mode_radio.isChecked()
                 selector.setEnabled(True if i == 0 else is_standard_mode)

                 # Try to restore previous selection by matching item data
                 restored = False
                 if current_data:
                      for idx in range(1, selector.count()): # Start from 1 to skip placeholder
                           if selector.itemData(idx) == current_data:
                                selector.setCurrentIndex(idx)
                                restored = True
                                break
                 if not restored:
                      selector.setCurrentIndex(0) # Default to "Select Model..."


        # Explicitly update enabled state after populating
        self._update_agent_selector_states() # Call the helper to ensure correct enable/disable state

    def start_battleground(self):
        # ... (implementation from previous step, ensure it reads radio buttons for mode) ...
        if self.worker is not None and self.worker.isRunning():
            self.show_info("A battleground run is already in progress.")
            return

        try:
            selected_ai_configs = self.get_selected_ai_configs()
        except ValueError as e:
            self.show_info(f"Selection Error: {e}")
            return

        two_player_param = "player_1" if self.two_player_checkbox.isChecked() else None
        cheat_flag = self.cheat_checkbox.isChecked()
        # Determine mode from radio buttons
        onev2_enabled = self.onev2_mode_radio.isChecked()
        duo_enabled = self.duo_mode_radio.isChecked()
        # Standard mode is implicitly handled if neither onev2 nor duo is checked

        rounds = self.rounds_spinbox.value()

        # Calculate Progress Max
        num_hardcoded = len(self.hardcoded_agents)
        num_historical = len(self.historical_model_configs)
        num_opponents_total = num_hardcoded + num_historical
        if num_opponents_total == 0:
             self.show_info("No opponents (hardcoded or historical) found to run against.")
             return

        if duo_enabled:
            total_matches = rounds * (num_opponents_total * (num_opponents_total - 1) // 2) if num_opponents_total > 1 else 0
        else:
            total_matches = rounds * num_opponents_total
        self.progress_bar.setMaximum(max(1, total_matches))
        self.progress_bar.setValue(0)

        if self.results_text.toPlainText().strip():
             self.previous_results = self.current_results
        self.results_text.clear()
        self.expert_activations = None # Clear previous expert data

        # Disable start button during run
        self.start_button.setEnabled(False)
        self.show_info(f"Starting battleground ({'1v2' if onev2_enabled else 'Duo' if duo_enabled else 'Standard'} mode)...")


        # Start Worker
        self.worker = BattlegroundWorker(
            selected_ai_agent_configs=selected_ai_configs,
            historical_model_configs=self.historical_model_configs,
            hardcoded_agent_classes=self.hardcoded_agents,
            rounds=rounds,
            device=self.device,
            two_player=two_player_param,
            cheat=cheat_flag,
            onev2=onev2_enabled,
            duo=duo_enabled
        )
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.results_signal.connect(self.on_results_received) # Connect to handler
        self.worker.expert_signal.connect(self.store_expert_activations)
        self.worker.error_signal.connect(self.on_worker_error) # Connect to handler
        self.worker.finished.connect(self.on_worker_finished) # Connect to handler
        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def store_expert_activations(self, expert_activations):
        """Store expert activations data"""
        self.expert_activations = expert_activations
        logger.info(f"Received expert activations for {len(expert_activations)} opponents")

    def on_results_received(self, results):
         """Handles receiving results from the worker."""
         self.display_results(results)
         self.show_info("Battleground finished.")

    def on_worker_error(self, error_message):
         """Handles errors reported by the worker."""
         self.show_info(f"Worker Error: {error_message}")
         logger.error(f"Worker thread reported error: {error_message}")
         self.start_button.setEnabled(True) # Re-enable start button on error

    def on_worker_finished(self):
         """Handles the worker thread finishing naturally."""
         self.start_button.setEnabled(True) # Re-enable start button
         # Optionally show a final "Finished" message if not already shown by results handler
         if not self.current_results: # If no results were emitted (e.g., zero rounds)
              self.show_info("Battleground worker finished.")

    # --- Display results with optional combining ---
    def display_results(self, results: Dict[str, list]):
        """Displays results, triggering HTML generation."""
        self.current_results = results
        # --- MODIFIED: Pass correct checkbox state ---
        combine = self.combine_results_checkbox.isChecked()
        # --- End Modification ---
        is_onev2 = self.onev2_mode_radio.isChecked()
        is_duo = self.duo_mode_radio.isChecked()

        html = self._generate_results_html(results, combine, is_onev2, is_duo)
        self.results_text.setHtml(html)

    def _generate_results_html(self, results, combine, is_onev2, is_duo):
        """Generates the results HTML table based on mode and combine flag."""
        html = """<table style="border: 1px solid #7289da; border-collapse: collapse; width: 100%;"><thead><tr style="background-color: #4f545c;">"""
        num_data_cols = 0 # Track number of columns between Name and Result

        # --- Table Header Logic ---
        if is_onev2 or is_duo: # 1 AI (P0) vs 2 Opponents (P1, P2)
            html += """<th style="border: 1px solid #7289da; padding: 8px;">Opponent Name / Pair</th>
                       <th style="border: 1px solid #7289da; padding: 8px;">AI Wins (P0)</th>"""
            if combine: # Combine Opponent wins
                html += """<th style="border: 1px solid #7289da; padding: 8px;">Opponent Wins (P1+P2)</th>
                           <th style="border: 1px solid #7289da; padding: 8px;">AI Win Rate</th>
                           <th style="border: 1px solid #7289da; padding: 8px;">Opponent Win Rate</th>"""
                num_data_cols = 4 # Wins(2) + Rates(2)
            else: # Show Opponents separately
                html += """<th style="border: 1px solid #7289da; padding: 8px;">Opponent1 Wins (P1)</th>
                           <th style="border: 1px solid #7289da; padding: 8px;">Opponent2 Wins (P2)</th>
                           <th style="border: 1px solid #7289da; padding: 8px;">AI Win Rate</th>
                           <th style="border: 1px solid #7289da; padding: 8px;">Opp1 Win Rate</th>
                           <th style="border: 1px solid #7289da; padding: 8px;">Opp2 Win Rate</th>"""
                num_data_cols = 6 # Wins(3) + Rates(3)
        else: # Standard Mode (2 AI (P0, P1) vs 1 Opponent (P2))
            html += """<th style="border: 1px solid #7289da; padding: 8px;">Opponent Name</th>"""
            if combine: # Combine AI wins
                 html += """<th style="border: 1px solid #7289da; padding: 8px;">AI Wins (P0+P1)</th>
                            <th style="border: 1px solid #7289da; padding: 8px;">Opponent Wins (P2)</th>
                            <th style="border: 1px solid #7289da; padding: 8px;">AI Win Rate</th>
                            <th style="border: 1px solid #7289da; padding: 8px;">Opponent Win Rate</th>"""
                 num_data_cols = 4 # Wins(2) + Rates(2)
            else: # Show AI separately
                 html += """<th style="border: 1px solid #7289da; padding: 8px;">AI1 Wins (P0)</th>
                            <th style="border: 1px solid #7289da; padding: 8px;">AI2 Wins (P1)</th>
                            <th style="border: 1px solid #7289da; padding: 8px;">Opponent Wins (P2)</th>
                            <th style="border: 1px solid #7289da; padding: 8px;">AI1 Win Rate</th>
                            <th style="border: 1px solid #7289da; padding: 8px;">AI2 Win Rate</th>
                            <th style="border: 1px solid #7289da; padding: 8px;">Opponent Win Rate</th>"""
                 num_data_cols = 6 # Wins(3) + Rates(3)

        html += """<th style="border: 1px solid #7289da; padding: 8px;">Result vs AI</th></tr></thead><tbody>""" # Result relative to AI performance

        # --- Table Rows ---
        # ... (Row generation logic remains the same, correctly uses combine/is_duo/is_onev2 flags) ...
        total_ai_wins_overall = 0; total_opp_wins_overall = 0; total_games_overall = 0
        min_ai_rate = 1.0; min_opp_name = None
        for opp_display_name, wins_list in results.items():
            if len(wins_list) < 3: wins_list.extend([0] * (3 - len(wins_list)))
            p0_wins, p1_wins, p2_wins = wins_list
            # Calculate wins based on mode
            if is_onev2 or is_duo: ai_wins = p0_wins; opp1_wins = p1_wins; opp2_wins = p2_wins; opp_wins_combined = opp1_wins + opp2_wins; total_games_match = ai_wins + opp_wins_combined
            else: ai1_wins = p0_wins; ai2_wins = p1_wins; ai_wins_combined = ai1_wins + ai2_wins; opp_wins = p2_wins; total_games_match = ai_wins_combined + opp_wins
            # Update totals
            total_games_overall += total_games_match
            current_ai_wins_in_match = ai_wins if (is_onev2 or is_duo) else ai_wins_combined
            current_opp_wins_in_match = opp_wins_combined if (is_onev2 or is_duo) else opp_wins
            total_ai_wins_overall += current_ai_wins_in_match
            total_opp_wins_overall += current_opp_wins_in_match
            # Calculate AI rate for this match
            ai_rate = current_ai_wins_in_match / total_games_match if total_games_match > 0 else 0.0
            # Track min rate
            if total_games_match > 0 and ai_rate < min_ai_rate: min_ai_rate = ai_rate; min_opp_name = opp_display_name
            result_str = "Win" if ai_rate > 0.5 else "Loss" if ai_rate < 0.5 else "Draw"
            # Build Row HTML
            html += f"""<tr><td style="border:1px solid #7289da;padding:6px;">{opp_display_name}</td>"""
            if is_onev2 or is_duo:
                html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{ai_wins}</td>"""
                if combine: opp_rate = 1.0 - ai_rate if total_games_match > 0 else 0.0; html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{opp_wins_combined}</td><td style="border:1px solid #7289da;padding:6px;text-align:center;">{ai_rate:.2%}</td><td style="border:1px solid #7289da;padding:6px;text-align:center;">{opp_rate:.2%}</td>"""
                else: opp1_rate = opp1_wins / total_games_match if total_games_match > 0 else 0.0; opp2_rate = opp2_wins / total_games_match if total_games_match > 0 else 0.0; html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{opp1_wins}</td><td style="border:1px solid #7289da;padding:6px;text-align:center;">{opp2_wins}</td><td style="border:1px solid #7289da;padding:6px;text-align:center;">{ai_rate:.2%}</td><td style="border:1px solid #7289da;padding:6px;text-align:center;">{opp1_rate:.2%}</td><td style="border:1px solid #7289da;padding:6px;text-align:center;">{opp2_rate:.2%}</td>"""
            else: # Standard
                if combine: opp_rate = 1.0 - ai_rate if total_games_match > 0 else 0.0; html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{ai_wins_combined}</td><td style="border:1px solid #7289da;padding:6px;text-align:center;">{opp_wins}</td><td style="border:1px solid #7289da;padding:6px;text-align:center;">{ai_rate:.2%}</td><td style="border:1px solid #7289da;padding:6px;text-align:center;">{opp_rate:.2%}</td>"""
                else: ai1_rate = ai1_wins / total_games_match if total_games_match > 0 else 0.0; ai2_rate = ai2_wins / total_games_match if total_games_match > 0 else 0.0; opp_rate = opp_wins / total_games_match if total_games_match > 0 else 0.0; html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{ai1_wins}</td><td style="border:1px solid #7289da;padding:6px;text-align:center;">{ai2_wins}</td><td style="border:1px solid #7289da;padding:6px;text-align:center;">{opp_wins}</td><td style="border:1px solid #7289da;padding:6px;text-align:center;">{ai1_rate:.2%}</td><td style="border:1px solid #7289da;padding:6px;text-align:center;">{ai2_rate:.2%}</td><td style="border:1px solid #7289da;padding:6px;text-align:center;">{opp_rate:.2%}</td>"""
            html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{result_str}</td></tr>"""


        # --- Overall Summary Row ---
        overall_ai_rate = total_ai_wins_overall / total_games_overall if total_games_overall > 0 else 0.0
        overall_opp_rate = total_opp_wins_overall / total_games_overall if total_games_overall > 0 else 0.0
        overall_result = "Win" if overall_ai_rate > 0.5 else "Loss" if overall_ai_rate < 0.5 else "Draw"
        html += f"""<tr style="background-color:#2f3136;font-weight:bold;">
                    <td style="border:1px solid #7289da;padding:6px;">Overall</td>
                    <td style="border:1px solid #7289da;padding:6px;text-align:center;">{total_ai_wins_overall}</td>
                    <td style="border:1px solid #7289da;padding:6px;text-align:center;">{total_opp_wins_overall}</td>"""
        # Add rates/padding depending on number of data cols
        if num_data_cols == 4: # Combined view
             html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{overall_ai_rate:.2%}</td>
                         <td style="border:1px solid #7289da;padding:6px;text-align:center;">{overall_opp_rate:.2%}</td>"""
        elif num_data_cols == 6: # Separate view
             html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">N/A</td>
                         <td style="border:1px solid #7289da;padding:6px;text-align:center;">N/A</td>
                         <td style="border:1px solid #7289da;padding:6px;text-align:center;">{overall_ai_rate:.2%}</td>
                         <td style="border:1px solid #7289da;padding:6px;text-align:center;">N/A</td>
                         <td style="border:1px solid #7289da;padding:6px;text-align:center;">N/A</td>""" # Adjust cols/rates as needed
        html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{overall_result}</td></tr>"""


        # --- Min Win Rate Row ---
        if min_opp_name is not None:
             min_result_str = "Win" if min_ai_rate > 0.5 else "Loss" if min_ai_rate < 0.5 else "Draw"
             html += f"""<tr style="background-color:#202225;color:#aaa;">
                         <td style="border:1px solid #7289da;padding:6px;">Lowest AI Rate vs:</td>
                         <td style="border:1px solid #7289da;padding:6px;" colspan="2">{min_opp_name}</td>
                         <td style="border:1px solid #7289da;padding:6px;text-align:center;">{min_ai_rate:.2%}</td>"""
             # Add padding to match data columns
             html += f"""<td style="border:1px solid #7289da;padding:6px;"></td>""" * (num_data_cols - 3) # 1 for name, 2 for value
             html += f"""<td style="border:1px solid #7289da;padding:6px;text-align:center;">{min_result_str}</td></tr>"""

        html += "</tbody></table>"
        return html

        
    def update_results_display(self):
        """ Updates the displayed results when Combine AI Columns is toggled. """
        if self.current_results:
            self.display_results(self.current_results)  # Reformat and display results
        
    def compare_results(self):
        # Compare previous_results and current_results and plot two bar charts.
        if self.previous_results is None or self.current_results is None:
            QtWidgets.QMessageBox.information(self, "Comparison", "No previous results to compare. Run at least two battles.")
            self.previous_results = self.current_results
            return

        LABEL_SHORT_NAMES = {
            "GreedyCardSpammer": "GCS",
            "StrategicChallenger": "SC",
            "TableNonTableAgent": "TNTA",
            "Classic": "CL",
            "TableFirstConservativeChallenger": "TFCC",
            "SelectiveTableConservativeChallenger": "STCC",
            "RandomAgent": "RA",
            "Version_E_player_1": "VE",
            "Version_C_player_0": "VC",
            "Version_A_player_2": "VA",
        }

        def make_acronym(name):
            # For pair names joined by +, like "Classic+GreedyCardSpammer"
            parts = name.split("+")
            short_parts = [LABEL_SHORT_NAMES.get(part, part[:4].upper()) for part in parts]
            return "+".join(short_parts)

        opp_names = list(self.current_results.keys())
        # Create shortened labels for plotting.
        display_names = [make_acronym(name) for name in opp_names]

        ai_prev_rates = []
        opp_prev_rates = []
        ai_curr_rates = []
        opp_curr_rates = []

        # Determine mode: if either onev2 or duo is checked.
        is_special_mode = self.onev2_mode_radio.isChecked() or self.duo_mode_radio.isChecked()

        for opp in opp_names:
            prev = self.previous_results.get(opp, [0, 0, 0])
            curr = self.current_results.get(opp, [0, 0, 0])
            if is_special_mode:
                # In 1v2/duo mode, AI wins are only from column 0;
                # opponent wins are the sum of columns 1 and 2.
                prev_ai_wins = prev[0]
                curr_ai_wins = curr[0]
                prev_opp_wins = prev[1] + prev[2]
                curr_opp_wins = curr[1] + curr[2]
            else:
                # Normal mode: combine AI wins from columns 0 and 1.
                prev_ai_wins = prev[0] + prev[1]
                curr_ai_wins = curr[0] + curr[1]
                prev_opp_wins = prev[2]
                curr_opp_wins = curr[2]
            
            prev_total = prev_ai_wins + prev_opp_wins
            curr_total = curr_ai_wins + curr_opp_wins

            ai_prev = prev_ai_wins / prev_total if prev_total > 0 else 0
            opp_prev = prev_opp_wins / prev_total if prev_total > 0 else 0
            ai_curr = curr_ai_wins / curr_total if curr_total > 0 else 0
            opp_curr = curr_opp_wins / curr_total if curr_total > 0 else 0

            ai_prev_rates.append(ai_prev)
            opp_prev_rates.append(opp_prev)
            ai_curr_rates.append(ai_curr)
            opp_curr_rates.append(opp_curr)

        x = np.arange(len(opp_names))
        width = 0.35

        # Create the figure for AI win rates.
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.bar(x - width/2, ai_prev_rates, width, label='Previous AI Win Rate')
        ax1.bar(x + width/2, ai_curr_rates, width, label='Current AI Win Rate')
        ax1.set_xticks(x)
        ax1.set_xticklabels(display_names, rotation=45)
        ax1.set_ylabel("Win Rate")
        ax1.set_title("AI Win Rate Comparison")
        # Place legend outside the plot area
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()

        # Create the figure for Opponent win rates.
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.bar(x - width/2, opp_prev_rates, width, label='Previous Opponent Win Rate')
        ax2.bar(x + width/2, opp_curr_rates, width, label='Current Opponent Win Rate')
        ax2.set_xticks(x)
        ax2.set_xticklabels(display_names, rotation=45)
        ax2.set_ylabel("Win Rate")
        ax2.set_title("Opponent Win Rate Comparison")
        # Place legend outside the plot area
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    def show_expert_usage(self):
        """Display expert activation information, adapting for Duo mode Belief Agents."""
        if not self.expert_activations:
            QtWidgets.QMessageBox.information(self, "Expert Usage", "No expert data available.")
            return

        dialog = QtWidgets.QDialog(self); dialog.setWindowTitle("Expert/Belief Activation Analysis")
        dialog.setMinimumSize(900, 700); layout = QtWidgets.QVBoxLayout(dialog)
        tab_widget = QtWidgets.QTabWidget()

        # Identify AI player IDs (as before)
        # ... (code to get ai_player_ids) ...
        ai_player_ids = []
        try: 
            configs = self.get_selected_ai_configs(); is_standard = self.standard_mode_radio.isChecked()
            if 'player_0' in configs: ai_player_ids.append(configs['player_0']['id'])
            if is_standard and 'player_1' in configs: ai_player_ids.append(configs['player_1']['id'])
        except ValueError:
            logger.warning("Could not get AI configs for expert analysis.")
        if not ai_player_ids and self.expert_activations: first_match_data = next(iter(self.expert_activations.values()), {}); ai_player_ids = list(first_match_data.keys())
        if not ai_player_ids: QtWidgets.QMessageBox.warning(self, "Expert Usage", "Could not identify AI player IDs."); return


        is_duo_mode = self.duo_mode_radio.isChecked()

        for idx, player_id in enumerate(ai_player_ids):
            player_tab = QtWidgets.QWidget()
            player_layout = QtWidgets.QVBoxLayout(player_tab)

            # Check if this player has belief data (check the structure of the first step's info)
            first_match_data = next(iter(self.expert_activations.values()), {})
            first_step_info = first_match_data.get(player_id, {}).get('steps', [None])[0]
            # Check if the first step info is a dict with keys that look like opponent IDs
            is_belief_agent_data = isinstance(first_step_info, dict) and \
                                   all(k.startswith('player_') or k.startswith('Hardcoded_') or k.startswith('Version_') for k in first_step_info.keys())

            # Build Table Header (as before)
            # ... (HTML header generation) ...
            html = f"""<h2>{'Belief Peak' if is_belief_agent_data else 'Expert/Gate'} Activations for AI Agent {idx+1} ({player_id})</h2>...""" # Shortened
            html += """<table ...><thead><tr><th>Opponent Match</th>"""
            if is_belief_agent_data and is_duo_mode: html += """<th>Opp1 Peak (Rate)</th><th>Opp2 Peak (Rate)</th>""" # Simpler Duo display
            else: html += """<th>Most Used</th><th>Rate</th>"""
            html += """<th>Total Steps</th></tr></thead><tbody>"""

            plot_match_names = []; plot_data1 = []; plot_data2 = [] # Reset plot data

            for match_name, match_data in self.expert_activations.items():
                player_expert_step_data = match_data.get(player_id, {}).get('steps')
                if not player_expert_step_data: continue # No data for this player/match

                html += f"""<tr><td ...>{match_name}</td>"""
                plot_match_names.append(match_name)
                total_steps_in_match = len(player_expert_step_data)

                if is_belief_agent_data:
                     # --- Aggregate Belief Peaks from Step Data ---
                     agg_peaks_per_opponent = defaultdict(lambda: defaultdict(int))
                     for step_info in player_expert_step_data:
                          if isinstance(step_info, dict): # Should be dict of {opp_id: {'expert_index': peak, 'source':...}}
                               for opp_id, peak_info in step_info.items():
                                    if peak_info and 'expert_index' in peak_info:
                                         peak_idx_str = str(peak_info['expert_index'])
                                         agg_peaks_per_opponent[opp_id][peak_idx_str] += 1
                     # --- End Aggregation ---

                     if is_duo_mode: # Duo mode Belief Agent display
                         opp_ids = sorted(list(agg_peaks_per_opponent.keys()))
                         opp1_html = "N/A (0.0%)"; opp1_plot = ("N/A", 0.0)
                         opp2_html = "N/A (0.0%)"; opp2_plot = ("N/A", 0.0)

                         if len(opp_ids) > 0: # Opponent 1
                              opp1_peaks = agg_peaks_per_opponent.get(opp_ids[0], {})
                              opp1_total = sum(opp1_peaks.values())
                              if opp1_total > 0:
                                   opp1_peak_expert, opp1_count = max(opp1_peaks.items(), key=lambda i: i[1])
                                   opp1_rate = opp1_count / opp1_total # Rate over steps where this opponent had a peak recorded
                                   opp1_html = f"T{opp1_peak_expert} ({opp1_rate:.1%})"
                                   opp1_plot = (opp1_peak_expert, opp1_rate)
                         if len(opp_ids) > 1: # Opponent 2
                              opp2_peaks = agg_peaks_per_opponent.get(opp_ids[1], {})
                              opp2_total = sum(opp2_peaks.values())
                              if opp2_total > 0:
                                   opp2_peak_expert, opp2_count = max(opp2_peaks.items(), key=lambda i: i[1])
                                   opp2_rate = opp2_count / opp2_total
                                   opp2_html = f"T{opp2_peak_expert} ({opp2_rate:.1%})"
                                   opp2_plot = (opp2_peak_expert, opp2_rate)

                         html += f"""<td ...>{opp1_html}</td><td ...>{opp2_html}</td>""" # Combine rate in cell
                         plot_data1.append(opp1_plot); plot_data2.append(opp2_plot)

                     else: # Non-Duo Belief Agent: Aggregate across opponents
                         all_peaks_agg = defaultdict(int)
                         for opp_peaks in agg_peaks_per_opponent.values():
                              for expert_idx, count in opp_peaks.items(): all_peaks_agg[expert_idx] += 1
                         total_agg_activations = sum(all_peaks_agg.values())
                         if total_agg_activations > 0:
                              peak_expert, peak_count = max(all_peaks_agg.items(), key=lambda i: i[1])
                              peak_rate = peak_count / total_agg_activations
                              html += f"""<td ...>Peak T{peak_expert}</td><td ...>{peak_rate:.1%}</td>"""
                              plot_data1.append((peak_expert, peak_rate)); plot_data2.append(("N/A", 0.0))
                         else: html += """<td>N/A</td><td>0.0%</td>"""; plot_data1.append(("N/A", 0.0)); plot_data2.append(("N/A", 0.0))

                else: # MoE or StackedObs Agent
                     # Aggregate counts from step data
                     expert_counts = defaultdict(int)
                     for step_info in player_expert_step_data:
                          if isinstance(step_info, dict) and 'expert_index' in step_info:
                               expert_idx_str = str(step_info['expert_index'])
                               expert_counts[expert_idx_str] += 1
                     # Calculate peak rate
                     total_activations = sum(expert_counts.values())
                     if total_activations > 0:
                          most_used_expert, max_count = max(expert_counts.items(), key=lambda i: i[1])
                          activation_rate = max_count / total_activations
                          html += f"""<td ...>E/G {most_used_expert}</td><td ...>{activation_rate:.1%}</td>"""
                          plot_data1.append((most_used_expert, activation_rate)); plot_data2.append(("N/A", 0.0))
                     else: html += """<td>N/A</td><td>0.0%</td>"""; plot_data1.append(("N/A", 0.0)); plot_data2.append(("N/A", 0.0))

                html += f"""<td>{total_steps_in_match}</td></tr>"""

            # --- End Row Population ---
            html += "</tbody></table>"
            text = QtWidgets.QTextEdit(); text.setReadOnly(True); text.setHtml(html)
            player_layout.addWidget(text)

            # Plotting (logic for grouped vs single bar remains the same)
            # ... (Plotting code uses plot_data1/plot_data2) ...
            if plot_match_names:
                 num_matches = len(plot_match_names); x = np.arange(num_matches)
                 figure = plt.figure(figsize=(max(8, num_matches * 0.7), 6)); ax = figure.add_subplot(111)
                 plot_experts1 = [p[0] for p in plot_data1]; plot_rates1 = [p[1] for p in plot_data1]
                 plot_experts2 = [p[0] for p in plot_data2]; plot_rates2 = [p[1] for p in plot_data2]
                 if is_belief_agent_data and is_duo_mode:
                     bar_width = 0.35; bars1 = ax.bar(x - bar_width/2, plot_rates1, bar_width, label='Opp 1 Peak Rate'); bars2 = ax.bar(x + bar_width/2, plot_rates2, bar_width, label='Opp 2 Peak Rate')
                     # ... annotations ...
                     for bar, expert in zip(bars1, plot_experts1):
                          if expert != "N/A": ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f"T{expert}", ha='center', va='bottom', fontsize=8)
                     for bar, expert in zip(bars2, plot_experts2):
                          if expert != "N/A": ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f"T{expert}", ha='center', va='bottom', fontsize=8)
                     ax.set_title(f'Opponent Belief Peak Analysis (AI {idx+1}: {player_id})'); ax.set_ylabel('Rate of Peak Belief Type')
                 else:
                     bar_width = 0.6; bars1 = ax.bar(x, plot_rates1, bar_width, label='Dominant Rate')
                     # ... annotations ...
                     for bar, expert in zip(bars1, plot_experts1):
                          if expert != "N/A": ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f"E/T{expert}", ha='center', va='bottom', fontsize=8)
                     ax.set_title(f'Dominant Activation (AI {idx+1}: {player_id})'); ax.set_ylabel('Activation Rate')
                 ax.set_xticks(x); ax.set_xticklabels(plot_match_names, rotation=45, ha='right', fontsize=9)
                 ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0); ax.set_ylim(0, 1.05); ax.grid(axis='y', linestyle='--', alpha=0.7)
                 figure.tight_layout(rect=[0, 0, 0.85, 1]); canvas = FigureCanvasQTAgg(figure)
                 player_layout.addWidget(canvas)


            tab_widget.addTab(player_tab, f"AI Agent {idx+1}")

        # ... (Add tab widget, close button, show dialog) ...
        layout.addWidget(tab_widget)
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close); button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        dialog.setLayout(layout)
        dialog.exec_()

if __name__ == "__main__":

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
