# src/main.py
import tkinter as tk
from tkinter import ttk
from tkinterdnd2 import TkinterDnD, DND_FILES
import logging
import threading
import time
import os
import torch
import numpy as np
from PIL import Image, ImageTk

from src.eval.gui_evaluate import TournamentManager
from src.env.liars_deck_env_core import LiarsDeckEnv
from src.model.models import PolicyNetwork, ValueNetwork, OpponentBehaviorPredictor
from src.training.train import train
from src import config

class TrainingApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Liars Deck Training Manager")
        self.geometry("1200x800")
        self.configure_logger()
        self.loaded_models = {}
        self.training_thread = None
        self.stop_training_flag = False
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def configure_logger(self):
        self.logger = logging.getLogger('MainApp')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s:%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def create_widgets(self):
        self.create_model_manager()
        self.create_training_controls()
        self.create_evaluation_panel()
        self.create_status_bar()

    def create_model_manager(self):
        # Model Management Frame
        model_frame = ttk.LabelFrame(self, text="Model Management", padding=10)
        model_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        # Model List
        self.model_list = tk.Listbox(model_frame, height=8, selectmode=tk.SINGLE)
        self.model_list.pack(fill=tk.BOTH, expand=True)
        self.model_list.drop_target_register(DND_FILES)
        self.model_list.dnd_bind('<<Drop>>', self.on_model_drop)

        # Model Controls
        btn_frame = ttk.Frame(model_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="New Agent", command=self.create_new_agent).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Delete", command=self.delete_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Refresh", command=self.update_model_list).pack(side=tk.RIGHT, padx=2)

    def create_training_controls(self):
        # Training Frame
        train_frame = ttk.LabelFrame(self, text="Training Configuration", padding=10)
        train_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        # Agent Selection
        ttk.Label(train_frame, text="Agents to Train:").grid(row=0, column=0, sticky=tk.W)
        self.agent_selector = tk.Listbox(train_frame, selectmode=tk.MULTIPLE, height=3)
        self.agent_selector.grid(row=0, column=1, sticky=tk.EW, padx=5)

        # Hyperparameters
        params = [
            ('Learning Rate', '3e-4'),
            ('Episodes', '1000'),
            ('Entropy Coef', '0.1')
        ]
        
        self.entries = {}
        for i, (label, default) in enumerate(params):
            ttk.Label(train_frame, text=label+":").grid(row=i+1, column=0, sticky=tk.W)
            entry = ttk.Entry(train_frame)
            entry.insert(0, default)
            entry.grid(row=i+1, column=1, sticky=tk.EW, padx=5)
            self.entries[label] = entry

        # Training Controls
        btn_frame = ttk.Frame(train_frame)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        self.train_btn = ttk.Button(btn_frame, text="Start Training", command=self.start_training)
        self.train_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Stop", command=self.stop_training).pack(side=tk.LEFT, padx=5)

        # Training Log
        self.log_text = tk.Text(train_frame, height=10, state=tk.DISABLED)
        self.log_text.grid(row=6, column=0, columnspan=2, sticky="nsew")

    def delete_model(self):
        selection = self.model_list.curselection()
        if not selection:
            self.log("No model selected to delete!", error=True)
            return
        
        model_name = self.model_list.get(selection[0])
        
        try:
            if model_name in self.loaded_models:
                model_path = self.loaded_models[model_name]['source_file']
                del self.loaded_models[model_name]
                
                if os.path.exists(model_path):
                    os.remove(model_path)
                    self.log(f"Deleted model: {model_name}")
                else:
                    self.log(f"Model file not found: {model_path}", error=True)
            
            self.update_model_list()
            
        except Exception as e:
            self.log(f"Error deleting model: {str(e)}", error=True)

    def create_evaluation_panel(self):
        eval_notebook = ttk.Notebook(self)
        eval_notebook.grid(row=0, column=1, rowspan=2, padx=10, pady=5, sticky="nsew")

        tournament_frame = ttk.Frame(eval_notebook)
        self.create_tournament_controls(tournament_frame)
        eval_notebook.add(tournament_frame, text="Tournament")

    def create_tournament_controls(self, parent):
        ttk.Label(parent, text="Select Participants:").grid(row=0, column=0, padx=5, pady=5)
        
        list_frame = ttk.Frame(parent)
        list_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        
        self.tournament_list = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, height=6)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.tournament_list.yview)
        self.tournament_list.configure(yscrollcommand=scrollbar.set)
        
        self.tournament_list.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        ttk.Button(btn_frame, text="Run Tournament", command=self.run_tournament).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Refresh List", command=self.update_model_list).pack(side=tk.LEFT, padx=5)
        
        self.tournament_results = tk.Text(parent, height=10, state=tk.DISABLED)
        self.tournament_results.grid(row=3, column=0, columnspan=2, sticky="nsew")

    def create_status_bar(self):
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")

    def on_model_drop(self, event):
        files = self.tk.splitlist(event.data)
        for f in files:
            if f.endswith('.pth'):
                try:
                    self.load_model(f)
                    self.update_model_list()
                except Exception as e:
                    self.log(f"Error loading {f}: {str(e)}", error=True)

    def load_model(self, path):
        """
        Load a model checkpoint (.pth). Supports:
          - Combined multi-agent checkpoints (keys: "policy_nets", "value_nets", "obp_model").
          - Single-agent checkpoints (keys: "policy_net", "value_net", "obp_model").
        """
        try:
            checkpoint = torch.load(path, map_location='cpu')
            model_name = os.path.basename(path)
            model_base = os.path.splitext(model_name)[0]  # Extract filename without extension

            if 'policy_nets' in checkpoint:
                # Combined checkpoint => multiple agents in one file
                for agent_name in checkpoint['policy_nets']:
                    new_agent_name = f"{model_base}.{agent_name}"
                    self.loaded_models[new_agent_name] = {
                        'policy_net': checkpoint['policy_nets'][agent_name],
                        'value_net': checkpoint['value_nets'][agent_name],
                        'obp_model': checkpoint.get('obp_model', None),
                        'source_file': path,
                        'is_combined': True
                    }
            else:
                # Single-agent checkpoint => keys: "policy_net", "value_net", "obp_model"
                # (or old ones might be "policy", "value", "obp" - adapt if needed)
                if 'policy_net' in checkpoint and 'value_net' in checkpoint:
                    self.loaded_models[model_name] = {
                        'policy_net': checkpoint['policy_net'],
                        'value_net': checkpoint['value_net'],
                        'obp_model': checkpoint.get('obp_model', None),
                        'source_file': path,
                        'is_combined': False
                    }
                elif 'policy' in checkpoint and 'value' in checkpoint:
                    # If you have older single-agent files that used "policy", "value", "obp"
                    self.loaded_models[model_name] = {
                        'policy_net': checkpoint['policy'],
                        'value_net': checkpoint['value'],
                        'obp_model': checkpoint.get('obp', None),
                        'source_file': path,
                        'is_combined': False
                    }
                else:
                    raise ValueError("Checkpoint format not recognized (missing policy_net/value_net or policy/value).")

            self.update_model_list()
            self.log(f"Loaded: {model_name}")

        except Exception as e:
            self.log(f"Error loading model from file: {str(e)}", error=True)

    def create_new_agent(self):
        """
        Creates a new agent with random initialization for Policy, Value, and OBP,
        then saves it with keys "policy_net", "value_net", "obp_model".
        """
        model_name = f"agent_{int(time.time())}.pth"
        model_path = os.path.join(config.CHECKPOINT_DIR, model_name)

        policy = PolicyNetwork(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            use_lstm=True,
            use_dropout=True,
            use_layer_norm=True
        )
        value = ValueNetwork(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            use_dropout=True,
            use_layer_norm=True
        )
        obp = OpponentBehaviorPredictor(
            input_dim=config.OPPONENT_INPUT_DIM,
            hidden_dim=config.OPPONENT_HIDDEN_DIM,
            output_dim=2
        )

        torch.save({
            'policy_net': policy.state_dict(),
            'value_net': value.state_dict(),
            'obp_model': obp.state_dict(),
        }, model_path)
        
        self.load_model(model_path)
        self.update_model_list()
        self.log(f"Created and loaded new agent: {model_name}")

    def update_model_list(self):
        self.model_list.delete(0, tk.END)
        self.tournament_list.delete(0, tk.END)
        self.agent_selector.delete(0, tk.END)
        
        agent_names = list(self.loaded_models.keys())
        
        for agent in agent_names:
            self.model_list.insert(tk.END, agent)
            self.tournament_list.insert(tk.END, agent)
            self.agent_selector.insert(tk.END, agent)

        # Highlight combined checkpoints in blue
        for i, agent in enumerate(agent_names):
            if self.loaded_models[agent].get('is_combined', False):
                self.model_list.itemconfig(i, {'fg': 'blue'})

    def start_training(self):
        selected_indices = self.agent_selector.curselection()
        if not selected_indices:
            self.log("No agents selected for training!", error=True)
            return

        selected_agents = [self.agent_selector.get(i) for i in selected_indices]
        
        self.stop_training_flag = False
        self.train_btn.config(state=tk.DISABLED)
        
        training_config = {
            'lr': float(self.entries['Learning Rate'].get()),
            'episodes': int(self.entries['Episodes'].get()),
            'entropy_coef': float(self.entries['Entropy Coef'].get())
        }
        
        self.training_thread = threading.Thread(
            target=self.run_training,
            args=(training_config, selected_agents),
            daemon=True
        )
        self.training_thread.start()

    def run_training(self, training_config, selected_agents):
        """
        Runs MARL training on the selected agents. We unify them into a single environment,
        mapping each selected agent to an env "player_i" name. Then we call `train(...)`.
        """
        try:
            num_players = len(selected_agents)
            env = LiarsDeckEnv(num_players=num_players)
            device = torch.device(config.DEVICE)
            
            # Build agent name mapping: user_agent_name -> env_agent_name
            env_agent_names = [f"player_{i}" for i in range(num_players)]
            agent_name_map = dict(zip(selected_agents, env_agent_names))
            
            policy_nets = {}
            value_nets = {}
            optimizers_policy = {}
            optimizers_value = {}
            entropy_coefs = {}

            # Single OBP shared among all selected agents for training
            obp_model = OpponentBehaviorPredictor(
                input_dim=config.OPPONENT_INPUT_DIM,
                hidden_dim=config.OPPONENT_HIDDEN_DIM,
                output_dim=2
            ).to(device)
            obp_optimizer = torch.optim.Adam(obp_model.parameters(), lr=training_config['lr'])

            # Load existing states into each net
            for user_agent_name in selected_agents:
                env_agent_name = agent_name_map[user_agent_name]
                model_dict = self.loaded_models[user_agent_name]

                # Build Policy/Value Networks
                policy = PolicyNetwork(
                    input_dim=config.INPUT_DIM,
                    hidden_dim=config.HIDDEN_DIM,
                    output_dim=config.OUTPUT_DIM,
                    use_lstm=True,
                    use_dropout=True,
                    use_layer_norm=True
                ).to(device)
                policy.load_state_dict(model_dict['policy_net'])

                value = ValueNetwork(
                    input_dim=config.INPUT_DIM,
                    hidden_dim=config.HIDDEN_DIM,
                    use_dropout=True,
                    use_layer_norm=True
                ).to(device)
                value.load_state_dict(model_dict['value_net'])

                # If the model had an obp_model, load that into our single shared obp_model
                if model_dict['obp_model'] is not None:
                    obp_model.load_state_dict(model_dict['obp_model'], strict=False)

                policy_nets[env_agent_name] = policy
                value_nets[env_agent_name] = value

                optimizers_policy[env_agent_name] = torch.optim.Adam(policy.parameters(), lr=training_config['lr'])
                optimizers_value[env_agent_name] = torch.optim.Adam(value.parameters(), lr=training_config['lr'])

                entropy_coefs[env_agent_name] = training_config['entropy_coef']

            # Build agents_dict for the new train function
            agents_dict = {
                env_agent: {
                    'policy_net': policy_nets[env_agent],
                    'value_net': value_nets[env_agent],
                    'optimizer_policy': optimizers_policy[env_agent],
                    'optimizer_value': optimizers_value[env_agent],
                    'entropy_coef': entropy_coefs[env_agent]
                }
                for env_agent in env_agent_names
            }

            # Call the training function
            results = train(
                agents_dict=agents_dict,
                env=env,
                device=device,
                obp_model=obp_model,
                obp_optimizer=obp_optimizer,
                num_episodes=training_config['episodes'],
                logger=self.logger,
                agent_mapping=None  # Adjust if environment agents need mapping
            )

            # Save the updated single-agent files
            for user_agent_name in selected_agents:
                env_agent_name = agent_name_map[user_agent_name]
                is_combined = self.loaded_models[user_agent_name].get('is_combined', False)

                # Overwrite the original single-agent checkpoint if it's not from a combined file
                if not is_combined:
                    torch.save({
                        'policy_net': policy_nets[env_agent_name].state_dict(),
                        'value_net': value_nets[env_agent_name].state_dict(),
                        'obp_model': results['obp_model'].state_dict()
                    }, self.loaded_models[user_agent_name]['source_file'])

            # Also save a new combined checkpoint for these agents
            combined_path = os.path.join(
                config.CHECKPOINT_DIR, 
                f"combined_checkpoint_ep{training_config['episodes']}.pth"
            )
            torch.save({
                'policy_nets': {a: net.state_dict() for a, net in policy_nets.items()},
                'value_nets': {a: net.state_dict() for a, net in value_nets.items()},
                'obp_model': results['obp_model'].state_dict() if results['obp_model'] else None,
                'episode': training_config['episodes']
            }, combined_path)

            self.log("Training completed successfully! Saved:")
            self.log(f"- Individual agent updates for: {', '.join(selected_agents)}")
            self.log(f"- Combined checkpoint: {os.path.basename(combined_path)}")

        except Exception as e:
            self.log(f"Training error: {str(e)}", error=True)
        finally:
            self.after(0, lambda: self.train_btn.config(state=tk.NORMAL))

    def stop_training(self):
        self.stop_training_flag = True
        self.log("Training stop requested...")

    def run_tournament(self):
        selected = self.tournament_list.curselection()
        if len(selected) < 2:
            self.log("Select at least 2 participants!", error=True)
            return
        
        participants = [self.tournament_list.get(i) for i in selected]
        self.log(f"Starting tournament with {len(participants)} agents...")
        
        # Create tournament manager
        self.tournament_manager = TournamentManager(
            loaded_models=self.loaded_models,
            update_callback=self._update_tournament_results,
            log_callback=self.log
        )
        self.tournament_manager.initialize_players(participants)
        
        # Clear results
        self.tournament_results.config(state=tk.NORMAL)
        self.tournament_results.delete(1.0, tk.END)
        self.tournament_results.config(state=tk.DISABLED)
        
        # Start thread so UI remains responsive
        threading.Thread(
            target=self.tournament_manager.run_tournament,
            daemon=True
        ).start()

    def _update_tournament_results(self, results):
        self.after(0, lambda: self._display_results(results))

    def _display_results(self, results):
        self.tournament_results.config(state=tk.NORMAL)
        
        if results.get('final', False):
            self.tournament_results.insert(tk.END, "=== Final Results ===\n")
        else:
            self.tournament_results.insert(tk.END, f"=== Round {results['round']} Results ===\n")
            
        for player in results['players']:
            line = f"{player['name']}: Score {player['score']:.2f} - Wins {player['wins']}\n"
            self.tournament_results.insert(tk.END, line)
            
        self.tournament_results.insert(tk.END, "\n")
        self.tournament_results.see(tk.END)
        self.tournament_results.config(state=tk.DISABLED)

    def log(self, message, error=False):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        if error:
            self.log_text.tag_add('error', 'end-1l linestart', 'end-1l lineend')
        self.log_text.config(state=tk.DISABLED)
        self.log_text.see(tk.END)
        self.status_var.set(message)

    def on_close(self):
        if self.training_thread and self.training_thread.is_alive():
            self.stop_training()
            self.training_thread.join()
        self.destroy()

if __name__ == "__main__":
    app = TrainingApp()
    app.mainloop()
