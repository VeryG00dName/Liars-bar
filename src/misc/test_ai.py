import os
import tkinter as tk
from tkinter import ttk, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
import torch
import numpy as np
from collections import defaultdict
from src import config
from src.model.models import PolicyNetwork, OpponentBehaviorPredictor  # Added OpponentBehaviorPredictor

class StrategyAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Strategy Analyzer")
        self.root.geometry("800x600")
        
        self.loaded_models = {}
        self.selected_agent = None
        self.test_cases = []
        
        self.create_file_drop_zone()
        self.create_model_info_panel()
        self.create_agent_selection()
        self.create_test_button()
        self.create_result_display()
        
        self.create_predefined_test_cases()

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

    def create_agent_selection(self):
        frame = ttk.LabelFrame(self.root, text="Agent Selection", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)
        self.agent_selector = ttk.Combobox(frame, state="readonly", width=50)
        self.agent_selector.pack(pady=5)

    def create_test_button(self):
        frame = ttk.Frame(self.root)
        frame.pack(pady=10)
        ttk.Button(frame, text="Analyze Strategy", command=self.run_analysis).pack(side=tk.LEFT, padx=5)

    def create_result_display(self):
        frame = ttk.LabelFrame(self.root, text="Analysis Results", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.result_text = tk.Text(frame, wrap=tk.WORD, state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def create_predefined_test_cases(self):
        # Base observation structure (without OBP predictions)
        base_obs_length = config.INPUT_DIM - (config.NUM_PLAYERS - 1)
        base_obs = np.zeros(base_obs_length, dtype=np.float32)
        
        # Test case 1: Empty hand
        self.test_cases.append(("Empty hand", base_obs.copy()))
        
        # Test case 2: Table card dominance
        table_obs = base_obs.copy()
        table_obs[0:2] = [1.0, 0.0]  # Full table cards
        self.test_cases.append(("Table card dominance", table_obs))
        
        # Test case 3: High bluff opponents
        bluff_obs = base_obs.copy()
        # Set opponent features (5 features per opponent × 2 opponents)
        bluff_obs[-10:] = [1.0, 0.8, 1.0, 0.9, 0.7] * 2  # High bluff probabilities
        self.test_cases.append(("High bluff opponents", bluff_obs))

    def on_file_drop(self, event):
        file_path = event.data.strip().replace("{", "").replace("}", "").strip('"')
        if not file_path.endswith(".pth"):
            self.show_info("Only .pth files are supported")
            return
        
        try:
            checkpoint = torch.load(file_path, map_location="cpu")
            self.loaded_models[file_path] = checkpoint
            self.file_list.insert(tk.END, os.path.basename(file_path))
            self.update_agent_selector()
            self.show_info(f"Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            self.show_info(f"Error: {str(e)}")

    def update_agent_selector(self):
        agents = set()
        for checkpoint in self.loaded_models.values():
            if 'policy_nets' in checkpoint:
                agents.update(checkpoint['policy_nets'].keys())
        self.agent_selector['values'] = list(agents)

    def run_analysis(self):
        if not self.loaded_models or not self.agent_selector.get():
            messagebox.showerror("Error", "Load a model and select an agent first!")
            return

        # Load selected agent
        selected_agent = self.agent_selector.get()
        checkpoint = next(iter(self.loaded_models.values()))
        
        if 'policy_nets' not in checkpoint or selected_agent not in checkpoint['policy_nets']:
            messagebox.showerror("Error", f"Agent '{selected_agent}' not found in the loaded model.")
            return

        policy_state = checkpoint['policy_nets'][selected_agent]
        
        # Initialize Policy Network
        hidden_dim = self.get_hidden_dim(policy_state)
        policy_net = PolicyNetwork(
                        input_dim=config.INPUT_DIM,
                        hidden_dim=hidden_dim,
                        output_dim=config.OUTPUT_DIM,
                        use_lstm=True,
                        use_dropout=True,
                        use_layer_norm=True
                    )
        policy_net.load_state_dict(policy_state)
        policy_net.eval()

        # Load OBP model from checkpoint
        if 'obp_model' not in checkpoint:
            messagebox.showerror("Error", "OBP model not found in the checkpoint.")
            return

        obp_state = checkpoint['obp_model']
        obp_hidden_dim = self.get_obp_hidden_dim(obp_state)
        obp_model = OpponentBehaviorPredictor(
            input_dim=5,  # Each opponent's feature vector size
            hidden_dim=obp_hidden_dim,
            output_dim=2
        )
        obp_model.load_state_dict(obp_state)
        obp_model.eval()

        # Run through test cases
        results = defaultdict(list)
        for case_name, obs in self.test_cases:
            action_mask = self.generate_action_mask(obs)
            action_probs = self.get_action_probs(policy_net, obp_model, obs, action_mask)
            results[case_name] = action_probs

        # Analyze and display results
        analysis = self.analyze_results(results)
        self.display_results(analysis)

    def get_hidden_dim(self, state_dict):
        for key in state_dict.keys():
            if key.endswith('.weight') and 'fc' in key:
                return state_dict[key].shape[0]
        return 128

    def get_obp_hidden_dim(self, state_dict, layer_prefix='fc1'):
        for key in state_dict.keys():
            if key.endswith('.weight') and 'fc' in key:
                return state_dict[key].shape[0]
        return 128  # Fallback default

    def generate_action_mask(self, obs):
        # Simplified action mask logic
        mask = [1]*7
        table_cards = int(obs[0] * 5)
        non_table_cards = int(obs[1] * 5)
        
        for i in range(3):
            mask[i] = 1 if table_cards >= (i+1) else 0
        for i in range(3,6):
            mask[i] = 1 if non_table_cards >= (i-2) else 0
        return mask

    def get_action_probs(self, policy_net, obp_model, obs, action_mask, num_players=3):
        num_opponents = num_players - 1
        opp_feature_dim = 5  # Each opponent's feature vector size
        
        # Verify observation length
        expected_obs_length = config.INPUT_DIM - num_opponents  # Original obs without OBP probs
        if len(obs) != expected_obs_length:
            raise ValueError(f"Observation length mismatch. Expected {expected_obs_length}, got {len(obs)}")
        
        # Extract opponent features from observation
        opp_features_start = len(obs) - (num_opponents * opp_feature_dim)
        obp_probs = []
        
        for i in range(num_opponents):
            start_idx = opp_features_start + i * opp_feature_dim
            end_idx = start_idx + opp_feature_dim
            opp_vec = obs[start_idx:end_idx]
            
            # Ensure we have valid features
            if len(opp_vec) != opp_feature_dim:
                raise ValueError(f"Invalid opponent features. Expected {opp_feature_dim} elements, got {len(opp_vec)}")
            
            opp_vec_tensor = torch.tensor(opp_vec, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = obp_model(opp_vec_tensor)
                probs = torch.softmax(logits, dim=-1)
                obp_probs.append(probs[0, 1].item())  # Bluff probability
        
        # Combine original obs with OBP predictions
        final_obs = np.concatenate([obs, np.array(obp_probs, dtype=np.float32)])
        
        # Get policy network predictions
        with torch.no_grad():
            obs_tensor = torch.tensor(final_obs, dtype=torch.float32).unsqueeze(0)
            probs, _ = policy_net(obs_tensor)
            masked_probs = probs * torch.tensor(action_mask)
            if masked_probs.sum() > 0:
                masked_probs /= masked_probs.sum()
            return masked_probs.numpy().flatten()

    def analyze_results(self, results):
        strategy_insights = []
        
        for case_name, probs in results.items():
            top_action = np.argmax(probs)
            action_dist = {i: f"{p*100:.1f}%" for i, p in enumerate(probs)}
            insight = f"{case_name}:\n- Most likely action: {top_action}\n- Action distribution: {action_dist}\n"
            strategy_insights.append(insight)
        
        # Detect overall patterns
        challenge_prob = np.mean([probs[6] for probs in results.values()])
        if challenge_prob > 0.3:
            strategy_insights.append("\nStrategic tendency: Aggressive challenger")
        elif challenge_prob < 0.1:
            strategy_insights.append("\nStrategic tendency: Avoids challenges")
            
        return "\n".join(strategy_insights)

    def display_results(self, analysis):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Strategy Analysis Results:\n\n")
        self.result_text.insert(tk.END, analysis)
        self.result_text.config(state=tk.DISABLED)

    def show_info(self, message):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, message)
        self.info_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = StrategyAnalyzerGUI(root)
    root.mainloop()
