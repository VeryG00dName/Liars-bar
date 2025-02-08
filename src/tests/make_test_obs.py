import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from src.env.liars_deck_env_utils_2 import encode_hand

class ObservationGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Observation Generator")
        self.root.geometry("1000x800")
        
        # Game state configuration
        self.num_players = tk.IntVar(value=3)
        self.table_card = tk.StringVar(value="King")
        self.last_action_type = tk.StringVar(value="None")
        self.last_action_count = tk.IntVar(value=0)
        self.player_hands = {}
        self.opponent_features = {}
        
        # Create UI components
        self.create_controls()
        self.create_player_configs()
        self.create_opponent_features()
        self.create_output_display()
        
    def create_controls(self):
        control_frame = ttk.LabelFrame(self.root, text="Global Settings", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Number of players selector
        ttk.Label(control_frame, text="Number of Players:").grid(row=0, column=0)
        player_selector = ttk.Spinbox(control_frame, from_=2, to=4, textvariable=self.num_players)
        player_selector.grid(row=0, column=1)
        player_selector.bind("<<Increment>>", lambda e: self.update_ui())
        player_selector.bind("<<Decrement>>", lambda e: self.update_ui())
        
        # Table card selector
        ttk.Label(control_frame, text="Table Card:").grid(row=0, column=2)
        table_card_dd = ttk.Combobox(control_frame, textvariable=self.table_card, 
                                   values=["King", "Queen", "Ace"])
        table_card_dd.grid(row=0, column=3)
        
        # Last action controls
        ttk.Label(control_frame, text="Last Action Type:").grid(row=1, column=0)
        action_type_dd = ttk.Combobox(control_frame, textvariable=self.last_action_type,
                                    values=["Play", "Challenge", "None"])
        action_type_dd.grid(row=1, column=1)
        
        ttk.Label(control_frame, text="Last Action Count:").grid(row=1, column=2)
        action_count_spin = ttk.Spinbox(control_frame, from_=0, to=5, 
                                      textvariable=self.last_action_count)
        action_count_spin.grid(row=1, column=3)
        
        # Generate button
        ttk.Button(control_frame, text="Generate Observation", 
                 command=self.generate_observation).grid(row=1, column=4)

    def create_player_configs(self):
        self.player_frame = ttk.LabelFrame(self.root, text="Player Configuration", padding=10)
        self.player_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        self.update_player_configs()

    def update_player_configs(self):
        # Clear existing widgets
        for widget in self.player_frame.winfo_children():
            widget.destroy()
            
        num_players = self.num_players.get()
        card_options = ["Ace", "King", "Queen", "Joker", "None"]
        
        for player_num in range(num_players):
            frame = ttk.Frame(self.player_frame)
            frame.pack(fill=tk.X, pady=2)
            
            # Player label
            ttk.Label(frame, text=f"Player {player_num} Hand:").pack(side=tk.LEFT)
            
            # Card selection
            self.player_hands[player_num] = []
            for i in range(5):
                card_var = tk.StringVar(value=card_options[0])
                self.player_hands[player_num].append(card_var)
                dd = ttk.Combobox(frame, textvariable=card_var, values=card_options, width=6)
                dd.pack(side=tk.LEFT, padx=2)

    def create_opponent_features(self):
        self.opponent_frame = ttk.LabelFrame(self.root, text="Opponent Features", padding=10)
        self.opponent_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        self.update_opponent_features()

    def update_opponent_features(self):
        for widget in self.opponent_frame.winfo_children():
            widget.destroy()
            
        num_players = self.num_players.get()
        for player_num in range(num_players):
            frame = ttk.Frame(self.opponent_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=f"Opponent {player_num} Features:").pack(side=tk.LEFT)
            
            # Last action type
            action_var = tk.StringVar(value="Play")
            ttk.Combobox(frame, textvariable=action_var, 
                        values=["Play", "Challenge"], width=8).pack(side=tk.LEFT)
            
            # Action count
            count_var = tk.IntVar(value=1)
            ttk.Spinbox(frame, from_=1, to=3, textvariable=count_var, width=3).pack(side=tk.LEFT)
            
            # Bluff frequency input
            bluff_frame = ttk.Frame(frame)
            bluff_frame.pack(side=tk.LEFT, padx=5)
            ttk.Label(bluff_frame, text="Bluff Freq:").pack(side=tk.LEFT)
            bluff_var = tk.StringVar(value="0.5")
            bluff_entry = ttk.Entry(bluff_frame, textvariable=bluff_var, width=5)
            bluff_entry.pack(side=tk.LEFT)
            
            self.opponent_features[player_num] = (action_var, count_var, bluff_var)

    def create_output_display(self):
        output_frame = ttk.LabelFrame(self.root, text="Generated Observation", padding=10)
        output_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        
        self.output_text = tk.Text(output_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def update_ui(self):
        self.update_player_configs()
        self.update_opponent_features()

    def validate_float(self, value):
        try:
            float_value = float(value)
            return 0.0 <= float_value <= 1.0
        except ValueError:
            return False

    def generate_observation(self):
        try:
            # Get player hands
            hands = {}
            for player_num in range(self.num_players.get()):
                hands[f"player_{player_num}"] = [var.get() for var in self.player_hands[player_num]]
            
            # Create hand vector for player 0 (filter out "None" values)
            table_card = self.table_card.get()
            filtered_hand = [card for card in hands["player_0"] if card != "None"]
            hand_vector = encode_hand(filtered_hand, table_card)
            
            # Create last action value
            if self.last_action_type.get() == "Play":
                last_action_val = np.array([self.last_action_count.get()], dtype=np.float32)
            else:
                last_action_val = np.array([0], dtype=np.float32)
            
            # Create active players vector
            active_players = np.ones(self.num_players.get(), dtype=np.float32)
            
            # Create opponent features
            opponent_features = []
            for player_num in range(1, self.num_players.get()):
                action_type, count, bluff = self.opponent_features[player_num]
                
                # Validate bluff frequency
                if not self.validate_float(bluff.get()):
                    raise ValueError(f"Invalid bluff frequency for opponent {player_num}. Must be between 0 and 1")
                
                # Action type one-hot encoding
                action_onehot = [0.0, 0.0, 0.0]
                if action_type.get() == "Play":
                    action_onehot[1] = 1.0
                elif action_type.get() == "Challenge":
                    action_onehot[2] = 1.0
                else:
                    action_onehot[0] = 1.0
                
                # Normalized count
                norm_count = count.get() / 5.0
                
                # Bluff frequency
                bluff_freq = float(bluff.get())
                
                opponent_features.extend(action_onehot + [norm_count, bluff_freq])
            
            # Create final observation vector
            observation = np.concatenate([
                hand_vector,
                last_action_val,
                active_players,
                np.array(opponent_features, dtype=np.float32)
            ])
            
            # Display results
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Generated Observation:\n\n")
            self.output_text.insert(tk.END, f"Table Card: {table_card}\n")
            self.output_text.insert(tk.END, f"Player 0 Hand: {hands['player_0']}\n")
            self.output_text.insert(tk.END, f"Last Action: {self.last_action_type.get()} ({self.last_action_count.get()})\n")
            self.output_text.insert(tk.END, "\nObservation Vector:\n")
            self.output_text.insert(tk.END, str(observation))
            self.output_text.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate observation: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ObservationGeneratorGUI(root)
    root.mainloop()