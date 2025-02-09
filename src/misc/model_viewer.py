# src/misc/model_viewer.py
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
import torch
import os
from collections import OrderedDict
import numpy as np  # for numerical operations

def aggregate_weights(matrix, target_count, axis=0):
    """Aggregates rows (or columns) of a 2D NumPy array into `target_count` groups by averaging."""
    if axis == 1:
        matrix = matrix.T
        aggregated = aggregate_weights(matrix, target_count, axis=0)
        return aggregated.T

    n, d = matrix.shape
    if n <= target_count:
        return matrix
    group_size = n // target_count
    aggregated_groups = []
    for i in range(target_count):
        start = i * group_size
        if i == target_count - 1:
            group = matrix[start:, :]
        else:
            group = matrix[start:start+group_size, :]
        group_avg = group.mean(axis=0, keepdims=True)
        aggregated_groups.append(group_avg)
    return np.concatenate(aggregated_groups, axis=0)

class ModelInfoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Information Viewer")
        self.root.geometry("900x1100")  # Increased window width for clarity
        
        self.label = tk.Label(self.root, text="Drag and drop a .pth file here", font=("Arial", 14))
        self.label.pack(pady=10)
        
        self.text_frame = tk.Frame(self.root)
        self.text_frame.pack(expand=True, fill="both", padx=10, pady=(0, 10))
        
        self.text = tk.Text(self.text_frame, wrap="word", state="disabled", font=("Arial", 12), height=10)
        self.text.pack(expand=True, fill="both", side="top")
        
        # Increased canvas height for better spacing
        self.canvas = tk.Canvas(self.root, bg="white", height=600)
        self.canvas.pack(expand=False, fill="both", padx=10, pady=(0, 10))
        
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self.on_drop)

    def on_drop(self, event):
        file_path = event.data.strip().replace("{", "").replace("}", "")
        if os.path.isfile(file_path) and file_path.endswith(".pth"):
            self.display_model_info(file_path)

    def display_model_info(self, file_path):
        try:
            data = torch.load(file_path, map_location="cpu")

            self.text.configure(state="normal")
            self.text.delete(1.0, tk.END)

            total_file_size = os.path.getsize(file_path) / (1024 * 1024)
            self.text.insert(tk.END, f"File: {file_path}\n")
            self.text.insert(tk.END, f"Total File Size: {total_file_size:.2f} MB\n\n")

            if "policy_nets" in data and isinstance(data["policy_nets"], dict) and "player_1" in data["policy_nets"]:
                policy_net = data["policy_nets"]["player_1"]
                self.text.insert(tk.END, "\nVisualizing policy network for player_1\n")
                self.draw_policy_network_visualization(policy_net)
            else:
                self.text.insert(tk.END, "\nNo policy network found for player_1.\n")
                self.canvas.delete("all")

            self.text.insert(tk.END, "\nModel loaded successfully.")
        except Exception as e:
            self.text.insert(tk.END, f"Error loading model: {str(e)}\n")

        self.text.configure(state="disabled")

    def draw_policy_network_visualization(self, policy_net):
        """Visualizes the policy network with increased spacing and only the strongest/weakest 3 connections per neuron."""
        self.canvas.delete("all")
        
        layers = []
        for key, value in policy_net.items():
            if isinstance(value, torch.Tensor) and value.dim() == 2 and key.endswith(".weight"):
                layers.append(value)
        
        if not layers:
            self.canvas.create_text(300, 300, text="No weight matrices found in policy network.", font=("Arial", 14))
            return

        if len(layers) >= 2:
            W_in = layers[0]  # First weight matrix (input to hidden)
            W_out = layers[-1]  # Last weight matrix (hidden to output)
            n_input = W_in.shape[1]
            n_output = W_out.shape[0]
            hidden_size_agg = 10  # Fixed aggregated hidden layer size

            W_in_np = W_in.cpu().detach().numpy()
            effective_in_hidden = aggregate_weights(W_in_np, hidden_size_agg, axis=0)

            W_out_np = W_out.cpu().detach().numpy()
            effective_hidden_out = aggregate_weights(W_out_np, hidden_size_agg, axis=1)

            sizes = [n_input, hidden_size_agg, n_output]
            weight_matrices = [effective_in_hidden, effective_hidden_out]
        else:
            W = layers[0].cpu().detach().numpy()
            sizes = [W.shape[1], W.shape[0]]
            weight_matrices = [W]

        self._draw_network(sizes, weight_matrices)

    def _draw_network(self, sizes, weight_matrices):
        """Draws the network while only visualizing the 3 strongest and 3 weakest connections per neuron."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1:
            canvas_width = 800
        if canvas_height <= 1:
            canvas_height = 600

        margin_x = 80
        margin_y = 50
        available_width = canvas_width - 2 * margin_x
        num_layers = len(sizes)
        
        x_positions = [margin_x + (available_width * i / (num_layers - 1)) for i in range(num_layers)]
        nodes = []
        
        for i, num_neurons in enumerate(sizes):
            available_height = canvas_height - 2 * margin_y
            layer_nodes = [(x_positions[i], margin_y + j * (available_height / (num_neurons - 1))) for j in range(num_neurons)]
            nodes.append(layer_nodes)

        radius = 8
        for layer in nodes:
            for (x, y) in layer:
                self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="lightgrey", outline="black")

        for layer_idx in range(num_layers - 1):
            weight_matrix = weight_matrices[layer_idx]
            max_abs = abs(weight_matrix).max() if weight_matrix.size > 0 else 1

            for out_idx, out_node in enumerate(nodes[layer_idx + 1]):
                for in_idx, in_node in enumerate(nodes[layer_idx]):
                    weight_val = weight_matrix[out_idx, in_idx]

                    sorted_weights = np.argsort(abs(weight_matrix[out_idx]))  # Sort by absolute value
                    top_3_strongest = sorted_weights[-3:]  # Three strongest connections
                    top_3_weakest = sorted_weights[:3]  # Three weakest connections

                    if in_idx in top_3_strongest or in_idx in top_3_weakest:
                        color = "red" if weight_val > 0 else "blue"
                        line_width = 1 + (abs(weight_val) / max_abs) * 4
                        self.canvas.create_line(in_node[0], in_node[1], out_node[0], out_node[1], fill=color, width=line_width)

        self.canvas.create_text(x_positions[0], canvas_height - margin_y/2, text="Input", font=("Arial", 12, "bold"))
        self.canvas.create_text(x_positions[1], canvas_height - margin_y/2, text="Hidden (aggregated)", font=("Arial", 12, "bold"))
        self.canvas.create_text(x_positions[-1], canvas_height - margin_y/2, text="Output", font=("Arial", 12, "bold"))

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ModelInfoApp(root)
    root.mainloop()
