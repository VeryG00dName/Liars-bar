import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
import torch
import os
from collections import OrderedDict

class ModelInfoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Information Viewer")
        self.root.geometry("700x600")

        self.label = tk.Label(self.root, text="Drag and drop a .pth file here", font=("Arial", 14))
        self.label.pack(pady=10)

        self.text = tk.Text(self.root, wrap="word", state="disabled", font=("Arial", 12))
        self.text.pack(expand=True, fill="both", padx=10, pady=10)

        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self.on_drop)

    def on_drop(self, event):
        file_path = event.data.strip().replace("{", "").replace("}", "")
        if os.path.isfile(file_path) and file_path.endswith(".pth"):
            self.display_model_info(file_path)

    def get_tensor_size_and_redundancy(self, obj, threshold=1e-3):
        """Recursively compute total size, exact zero weights, and near-zero weights."""
        total_size = 0
        total_params = 0
        zero_params = 0
        near_zero_params = 0

        if isinstance(obj, torch.Tensor):
            total_params = obj.numel()
            zero_params = torch.sum(obj == 0).item()
            near_zero_params = torch.sum(torch.abs(obj) < threshold).item()
            total_size = obj.element_size() * total_params

        elif isinstance(obj, dict) or isinstance(obj, OrderedDict):
            for sub_value in obj.values():
                sub_size, sub_total, sub_zero, sub_near_zero = self.get_tensor_size_and_redundancy(sub_value, threshold)
                total_size += sub_size
                total_params += sub_total
                zero_params += sub_zero
                near_zero_params += sub_near_zero

        elif isinstance(obj, list):
            for sub_value in obj:
                sub_size, sub_total, sub_zero, sub_near_zero = self.get_tensor_size_and_redundancy(sub_value, threshold)
                total_size += sub_size
                total_params += sub_total
                zero_params += sub_zero
                near_zero_params += sub_near_zero

        return total_size, total_params, zero_params, near_zero_params

    def display_model_info(self, file_path):
        try:
            data = torch.load(file_path, map_location="cpu")

            self.text.configure(state="normal")
            self.text.delete(1.0, tk.END)

            total_file_size = os.path.getsize(file_path) / (1024 * 1024)
            self.text.insert(tk.END, f"File: {file_path}\n")
            self.text.insert(tk.END, f"Total File Size: {total_file_size:.2f} MB\n\n")

            threshold = 1e-3  # Threshold for near-zero weights

            if isinstance(data, dict):
                self.text.insert(tk.END, "Detected keys in the model:\n")

                for key, value in data.items():
                    size_bytes, total_params, zero_params, near_zero_params = self.get_tensor_size_and_redundancy(value, threshold)
                    
                    size_str = f" (Size: {size_bytes / (1024 * 1024):.2f} MB)" if size_bytes > 0 else ""
                    zero_str = f", Zero Weights: {zero_params} ({(zero_params / total_params) * 100:.2f}%)" if total_params > 0 else ""
                    near_zero_str = f", Near-Zero Weights: {near_zero_params} ({(near_zero_params / total_params) * 100:.2f}%)" if total_params > 0 else ""

                    self.text.insert(tk.END, f"- {key}: {type(value)}{size_str}{zero_str}{near_zero_str}\n")

                    if isinstance(value, dict) or isinstance(value, OrderedDict):
                        self.text.insert(tk.END, "  Sub-keys:\n")
                        for sub_key, sub_value in value.items():
                            sub_size, sub_total, sub_zero, sub_near_zero = self.get_tensor_size_and_redundancy(sub_value, threshold)
                            
                            sub_size_str = f" (Size: {sub_size / (1024 * 1024):.2f} MB)" if sub_size > 0 else ""
                            sub_zero_str = f", Zero Weights: {sub_zero} ({(sub_zero / sub_total) * 100:.2f}%)" if sub_total > 0 else ""
                            sub_near_zero_str = f", Near-Zero Weights: {sub_near_zero} ({(sub_near_zero / sub_total) * 100:.2f}%)" if sub_total > 0 else ""

                            self.text.insert(tk.END, f"    - {sub_key}: {type(sub_value)}{sub_size_str}{sub_zero_str}{sub_near_zero_str}\n")

                    if key.lower() in ["episode", "epoch"]:
                        self.text.insert(tk.END, f"  {key.capitalize()}: {value}\n")
            else:
                self.text.insert(tk.END, "Unknown model format.\n")

            self.text.insert(tk.END, "\nModel loaded successfully.")
        except Exception as e:
            self.text.insert(tk.END, f"Error loading model: {str(e)}\n")

        self.text.configure(state="disabled")

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = ModelInfoApp(root)
    root.mainloop()
