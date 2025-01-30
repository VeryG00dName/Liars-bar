import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
import torch
import os

class ModelInfoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Information Viewer")
        self.root.geometry("600x400")

        self.label = tk.Label(self.root, text="Drag and drop a .pth file here", font=("Arial", 16))
        self.label.pack(pady=20)

        self.text = tk.Text(self.root, wrap="word", state="disabled", font=("Arial", 12))
        self.text.pack(expand=True, fill="both", padx=10, pady=10)

        # Enable drag-and-drop support
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

            self.text.insert(tk.END, f"File: {file_path}\n\n")

            if isinstance(data, dict):
                self.text.insert(tk.END, "Detected keys in the model:\n")
                for key, value in data.items():
                    self.text.insert(tk.END, f"- {key}: {type(value)}\n")

                    if isinstance(value, dict):
                        self.text.insert(tk.END, "  Sub-keys:\n")
                        for sub_key in value.keys():
                            self.text.insert(tk.END, f"    - {sub_key}\n")

                    if key.lower() in ["episode", "epoch"]:
                        self.text.insert(tk.END, f"  {key.capitalize()}: {value}\n")
            else:
                self.text.insert(tk.END, "Unknown model format.\n")

            self.text.insert(tk.END, "\nModel loaded successfully.")
        except Exception as e:
            self.text.insert(tk.END, f"Error loading model: {str(e)}\n")

        self.text.configure(state="disabled")

if __name__ == "__main__":
    root = TkinterDnD.Tk()  # Use TkinterDnD.Tk for drag-and-drop support
    app = ModelInfoApp(root)
    root.mainloop()
