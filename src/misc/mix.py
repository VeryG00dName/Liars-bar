# src/misc/mix_gui.py

import os
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
import torch
from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MixGUI")


class MixGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mix AI Agents and Players")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)

        self.loaded_checkpoints = {}  # key: file_path, value: checkpoint dict
        self.agent_display_map = {}   # key: display_name, value: (file_path, agent_name)

        self.create_file_drop_zone()
        self.create_model_info_panel()
        self.create_agent_selection_panel()
        self.create_control_buttons()
        self.create_save_panel()

        self.combined_checkpoint = None

    def create_file_drop_zone(self):
        frame = ttk.LabelFrame(self.root, text="Model Files", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)

        self.file_listbox = tk.Listbox(frame, height=5, selectmode=tk.MULTIPLE)
        self.file_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        drop_label = ttk.Label(frame, text="Drag and drop .pth files here")
        drop_label.pack(pady=5)

        # Register drag-and-drop
        for widget in [frame, self.file_listbox, drop_label]:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", self.on_file_drop)

    def create_model_info_panel(self):
        frame = ttk.LabelFrame(self.root, text="Loaded Models Information", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.info_text = tk.Text(frame, wrap=tk.WORD, state=tk.DISABLED, height=10)
        self.info_text.pack(fill=tk.BOTH, expand=True)

    def create_agent_selection_panel(self):
        frame = ttk.LabelFrame(self.root, text="Agents Selection for Mixing", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)

        # Create labels and drop-down menus for each player slot
        self.agent_vars = {
            "player_0": tk.StringVar(),
            "player_1": tk.StringVar(),
            "player_2": tk.StringVar(),
        }

        for idx, player in enumerate(["player_0", "player_1", "player_2"]):
            label = ttk.Label(frame, text=f"{player}:")
            label.grid(row=idx, column=0, sticky=tk.W, pady=5, padx=5)

            combobox = ttk.Combobox(
                frame,
                textvariable=self.agent_vars[player],
                state="readonly",
                width=80
            )
            combobox.grid(row=idx, column=1, sticky=tk.EW, pady=5, padx=5)
            combobox['values'] = []
            setattr(self, f"combobox_{player}", combobox)

        frame.columnconfigure(1, weight=1)

    def create_control_buttons(self):
        frame = ttk.Frame(self.root)
        frame.pack(pady=10)

        ttk.Button(frame, text="Refresh Selection", command=self.refresh_agent_dropdowns).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Mix Checkpoints", command=self.mix_checkpoints).pack(side=tk.LEFT, padx=5)

    def create_save_panel(self):
        frame = ttk.LabelFrame(self.root, text="Save Combined Checkpoint", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=5)

        self.save_path_var = tk.StringVar()

        save_entry = ttk.Entry(frame, textvariable=self.save_path_var, width=80)
        save_entry.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)

        ttk.Button(frame, text="Browse", command=self.browse_save_location).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(frame, text="Save Combined Checkpoint", command=self.save_combined_checkpoint).pack(side=tk.LEFT, padx=5, pady=5)

    def on_file_drop(self, event):
        files = self.root.tk.splitlist(event.data)
        for file_path in files:
            file_path = self.normalize_file_path(file_path)
            if not file_path.endswith(".pth"):
                self.show_info(f"Only .pth files are supported. Skipping: {os.path.basename(file_path)}")
                continue
            if file_path in self.loaded_checkpoints:
                self.show_info(f"Model already loaded: {os.path.basename(file_path)}")
                continue
            try:
                self.load_checkpoint(file_path)
                self.file_listbox.insert(tk.END, os.path.basename(file_path))
                self.update_model_info()
                self.refresh_agent_dropdowns()
                self.show_info(f"Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                self.show_info(f"Error loading {os.path.basename(file_path)}: {e}")

    def normalize_file_path(self, file_path):
        if os.name == 'nt':
            file_path = file_path.replace("{", "").replace("}", "").strip('"')
        else:
            if file_path.startswith("file://"):
                file_path = file_path[7:]
        return os.path.normpath(file_path)

    def load_checkpoint(self, file_path):
        checkpoint = torch.load(file_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid checkpoint format")
        required_keys = ["policy_nets", "value_nets", "optimizers_policy", "optimizers_value"]
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Missing required key '{key}' in checkpoint")
        # Validate that there are three agents
        if len(checkpoint["policy_nets"]) != 3:
            raise ValueError("Checkpoint does not contain exactly three agents in 'policy_nets'")
        self.loaded_checkpoints[file_path] = checkpoint

    def show_info(self, message):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, message + "\n")
        self.info_text.see(tk.END)
        self.info_text.config(state=tk.DISABLED)

    def update_model_info(self):
        self.show_info(f"Total Models Loaded: {len(self.loaded_checkpoints)}")

    def refresh_agent_dropdowns(self):
        # Clear the existing agent_display_map
        self.agent_display_map.clear()

        # Collect all agents from all loaded checkpoints
        agents = []
        for file_path, checkpoint in self.loaded_checkpoints.items():
            base_dir = os.path.basename(os.path.dirname(file_path))
            base_name = os.path.basename(file_path)
            for agent_name in checkpoint["policy_nets"].keys():
                # Create a display name with directory prefix
                display_name = f"{base_dir}_{agent_name}"
                agents.append(display_name)
                # Map display_name to (file_path, agent_name)
                self.agent_display_map[display_name] = (file_path, agent_name)

        # Update each drop-down with the list of agents
        for player in ["player_0", "player_1", "player_2"]:
            combobox = getattr(self, f"combobox_{player}")
            combobox['values'] = agents
            # Reset selection if previously selected agent is no longer available
            current_selection = self.agent_vars[player].get()
            if current_selection not in agents:
                self.agent_vars[player].set('')

    def mix_checkpoints(self):
        selected_agents = self.get_selected_agents()
        if not selected_agents:
            messagebox.showwarning("No Selection", "Please select one agent for each player slot to mix.")
            return

        # Validate that all three players have selections
        if len(selected_agents) != 3:
            messagebox.showwarning("Incomplete Selection", "Please select one agent for each player slot.")
            return

        try:
            combined = self.combine_checkpoints(selected_agents)
            self.combined_checkpoint = combined
            self.show_info("Checkpoints successfully mixed.")
            messagebox.showinfo("Success", "Checkpoints successfully mixed.")
        except Exception as e:
            logger.error(f"Mixing failed: {e}")
            self.show_info(f"Error during mixing: {e}")
            messagebox.showerror("Mixing Failed", f"An error occurred while mixing checkpoints:\n{e}")

    def get_selected_agents(self):
        """
        Retrieves the selected agents from the drop-down menus.

        Returns:
            list of tuples: Each tuple contains (file_path, agent_name)
        """
        selected_agents = []
        for player in ["player_0", "player_1", "player_2"]:
            selection = self.agent_vars[player].get()
            if not selection:
                continue
            # Retrieve the corresponding file_path and agent_name from the mapping
            if selection in self.agent_display_map:
                file_path, agent_name = self.agent_display_map[selection]
                selected_agents.append((file_path, agent_name))
            else:
                logger.error(f"Selection '{selection}' not found in agent_display_map.")
                self.show_info(f"Error: Selection '{selection}' not found.")
        return selected_agents

    def combine_checkpoints(self, selected_agents):
        """
        Combines selected agents from different checkpoints into a single checkpoint.

        Args:
            selected_agents (list): List of tuples (file_path, agent_name)

        Returns:
            dict: Combined checkpoint
        """
        combined = {
            "policy_nets": {},
            "value_nets": {},
            "optimizers_policy": {},
            "optimizers_value": {},
            "obp_model": None,
            "obp_optimizer": None,
            "episode": None
        }

        # Assign selected agents to player_0, player_1, player_2
        for idx, (file_path, agent_name) in enumerate(selected_agents):
            target_player = f"player_{idx}"  # player_0, player_1, player_2
            checkpoint = self.loaded_checkpoints[file_path]

            # Map agent to combined checkpoint
            combined["policy_nets"][target_player] = checkpoint["policy_nets"][agent_name]
            combined["value_nets"][target_player] = checkpoint["value_nets"][agent_name]
            combined["optimizers_policy"][target_player] = checkpoint["optimizers_policy"][agent_name]
            combined["optimizers_value"][target_player] = checkpoint["optimizers_value"][agent_name]

            # If the first agent is from player_0, copy global settings
            if idx == 0 and agent_name == "player_0":
                combined["obp_model"] = checkpoint.get("obp_model", None)
                combined["obp_optimizer"] = checkpoint.get("obp_optimizer", None)
                combined["episode"] = checkpoint.get("episode", None)

        return combined

    def browse_save_location(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch Checkpoint", "*.pth")],
            initialdir=config.CHECKPOINT_DIR,
            title="Select Save Location"
        )
        if file_path:
            self.save_path_var.set(file_path)

    def save_combined_checkpoint(self):
        if self.combined_checkpoint is None:
            messagebox.showwarning("No Combined Checkpoint", "Please mix checkpoints before saving.")
            return

        save_path = self.save_path_var.get()
        if not save_path:
            messagebox.showwarning("No Save Path", "Please specify a save location.")
            return

        # Extract the filename and check if it's 'combined_checkpoint.pth'
        filename = os.path.basename(save_path)
        dir_path = os.path.dirname(save_path)

        if filename == "combined_checkpoint.pth":
            # Create a prefix from source directory names
            if not hasattr(self, 'source_dirs') or not self.source_dirs:
                messagebox.showerror("Missing Source Directories", "Source directory names are missing.")
                return
            prefix = "_".join(self.source_dirs)
            new_filename = f"{prefix}_combined_checkpoint.pth"
            new_save_path = os.path.join(dir_path, new_filename)
            # Update the save path
            save_path = new_save_path
            self.save_path_var.set(new_save_path)
            self.show_info(f"Filename changed to {new_filename} to include source directory names.")

        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.combined_checkpoint, save_path)
            self.show_info(f"Combined checkpoint saved to {save_path}")
            messagebox.showinfo("Success", f"Combined checkpoint saved to:\n{save_path}")
        except Exception as e:
            logger.error(f"Failed to save combined checkpoint: {e}")
            self.show_info(f"Error saving checkpoint: {e}")
            messagebox.showerror("Save Failed", f"An error occurred while saving the checkpoint:\n{e}")


if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = MixGUI(root)
    root.mainloop()
