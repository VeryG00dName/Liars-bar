# src/tests/data_cleaner_gui.py
import os
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
from collections import Counter

class TrainingDataCleanerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Training Data Cleaner")
        self.root.geometry("500x400")
        # Path to your training data pickle file.
        self.data_path = os.path.join(os.getcwd(), "opponent_training_data.pkl")
        self.load_data()
        self.create_widgets()
        self.populate_listbox()

    def load_data(self):
        if not os.path.exists(self.data_path):
            messagebox.showerror("Error", f"Data file not found at: {self.data_path}")
            self.training_data = []
        else:
            with open(self.data_path, "rb") as f:
                self.training_data = pickle.load(f)
        self.update_counts()

    def update_counts(self):
        # Build a dictionary mapping each label to its count.
        self.label_counts = Counter(label for _, label in self.training_data)

    def create_widgets(self):
        # Frame for the listbox and scrollbar.
        frame = ttk.Frame(self.root)
        frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.listbox = tk.Listbox(frame, font=("TkDefaultFont", 11))
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)
        
        # Buttons frame.
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        self.delete_button = ttk.Button(btn_frame, text="Delete Selected Section", command=self.delete_selected)
        self.delete_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(btn_frame, text="Save Cleaned Data", command=self.save_data)
        self.save_button.pack(side=tk.LEFT, padx=5)

    def populate_listbox(self):
        # Clear current listbox entries.
        self.listbox.delete(0, tk.END)
        # Insert each label and its count.
        for label, count in sorted(self.label_counts.items()):
            self.listbox.insert(tk.END, f"{label}: {count} samples")

    def delete_selected(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a section to delete.")
            return

        index = selection[0]
        item_text = self.listbox.get(index)
        # Assume label is the part before the colon.
        label_to_delete = item_text.split(":")[0]
        
        # Confirm deletion.
        confirm = messagebox.askyesno("Confirm Deletion",
                                      f"Are you sure you want to delete all samples for '{label_to_delete}'?")
        if confirm:
            # Remove all samples with the selected label.
            self.training_data = [sample for sample in self.training_data if sample[1] != label_to_delete]
            self.update_counts()
            self.populate_listbox()
            messagebox.showinfo("Deleted", f"Section '{label_to_delete}' deleted.")

    def save_data(self):
        with open(self.data_path, "wb") as f:
            pickle.dump(self.training_data, f)
        messagebox.showinfo("Saved", f"Cleaned data saved to {self.data_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingDataCleanerGUI(root)
    root.mainloop()
