import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import re
from collections import defaultdict

def parse_multiversion_text(raw_text):
    blocks = re.split(r'\n\s*version\s+', raw_text, flags=re.IGNORECASE)
    data = []

    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue

        version = lines[0].strip().lower()
        i = 0
        # Skip header if detected
        while i < len(lines) and not ('+' in lines[i] and '%' not in lines[i]):
            i += 1

        # Parse match data in blocks of 6 lines
        while i + 5 < len(lines):
            try:
                matchup = lines[i].strip()
                ai_wins = int(lines[i+1].strip())
                opp_wins = int(lines[i+2].strip())
                ai_rate = float(lines[i+3].strip().replace('%', ''))
                # opp_rate = float(lines[i+4].strip().replace('%', ''))
                # result = lines[i+5].strip()

                norm_key = '+'.join(sorted(matchup.split('+')))

                data.append({
                    'Original Matchup': matchup,
                    'Normalized Matchup': norm_key,
                    'Version': version,
                    'AI Win Rate': ai_rate
                })
                i += 6
            except:
                i += 1
    return pd.DataFrame(data)

def process_and_export():
    raw_input = text_box.get("1.0", tk.END)
    df = parse_multiversion_text(raw_input)

    if df.empty:
        messagebox.showerror("Error", "No valid data found.")
        return

    best_versions = []
    grouped = df.groupby("Normalized Matchup")

    for key, group in grouped:
        best = group.sort_values(by="AI Win Rate", ascending=False).iloc[0]
        best_versions.append({
            "Matchup": key,
            "Better Version": best["Version"]
        })

    out_df = pd.DataFrame(best_versions)
    out_df.to_csv("better_version_comparison.csv", index=False)
    messagebox.showinfo("Success", "Saved as better_version_comparison.csv")

# GUI Setup
root = tk.Tk()
root.title("AI Version Matchup Comparator")
root.geometry("900x700")

tk.Label(root, text="Paste Match Results Below (with version blocks):").pack()
text_box = tk.Text(root, height=35, width=120)
text_box.pack()

tk.Button(root, text="Process and Export CSV", command=process_and_export).pack(pady=10)

root.mainloop()
