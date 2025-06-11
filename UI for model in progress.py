import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import jcamp  #if it doesn't work: put pip install jcamp in command prompt

def open_jcamp_file():
    path = filedialog.askopenfilename(
        title="Open JCAMP File", filetypes=[("JCAMP files", "*.jdx")]
    )
    if path:
        display_answer_data(path)

def display_answer_data(path):
    try:
        data = jcamp.jcamp_reader(path)
        x = np.array(data.get("x", []))
        y = np.array(data.get("y", []))
        if x.size == 0 or y.size == 0:
            raise ValueError("No spectral data found in file.")

        # insert model part here
        
        presence="AI_Output"[0]
        phenol_score,aldehyde_score,benzene_score="AI_output"[1] 

        scores = [round(val, 2) for val in (phenol_score, aldehyde_score, benzene_score)]
        groups = ["Phenol", "Aldehyde", "Benzene Ring"]

        tree.delete(*tree.get_children())
        tree["columns"] = ("Functional Group","Present", "Confidence (%)")
        for col in tree["columns"]:
            tree.heading(col, text=col)
            tree.column(col, width=150)

        for g, sc in zip(groups,presence, scores):
            tree.insert("", "end", values=(g, sc))

        status_label.config(text=f"JCAMP file loaded: {path}")

    except Exception as e:
        status_label.config(text=f"Error: {str(e)}")

root = tk.Tk()
root.title("JCAMP File Viewer")

open_button = tk.Button(root, text="Open JCAMP File", command=open_jcamp_file)
open_button.pack(padx=20, pady=10)

tree = ttk.Treeview(root, show="headings")
tree.pack(padx=20, pady=20, fill="both", expand=True)

status_label = tk.Label(root, text="", padx=20, pady=10)
status_label.pack()

root.mainloop()
