import tkinter as tk
import tkinter.filedialog
import os
import sys
from tkinter import ttk
from src.factory_game import FactoryGame

LAYOUTS_DIR = "data/layouts"
SCENARIOS_DIR = os.path.join(LAYOUTS_DIR, "scenarios")
MODELS_DIR = "data/models"

def start_game(root, mode, layout_file=None, nn_mode="new"):
    root.withdraw()  # Hide launcher
    game_root = tk.Tk()
    game = FactoryGame(game_root, mode, layout_file, launcher_root=root, nn_mode=nn_mode)  # Pass nn_mode
    if nn_mode.startswith("load:"):
        game.nn_agent.model = tf.keras.models.load_model(nn_mode[5:])
        print(f"Loaded NN model from {nn_mode[5:]}")
    game_root.protocol("WM_DELETE_WINDOW", lambda: on_close(game_root, root))
    game_root.mainloop()

def on_close(game_root, launcher_root):
    game_root.destroy()  # Close game window
    launcher_root.deiconify()  # Show launcher
    launcher_root.update_idletasks()  # Process UI events
    launcher_root.lift()  # Bring launcher to front
    print("Launcher reopened")  # Log when launcher actually shows

def load_layout(root):
    layout_file = tkinter.filedialog.askopenfilename(initialdir=os.path.join(LAYOUTS_DIR, "saved"), filetypes=[("JSON files", "*.json")])
    if layout_file:
        start_game(root, "load", layout_file)

def choose_scenario(root):
    scenario_window = tk.Toplevel(root)
    scenario_window.title("Choose Scenario")
    scenario_window.geometry("300x200")

    tk.Label(scenario_window, text="Select a Scenario:", font=("Arial", 12)).pack(pady=10)

    scenarios = [d for d in os.listdir(SCENARIOS_DIR) if os.path.isdir(os.path.join(SCENARIOS_DIR, d))]
    if not scenarios:
        tk.Label(scenario_window, text=f"No scenario folders found in {SCENARIOS_DIR}!", font=("Arial", 10)).pack(pady=10)
        tk.Button(scenario_window, text="Close", command=scenario_window.destroy).pack(pady=5)
        return

    scenario_var = tk.StringVar(value=scenarios[0])
    for scenario in scenarios:
        tk.Radiobutton(scenario_window, text=scenario, variable=scenario_var, value=scenario).pack(anchor=tk.W, padx=10)

    def choose_nn():
        scenario_window.destroy()
        nn_window = tk.Toplevel(root)
        nn_window.title("NN Options")
        nn_window.geometry("300x300")
        
        tk.Label(nn_window, text="Use Neural Network?", font=("Arial", 12)).pack(pady=10)
        
        nn_var = tk.StringVar(value="new")
        tk.Radiobutton(nn_window, text="No NN", variable=nn_var, value="none").pack(anchor=tk.W, padx=10)
        tk.Radiobutton(nn_window, text="New NN", variable=nn_var, value="new").pack(anchor=tk.W, padx=10)
        tk.Radiobutton(nn_window, text="Load Existing NN", variable=nn_var, value="load").pack(anchor=tk.W, padx=10)
        
        model_var = tk.StringVar()
        model_dropdown = ttk.Combobox(nn_window, textvariable=model_var, state="disabled", width=40)  # Doubled width from ~20 to 40
        model_dropdown.pack(pady=5, padx=10)
        
        def update_dropdown(*args):
            if nn_var.get() == "load":
                os.makedirs(MODELS_DIR, exist_ok=True)  # Ensure directory exists
                models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".h5")]
                model_dropdown["values"] = models if models else ["No models found"]
                model_dropdown.state(["!disabled"])
                model_dropdown.set(models[0] if models else "No models found")
            else:
                model_dropdown.state(["disabled"])
                model_dropdown.set("")
        
        nn_var.trace("w", update_dropdown)
        
        def start_game_with_nn():
            nn_mode = nn_var.get()
            if nn_mode == "load" and model_var.get() and model_var.get() != "No models found":
                nn_mode = f"load:{os.path.join(MODELS_DIR, model_var.get())}"
            layout_file = os.path.join(SCENARIOS_DIR, scenario_var.get(), "scenario.json")
            nn_window.destroy()
            start_game(root, "new", layout_file, nn_mode)
        
        tk.Button(nn_window, text="Start", command=start_game_with_nn).pack(pady=10)
        tk.Button(nn_window, text="Cancel", command=nn_window.destroy).pack(pady=5)

    tk.Button(scenario_window, text="Next", command=choose_nn).pack(pady=10)
    tk.Button(scenario_window, text="Cancel", command=scenario_window.destroy).pack(pady=5)

def quit_launcher(root):
    root.destroy()
    sys.exit()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Future Factory Launcher")
    root.geometry("300x300")

    tk.Label(root, text="Future Factory", font=("Arial", 16, "bold")).pack(pady=20)
    tk.Button(root, text="Choose Scenario", command=lambda: choose_scenario(root), width=20).pack(pady=5)
    tk.Button(root, text="Continue Game", command=lambda: start_game(root, "continue"), width=20).pack(pady=5)
    tk.Button(root, text="Load Saved Layout", command=lambda: load_layout(root), width=20).pack(pady=5)
    tk.Button(root, text="Quit", command=lambda: quit_launcher(root), width=20).pack(pady=5)

    root.mainloop()