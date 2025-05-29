import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import tkinter.simpledialog
import tkinter.messagebox
import os
import json
import sys
import random
import csv
import numpy as np
from src.nn_agent import NNAgent
from src.DataLogger import DataLogger
from src.ocsvm_predictor import OCSVMPredictor

 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MASTER_DIR = os.path.join(DATA_DIR, "master")
LAYOUTS_DIR = os.path.join(DATA_DIR, "layouts")
SAVED_DIR = os.path.join(LAYOUTS_DIR, "saved")
LOGS_DIR = os.path.join(DATA_DIR, "logs", "game_logs")

class Material:
    def __init__(self, id, state, parent_ids=None, quality=1.0, fault_detected=False):
        self.id = id
        self.state = state
        self.location = None
        self.parent_ids = parent_ids or []
        self.quality = quality
        self.fault_detected = fault_detected

class FactoryGame:
    def __init__(self, root, mode="new", layout_file=None, launcher_root=None, nn_mode="new"):
        # --- Initialization Basics ---
        self.root = root
        self.launcher_root = launcher_root
        self.root.title("Future Factory")
        self.game_time = 0
        self.auto_running = False
        self.GRID_SIZE = 30
        self.CANVAS_WIDTH = 900
        self.CANVAS_HEIGHT = 750
        self.GRID_OFFSET = 2

        # NN Mode Setup
        self.use_nn = nn_mode != "none"
        self.train_nn = self.use_nn
        print(f"NN Mode: {'Enabled' if self.use_nn else 'Disabled'}, Training: {'Enabled' if self.train_nn else 'Disabled'}")

        # Logging control
        self.logging_enabled = True
        def log(self, message):
            if self.logging_enabled:
                print(message)
        self.log = log.__get__(self, self.__class__)

        # --- Directory Setup ---
        os.makedirs(MASTER_DIR, exist_ok=True)
        os.makedirs(LAYOUTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(LAYOUTS_DIR, "scenarios"), exist_ok=True)
        os.makedirs(SAVED_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)

        # --- Mode and Scenario Handling ---
        if mode == "continue":
            layout_file = os.path.join(LAYOUTS_DIR, "current.json")

        if mode == "new" and layout_file:
            scenario_dir = os.path.dirname(layout_file)
            self.scenario = os.path.basename(scenario_dir)
        elif mode in ("continue", "load") and layout_file:
            with open(layout_file, "r") as f:
                layout_temp = json.load(f)
            scenario = layout_temp.get("scenario", "car_factory")
            scenario_dir = os.path.join(LAYOUTS_DIR, "scenarios", scenario)
            self.scenario = scenario
        else:
            scenario_dir = os.path.join(LAYOUTS_DIR, "scenarios", "car_factory")
            self.scenario = "car_factory"

        tools_file = os.path.join(scenario_dir, "tools.json")
        scenario_file = os.path.join(scenario_dir, "scenario.json")

        # --- File Loading ---
        try:
            with open(layout_file, "r") as f:
                layout = json.load(f)
            with open(tools_file, "r") as f:
                tools_data = json.load(f)
            with open(scenario_file, "r") as f:
                scenario_data = json.load(f)
            self.max_turns = scenario_data.get("max_turns", float('inf'))
            self.material_dict = tools_data.get("materials", {})
            self.tool_dict = tools_data["tools"]
            print(f"Loaded materials and tools from: {tools_file}")
        except FileNotFoundError as e:
            print(f"File not found: {e} - game cannot proceed")
            raise

        # --- Delivery Setup ---
        self.delivery_interval = 10
        self.delivery_template = [
            {"material": "COMP_STEEL", "count": 5, "inv_id": "INV_STEEL"},
            {"material": "RAW_COAL", "count": 5, "inv_id": "INV_COAL"},
            {"material": "RAW_COPPER", "count": 4, "inv_id": "INV_COPPER"},
            {"material": "RAW_GLASS", "count": 3, "inv_id": "INV_GLASS"},
            {"material": "RAW_RUBBER", "count": 10, "inv_id": "INV_RUBBER"}
        ]

        # --- Game State Setup ---
        try:
            seen = set()
            if mode in ("continue", "load"):
                self.machines = [(m["id"], m["x"], m["y"]) for m in layout.get("machines", []) if not (m["id"] in seen or seen.add(m["id"]))]
                self.inventories = [(i["id"], i["x"], i["y"]) for i in layout.get("inventories", [])]
                self.arrows = layout.get("arrows", {})
                self.machine_tools = layout.get("machine_tools", {})  # Loaded as dicts from save
                self.inventory_accepts = layout.get("inventory_accepts", {})
                self.machine_states = layout.get("machine_states", {m[0]: 1 for m in self.machines})
                self.machine_cycles = layout.get("machine_cycles", {m[0]: 0 for m in self.machines})
                self.game_time = layout.get("game_time", 0)
                self.turns = layout.get("turns", 0)
                self.money = layout.get("money", 5000)
                self.orders = layout.get("orders", [])
                self.auto_running = False
                # Lean Tracking: INV metrics only (WS moved to DataLogger)
                self.inv_turnover_history = {i[0]: [] for i in self.inventories}
                self.inv_availability_history = {i[0]: [] for i in self.inventories}
                # Maintenance and breakdown tracking
                self.breakdown_risk = layout.get("breakdown_risk", {m[0]: 0 for m in self.machines})
                self.alert_active = layout.get("alert_active", {m[0]: False for m in self.machines})
                self.turns_since_alert = layout.get("turns_since_alert", {m[0]: 0 for m in self.machines})
                self.maintenance_turns = layout.get("maintenance_turns", {m[0]: 0 for m in self.machines})
                self.is_breakdown = layout.get("is_breakdown", {m[0]: False for m in self.machines})
                self.is_maintenance = layout.get("is_maintenance", {m[0]: False for m in self.machines})
                self.is_repair = layout.get("is_repair", {m[0]: False for m in self.machines})
                self.is_paused = layout.get("is_paused", {m[0]: False for m in self.machines})
                self.maintenance_queued = layout.get("maintenance_queued", {m[0]: False for m in self.machines})
                self.last_vibration = {}
                self.spike_duration = {}
                self.alert_duration = {m[0]: 0 for m in self.machines}  # Track high-vibe duration for breakdowns
                self.prev_states = {m[0]: 1 for m in self.machines}
            else:
                self.machines = [(m["id"], m["x"], m["y"]) for m in layout.get("initial_machines", []) if not (m["id"] in seen or seen.add(m["id"]))]
                self.inventories = [(i["id"], i["x"], i["y"]) for i in layout["initial_inventories"]]
                self.arrows = layout.get("initial_arrows", {})
                # Map tool names to full tool dicts from tools.json
                initial_tools = layout.get("initial_machine_tools", {})
                self.machine_tools = {ws_id: self.tool_dict[tool_name] for ws_id, tool_name in initial_tools.items() if tool_name in self.tool_dict}
                self.inventory_accepts = layout.get("initial_inventory_accepts", {})
                self.machine_states = layout.get("initial_machine_states", {m[0]: 1 for m in self.machines})
                self.machine_cycles = layout.get("initial_machine_cycles", {m[0]: 0 for m in self.machines})
                self.turns = 0
                self.money = 5000
                self.orders = layout.get("orders", [])
                self.deliveries = []
                # Lean Tracking: INV metrics only (WS moved to DataLogger)
                self.inv_turnover_history = {i[0]: [] for i in self.inventories}
                self.inv_availability_history = {i[0]: [] for i in self.inventories}
                # Maintenance and breakdown tracking
                self.breakdown_risk = {m[0]: 0 for m in self.machines}
                self.alert_active = {m[0]: False for m in self.machines}
                self.turns_since_alert = {m[0]: 0 for m in self.machines}
                self.maintenance_turns = {m[0]: 0 for m in self.machines}
                self.is_breakdown = {m[0]: False for m in self.machines}
                self.is_maintenance = {m[0]: False for m in self.machines}
                self.is_repair = {m[0]: False for m in self.machines}
                self.is_paused = {m[0]: False for m in self.machines}
                self.maintenance_queued = {m[0]: False for m in self.machines}
                self.fixed_costs = 100
                self.last_vibration = {}
                self.spike_duration = {}
                self.alert_duration = {m[0]: 0 for m in self.machines}  # Track high-vibe duration for breakdowns
                self.prev_states = {m[0]: 1 for m in self.machines}

            # --- Inventory Caps Initialization ---
            self.inventory_caps = {inv_id: None for inv_id, _, _ in self.inventories}  # Default uncapped
            if mode == "new" and "initial_inventory_caps" in layout:
                for inv_id, cap in layout["initial_inventory_caps"].items():
                    if inv_id in self.inventory_caps:
                        self.inventory_caps[inv_id] = int(cap) if cap is not None else None  # Scenario override
            elif mode in ("continue", "load") and "inventory_caps" in layout:
                self.inventory_caps = {inv_id: int(cap) if cap is not None else None for inv_id, cap in layout["inventory_caps"].items()}

            # --- DataLogger Initialization ---
            self.data_logger = DataLogger(window_size=100)
            for mach_id, _, _ in self.machines:
                self.data_logger._initialize_object(mach_id, "WS")
            for inv_id, _, _ in self.inventories:
                self.data_logger._initialize_object(inv_id, "INV")

            all_materials = list(self.material_dict.keys())
            for inv_id, _, _ in self.inventories:
                if inv_id not in self.inventory_accepts:
                    self.inventory_accepts[inv_id] = all_materials

            valid_ids = {m[0]: True for m in self.machines} | {i[0]: True for i in self.inventories}
            self.arrows = {k: v for k, v in self.arrows.items() if v["start_id"] in valid_ids and v["end_id"] in valid_ids}

            # Pre-index arrows for faster lookups and cache sorting
            self.downstream_arrows = {}
            self.upstream_arrows = {}
            for arrow_id, arrow in self.arrows.items():
                start_id = arrow["start_id"]
                end_id = arrow["end_id"]
                self.downstream_arrows.setdefault(start_id, []).append(arrow)
                self.upstream_arrows.setdefault(end_id, []).append(arrow)
            for start_id in self.downstream_arrows:
                self.downstream_arrows[start_id].sort(key=lambda x: x.get("priority", 0.5), reverse=True)
            for end_id in self.upstream_arrows:
                self.upstream_arrows[end_id].sort(key=lambda x: x.get("priority", 0.5), reverse=True)

            # --- Neural Network Agent Setup ---
            num_machines = len(self.machines)
            input_size = (num_machines * 11) + 3
            action_size = num_machines + 1
            self.nn_agent = NNAgent(input_size=input_size, action_size=action_size)

            # --- OC-SVM Predictor Setup ---
            self.ocsvm_predictor = OCSVMPredictor(nu=0.05)  # 5% outliers
            self.training_turn = 32  # Train after 32 turns, aligned with NN batch size
            self.use_ocsvm = True  # Toggle OC-SVM usage

            # Hover state
            self.last_hover_shape = None

            print(f"Loaded from {layout_file}: {len(self.machines)} machines, {len(self.inventories)} inventories, {len(self.arrows)} arrows, {len(self.machine_tools)} tools")
            print(f"Order queue initialized with {len(self.orders)} orders")
            print(f"Delivery schedule initialized with dynamic template every {self.delivery_interval} turns")
            print(f"Game state: Turns={self.turns}, Money=${self.money}")
            self.log(f"Pre-indexed {len(self.arrows)} arrows: {len(self.downstream_arrows)} downstream, {len(self.upstream_arrows)} upstream")

        except KeyError as e:
            print(f"Missing required field in {layout_file}: {e} - game cannot proceed")
            raise
        except Exception as e:
            print(f"Error initializing game state: {e}")
            raise

        # --- Material Loading ---
        self.materials_at_machine = {m[0]: [] for m in self.machines}
        self.materials = {}
        if mode in ("continue", "load") and "materials_at_machine" in layout:
            for m_id, mat_list in layout["materials_at_machine"].items():
                self.materials_at_machine[m_id] = []
                for mat_data in mat_list:
                    mat = Material(mat_data["id"], mat_data["state"], 
                                  parent_ids=mat_data.get("parent_ids", []),
                                  quality=mat_data.get("quality", 1.0),
                                  fault_detected=mat_data.get("fault_detected", False))
                    mat.location = m_id
                    self.materials_at_machine[m_id].append(mat)
                    self.materials[mat.id] = mat

        self.last_deleted = None
        self.dragging = None
        self.flow_mode = False
        self.flow_start = None
        self.flow_preview = None
        self.hover_shape = None
        self.hover_arrow = None
        self.hover_dot = None
        self.is_new_item = False

        initial_stocks = layout.get("initial_stocks", {}) if mode == "new" else {}
        self.inventory_stocks = {}
        for inv_id, x, y in self.inventories:
            if mode == "new" and inv_id in initial_stocks:
                self.inventory_stocks[inv_id] = {}
                for mat_type, count in initial_stocks[inv_id].items():
                    mats = [Material(f"{mat_type.split('_')[-1]}{i}", mat_type) for i in range(1, count + 1)]
                    for mat in mats:
                        mat.location = inv_id
                        self.materials[mat.id] = mat
                    self.inventory_stocks[inv_id][mat_type] = mats
            else:
                self.inventory_stocks[inv_id] = {}

        if mode in ("continue", "load") and "inventory_stocks" in layout:
            saved_stocks = layout["inventory_stocks"]
            for inv_id in self.inventories:
                inv_id = inv_id[0]
                if inv_id in saved_stocks:
                    self.inventory_stocks[inv_id] = {}
                    for mat_type, mat_list in saved_stocks[inv_id].items():
                        mats = [Material(mat_data["id"], mat_type, 
                                        quality=mat_data.get("quality", 1.0), 
                                        fault_detected=mat_data.get("fault_detected", False)) 
                                for mat_data in mat_list]
                        for mat in mats:
                            mat.location = inv_id
                            self.materials[mat.id] = mat
                        self.inventory_stocks[inv_id][mat_type] = mats

        # --- ID Counters ---
        self.next_machine_id = max([int(m[0][2:]) for m in self.machines], default=0) + 1 if self.machines else 1
        self.next_inv_id = max([int(i[0][3:]) if i[0].startswith("INV") and i[0][3:].isdigit() else 0 for i in self.inventories], default=0) + 1
        self.next_flow_id = max([int(k[4:]) for k in self.arrows.keys()], default=0) + 1 if self.arrows else 1
        self.next_material_id = max([int(k[3:]) for k in self.materials.keys() if k.startswith("MAT")], default=100) + 1

        # Add visual_positions for dragging
        self.visual_positions = {}
        self.pending_new = None

        # --- Canvas Setup ---
        self.canvas = tk.Canvas(root, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT, bg="white")
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        self.selected_item = None
        
        # --- Info Frame Setup ---
        self.info_frame = tk.Frame(root, width=300, height=self.CANVAS_HEIGHT, bg="lightgrey")
        self.info_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self.info_frame.pack_propagate(False)
        tk.Label(self.info_frame, text="Info Board", font=("Arial", 12, "bold"), bg="lightgrey").pack(pady=5)
        self.info_object_id = tk.Label(self.info_frame, text="Object ID: -", bg="lightgrey", wraplength=280)
        self.info_object_id.pack(pady=2)
        self.info_name = tk.Label(self.info_frame, text="Tool: -", bg="lightgrey", wraplength=280)
        self.info_name.pack(pady=2)
        self.info_inputs = tk.Label(self.info_frame, text="Inputs: -", bg="lightgrey", wraplength=280)
        self.info_inputs.pack(pady=2)
        self.info_outputs = tk.Label(self.info_frame, text="Outputs: -", bg="lightgrey", wraplength=280)
        self.info_outputs.pack(pady=2)
        self.info_cycle = tk.Label(self.info_frame, text="Cycle Time: -", bg="lightgrey")
        self.info_cycle.pack(pady=2)
        self.info_costs = tk.Label(self.info_frame, text="Running Costs: -", bg="lightgrey", wraplength=280)
        self.info_costs.pack(pady=2)
        self.info_availability = tk.Label(self.info_frame, text="Availability: -", bg="lightgrey")
        self.info_availability.pack(pady=2)
        self.info_fpy = tk.Label(self.info_frame, text="FPY: -", bg="lightgrey")
        self.info_fpy.pack(pady=2)
        self.info_throughput = tk.Label(self.info_frame, text="Throughput Rate: -", bg="lightgrey")
        self.info_throughput.pack(pady=2)
        self.info_cte = tk.Label(self.info_frame, text="CTE: -", bg="lightgrey")
        self.info_cte.pack(pady=2)
        self.info_oee = tk.Label(self.info_frame, text="OEE: -", bg="lightgrey")
        self.info_oee.pack(pady=2)
        self.info_wear = tk.Label(self.info_frame, text="Wear: -", bg="lightgrey")
        self.info_wear.pack(pady=2)
        self.info_fault_chance = tk.Label(self.info_frame, text="Fault Chance: -", bg="lightgrey")
        self.info_fault_chance.pack(pady=2)
        self.info_held_materials = tk.Label(self.info_frame, text="Held Materials: -", bg="lightgrey", wraplength=280)
        self.info_held_materials.pack(pady=2)
        
        # --- UI Frame Setup ---
        ui_frame = tk.Frame(root, width=150)
        ui_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        ui_frame.pack_propagate(False)

        self.status_label = tk.Label(ui_frame, text=f"Turns: {self.turns}", font=("Arial", 12))
        self.status_label.pack(pady=5)
        self.money_label = tk.Label(ui_frame, text=f"Money: ${self.money}", font=("Arial", 12))
        self.money_label.pack(pady=5)

        tk.Label(ui_frame, text="Actions:", font=("Arial", 10, "bold")).pack(pady=5)
        tk.Button(ui_frame, text="Buy Working Station ($100)", command=self.buy_machine, width=20, height=1).pack(pady=2)
        tk.Button(ui_frame, text="Buy Inventory Rack ($100)", command=self.buy_inventory, width=20, height=1).pack(pady=2)
        tk.Button(ui_frame, text="Undo Last Delete", command=self.undo_delete, width=20, height=1).pack(pady=2)
        tk.Button(ui_frame, text="Run Turn", command=self.run_turn, width=20, height=1).pack(pady=2)
        self.toggle_auto_button = tk.Button(ui_frame, text="Start Auto-Turn", command=self.toggle_auto_turn, width=20, height=1)
        self.toggle_auto_button.pack(pady=2)
        tk.Button(ui_frame, text="View Orders", command=self.view_orders, width=20, height=1).pack(pady=2)
        tk.Button(ui_frame, text="Analytics", command=self.generate_analytics_report, width=20, height=1).pack(pady=2)
        self.toggle_nn_button = tk.Button(ui_frame, text="Enable NN" if not self.use_nn else "Disable NN", command=self.toggle_nn, width=20, height=1)
        self.toggle_nn_button.pack(pady=2)
        self.train_button = tk.Button(ui_frame, text="Enable Training" if not self.train_nn else "Disable Training", command=self.toggle_training, width=20, height=1)
        self.train_button.pack(pady=2)
        self.toggle_log_button = tk.Button(ui_frame, text="Disable Logging" if self.logging_enabled else "Enable Logging", command=self.toggle_logging, width=20, height=1)
        self.toggle_log_button.pack(pady=2)
        self.update_train_button_state()

        tk.Frame(ui_frame, height=100).pack(fill=tk.X, expand=True)

        tk.Label(ui_frame, text="Options:", font=("Arial", 10, "bold")).pack(pady=5)
        tk.Button(ui_frame, text="Save Layout", command=self.save_named_layout, width=20, height=1).pack(pady=2)
        tk.Button(ui_frame, text="Show Instructions", command=self.show_instructions, width=20, height=1).pack(pady=2)
        tk.Button(ui_frame, text="Back to Launcher", command=self.back_to_launcher, width=20, height=1).pack(pady=2)
        tk.Button(ui_frame, text="Quit", command=self.quit_game, width=20, height=1).pack(pady=2)

        # --- Canvas Bindings and Final Setup ---
        self.canvas.bind("<Button-1>", self.handle_left_click)
        self.canvas.bind("<B1-Motion>", self.move_shape)
        self.canvas.bind("<ButtonRelease-1>", self.handle_release)
        self.canvas.bind("<Motion>", self.hover_check)
        self.canvas.bind("<Button-3>", self.handle_right_click)
        self.canvas.bind("<Double-1>", self.handle_double_click)

        self.draw_grid()
        self.draw_layout()

        self.canvas.bind("<Button-3>", self.handle_right_click)
        print("Right-click binding set for canvas")
        
    def run_turn(self):
        # --- Turn Initialization ---
        self.turns += 1
        self.game_time += 1
        self.remaining_machines = set(m[0] for m in self.machines if self.machine_states.get(m[0], 1) != 4)
        self.log(f"\nTurn {self.turns} Start:")
        ws_states = {m[0]: f"S:{self.machine_states.get(m[0], 1)} ({self.machine_cycles.get(m[0], 0)}/{self.machine_tools.get(m[0], {}).get('cycle_time', 0)})" for m in self.machines}
        self.log("WS States: " + str(ws_states))
        
        # --- Export History Every 100 Turns ---
        if self.turns > 1 and self.turns % 100 == 0:  # Avoid turn 0, start at 100
            start_turn = self.turns - 100
            end_turn = self.turns - 1
            self.data_logger.export_to_csv(start_turn, end_turn)

        # --- NN Data Initialization ---
        if not hasattr(self, 'nn_data'):
            self.nn_data = {m[0]: [] for m in self.machines}
        if not hasattr(self, 'revenue_streams'):
            self.revenue_streams = []

        # --- Reward Tracking ---
        cycles_completed = 0
        breakdowns = 0

        # --- Fixed Costs ---
        self.money -= self.fixed_costs
        self.log(f"Turn {self.turns}: Fixed costs deducted: ${self.fixed_costs}")

        # --- Deliveries and Material Costs ---
        if self.turns % self.delivery_interval == 1:
            material_cost = 0
            for item in self.delivery_template:
                mat_type = item["material"]
                count = item["count"]
                inv_id = item["inv_id"]
                if inv_id in self.inventory_stocks and mat_type in self.inventory_accepts.get(inv_id, []):
                    if mat_type not in self.inventory_stocks[inv_id]:
                        self.inventory_stocks[inv_id][mat_type] = []
                    for i in range(count):
                        mat = Material(f"{mat_type.split('_')[-1]}{self.next_material_id}", mat_type)
                        self.next_material_id += 1
                        mat.location = inv_id
                        self.materials[mat.id] = mat
                        self.inventory_stocks[inv_id][mat_type].append(mat)
                        self.log(f"Turn {self.turns}: Delivered {self.material_dict.get(mat_type, {}).get('display', mat_type)} (ID: {mat.id}) to {inv_id}")
                    cost = self.material_dict.get(mat_type, {}).get("cost", 0) * count
                    material_cost += cost
                else:
                    self.log(f"Turn {self.turns}: Delivery failed - {inv_id} does not exist or does not accept {mat_type}")
            self.money -= material_cost
            self.log(f"Turn {self.turns}: Material costs deducted: ${material_cost}")

        # --- Running Costs ---
        total_cost = 0
        for mach_type, _, _ in self.machines:
            state = self.machine_states.get(mach_type, 1)
            tool = self.machine_tools.get(mach_type, {})
            costs = tool.get("costs", {"idle": 0, "processing": 0, "ready": 0})
            cost = costs["idle"] if state == 1 else costs["processing"] if state == 2 else costs["ready"] if state == 3 else 0
            total_cost += cost
            self.data_logger.log_cost(self.turns, mach_type, "WS", cost)  # Log energy consumption
            if cost > 0:
                self.log(f"Turn {self.turns}: {mach_type} running cost: ${cost} (State: S:{state})")
        self.money -= total_cost
        if total_cost > 0:
            self.log(f"Turn {self.turns}: Total WS running costs: ${total_cost}")

        # --- NN Decision ---
        state_vector = None
        action = None
        if self.use_nn:
            state_vector = self.get_current_state()
            state_vector_flat = np.concatenate([
                np.array([state_vector[mach_id][key] for key in state_vector[mach_id]])
                for mach_id in [m[0] for m in self.machines]
            ] + [
                np.array([state_vector["global"][key] for key in state_vector["global"]])
            ])
            action = self.nn_agent.choose_action(state_vector_flat)
            if action > 0:
                machine_idx = action - 1
                if machine_idx < len(self.machines):
                    machine_id = self.machines[machine_idx][0]
                    self.schedule_maintenance(machine_id)
                    self.log(f"Turn {self.turns}: NN scheduled maintenance on {machine_id}")
            else:
                self.log(f"Turn {self.turns}: NN chose to do nothing")

        # --- Process Machines (State 2) ---
        for mach_type in list(self.remaining_machines):
            state = self.machine_states.get(mach_type, 1)
            if state == 2:
                tool = self.machine_tools.get(mach_type, {})
                self.machine_cycles[mach_type] -= 1
                mats = self.materials_at_machine[mach_type]
                
                if self.machine_cycles[mach_type] <= 0:
                    if tool and "outputs" in tool:
                        parent_ids = [mat.id for mat in mats]
                        self.materials_at_machine[mach_type] = []
                        if "wear" not in tool:
                            tool["wear"] = 0
                        lifecycle = tool.get("lifecycle", 50)  # Legacy field, not used for vibration
                        output_qualities = []
                        output_count = sum(tool["outputs"].values())
                        for out_type, count in tool["outputs"].items():
                            for _ in range(count):
                                new_mat = Material(f"MAT{self.next_material_id}", out_type, parent_ids=parent_ids)
                                self.next_material_id += 1
                                new_mat.location = mach_type
                                self.materials_at_machine[mach_type].append(new_mat)
                                self.materials[new_mat.id] = new_mat
                                output_qualities.append(new_mat.quality)
                                self.log(f"Turn {self.turns}: {mach_type} completed - {self.material_dict.get(out_type, {}).get('display', out_type)} (ID: {new_mat.id})")
                        tool["wear"] += 1
                        self.breakdown_risk[mach_type] = self.breakdown_risk.get(mach_type, 0) + 1
                        self.data_logger.log_output(self.turns, mach_type, "WS", output_qualities, output_count)
                        self.machine_states[mach_type] = 3
                        cycles_completed += 1
                        self.remaining_machines.remove(mach_type)
                    else:
                        self.log(f"Turn {self.turns}: {mach_type} has no tool - skipping")
                        self.machine_states[mach_type] = 1
                        self.materials_at_machine[mach_type] = []
                        self.remaining_machines.remove(mach_type)

        # --- Breakdown Risk and Alerts (Vibration-Driven) ---
        for mach_type, _, _ in self.machines:
            mach_id = mach_type
            state = self.machine_states.get(mach_id, 1)
            if state != 4:  # Only active machines can break down
                pass  # Logic moved to vibration section

        # --- Down State (S:4) ---
        for mach_type, _, _ in self.machines:
            mach_id = mach_type
            state = self.machine_states.get(mach_id, 1)
            if state == 4:
                if self.is_maintenance.get(mach_id, False) or self.is_repair.get(mach_id, False):
                    if self.maintenance_turns[mach_id] == 40 or self.maintenance_turns[mach_id] == 5:  # Log start
                        event_type = "repair_start" if self.is_repair.get(mach_id, False) else "maintenance_start"
                        self.data_logger.log_maintenance(self.turns, mach_id, "WS", event_type, self.maintenance_turns[mach_id])
                    self.maintenance_turns[mach_id] -= 1
                    self.log(f"Turn {self.turns}: {mach_id} in {'repair' if self.is_repair.get(mach_id, False) else 'maintenance'} - {self.maintenance_turns[mach_id]} turns remaining")
                    if self.maintenance_turns[mach_id] <= 0:
                        self.machine_states[mach_id] = 1
                        reason = "repair" if self.is_repair.get(mach_id, False) else "maintenance"
                        self.data_logger.log_maintenance(self.turns, mach_id, "WS", f"{reason}_end", 0)
                        self.is_maintenance[mach_id] = False
                        self.is_repair[mach_id] = False
                        self.breakdown_risk[mach_id] = 0
                        self.alert_active[mach_id] = False
                        self.alert_duration[mach_id] = 0  # Reset breakdown counter
                        # Reset vibration tracking and OC-SVM history
                        if mach_id in self.last_vibration:
                            del self.last_vibration[mach_id]
                        if mach_id in self.spike_duration:
                            del self.spike_duration[mach_id]
                        if self.use_ocsvm:
                            self.ocsvm_predictor.reset(mach_id)
                        self.log(f"Turn {self.turns}: {mach_id} {reason} complete, back to S:1")
                        self.remaining_machines.add(mach_id)
                elif self.is_breakdown.get(mach_id, False):
                    self.log(f"Turn {self.turns}: {mach_id} in breakdown - awaiting action")
                elif self.is_paused.get(mach_id, False):
                    self.log(f"Turn {self.turns}: {mach_id} paused - awaiting unpause")

        # --- Unload Materials (State 3) ---
        inv_outputs = {inv_id: 0 for inv_id in self.inventory_stocks.keys()}
        queue_starts = getattr(self, 'queue_starts', {m[0]: None for m in self.machines})  # Track S:3 start
        for mach_type in list(self.remaining_machines):
            state = self.machine_states.get(mach_type, 1)
            if state == 3:
                if queue_starts[mach_type] is None:
                    queue_starts[mach_type] = self.turns  # Start tracking queue time
                downstreams = self.downstream_arrows.get(mach_type, [])
                downstreams.sort(key=lambda x: x.get("priority", 0.5), reverse=True)
                mats = self.materials_at_machine[mach_type][:]
                delivered_any = False
                blocked = False
                for mat in mats[:]:
                    delivered = False
                    for arrow in downstreams:
                        down_id = arrow["end_id"]
                        if down_id in self.inventory_stocks:
                            accepts = self.inventory_accepts.get(down_id, [])
                            if mat.state in accepts:
                                # Check inventory capacity
                                cap = self.inventory_caps.get(down_id)
                                current_stock = sum(len(mats_list) for mats_list in self.inventory_stocks[down_id].values())
                                if cap is not None and current_stock >= cap:
                                    self.log(f"Turn {self.turns}: {mach_type} blocked - {down_id} full (cap: {cap}, stock: {current_stock})")
                                    self.data_logger.log_inv_blockage(self.turns, down_id, mach_type)
                                    blocked = True
                                    self.remaining_machines.remove(mach_type)  # Action done—remove now
                                    break  # Skip to next machine
                                # Deliver if under cap
                                self.inventory_stocks[down_id].setdefault(mat.state, []).append(mat)
                                mat.location = down_id
                                self.materials_at_machine[mach_type].remove(mat)
                                delivered = True
                                delivered_any = True
                                inv_outputs[down_id] += 1
                                self.data_logger.log_inv_movement(self.turns, down_id, "in", mat.id, mat.state, source=mach_type)
                                self.log(f"Turn {self.turns}: {mach_type} delivered {self.material_dict.get(mat.state, {}).get('display', mat.state)} (ID: {mat.id}) to {down_id}")
                                break
                        elif down_id in self.machine_states and self.machine_states[down_id] == 1 and down_id in self.remaining_machines:
                            down_tool = self.machine_tools.get(down_id, {})
                            if mat.state in down_tool.get("inputs", {}):
                                mat.location = down_id
                                self.materials_at_machine[down_id].append(mat)
                                self.materials_at_machine[mach_type].remove(mat)
                                delivered = True
                                delivered_any = True
                                self.log(f"Turn {self.turns}: {mach_type} delivered {self.material_dict.get(mat.state, {}).get('display', mat.state)} (ID: {mat.id}) to {down_id}")
                                break
                    if not delivered and down_id not in self.inventory_stocks:
                        self.log(f"Turn {self.turns}: {mach_type} could not deliver {self.material_dict.get(mat.state, {}).get('display', mat.state)} (ID: {mat.id}) - no valid downstream")
                if blocked:
                    self.log(f"Turn {self.turns}: {mach_type} remains in S:3 - blocked")
                elif delivered_any and not self.materials_at_machine[mach_type]:
                    queue_duration = self.turns - queue_starts[mach_type]
                    self.data_logger.log_queue(self.turns, mach_type, "WS", queue_duration)
                    self.log(f"Turn {self.turns}: {mach_type} queued for {queue_duration} turns")
                    queue_starts[mach_type] = None
                    if self.maintenance_queued.get(mach_type, False):
                        if self.is_breakdown.get(mach_type, False):
                            self.machine_states[mach_type] = 4
                            self.maintenance_turns[mach_type] = 40
                            self.is_repair[mach_type] = True
                            self.is_breakdown[mach_type] = False
                            self.breakdown_risk[mach_type] = 0
                            self.alert_active[mach_type] = False
                            self.maintenance_queued[mach_type] = False
                            self.data_logger.log_maintenance(self.turns, mach_type, "WS", "repair_start", 40)
                            self.log(f"Turn {self.turns}: {mach_type} started repair (40 turns) post-breakdown")
                        elif self.is_paused.get(mach_type, False):
                            self.machine_states[mach_type] = 4
                            self.is_paused[mach_type] = True
                            self.maintenance_queued[mach_type] = False
                            self.log(f"Turn {self.turns}: {mach_type} paused by player after processing")
                        else:
                            self.machine_states[mach_type] = 4
                            self.maintenance_turns[mach_type] = 5
                            self.is_maintenance[mach_type] = True
                            self.breakdown_risk[mach_type] = 0
                            self.alert_active[mach_type] = False
                            self.maintenance_queued[mach_type] = False
                            self.data_logger.log_maintenance(self.turns, mach_type, "WS", "maintenance_start", 5)
                            self.log(f"Turn {self.turns}: {mach_type} started quick maintenance (5 turns)")
                    else:
                        self.machine_states[mach_type] = 1
                        self.log(f"Turn {self.turns}: {mach_type} emptied - Transitioned to S:1")
                    self.remaining_machines.remove(mach_type)  # Remove after success
                elif not delivered_any and mats:
                    self.log(f"Turn {self.turns}: {mach_type} remains in S:3 - blocked or no valid downstream")
                    self.remaining_machines.remove(mach_type)  # Remove if stuck but not capped
        self.queue_starts = queue_starts

        # --- Load Materials (State 1) ---
        for mach_type in list(self.remaining_machines):
            state = self.machine_states.get(mach_type, 1)
            if state == 1:
                tool = self.machine_tools.get(mach_type, {})
                if not tool or not tool.get("inputs"):
                    self.data_logger.log_shortage(self.turns, mach_type, "WS", False, [])  # Log no shortage
                    self.log(f"Turn {self.turns}: {mach_type} skipped - no tool/inputs")
                    self.remaining_machines.remove(mach_type)
                    continue
                upstreams = self.upstream_arrows.get(mach_type, [])
                upstreams.sort(key=lambda x: x.get("priority", 0.5), reverse=True)
                required = tool["inputs"]
                current_mats = {mat.state: sum(1 for m in self.materials_at_machine[mach_type] if m.state == mat.state) for mat in self.materials_at_machine[mach_type]}
                needed = {k: max(0, v - current_mats.get(k, 0)) for k, v in required.items()}
                for arrow in upstreams:
                    up_id = arrow["start_id"]
                    if up_id in self.inventory_stocks:
                        stock = self.inventory_stocks[up_id]
                        for mat_type, count in needed.items():
                            if count > 0 and mat_type in stock and stock[mat_type]:
                                pull_count = min(count, len(stock[mat_type]))
                                for _ in range(pull_count):
                                    mat = stock[mat_type].pop(0)
                                    mat.location = mach_type
                                    self.materials_at_machine[mach_type].append(mat)
                                    needed[mat_type] -= 1
                                    inv_outputs[up_id] += 1
                                    self.data_logger.log_inv_movement(self.turns, up_id, "out", mat.id, mat_type, source=up_id, dest=mach_type)  # Log INV out
                                    self.log(f"Turn {self.turns}: Pulled {self.material_dict.get(mat_type, {}).get('display', mat_type)} (ID: {mat.id}) from {up_id} to {mach_type}")
                    elif up_id in self.machine_states and self.machine_states[up_id] == 3 and up_id in self.remaining_machines:
                        up_mats = self.materials_at_machine[up_id]
                        for mat in up_mats[:]:
                            if mat.state in needed and needed[mat.state] > 0:
                                up_mats.remove(mat)
                                mat.location = mach_type
                                self.materials_at_machine[mach_type].append(mat)
                                needed[mat.state] -= 1
                                self.log(f"Turn {self.turns}: Pulled {self.material_dict.get(mat.state, {}).get('display', mat.state)} (ID: {mat.id}) from {up_id} to {mach_type}")
                                if not self.materials_at_machine[up_id]:
                                    self.machine_states[up_id] = 1
                                    self.log(f"Turn {self.turns}: {up_id} emptied - Transitioned to S:1")
                                break
                current_mats = {mat.state: sum(1 for m in self.materials_at_machine[mach_type] if m.state == mat.state) for mat in self.materials_at_machine[mach_type]}
                shortage = not all(current_mats.get(k, 0) >= v for k, v in required.items())
                if shortage:
                    missing = {k: v - current_mats.get(k, 0) for k, v in required.items() if current_mats.get(k, 0) < v}
                    self.data_logger.log_shortage(self.turns, mach_type, "WS", True, list(missing.keys()))
                    self.log(f"Turn {self.turns}: {mach_type} shortage - missing: {missing}")
                else:
                    self.data_logger.log_shortage(self.turns, mach_type, "WS", False, [])
                    self.machine_states[mach_type] = 2
                    self.machine_cycles[mach_type] = tool["cycle_time"]
                    self.log(f"Turn {self.turns}: {mach_type} has all inputs, starting processing (S:2)")
                self.remaining_machines.remove(mach_type)

        # --- Revenue from Car Sales ---
        final_inv = "INV_Final"
        if final_inv in self.inventory_stocks and "NEW_CAR" in self.inventory_stocks[final_inv]:
            qty = len(self.inventory_stocks[final_inv]["NEW_CAR"])
            if qty > 0:
                self.revenue_streams.append([self.turns, 300, 12])
                sold = self.inventory_stocks[final_inv]["NEW_CAR"][:qty]
                del self.inventory_stocks[final_inv]["NEW_CAR"][:qty]
                self.log(f"Turn {self.turns}: Sold {qty} Car(s) from {final_inv}, starting revenue stream")
                inv_outputs[final_inv] += qty

        # Calculate revenue for this turn
        revenue_per_turn = 0
        active_streams = []
        for stream in self.revenue_streams[:]:
            start_turn, amount, duration = stream
            if self.turns <= start_turn + duration - 1:
                revenue_per_turn += amount
                active_streams.append(stream)
            else:
                self.log(f"Turn {self.turns}: Revenue stream of ${amount}/turn from turn {start_turn} expired")
        self.revenue_streams = active_streams
        self.money += revenue_per_turn
        if revenue_per_turn > 0:
            self.log(f"Turn {self.turns}: Revenue this turn: ${revenue_per_turn}")

        # --- Inventory Turnover Logging ---
        for inv_id in inv_outputs:
            if inv_id not in self.inv_turnover_history:
                self.inv_turnover_history[inv_id] = []
            self.inv_turnover_history[inv_id].append(inv_outputs[inv_id])
            if len(self.inv_turnover_history[inv_id]) > 10:
                self.inv_turnover_history[inv_id].pop(0)

        # --- NN Reward and Training ---
        if self.use_nn:
            next_state = self.get_current_state()
            next_state["global"]["cycles_completed"] = cycles_completed
            next_state["global"]["breakdowns"] = breakdowns
            total_costs_per_turn = self.fixed_costs + total_cost + (material_cost if self.turns % self.delivery_interval == 1 else 0)
            net_profit = revenue_per_turn - total_costs_per_turn
            reward = net_profit / 100
            self.log(f"Turn {self.turns}: Net Profit = ${net_profit}, Reward = {reward} (Revenue: ${revenue_per_turn}, Costs: ${total_costs_per_turn})")
            next_state_vector = np.concatenate([
                np.array([next_state[mach_id][key] for key in next_state[mach_id]])
                for mach_id in [m[0] for m in self.machines]
            ] + [
                np.array([next_state["global"][key] for key in next_state["global"]])
            ])
            done = self.turns >= self.max_turns
            self.nn_agent.store_experience(state_vector_flat, action, reward, next_state_vector, done)
            if self.train_nn:
                self.nn_agent.train()

        # --- Log Machine State, Cycle Times, and Vibration ---
        cycle_starts = getattr(self, 'cycle_starts', {m[0]: None for m in self.machines})
        for mach_type, _, _ in self.machines:
            current_state = self.machine_states.get(mach_type, 1)
            prev_state = self.prev_states.get(mach_type, 1)  # Get previous state
            self.data_logger.log_state(self.turns, mach_type, "WS", current_state)
            
            # Cycle time
            if current_state == 2 and cycle_starts.get(mach_type) is None:
                cycle_starts[mach_type] = self.turns
            elif current_state == 3 and cycle_starts.get(mach_type) is not None:
                tool = self.machine_tools.get(mach_type, {})
                cycle_time = tool.get("cycle_time", 0)
                start_turn = cycle_starts[mach_type]
                actual_cycle_time = self.turns - start_turn
                self.data_logger.log_cycle_time(self.turns, mach_type, "WS", actual_cycle_time)
                self.log(f"Turn {self.turns}: {mach_type} cycle completed, actual cycle time: {actual_cycle_time} (ideal: {cycle_time})")
                cycle_starts[mach_type] = None
            # Downtime
            elif current_state == 4:
                reason = (
                    "breakdown" if self.is_breakdown.get(mach_type, False) else
                    "maintenance" if self.is_maintenance.get(mach_type, False) else
                    "repair" if self.is_repair.get(mach_type, False) else
                    "paused" if self.is_paused.get(mach_type, False) else
                    "unknown"
                )
                self.data_logger.log_downtime(self.turns, mach_type, "WS", reason)
                self.log(f"Turn {self.turns}: {mach_type} in downtime, reason: {reason}")
            
            # Vibration (Simplified—Every Turn, All States Except Post-Breakdown S:4)
            tool = self.machine_tools.get(mach_type, {})
            lifecycle = tool.get("lifecycle", 50)  # Use lifecycle for wear
            wear = self.breakdown_risk.get(mach_type, 0)
            wear_ratio = wear / lifecycle  # 0.0 to 1.0
            
            if current_state == 4 and self.is_breakdown.get(mach_type, False):  # Only 0 after breakdown
                rms = 0.0
            else:  # S:1, S:2, S:3, S:4 (maintenance/repair/paused) all vibrate
                if wear_ratio <= 0.5:  # 0–50%: Healthy
                    base_rms = random.uniform(0.02, 0.04)
                elif wear_ratio <= 0.8:  # 50–80%: Dulling
                    last_rms = self.last_vibration.get(mach_type, 0.035)  # Default mid-healthy
                    drift = 0.0001 * (lifecycle / 50)  # Scale drift to lifecycle
                    base_rms = random.uniform(last_rms, min(last_rms + 0.02, 0.06))
                    base_rms += drift
                    base_rms = min(base_rms, 0.06)
                elif wear_ratio <= 0.95:  # 80–95%: Worn
                    base_rms = random.uniform(0.06, 0.12)
                else:  # 95–100%: Near failure
                    base_rms = random.uniform(0.10, 0.20)

                # Add random spikes
                spike_chance = 0.005 if wear_ratio <= 0.95 else 0.05  # 0.5% early, 5% late
                if random.random() < spike_chance:
                    if wear_ratio <= 0.95:  # Early spike
                        rms = random.uniform(0.15, 0.25)
                    else:  # Late spike
                        rms = random.uniform(0.30, 0.50)
                        self.spike_duration[mach_type] = random.randint(1, max(1, min(3, int(lifecycle * 0.1))))
                else:
                    rms = base_rms
                    if mach_type in self.spike_duration and self.spike_duration[mach_type] > 0:
                        rms = random.uniform(0.30, 0.50)  # Continue spike
                        self.spike_duration[mach_type] -= 1
                        if self.spike_duration[mach_type] <= 0:
                            del self.spike_duration[mach_type]
                self.last_vibration[mach_type] = base_rms  # Update last base RMS

            self.data_logger.log_vibration(self.turns, mach_type, "WS", rms)

            # Breakdown Check (Vibration-Driven)
            if current_state != 4:  # Only active machines
                if rms >= 0.30:  # Severe spike threshold
                    self.alert_duration[mach_type] += 1
                    self.log(f"Turn {self.turns}: {mach_type} high vibration (RMS: {rms:.3f}, Duration: {self.alert_duration[mach_type]})")
                    if self.alert_duration[mach_type] >= 5 and not self.maintenance_queued[mach_type]:
                        self.machine_states[mach_type] = 4
                        self.is_breakdown[mach_type] = True
                        self.alert_active[mach_type] = False
                        self.alert_duration[mach_type] = 0
                        breakdowns += 1
                        self.log(f"Turn {self.turns}: {mach_type} broke down due to persistent high vibration")
                else:
                    self.alert_duration[mach_type] = 0  # Reset if no spike

            # OC-SVM Training and Prediction
            if self.use_ocsvm:
                # Train at specified turn (32 from __init__)
                if self.turns == self.training_turn and not self.ocsvm_predictor.trained:
                    vibration_data = {}
                    for ws_id in [m[0] for m in self.machines]:
                        vibe_logs = self.data_logger.history.get(ws_id, {}).get("vibration_level", [])
                        healthy_vibes = [v["vibration_level"] for t, v in vibe_logs if t <= self.training_turn and 0.02 <= v["vibration_level"] <= 0.06]
                        if healthy_vibes:
                            vibration_data[ws_id] = healthy_vibes
                    self.ocsvm_predictor.train(vibration_data)
                    self.log(f"Turn {self.turns}: OC-SVM trained on healthy vibration data (0.02–0.06 g’s).")
                
                # Predict anomalies after training
                if self.ocsvm_predictor.trained:
                    should_alert = self.ocsvm_predictor.predict(mach_type, self.turns, rms)
                    if should_alert and current_state in [1, 2, 3]:
                        self.alert_active[mach_type] = True
                        self.log(f"Turn {self.turns}: OC-SVM detected anomaly on {mach_type} (RMS: {rms:.3f})")
                    elif current_state != 4:  # Reset alert if no anomaly and not down
                        self.alert_active[mach_type] = False

            self.prev_states[mach_type] = current_state  # Update previous state

        self.cycle_starts = cycle_starts

        # --- UI and Layout Update ---
        self.update_status()
        self.draw_layout()

        # --- Info Board Refresh ---
        if self.selected_item:
            self.update_info_board()
            
    def get_current_state(self):
        state = {}
        # Per-machine state
        for mach_type, _, _ in self.machines:
            mach_id = mach_type
            tool = self.machine_tools.get(mach_id, {})
            cycles_left = self.machine_cycles.get(mach_id, 0) if self.machine_states.get(mach_id, 1) == 2 else 0
            cycle_time = tool.get("cycle_time", 0) if tool else 0
            state[mach_id] = {
                "state": self.machine_states.get(mach_id, 1),
                "wear": self.breakdown_risk.get(mach_id, 0),
                "lifecycle": tool.get("lifecycle", 50),
                "alert_active": self.alert_active.get(mach_id, False),
                "turns_since_alert": self.turns_since_alert.get(mach_id, 0),
                "maintenance_queued": self.maintenance_queued.get(mach_id, False),
                "is_breakdown": self.is_breakdown.get(mach_id, False),
                "is_maintenance": self.is_maintenance.get(mach_id, False),
                "is_repair": self.is_repair.get(mach_id, False),
                "is_paused": self.is_paused.get(mach_id, False),
                "cycle_progress": cycles_left if cycles_left > 0 else cycle_time if self.machine_states.get(mach_id, 1) == 3 else 0
            }
        
        # Global state
        state["global"] = {
            "cycles_completed": 0,      # Updated in run_turn when cycles finish
            "breakdowns": 0,            # Updated in run_turn when breakdowns occur
            "total_turns": self.turns
        }
        
        return state
 
    def schedule_maintenance(self, machine_id):
        state = self.machine_states.get(machine_id, 1)
        if state in [1, 3]:  # Idle or Ready → Act now
            if self.is_breakdown.get(machine_id, False):
                self.machine_states[machine_id] = 4
                self.maintenance_turns[machine_id] = 40
                self.is_repair[machine_id] = True
                self.is_breakdown[machine_id] = False
                self.breakdown_risk[machine_id] = 0
                self.alert_active[machine_id] = False
                self.money -= 500  # Repair cost
                self.log(f"Turn {self.turns}: {machine_id} scheduled for repair (40 turns), cost $500")
            else:
                self.machine_states[machine_id] = 4
                self.maintenance_turns[machine_id] = 5
                self.is_maintenance[machine_id] = True
                self.breakdown_risk[machine_id] = 0
                self.alert_active[machine_id] = False
                self.money -= 50  # Maintenance cost
                self.log(f"Turn {self.turns}: {machine_id} scheduled for quick maintenance (5 turns), cost $50")
            self.draw_layout()
        elif state == 2:  # Processing → Queue it
            self.maintenance_queued[machine_id] = True
            if self.is_breakdown.get(machine_id, False):
                self.log(f"Turn {self.turns}: Repair queued for {machine_id} after processing (40 turns)")
            else:
                self.log(f"Turn {self.turns}: Maintenance queued for {machine_id} after processing (5 turns)")
        elif state == 4:  # Already in S:4
            if self.is_breakdown.get(machine_id, False):
                self.maintenance_turns[machine_id] = 40
                self.is_repair[machine_id] = True
                self.is_breakdown[machine_id] = False
                self.breakdown_risk[machine_id] = 0
                self.alert_active[machine_id] = False
                self.money -= 500  # Repair cost
                self.log(f"Turn {self.turns}: {machine_id} scheduled for repair (40 turns) post-breakdown, cost $500")
            elif not (self.is_maintenance[machine_id] or self.is_repair[machine_id] or self.is_paused[machine_id]):
                self.log(f"Turn {self.turns}: {machine_id} in unexpected S:4 state - action ignored")
            self.draw_layout()
            
    def unpause_machine(self, machine_id):
        state = self.machine_states.get(machine_id, 1)
        if state == 4 and self.is_paused[machine_id]:
            self.machine_states[machine_id] = 1
            self.is_paused[machine_id] = False
            self.log(f"Turn {self.turns}: {machine_id} unpaused by player, back to S:1")
            self.remaining_machines.add(machine_id)
            self.draw_layout()
        else:
            self.log(f"Turn {self.turns}: {machine_id} not paused - unpause ignored")
            
    def unpause_machine(self, machine_id):
        state = self.machine_states.get(machine_id, 1)
        if state == 4 and self.is_paused[machine_id]:
            self.machine_states[machine_id] = 1
            self.is_paused[machine_id] = False
            self.log(f"Turn {self.turns}: {machine_id} unpaused by player, back to S:1")
            self.remaining_machines.add(machine_id)
            self.draw_layout()
        else:
            self.log(f"Turn {self.turns}: {machine_id} not paused - unpause ignored")
                
    def toggle_auto_turn(self):
        if self.auto_running:
            self.auto_running = False
            self.toggle_auto_button.config(text="Start Auto-Turn")
            print("Auto-turn stopped")
        else:
            self.auto_running = True
            self.toggle_auto_button.config(text="Stop Auto-Turn")
            self.run_auto_turn()
            print("Auto-turn started")
            
    def toggle_nn(self):
        self.use_nn = not self.use_nn
        self.toggle_nn_button.config(text="Enable NN" if not self.use_nn else "Disable NN")
        self.update_train_button_state()
        print(f"NN {'enabled' if self.use_nn else 'disabled'}")

    def toggle_training(self):
        if self.use_nn:
            self.train_nn = not self.train_nn
            self.train_button.config(text="Enable Training" if not self.train_nn else "Disable Training")
            self.update_train_button_state()
            print(f"NN training {'enabled' if self.train_nn else 'disabled'}")

    def update_train_button_state(self):
        if self.use_nn:
            self.train_button.config(state="normal", text="Enable Training" if not self.train_nn else "Disable Training")
        else:
            self.train_button.config(state="disabled", text="Enable Training")
            
    def toggle_logging(self):
        self.logging_enabled = not self.logging_enabled
        self.toggle_log_button.config(text="Disable Logging" if self.logging_enabled else "Enable Logging")
        print(f"Logging {'enabled' if self.logging_enabled else 'disabled'}")
            
    def run_auto_turn(self):
        if self.auto_running:
            self.run_turn()
            self.update_status()
            self.root.after(100, self.run_auto_turn)
 
    def generate_analytics_report(self):
        report = {
            "Turns": self.turns,
            "Money": self.money,
            "Orders Completed": len([o for o in self.orders if o["due_turn"] <= self.turns]),
            "Orders Remaining": len(self.orders),
            "OEE": {},
            "FPY": {},
            "Throughput": {},
            "Stock Turnover": {}
        }
        
        # Machine metrics
        for machine_id in self.machine_state_history:
            history = self.machine_state_history[machine_id]
            output_history = self.machine_output_history[machine_id]
            cycle_times = self.machine_cycle_times[machine_id]
            tool = self.machine_tools.get(machine_id, {})
            
            if history and output_history and cycle_times:
                # Last 100 turns (or all if < 100)
                recent_history = history[-100:] if len(history) >= 100 else history
                total_turns = len(recent_history)
                
                # Availability: Turns in S:2 (Processing) over last 100 turns
                processing_turns = sum(1 for s in recent_history if s == 2)
                availability = processing_turns / total_turns if total_turns > 0 else 0
                
                # Performance: Actual cycles completed vs. ideal in S:2 time
                cycle_time = tool.get("cycle_time", 1)
                actual_cycles = len([o for o in output_history[-100:] if o])  # Cycles in last 100 turns
                ideal_cycles = processing_turns / cycle_time if cycle_time > 0 else 0
                performance = actual_cycles / ideal_cycles if ideal_cycles > 0 else 1.0  # Cap at 100%
                
                # Quality: Good units vs. total (placeholder until faults added)
                total_output = actual_cycles
                good_output = sum(len(outputs) for outputs, _ in output_history[-100:] if all(q >= 0.95 for q in outputs))
                quality = good_output / total_output if total_output > 0 else 1.0
                
                # OEE
                oee = availability * performance * quality * 100  # As percentage
                report["OEE"][machine_id] = round(oee, 1)
                report["FPY"][machine_id] = round(quality * 100, 1)  # Keep FPY as quality for now
                report["Throughput"][machine_id] = round(actual_cycles / total_turns, 2) if total_turns > 0 else 0  # Units per turn
                
        # Inventory metrics
        total_turnover = 0
        inv_count = 0
        for inv_id in self.inv_turnover_history:
            turnover = sum(self.inv_turnover_history[inv_id]) / len(self.inv_turnover_history[inv_id]) if self.inv_turnover_history[inv_id] else 0
            report["Stock Turnover"][inv_id] = round(turnover, 2)
            total_turnover += turnover
            inv_count += 1
        avg_stock_turnover = round(total_turnover / inv_count, 2) if inv_count > 0 else 0
        
        # Additional metrics
        avg_oee = round(sum(report["OEE"].values()) / len(report["OEE"]), 1) if report["OEE"] else 0
        order_completion_rate = round((report["Orders Completed"] / (report["Orders Completed"] + report["Orders Remaining"])) * 100, 1) if (report["Orders Completed"] + report["Orders Remaining"]) > 0 else 0
        
        # Display in window
        report_text = (
            f"Turns: {report['Turns']}\n"
            f"Money: ${report['Money']}\n"
            f"Orders Completed: {report['Orders Completed']}\n"
            f"Orders Remaining: {report['Orders Remaining']}\n"
            f"Order Completion Rate: {order_completion_rate}%\n"
            f"Average OEE: {avg_oee}%\n"
            f"Average Stock Turnover: {avg_stock_turnover}/turn\n"
            f"\nMachine OEE (Last 100 Turns):\n"
        )
        for mach_id, oee in report["OEE"].items():
            report_text += f"{mach_id}: {oee}%\n"
        
        analytics_window = tk.Toplevel(self.root)
        analytics_window.title("Analytics Report")
        analytics_window.geometry("1000x800")  # Bigger window
        tk.Label(analytics_window, text=report_text, justify=tk.LEFT, font=("Arial", 12)).pack(pady=10)
        button_frame = tk.Frame(analytics_window)
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="Export CSV", command=lambda: self.export_analytics_to_csv(report), width=20).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Close", command=analytics_window.destroy, width=20).pack(side=tk.LEFT, padx=5)
        print("Analytics report displayed")
        
    def export_analytics_to_csv(self, report):
        # Open file dialog for saving CSV
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Analytics Report as CSV",
            initialdir=LOGS_DIR,
            initialfile=f"analytics_turn_{self.turns}.csv"
        )
        
        if not file_path:
            print("Export cancelled - no file path provided")
            return
        
        # Prepare data from history
        headers = [
            "Turns", "Money", "Orders Completed", "Orders Remaining", "Order Completion Rate",
            "Average OEE", "Average FPY", "Average Throughput", "Average Stock Turnover",
            "Average Inventory Availability", "Total Downtime Turns"
        ]
        
        # Calculate metrics from history
        total_turns = self.turns
        orders_completed = report["Orders Completed"]
        orders_remaining = report["Orders Remaining"]
        order_completion_rate = round((orders_completed / (orders_completed + orders_remaining)) * 100, 1) if (orders_completed + orders_remaining) > 0 else 0
        
        # Machine metrics
        oee_list = []
        fpy_list = []
        throughput_list = []
        downtime_turns = 0
        for machine_id in self.machine_state_history:
            history = self.machine_state_history[machine_id]
            output_history = self.machine_output_history[machine_id]
            cycle_times = self.machine_cycle_times[machine_id]
            tool = self.machine_tools.get(machine_id, {})
            
            if history and output_history and cycle_times:
                availability = sum(1 for i in range(1, len(history)) if (history[i-1], history[i]) in [(1, 2), (2, 2), (2, 3), (3, 1)]) / (len(history) - 1) * 100
                total_outputs = sum(count for _, count in output_history)
                good_outputs = sum(sum(1 for q in qualities if q >= 0.9) for qualities, _ in output_history)
                fpy = (good_outputs / total_outputs) * 100 if total_outputs > 0 else 100
                throughput = 1 / tool.get("cycle_time", 1) if tool.get("cycle_time", 1) > 0 else 0
                actual_outputs = sum(count for _, count in output_history)
                performance = (actual_outputs / total_turns) / throughput * 100 if throughput > 0 else 100
                oee = (availability / 100) * (performance / 100) * (fpy / 100) * 100
                oee_list.append(oee)
                fpy_list.append(fpy)
                throughput_list.append(throughput)
                downtime_turns += sum(1 for state in history if state == 1)  # Idle state
        
        avg_oee = round(sum(oee_list) / len(oee_list), 1) if oee_list else 0
        avg_fpy = round(sum(fpy_list) / len(fpy_list), 1) if fpy_list else 0
        avg_throughput = round(sum(throughput_list) / len(throughput_list), 2) if throughput_list else 0
        
        # Inventory metrics
        total_turnover = 0
        inv_count = 0
        total_availability = 0
        for inv_id in self.inv_turnover_history:
            turnover = sum(self.inv_turnover_history[inv_id]) / len(self.inv_turnover_history[inv_id]) if self.inv_turnover_history[inv_id] else 0
            total_turnover += turnover
            inv_count += 1
            availability = sum(1 for avail in self.inv_availability_history[inv_id]) / len(self.inv_availability_history[inv_id]) * 100 if self.inv_availability_history[inv_id] else 0
            total_availability += availability
        avg_stock_turnover = round(total_turnover / inv_count, 2) if inv_count > 0 else 0
        avg_inv_availability = round(total_availability / inv_count, 1) if inv_count > 0 else 0
        
        # Data row
        data = [
            total_turns,
            self.money,
            orders_completed,
            orders_remaining,
            order_completion_rate,
            avg_oee,
            avg_fpy,
            avg_throughput,
            avg_stock_turnover,
            avg_inv_availability,
            downtime_turns
        ]
        
        # Write to CSV
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(data)
        
        print(f"Analytics report exported to {file_path}")
     
    def cancel_purchase(self, event):
        if self.pending_new:
            del self.visual_positions[self.pending_new]
            self.pending_new = None
            self.canvas.unbind("<Motion>")
            self.canvas.bind("<Motion>", self.hover_check)
            self.canvas.unbind("<Button-3>")  # Explicitly clear
            self.canvas.bind("<Button-3>", self.handle_right_click)
            self.draw_layout()
            print(f"Purchase of {self.pending_new} cancelled—no money deducted")
        elif self.is_new_item and self.dragging:
            shape_id = self.dragging[0]
            if shape_id.startswith("WS"):
                del self.machine_states[shape_id]
                del self.machine_cycles[shape_id]
                del self.materials_at_machine[shape_id]
            else:
                del self.inventory_stocks[shape_id]
                del self.inventory_accepts[shape_id]
            self.dragging = None
            self.is_new_item = False
            self.canvas.unbind("<Motion>")
            self.canvas.bind("<Motion>", self.hover_check)
            self.canvas.unbind("<Button-3>")  # Clear here too
            self.canvas.bind("<Button-3>", self.handle_right_click)
            self.draw_layout()
            print(f"Purchase of {shape_id} cancelled—no money deducted")

    def buy_machine(self):
        if self.pending_new:  # Prevent multiple pending buys
            print("Finish placing the current object first!")
            return
        new_id = f"WS{self.next_machine_id}"
        self.next_machine_id += 1
        self.pending_new = new_id
        self.visual_positions[new_id] = (15, 10)  # Center-ish start
        self.canvas.bind("<Motion>", self.move_shape)  # Free movement
        self.canvas.bind("<Button-1>", self.handle_left_click)  # Confirm
        self.canvas.bind("<Button-3>", self.cancel_purchase)  # Cancel
        self.draw_layout()
        print(f"New Working Station {new_id} spawned—left-click to place, right-click to cancel (cost: $100)")

    def buy_inventory(self):
        if self.pending_new:
            print("Finish placing the current object first!")
            return
        new_id = f"INV{self.next_inv_id}"
        self.next_inv_id += 1
        self.pending_new = new_id
        self.visual_positions[new_id] = (15, 10)
        self.canvas.bind("<Motion>", self.move_shape)
        self.canvas.bind("<Button-1>", self.handle_left_click)
        self.canvas.bind("<Button-3>", self.cancel_purchase)
        self.draw_layout()
        print(f"New Inventory Rack {new_id} spawned—left-click to place, right-click to cancel (cost: $100)")
            
    def sell_new_car(self):
        final_inv = "INV_Final"  # Assuming this is your final inventory ID
        if final_inv in self.inventory_stocks and "NEW_CAR" in self.inventory_stocks[final_inv]:
            cars = self.inventory_stocks[final_inv]["NEW_CAR"]
            if cars:
                car = cars.pop(0)  # Sell the first car
                self.money += 1000
                self.update_status()
                # Log full lineage
                lineage = self.trace_material_lineage(car)
                print(f"Sold NEW_CAR (ID: {car.id}) from {final_inv} for $1000")
                print(f"Lineage: {lineage}")
                self.draw_layout()
            else:
                print(f"No NEW_CAR available in {final_inv} to sell!")
        else:
            print(f"No NEW_CAR in {final_inv} to sell!")
            
    def view_orders(self):
        order_window = tk.Toplevel(self.root)
        order_window.title("Order Queue")
        order_window.geometry("300x400")
        tk.Label(order_window, text="Pending Orders", font=("Arial", 12, "bold")).pack(pady=5)
        order_frame = tk.Frame(order_window)
        order_frame.pack(fill=tk.BOTH, expand=True)
        order_scroll = tk.Scrollbar(order_frame, orient=tk.VERTICAL)
        order_listbox = tk.Listbox(order_frame, height=15, yscrollcommand=order_scroll.set)
        order_scroll.config(command=order_listbox.yview)
        order_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        order_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for order in sorted(self.orders, key=lambda x: x["due_turn"]):
            order_listbox.insert(tk.END, f"{order['quantity']} {order['item']} due by Turn {order['due_turn']}")
        tk.Button(order_window, text="Close", command=order_window.destroy, width=10).pack(pady=10)
        print("Viewing order queue")            
            
    def trace_material_lineage(self, material):
        lineage = {}
        def recurse_parents(mat):
            if mat.id not in lineage:  # Avoid duplicates
                lineage[mat.id] = mat.state
                for parent_id in mat.parent_ids:
                    if parent_id in self.materials:
                        recurse_parents(self.materials[parent_id])
        recurse_parents(material)
        return lineage

    def configure_tool(self, event):
        x, y = event.x, event.y
        for machine_type, grid_x, grid_y in self.machines:
            cx = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
            cy = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
            if abs(cx - x) < 40 and abs(cy - y) < 40:
                tool_window = tk.Toplevel(self.root)
                tool_window.title(f"Configure Tool for {machine_type}")
                tool_window.geometry(f"400x400+{self.CANVAS_WIDTH + 20}+10")

                from tkinter import ttk
                notebook = ttk.Notebook(tool_window)
                notebook.pack(pady=10, fill=tk.BOTH, expand=True)

                categories = {
                    "Raw Processing": [],
                    "Components": [],
                    "Final Assembly": [],
                    "Quality": []
                }
                for tool_name, tool_data in self.tool_dict.items():
                    category = tool_data.get("category", "Raw Processing")
                    if category in categories:
                        categories[category].append(tool_name)

                tab_frames = {}
                listboxes = {}
                for category in categories:
                    tab_frames[category] = tk.Frame(notebook)
                    notebook.add(tab_frames[category], text=category)
                    listboxes[category] = tk.Listbox(tab_frames[category], height=10, selectmode="single")
                    listboxes[category].pack(pady=5, fill=tk.BOTH, expand=True)
                    scrollbar = tk.Scrollbar(tab_frames[category], orient=tk.VERTICAL, command=listboxes[category].yview)
                    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                    listboxes[category].config(yscrollcommand=scrollbar.set)
                    listboxes[category].insert(tk.END, "NO TOOL")
                    seen_tools = set()
                    for tool_name in sorted(categories[category]):
                        if tool_name not in seen_tools:
                            seen_tools.add(tool_name)
                            inputs = ", ".join(self.simplify_material_name(k) for k in self.tool_dict[tool_name]["inputs"].keys())
                            outputs = ", ".join(self.simplify_material_name(k) for k in self.tool_dict[tool_name]["outputs"].keys())
                            listboxes[category].insert(tk.END, f"{tool_name}: {outputs} | {inputs}")

                selected_label = tk.Label(tool_window, text="Selected: None", font=("Arial", 10))
                selected_label.pack(pady=5)

                def update_info(*args):
                    for category in categories:
                        sel = listboxes[category].curselection()
                        if sel:
                            selected_tool = listboxes[category].get(sel[0]).split(":")[0]
                            selected_label.config(text=f"Selected: {selected_tool if selected_tool != 'NO TOOL' else 'None'}")
                            break

                for category in categories:
                    listboxes[category].bind("<<ListboxSelect>>", update_info)

                def save_tool():
                    for category in categories:
                        sel = listboxes[category].curselection()
                        if sel:
                            selected_tool = listboxes[category].get(sel[0]).split(":")[0]
                            if selected_tool == "NO TOOL":
                                if machine_type in self.machine_tools:
                                    del self.machine_tools[machine_type]
                                self.log(f"Turn {self.turns}: Removed tool from {machine_type}")
                            else:
                                self.machine_tools[machine_type] = self.tool_dict[selected_tool].copy()
                                self.machine_tools[machine_type]["name"] = selected_tool  # Set name explicitly
                                self.machine_tools[machine_type]["wear"] = self.machine_tools[machine_type].get("wear", 0)
                                self.log(f"Turn {self.turns}: Tool set for {machine_type}: {selected_tool}, Cycle Time={self.tool_dict[selected_tool]['cycle_time']}")
                            break
                    tool_window.destroy()
                    self.draw_layout()

                def trigger_maintenance():
                    self.schedule_maintenance(machine_type)  # Call the separate method
                    tool_window.destroy()

                button_frame = tk.Frame(tool_window)
                button_frame.pack(pady=10)
                tk.Button(button_frame, text="Confirm", command=save_tool).pack(side=tk.LEFT, padx=5)
                tk.Button(button_frame, text="Maintenance", command=trigger_maintenance).pack(side=tk.LEFT, padx=5)
                tk.Button(button_frame, text="Cancel", command=tool_window.destroy).pack(side=tk.LEFT, padx=5)
                break

    def configure_inventory(self, event):
        x, y = event.x, event.y
        for inv_type, grid_x, grid_y in self.inventories:
            cx = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
            cy = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
            if abs(cx - x) < 50 and abs(cy - y) < 30:
                inv_window = tk.Toplevel(self.root)
                inv_window.title(f"Configure Acceptance for {inv_type}")
                inv_window.geometry(f"400x600+{self.CANVAS_WIDTH + 20}+10")

                # Acceptance list
                tk.Label(inv_window, text="Select Accepted Materials:").pack(pady=5)
                inv_frame = tk.Frame(inv_window)
                inv_frame.pack(pady=5, fill=tk.BOTH, expand=True)
                inv_scroll = tk.Scrollbar(inv_frame, orient=tk.VERTICAL)
                inv_listbox = tk.Listbox(inv_frame, height=10, selectmode="multiple", yscrollcommand=inv_scroll.set)
                inv_scroll.config(command=inv_listbox.yview)
                inv_scroll.pack(side=tk.RIGHT, fill=tk.Y)
                inv_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

                # Build display-to-full-name mapping
                display_to_full = {self.material_dict[mat]["display"]: mat for mat in self.material_dict.keys()}
                display_names = sorted(display_to_full.keys())
                for display_name in display_names:
                    inv_listbox.insert(tk.END, display_name)
                current_accepts = self.inventory_accepts.get(inv_type, [])
                for i, display_name in enumerate(display_names):
                    if display_to_full[display_name] in current_accepts:
                        inv_listbox.select_set(i)

                # Capacity field
                cap_frame = tk.Frame(inv_window)
                cap_frame.pack(pady=10)
                tk.Label(cap_frame, text="Capacity:").pack(side=tk.LEFT, padx=5)
                cap_entry = tk.Entry(cap_frame, width=10)
                cap_entry.pack(side=tk.LEFT, padx=5)
                current_cap = self.inventory_caps.get(inv_type)
                if current_cap is not None:
                    cap_entry.insert(0, str(current_cap))

                # Hint and error labels
                hint_label = tk.Label(cap_frame, text="Natural number (0+) or empty for uncapped", fg="grey")
                hint_label.pack(side=tk.LEFT, padx=5)
                error_label = tk.Label(inv_window, text="", fg="red")
                error_label.pack(pady=5)

                def save_inventory():
                    # Handle acceptance
                    selected_indices = inv_listbox.curselection()
                    selected_displays = [display_names[i] for i in selected_indices]
                    selected_mats = [display_to_full[display] for display in selected_displays]
                    
                    # Handle capacity
                    cap_text = cap_entry.get().strip()
                    if cap_text == "":
                        self.inventory_accepts[inv_type] = selected_mats
                        self.inventory_caps[inv_type] = None
                        print(f"Turn {self.turns}: Set {inv_type} to accept: {sorted(selected_mats)}, uncapped")
                        inv_window.destroy()
                        self.draw_layout()
                    else:
                        try:
                            cap_value = int(cap_text)
                            if cap_value >= 0:
                                self.inventory_accepts[inv_type] = selected_mats
                                self.inventory_caps[inv_type] = cap_value
                                print(f"Turn {self.turns}: Set {inv_type} to accept: {sorted(selected_mats)}, capped at {cap_value}")
                                inv_window.destroy()
                                self.draw_layout()
                            else:
                                error_label.config(text="Error: Capacity must be 0 or higher")
                        except ValueError:
                            error_label.config(text="Error: Enter a valid natural number")

                def clear_inventory():
                    inv_listbox.selection_clear(0, tk.END)
                    cap_entry.delete(0, tk.END)
                    error_label.config(text="")
                    print(f"Turn {self.turns}: Cleared selection and capacity for {inv_type} - ready to pick")

                def cancel_inventory():
                    print(f"Turn {self.turns}: Cancelled config for {inv_type}")
                    inv_window.destroy()

                # Buttons
                button_frame = tk.Frame(inv_window)
                button_frame.pack(pady=10)
                tk.Button(button_frame, text="Confirm", command=save_inventory).pack(side=tk.LEFT, padx=5)
                tk.Button(button_frame, text="Clear", command=clear_inventory).pack(side=tk.LEFT, padx=5)
                tk.Button(button_frame, text="Cancel", command=cancel_inventory).pack(side=tk.LEFT, padx=5)
                break

    def simplify_material_name(self, full_name):
        return full_name.split('_')[-1].lower()

    def move_shape(self, event):
        if self.dragging:  # Existing drag
            grid_x = event.x // self.GRID_SIZE
            grid_y = event.y // self.GRID_SIZE
            self.visual_positions[self.dragging] = (int(grid_x), int(grid_y))
            self.draw_layout()
        elif self.pending_new:  # New buy
            grid_x = event.x // self.GRID_SIZE
            grid_y = event.y // self.GRID_SIZE
            self.visual_positions[self.pending_new] = (int(grid_x), int(grid_y))
            self.draw_layout()
        elif self.flow_mode and self.flow_start:
            self.flow_preview = (event.x, event.y)
            self.draw_layout()

    def draw_grid(self):
        for i in range(self.CANVAS_WIDTH // self.GRID_SIZE + 1):
            self.canvas.create_line(i * self.GRID_SIZE, 0, i * self.GRID_SIZE, self.CANVAS_HEIGHT, fill="grey", dash=(2, 2))
        for i in range(self.CANVAS_HEIGHT // self.GRID_SIZE + 1):
            self.canvas.create_line(0, i * self.GRID_SIZE, self.CANVAS_WIDTH, i * self.GRID_SIZE, fill="grey", dash=(2, 2))

    def draw_layout(self):
        self.canvas.delete("all")
        self.draw_grid()

        # Draw inventories
        for inv_id, grid_x, grid_y in self.inventories:
            if inv_id in self.visual_positions:  # Dragged position
                grid_x, grid_y = self.visual_positions[inv_id]
            x = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
            y = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
            color = "red" if self.hover_shape == (inv_id, grid_x, grid_y) else "black"
            points = [x, y-30, x+50, y+30, x-50, y+30]
            self.canvas.create_polygon(points, outline=color, fill="", width=2)
            self.canvas.create_text(x, y-5, text=inv_id, fill="black")
            stock = self.inventory_stocks.get(inv_id, {})
            stock_text = "empty"
            if stock:
                stock_counts = {k: len(v) for k, v in stock.items() if v}
                if stock_counts:
                    first_mat = next(iter(stock_counts.keys()))
                    stock_text = f"{self.simplify_material_name(first_mat)}:{stock_counts[first_mat]}"
            self.canvas.create_text(x, y+25, text=stock_text, fill="black", font=("Arial", 10))
            dot_color = "red" if self.hover_dot and self.hover_dot[0] == inv_id else "black"
            self.canvas.create_oval(x-2, y-32, x+2, y-28, fill="red" if self.hover_dot and self.hover_dot[3] == "top" and self.hover_dot[0] == inv_id else dot_color)
            self.canvas.create_oval(x-27, y-2, x-23, y+2, fill="red" if self.hover_dot and self.hover_dot[3] == "left" and self.hover_dot[0] == inv_id else dot_color)
            self.canvas.create_oval(x+23, y-2, x+27, y+2, fill="red" if self.hover_dot and self.hover_dot[3] == "right" and self.hover_dot[0] == inv_id else dot_color)
            self.canvas.create_oval(x-2, y+28, x+2, y+32, fill="red" if self.hover_dot and self.hover_dot[3] == "bottom" and self.hover_dot[0] == inv_id else dot_color)

        # Draw machines
        for mach_id, grid_x, grid_y in self.machines:
            if mach_id in self.visual_positions:  # Dragged position
                grid_x, grid_y = self.visual_positions[mach_id]
            x = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
            y = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
            color = "red" if self.hover_shape == (mach_id, grid_x, grid_y) else "black"
            self.canvas.create_rectangle(x-40, y-40, x+40, y+40, outline=color, fill="", width=2)
            self.canvas.create_text(x, y, text=mach_id, fill="black")
            tool = self.machine_tools.get(mach_id, {})
            if tool and "outputs" in tool:
                output_text = self.simplify_material_name(next(iter(tool["outputs"].keys())))
                self.canvas.create_text(x, y+10, text=output_text, fill="black", font=("Arial", 8))
            state = self.machine_states.get(mach_id, 1)
            if state == 1:
                state_text = "Idle S:1"
            elif state == 2:
                cycles = self.machine_cycles.get(mach_id, 0)
                max_cycle = tool.get("cycle_time", 0) if tool else 0
                state_text = f"Processing S:2 ({cycles}/{max_cycle})"
            elif state == 3:
                state_text = "Ready S:3"
            elif state == 4:
                turns_left = self.maintenance_turns.get(mach_id, 0)
                if self.is_breakdown.get(mach_id, False):
                    state_text = "Breakdown S:4"
                elif self.is_maintenance.get(mach_id, False):
                    state_text = f"Maintenance S:4 ({turns_left}/5)"
                elif self.is_repair.get(mach_id, False):
                    state_text = f"Repair S:4 ({turns_left}/40)"
                elif self.is_paused.get(mach_id, False):
                    state_text = "Paused S:4"
                else:
                    state_text = "Blocked S:4"
            if self.alert_active.get(mach_id, False) and self.turns_since_alert.get(mach_id, 0) < 20:
                self.canvas.create_text(x, y-30, text="ALERT", fill="black", font=("Arial", 8))
                self.canvas.create_text(x, y-10, text=state_text, fill="blue", font=("Arial", 8))
            else:
                self.canvas.create_text(x, y-10, text=state_text, fill="blue", font=("Arial", 8))
            cycles = self.machine_cycles.get(mach_id, 0)
            max_cycle = tool.get("cycle_time", 0) if tool else 0
            display_cycle = max_cycle - cycles if state == 2 else (max_cycle if state == 3 else 0)
            cycle_text = f"CYCLE TIME:\n{display_cycle}/{max_cycle}" if max_cycle > 0 else "NO TOOL"
            self.canvas.create_text(x, y+27, text=cycle_text, fill="black", font=("Arial", 8))
            dot_color = "red" if self.hover_dot and self.hover_dot[0] == mach_id else "black"
            self.canvas.create_oval(x-2, y-42, x+2, y-38, fill="red" if self.hover_dot and self.hover_dot[3] == "top" and self.hover_dot[0] == mach_id else dot_color)
            self.canvas.create_oval(x-2, y+38, x+2, y+42, fill="red" if self.hover_dot and self.hover_dot[3] == "bottom" and self.hover_dot[0] == mach_id else dot_color)
            self.canvas.create_oval(x-42, y-2, x-38, y+2, fill="red" if self.hover_dot and self.hover_dot[3] == "left" and self.hover_dot[0] == mach_id else dot_color)
            self.canvas.create_oval(x+38, y-2, x+42, y+2, fill="red" if self.hover_dot and self.hover_dot[3] == "right" and self.hover_dot[0] == mach_id else dot_color)

        # Draw pending new object
        if self.pending_new:
            grid_x, grid_y = self.visual_positions[self.pending_new]
            x = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
            y = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
            if self.pending_new.startswith("WS"):
                self.canvas.create_rectangle(x-40, y-40, x+40, y+40, outline="black", fill="", width=2)
                self.canvas.create_text(x, y, text=self.pending_new, fill="black")
            else:
                points = [x, y-30, x+50, y+30, x-50, y+30]
                self.canvas.create_polygon(points, outline="black", fill="", width=2)
                self.canvas.create_text(x, y-5, text="INV", fill="black")

        # Draw arrows (update to use visual_positions)
        for flow_id, flow_data in self.arrows.items():
            start_id = flow_data["start_id"]
            end_id = flow_data["end_id"]
            start_pos = flow_data["start_pos"]
            end_pos = flow_data["end_pos"]
            start_coords = None
            end_coords = None

            # Use visual_positions if dragged, otherwise static
            for shape_id, grid_x, grid_y in self.machines + self.inventories:
                if shape_id in self.visual_positions:
                    grid_x, grid_y = self.visual_positions[shape_id]
                x = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
                y = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
                if shape_id == start_id and start_coords is None:
                    start_coords = {
                        "top": (x, y-40 if shape_id.startswith("WS") else y-30),
                        "bottom": (x, y+40 if shape_id.startswith("WS") else y+30),
                        "left": (x-40 if shape_id.startswith("WS") else x-25, y),
                        "right": (x+40 if shape_id.startswith("WS") else x+25, y)
                    }[start_pos]
                if shape_id == end_id and end_coords is None:
                    end_coords = {
                        "top": (x, y-40 if shape_id.startswith("WS") else y-30),
                        "bottom": (x, y+40 if shape_id.startswith("WS") else y+30),
                        "left": (x-40 if shape_id.startswith("WS") else x-25, y),
                        "right": (x+40 if shape_id.startswith("WS") else x+25, y)
                    }[end_pos]

            if start_coords and end_coords:
                color = "red" if self.hover_arrow == flow_id else "black"
                self.canvas.create_line(start_coords[0], start_coords[1], end_coords[0], end_coords[1], fill=color, width=2, arrow=tk.LAST)

        if self.flow_mode and self.flow_start and self.flow_preview:
            self.canvas.create_line(self.flow_start[1], self.flow_start[2], self.flow_preview[0], self.flow_preview[1], arrow=tk.LAST, fill="black", dash=(4, 4), width=2)

    def get_status(self):
        return f"Turns: {self.turns} | Money: ${self.money}"

    def update_status(self):
        self.status_label.config(text=f"Turns: {self.turns}")
        self.money_label.config(text=f"Money: ${self.money}")
        
    def export_nn_data(self):
        log_dir = os.path.join("data", "logs", "game_logs")  # Move to logs folder
        os.makedirs(log_dir, exist_ok=True)  # Create dir if missing
        filename = os.path.join(log_dir, f"nn_data_turn_{self.turns}.csv")
        fieldnames = [
            "machine_id", "turn", "state", "wear", "lifecycle", "alert_active",
            "turns_since_alert", "materials_processed", "downtime", "maintenance_queued",
            "is_breakdown", "is_maintenance", "is_repair", "is_paused"
        ]
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for mach_id in self.machine_states.keys():  # Use all WS IDs
                if mach_id in self.nn_data:
                    for data in self.nn_data[mach_id]:
                        writer.writerow({"machine_id": mach_id, **data})
        self.log(f"Exported NN data to {filename}")

    def handle_left_click(self, event):
        if self.pending_new:  # Confirm new WS/INV placement
            if self.money >= 100:
                new_x, new_y = self.visual_positions[self.pending_new]
                self.money -= 100
                if self.pending_new.startswith("WS"):
                    self.machines.append((self.pending_new, new_x, new_y))
                    self.machine_states[self.pending_new] = 1
                    self.machine_cycles[self.pending_new] = 0
                    self.materials_at_machine[self.pending_new] = []
                    self.breakdown_risk[self.pending_new] = 0
                    self.alert_active[self.pending_new] = False
                    self.turns_since_alert[self.pending_new] = 0
                    self.maintenance_turns[self.pending_new] = 0
                    self.is_breakdown[self.pending_new] = False
                    self.maintenance_queued[self.pending_new] = False
                    self.data_logger._initialize_object(self.pending_new, "WS")
                    print(f"Placed {self.pending_new} at ({new_x}, {new_y}) - $100 deducted")
                else:
                    self.inventories.append((self.pending_new, new_x, new_y))
                    self.inventory_accepts[self.pending_new] = list(self.material_dict.keys())
                    self.inventory_stocks[self.pending_new] = {}
                    self.inv_turnover_history[self.pending_new] = []
                    self.inv_availability_history[self.pending_new] = []
                    self.data_logger._initialize_object(self.pending_new, "INV")
                    print(f"Placed {self.pending_new} at ({new_x}, {new_y}) - $100 deducted")
                del self.visual_positions[self.pending_new]
                self.pending_new = None
                self.canvas.unbind("<Motion>")
                self.canvas.bind("<Motion>", self.hover_check)
                self.canvas.bind("<Button-3>", self.handle_right_click)  # Restore deletion
                self.update_status()
                self.draw_layout()
            else:
                print(f"Not enough money to place {self.pending_new}!")
                self.cancel_purchase(event)
        elif self.is_new_item and self.dragging:  # Legacy placement
            key = (self.dragging[0], self.dragging[1], self.dragging[2])
            self.money -= 100
            if self.dragging[0].startswith("WS"):
                self.machines.append(key)
                print(f"Placed {self.dragging[0]} at ({self.dragging[1]}, {self.dragging[2]}) - $100 deducted")
            else:
                self.inventories.append(key)
                print(f"Placed {self.dragging[0]} at ({self.dragging[1]}, {self.dragging[2]}) - $100 deducted")
            self.is_new_item = False
            self.dragging = None
            self.canvas.unbind("<Motion>")
            self.canvas.bind("<Motion>", self.hover_check)
            self.canvas.bind("<Button-3>", self.handle_right_click)  # Restore here too
            self.update_status()
            self.draw_layout()
            self.canvas.bind("<Button-3>", self.handle_right_click)  # Extra safety
        else:
            x, y = event.x, event.y
            dot_hit = False
            for mach_type, grid_x, grid_y in self.machines:
                cx = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
                cy = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
                dots = [
                    (cx, cy-40, "top"),
                    (cx-40, cy, "left"),
                    (cx+40, cy, "right"),
                    (cx, cy+40, "bottom")
                ]
                for dx, dy, pos in dots:
                    if abs(x - dx) < 6 and abs(y - dy) < 6:
                        dot_hit = True
                        if self.flow_mode and self.flow_start:
                            start_id, sx, sy, start_pos = self.flow_start
                            flow_id = f"FLOW{self.next_flow_id}"
                            self.next_flow_id += 1
                            priority = 1.0 if mach_type.startswith("WS") and start_id.startswith("WS") else 0.5
                            arrow = {
                                "start_id": start_id,
                                "start_pos": start_pos,
                                "end_id": mach_type,
                                "end_pos": pos,
                                "priority": priority
                            }
                            self.arrows[flow_id] = arrow
                            self.downstream_arrows.setdefault(start_id, []).append(arrow)
                            self.upstream_arrows.setdefault(mach_type, []).append(arrow)
                            print(f"Arrow finished connecting: {flow_id} - {start_id} ({start_pos}) to {mach_type} ({pos}) at ({grid_x}, {grid_y}) with priority {priority}")
                            self.flow_mode = False
                            self.flow_start = None
                            self.flow_preview = None
                            self.canvas.unbind("<Motion>")
                            self.canvas.bind("<Motion>", self.hover_check)
                            self.canvas.bind("<Button-3>", self.handle_right_click)  # Restore in flow too
                            self.draw_layout()
                            self.canvas.bind("<Button-3>", self.handle_right_click)  # Extra safety
                        elif not self.flow_mode:
                            self.flow_mode = True
                            self.flow_start = (mach_type, dx, dy, pos)
                            self.flow_preview = (dx, dy)
                            self.canvas.bind("<Motion>", self.move_shape)
                            self.draw_layout()
                            self.log(f"Arrow started drawing at {mach_type} ({pos}) - ({grid_x}, {grid_y})")
                        return
            for inv_type, grid_x, grid_y in self.inventories:
                cx = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
                cy = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
                dots = [
                    (cx, cy-30, "top"),
                    (cx-25, cy, "left"),
                    (cx+25, cy, "right"),
                    (cx, cy+30, "bottom")
                ]
                for dx, dy, pos in dots:
                    if abs(x - dx) < 6 and abs(y - dy) < 6:
                        dot_hit = True
                        if self.flow_mode and self.flow_start:
                            start_id, sx, sy, start_pos = self.flow_start
                            flow_id = f"FLOW{self.next_flow_id}"
                            self.next_flow_id += 1
                            priority = 1.0 if inv_type.startswith("WS") and start_id.startswith("WS") else 0.5
                            arrow = {
                                "start_id": start_id,
                                "start_pos": start_pos,
                                "end_id": inv_type,
                                "end_pos": pos,
                                "priority": priority
                            }
                            self.arrows[flow_id] = arrow
                            self.downstream_arrows.setdefault(start_id, []).append(arrow)
                            self.upstream_arrows.setdefault(inv_type, []).append(arrow)
                            print(f"Arrow finished connecting: {flow_id} - {start_id} ({start_pos}) to {inv_type} ({pos}) at ({grid_x}, {grid_y}) with priority {priority}")
                            self.flow_mode = False
                            self.flow_start = None
                            self.flow_preview = None
                            self.canvas.unbind("<Motion>")
                            self.canvas.bind("<Motion>", self.hover_check)
                            self.canvas.bind("<Button-3>", self.handle_right_click)  # Restore in flow
                            self.draw_layout()
                            self.canvas.bind("<Button-3>", self.handle_right_click)  # Extra safety
                        elif not self.flow_mode:
                            self.flow_mode = True
                            self.flow_start = (inv_type, dx, dy, pos)
                            self.flow_preview = (dx, dy)
                            self.canvas.bind("<Motion>", self.move_shape)
                            self.draw_layout()
                            self.log(f"Arrow started drawing at {inv_type} ({pos}) - ({grid_x}, {grid_y})")
                        return
            if self.flow_mode and not dot_hit:
                return
            for i, (shape_type, grid_x, grid_y) in enumerate(self.machines):
                cx = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
                cy = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
                state = self.machine_states.get(shape_type, 1)
                if abs(cx - x) < 40 and abs(cy - y) < 40:
                    if state != 4:
                        self.dragging = shape_type
                        self.visual_positions[shape_type] = (grid_x, grid_y)
                        self.is_new_item = False
                        self.selected_item = shape_type
                        self.update_info_board()
                        self.canvas.bind("<Motion>", self.move_shape)
                        self.draw_layout()
                        self.log(f"Dragging {shape_type} from ({grid_x}, {grid_y})")
                    else:
                        self.selected_item = shape_type
                        self.update_info_board()
                        self.draw_layout()
                        self.log(f"Selected {shape_type} at ({grid_x}, {grid_y}) - in S:4, no drag")
                    return
            for i, (shape_type, grid_x, grid_y) in enumerate(self.inventories):
                cx = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
                cy = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
                if abs(cx - x) < 50 and abs(cy - y) < 30:
                    self.dragging = shape_type
                    self.visual_positions[shape_type] = (grid_x, grid_y)
                    self.is_new_item = False
                    self.selected_item = shape_type
                    self.update_info_board()
                    self.canvas.bind("<Motion>", self.move_shape)
                    self.draw_layout()
                    self.log(f"Dragging {shape_type} from ({grid_x}, {grid_y})")
                    return

    def handle_release(self, event):
        if self.dragging and not self.is_new_item:
            new_x, new_y = self.visual_positions[self.dragging]
            new_x, new_y = int(new_x), int(new_y)  # Ensure ints
            if self.dragging.startswith("WS"):
                self.machines = [(m_id, new_x if m_id == self.dragging else x, new_y if m_id == self.dragging else y) 
                                for m_id, x, y in self.machines]
                print(f"Moved {self.dragging} to ({new_x}, {new_y})")
            else:
                self.inventories = [(i_id, new_x if i_id == self.dragging else x, new_y if i_id == self.dragging else y) 
                                   for i_id, x, y in self.inventories]
                print(f"Moved {self.dragging} to ({new_x}, {new_y})")
            del self.visual_positions[self.dragging]
            self.dragging = None
            self.canvas.unbind("<Motion>")
            self.canvas.bind("<Motion>", self.hover_check)
            self.draw_layout()
        elif self.dragging and self.is_new_item:
            if self.money >= 100:
                new_x, new_y = self.visual_positions[self.dragging]
                new_x, new_y = int(new_x), int(new_y)  # Ensure ints
                self.money -= 100
                if self.dragging.startswith("WS"):
                    self.machines.append((self.dragging, new_x, new_y))
                    self.machine_states[self.dragging] = 1
                    self.machine_cycles[self.dragging] = 0
                    print(f"Placed {self.dragging} at ({new_x}, {new_y}) - $100 deducted")
                else:
                    self.inventories.append((self.dragging, new_x, new_y))
                    if self.dragging not in self.inventory_stocks:
                        self.inventory_stocks[self.dragging] = {}
                    print(f"Placed {self.dragging} at ({new_x}, {new_y}) - $100 deducted")
                del self.visual_positions[self.dragging]
                self.is_new_item = False
                self.dragging = None
                self.flow_mode = False
                self.flow_start = None
                self.flow_preview = None
                self.canvas.unbind("<Motion>")
                self.canvas.bind("<Motion>", self.hover_check)
                self.update_status()
                self.draw_layout()
            else:
                print(f"Not enough money to place {self.dragging}!")
                self.cancel_purchase(event)
                
    def handle_right_click(self, event):
        self.log(f"Right-click attempted at ({event.x}, {event.y})")
        self.draw_layout()
        x, y = event.x, event.y

        if self.flow_mode and self.flow_start:
            start_id, sx, sy, start_pos = self.flow_start
            grid_x = sx // self.GRID_SIZE
            grid_y = sy // self.GRID_SIZE
            self.flow_mode = False
            self.flow_start = None
            self.flow_preview = None
            self.canvas.unbind("<Motion>")
            self.canvas.bind("<Motion>", self.hover_check)
            self.draw_layout()
            self.log(f"Arrow cancelled at {start_id} ({start_pos}) - ({grid_x}, {grid_y})")
            return

        for i, (shape_type, grid_x, grid_y) in enumerate(self.machines):
            cx = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
            cy = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
            if abs(cx - x) < 40 and abs(cy - y) < 40:
                self.machines = [(t, gx, gy) for t, gx, gy in self.machines if not (t == shape_type and gx == grid_x and gy == grid_y)]
                deleted_arrows = {k: v for k, v in self.arrows.items() if v["start_id"] == shape_type or v["end_id"] == shape_type}
                self.arrows = {k: v for k, v in self.arrows.items() if v["start_id"] != shape_type and v["end_id"] != shape_type}
                for arrow_id, arrow in deleted_arrows.items():
                    start_id = arrow["start_id"]
                    end_id = arrow["end_id"]
                    if start_id in self.downstream_arrows and arrow in self.downstream_arrows[start_id]:
                        self.downstream_arrows[start_id].remove(arrow)
                        if not self.downstream_arrows[start_id]:
                            del self.downstream_arrows[start_id]
                    if end_id in self.upstream_arrows and arrow in self.upstream_arrows[end_id]:
                        self.upstream_arrows[end_id].remove(arrow)
                        if not self.upstream_arrows[end_id]:
                            del self.upstream_arrows[end_id]
                self.last_deleted = ("machine", shape_type, grid_x, grid_y, deleted_arrows)
                print(f"Deleted {shape_type} at ({grid_x}, {grid_y}) with {len(deleted_arrows)} arrows")
                self.draw_layout()
                return

        for i, (shape_type, grid_x, grid_y) in enumerate(self.inventories):
            cx = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
            cy = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
            if abs(cx - x) < 50 and abs(cy - y) < 30:
                self.inventories = [(t, gx, gy) for t, gx, gy in self.inventories if not (t == shape_type and gx == grid_x and gy == grid_y)]
                deleted_arrows = {k: v for k, v in self.arrows.items() if v["start_id"] == shape_type or v["end_id"] == shape_type}
                self.arrows = {k: v for k, v in self.arrows.items() if v["start_id"] != shape_type and v["end_id"] != shape_type}
                for arrow_id, arrow in deleted_arrows.items():
                    start_id = arrow["start_id"]
                    end_id = arrow["end_id"]
                    if start_id in self.downstream_arrows and arrow in self.downstream_arrows[start_id]:
                        self.downstream_arrows[start_id].remove(arrow)
                        if not self.downstream_arrows[start_id]:
                            del self.downstream_arrows[start_id]
                    if end_id in self.upstream_arrows and arrow in self.upstream_arrows[end_id]:
                        self.upstream_arrows[end_id].remove(arrow)
                        if not self.upstream_arrows[end_id]:
                            del self.upstream_arrows[end_id]
                self.last_deleted = ("inventory", shape_type, grid_x, grid_y, deleted_arrows)
                print(f"Deleted {shape_type} at ({grid_x}, {grid_y}) with {len(deleted_arrows)} arrows")
                self.draw_layout()
                return

        for arrow_id, arrow_data in self.arrows.items():
            start_id = arrow_data["start_id"]
            end_id = arrow_data["end_id"]
            start_coords = None
            end_coords = None
            for shape_type, grid_x, grid_y in self.machines + self.inventories:
                cx = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
                cy = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
                if shape_type == start_id:
                    start_coords = {
                        "top": (cx, cy-40 if shape_type.startswith("WS") else cy-30),
                        "left": (cx-40 if shape_type.startswith("WS") else cx-25, cy),
                        "right": (cx+40 if shape_type.startswith("WS") else cx+25, cy),
                        "bottom": (cx, cy+40 if shape_type.startswith("WS") else cy+30)
                    }[arrow_data["start_pos"]]
                if shape_type == end_id:
                    end_coords = {
                        "top": (cx, cy-40 if shape_type.startswith("WS") else cy-30),
                        "left": (cx-40 if shape_type.startswith("WS") else cx-25, cy),
                        "right": (cx+40 if shape_type.startswith("WS") else cx+25, cy),
                        "bottom": (cx, cy+40 if shape_type.startswith("WS") else cy+30)
                    }[arrow_data["end_pos"]]
            if start_coords and end_coords:
                mid_x = (start_coords[0] + end_coords[0]) / 2
                mid_y = (start_coords[1] + end_coords[1]) / 2
                if abs(mid_x - x) < 10 and abs(mid_y - y) < 10:
                    arrow_data = self.arrows[arrow_id]
                    del self.arrows[arrow_id]
                    start_id = arrow_data["start_id"]
                    end_id = arrow_data["end_id"]
                    if start_id in self.downstream_arrows and arrow_data in self.downstream_arrows[start_id]:
                        self.downstream_arrows[start_id].remove(arrow_data)
                        if not self.downstream_arrows[start_id]:
                            del self.downstream_arrows[start_id]
                    if end_id in self.upstream_arrows and arrow_data in self.upstream_arrows[end_id]:
                        self.upstream_arrows[end_id].remove(arrow_data)
                        if not self.upstream_arrows[end_id]:
                            del self.upstream_arrows[end_id]
                    self.last_deleted = ("arrow", arrow_id, arrow_data)
                    print(f"Deleted arrow {arrow_id}")
                    self.draw_layout()
                    return

        self.log("No actionable object detected for deletion at coordinates")

    def handle_double_click(self, event):
        print(f"Double-click at ({event.x}, {event.y}), hover_shape={self.hover_shape}")
        x, y = event.x, event.y
        for shape_type, grid_x, grid_y in self.machines:
            cx = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
            cy = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
            if abs(cx - x) < 40 and abs(cy - y) < 40:
                self.configure_tool(event)
                print(f"Opened config for WS {shape_type} at ({grid_x}, {grid_y})")
                return
        for shape_type, grid_x, grid_y in self.inventories:
            cx = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
            cy = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
            if abs(cx - x) < 50 and abs(cy - y) < 30:
                self.configure_inventory(event)
                print(f"Opened config for INV {shape_type} at ({grid_x}, {grid_y})")
                return
        print(f"Double-click missed - no object hit at ({x}, {y})")

    def hover_check(self, event):
        x, y = event.x, event.y
        self.hover_shape = None
        for mach_type, grid_x, grid_y in self.machines:
            cx = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
            cy = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
            if abs(cx - x) < 40 and abs(cy - y) < 40:
                self.hover_shape = (mach_type, grid_x, grid_y)
                return
            dots = [
                (cx, cy-40, "top"),
                (cx-40, cy, "left"),
                (cx+40, cy, "right"),
                (cx, cy+40, "bottom")
            ]
            for dx, dy, pos in dots:
                if abs(x - dx) < 6 and abs(y - dy) < 6:
                    self.hover_shape = (mach_type, grid_x, grid_y)
                    return
        for inv_type, grid_x, grid_y in self.inventories:
            cx = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
            cy = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
            if abs(cx - x) < 50 and abs(cy - y) < 30:
                self.hover_shape = (inv_type, grid_x, grid_y)
                return
            dots = [
                (cx, cy-30, "top"),
                (cx-25, cy, "left"),
                (cx+25, cy, "right"),
                (cx, cy+30, "bottom")
            ]
            for dx, dy, pos in dots:
                if abs(x - dx) < 6 and abs(y - dy) < 6:
                    self.hover_shape = (inv_type, grid_x, grid_y)
                    return
        for arrow_id, arrow_data in self.arrows.items():
            start_id = arrow_data["start_id"]
            end_id = arrow_data["end_id"]
            start_coords = None
            end_coords = None
            for shape_type, grid_x, grid_y in self.machines + self.inventories:
                cx = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
                cy = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
                if shape_type == start_id:
                    start_coords = {
                        "top": (cx, cy-40 if shape_type.startswith("WS") else cy-30),
                        "left": (cx-40 if shape_type.startswith("WS") else cx-25, cy),
                        "right": (cx+40 if shape_type.startswith("WS") else cx+25, cy),
                        "bottom": (cx, cy+40 if shape_type.startswith("WS") else cy+30)
                    }[arrow_data["start_pos"]]
                if shape_type == end_id:
                    end_coords = {
                        "top": (cx, cy-40 if shape_type.startswith("WS") else cy-30),
                        "left": (cx-40 if shape_type.startswith("WS") else cx-25, cy),
                        "right": (cx+40 if shape_type.startswith("WS") else cx+25, cy),
                        "bottom": (cx, cy+40 if shape_type.startswith("WS") else cy+30)
                    }[arrow_data["end_pos"]]
            if start_coords and end_coords:
                mid_x = (start_coords[0] + end_coords[0]) / 2
                mid_y = (start_coords[1] + end_coords[1]) / 2
                if abs(mid_x - x) < 10 and abs(mid_y - y) < 10:
                    self.hover_shape = None
                    return
        self.last_hover_shape = self.hover_shape
        
    def update_info_board(self):
        # Default empty state
        self.info_object_id.config(text="Object ID: -")
        self.info_name.config(text="Tool: -")
        self.info_inputs.config(text="Inputs: -")
        self.info_outputs.config(text="Outputs: -")
        self.info_cycle.config(text="Cycle Time: -")
        self.info_costs.config(text="Running Costs: -")
        self.info_availability.config(text="Availability: -")
        self.info_fpy.config(text="FPY: -")
        self.info_throughput.config(text="Throughput Rate: -")
        self.info_cte.config(text="CTE: N/A")
        self.info_oee.config(text="OEE: -")
        self.info_wear.config(text="Wear: -")
        self.info_fault_chance.config(text="Fault Chance: -")
        self.info_held_materials.config(text="Held Materials: -")

        if self.selected_item:
            self.info_object_id.config(text=f"Object ID: {self.selected_item}")
            
            # Workstation (WS) Info
            if self.selected_item.startswith("WS"):
                tool = self.machine_tools.get(self.selected_item, {})
                state = self.machine_states.get(self.selected_item, 1)
                if tool:
                    self.info_name.config(text=f"Tool: {tool.get('name', self.selected_item)}")
                    self.info_inputs.config(text=f"Inputs: {', '.join(self.simplify_material_name(i) for i in tool.get('inputs', {}).keys())}")
                    self.info_outputs.config(text=f"Outputs: {', '.join(self.simplify_material_name(o) for o in tool.get('outputs', {}).keys())}")
                    cycle_time = tool.get("cycle_time", 0)
                    self.info_cycle.config(text=f"Cycle Time: {cycle_time}")
                    self.info_costs.config(text=f"Running Costs: Idle=${tool.get('costs', {}).get('idle', 0)}, Proc=${tool.get('costs', {}).get('processing', 0)}, Ready=${tool.get('costs', {}).get('ready', 0)}")

                    # WS Metrics (Last 100 Turns)
                    history = self.data_logger.history.get(self.selected_item, {})
                    state_history = history.get("state", [])  # [(turn, {"state": int}), ...]
                    output_history = history.get("output", [])  # [(turn, {"qualities": list, "count": int}), ...]
                    recent_state_history = state_history[-100:] if len(state_history) >= 100 else state_history
                    recent_output_history = output_history[-100:] if len(output_history) >= 100 else output_history
                    total_turns = len(recent_state_history)

                    if total_turns > 0:
                        # Availability: Turns in S:2 over last 100 turns
                        processing_turns = sum(1 for _, entry in recent_state_history if entry["state"] == 2)
                        availability = processing_turns / total_turns * 100
                        self.info_availability.config(text=f"Availability: {int(availability)}%")

                        # Performance: Actual cycles vs. ideal in S:2 time
                        actual_cycles = len([o for _, o in recent_output_history if o])
                        ideal_cycles = processing_turns / cycle_time if cycle_time > 0 else 0
                        performance = actual_cycles / ideal_cycles * 100 if ideal_cycles > 0 else 100
                        self.info_throughput.config(text=f"Throughput Rate: {actual_cycles / total_turns:.2f}/turn")

                        # Quality: Good units vs. total
                        total_output = sum(entry["count"] for _, entry in recent_output_history)
                        good_output = sum(entry["count"] for _, entry in recent_output_history if all(q >= 0.95 for q in entry["qualities"]))
                        quality = good_output / total_output * 100 if total_output > 0 else 100
                        self.info_fpy.config(text=f"FPY: {int(quality)}%")

                        # OEE
                        oee = (availability / 100) * (performance / 100) * (quality / 100) * 100
                        self.info_oee.config(text=f"OEE: {int(oee)}%")
                    else:
                        self.info_availability.config(text="Availability: N/A")
                        self.info_throughput.config(text="Throughput Rate: N/A")
                        self.info_fpy.config(text="FPY: N/A")
                        self.info_oee.config(text="OEE: N/A")

                    # Wear and Fault Chance
                    wear = self.breakdown_risk.get(self.selected_item, 0)
                    lifecycle = tool.get("lifecycle", 50)
                    self.info_wear.config(text=f"Wear: {wear}/{lifecycle} cycles")
                    wear_pct = wear / lifecycle if lifecycle > 0 else 0
                    fault_chance = 0 if wear_pct <= 0.5 else (wear_pct - 0.5) * 200
                    self.info_fault_chance.config(text=f"Fault Chance: {int(fault_chance)}%")
                else:
                    self.info_name.config(text="Tool: NONE")
                    wear = self.breakdown_risk.get(self.selected_item, 0)
                    lifecycle = self.machine_tools.get(self.selected_item, {}).get("lifecycle", 50)
                    self.info_wear.config(text=f"Wear: {wear}/{lifecycle} cycles")

            # Inventory (INV) Info
            elif self.selected_item.startswith("INV"):
                self.info_name.config(text="Tool: Inventory Rack")
                accepts = self.inventory_accepts.get(self.selected_item, [])
                self.info_inputs.config(text=f"Accepts: {', '.join(self.simplify_material_name(a) for a in accepts)}")
                self.info_outputs.config(text="Outputs: N/A")
                self.info_cycle.config(text="Cycle Time: N/A")
                self.info_costs.config(text="Running Costs: $0")

                # INV Metrics (Last 100 Turns)
                history = self.data_logger.history.get(self.selected_item, {})
                stock_history = history.get("inv_stock", [])  # [(turn, {"total_stock": int, "stock_by_type": dict}), ...]
                movement_in = history.get("inv_movement_in", [])  # [(turn, {"mat_id": str, "mat_type": str, "source": str}), ...]
                movement_out = history.get("inv_movement_out", [])  # [(turn, {"mat_id": str, "mat_type": str, "dest": str}), ...]
                blockage_history = history.get("inv_blockage", [])  # [(turn, {"downstream_id": str}), ...]
                recent_stock = stock_history[-100:] if len(stock_history) >= 100 else stock_history
                recent_in = movement_in[-100:] if len(movement_in) >= 100 else movement_in
                recent_out = movement_out[-100:] if len(movement_out) >= 100 else movement_out
                recent_blockages = blockage_history[-100:] if len(blockage_history) >= 100 else blockage_history

                total_turns = max(len(recent_stock), 1)  # Avoid division by zero
                # Availability: Always 100% for INV (no downtime)
                self.info_availability.config(text="Availability: 100%")
                
                # Turnover Rate: Items in/out per turn
                in_count = len(recent_in)
                out_count = len(recent_out)
                turnover = (in_count + out_count) / total_turns if total_turns > 0 else 0
                self.info_throughput.config(text=f"Turnover Rate: {turnover:.2f}/turn")

                # Blockage Frequency: % of turns blocked
                blockage_turns = len(set(t for t, _ in recent_blockages))  # Unique turns with blockages
                blockage_pct = (blockage_turns / total_turns * 100) if total_turns > 0 else 0
                self.info_fpy.config(text=f"Blockage Freq: {int(blockage_pct)}%")

                # OEE: N/A for INV, use as placeholder for avg stock
                avg_stock = sum(s["total_stock"] for _, s in recent_stock) / total_turns if recent_stock else 0
                self.info_oee.config(text=f"Avg Stock: {int(avg_stock)}")

                # Wear and Fault Chance: N/A for INV
                self.info_wear.config(text="Wear: N/A")
                self.info_fault_chance.config(text="Fault Chance: N/A")

            # Shared: Held Materials
            held_mats = self.materials_at_machine.get(self.selected_item, [])
            held_stocks = self.inventory_stocks.get(self.selected_item, {})
            if held_mats:
                mat_list = [
                    f"{self.material_dict.get(mat.state, {}).get('display', mat.state)} Q:{int(mat.quality * 100)}% {mat.id}"
                    for mat in held_mats
                ]
                self.info_held_materials.config(text=f"Held Materials: {', '.join(mat_list)}")
            elif held_stocks:
                mat_list = []
                for mat_type, mats in held_stocks.items():
                    for mat in mats:
                        quality = int(mat.quality * 100)
                        display_name = self.material_dict.get(mat_type, {}).get("display", mat_type)
                        mat_list.append(f"{display_name} Q:{quality}% {mat.id}")
                self.info_held_materials.config(text=f"Held Materials: {', '.join(mat_list)}")
            else:
                self.info_held_materials.config(text="Held Materials: None")
                    
    def delete_shape(self, grid_x, grid_y):
        for i, (shape_type, x, y) in enumerate(self.machines):
            if x == grid_x and y == grid_y:
                arrows_to_delete = [k for k, v in self.arrows.items() if v["start_id"] == shape_type or v["end_id"] == shape_type]
                self.last_deleted = {
                    "type": "machine",
                    "shape": (shape_type, x, y),
                    "arrows": {k: self.arrows[k] for k in arrows_to_delete},
                    "tool": self.machine_tools.get(shape_type, None),
                    "state": self.machine_states.get(shape_type, 1),
                    "cycles": self.machine_cycles.get(shape_type, 0),
                    "mats_at_machine": self.materials_at_machine.get(shape_type, {})
                }
                self.machines.pop(i)
                for arrow in arrows_to_delete:
                    del self.arrows[arrow]
                if shape_type in self.machine_tools:
                    del self.machine_tools[shape_type]
                if shape_type in self.machine_states:
                    del self.machine_states[shape_type]
                if shape_type in self.machine_cycles:
                    del self.machine_cycles[shape_type]
                if shape_type in self.materials_at_machine:
                    del self.materials_at_machine[shape_type]
                print(f"Deleted {shape_type} at ({grid_x}, {grid_y})")
                self.draw_layout()
                return
        for i, (shape_type, x, y) in enumerate(self.inventories):
            if x == grid_x and y == grid_y:
                arrows_to_delete = [k for k, v in self.arrows.items() if v["start_id"] == shape_type or v["end_id"] == shape_type]
                self.last_deleted = {
                    "type": "inventory",
                    "shape": (shape_type, x, y),
                    "arrows": {k: self.arrows[k] for k in arrows_to_delete},
                    "tool": None,
                    "state": None,
                    "cycles": None,
                    "mats_at_machine": {}
                }
                self.inventories.pop(i)
                for arrow in arrows_to_delete:
                    del self.arrows[arrow]
                if shape_type in self.inventory_stocks:
                    del self.inventory_stocks[shape_type]
                print(f"Deleted {shape_type} at ({grid_x}, {grid_y})")
                self.draw_layout()
                return
        print(f"No shape found to delete at ({grid_x}, {grid_y})")

    def delete_arrow(self, arrow_id):
        if arrow_id in self.arrows:
            self.last_deleted = {
                "type": "arrow",
                "arrow": (arrow_id, self.arrows[arrow_id])
            }
            del self.arrows[arrow_id]
            print(f"Deleted arrow {arrow_id}")
            self.draw_layout()

    def undo_delete(self):
        if self.last_deleted:
            del_type = self.last_deleted[0]
            if del_type == "machine":
                _, machine_type, grid_x, grid_y, deleted_arrows = self.last_deleted
                self.machines.append((machine_type, grid_x, grid_y))
                if machine_type not in self.machine_states:
                    self.machine_states[machine_type] = 1
                    self.machine_cycles[machine_type] = 0
                    self.materials_at_machine[machine_type] = []
                self.arrows.update(deleted_arrows)
                for arrow_id, arrow in deleted_arrows.items():
                    start_id = arrow["start_id"]
                    end_id = arrow["end_id"]
                    self.downstream_arrows.setdefault(start_id, []).append(arrow)
                    self.upstream_arrows.setdefault(end_id, []).append(arrow)
                print(f"Restored {machine_type} at ({grid_x}, {grid_y}) with {len(deleted_arrows)} arrows")
            elif del_type == "inventory":
                _, inv_type, grid_x, grid_y, deleted_arrows = self.last_deleted
                self.inventories.append((inv_type, grid_x, grid_y))
                if inv_type not in self.inventory_stocks:
                    self.inventory_stocks[inv_type] = {}
                self.arrows.update(deleted_arrows)
                for arrow_id, arrow in deleted_arrows.items():
                    start_id = arrow["start_id"]
                    end_id = arrow["end_id"]
                    self.downstream_arrows.setdefault(start_id, []).append(arrow)
                    self.upstream_arrows.setdefault(end_id, []).append(arrow)
                print(f"Restored {inv_type} at ({grid_x}, {grid_y}) with {len(deleted_arrows)} arrows")
            elif del_type == "arrow":
                _, flow_id, arrow_data = self.last_deleted
                self.arrows[flow_id] = arrow_data
                start_id = arrow_data["start_id"]
                end_id = arrow_data["end_id"]
                self.downstream_arrows.setdefault(start_id, []).append(arrow_data)
                self.upstream_arrows.setdefault(end_id, []).append(arrow_data)
                print(f"Restored arrow {flow_id}")
            self.last_deleted = None
            self.draw_layout()

    def save_layout(self):
        import os
        import json
        layout_file = os.path.join(LAYOUTS_DIR, "current.json")
        layout = {
            "scenario": self.scenario,  # Save scenario name
            "machines": [{"id": m[0], "x": m[1], "y": m[2]} for m in self.machines],
            "inventories": [{"id": i[0], "x": i[1], "y": i[2]} for i in self.inventories],
            "arrows": self.arrows,
            "machine_tools": {k: v for k, v in self.machine_tools.items()},
            "inventory_accepts": self.inventory_accepts,
            "machine_states": self.machine_states,
            "machine_cycles": self.machine_cycles,
            "game_time": self.game_time,
            "turns": self.turns,
            "money": self.money,
            "orders": self.orders,
            "deliveries": self.deliveries,
            "materials_at_machine": {
                m_id: [{"id": mat.id, "state": mat.state, "parent_ids": mat.parent_ids, "quality": mat.quality, "fault_detected": mat.fault_detected} 
                       for mat in mat_list]
                for m_id, mat_list in self.materials_at_machine.items()
            },
            "inventory_stocks": {
                inv_id: {mat_type: [{"id": mat.id, "quality": mat.quality, "fault_detected": mat.fault_detected} 
                                   for mat in mat_list] 
                         for mat_type, mat_list in stocks.items()}
                for inv_id, stocks in self.inventory_stocks.items()
            },
            "breakdown_risk": self.breakdown_risk,
            "alert_active": self.alert_active,
            "turns_since_alert": self.turns_since_alert,
            "maintenance_turns": self.maintenance_turns,
            "is_breakdown": self.is_breakdown,
            "is_maintenance": self.is_maintenance,
            "is_repair": self.is_repair,
            "is_paused": self.is_paused,
            "maintenance_queued": self.maintenance_queued
        }
        with open(layout_file, "w") as f:
            json.dump(layout, f, indent=4)
        print(f"Layout saved to {layout_file}")

    def save_named_layout(self):
        import os
        import json
        import tkinter.simpledialog
        layout_name = tkinter.simpledialog.askstring("Save Layout", "Enter layout name (e.g., car_flow_stable):", parent=self.root)
        if not layout_name:
            print("Save cancelled - no name provided")
            return
        layout_file = os.path.join(SAVED_DIR, f"{layout_name}.json")
        layout = {
            "scenario": self.scenario,  # Save scenario name
            "machines": [{"id": m[0], "x": m[1], "y": m[2]} for m in self.machines],
            "inventories": [{"id": i[0], "x": i[1], "y": i[2]} for i in self.inventories],
            "arrows": self.arrows,
            "machine_tools": {k: v for k, v in self.machine_tools.items()},
            "inventory_accepts": self.inventory_accepts,
            "machine_states": self.machine_states,
            "machine_cycles": self.machine_cycles,
            "game_time": self.game_time,
            "turns": self.turns,
            "money": self.money,
            "orders": self.orders,
            "deliveries": self.deliveries,
            "materials_at_machine": {
                m_id: [{"id": mat.id, "state": mat.state, "parent_ids": mat.parent_ids, "quality": mat.quality, "fault_detected": mat.fault_detected} 
                       for mat in mat_list]
                for m_id, mat_list in self.materials_at_machine.items()
            },
            "inventory_stocks": {
                inv_id: {mat_type: [{"id": mat.id, "quality": mat.quality, "fault_detected": mat.fault_detected} 
                                   for mat in mat_list] 
                         for mat_type, mat_list in stocks.items()}
                for inv_id, stocks in self.inventory_stocks.items()
            },
            "breakdown_risk": self.breakdown_risk,
            "alert_active": self.alert_active,
            "turns_since_alert": self.turns_since_alert,
            "maintenance_turns": self.maintenance_turns,
            "is_breakdown": self.is_breakdown,
            "is_maintenance": self.is_maintenance,
            "is_repair": self.is_repair,
            "is_paused": self.is_paused,
            "maintenance_queued": self.maintenance_queued
        }
        with open(layout_file, "w") as f:
            json.dump(layout, f, indent=4)
        print(f"Layout saved as {layout_file}")

    def quit_game(self):
        self.save_layout()
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = f"data/models/model_{timestamp}.h5"
        os.makedirs("data/models", exist_ok=True)
        print(f"Saving model to: {os.path.abspath(save_path)}")
        self.nn_agent.save_model(save_path)
        self.root.destroy()
        exit()
        print("Game closed")

    def back_to_launcher(self):
        self.save_layout()  # Save to current.json
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = f"data/models/model_{timestamp}.h5"
        os.makedirs("data/models", exist_ok=True)
        print(f"Saving model to: {os.path.abspath(save_path)}")
        self.nn_agent.save_model(save_path)
        self.root.destroy()  # Close game window
        if self.launcher_root:
            self.launcher_root.deiconify()  # Show launcher directly
            self.launcher_root.update_idletasks()  # Process UI events
            self.launcher_root.lift()  # Bring to front
        print("Game closed, launcher should reappear")
        
    def show_instructions(self):
        instructions_file = os.path.join(BASE_DIR, "data", "master", "instructions.txt")
        try:
            with open(instructions_file, "r") as f:
                instructions_text = f.read()
            instr_window = tk.Toplevel(self.root)
            instr_window.title("Instructions")
            instr_window.geometry("900x650")
            text_widget = tk.Text(instr_window, wrap=tk.WORD, height=15, width=50)
            text_widget.insert(tk.END, instructions_text)
            text_widget.config(state=tk.DISABLED)  # Read-only
            text_widget.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
            tk.Button(instr_window, text="Close", command=instr_window.destroy, width=10).pack(pady=5)
        except FileNotFoundError:
            print(f"Instructions file not found at {instructions_file}")
            error_window = tk.Toplevel(self.root)
            error_window.title("Error")
            tk.Label(error_window, text="Instructions file not found!", fg="red").pack(pady=10)
            tk.Button(error_window, text="Close", command=error_window.destroy).pack(pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    game = FactoryGame(root)
    root.mainloop()