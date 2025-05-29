import json
import csv
import os

class DataLogger:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.history = {}  # {obj_id: {metric: [(turn, value_dict)]}}

    def _initialize_object(self, obj_id, obj_type):
        """Initialize history for an object if not already present."""
        if obj_id not in self.history:
            self.history[obj_id] = {
                "state": [],              # (turn, {"state": int})
                "cycle_time": [],         # (turn, {"actual_duration": int})
                "downtime": [],           # (turn, {"reason": str})
                "shortage": [],           # (turn, {"shortage_flag": bool, "missing_inputs": list})
                "maintenance": [],        # (turn, {"event_type": str, "duration": int})
                "queue": [],              # (turn, {"queue_duration": int})
                "cost": [],               # (turn, {"cost": float})
                "vibration_level": [],    # (turn, {"vibration_level": float})
                "output": [],             # (turn, {"qualities": list, "count": int})
                "inv_stock": [],          # (turn, {"total_stock": int, "stock_by_type": dict})
                "inv_movement_in": [],    # (turn, {"mat_id": str, "mat_type": str, "source": str})
                "inv_movement_out": [],   # (turn, {"mat_id": str, "mat_type": str, "dest": str})
                "inv_blockage": []        # (turn, {"downstream_id": str})
            }
        # Set object_type once during initialization
        self.history[obj_id]["object_type"] = obj_type

    def _trim_history(self, history_list):
        """Trim a specific metric’s history to window_size."""
        if len(history_list) > self.window_size:
            history_list[:] = history_list[-self.window_size:]

    def log_state(self, turn, object_id, object_type, state):
        self._initialize_object(object_id, object_type)
        entry = {"state": state}
        self.history[object_id]["state"].append((turn, entry))
        self._trim_history(self.history[object_id]["state"])

    def log_cycle_time(self, turn, object_id, object_type, actual_duration):
        self._initialize_object(object_id, object_type)
        entry = {"actual_duration": actual_duration}
        self.history[object_id]["cycle_time"].append((turn, entry))
        self._trim_history(self.history[object_id]["cycle_time"])

    def log_downtime(self, turn, object_id, object_type, reason):
        self._initialize_object(object_id, object_type)
        entry = {"reason": reason}
        self.history[object_id]["downtime"].append((turn, entry))
        self._trim_history(self.history[object_id]["downtime"])

    def log_shortage(self, turn, object_id, object_type, shortage_flag, missing_inputs):
        """Log material shortages blocking S:1 to S:2."""
        self._initialize_object(object_id, object_type)
        entry = {"shortage_flag": shortage_flag, "missing_inputs": missing_inputs}
        self.history[object_id]["shortage"].append((turn, entry))
        self._trim_history(self.history[object_id]["shortage"])

    def log_maintenance(self, turn, object_id, object_type, event_type, duration):
        """Log maintenance events (start/end) with duration."""
        self._initialize_object(object_id, object_type)
        entry = {"event_type": event_type, "duration": duration}
        self.history[object_id]["maintenance"].append((turn, entry))
        self._trim_history(self.history[object_id]["maintenance"])

    def log_queue(self, turn, object_id, object_type, queue_duration):
        """Log time spent in S:3 waiting to unload."""
        self._initialize_object(object_id, object_type)
        entry = {"queue_duration": queue_duration}
        self.history[object_id]["queue"].append((turn, entry))
        self._trim_history(self.history[object_id]["queue"])

    def log_cost(self, turn, object_id, object_type, cost):
        """Log energy consumption cost per turn."""
        self._initialize_object(object_id, object_type)
        entry = {"cost": cost}
        self.history[object_id]["cost"].append((turn, entry))
        self._trim_history(self.history[object_id]["cost"])

    def log_vibration(self, turn, object_id, object_type, vibration_level):
        """Log simulated vibration level for predictive maintenance."""
        self._initialize_object(object_id, object_type)
        entry = {"vibration_level": vibration_level}
        self.history[object_id]["vibration_level"].append((turn, entry))
        self._trim_history(self.history[object_id]["vibration_level"])

    def log_output(self, turn, object_id, object_type, qualities, count):
        """Log WS output (qualities and count) when a cycle completes."""
        self._initialize_object(object_id, object_type)
        entry = {"qualities": qualities, "count": count}
        self.history[object_id]["output"].append((turn, entry))
        self._trim_history(self.history[object_id]["output"])

    def log_inv_stock(self, turn, inv_id, total_stock, stock_by_type):
        """Log total stock and breakdown by material type for an inventory."""
        self._initialize_object(inv_id, "INV")
        entry = {"total_stock": total_stock, "stock_by_type": stock_by_type}
        self.history[inv_id]["inv_stock"].append((turn, entry))
        self._trim_history(self.history[inv_id]["inv_stock"])

    def log_inv_movement(self, turn, inv_id, direction, mat_id, mat_type, source=None, dest=None):
        """Log material entering or leaving an inventory, storing in/out separately."""
        self._initialize_object(inv_id, "INV")
        entry = {"mat_id": mat_id, "mat_type": mat_type, "source": source, "dest": dest}
        key = "inv_movement_in" if direction == "in" else "inv_movement_out"
        self.history[inv_id][key].append((turn, entry))
        self._trim_history(self.history[inv_id][key])

    def log_inv_blockage(self, turn, inv_id, downstream_id):
        """Log when an inventory blocks a WS due to capacity."""
        self._initialize_object(inv_id, "INV")
        entry = {"downstream_id": downstream_id}
        self.history[inv_id]["inv_blockage"].append((turn, entry))
        self._trim_history(self.history[inv_id]["inv_blockage"])

    def export_to_csv(self, start_turn, end_turn, output_dir="logs"):
        """Export all histories to a CSV file for the given turn range, consolidated by turn and object."""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/history_{start_turn}-{end_turn}.csv"
        
        # Define CSV headers with split in/out columns
        headers = [
            "turn", "object_id", "object_type", "state", "downtime_reason", "shortage_flag", "missing_inputs",
            "maintenance_event", "maintenance_duration", "queue_duration", "cost", "vibration_level",
            "output_qualities", "output_count", "cycle_time", "total_stock", "stock_by_type",
            "mat_id_in", "mat_type_in", "source_in", "mat_id_out", "mat_type_out", "dest_out",
            "blocked_downstream_id"
        ]
        
        # Merge all histories into a dict keyed by (turn, object_id)
        merged_data = {}
        all_objects = self.history.keys()
        
        for obj_id in all_objects:
            obj_type = self.history[obj_id]["object_type"]
            for metric, entries in self.history[obj_id].items():
                if metric == "object_type":
                    continue
                for turn_val, entry in entries:
                    if start_turn <= turn_val <= end_turn:
                        key = (turn_val, obj_id)
                        merged_data.setdefault(key, {"turn": turn_val, "object_id": obj_id, "object_type": obj_type})
                        if metric == "state":
                            merged_data[key]["state"] = entry["state"]
                        elif metric == "cycle_time":
                            merged_data[key]["cycle_time"] = entry["actual_duration"]
                        elif metric == "downtime":
                            merged_data[key]["downtime_reason"] = entry["reason"]
                        elif metric == "shortage":
                            merged_data[key]["shortage_flag"] = entry["shortage_flag"]
                            merged_data[key]["missing_inputs"] = json.dumps(entry["missing_inputs"])
                        elif metric == "maintenance":
                            merged_data[key]["maintenance_event"] = entry["event_type"]
                            merged_data[key]["maintenance_duration"] = entry["duration"]
                        elif metric == "queue":
                            merged_data[key]["queue_duration"] = entry["queue_duration"]
                        elif metric == "cost":
                            merged_data[key]["cost"] = entry["cost"]
                        elif metric == "vibration_level":
                            merged_data[key]["vibration_level"] = entry["vibration_level"]
                        elif metric == "output":
                            merged_data[key]["output_qualities"] = json.dumps(entry["qualities"])
                            merged_data[key]["output_count"] = entry["count"]
                        elif metric == "inv_stock":
                            merged_data[key]["total_stock"] = entry["total_stock"]
                            merged_data[key]["stock_by_type"] = json.dumps(entry["stock_by_type"])
                        elif metric == "inv_movement_in":
                            merged_data[key]["mat_id_in"] = entry["mat_id"]
                            merged_data[key]["mat_type_in"] = entry["mat_type"]
                            merged_data[key]["source_in"] = entry["source"]
                        elif metric == "inv_movement_out":
                            merged_data[key]["mat_id_out"] = entry["mat_id"]
                            merged_data[key]["mat_type_out"] = entry["mat_type"]
                            merged_data[key]["dest_out"] = entry["dest"]
                        elif metric == "inv_blockage":
                            merged_data[key]["blocked_downstream_id"] = entry["downstream_id"]
        
        # Convert to list and sort by turn, then object_id
        rows = list(merged_data.values())
        rows.sort(key=lambda x: (x["turn"], x["object_id"]))
        
        # Write to CSV
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                # Fill missing fields with None or empty strings
                for h in headers:
                    row.setdefault(h, None if h in ["turn", "cost", "vibration_level", "output_count", "cycle_time", "total_stock", "maintenance_duration", "queue_duration"] else "")
                writer.writerow(row)
        
        print(f"Exported consolidated history for turns {start_turn}-{end_turn} to {filename}")