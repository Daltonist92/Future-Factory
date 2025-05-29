from sklearn.svm import OneClassSVM
import numpy as np
from collections import deque

class OCSVMPredictor:
    def __init__(self, nu=0.05, kernel="rbf", gamma="scale"):
        """Initialize OC-SVM predictor."""
        self.ocsvm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)  # nu=0.05 allows 5% outliers
        self.trained = False
        self.window_size = 10  # Lookback window for anomalies
        self.alert_threshold = 3  # 3+ anomalies in window to alert
        self.vibration_history = {}  # Per WS: deque of (turn, rms, anomaly)

    def train(self, vibration_data):
        """Train OC-SVM on healthy vibration data.
        
        Args:
            vibration_data (dict): {ws_id: list of RMS values}
        """
        # Combine all WS data for a global healthy baseline
        all_data = []
        for ws_id, rms_list in vibration_data.items():
            if rms_list:  # Only include non-empty lists
                all_data.extend(rms_list)
                # Initialize history deque for this WS
                if ws_id not in self.vibration_history:
                    self.vibration_history[ws_id] = deque(maxlen=self.window_size)
        
        if not all_data:
            print("No vibration data to train OC-SVM!")
            return
        
        # Train on healthy data (assumes input is healthy, e.g., 0–50% wear)
        X = np.array(all_data).reshape(-1, 1)  # 2D array for sklearn
        self.ocsvm.fit(X)
        self.trained = True
        print(f"OC-SVM trained on {len(all_data)} healthy vibration samples.")

    def predict(self, ws_id, turn, rms):
        """Predict if current RMS is anomalous and check for alert.
        
        Args:
            ws_id (str): Workstation ID (e.g., 'WS1')
            turn (int): Current turn number
            rms (float): Vibration RMS value
            
        Returns:
            bool: True if alert triggered (3+ anomalies in window)
        """
        if not self.trained:
            return False
        
        # Skip if machine is off (RMS = 0)
        if rms == 0:
            return False
        
        # Predict anomaly (1 = inlier, -1 = outlier)
        X = np.array([[rms]])  # 2D array for sklearn
        is_anomaly = self.ocsvm.predict(X)[0] == -1
        
        # Update history
        if ws_id not in self.vibration_history:
            self.vibration_history[ws_id] = deque(maxlen=self.window_size)
        self.vibration_history[ws_id].append((turn, rms, is_anomaly))
        
        # Check window for alert
        window = list(self.vibration_history[ws_id])
        anomaly_count = sum(1 for _, _, anomaly in window if anomaly)
        return anomaly_count >= self.alert_threshold

    def reset(self, ws_id):
        """Reset history for a workstation (e.g., after repair)."""
        if ws_id in self.vibration_history:
            self.vibration_history[ws_id].clear()