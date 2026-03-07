import collections
import torch
import math
import random
import joblib
import numpy as np
import sys
import os

# Ensure config can be reached
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SCALER_PATH

class TrafficGenerator:
    """
    Generates synthetic real-time Request-Per-Second (RPS) metrics
    to simulate traffic for the Autoscaler.
    """
    def __init__(self, window_size=30, scaler_path=SCALER_PATH):
        self.window_size = window_size
        self.history = collections.deque(maxlen=window_size)
        self.tick_count = 0
        self.scaler = joblib.load(scaler_path)

        # Load actual validation dataset to perfectly match model distribution
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_path = os.path.join(root, "data/raw/rps_raw.npy")
        self.real_data = np.load(raw_path).flatten()
        self.start_idx = int(len(self.real_data) * 0.8) # Test split
        
        # Pre-fill history to avoid empty tensor errors during warm-up
        for t in range(-window_size, 0):
            self.history.append(self.real_data[self.start_idx + t])
        
    def get_next(self):
        """Yields the actual RPS for the current simulation tick."""
        self.tick_count += 1
        
        # Get actual historical RPS
        try:
            base = float(self.real_data[self.start_idx + self.tick_count])
        except IndexError:
            base = 200.0 # safety fallback
        
        # Inject an unexpected burst to test the BurstDetector
        # Burst from tick 50 to 90
        if 50 <= self.tick_count <= 90:
            current_rps = base + 300
        else:
            current_rps = base
            
        self.history.append(current_rps)
        return current_rps
        
    def get_tensor(self):
        """Returns the last `window_size` elements as a (1, W, 1) scaled tensor."""
        data = list(self.history)
        
        # Fallback padding if needed
        if len(data) < self.window_size:
            data = [data[0]] * (self.window_size - len(data)) + data
            
        # Scale data using the saved scaler (expects 2D array)
        data_np = np.array(data).reshape(-1, 1)
        data_scaled = self.scaler.transform(data_np)
            
        tensor = torch.tensor(data_scaled, dtype=torch.float32).view(1, self.window_size, 1)
        return tensor
