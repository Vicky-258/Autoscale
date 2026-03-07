import numpy as np
import torch
import matplotlib.pyplot as plt

from config.settings import DEVICE, MODEL_PATH, SCALER_PATH
from predictor.gru_model import GRUModel
from predictor.predictor import GRUPredictor


# -------------------------
# Interval violation logic
# -------------------------
def interval_violation(y, low, high):
    return 0 if low <= y <= high else 1


# -------------------------
# Standalone Visualization
# -------------------------
if __name__ == "__main__":
    X = np.load("data/processed/X.npy")   # (N, 30)
    plt.show()
