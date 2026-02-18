# config/settings.py

from pathlib import Path
import torch


# =========================
# Project Root
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent


# =========================
# Paths
# =========================

DATA_RAW_PATH = BASE_DIR / "data/raw/rps_raw.npy"
X_PATH = BASE_DIR / "data/processed/X.npy"
Y_PATH = BASE_DIR / "data/processed/y.npy"
SCALER_PATH = BASE_DIR / "data/processed/scaler.pkl"
ARTIFACT_DIR = Path("performance/artifacts")

MODEL_PATH = BASE_DIR / "predictor/weights/gru_v1.pt"

CALIBRATION_PATH = BASE_DIR / "uncertainty/calibration.json"


# =========================
# Model Settings
# =========================

WINDOW_SIZE = 30
HORIZON = 12
TRAIN_SPLIT = 0.8
PERCENTILE = 97.5


# =========================
# Autoscaler Settings
# =========================

DEPLOYMENT_NAME = "Autoscaler"
LOOP_INTERVAL = 30

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"