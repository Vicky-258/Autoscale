# config/settings.py

from pathlib import Path
import json
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
# Burst Detection Settings
# =========================
BURST_WINDOW_SIZE = 20
BURST_RUN_RATIO = 0.3
DRIFT_RUN_RATIO = 0.3
BURST_PERSISTENCE = 3
RECOVERY_TIME = 3



# =========================
# Autoscaler Settings
# =========================

DEPLOYMENT_NAME = "Autoscaler"
LOOP_INTERVAL = 30

MIN_REPLICAS = 2
MAX_REPLICAS = 20
MAX_SCALE_UP_STEP = 3
MAX_SCALE_DOWN_STEP = 2

SCALE_UP_COOLDOWN = 30
SCALE_DOWN_COOLDOWN = 120
BURST_COOLDOWN_OVERRIDE = 10

CAPACITY_PER_POD = 100
MAX_BURST_BOOST = 1.0


# =========================
# System Constants
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import sys
import json

ERROR_BOUND = 0.0

if CALIBRATION_PATH.exists():
    with open(CALIBRATION_PATH) as _f:
        _calibration = json.load(_f)
    ERROR_BOUND = _calibration["raw_bounds"][0]
else:
    # If not running calibrate right now, print a warning
    if "calibrate" not in sys.argv[0] and "calibrate" not in getattr(sys.modules.get("__main__"), "__file__", ""):
        print(f"⚠ Calibration file missing: {CALIBRATION_PATH}")
        print("⚠ Please run: python -m uncertainty.calibrate")