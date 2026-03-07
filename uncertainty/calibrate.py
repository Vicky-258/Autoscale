# uncertainty/calibrate.py

import json
import numpy as np
import torch
import joblib
from pathlib import Path

from config.settings import SCALER_PATH, MODEL_PATH, X_PATH, Y_PATH, CALIBRATION_PATH
from main import DEVICE
from predictor.gru_model import GRUModel
from predictor.predictor import GRUPredictor


WINDOW_SIZE = 30
HORIZON = 12
TRAIN_SPLIT = 0.8
PERCENTILE = 97.5


def calibrate(
    model_path: str,
    scaler_path: str,
    X_path: str,
    y_path: str,
    output_path: str,
):
    print("🔍 Starting uncertainty calibration...")

    # -------------------------
    # Load data
    # -------------------------
    X = np.load(X_path)
    y = np.load(y_path)

    N = X.shape[0]
    split = int(TRAIN_SPLIT * N)

    X_val = X[split:]
    y_val = y[split:]

    print(f"Validation samples: {X_val.shape[0]}")

    # -------------------------
    # Load model
    # -------------------------
    model = GRUModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    predictor = GRUPredictor(model, scaler_path=SCALER_PATH, device=DEVICE)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)

    with torch.no_grad():
        preds = predictor.predict_next_12(X_val_t)

    preds_np = preds.numpy()

    # -------------------------
    # Residuals (Raw RPS differences)
    # -------------------------
    scaler = joblib.load(scaler_path)
    scale_range = float(scaler.data_max_[0] - scaler.data_min_[0])
    
    # y_val is currently normalized, shape (N, 12). Inverse transform it.
    y_val_raw = scaler.inverse_transform(y_val)
    
    # Calculate absolute error between raw predicted RPS and raw actual RPS
    residuals = np.abs(y_val_raw - preds_np)

    # Compute percentiles directly on the absolute raw errors
    raw_bounds = [
        float(np.percentile(residuals[:, h], PERCENTILE))
        for h in range(HORIZON)
    ]

    # -------------------------
    # Save calibration file
    # -------------------------
    calibration_data = {
        "percentile": PERCENTILE,
        "scale_range": scale_range,
        "min_rps": float(scaler.data_min_[0]),
        "max_rps": float(scaler.data_max_[0]),
        "raw_bounds": raw_bounds,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(calibration_data, f, indent=4)

    print("Calibration complete.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    calibrate(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        X_path=X_PATH,
        y_path=Y_PATH,
        output_path=CALIBRATION_PATH,
    )