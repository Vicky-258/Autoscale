import time
import torch
from datetime import datetime
import json

from autoscaler.policy import decide_replicas, ScalingState
from predictor.models.gru_model import GRUModel
from predictor.predictor import GRUPredictor
from burst_detector.detector import BurstDetector
from control.k8_scaler import (
    load_config,
    get_current_replicas,
    set_replicas
)
from config.settings import (
    DEPLOYMENT_NAME,
    LOOP_INTERVAL,
    DEVICE,
    MODEL_PATH,
    SCALER_PATH,
    CALIBRATION_PATH,
)

# --- Load Calibration ---
with open(CALIBRATION_PATH) as f:
    calibration = json.load(f)

RAW_BOUNDS = calibration["raw_bounds"]

# We use t+1 only
ERROR_BOUND = RAW_BOUNDS[0]

print(f"📊 Loaded calibrated error bound (t+1): ±{ERROR_BOUND:.2f} RPS")


# =========================
# Utilities (Replace Later)
# =========================

def get_latest_rps():
    """
    Replace with real metric ingestion.
    """
    import random
    return random.uniform(200, 800)


def build_input_tensor():
    """
    Replace with real sliding window of last 30 RPS values.
    Shape: (1, 30, 1)
    """
    return torch.randn(1, 30, 1)


# =========================
# System State Resolver
# =========================

def resolve_system_state(burst_state: str, model_alive: bool) -> str:
    """
    Map burst classifier + model health to system state.
    """

    if not model_alive:
        return "UNCERTAIN"

    if burst_state == "BURST":
        return "BURST"

    if burst_state == "PERIODIC_SPIKE":
        return "UNCERTAIN"

    return "NORMAL"


# =========================
# Main Control Loop
# =========================

def run_autoscaler():

    load_config()

    # --- Load Model ---
    model = GRUModel()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.eval()

    predictor = GRUPredictor(model, scaler_path=SCALER_PATH, device=DEVICE)

    # --- Burst Detector ---
    burst_detector = BurstDetector()

    # --- Scaling Memory ---
    state = ScalingState()

    print("🚀 AutoScale (Day 5) started...")

    while True:

        now = datetime.utcnow()

        # -------------------
        # OBSERVE
        # -------------------
        current_rps = get_latest_rps()
        x_input = build_input_tensor()

        # --- Prediction with safety ---
        try:
            forecast = predictor.predict_next_12(x_input)
            predicted_rps = forecast[0].item()
            model_alive = True
        except Exception as e:
            print("⚠ Predictor failure:", e)
            predicted_rps = current_rps
            model_alive = False

        # --- Basic interval (replace later with residual-based module) ---
        lower = predicted_rps - ERROR_BOUND
        upper = predicted_rps + ERROR_BOUND

        burst_state, explanation = burst_detector.update(
            actual=current_rps,
            lower=lower,
            upper=upper
        )

        system_state = resolve_system_state(burst_state, model_alive)

        current_replicas = get_current_replicas(DEPLOYMENT_NAME)

        if current_replicas is None:
            print("⚠ Could not fetch replicas. Skipping cycle.")
            time.sleep(LOOP_INTERVAL)
            continue

        # -------------------
        # DECIDE
        # -------------------

        if system_state == "NORMAL":

            desired = decide_replicas(
                predicted_rps=predicted_rps,
                current_replicas=current_replicas,
                is_bursting=False,
                burst_intensity=0.0,
                state=state,
                now=now
            )

        elif system_state == "BURST":

            desired = decide_replicas(
                predicted_rps=predicted_rps,
                current_replicas=current_replicas,
                is_bursting=True,
                burst_intensity=0.5,
                state=state,
                now=now
            )

        elif system_state == "UNCERTAIN":

            # Conservative logic:
            # Never scale below current demand
            conservative_pred = max(predicted_rps, current_rps)

            desired = decide_replicas(
                predicted_rps=conservative_pred,
                current_replicas=current_replicas,
                is_bursting=False,
                burst_intensity=0.0,
                state=state,
                now=now
            )

        # -------------------
        # ACT
        # -------------------

        if desired != current_replicas:

            print(
                f"[{now}] "
                f"State={system_state} | "
                f"Scaling {current_replicas} → {desired} | "
                f"Pred={predicted_rps:.2f} | "
                f"Actual={current_rps:.2f} | "
                f"Burst={burst_state}"
            )

            set_replicas(DEPLOYMENT_NAME, desired)

        else:

            print(
                f"[{now}] "
                f"State={system_state} | "
                f"Stable | Replicas={current_replicas} | "
                f"Pred={predicted_rps:.2f} | "
                f"Actual={current_rps:.2f} | "
                f"Burst={burst_state}"
            )

        time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    run_autoscaler()