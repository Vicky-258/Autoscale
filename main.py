import time
import torch
from datetime import datetime, timezone
import math

from autoscaler.policy import decide_replicas, ScalingState
from burst_detection.states import BurstState
from autoscaler.states import SystemState
from predictor.gru_model import GRUModel
from predictor.predictor import GRUPredictor
from burst_detection.detector import BurstDetector
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
    ERROR_BOUND,
)
from config.logger import setup_logger

logger = setup_logger("autoscale.main")

logger.info(f"Loaded calibrated error bound (t+1): ±{ERROR_BOUND:.2f} RPS")


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

def resolve_system_state(burst_state: BurstState, model_alive: bool) -> SystemState:
    """
    Map burst classifier + model health to system state.
    """

    if not model_alive:
        return SystemState.UNCERTAIN

    if burst_state == BurstState.BURST:
        return SystemState.BURST

    if burst_state == BurstState.PERIODIC_SPIKE:
        return SystemState.UNCERTAIN

    return SystemState.NORMAL


# =========================
# Main Control Loop
# =========================

def run_autoscaler(
    get_rps_fn,
    get_tensor_fn,
    get_replicas_fn,
    set_replicas_fn,
    sleep_fn=time.sleep,
    get_time_fn=lambda: datetime.now(timezone.utc),
    metrics_callback=None
):

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
    state_file = "scaling_state.json"
    state = ScalingState.load(state_file)

    print("AutoScale started...")

    import traceback
    
    while True:
    
        try:
    
            now = get_time_fn()
    
            # -------------------
            # OBSERVE
            # -------------------
            current_rps = get_rps_fn()
            x_input = get_tensor_fn()
    
            # --- Prediction with safety ---
            try:
                forecast = predictor.predict_next_12(x_input)
                predicted_rps = forecast[0, 0].item()
                if not math.isfinite(predicted_rps):
                    raise ValueError("Model predicted NaN or Inf")
                model_alive = True
            except Exception as e:
                logger.error(f"⚠ Predictor failure: {e}")
                predicted_rps = current_rps
                model_alive = False
    
            lower = predicted_rps - ERROR_BOUND
            upper = predicted_rps + ERROR_BOUND
    
            burst_state, explanation = burst_detector.update(
                actual=current_rps,
                lower=lower,
                upper=upper,
                is_valid=model_alive
            )
    
            system_state = resolve_system_state(burst_state, model_alive)
    
            current_replicas = get_replicas_fn(DEPLOYMENT_NAME)
    
            if current_replicas is None:
                logger.warning("⚠ Could not fetch replicas. Skipping cycle.")
                sleep_fn(LOOP_INTERVAL)
                continue
    
            # -------------------
            # DECIDE
            # -------------------
            if system_state == SystemState.NORMAL:
    
                desired = decide_replicas(
                    predicted_rps=predicted_rps,
                    current_replicas=current_replicas,
                    is_bursting=False,
                    burst_intensity=0.0,
                    state=state,
                    now=now
                )
    
            elif system_state == SystemState.BURST:
    
                desired = decide_replicas(
                    predicted_rps=predicted_rps,
                    current_replicas=current_replicas,
                    is_bursting=True,
                    burst_intensity=0.5,
                    state=state,
                    now=now
                )
    
            elif system_state == SystemState.UNCERTAIN:
    
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
    
                logger.info(
                    f"State={system_state.name} | "
                    f"Scaling {current_replicas} → {desired} | "
                    f"Pred={predicted_rps:.2f} | "
                    f"Actual={current_rps:.2f} | "
                    f"Burst={burst_state.name}"
                )
    
                scaled = set_replicas_fn(DEPLOYMENT_NAME, desired)
                if scaled is not None:
                    from autoscaler.states import ScalingDirection
                    state.last_scale_time = now
                    state.last_scale_direction = ScalingDirection.UP if desired > current_replicas else ScalingDirection.DOWN
                    state.save(state_file)
                else:
                    logger.warning("⚠ Scaling request failed in API. Ghost scaling prevented.")
    
            else:
    
                logger.info(
                    f"State={system_state.name} | "
                    f"Stable | Replicas={current_replicas} | "
                    f"Pred={predicted_rps:.2f} | "
                    f"Actual={current_rps:.2f} | "
                    f"Burst={burst_state.name}"
                )
                
            if metrics_callback:
                metrics_callback({
                    'actual_rps': current_rps,
                    'predicted_rps': predicted_rps,
                    'lower_bound': lower,
                    'upper_bound': upper,
                    'burst_state': burst_state.name,
                    'system_state': system_state.name,
                    'current_replicas': current_replicas,
                    'desired_replicas': desired
                })
    
        except Exception as e:
    
            logger.error(f"🚨 Control loop failure: {e}")
            traceback.print_exc()
    
        finally:
    
            sleep_fn(LOOP_INTERVAL)


if __name__ == "__main__":
    run_autoscaler(
        get_rps_fn=get_latest_rps,
        get_tensor_fn=build_input_tensor,
        get_replicas_fn=get_current_replicas,
        set_replicas_fn=set_replicas,
        sleep_fn=time.sleep,
        get_time_fn=lambda: datetime.now(timezone.utc)
    )