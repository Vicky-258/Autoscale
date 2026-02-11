import joblib
import numpy as np
from pathlib import Path

# ======================
# Paths
# ======================

ARTIFACT_DIR = Path("performance/artifacts")
MODEL_PATH = ARTIFACT_DIR / "svr_model.pkl"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"

class PerformanceModel:
    """
    Runtime wrapper around trained SVR performance model.
    Predicts steady-state p95 latency.
    """

    def __init__(self):
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            raise RuntimeError(
                "SVR artifacts not found. Train the model first."
            )

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

    def predict(self, total_rps: float, replicas: int) -> float:
        """
        Predict p95 latency (ms) for a calm system.

        Args:
            total_rps: steady-state request rate
            replicas: active replica count

        Returns:
            predicted p95 latency in ms
        """
        X = np.array([[total_rps, replicas]])
        X_scaled = self.scaler.transform(X)

        latency = self.model.predict(X_scaled)[0]
        return float(latency)
