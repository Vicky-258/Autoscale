import pandas as pd
import joblib
from pathlib import Path
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# ======================
# Paths
# ======================

DATASET_PATH = Path("svr_dataset.csv")
ARTIFACT_DIR = Path("performance/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "svr_model.pkl"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"

# ======================
# Load dataset
# ======================

df = pd.read_csv(DATASET_PATH)

X = df[["total_rps", "replicas"]].values
y = df["p95_latency_ms"].values

# ======================
# Feature scaling
# ======================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================
# Train SVR (conservative)
# ======================

svr = SVR(
    kernel="rbf",
    C=1.0,        # low penalty → smooth surface
    gamma=0.1,    # wide influence → conservative
    epsilon=10.0  # ignore small p95 noise
)

svr.fit(X_scaled, y)

# ======================
# Save artifacts
# ======================

joblib.dump(svr, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("SVR training complete.")
print(f"Model saved to   : {MODEL_PATH}")
print(f"Scaler saved to  : {SCALER_PATH}")
