import numpy as np
import torch
import matplotlib.pyplot as plt

from config.settings import DEVICE, MODEL_PATH, SCALER_PATH
from predictor.models.gru_model import GRUModel
from predictor.predictor import GRUPredictor


# -------------------------
# Interval violation logic
# -------------------------
def interval_violation(y, low, high):
    return 0 if low <= y <= high else 1


# -------------------------
# Load data
# -------------------------
X = np.load("data/processed/X.npy")   # (N, 30)
y = np.load("data/processed/y.npy")   # (N, 12)

N = X.shape[0]
split = int(0.8 * N)

X_val = X[split:]
y_val = y[split:]


# -------------------------
# Load model + predictor
# -------------------------
model = GRUModel()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

predictor = GRUPredictor(model, scaler_path=SCALER_PATH, device=DEVICE)

X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)

with torch.no_grad():
    preds = predictor.predict_next_12(X_val_t)

preds_np = preds.numpy()


# -------------------------
# Build prediction interval
# (fixed 97.5% residual envelope)
# -------------------------
residuals = y_val - preds_np
upper_bounds = [
    np.percentile(residuals[:, h], 97.5)
    for h in range(12)
]


# -------------------------
# Select horizon to inspect
# -------------------------
h = 7  # t+8 (0-indexed)

actual = y_val[:, h]
pred   = preds_np[:, h]

upper = pred + upper_bounds[h]
lower = pred - upper_bounds[h]


# -------------------------
# Violation sequence
# -------------------------
violations = [
    interval_violation(y, lo, hi)
    for y, lo, hi in zip(actual, lower, upper)
]


# -------------------------
# Visualization
# -------------------------
start = 500
end   = 800

x = range(end - start)

plt.figure(figsize=(12, 4))

plt.plot(x, actual[start:end], label="Actual", linewidth=2)
plt.plot(x, pred[start:end], linestyle="--", label="Prediction")

plt.fill_between(
    x,
    lower[start:end],
    upper[start:end],
    alpha=0.3,
    label="Prediction Interval"
)

# Violation markers
violation_x = [
    i for i, v in enumerate(violations[start:end]) if v == 1
]
violation_y = [
    actual[start + i] for i in violation_x
]

plt.scatter(
    violation_x,
    violation_y,
    color="red",
    s=25,
    label="Violation"
)

plt.title("Day 3 — Interval Violations (t+8)")
plt.legend()
plt.tight_layout()
plt.show()
