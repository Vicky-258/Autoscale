import numpy as np
import torch

from predictor.models.gru_model import GRUModel
from predictor.predictor import GRUPredictor

def interval_violation(y, low, high):
    if low <= y <= high:
        return 0
    else:
        return 1

X = np.load("data/processed/X.npy")   # (N, 30)
y = np.load("data/processed/y.npy")   # (N, 12)

print("X:", X.shape, "y:", y.shape)

N = X.shape[0]
split = int(0.8 * N)

X_val = X[split:]
y_val = y[split:]

print("Val shapes:", X_val.shape, y_val.shape)

model = GRUModel()
model.load_state_dict(torch.load("predictor/weights/gru_v1.pt"))
model.eval()

predictor = GRUPredictor(model)

X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)

with torch.no_grad():
    preds = predictor.predict_next_12(X_val_t)

preds_np = preds.numpy()
residuals = y_val - preds_np

print("Preds:", preds_np.shape)
print("Residuals:", residuals.shape)

import numpy as np

residuals_h = residuals[:, 5]

p90  = np.percentile(residuals_h, 90)
p95  = np.percentile(residuals_h, 95)
p975 = np.percentile(residuals_h, 97.5)

print("t+6 percentiles:")
print("90% :", p90)
print("95% :", p95)
print("97.5%:", p975)

upper_bounds = []

for h in range(12):
    ub = np.percentile(residuals[:, h], 97.5)
    upper_bounds.append(ub)

print(*upper_bounds)

import matplotlib.pyplot as plt

h = 7  # t+8 (0-indexed)

actual = y_val[:, h]
pred = preds_np[:, h]

upper = pred + upper_bounds[h]
lower = pred - upper_bounds[h]

start = 500
end = 800

plt.figure(figsize=(12, 4))

x = range(end - start)

plt.plot(x, actual[start:end], label="Actual", linewidth=2)
plt.plot(x, pred[start:end], linestyle="--", label="Prediction")

plt.fill_between(
    x,
    lower[start:end],
    upper[start:end],
    alpha=0.3,
    label="Prediction Interval"
)

plt.title("t+8 Prediction with Uncertainty Envelope")
plt.legend()
plt.tight_layout()
plt.show()
