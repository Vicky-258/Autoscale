import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from predictor.gru_model import GRUModel

# ----------------------------
# Load data
# ----------------------------
X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

y_multi = y[:, :12]          # (N, 12)
X = X[..., np.newaxis]       # (N, 30, 1)

X = torch.tensor(X, dtype=torch.float32)
y_multi = torch.tensor(y_multi, dtype=torch.float32)

# ----------------------------
# Train / Val split
# ----------------------------
split = int(0.8 * len(X))

X_train, X_val = X[:split], X[split:]
y_train, y_val = y_multi[:split], y_multi[split:]

train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)

# ----------------------------
# Model
# ----------------------------
model = GRUModel()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 15

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train MSE: {train_loss:.4f} | "
        f"Val MSE: {val_loss:.4f}"
    )

# ----------------------------
# Save model (FREEZE POINT)
# ----------------------------
model_path = "predictor/weights/gru_v1.pt"
torch.save(model.state_dict(), model_path)
print(f"\n✅ Model saved to {model_path}")

# ----------------------------
# Qualitative check (plot)
# ----------------------------
model.eval()
idx = 100

with torch.no_grad():
    pred = model(X_val[idx:idx+1]).squeeze().numpy()

true = y_val[idx].numpy()

plt.figure(figsize=(8, 4))
plt.plot(range(12), true, marker="o", label="True")
plt.plot(range(12), pred, marker="x", label="Predicted")
plt.legend()
plt.title("GRU — 12-Step Forecast")
plt.show()

# ----------------------------
# Quantitative evaluation
# ----------------------------
with torch.no_grad():
    preds = model(X_val)

errors = (preds - y_val) ** 2
mse_per_horizon = errors.mean(dim=0)
overall_mse = errors.mean()

print("\nOverall MSE:", overall_mse.item())
for i, mse in enumerate(mse_per_horizon):
    print(f"t+{i+1}: MSE = {mse.item():.4f}")
