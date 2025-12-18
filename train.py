import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

y_single = y[:, 0]        # single-step target
X = X[..., np.newaxis]   # (N, 30, 1)

X = torch.tensor(X, dtype=torch.float32)
y_single = torch.tensor(y_single, dtype=torch.float32)

split = int(0.8 * len(X))

X_train, X_val = X[:split], X[split:]
y_train, y_val = y_single[:split], y_single[split:]

train_ds = TensorDataset(X_train, y_train)
val_ds   = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=32,
            batch_first=True
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # (batch, 30, 32)
        last = out[:, -1, :]           # last timestep
        return self.fc(last).squeeze()

model = LSTMModel()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 15

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

model.eval()

with torch.no_grad():
    preds = model(X_val[:200]).numpy()

actual = y_val[:200].numpy()

plt.figure(figsize=(10, 4))
plt.plot(actual, label="Actual")
plt.plot(preds, label="Predicted")
plt.legend()
plt.title("Single-step Prediction (Validation)")
plt.show()
