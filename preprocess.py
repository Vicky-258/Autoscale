import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib

WINDOW_SIZE = 30
HORIZON = 12
TRAIN_SPLIT = 0.8

rps = np.load("data/raw/rps_raw.npy")

print("Total points:", len(rps))
print("Min / Max:", rps.min(), rps.max())

rps_2d = rps.reshape(-1, 1)

split_idx = int(TRAIN_SPLIT * len(rps_2d))
train_rps = rps_2d[:split_idx]

scaler = MinMaxScaler()
scaler.fit(train_rps)

rps_norm = scaler.transform(rps_2d).flatten()

print("Normalized Min / Max:", rps_norm.min(), rps_norm.max())

def create_sliding_windows(series, window_size, horizon):
    X, y = [], []

    for i in range(len(series) - window_size - horizon + 1):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size : i + window_size + horizon])

    return np.array(X), np.array(y)

X, y = create_sliding_windows(
    rps_norm,
    WINDOW_SIZE,
    HORIZON
)

print("X shape:", X.shape)
print("y shape:", y.shape)

idx = 200  # any index

plt.figure(figsize=(10, 4))
plt.plot(range(WINDOW_SIZE), X[idx], label="Past (X)")
plt.plot(
    range(WINDOW_SIZE, WINDOW_SIZE + HORIZON),
    y[idx],
    label="Future (y)"
)
plt.legend()
plt.title("One Training Sample")
plt.show()
