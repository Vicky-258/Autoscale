import pandas as pd
import matplotlib.pyplot as plt
from performance.model import PerformanceModel
import numpy as np

# Load SVR dataset
df = pd.read_csv("svr_dataset.csv")

total_rps = df["total_rps"].values
replicas = df["replicas"].values
y_actual = df["p95_latency_ms"].values

perf_model = PerformanceModel()

y_pred = []
for rps, rep in zip(total_rps, replicas):
    pred = perf_model.predict(rps, int(rep))
    y_pred.append(pred)

y_pred = np.array(y_pred)

plt.figure(figsize=(6,6))
plt.scatter(y_actual, y_pred, alpha=0.7)

min_v = min(y_actual.min(), y_pred.min())
max_v = max(y_actual.max(), y_pred.max())
plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")

plt.xlabel("Actual p95 Latency (ms)")
plt.ylabel("Predicted p95 Latency (ms)")
plt.title("Day 5 — SVR Predicted vs Actual (Production Path)")
plt.grid(True)
plt.show()
