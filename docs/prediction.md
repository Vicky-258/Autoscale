# Predictive Pipeline & ML

The system bridges AI and statistical bounds through a deterministic predictor `predictor.py`.

## Model Pipeline

The predictive engine runs on a PyTorch `GRUModel` (Gated Recurrent Unit).
- **Input Form:** A rolling tensor window of [Batchx30x1] containing the normalized RPS footprint of the last 30 intervals (15 minutes).
- **Forecast Phase:** The network recursively unrolls forward to forecast $t+1$ to $t+12$ sequences (6 minutes).

## Health Protections

Machine Learning models, if subjected to undefined feature math, can panic and spit out `NaN` or `Infinity` (`Float.INF`) values. Returning these to a Kubernetes capacity API could cause horizontal scaling pods to crash catastrophically or spin off unlimited scaling bills.

### Panic Failsafe

The system structurally encapsulates the PyTorch layer.
```python
if not math.isfinite(predicted_rps):
    raise ValueError("Model predicted NaN or Inf")
```
By forcing a hard panic on infinite numbers, `set_replicas` avoids triggering the mathematical payload. Instead, `model_alive = False` is broadcast across the system. This traps the State Machine locally into `SystemState.UNCERTAIN`, where it defensively assumes exactly 1-to-1 parity matching (mirroring `actual_rps`) and scales strictly linearly without using predictive bounds until the network stabilizes.
