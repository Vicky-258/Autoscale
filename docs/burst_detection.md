# Burst Detection & Anomaly Routing

The Burst Direction mechanism acts as the statistical bridge between the continuous predictions of the underlying GRU network and the discrete actions of the autoscaler policy.

## The Uncertainty Error Bound (`±x RPS`)

A neural network is almost never 100% accurate. The model predicts a continuous value constraint (e.g. `202.90 RPS`), but actual traffic inherently involves natural chaotic variance.

To safely scale without thrashing during natural variance, the `uncertainty/calibrate.py` script validates the model against a historical hold-out validation dataset (`20%` of training data).
1. It predicts RPS for the entire validation set.
2. It calculates the absolute residual error between Real and Predicted RPS.
3. It takes the 97.5th percentile of the absolute residual errors.

This mathematical bound is rendered in `uncertainty/calibration.json` and loaded by `config/settings.py` as the `ERROR_BOUND`.

During operation, the model’s prediction creates an upper and lower envelope using this static bound:
```
Lower = predicted_rps - ERROR_BOUND
Upper = predicted_rps + ERROR_BOUND
```

## The Sliding Window & Counters

If actual RPS falls outside the `Lower` & `Upper` envelope, it is flagged as an `interval violation`.
However, a single violation does not equate to a true traffic burst—it could just be random noise.

`burst_detection.window.SlidingWindow` retains a fixed history of the last 20 ticks.
Each tick, `counters.update_counters()` analyzes the window to quantify:
- `freq`: The total percentage of the window that consists of interval violations.
- `runs`: A consecutive sequence of interval violations uninterrupted by normal behavior.
- `max_run`: The longest unbroken sequence of violations in the window.

## Burst Classification Patterns

`burst_detection.classifier.classify_pattern` analyzes these counters to classify the current sliding window:
- **`PATTERN_DOMINANT`**: Very high frequency (`>30%` of the latest window) AND a long unbroken run (`max_run > 0.3 * window_size`). This indicates a persistent anomaly.
- **`PATTERN_DRIFT`**: High frequency, but shattered by multiple `runs` of violations without a dominant contiguous block.
- **`PATTERN_SPORADIC`**: Little to no violations.

## State Transition Machine

The final layer (`BurstClassifier.update`) converts the pattern into a declarative `BurstState` for the `autoscaler`.
- **`NORMAL`**: Sporadic traffic.
- **`BURST`**: Transitions into BURST if a `PATTERN_DOMINANT` is detected. It requires `BURST_PERSISTENCE` (e.g., 3 consecutive ticks) of dominance to commit.
- **`PERIODIC_SPIKE`**: Transitions here if it detects a scattered `PATTERN_DRIFT`, advising the autoscaler to play it safe.

### Recovery
Once a Burst subsides and traffic returns to `NORMAL`, the classifier will NOT exit the `BURST` state immediately. It relies on a `recovery_counter` that counts down down from `RECOVERY_TIME` (3 ticks). This ensures the burst has genuinely concluded and isn't a temporary micro-dip.
