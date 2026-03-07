# Autoscaler Policy & Deterministic Rules

While the Machine Learning model predicts the raw Traffic RPS and the Burst Classifier detects anomaly states, the final decision on how many Pods the cluster should run is mathematically guarded by the `autoscaler/policy.py` engine.

## Base Calculation

The baseline `recommended` scale is derived linearly:
```python
recommended = math.ceil(predicted_rps / CAPACITY_PER_POD)
```
Where `CAPACITY_PER_POD` is staticaly defined in `config/settings.py` (e.g., 100 RPS per replica).

## Burst Modifiers

If the `BurstDetector` explicitly flags a `SystemState.BURST` (or if the Model dies/panics, entering `SystemState.UNCERTAIN`), the engine injects a scaling bias overriding the linear math.

```python
if is_bursting:
    boost = min(burst_intensity, MAX_BURST_BOOST)
    recommended = math.ceil(recommended * (1 + boost))
```
*Note: A `MAX_BURST_BOOST` of 1.0 limits the anomaly scale-up to double (200%) the original linear recommendation to protect against runaway scaling bills.*

## Hard Bounds & Step Limits

To prevent catastrophic scaling events, two rigid logical limits are enforced:
1. **Absolute Bounds**: The final recommendation is physically clamped between `MIN_REPLICAS` (2) and `MAX_REPLICAS` (20).
2. **Step Limits**: The cluster is mathematically forbidden from altering its size drastically in a single scaling event.
   - If recommended is 15 but current is 2, and `MAX_SCALE_UP_STEP=3`, the system only steps to 5.
   - It will iteratively step upwards (+3) over subsequent ticks, ensuring cluster resource provisioning can catch up smoothly.
   - Scale-down steps (`MAX_SCALE_DOWN_STEP=2`) decay slower than scale-ups to provide a cooling off buffer.

## Cooldown Mechanisms

Frequent scaling thrashes cluster infrastructure and can generate internal failure loops. To protect against this, scaling actions lock the system using a cooldown chronometer.
- `SCALE_UP_COOLDOWN` (e.g., 30s): Standard delay before scaling up again.
- `SCALE_DOWN_COOLDOWN` (e.g., 120s): Longer delay when reducing capacity to guard against trailing traffic spikes.

### The Burst Override Bypass

If standard traffic predicts high capacity, it must obey the 30-second `SCALE_UP_COOLDOWN`. 
However, if a true traffic shockwave hits and the state machine asserts `is_bursting=True`, the engine shifts to the emergency `BURST_COOLDOWN_OVERRIDE` (10s), forcefully breaking standard cooldown locks and injecting rescue capacity within 10 seconds of the last scale event.
