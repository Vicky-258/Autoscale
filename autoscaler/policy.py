import math
from dataclasses import dataclass


# ======================
# Invariants (Day 1 Contract)
# ======================

MIN_REPLICAS = 2
MAX_REPLICAS = 20

MAX_SCALE_UP_STEP = 3
MAX_SCALE_DOWN_STEP = 2

SCALE_UP_COOLDOWN = 30        # seconds
SCALE_DOWN_COOLDOWN = 120     # seconds

CAPACITY_PER_POD = 100
MAX_BURST_BOOST = 1.0  # cap burst multiplier addition (100% max boost)


# ======================
# Pure Decision Function
# ======================

def decide_replicas(
    predicted_rps: float,
    current_replicas: int,
    is_bursting: bool,
    burst_intensity: float,
    current_time: float,
    last_scale_time: float,
) -> int:
    """
    Deterministic scaling policy.

    Returns final desired replica count.
    """

    # ----------------------
    # 1️⃣ Base Recommendation
    # ----------------------
    recommended = math.ceil(predicted_rps / CAPACITY_PER_POD)

    # ----------------------
    # 2️⃣ Burst Adjustment
    # ----------------------
    if is_bursting:
        boost = min(burst_intensity, MAX_BURST_BOOST)
        recommended = math.ceil(recommended * (1 + boost))

    # ----------------------
    # 3️⃣ Apply Hard Bounds
    # ----------------------
    recommended = max(MIN_REPLICAS, min(MAX_REPLICAS, recommended))

    # ----------------------
    # 4️⃣ Apply Step Limits
    # ----------------------
    if recommended > current_replicas:
        candidate = min(
            recommended,
            current_replicas + MAX_SCALE_UP_STEP
        )
        scaling_direction = "up"

    elif recommended < current_replicas:
        candidate = max(
            recommended,
            current_replicas - MAX_SCALE_DOWN_STEP
        )
        scaling_direction = "down"

    else:
        return current_replicas  # no change

    # ----------------------
    # 5️⃣ Apply Cooldown
    # ----------------------
    time_since_last = current_time - last_scale_time

    if scaling_direction == "up":
        if time_since_last < SCALE_UP_COOLDOWN:
            return current_replicas
    else:
        if time_since_last < SCALE_DOWN_COOLDOWN:
            return current_replicas

    return candidate