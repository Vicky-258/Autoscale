import math
from dataclasses import dataclass
from datetime import datetime


# ======================
# Invariants
# ======================

MIN_REPLICAS = 2
MAX_REPLICAS = 20

MAX_SCALE_UP_STEP = 3
MAX_SCALE_DOWN_STEP = 2

SCALE_UP_COOLDOWN = 30        # seconds
SCALE_DOWN_COOLDOWN = 120     # seconds

CAPACITY_PER_POD = 100
MAX_BURST_BOOST = 1.0  # max 100% additional burst boost


# ======================
# Scaling State
# ======================

@dataclass
class ScalingState:
    last_scale_time: datetime | None = None
    last_scale_direction: str | None = None  # "up" or "down"


# ======================
# Internal Helpers
# ======================

def _apply_bounds(recommended: int) -> int:
    return max(MIN_REPLICAS, min(MAX_REPLICAS, recommended))


def _apply_step_limits(current: int, recommended: int) -> tuple[int, str | None]:
    if recommended > current:
        candidate = min(recommended, current + MAX_SCALE_UP_STEP)
        return candidate, "up"

    elif recommended < current:
        candidate = max(recommended, current - MAX_SCALE_DOWN_STEP)
        return candidate, "down"

    return current, None


def _cooldown_active(state: ScalingState, direction: str, now: datetime) -> bool:
    if state.last_scale_time is None:
        return False

    elapsed = (now - state.last_scale_time).total_seconds()

    if direction == "up":
        return elapsed < SCALE_UP_COOLDOWN
    else:
        return elapsed < SCALE_DOWN_COOLDOWN


# ======================
# Public Decision Function
# ======================

def decide_replicas(
    predicted_rps: float,
    current_replicas: int,
    is_bursting: bool,
    burst_intensity: float,
    state: ScalingState,
    now: datetime,
) -> int:
    """
    Deterministic scaling policy.

    Returns final desired replica count.
    """

    # ----------------------
    # Base Recommendation
    # ----------------------
    recommended = math.ceil(predicted_rps / CAPACITY_PER_POD)

    # ----------------------
    # Burst Adjustment
    # ----------------------
    if is_bursting:
        boost = min(burst_intensity, MAX_BURST_BOOST)
        recommended = math.ceil(recommended * (1 + boost))

    # ----------------------
    # Apply Hard Bounds
    # ----------------------
    recommended = _apply_bounds(recommended)

    # ----------------------
    # Apply Step Limits
    # ----------------------
    candidate, direction = _apply_step_limits(current_replicas, recommended)

    if direction is None:
        return current_replicas  # no change

    # ----------------------
    # Apply Cooldown
    # ----------------------
    if _cooldown_active(state, direction, now):
        return current_replicas

    # ----------------------
    # Commit Scaling Event
    # ----------------------
    if candidate != current_replicas:
        state.last_scale_time = now
        state.last_scale_direction = direction

    return candidate