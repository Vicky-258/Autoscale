import math
from dataclasses import dataclass
from datetime import datetime
from autoscaler.states import ScalingDirection
from config.settings import (
    MIN_REPLICAS,
    MAX_REPLICAS,
    MAX_SCALE_UP_STEP,
    MAX_SCALE_DOWN_STEP,
    SCALE_UP_COOLDOWN,
    SCALE_DOWN_COOLDOWN,
    BURST_COOLDOWN_OVERRIDE,
    CAPACITY_PER_POD,
    MAX_BURST_BOOST,
)  # max 100% additional burst boost


# ======================
# Scaling State
# ======================

@dataclass
class ScalingState:
    last_scale_time: datetime | None = None
    last_scale_direction: ScalingDirection | None = None

    def save(self, filepath: str):
        import json
        data = {
            "last_scale_time": self.last_scale_time.isoformat() if self.last_scale_time else None,
            "last_scale_direction": self.last_scale_direction.value if self.last_scale_direction else None
        }
        with open(filepath, "w") as f:
            json.dump(data, f)
            
    @classmethod
    def load(cls, filepath: str) -> "ScalingState":
        import json
        import os
        if not os.path.exists(filepath):
            return cls()
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            time_str = data.get("last_scale_time")
            dir_str = data.get("last_scale_direction")
            
            return cls(
                last_scale_time=datetime.fromisoformat(time_str) if time_str else None,
                last_scale_direction=ScalingDirection(dir_str) if dir_str else None
            )
        except Exception as e:
            # Replaced print with a comment or pass, as we'll handle logging mostly in main/control
            return cls()



# ======================
# Internal Helpers
# ======================

def _apply_bounds(recommended: int) -> int:
    return max(MIN_REPLICAS, min(MAX_REPLICAS, recommended))


def _apply_step_limits(current: int, recommended: int) -> tuple[int, ScalingDirection | None]:
    if recommended > current:
        candidate = min(recommended, current + MAX_SCALE_UP_STEP)
        return candidate, ScalingDirection.UP

    elif recommended < current:
        candidate = max(recommended, current - MAX_SCALE_DOWN_STEP)
        return candidate, ScalingDirection.DOWN

    return current, None


def _cooldown_active(state: ScalingState, direction: ScalingDirection, now: datetime, is_bursting: bool = False) -> bool:
    if state.last_scale_time is None:
        return False

    elapsed = (now - state.last_scale_time).total_seconds()

    if is_bursting and direction == ScalingDirection.UP:
        return elapsed < BURST_COOLDOWN_OVERRIDE

    if direction == ScalingDirection.UP:
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
    if _cooldown_active(state, direction, now, is_bursting):
        return current_replicas

    # ----------------------
    # Commit Scaling Event
    # ----------------------
    # State mutation strictly deferred to main.py to prevent ghost scaling.
    return candidate