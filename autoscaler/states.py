from enum import Enum

class SystemState(Enum):
    NORMAL = "normal"
    BURST = "burst"
    UNCERTAIN = "uncertain"

class ScalingDirection(Enum):
    UP = "up"
    DOWN = "down"
