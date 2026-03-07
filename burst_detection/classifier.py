from burst_detection.patterns import Pattern
from burst_detection.states import BurstState

class BurstClassifier:
    def __init__(self, burst_persistence=3, recovery_time=5):
        self.burst_counter = 0
        self.recovery_counter = 0
        self.state = BurstState.NORMAL

        self.burst_persistence = burst_persistence
        self.recovery_time = recovery_time

    def update(self, pattern):
        explanation = ""

        if pattern == Pattern.BURST:
            self.burst_counter += 1
            self.recovery_counter = 0

            if self.burst_counter >= self.burst_persistence:
                self.state = BurstState.BURST
                explanation = "Violations persisted across multiple windows."
            else:
                explanation = "Burst-like pattern detected but not persistent."

        elif pattern == Pattern.DRIFT:
            self.burst_counter = 0
            self.recovery_counter = 0
            self.state = BurstState.PERIODIC_SPIKE
            explanation = "Violations are periodic without persistence."

        else:  # STABLE or NOISE
            self.burst_counter = 0
            self.recovery_counter += 1

            if self.recovery_counter >= self.recovery_time:
                self.state = BurstState.NORMAL
                explanation = "System behavior has stabilized."

        return self.state, explanation
