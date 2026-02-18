class BurstClassifier:
    def __init__(self, burst_persistence=3, recovery_time=5):
        self.burst_counter = 0
        self.recovery_counter = 0
        self.state = "NORMAL"

        self.burst_persistence = burst_persistence
        self.recovery_time = recovery_time

    def update(self, pattern):
        explanation = ""

        if pattern == "BURST":
            self.burst_counter += 1
            self.recovery_counter = 0

            if self.burst_counter >= self.burst_persistence:
                self.state = "BURST"
                explanation = "Violations persisted across multiple windows."
            else:
                explanation = "Burst-like pattern detected but not persistent."

        elif pattern == "DRIFT":
            self.burst_counter = 0
            self.recovery_counter = 0
            self.state = "PERIODIC_SPIKE"
            explanation = "Violations are periodic without persistence."

        else:  # STABLE or NOISE
            self.burst_counter = 0
            self.recovery_counter += 1

            if self.recovery_counter >= self.recovery_time:
                self.state = "NORMAL"
                explanation = "System behavior has stabilized."

        return self.state, explanation
