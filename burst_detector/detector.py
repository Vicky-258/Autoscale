# burst_detector/detector.py

from burst_detector.config import *
from uncertainty.violations import interval_violation
from temporal.window import SlidingWindow
from temporal.counters import (
    violation_frequency,
    max_run_length,
    run_count,
)
from temporal.patterns import classify_pattern
from temporal.classifier import BurstClassifier


class BurstDetector:
    """
    Inference-only burst detector.
    No training dependencies.
    """

    def __init__(self):
        self.window = SlidingWindow(WINDOW_SIZE)
        self.classifier = BurstClassifier(
            burst_persistence=BURST_PERSISTENCE,
            recovery_time=RECOVERY_TIME
        )
        self.state = "NORMAL"

    def update(self, actual, lower, upper):
        """
        Update detector with latest observation.
        """

        v = interval_violation(actual, lower, upper)
        self.window.add(v)

        window = self.window.get()

        freq = violation_frequency(window)
        max_run = max_run_length(window)
        runs = run_count(window)

        pattern = classify_pattern(
            freq=freq,
            max_run=max_run,
            run_count=runs,
            window_size=len(window)
        )

        self.state, explanation = self.classifier.update(pattern)
        return self.state, explanation

    def get_state(self):
        return self.state
