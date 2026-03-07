from config.settings import BURST_WINDOW_SIZE, BURST_PERSISTENCE, RECOVERY_TIME
from uncertainty.violations import interval_violation
from burst_detection.window import SlidingWindow
from burst_detection.counters import (
    violation_frequency,
    max_run_length,
    run_count,
)
from burst_detection.patterns import classify_pattern
from burst_detection.classifier import BurstClassifier
from burst_detection.states import BurstState


class BurstDetector:
    """
    Inference-only burst detector.
    """

    def __init__(self):
        self.window = SlidingWindow(BURST_WINDOW_SIZE)
        self.classifier = BurstClassifier(
            burst_persistence=BURST_PERSISTENCE,
            recovery_time=RECOVERY_TIME
        )
        self.state = BurstState.NORMAL

    def update(self, actual, lower, upper, is_valid=True):
        """
        Update detector with latest observation.
        """
        if not is_valid:
            # Model is returning invalid/fallback bounds. 
            # Skip updating the window to prevent false decay.
            return self.state, "Detector paused due to invalid prediction bounds."

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
            window_size=self.window.size
        )

        self.state, explanation = self.classifier.update(pattern)
        return self.state, explanation

    def get_state(self):
        return self.state
