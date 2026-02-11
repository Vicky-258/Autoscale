"""
Invariants:

1. Pattern classification is temporal, not pointwise.
2. Exactly ONE pattern is active per window.
3. Single violations are NEVER bursts.
4. Bursts require consecutive violations.
5. Periodic spacing implies drift, not burst.
6. Thresholds exist to express structure, not precision.
"""

from enum import Enum

class Pattern(Enum):
    STABLE = "stable"
    NOISE = "noise"
    BURST = "burst"
    DRIFT = "drift"


def classify_pattern(freq, max_run, run_count, window_size):
    """
    Temporal pattern classification.

    freq       : total violations in window
    max_run    : longest consecutive violation run
    run_count  : number of violation runs
    window_size: sliding window length
    """

    # No disagreement at all
    if freq == 0:
        return Pattern.STABLE

    # Relative dominance metrics
    run_ratio = max_run / window_size
    freq_ratio = freq / window_size

    # Sustained consecutive disagreement
    # (dominant continuous violation)
    if run_ratio >= 0.3:
        return Pattern.BURST

    # Repeated but non-consecutive disagreement
    # (many short runs, no persistence)
    if max_run == 1 and run_count >= 0.3 * window_size:
        return Pattern.DRIFT

    # Everything else
    return Pattern.NOISE



