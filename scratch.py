TEST_CASES = {
    "borderline_flicker": [
        "NOISE", "NOISE", "DRIFT", "NOISE", "DRIFT", "NOISE"
    ],

    "short_burst": [
        "BURST", "BURST", "NOISE", "STABLE"
    ],

    "noisy_but_stable": [
        "NOISE", "NOISE", "DRIFT", "NOISE", "NOISE", "DRIFT"
    ],

    "slow_ramp_incident": [
        "NOISE", "DRIFT", "BURST", "BURST", "BURST"
    ],

    "worst_case_flap_attempt": [
        "BURST", "NOISE", "BURST", "NOISE", "BURST", "NOISE"
    ]
}

from temporal.classifier import BurstClassifier

def run_stress_test(name, pattern_sequence):
    print(f"\n=== Day 6 Test: {name} ===")

    classifier = BurstClassifier(
        burst_persistence=3,
        recovery_time=3
    )

    burst_triggered = False

    for t, pattern in enumerate(pattern_sequence):
        state, explanation = classifier.update(pattern)

        if state == "BURST":
            burst_triggered = True

        print(
            f"t={t:02d} | pattern={pattern:<7} "
            f"→ state={state:<16} | {explanation}"
        )

    return burst_triggered

EXPECTED = {
    "borderline_flicker": False,     # must NOT burst
    "short_burst": False,            # must NOT burst
    "noisy_but_stable": False,       # must NOT burst
    "slow_ramp_incident": True,      # SHOULD burst
    "worst_case_flap_attempt": False # must NOT burst
}

def run_all_tests():
    print("\n=== DAY 6 STRESS TEST SUMMARY ===")

    for name, sequence in TEST_CASES.items():
        burst_seen = run_stress_test(name, sequence)
        expected = EXPECTED[name]

        verdict = "PASS" if burst_seen == expected else "FAIL"

        print(
            f"\nResult: {name} → {verdict} "
            f"(burst_seen={burst_seen}, expected={expected})"
        )

run_all_tests()