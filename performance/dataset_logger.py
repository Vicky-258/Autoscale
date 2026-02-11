import time
import requests
from pathlib import Path


class DatasetLogger:
    """
    Collects steady-state performance samples for performance modeling.

    IMPORTANT CONTRACT:
    - step() MUST be called frequently (e.g. every 1–5 seconds).
    - Sampling frequency is internally controlled via sample_interval.
    - Only logs data during calm, stable system behavior.
    """

    def __init__(
        self,
        replica_metrics_urls,
        output_path,
        sample_interval=60,
        warmup_seconds=30,
        debug=False,
    ):
        self.replica_metrics_urls = replica_metrics_urls
        self.output_path = Path(output_path)

        self.sample_interval = sample_interval
        self.warmup_seconds = warmup_seconds
        self.debug = debug

        self.last_sample_time = 0.0
        self.normal_since = None
        self.last_replica_count = None

        # Initialize CSV file
        if not self.output_path.exists():
            self.output_path.write_text(
                "timestamp,total_rps,replicas,p95_latency_ms\n"
            )

    # ======================
    # Gating logic
    # ======================

    def burst_state(self):
        """
        Placeholder for burst detector.
        Must return: 'NORMAL', 'BURST', or 'PERIODIC_SPIKE'
        """
        return "NORMAL"

    # ======================
    # Main step function
    # ======================

    def step(self):
        now = time.time()

        # -------- Gate 1: Burst detector --------
        if self.burst_state() != "NORMAL":
            if self.debug:
                print("[logger] blocked: burst state")
            self.normal_since = None
            return

        # -------- Gate 2: Track stable-normal window --------
        if self.normal_since is None:
            self.normal_since = now
            if self.debug:
                print("[logger] entering NORMAL state")
            return

        if now - self.normal_since < self.warmup_seconds:
            if self.debug:
                print("[logger] blocked: warmup window")
            return

        # -------- Gate 3: Sampling interval --------
        if now - self.last_sample_time < self.sample_interval:
            return

        self.last_sample_time = now

        # -------- Collect metrics (all replicas must succeed) --------
        metrics = []
        for url in self.replica_metrics_urls:
            try:
                resp = requests.get(url, timeout=2)
                metrics.append(resp.json())
            except Exception:
                if self.debug:
                    print("[logger] blocked: metrics fetch failure")
                return  # discard sample, do NOT reset state

        replica_count = len(metrics)

        # -------- Gate 4: Replica stability --------
        if self.last_replica_count is None:
            self.last_replica_count = replica_count
            if self.debug:
                print("[logger] waiting for replica stability")
            return

        if replica_count != self.last_replica_count:
            if self.debug:
                print("[logger] blocked: replica count changed")
            self.last_replica_count = replica_count
            self.normal_since = None
            return

        self.last_replica_count = replica_count

        # -------- Aggregate metrics --------
        total_rps = sum(m.get("rps", 0) for m in metrics)
        p95_latency = max(m.get("p95_latency_ms", 0) for m in metrics)

        # -------- Gate 5: Discard garbage rows --------
        if total_rps <= 0 or p95_latency <= 0:
            if self.debug:
                print("[logger] blocked: zero-rps or zero-latency")
            return

        # -------- Write sample --------
        self._write_row(
            timestamp=now,
            total_rps=total_rps,
            replicas=replica_count,
            p95_latency=p95_latency,
        )

        if self.debug:
            print(
                f"[logger] logged: rps={total_rps:.2f}, "
                f"replicas={replica_count}, p95={p95_latency:.2f}"
            )

    # ======================
    # CSV writer
    # ======================

    def _write_row(self, timestamp, total_rps, replicas, p95_latency):
        with self.output_path.open("a") as f:
            f.write(
                f"{timestamp},{total_rps},{replicas},{p95_latency}\n"
            )
