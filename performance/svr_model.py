# performance/svr_model.py

from typing import Optional
import logging

from performance.model import PerformanceModel
from burst_detector.detector import BurstDetector

logger = logging.getLogger("svr_advisor")


class SVRAdvisor:
    """
    Read-only latency advisor.
    Never triggers scaling.
    Never runs outside NORMAL state.
    """

    def __init__(
        self,
        detector: BurstDetector,
        min_rps: float,
        max_rps: float,
        min_replicas: int,
        max_replicas: int,
    ):
        self.detector = detector
        self.model = PerformanceModel()

        # Trust envelope
        self.min_rps = min_rps
        self.max_rps = max_rps
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas

    def predict_latency(
        self,
        workload_rps: float,
        replicas: int,
    ) -> Optional[float]:
        """
        Returns predicted p95 latency (ms) if safe.
        Returns None otherwise.
        """

        state = self.detector.get_state()
        if state != "NORMAL":
            logger.debug(
                f"SVR rejected: detector state = {state}"
            )
            return None

        if not (self.min_rps <= workload_rps <= self.max_rps):
            logger.warning(
                f"SVR rejected: rps {workload_rps:.2f} out of range "
                f"[{self.min_rps}, {self.max_rps}]"
            )
            return None

        if not (self.min_replicas <= replicas <= self.max_replicas):
            logger.warning(
                f"SVR rejected: replicas {replicas} out of range "
                f"[{self.min_replicas}, {self.max_replicas}]"
            )
            return None

        latency = self.model.predict(workload_rps, replicas)

        logger.info(
            "SVR_ADVISOR | "
            f"state=NORMAL rps={workload_rps:.2f} "
            f"replicas={replicas} "
            f"predicted_p95={latency:.2f}ms"
        )

        return latency
