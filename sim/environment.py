import sys
import os

# Add parent directory to path to reach config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.logger import setup_logger

logger = setup_logger("autoscale.sim.environment")

class MockKubernetes:
    """
    Mocks the Kubernetes AppsV1 API and state.
    """
    def __init__(self, initial_replicas=3):
        self.current_replicas = initial_replicas
        self.desired_replicas = initial_replicas
        
    def get_replicas(self, deployment_name):
        # In a more advanced simulator, we could introduce lag
        # whereby current_replicas slowly approaches desired_replicas.
        # For now, it's instantaneous.
        self.current_replicas = self.desired_replicas
        return self.current_replicas
        
    def set_replicas(self, deployment_name, replicas):
        logger.info(f"[SIM Engine] API PATCH: Scaling {deployment_name} to {replicas} replicas")
        self.desired_replicas = replicas
        return replicas
