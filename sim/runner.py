import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path to import main components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import run_autoscaler
from sim.traffic import TrafficGenerator
from sim.environment import MockKubernetes
from config.logger import setup_logger

logger = setup_logger("autoscale.sim.runner")

import datetime

class StopSimulation(Exception):
    """Raised to break out of the infinite control loop deterministically."""
    pass

class SimulationClock:
    def __init__(self, max_ticks):
        self.ticks = 0
        self.max_ticks = max_ticks
        self.current_time = datetime.datetime.now(datetime.timezone.utc)
        
    def sleep_and_tick(self, duration_s):
        """Replaces time.sleep(), fast-forwarding the clock instantly."""
        self.ticks += 1
        self.current_time += datetime.timedelta(seconds=duration_s)
        if self.ticks >= self.max_ticks:
            raise StopSimulation("Simulation Complete")

def run_simulation(total_ticks=200):
    logger.info(f"🚀 Starting AutoScale Simulation Engine for {total_ticks} ticks...")
    
    traffic_gen = TrafficGenerator()
    k8s_mock = MockKubernetes(initial_replicas=3)
    clock = SimulationClock(max_ticks=total_ticks)
    
    metrics = []

    def log_metrics(state_dict):
        state_dict['tick'] = clock.ticks
        metrics.append(state_dict)

    try:
        # Run the core loop with simulated dependencies
        run_autoscaler(
            get_rps_fn=traffic_gen.get_next,
            get_tensor_fn=traffic_gen.get_tensor,
            get_replicas_fn=k8s_mock.get_replicas,
            set_replicas_fn=k8s_mock.set_replicas,
            sleep_fn=clock.sleep_and_tick,
            get_time_fn=lambda: clock.current_time,
            metrics_callback=log_metrics
        )
    except StopSimulation:
        logger.info("🛑 Simulation gracefully reached max ticks.")
        
    # --- Reporting & Visualization ---
    df = pd.DataFrame(metrics)
    if not df.empty:
        df.set_index('tick', inplace=True)
        generate_report(df)
        
def generate_report(df):
    logger.info("📊 Generating Simulation Report...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot 1: Traffic + Predictions
    ax1.plot(df.index, df['actual_rps'], label='Actual RPS', color='blue', alpha=0.6)
    ax1.plot(df.index, df['predicted_rps'], label='Predicted RPS (t+12)', color='orange', linestyle='--')
    ax1.fill_between(df.index, df['lower_bound'], df['upper_bound'], color='orange', alpha=0.2, label='Confidence Interval')
    ax1.set_ylabel('Requests Per Second (RPS)')
    ax1.set_title('Simulated Traffic & Forecasting')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Replicas Allocation
    ax2.step(df.index, df['current_replicas'], label='Replicas', color='green', where='post')
    ax2.set_ylabel('Replicas')
    ax2.set_xlabel('Simulation Ticks (Fast-Forwarded)')
    ax2.set_title('Autoscaler Replica Strategy')
    
    # Highlight bursting periods
    bursts = df[df['burst_state'] == 'BURST']
    for tick in bursts.index:
        ax2.axvline(x=tick, color='red', alpha=0.1, linestyle='-')
        ax1.axvline(x=tick, color='red', alpha=0.1, linestyle='-')

    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sim/report.png')
    logger.info("✅ Simulation complete! Visualization saved to sim/report.png.")

if __name__ == "__main__":
    # Remove existing state if any, to start clean
    if os.path.exists("scaling_state.json"):
        os.remove("scaling_state.json")
    
    run_simulation(total_ticks=200)
