# The Simulation Environment (`sim/`)

Testing an autoscaler in live Kubernetes is difficult and time-intensive since real-world cooldowns, metric ingestion delays, and scale-up lags make testing the logical matrix impossible to accelerate.

The `sim/runner.py` completely decouples the PyTorch GRU predictor and Python logic loop from physical time and infrastructure.

## Traffic Generator

The `TrafficGenerator` acts as the `get_rps_fn` for the `run_autoscaler` loop.
1. It simulates baseline traffic using a cyclic sine wave (e.g., oscillating horizontally).
2. It mathematically injects localized bursts `+300 RPS` to synthetically force interval violations on the static standard deviation of the predictor.
3. This creates a fully controlled test variable that guarantees the GRU's confidence bounds will be breached exactly when requested.

## Mock Kubernetes Cluster

`environment.py` hosts a local state copy of `MockKubernetes`. 
Instead of sending real Helm APIs to Kubernetes, it acts as the `get_replicas_fn` and `set_replicas_fn` in memory. This returns API changes instantly, enabling 200 ticks of scaling policy logic to test itself in 0.1 seconds.

## Synthetic Chronometer

The logic heavily depends on physical time `datetime.now(timezone.utc)` to compute the cooldown blocks (30s timeouts).
In order to fast-forward tests, `SimulationClock` synthesizes a chronological clock. It intercepts `time.sleep` requests and forcibly dials the `current_time` variable forward by 30 seconds for every loop cycle iteration. This passes test time at exactly the expected production cadence, proving the engine mathematically works before deployment into the real world.
