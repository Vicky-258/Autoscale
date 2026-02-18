import time
import math
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from starlette.responses import JSONResponse
import threading
import statistics

# ======================
# Configuration knobs
# ======================

WORKER_COUNT = 1          # capacity per replica
WORK_MS = 30              # CPU work per request (milliseconds)

# ======================
# Metrics storage
# ======================

latency_buffer = []
request_count = 0
lock = threading.Lock()

WINDOW_SECONDS = 60

last_window_rps = 0
last_window_p95 = 0.0

# ======================
# Infrastructure
# ======================

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=WORKER_COUNT)

# ======================
# CPU-bound work
# ======================

def cpu_work(duration_ms: int):
    """
    Busy computation to simulate real CPU work.
    """
    end_time = time.perf_counter() + duration_ms / 1000
    x = 0.0
    while time.perf_counter() < end_time:
        x += math.sqrt(12345.6789)
    return x

# ======================
# Request handler
# ======================

@app.get("/work")
def handle_request():
    global request_count

    start = time.perf_counter()

    future = executor.submit(cpu_work, WORK_MS)
    future.result()

    latency_ms = (time.perf_counter() - start) * 1000

    with lock:
        latency_buffer.append(latency_ms)
        request_count += 1

    return JSONResponse(
        {
            "status": "ok",
            "latency_ms": latency_ms
        }
    )

@app.get("/metrics")
def metrics():
    return {
        "rps": last_window_rps,
        "p95_latency_ms": last_window_p95
    }


def metrics_aggregator():
    global last_window_rps, last_window_p95
    global latency_buffer, request_count

    while True:
        time.sleep(WINDOW_SECONDS)

        with lock:
            latencies = latency_buffer
            count = request_count

            latency_buffer = []
            request_count = 0

        if latencies:
            latencies.sort()
            p95_index = max(0, int(0.95 * len(latencies)) - 1)
            last_window_p95 = latencies[p95_index]
        else:
            last_window_p95 = 0.0

        last_window_rps = count / WINDOW_SECONDS

threading.Thread(target=metrics_aggregator, daemon=True).start()
