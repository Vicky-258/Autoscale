import time
import requests
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# ======================
# Configuration
# ======================

SERVICE_URLS = [
    "http://127.0.0.1:8000/work",
    "http://127.0.0.1:8001/work"
]

CONCURRENCY = 20          # number of in-flight requests
DURATION_SECONDS = 300    # how long to apply pressure
REQUEST_TIMEOUT = 5.0

# ======================
# Request sender
# ======================

def send_request(url):
    try:
        requests.get(url, timeout=REQUEST_TIMEOUT)
    except Exception:
        pass  # timeouts are expected under saturation

# ======================
# Sustained pressure loop
# ======================

end_time = time.time() + DURATION_SECONDS
sent = 0

with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
    futures = []

    # Prime the pool with initial concurrent requests
    for i in range(CONCURRENCY):
        url = SERVICE_URLS[i % len(SERVICE_URLS)]
        futures.append(executor.submit(send_request, url))

    while time.time() < end_time:
        done, _ = wait(futures, return_when=FIRST_COMPLETED)

        for f in done:
            futures.remove(f)
            sent += 1

            url = SERVICE_URLS[sent % len(SERVICE_URLS)]
            futures.append(executor.submit(send_request, url))

        print(f"sent so far: {sent}")

print(f"Total requests sent: {sent}")
