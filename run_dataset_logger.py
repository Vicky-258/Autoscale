import time
from performance.dataset_logger import DatasetLogger

logger = DatasetLogger(
    replica_metrics_urls=[
        "http://127.0.0.1:8000/metrics",
        "http://127.0.0.1:8001/metrics"
    ],
    output_path="svr_dataset.csv",
    debug=True,   
)


while True:
    logger.step()
    time.sleep(1)
