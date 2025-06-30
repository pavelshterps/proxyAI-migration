import time
from prometheus_client import start_http_server
from config.settings import settings
import tasks  # noqa: F401

if __name__ == "__main__":
    start_http_server(settings.metrics_port)
    print(f"Metrics server running on :{settings.metrics_port}")
    while True:
        time.sleep(1)