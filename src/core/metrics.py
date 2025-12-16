from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP Requests", ["method", "endpoint", "http_status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds", "HTTP Request Latency", ["method", "endpoint"]
)

IN_PROGRESS_REQUESTS = Gauge(
    "in_progress_requests", "Number of In-Progress HTTP Requests"
)
