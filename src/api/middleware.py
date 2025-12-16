from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time

from core.metrics import *
from util import Timer

from loguru import logger


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def update(self, request: Request, response: Response, duration: float):
        common = {
            "method": request.method,
            "endpoint": request.url.path,
        }
        REQUEST_COUNT.labels(**common, http_status=response.status_code).inc()
        REQUEST_LATENCY.labels(**common).observe(duration)
        IN_PROGRESS_REQUESTS.dec()

    async def dispatch(self, request: Request, call_next):
        IN_PROGRESS_REQUESTS.inc()

        with Timer() as t:
            response: Response = await call_next(request)
        logger.debug(f"Request to {request.url.path} took {t.elapsed:.2f} ms")
        await self.update(request, response, t.elapsed)
        return response
