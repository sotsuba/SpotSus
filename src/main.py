from fastapi import FastAPI

from .core.lifespan import lifespan
from .api.middleware import PrometheusMiddleware
from .api.router import router


app = FastAPI(lifespan=lifespan, title="SpotSus", version="1.0.0")
app.add_middleware(PrometheusMiddleware)
app.include_router(router)
