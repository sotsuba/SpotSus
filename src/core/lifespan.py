from contextlib import asynccontextmanager
from loguru import logger
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from fastapi import FastAPI
from fastapi_limiter import FastAPILimiter

from core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup(app)
    yield
    await shutdown(app)


# TODO: Remember to initialize model here.
# TODO: Codebase looks messy, consider refactoring.
async def startup(app: FastAPI) -> None:
    """Initialize all application resources."""
    logger.info("Starting SpotSus...")

    configure_logging()
    try:
        app.state.redis = await Redis.from_url(
            "redis://0.0.0.0:6379/0",
        )
        await app.state.redis.ping()
        logger.success("Redis connection established.")
    except Exception as e:
        logger.error(f"Redis conenction failed: {e}")
        app.state.redis = None

    if app.state.redis:
        try:
            await FastAPILimiter.init(app.state.redis)
            logger.success("Rate limiter initialized.")
        except Exception as e:
            logger.warning(f"Rate limiter failed to initialize: {e}")

    logger.success("SpotSus started successfully.")


# TODO: Codebase looks messy, consider refactoring.
async def shutdown(app: FastAPI) -> None:
    """Cleanup all application resources."""
    logger.info("Shutting down Spotsus...")

    try:
        await FastAPILimiter.close()
        logger.info("Rate limiter closed.")
    except Exception as e:
        logger.warning(f"Rate limiter close failed: {e}")

    if app.state.redis:
        try:
            await app.state.redis.close()
            logger.info("Redis connection closed.")
        except Exception as e:
            logger.warning(f"Redis close failed: {e}")

    logger.info("SpotSus shutdown completely.")
