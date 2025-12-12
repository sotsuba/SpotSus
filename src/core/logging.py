from loguru import logger
import sys
from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def configure_logging(log_level: LogLevel = LogLevel.INFO):
    logger.remove()
    # Sink 1: Standard Output
    logger.add(
        sys.stdout,
        # serialize=True,
        level=log_level,
        enqueue=True,
    )

    # Sink 2: File Output
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        level=LogLevel.DEBUG,
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
