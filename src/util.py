import time
from loguru import logger


class Timer:
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.duration = (time.perf_counter() - self.start_time) * 1000

        if exc_type is not None:
            logger.error(f"An exception occurred: {exc_value}")
        logger.info(f"Elapsed time: {self.duration:.2f} ms")

        return False

    @property
    def elapsed(self) -> float:
        return self.duration
