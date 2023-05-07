from datetime import datetime

from .logger import logger


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        logger.info(f"{func.__name__} took {end_time - start_time}")
        return result

    return wrapper
