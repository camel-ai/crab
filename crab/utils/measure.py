import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


# Misc logger setup so a debug log statement gets printed on stdout.
handler = logging.StreamHandler()
log_format = "%(asctime)s %(levelname)s -- %(message)s"
formatter = logging.Formatter(log_format)
handler.setFormatter(formatter)
logger.addHandler(handler)


def timed(func):
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        func_class = args[0].__class__.__name__ if args else ""
        info = "{}.{} ran in {}s".format(
            func_class,
            func.__name__,
            round(end - start, 2),
        )
        if hasattr(args[0], "name"):
            info += f" with name {args[0].name}"
        logger.info(info)
        return result

    return wrapper
