import logging
import sys


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure the ``agent`` logger (idempotent). Returns it."""
    logger = logging.getLogger("agent")
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
    return logger
