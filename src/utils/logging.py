"""
Logging configuration using loguru.
"""

import sys
from loguru import logger
from config.settings import LOG_LEVEL, PROJECT_ROOT

# Remove default handler
logger.remove()

# Console handler
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
)

# File handler
logger.add(
    PROJECT_ROOT / "mlb_predictor.log",
    level="DEBUG",
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} | {message}",
)


def get_logger(name: str):
    """Get a logger bound with a module name."""
    return logger.bind(name=name)
