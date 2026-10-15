"""
RoadSense AI - Structured Application Logging
Provides clean, standardized logging across CLI scripts and Streamlit services.
Guarantees no sensitive user data or raw file contents are dumped to logs.
"""

import logging
import sys
from typing import Optional

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_logger(name: str = "roadsense", level: Optional[int] = None) -> logging.Logger:
    """
    Get or configure a named logger.
    Defaults to INFO level with clean stderr stream output.
    """
    logger = logging.getLogger(name)
    if level is None:
        level = logging.INFO
    logger.setLevel(level)

    # Avoid duplicate handlers if already initialized
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
