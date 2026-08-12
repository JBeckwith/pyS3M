#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Logging Framework for pyS3M

Provides standardised logging functionality for scientific computing workflows
with support for both console and file output, performance monitoring,
and analysis-specific logging patterns.

Created on August 29, 2025
@author: Claude Code
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import threading

# Global logger configuration
_loggers: Dict[str, logging.Logger] = {}
_log_lock = threading.RLock()


class PyBayerSMLMFormatter(logging.Formatter):
    """Custom formatter for pyS3M with scientific analysis context."""

    def __init__(self):
        super().__init__()

    def format(self, record):
        # Add memory usage if available
        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            record.memory_mb = f"{memory_mb:.1f}MB"
        except ImportError:
            record.memory_mb = "N/A"

        # Standard format with scientific context
        fmt = (
            "%(asctime)s | %(name)-20s | %(levelname)-8s | "
            "MEM:%(memory_mb)s | %(message)s"
        )

        # Add function name for DEBUG level
        if record.levelno == logging.DEBUG:
            fmt = (
                "%(asctime)s | %(name)-20s | %(levelname)-8s | "
                "%(funcName)s:%(lineno)d | MEM:%(memory_mb)s | %(message)s"
            )

        self._style._fmt = fmt
        return super().format(record)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_dir: Optional[Path] = None,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up a standardised logger for pyS3M modules.

    Args:
        name: Logger name (typically module name)
        level: Logging level
        log_to_file: Whether to write logs to file
        log_dir: Directory for log files (default: project/logs)
        console_output: Whether to output to console

    Returns:
        logging.Logger: Configured logger instance
    """
    with _log_lock:
        if name in _loggers:
            return _loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Clear any existing handlers
        logger.handlers.clear()

        formatter = PyBayerSMLMFormatter()

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler
        if log_to_file:
            if log_dir is None:
                project_root = Path(__file__).parent.parent
                log_dir = project_root / "logs"

            log_dir = Path(log_dir)
            log_dir.mkdir(exist_ok=True)

            # Create timestamped log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{name}_{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            logger.info(f"Logging to file: {log_file}")

        # Prevent propagation to root logger
        logger.propagate = False

        _loggers[name] = logger
        return logger


def configure_matplotlib_logging():
    """Configure matplotlib to use appropriate logging level."""
    matplotlib_logger = logging.getLogger("matplotlib")
    matplotlib_logger.setLevel(logging.WARNING)  # Reduce matplotlib noise


# Initialize with sensible defaults
configure_matplotlib_logging()

# Example usage patterns for documentation
if __name__ == "__main__":
    logger = setup_logger("test_module")
    logger.info("This is a test log message")
