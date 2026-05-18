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
import time
import functools
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Dict, Callable
from contextlib import contextmanager
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


class AnalysisProgressLogger:
    """Specialised logger for analysis progress with timing information."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_time = None
        self.step_times = []

    def start_analysis(self, analysis_name: str, n_items: int = None):
        """Start analysis timing."""
        self.start_time = time.time()
        self.step_times = []
        if n_items:
            self.logger.info(f"Starting {analysis_name} - {n_items} items to process")
        else:
            self.logger.info(f"Starting {analysis_name}")

    def log_progress(self, step: str, current: int = None, total: int = None):
        """Log progress step with timing."""
        current_time = time.time()
        if self.start_time:
            elapsed = current_time - self.start_time
            self.step_times.append(elapsed)

        if current is not None and total is not None:
            percent = (current / total) * 100
            if len(self.step_times) > 1:
                rate = current / elapsed if elapsed > 0 else 0
                eta = (total - current) / rate if rate > 0 else float("inf")
                self.logger.info(
                    f"{step} - {current}/{total} ({percent:.1f}%) - "
                    f"Rate: {rate:.1f} items/s - ETA: {eta:.1f}s"
                )
            else:
                self.logger.info(f"{step} - {current}/{total} ({percent:.1f}%)")
        else:
            if self.start_time:
                self.logger.info(f"{step} - Elapsed: {elapsed:.1f}s")
            else:
                self.logger.info(step)

    def finish_analysis(self, analysis_name: str, success: bool = True):
        """Complete analysis with summary."""
        if self.start_time:
            total_time = time.time() - self.start_time
            status = "COMPLETED" if success else "FAILED"
            self.logger.info(
                f"{analysis_name} {status} - Total time: {total_time:.2f}s"
            )
        else:
            status = "COMPLETED" if success else "FAILED"
            self.logger.info(f"{analysis_name} {status}")


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


def get_analysis_logger(module_name: str) -> AnalysisProgressLogger:
    """
    Get an analysis progress logger for scientific computing workflows.

    Args:
        module_name: Name of the analysis module

    Returns:
        AnalysisProgressLogger: Progress logger instance
    """
    base_logger = setup_logger(f"analysis.{module_name}")
    return AnalysisProgressLogger(base_logger)


def log_performance(func: Callable) -> Callable:
    """
    Decorator to log function performance and memory usage.

    Usage::

        @log_performance
        def expensive_analysis(data):
            # analysis code
            return results
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = setup_logger(f"performance.{func.__module__}.{func.__name__}")

        # Log function start
        start_time = time.time()
        logger.info(
            f"Starting {func.__name__} with {len(args)} args, {len(kwargs)} kwargs"
        )

        try:
            # Execute function
            result = func(*args, **kwargs)

            # Log successful completion
            elapsed = time.time() - start_time
            logger.info(f"Completed {func.__name__} in {elapsed:.3f}s")

            return result

        except Exception as e:
            # Log error
            elapsed = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {elapsed:.3f}s: {str(e)}")
            raise

    return wrapper


@contextmanager
def log_analysis_block(analysis_name: str, logger_name: Optional[str] = None):
    """
    Context manager for logging analysis blocks with automatic timing.

    Usage:
        with log_analysis_block("Drift Correction", "drift_analysis"):
            # analysis code
            pass
    """
    if logger_name is None:
        logger_name = "analysis.block"

    logger = setup_logger(logger_name)
    start_time = time.time()

    logger.info(f"Starting analysis block: {analysis_name}")

    try:
        yield logger
        elapsed = time.time() - start_time
        logger.info(f"Completed analysis block: {analysis_name} ({elapsed:.3f}s)")

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            f"Failed analysis block: {analysis_name} after {elapsed:.3f}s - {str(e)}"
        )
        raise


# Global configuration
def set_global_log_level(level: int):
    """Set logging level for all pyS3M loggers."""
    with _log_lock:
        for logger in _loggers.values():
            logger.setLevel(level)


def configure_matplotlib_logging():
    """Configure matplotlib to use appropriate logging level."""
    matplotlib_logger = logging.getLogger("matplotlib")
    matplotlib_logger.setLevel(logging.WARNING)  # Reduce matplotlib noise


# Initialize with sensible defaults
configure_matplotlib_logging()

# Example usage patterns for documentation
if __name__ == "__main__":
    # Basic logger
    logger = setup_logger("test_module")
    logger.info("This is a test log message")

    # Analysis progress logger
    progress = get_analysis_logger("test_analysis")
    progress.start_analysis("Test Analysis", 100)
    progress.log_progress("Processing data", 25, 100)
    progress.finish_analysis("Test Analysis", success=True)

    # Performance logging decorator
    @log_performance
    def test_function():
        time.sleep(0.1)
        return "result"

    result = test_function()

    # Analysis block context manager
    with log_analysis_block("Test Block"):
        time.sleep(0.05)
