#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Progress bar utilities for pyS3M.

This module provides clean, consistent progress bar functionality across the
entire pyS3M package using tqdm with proper cleanup and context management.

Created on August 21, 2025
@author: Claude Code Assistant
"""

import sys
from contextlib import contextmanager
from typing import Optional, Any, Dict, Iterable
import os

try:
    # Default to text-based tqdm for better compatibility with carriage return
    # updates (used in both notebook and terminal environments).
    from tqdm import tqdm as base_tqdm

    TQDM_AVAILABLE = True

except ImportError:
    # Fallback if tqdm not available - use mock implementation
    base_tqdm = None
    TQDM_AVAILABLE = False


class MockProgressUtils:
    """Fallback progress utilities when tqdm is not available."""

    @staticmethod
    def clean_progress_bar(iterable=None, total=None, desc="Processing", **kwargs):
        """Mock context manager that returns iterable unchanged."""

        class MockContext:
            def __init__(self, iterable, total):
                self.iterable = (
                    iterable if iterable is not None else range(total) if total else []
                )

            def __enter__(self):
                return self.iterable

            def __exit__(self, *args):
                pass

            def update(self, n=1):
                """Mock update method for manual progress updates."""
                pass

            def write(self, msg):
                """Mock write method for messages."""
                print(msg)

            def close(self):
                """Mock close method."""
                pass

        return MockContext(iterable, total)


class ProgressBarConfig:
    """Configuration constants for consistent progress bar styling."""

    # Default styling
    DEFAULT_LEAVE = False  # Don't leave progress bars after completion in terminal
    DEFAULT_DYNAMIC_NCOLS = True  # Adapt to terminal width
    DEFAULT_MINITERS = 1  # Update frequency
    DEFAULT_MININTERVAL = (
        0.05  # Minimum time between updates (seconds) - faster for notebooks
    )

    # Color and styling (if terminal supports it)
    DEFAULT_COLOUR = "green"
    DEFAULT_BAR_FORMAT = (
        "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    # Performance settings
    DEFAULT_SMOOTHING = 0.3  # Smoothing factor for speed estimation

    @classmethod
    def get_default_kwargs(cls) -> Dict[str, Any]:
        """Get default keyword arguments for tqdm progress bars."""
        base_kwargs = {
            "leave": cls.DEFAULT_LEAVE,
            "dynamic_ncols": cls.DEFAULT_DYNAMIC_NCOLS,
            "miniters": cls.DEFAULT_MINITERS,
            "mininterval": cls.DEFAULT_MININTERVAL,
            "smoothing": cls.DEFAULT_SMOOTHING,
        }

        # Configure for both notebook and terminal environments using text-based tqdm
        base_kwargs["file"] = sys.stdout
        base_kwargs["leave"] = (
            False  # Don't leave progress bars to avoid line interference
        )

        return base_kwargs


@contextmanager
def clean_progress_bar(
    iterable: Optional[Iterable] = None,
    total: Optional[int] = None,
    desc: str = "Processing",
    leave: Optional[bool] = None,
    colour: Optional[str] = None,
    position: Optional[int] = None,
    **kwargs,
):
    """
    Context manager for tqdm progress bars with guaranteed cleanup.

    This function provides a clean interface for progress bars that ensures
    proper cleanup regardless of how the operation completes (success, failure,
    or interruption).

    Args:
        iterable: Iterable to wrap (optional, for direct iteration)
        total: Total number of iterations (required if no iterable provided)
        desc: Description text shown with the progress bar
        leave: Whether to leave the progress bar after completion (default: False)
        colour: Progress bar color (green, blue, red, etc.)
        position: Position for nested progress bars (0=top, 1=second, etc.)
        **kwargs: Additional tqdm parameters

    Yields:
        tqdm.tqdm: Progress bar object for manual updates

    Examples::

        # Basic usage with iterable
        with clean_progress_bar(range(100), desc="Processing items") as pbar:
            for item in pbar:
                pass  # process item

        # Manual progress updates
        with clean_progress_bar(total=100, desc="Fitting puncta") as pbar:
            for i in range(100):
                process_item(i)
                pbar.update(1)

        # Nested progress bars
        with clean_progress_bar(range(10), desc="Outer", position=0) as outer:
            for i in outer:
                with clean_progress_bar(range(5), desc="Inner", position=1) as inner:
                    for j in inner:
                        pass
    """
    # If tqdm not available, use mock implementation
    if not TQDM_AVAILABLE:
        yield MockProgressUtils.clean_progress_bar(
            iterable=iterable, total=total, desc=desc, **kwargs
        )
        return

    # Honour global disable flag set via set_progress_enabled(False)
    if progress_manager.is_globally_disabled():
        kwargs["disable"] = True

    # Merge default configuration with provided arguments
    pbar_kwargs = ProgressBarConfig.get_default_kwargs()

    # Override defaults with provided arguments
    if leave is not None:
        pbar_kwargs["leave"] = leave
    if colour is not None:
        # Apply colour in both notebook and terminal environments
        pbar_kwargs["colour"] = colour
    if position is not None:
        pbar_kwargs["position"] = position

    # Add any additional kwargs
    pbar_kwargs.update(kwargs)

    # Create progress bar
    pbar = None
    try:
        if iterable is not None:
            pbar = base_tqdm(iterable, desc=desc, **pbar_kwargs)
        else:
            if total is None:
                raise ValueError("Either 'iterable' or 'total' must be provided")
            pbar = base_tqdm(total=total, desc=desc, **pbar_kwargs)

        yield pbar

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        if pbar is not None:
            pbar.write("Operation cancelled by user")
        raise
    except Exception as e:
        # Handle other exceptions
        if pbar is not None:
            pbar.write(f"Error during operation: {str(e)[:50]}...")
        raise
    finally:
        # Always clean up progress bar
        if pbar is not None:
            pbar.close()

        # Don't add newline in notebooks - let the calling code handle line management

        # Force flush to ensure clean terminal state
        sys.stdout.flush()
        if hasattr(sys.stderr, "flush"):
            sys.stderr.flush()


class ProgressBarManager:
    """
    Manager class for coordinating multiple progress bars and handling global settings.

    This class provides centralized control over progress bar behavior, including
    the ability to disable all progress bars globally for batch processing or
    when running in non-interactive environments.
    """

    def __init__(self):
        self._global_disable = False
        self._default_config = ProgressBarConfig.get_default_kwargs()

    def set_global_disable(self, disable: bool):
        """Enable or disable all progress bars globally."""
        self._global_disable = disable

    def is_globally_disabled(self) -> bool:
        """Check if progress bars are globally disabled."""
        return self._global_disable or os.environ.get(
            "PYBAYERSMLM_NO_PROGRESS", ""
        ).lower() in ("1", "true", "yes")


# Global progress bar manager instance
progress_manager = ProgressBarManager()


def set_progress_enabled(enabled: bool):
    """
    Globally enable or disable progress bars for the entire package.

    Args:
        enabled: True to enable progress bars, False to disable

    Example:
        # Disable progress bars for batch processing
        set_progress_enabled(False)

        # Re-enable for interactive use
        set_progress_enabled(True)
    """
    progress_manager.set_global_disable(not enabled)


# Convenience functions for common use cases
def fitting_progress_bar(total: int, **kwargs) -> "contextmanager":
    """Progress bar optimised for fitting operations."""
    kwargs.setdefault("desc", "Fitting puncta")
    kwargs.setdefault("colour", "green")
    return clean_progress_bar(total=total, **kwargs)


def analysis_progress_bar(total: int, **kwargs) -> "contextmanager":
    """Progress bar optimised for analysis operations."""
    kwargs.setdefault("desc", "Analysing data")
    kwargs.setdefault("colour", "blue")
    return clean_progress_bar(total=total, **kwargs)


# For backward compatibility, export common tqdm functionality
__all__ = [
    "clean_progress_bar",
    "ProgressBarConfig",
    "ProgressBarManager",
    "progress_manager",
    "set_progress_enabled",
    "fitting_progress_bar",
    "analysis_progress_bar",
]
