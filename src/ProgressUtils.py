#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Progress bar utilities for pyBayerSMLM.

This module provides clean, consistent progress bar functionality across the
entire pyBayerSMLM package using tqdm with proper cleanup and context management.

Created on August 21, 2025
@author: Claude Code Assistant
"""

import sys
from contextlib import contextmanager
from typing import Optional, Any, Dict, Union, Iterable
import os

try:
    # Check if we're in a Jupyter notebook
    if "ipykernel" in sys.modules:
        from tqdm.notebook import tqdm as notebook_tqdm
        from tqdm import tqdm as text_tqdm

        NOTEBOOK_ENV = True
    else:
        from tqdm import tqdm as text_tqdm
        notebook_tqdm = None

        NOTEBOOK_ENV = False
except ImportError:
    # Fallback if tqdm not available
    from tqdm import tqdm as text_tqdm
    notebook_tqdm = None

    NOTEBOOK_ENV = False

# Default to text-based tqdm for better compatibility with carriage return updates
base_tqdm = text_tqdm


class ProgressBarConfig:
    """Configuration constants for consistent progress bar styling."""

    # Default styling
    DEFAULT_LEAVE = False  # Don't leave progress bars after completion in terminal
    DEFAULT_DYNAMIC_NCOLS = True  # Adapt to terminal width
    DEFAULT_MINITERS = 1  # Update frequency
    DEFAULT_MININTERVAL = 0.05  # Minimum time between updates (seconds) - faster for notebooks

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
        base_kwargs["leave"] = False  # Don't leave progress bars to avoid line interference
            
        return base_kwargs

    @classmethod
    def get_styled_kwargs(cls, colour: Optional[str] = None) -> Dict[str, Any]:
        """Get styled keyword arguments with color support."""
        kwargs = cls.get_default_kwargs()

        # Add styling if terminal supports it
        if colour and hasattr(base_tqdm, "_supports_color"):
            kwargs["colour"] = colour
            kwargs["bar_format"] = cls.DEFAULT_BAR_FORMAT

        return kwargs


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

    Examples:
        # Basic usage with iterable
        with clean_progress_bar(range(100), desc="Processing items") as pbar:
            for item in pbar:
                # Process item
                pass

        # Manual progress updates
        with clean_progress_bar(total=100, desc="Fitting puncta") as pbar:
            for i in range(100):
                # Do work
                process_item(i)
                pbar.update(1)

        # Nested progress bars
        with clean_progress_bar(range(10), desc="Outer", position=0) as outer:
            for i in outer:
                with clean_progress_bar(range(5), desc="Inner", position=1) as inner:
                    for j in inner:
                        # Process
                        pass
    """
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


def progress_wrapper(
    func, iterable: Iterable, desc: str = "Processing", **progress_kwargs
):
    """
    Generator function that applies a function to an iterable with progress tracking.

    Args:
        func: Function to apply to each item
        iterable: Items to process
        desc: Progress bar description
        **progress_kwargs: Additional progress bar arguments

    Yields:
        Generator yielding (item, result) tuples

    Example:
        items = [1, 2, 3, 4, 5]

        for item, result in progress_wrapper(lambda x: x**2, items, "Squaring numbers"):
            # Process results
            pass
    """
    with clean_progress_bar(iterable=iterable, desc=desc, **progress_kwargs) as pbar:
        for item in pbar:
            result = func(item)
            yield item, result


def silent_progress_bar(*args, **kwargs):
    """
    Create a silent progress bar that can be used when progress display is disabled.

    This is useful for maintaining the same API when progress bars should be
    conditionally disabled (e.g., in non-interactive environments).

    Returns:
        tqdm object with disabled output
    """
    kwargs["disable"] = True
    kwargs["leave"] = False
    return base_tqdm(*args, **kwargs)


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

    def get_progress_bar(self, *args, **kwargs):
        """Get a progress bar with global settings applied."""
        if self.is_globally_disabled():
            kwargs["disable"] = True

        # Apply default configuration
        final_kwargs = self._default_config.copy()
        final_kwargs.update(kwargs)

        return base_tqdm(*args, **final_kwargs)

    @contextmanager
    def managed_progress_bar(self, *args, **kwargs):
        """Context manager for managed progress bars."""
        pbar = None
        try:
            pbar = self.get_progress_bar(*args, **kwargs)
            yield pbar
        finally:
            if pbar is not None:
                pbar.close()
            sys.stdout.flush()


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


def is_progress_enabled() -> bool:
    """Check if progress bars are currently enabled."""
    return not progress_manager.is_globally_disabled()


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


def io_progress_bar(total: int, **kwargs) -> "contextmanager":
    """Progress bar optimised for I/O operations."""
    kwargs.setdefault("desc", "Processing files")
    kwargs.setdefault("colour", "cyan")
    return clean_progress_bar(total=total, **kwargs)


def simulation_progress_bar(total: int, **kwargs) -> "contextmanager":
    """Progress bar optimised for simulation operations."""
    kwargs.setdefault("desc", "Running simulation")
    kwargs.setdefault("colour", "magenta")
    return clean_progress_bar(total=total, **kwargs)


# For backward compatibility, export common tqdm functionality
__all__ = [
    "clean_progress_bar",
    "progress_wrapper",
    "ProgressBarConfig",
    "ProgressBarManager",
    "progress_manager",
    "set_progress_enabled",
    "is_progress_enabled",
    "fitting_progress_bar",
    "analysis_progress_bar",
    "io_progress_bar",
    "simulation_progress_bar",
    "silent_progress_bar",
    "NOTEBOOK_ENV",
]
