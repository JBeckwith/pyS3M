"""
ImportManager.py

Centralised import management for pyS3M.
Handles optional dependencies, module availability checks, and consistent import patterns.

:authors: Claude Code
:copyright: Copyright (c) 2025 pyS3M
"""

import sys
from pathlib import Path
import warnings
from typing import Dict, Optional, Any, List, Callable
from dataclasses import dataclass
from enum import Enum


class ImportStatus(Enum):
    """Status of module import attempts."""

    AVAILABLE = "available"
    MISSING = "missing"
    ERROR = "error"


@dataclass
class ModuleInfo:
    """Information about a module and its import status."""

    name: str
    module: Optional[Any]
    status: ImportStatus
    error_message: Optional[str] = None
    fallback: Optional[Any] = None
    required: bool = False


class ImportManager:
    """Centralised manager for handling imports across pyS3M.

    This class provides a single point of control for managing optional
    dependencies, handling import failures gracefully, and providing
    consistent error messages across the codebase.
    """

    def __init__(self):
        """Initialize the import manager."""
        self.modules: Dict[str, ModuleInfo] = {}
        self._setup_path()
        self._register_core_modules()

    def _setup_path(self):
        """Setup sys.path for local module imports."""
        _dir = str(Path(__file__).parent)
        if _dir not in sys.path:
            sys.path.append(_dir)

    def _register_core_modules(self):
        """Register core pyS3M modules and common dependencies.

        NOTE: Only registers external packages here. Local modules are registered
        lazily on-demand to avoid circular imports.
        """

        # Core scientific libraries (usually required)
        core_modules = [
            ("numpy", "np", True),
            ("matplotlib", None, True),
            ("matplotlib.pyplot", "plt", True),
            ("scipy", None, True),
        ]

        for module_name, alias, required in core_modules:
            self.register_module(module_name, alias=alias, required=required)

        # Optional scientific libraries
        optional_modules = [
            ("datashader", "ds"),
            ("pandas", "pd"),
            ("colorcet", "cc"),
            ("seaborn", "sns"),
            ("plotly", None),
            ("bokeh", None),
        ]

        for module_name, alias in optional_modules:
            self.register_module(module_name, alias=alias, required=False)

        # Register common matplotlib submodules
        self.register_module("matplotlib.colors", required=False)
        self.register_module("matplotlib.patches", required=False)

        # Register scipy submodules
        self.register_module("scipy.interpolate", required=False)

        # Register numpy submodules
        self.register_module("numpy.fft", required=True)

        # NOTE: Local pyS3M modules (PlottingFunctions, render, imageprocess, etc.)
        # are NOT registered here to avoid circular imports. They will be loaded
        # lazily on first access via get_module().

    def register_module(
        self,
        module_name: str,
        alias: Optional[str] = None,
        required: bool = False,
        fallback: Optional[Any] = None,
        custom_import: Optional[Callable] = None,
    ) -> ModuleInfo:
        """Register a module for managed importing.

        Args:
            module_name: Name of the module to import
            alias: Optional alias to use when importing (e.g., 'np' for numpy)
            required: Whether this module is required for basic functionality
            fallback: Fallback object to use if import fails
            custom_import: Custom import function for complex cases

        Returns:
            ModuleInfo object with import status
        """
        try:
            if custom_import:
                module = custom_import()
            else:
                if alias:
                    # Import with alias (e.g., import numpy as np)
                    module = __import__(module_name, fromlist=[""])
                    globals()[alias] = module
                else:
                    module = __import__(module_name, fromlist=[""])

            module_info = ModuleInfo(
                name=module_name,
                module=module,
                status=ImportStatus.AVAILABLE,
                required=required,
                fallback=fallback,
            )

        except ImportError as e:
            error_msg = f"Could not import {module_name}: {e}"

            if required:
                # For required modules, we still want to raise the error
                warnings.warn(f"CRITICAL: {error_msg}", UserWarning)
                module = None
            else:
                # For optional modules, use fallback and warn
                warnings.warn(f"Optional dependency missing: {error_msg}", UserWarning)
                module = fallback

            module_info = ModuleInfo(
                name=module_name,
                module=module,
                status=ImportStatus.MISSING,
                error_message=error_msg,
                required=required,
                fallback=fallback,
            )

        except Exception as e:
            error_msg = f"Error importing {module_name}: {e}"
            warnings.warn(error_msg, UserWarning)

            module_info = ModuleInfo(
                name=module_name,
                module=fallback,
                status=ImportStatus.ERROR,
                error_message=error_msg,
                required=required,
                fallback=fallback,
            )

        self.modules[module_name] = module_info
        return module_info

    def get_module(self, module_name: str) -> Optional[Any]:
        """Get a registered module by name.

        Uses lazy loading: if module not registered, attempts to import it.
        This avoids circular import issues with local modules.

        Args:
            module_name: Name of the module to retrieve

        Returns:
            The imported module, fallback, or None if not available
        """
        # If not registered, try lazy loading
        if module_name not in self.modules:
            # Try to import on-demand (lazy loading)
            try:
                module = __import__(module_name, fromlist=[""])
                module_info = ModuleInfo(
                    name=module_name,
                    module=module,
                    status=ImportStatus.AVAILABLE,
                    required=False,
                )
                self.modules[module_name] = module_info
                return module
            except ImportError as e:
                # Module not available, return None
                warnings.warn(f"Module {module_name} could not be imported: {e}")
                return None

        module_info = self.modules[module_name]
        return module_info.module

    def is_available(self, module_name: str) -> bool:
        """Check if a module is available for use.

        Args:
            module_name: Name of the module to check

        Returns:
            True if module is available, False otherwise
        """
        if module_name not in self.modules:
            return False

        module_info = self.modules[module_name]
        return module_info.status == ImportStatus.AVAILABLE

    def get_status_report(self) -> str:
        """Generate a status report of all registered modules.

        Returns:
            Formatted string with module status information
        """
        report_lines = ["Module Import Status Report", "=" * 30]

        available = []
        missing = []
        errors = []

        for name, info in self.modules.items():
            status_symbol = {
                ImportStatus.AVAILABLE: "✅",
                ImportStatus.MISSING: "❌",
                ImportStatus.ERROR: "⚠️",
            }[info.status]

            line = f"{status_symbol} {name}"
            if info.required:
                line += " (required)"

            if info.status == ImportStatus.AVAILABLE:
                available.append(line)
            elif info.status == ImportStatus.MISSING:
                missing.append(line)
            else:
                errors.append(line)

        if available:
            report_lines.extend(["", "Available:"] + available)
        if missing:
            report_lines.extend(["", "Missing:"] + missing)
        if errors:
            report_lines.extend(["", "Errors:"] + errors)

        return "\n".join(report_lines)

    def require_module(self, module_name: str, feature_name: str = "") -> Any:
        """Get a module that is required for a specific feature.

        Args:
            module_name: Name of the required module
            feature_name: Name of the feature that requires this module (for error messages)

        Returns:
            The imported module

        Raises:
            RuntimeError: If the module is not available
        """
        if not self.is_available(module_name):
            feature_desc = f" for {feature_name}" if feature_name else ""
            raise RuntimeError(
                f"Module '{module_name}' is required{feature_desc} but not available"
            )

        return self.get_module(module_name)

    def safe_import_with_fallback(
        self,
        module_name: str,
        fallback_action: Optional[Callable] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Any]:
        """Safely import a module with optional fallback action.

        Args:
            module_name: Name of the module to import
            fallback_action: Function to call if module is not available
            error_message: Custom error message to display

        Returns:
            The module if available, or result of fallback_action
        """
        if self.is_available(module_name):
            return self.get_module(module_name)

        if error_message:
            print(f"⚠️ {error_message}")

        if fallback_action:
            return fallback_action()

        return None

    def create_plotting_environment(self) -> Dict[str, Any]:
        """Create a plotting environment with available modules.

        Returns:
            Dictionary containing available plotting modules
        """
        env = {}

        # Core plotting
        if self.is_available("matplotlib.pyplot"):
            env["plt"] = self.get_module("matplotlib.pyplot")

        # Enhanced plotting
        optional_plotting = [
            ("seaborn", "sns"),
            ("plotly", "plotly"),
            ("datashader", "ds"),
            ("colorcet", "cc"),
        ]

        for module_name, key in optional_plotting:
            if self.is_available(module_name):
                env[key] = self.get_module(module_name)

        return env

    def setup_datashader_environment(self) -> Optional[Dict[str, Any]]:
        """Setup datashader environment if available.

        Returns:
            Dictionary with datashader modules, or None if not available
        """
        required_modules = ["datashader", "pandas", "colorcet"]

        if not all(self.is_available(mod) for mod in required_modules):
            return None

        return {
            "ds": self.get_module("datashader"),
            "pd": self.get_module("pandas"),
            "cc": self.get_module("colorcet"),
        }


# Global instance for use across pyS3M
_import_manager = ImportManager()


# Convenience functions for common use cases
def get_module(module_name: str) -> Optional[Any]:
    """Get a module from the global import manager."""
    return _import_manager.get_module(module_name)


def is_available(module_name: str) -> bool:
    """Check if a module is available."""
    return _import_manager.is_available(module_name)


def require_module(module_name: str, feature_name: str = "") -> Any:
    """Require a module for a specific feature."""
    return _import_manager.require_module(module_name, feature_name)


def get_status_report() -> str:
    """Get status report of all modules."""
    return _import_manager.get_status_report()


def safe_import(
    module_name: str,
    fallback_action: Optional[Callable] = None,
    error_message: Optional[str] = None,
) -> Optional[Any]:
    """Safely import a module with fallback."""
    return _import_manager.safe_import_with_fallback(
        module_name, fallback_action, error_message
    )


# Pre-configured environment setups
def get_plotting_env() -> Dict[str, Any]:
    """Get available plotting environment."""
    return _import_manager.create_plotting_environment()


def get_datashader_env() -> Optional[Dict[str, Any]]:
    """Get datashader environment if available."""
    return _import_manager.setup_datashader_environment()


# Module-specific convenience functions
def get_datashader():
    """Get datashader module if available."""
    return get_module("datashader")


def get_pandas():
    """Get pandas module if available."""
    return get_module("pandas")


def get_plotting_functions():
    """Get PlottingBase module (replaces deprecated PlottingFunctions)."""
    return get_module("PlottingBase")


def get_postprocess():
    """Get postprocess module if available."""
    return get_module("postprocess")


def get_render():
    """Get render module if available."""
    return get_module("render")


def get_imageprocess():
    """Get imageprocess module if available."""
    return get_module("imageprocess")


if __name__ == "__main__":
    # Print status report when run directly
    print(get_status_report())
