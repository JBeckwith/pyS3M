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
from typing import Dict, Optional, Any, Callable
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


# Global instance for use across pyS3M
_import_manager = ImportManager()


# Convenience functions for common use cases
def get_module(module_name: str) -> Optional[Any]:
    """Get a module from the global import manager."""
    return _import_manager.get_module(module_name)


def is_available(module_name: str) -> bool:
    """Check if a module is available."""
    return _import_manager.is_available(module_name)


if __name__ == "__main__":
    print(get_module("numpy"))
