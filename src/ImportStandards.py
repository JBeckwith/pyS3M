#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import Standards for pyBayerSMLM

This module defines standardised import patterns and utilities for consistent
module loading across the codebase.

Created on August 29, 2025
@author: Claude Code
"""

import os
import sys
from pathlib import Path
import logging


def setup_module_path() -> str:
    """
    Standardised module path setup for src/ directory imports.

    Returns:
        str: Absolute path to the module directory
    """
    module_dir = os.path.abspath(os.path.dirname(__file__))
    if module_dir not in sys.path:
        sys.path.append(module_dir)
    return module_dir


def setup_project_path() -> str:
    """
    Setup project root path for accessing notebooks and data directories.

    Returns:
        str: Absolute path to the project root directory
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    return project_dir


# Standard import aliases (for documentation)
STANDARD_IMPORTS = {
    "numpy": "import numpy as np",
    "pandas": "import pandas as pd",
    "matplotlib_pyplot": "import matplotlib.pyplot as plt",
    "matplotlib": "import matplotlib",
    "scipy": "import scipy",
    "sklearn": "from sklearn import *",
    "pathlib": "from pathlib import Path",
}

# Legacy Picasso module imports (deprecated - being standardized)
LEGACY_IMPORTS = {
    "numpy_legacy": "import numpy as np",  # Standardized from _np
    "matplotlib_legacy": "import matplotlib.pyplot as plt",  # Standardized from _plt
}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get standardised logger for pyBayerSMLM modules.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
