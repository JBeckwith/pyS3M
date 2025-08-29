#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard imports header for pyBayerSMLM modules

This provides a standardised way to import common dependencies and set up
module paths for consistent imports across the codebase.

Usage:
    from StandardImports import *

Created on August 29, 2025  
@author: Claude Code
"""

# Standard library imports
import os
import sys
from pathlib import Path

# Set up module path for src/ imports
module_dir = os.path.abspath(os.path.dirname(__file__))
if module_dir not in sys.path:
    sys.path.append(module_dir)

# Scientific computing imports (standard pattern)
import numpy as np
import pandas as pd
import scipy

# Matplotlib with standard backend for batch processing
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for memory efficiency
import matplotlib.pyplot as plt

# Optional imports with error handling
try:
    import polars as pl
except ImportError:
    pl = None

try:
    from sklearn import *

    sklearn_available = True
except ImportError:
    sklearn_available = False

try:
    import dask.array as da
except ImportError:
    da = None

# Project-specific imports
from Constants import CalibrationConstants, ProcessingConstants, DefaultParameters

__all__ = [
    "np",
    "pd",
    "scipy",
    "plt",
    "matplotlib",
    "pl",
    "da",
    "Path",
    "os",
    "sys",
    "module_dir",
    "CalibrationConstants",
    "ProcessingConstants",
    "DefaultParameters",
]
