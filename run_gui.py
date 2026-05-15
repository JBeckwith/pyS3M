#!/usr/bin/env python3
"""Entry point for the pyBayerSMLM desktop GUI."""
import sys
import os

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Must be set before any other matplotlib import
import matplotlib
matplotlib.use("QtAgg")

from gui.app import run

if __name__ == "__main__":
    run()
