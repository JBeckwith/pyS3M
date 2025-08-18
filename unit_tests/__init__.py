"""
Unit Tests for pyBayerSMLM

This package contains unit tests for the pyBayerSMLM codebase.
Tests are organized by module and functionality.

Usage:
    # Run all tests
    python -m pytest unit_tests/
    
    # Run specific test file
    python unit_tests/test_drift_correction.py
    
    # Run with coverage
    python -m pytest unit_tests/ --cov=src --cov-report=html
"""

import sys
import os

# Add src directory to Python path for imports
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)