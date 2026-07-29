#!/usr/bin/env python3
"""
Test runner for pyS3M unit tests.

This script provides a convenient way to run all unit tests
and generate coverage reports.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"


def run_all_tests(verbose=False, coverage=False):
    """Run all unit tests in the unit_tests directory."""

    print("=" * 60)
    print("Running pyS3M Unit Tests")
    print("=" * 60)

    # Get all test files
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob("test_*.py"))

    if not test_files:
        print("No test files found!")
        return False

    print(f"Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  - {test_file.name}")
    print()

    # Run each test file
    all_passed = True
    results = {}

    for test_file in test_files:
        print(f"Running {test_file.name}...")
        print("-" * 40)

        try:
            # Run the test file as a script
            result = subprocess.run(
                [sys.executable, str(test_file)], capture_output=False, cwd=test_dir
            )

            success = result.returncode == 0
            results[test_file.name] = success

            if not success:
                all_passed = False
                print(f"❌ {test_file.name} FAILED")
            else:
                print(f"✅ {test_file.name} PASSED")

        except Exception as e:
            print(f"❌ {test_file.name} ERROR: {e}")
            results[test_file.name] = False
            all_passed = False

        print()

    # Summary
    print("=" * 60)
    print("Test Summary:")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<30} {status}")

    print()
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check output above for details.")

    return all_passed


def run_specific_test(test_name, verbose=False):
    """Run a specific test file."""

    test_dir = Path(__file__).parent
    test_file = test_dir / test_name

    if not test_file.exists():
        print(f"Test file not found: {test_name}")
        return False

    print(f"Running {test_name}...")
    print("=" * 60)

    try:
        result = subprocess.run([sys.executable, str(test_file)], cwd=test_dir)

        return result.returncode == 0

    except Exception as e:
        print(f"Error running test: {e}")
        return False


def setup_pytest():
    """Set up pytest configuration if available."""

    try:
        import pytest

        print("pytest is available. You can also run:")
        print(f"  pytest {Path(__file__).parent}")
        print("  pytest --cov=src --cov-report=html")
        return True
    except ImportError:
        print("pytest not available. Install with: pip install pytest pytest-cov")
        return False


def main():
    """Main test runner."""

    parser = argparse.ArgumentParser(description="Run pyS3M unit tests")
    parser.add_argument("test", nargs="?", help="Specific test file to run")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "-c", "--coverage", action="store_true", help="Generate coverage report"
    )
    parser.add_argument("--list", action="store_true", help="List available tests")

    args = parser.parse_args()

    # List tests
    if args.list:
        test_dir = Path(__file__).parent
        test_files = list(test_dir.glob("test_*.py"))
        print("Available tests:")
        for test_file in test_files:
            print(f"  - {test_file.name}")
        return

    # Check for pytest
    setup_pytest()
    print()

    # Run specific test
    if args.test:
        success = run_specific_test(args.test, args.verbose)
        sys.exit(0 if success else 1)

    # Run all tests
    success = run_all_tests(args.verbose, args.coverage)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
