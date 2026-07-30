import shutil
from pathlib import Path

import pytest

TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"


@pytest.fixture
def test_output_dir():
    """Directory for test-generated figures/artifacts.

    Unlike `tmp_path` (a fresh, hidden-away system temp directory per test,
    gone the moment that test finishes), this sits inside the repo so
    generated figures are easy to find and inspect during a test run --
    but it's wiped in one go once the whole session finishes
    (`pytest_sessionfinish` below), so nothing accumulates or gets committed.
    """
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    return TEST_OUTPUT_DIR


def pytest_sessionfinish(session, exitstatus):
    """Remove all test-generated output once the full test session completes."""
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
