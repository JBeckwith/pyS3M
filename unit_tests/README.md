# pyS3M Unit Tests

This directory contains unit tests for the pyS3M package. The tests are designed to validate functionality, catch regressions, and ensure code quality.

## Test Structure

```
unit_tests/
├── __init__.py                # Package initialization  
├── README.md                  # This file
├── pytest.ini                # pytest configuration
├── test_requirements.txt      # Testing dependencies
├── run_tests.py              # Test runner script
├── test_drift_correction.py  # Drift correction tests
└── [future test files]       # Additional test modules
```

## Running Tests

### Method 1: Using the Test Runner Script

```bash
# Run all tests
python unit_tests/run_tests.py

# Run specific test
python unit_tests/run_tests.py test_drift_correction.py

# List available tests
python unit_tests/run_tests.py --list

# Verbose output
python unit_tests/run_tests.py -v
```

### Method 2: Using pytest (Recommended)

First, install testing requirements:
```bash
pip install -r unit_tests/test_requirements.txt
```

Run tests:
```bash
# Run all tests
pytest unit_tests/

# Run with coverage
pytest unit_tests/ --cov=src --cov-report=html

# Run specific test file
pytest unit_tests/test_drift_correction.py

# Run with markers
pytest unit_tests/ -m "not slow"  # Skip slow tests
pytest unit_tests/ -m "unit"      # Only unit tests
```

### Method 3: Direct Script Execution

```bash
# Run individual test files directly
python unit_tests/test_drift_correction.py
```

## Test Categories

Tests are organized into categories using pytest markers:

- **`@pytest.mark.unit`**: Fast unit tests (< 1 second)
- **`@pytest.mark.integration`**: Integration tests (1-10 seconds)  
- **`@pytest.mark.slow`**: Slow tests (> 10 seconds)
- **`@pytest.mark.requires_gpu`**: Tests requiring GPU hardware
- **`@pytest.mark.requires_render`**: Tests requiring render modules

## Test Files

### `test_drift_correction.py`

Tests for the unified drift correction module (`DriftCorrectionFunctions.py`):

- ✅ Strategy pattern factory
- ✅ Parameter validation
- ✅ Method selection logic
- ✅ AIM drift correction (placeholder)
- ⚠️ RCC drift correction (requires render modules)
- ✅ Auto method selection
- ✅ Backward compatibility functions

## Writing New Tests

When adding new test files, follow these conventions:

### File Naming
- Test files: `test_[module_name].py`
- Test classes: `Test[ClassName]`
- Test functions: `test_[function_name]`

### Test Structure
```python
#!/usr/bin/env python3
"""
Test module for [ModuleName].

Description of what this module tests.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from ModuleToTest import ClassToTest


class TestClassName:
    """Test class for [ClassName]."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing test data."""
        return create_test_data()
    
    @pytest.mark.unit
    def test_basic_functionality(self, sample_data):
        """Test basic functionality."""
        # Test implementation
        assert True
    
    @pytest.mark.integration
    def test_integration_workflow(self, sample_data):
        """Test complete workflow."""
        # Integration test
        assert True
    
    @pytest.mark.slow
    def test_performance(self, sample_data):
        """Test performance with large data."""
        # Performance test
        assert True


def test_standalone_function():
    """Test standalone functions."""
    assert True
```

### Test Data
- Use fixtures for reusable test data
- Create realistic synthetic data when possible
- Keep test data small for fast execution
- Use `@pytest.mark.slow` for tests with large datasets

### Assertions
- Use descriptive assertion messages
- Test both success and failure cases
- Validate input/output types and shapes
- Check edge cases and boundary conditions

## Coverage Reports

Generate HTML coverage reports:

```bash
pytest unit_tests/ --cov=src --cov-report=html
open htmlcov/index.html  # View coverage report
```

Target coverage goals:
- **New modules**: >90% line coverage
- **Refactored modules**: >85% line coverage  
- **Legacy modules**: >70% line coverage

## Continuous Integration

Tests should pass on:
- Python 3.8, 3.9, 3.10, 3.11
- Linux, macOS, Windows (where applicable)
- With and without optional dependencies

## Performance Testing

For performance-critical modules, include benchmark tests:

```python
@pytest.mark.benchmark
def test_performance_benchmark(benchmark):
    """Benchmark performance of critical function."""
    result = benchmark(expensive_function, test_data)
    assert result is not None
```

## Mock Testing

Use pytest-mock for external dependencies:

```python
def test_with_mock_render(mocker):
    """Test with mocked render module."""
    mock_render = mocker.patch('DriftCorrectionFunctions._render')
    # Test implementation
```

## Test Data Files

For tests requiring data files:

```
unit_tests/
├── test_data/
│   ├── sample_localizations.csv
│   ├── test_image.tiff
│   └── metadata.yaml
└── fixtures/
    ├── __init__.py
    └── data_generators.py
```

## Debugging Failed Tests

```bash
# Run with detailed output
pytest unit_tests/ -vvv

# Stop at first failure
pytest unit_tests/ -x

# Run last failed tests only
pytest unit_tests/ --lf

# Run with pdb debugger
pytest unit_tests/ --pdb
```

## Contributing

When contributing tests:

1. **Write tests first** (TDD approach recommended)
2. **Test both success and failure** paths
3. **Use appropriate markers** for test categorization
4. **Include docstrings** explaining what is tested
5. **Keep tests fast** (use mocks for slow operations)
6. **Update this README** when adding new test categories

## Future Test Modules

Planned test modules to be added:

- `test_image_analysis.py` - ImageAnalysisFunctions tests
- `test_spectral_functions.py` - SpectralFunctions tests  
- `test_plotting_functions.py` - PlottingFunctions tests
- `test_psf_functions.py` - PSFFunctions tests
- `test_integration.py` - End-to-end workflow tests
- `test_performance.py` - Performance benchmarks