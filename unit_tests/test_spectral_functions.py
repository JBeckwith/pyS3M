#!/usr/bin/env python3
"""
Test module for SpectralFunctions.

Tests the refactored SpectralFunctions module including strategy patterns,
database operations, and spectral data processing.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock
import sqlite3

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from SpectralFunctions import (
    Spectral_Funcs,
    SpectrumProcessor,
    DyeSpectrumProcessor,
    FilterSpectrumProcessor,
    DatabaseQueryHandler,
    SpectralDataType,
    SpectralConstants,
)


class TestSpectralDataType:
    """Test SpectralDataType enum."""

    @pytest.mark.unit
    def test_enum_values(self):
        """Test that enum has expected values."""
        assert SpectralDataType.DYE.value == "dye"
        assert SpectralDataType.FILTER.value == "filter"

    @pytest.mark.unit
    def test_enum_completeness(self):
        """Test that enum contains expected number of values."""
        enum_values = list(SpectralDataType)
        assert len(enum_values) == 2
        assert SpectralDataType.DYE in enum_values
        assert SpectralDataType.FILTER in enum_values


class TestSpectralConstants:
    """Test SpectralConstants configuration."""

    @pytest.mark.unit
    def test_constants_exist(self):
        """Test that required constants are defined."""
        assert hasattr(SpectralConstants, "SPECTRA_DIR")
        assert hasattr(SpectralConstants, "DB_PATH")
        assert hasattr(SpectralConstants, "WAVELENGTH_RANGE")
        assert hasattr(SpectralConstants, "WAVELENGTH_STEP")

    @pytest.mark.unit
    def test_wavelength_config(self):
        """Test wavelength configuration values."""
        assert len(SpectralConstants.WAVELENGTH_RANGE) == 2
        assert (
            SpectralConstants.WAVELENGTH_RANGE[0]
            < SpectralConstants.WAVELENGTH_RANGE[1]
        )
        assert SpectralConstants.WAVELENGTH_STEP > 0


class TestDatabaseQueryHandler:
    """Test DatabaseQueryHandler class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name

        # Create test database with sample data
        conn = sqlite3.connect(temp_path)
        cursor = conn.cursor()

        # Create test tables
        cursor.execute(
            """
            CREATE TABLE dyes (
                id INTEGER PRIMARY KEY,
                name TEXT,
                wavelength REAL,
                value REAL
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE filters (
                id INTEGER PRIMARY KEY,
                name TEXT,
                wavelength REAL,
                transmission REAL
            )
        """
        )

        # Insert test data
        test_dyes = [
            (1, "ATTO488", 488.0, 0.8),
            (2, "ATTO565", 565.0, 0.9),
            (3, "ATTO647N", 647.0, 0.7),
        ]

        test_filters = [
            (1, "BP525/50", 525.0, 0.95),
            (2, "BP585/40", 585.0, 0.92),
            (3, "BP700/75", 700.0, 0.88),
        ]

        cursor.executemany("INSERT INTO dyes VALUES (?,?,?,?)", test_dyes)
        cursor.executemany("INSERT INTO filters VALUES (?,?,?,?)", test_filters)

        conn.commit()
        conn.close()

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.mark.unit
    def test_handler_initialization(self, temp_db_path):
        """Test DatabaseQueryHandler initialization."""
        handler = DatabaseQueryHandler(temp_db_path)
        assert handler.db_path == temp_db_path

    @pytest.mark.unit
    def test_query_dye_data(self, temp_db_path):
        """Test querying dye data."""
        handler = DatabaseQueryHandler(temp_db_path)

        # Mock the actual query since we don't have the real database structure
        with patch.object(handler, "_execute_query") as mock_query:
            mock_query.return_value = pd.DataFrame(
                {"wavelength": [488.0, 565.0, 647.0], "value": [0.8, 0.9, 0.7]}
            )

            result = handler.query_dye_data("ATTO488")
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            mock_query.assert_called_once()

    @pytest.mark.unit
    def test_query_filter_data(self, temp_db_path):
        """Test querying filter data."""
        handler = DatabaseQueryHandler(temp_db_path)

        with patch.object(handler, "_execute_query") as mock_query:
            mock_query.return_value = pd.DataFrame(
                {
                    "wavelength": [500.0, 550.0, 600.0],
                    "transmission": [0.95, 0.98, 0.92],
                }
            )

            result = handler.query_filter_data("BP525/50")
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            mock_query.assert_called_once()

    @pytest.mark.unit
    def test_connection_context_manager(self, temp_db_path):
        """Test database connection context manager."""
        handler = DatabaseQueryHandler(temp_db_path)

        # Test that connection is properly opened and closed
        with handler._get_connection() as conn:
            assert conn is not None
            # Connection should be open
            conn.execute("SELECT 1").fetchone()


class TestDyeSpectrumProcessor:
    """Test DyeSpectrumProcessor implementation."""

    @pytest.fixture
    def processor(self):
        """Create DyeSpectrumProcessor instance."""
        return DyeSpectrumProcessor()

    @pytest.fixture
    def mock_handler(self):
        """Create mock DatabaseQueryHandler."""
        handler = Mock()
        handler.query_dye_data.return_value = pd.DataFrame(
            {"wavelength": np.arange(400, 700, 10), "extinction": np.random.random(30)}
        )
        return handler

    @pytest.mark.unit
    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor is not None
        assert isinstance(processor, SpectrumProcessor)
        assert processor.data_type == SpectralDataType.DYE

    @pytest.mark.unit
    def test_process_spectrum_data(self, processor, mock_handler):
        """Test spectrum data processing."""
        with patch.object(
            processor, "_get_database_handler", return_value=mock_handler
        ):
            result = processor.process_spectrum("ATTO488")

            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert "wavelength" in result.columns
            mock_handler.query_dye_data.assert_called_once_with("ATTO488")

    @pytest.mark.unit
    def test_validate_data_columns(self, processor):
        """Test data column validation."""
        # Valid data
        valid_data = pd.DataFrame(
            {"wavelength": [400, 500, 600], "extinction": [0.1, 0.8, 0.3]}
        )

        # Should not raise exception
        processor._validate_data(valid_data)

        # Invalid data missing wavelength
        invalid_data = pd.DataFrame({"extinction": [0.1, 0.8, 0.3]})

        with pytest.raises(ValueError, match="Missing required column"):
            processor._validate_data(invalid_data)


class TestFilterSpectrumProcessor:
    """Test FilterSpectrumProcessor implementation."""

    @pytest.fixture
    def processor(self):
        """Create FilterSpectrumProcessor instance."""
        return FilterSpectrumProcessor()

    @pytest.fixture
    def mock_handler(self):
        """Create mock DatabaseQueryHandler."""
        handler = Mock()
        handler.query_filter_data.return_value = pd.DataFrame(
            {
                "wavelength": np.arange(400, 700, 10),
                "transmission": np.random.random(30),
            }
        )
        return handler

    @pytest.mark.unit
    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor is not None
        assert isinstance(processor, SpectrumProcessor)
        assert processor.data_type == SpectralDataType.FILTER

    @pytest.mark.unit
    def test_process_spectrum_data(self, processor, mock_handler):
        """Test spectrum data processing."""
        with patch.object(
            processor, "_get_database_handler", return_value=mock_handler
        ):
            result = processor.process_spectrum("BP525/50")

            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert "wavelength" in result.columns
            mock_handler.query_filter_data.assert_called_once_with("BP525/50")

    @pytest.mark.unit
    def test_validate_transmission_data(self, processor):
        """Test transmission data validation."""
        # Valid transmission data
        valid_data = pd.DataFrame(
            {"wavelength": [400, 500, 600], "transmission": [0.1, 0.95, 0.8]}
        )

        processor._validate_data(valid_data)

        # Invalid transmission values
        invalid_data = pd.DataFrame(
            {
                "wavelength": [400, 500, 600],
                "transmission": [0.1, 1.5, 0.8],  # 1.5 > 1.0
            }
        )

        with pytest.raises(ValueError, match="Invalid transmission values"):
            processor._validate_data(invalid_data)


class TestSpectralFunctions:
    """Test main Spectral_Funcs class."""

    @pytest.fixture
    def spectral_functions(self):
        """Create Spectral_Funcs instance."""
        return Spectral_Funcs()

    @pytest.mark.unit
    def test_class_initialization(self, spectral_functions):
        """Test class initialization."""
        assert spectral_functions is not None
        # Test that processors are properly initialized
        assert hasattr(spectral_functions, "_dye_processor")
        assert hasattr(spectral_functions, "_filter_processor")

    @pytest.mark.unit
    def test_get_processor_dye(self, spectral_functions):
        """Test getting dye processor."""
        processor = spectral_functions._get_processor(SpectralDataType.DYE)
        assert isinstance(processor, DyeSpectrumProcessor)

    @pytest.mark.unit
    def test_get_processor_filter(self, spectral_functions):
        """Test getting filter processor."""
        processor = spectral_functions._get_processor(SpectralDataType.FILTER)
        assert isinstance(processor, FilterSpectrumProcessor)

    @pytest.mark.unit
    def test_get_processor_invalid_type(self, spectral_functions):
        """Test getting processor with invalid type."""
        with pytest.raises(ValueError, match="Unsupported data type"):
            spectral_functions._get_processor("invalid")

    @pytest.mark.integration
    def test_get_dye_data_integration(self, spectral_functions):
        """Test getting dye data with mocked processor."""
        mock_data = pd.DataFrame(
            {"wavelength": np.arange(400, 700, 10), "extinction": np.random.random(30)}
        )

        with patch.object(
            spectral_functions._dye_processor,
            "process_spectrum",
            return_value=mock_data,
        ):
            result = spectral_functions.get_dye_data("ATTO488")

            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(mock_data)
            assert "wavelength" in result.columns

    @pytest.mark.integration
    def test_get_filter_data_integration(self, spectral_functions):
        """Test getting filter data with mocked processor."""
        mock_data = pd.DataFrame(
            {"wavelength": np.arange(450, 650, 5), "transmission": np.random.random(40)}
        )

        with patch.object(
            spectral_functions._filter_processor,
            "process_spectrum",
            return_value=mock_data,
        ):
            result = spectral_functions.get_filter_data("BP525/50")

            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(mock_data)
            assert "wavelength" in result.columns

    @pytest.mark.unit
    def test_backward_compatibility_method(self, spectral_functions):
        """Test legacy get_dye_or_filter_data method."""
        mock_data = pd.DataFrame(
            {"wavelength": [400, 500, 600], "value": [0.1, 0.8, 0.3]}
        )

        # Test dye data (is_dye=True)
        with patch.object(spectral_functions, "get_dye_data", return_value=mock_data):
            result = spectral_functions.get_dye_or_filter_data("ATTO488", is_dye=True)
            assert isinstance(result, pd.DataFrame)
            spectral_functions.get_dye_data.assert_called_once_with("ATTO488")

        # Test filter data (is_dye=False)
        with patch.object(
            spectral_functions, "get_filter_data", return_value=mock_data
        ):
            result = spectral_functions.get_dye_or_filter_data("BP525/50", is_dye=False)
            assert isinstance(result, pd.DataFrame)
            spectral_functions.get_filter_data.assert_called_once_with("BP525/50")


class TestSpectralProcessingWorkflow:
    """Test complete spectral processing workflows."""

    @pytest.fixture
    def spectral_functions(self):
        """Create Spectral_Funcs instance."""
        return Spectral_Funcs()

    @pytest.mark.integration
    def test_complete_dye_workflow(self, spectral_functions):
        """Test complete dye data processing workflow."""
        # Mock the entire chain from database to final result
        mock_raw_data = pd.DataFrame(
            {
                "wavelength": np.arange(400, 700, 5),
                "extinction_coeff": np.exp(
                    -((np.arange(400, 700, 5) - 488) ** 2) / (2 * 30**2)
                ),
            }
        )

        with patch.object(
            DatabaseQueryHandler, "query_dye_data", return_value=mock_raw_data
        ):
            result = spectral_functions.get_dye_data("ATTO488")

            # Verify result structure and content
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert "wavelength" in result.columns

            # Verify wavelength range is reasonable
            wavelengths = result["wavelength"].values
            assert wavelengths.min() >= 350  # Reasonable minimum
            assert wavelengths.max() <= 800  # Reasonable maximum

    @pytest.mark.integration
    def test_complete_filter_workflow(self, spectral_functions):
        """Test complete filter data processing workflow."""
        # Mock bandpass filter transmission curve
        wavelengths = np.arange(400, 700, 5)
        center = 525
        width = 50
        transmission = np.exp(-((wavelengths - center) ** 2) / (2 * (width / 4) ** 2))

        mock_raw_data = pd.DataFrame(
            {"wavelength": wavelengths, "transmission": transmission}
        )

        with patch.object(
            DatabaseQueryHandler, "query_filter_data", return_value=mock_raw_data
        ):
            result = spectral_functions.get_filter_data("BP525/50")

            # Verify result structure
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert "wavelength" in result.columns
            assert "transmission" in result.columns

            # Verify transmission values are valid (0-1 range)
            transmission_vals = result["transmission"].values
            assert np.all(transmission_vals >= 0)
            assert np.all(transmission_vals <= 1)

    @pytest.mark.integration
    def test_error_handling_workflow(self, spectral_functions):
        """Test error handling in complete workflow."""
        # Test database connection error
        with patch.object(
            DatabaseQueryHandler,
            "query_dye_data",
            side_effect=Exception("Database error"),
        ):
            with pytest.raises(Exception, match="Database error"):
                spectral_functions.get_dye_data("NonexistentDye")

        # Test empty result handling
        with patch.object(
            DatabaseQueryHandler, "query_filter_data", return_value=pd.DataFrame()
        ):
            # Should handle empty results gracefully
            result = spectral_functions.get_filter_data("NonexistentFilter")
            # Depending on implementation, might return empty DataFrame or raise exception
            assert isinstance(result, pd.DataFrame)

    @pytest.mark.integration
    def test_performance_with_large_datasets(self, spectral_functions):
        """Test performance with large spectral datasets."""
        # Create large mock dataset
        large_wavelengths = np.arange(300, 900, 0.1)  # High resolution spectrum
        large_data = pd.DataFrame(
            {
                "wavelength": large_wavelengths,
                "extinction": np.random.random(len(large_wavelengths)),
            }
        )

        with patch.object(
            DatabaseQueryHandler, "query_dye_data", return_value=large_data
        ):
            # Should handle large datasets efficiently
            import time

            start_time = time.time()

            result = spectral_functions.get_dye_data("HighResDye")

            end_time = time.time()
            processing_time = end_time - start_time

            # Verify result
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(large_data)

            # Performance should be reasonable (< 1 second for processing)
            assert processing_time < 1.0
