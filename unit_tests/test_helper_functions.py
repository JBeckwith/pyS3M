#!/usr/bin/env python3
"""
Test module for HelperFunctions.

Tests the helper functions for image analysis, database cleaning,
and file operations.
"""

import pytest
import numpy as np
import polars as pl
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from HelperFunctions import Helper_Functions


class TestHelperFunctions:
    """Test the Helper_Functions class."""

    @pytest.fixture
    def helper_functions(self):
        """Create an instance of Helper_Functions."""
        return Helper_Functions()

    @pytest.fixture
    def sample_database(self):
        """Create a sample polars DataFrame for testing database operations."""
        data = {
            "value1": ["1.5", "2.0", "3.5", "4.0"],
            "value2": ["10.1", "20.2", "30.3", "40.4"],
            "value3": ["100", "200", "300", "400"],
            "filename": ["file1.txt", "file2.txt", "file3.txt", "file4.txt"],
        }
        return pl.DataFrame(data)

    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = [
                "test_string1_data.txt",
                "test_string1_string2.csv",
                "other_string1_file.log",
                "string1_test_string2.dat",
                "no_match.txt",
                "string2_only.csv",
            ]

            for filename in test_files:
                filepath = os.path.join(temp_dir, filename)
                with open(filepath, "w") as f:
                    f.write("test content")

            yield temp_dir

    @pytest.mark.unit
    def test_class_initialization(self, helper_functions):
        """Test that the Helper_Functions class initializes properly."""
        assert helper_functions is not None
        assert hasattr(helper_functions, "clean_database")
        assert hasattr(helper_functions, "file_search")

    @pytest.mark.unit
    def test_clean_database_basic_functionality(
        self, helper_functions, sample_database
    ):
        """Test basic database cleaning functionality."""
        columns = ["value1", "value2", "value3", "filename"]

        cleaned_db = helper_functions.clean_database(sample_database, columns)

        # Check that cleaned database is still a DataFrame
        assert isinstance(cleaned_db, pl.DataFrame)

        # Check that we have the same number of rows and columns
        assert cleaned_db.shape == sample_database.shape

        # Check that numeric columns were converted to float
        for col in columns[:-1]:  # All columns except filename
            assert cleaned_db[col].dtype in [pl.Float32, pl.Float64]

        # Check that filename column remains as string
        assert cleaned_db["filename"].dtype == pl.Utf8

    @pytest.mark.unit
    def test_clean_database_numeric_conversion(self, helper_functions):
        """Test that string numbers are properly converted to floats."""
        # Create test data with string representations of numbers
        data = {
            "numeric_str1": ["1.23", "4.56", "7.89"],
            "numeric_str2": ["10", "20", "30"],
            "filename": ["file1.txt", "file2.txt", "file3.txt"],
        }
        test_db = pl.DataFrame(data)
        columns = ["numeric_str1", "numeric_str2", "filename"]

        cleaned_db = helper_functions.clean_database(test_db, columns)

        # Check that numeric columns were properly converted
        expected_values1 = [1.23, 4.56, 7.89]
        expected_values2 = [10.0, 20.0, 30.0]

        np.testing.assert_array_almost_equal(
            cleaned_db["numeric_str1"].to_numpy(), expected_values1
        )
        np.testing.assert_array_almost_equal(
            cleaned_db["numeric_str2"].to_numpy(), expected_values2
        )

    @pytest.mark.unit
    def test_clean_database_preserves_filename(self, helper_functions, sample_database):
        """Test that filename column is preserved during cleaning."""
        columns = ["value1", "value2", "value3", "filename"]

        cleaned_db = helper_functions.clean_database(sample_database, columns)

        # Check that filename values are unchanged
        original_filenames = sample_database["filename"].to_list()
        cleaned_filenames = cleaned_db["filename"].to_list()

        assert original_filenames == cleaned_filenames

    @pytest.mark.unit
    def test_clean_database_empty_dataframe(self, helper_functions):
        """Test database cleaning with empty DataFrame."""
        empty_db = pl.DataFrame({"col1": [], "col2": [], "filename": []})
        columns = ["col1", "col2", "filename"]

        cleaned_db = helper_functions.clean_database(empty_db, columns)

        assert cleaned_db.shape == (0, 3)
        assert list(cleaned_db.columns) == columns

    @pytest.mark.unit
    def test_file_search_both_strings_present(self, helper_functions, temp_directory):
        """Test file search when both search strings are present."""
        files = helper_functions.file_search(temp_directory, "string1", "string2")

        # Should find files that contain both string1 and string2
        expected_files = {"test_string1_string2.csv", "string1_test_string2.dat"}

        found_files = {os.path.basename(f) for f in files}
        assert found_files == expected_files

    @pytest.mark.unit
    def test_file_search_only_first_string(self, helper_functions, temp_directory):
        """Test file search when only first string is present."""
        files = helper_functions.file_search(temp_directory, "string1", "nonexistent")

        # Should return empty numpy array since second string doesn't match any files
        assert len(files) == 0
        assert isinstance(files, np.ndarray)

    @pytest.mark.unit
    def test_file_search_no_matches(self, helper_functions, temp_directory):
        """Test file search when no files match first string."""
        files = helper_functions.file_search(temp_directory, "nonexistent", "string2")

        # Should return empty numpy array
        assert len(files) == 0
        assert isinstance(files, np.ndarray)

    @pytest.mark.unit
    def test_file_search_case_sensitivity(self, helper_functions, temp_directory):
        """Test file search case sensitivity."""
        files = helper_functions.file_search(temp_directory, "STRING1", "STRING2")

        # Should return empty numpy array since search is case-sensitive by default
        assert len(files) == 0
        assert isinstance(files, np.ndarray)

    @pytest.mark.unit
    def test_file_search_nonexistent_directory(self, helper_functions):
        """Test file search with nonexistent directory."""
        # os.walk doesn't raise an exception for nonexistent directories, it just returns empty
        files = helper_functions.file_search(
            "/nonexistent/directory", "string1", "string2"
        )
        assert len(files) == 0
        assert isinstance(files, np.ndarray)

    @pytest.mark.unit
    def test_file_search_returns_full_paths(self, helper_functions, temp_directory):
        """Test that file search returns full file paths."""
        files = helper_functions.file_search(temp_directory, "string1", "string2")

        # All returned paths should be absolute and exist
        for file_path in files:
            assert os.path.isabs(file_path)
            assert os.path.exists(file_path)
            assert os.path.isfile(file_path)

    @pytest.mark.unit
    def test_file_search_empty_strings(self, helper_functions, temp_directory):
        """Test file search with empty search strings."""
        # Empty string should match all files
        files = helper_functions.file_search(temp_directory, "", "")

        # Should return all files in the directory
        assert len(files) == 6  # We created 6 test files

    @pytest.mark.integration
    def test_database_cleaning_workflow(self, helper_functions):
        """Test a complete database cleaning workflow."""
        # Create a realistic database with mixed data types
        data = {
            "x_position": ["12.34", "56.78", "90.12"],
            "y_position": ["23.45", "67.89", "01.23"],
            "intensity": ["1000", "1500", "2000"],
            "frame_number": ["1", "2", "3"],
            "analysis_file": ["data_001.csv", "data_002.csv", "data_003.csv"],
        }
        test_db = pl.DataFrame(data)
        columns = [
            "x_position",
            "y_position",
            "intensity",
            "frame_number",
            "analysis_file",
        ]

        # Clean the database
        cleaned_db = helper_functions.clean_database(test_db, columns)

        # Verify the results
        assert cleaned_db.shape == (3, 5)

        # Check numeric conversions
        assert cleaned_db["x_position"].dtype in [pl.Float32, pl.Float64]
        assert cleaned_db["y_position"].dtype in [pl.Float32, pl.Float64]
        assert cleaned_db["intensity"].dtype in [pl.Float32, pl.Float64]
        assert cleaned_db["frame_number"].dtype in [pl.Float32, pl.Float64]

        # Check filename preservation
        assert cleaned_db["analysis_file"].dtype == pl.Utf8

        # Check values
        np.testing.assert_array_almost_equal(
            cleaned_db["x_position"].to_numpy(), [12.34, 56.78, 90.12]
        )
        np.testing.assert_array_almost_equal(
            cleaned_db["intensity"].to_numpy(), [1000.0, 1500.0, 2000.0]
        )


if __name__ == "__main__":
    # Run tests directly when script is executed
    pytest.main([__file__, "-v"])
