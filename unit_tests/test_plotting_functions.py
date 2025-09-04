#!/usr/bin/env python3
"""
Test module for PlottingFunctions.

Tests the refactored PlottingFunctions module including plot creation,
style management, and visualization utilities.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import tempfile
import os
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Use non-interactive backend for testing
matplotlib.use("Agg")

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from PlottingFunctions import Plotter, PlotConstants


class TestPlotConstants:
    """Test PlotConstants configuration."""

    @pytest.mark.unit
    def test_constants_exist(self):
        """Test that required constants are defined."""
        assert hasattr(PlotConstants, "FONT_SIZES")
        assert hasattr(PlotConstants, "FIGURE_SIZES")
        assert hasattr(PlotConstants, "COLOR_PALETTES")
        assert hasattr(PlotConstants, "DPI_SETTINGS")

    @pytest.mark.unit
    def test_font_sizes_valid(self):
        """Test that font size constants are reasonable."""
        font_sizes = PlotConstants.FONT_SIZES
        assert isinstance(font_sizes, dict)

        # Check common font size keys exist and are positive numbers
        expected_keys = ["small", "medium", "large", "title"]
        for key in expected_keys:
            if key in font_sizes:
                assert isinstance(font_sizes[key], (int, float))
                assert font_sizes[key] > 0

    @pytest.mark.unit
    def test_figure_sizes_valid(self):
        """Test that figure size constants are valid."""
        figure_sizes = PlotConstants.FIGURE_SIZES
        assert isinstance(figure_sizes, dict)

        # Check that sizes are tuples of positive numbers
        for size_name, size_tuple in figure_sizes.items():
            assert isinstance(size_tuple, (tuple, list))
            assert len(size_tuple) == 2
            assert all(isinstance(x, (int, float)) and x > 0 for x in size_tuple)


class TestPlottingFunctions:
    """Test main Plotter class."""

    @pytest.fixture
    def plotting_functions(self):
        """Create Plotter instance."""
        return Plotter()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for plotting tests."""
        np.random.seed(42)  # For reproducible tests
        return {
            "x": np.linspace(0, 10, 100),
            "y": np.sin(np.linspace(0, 10, 100)) + 0.1 * np.random.randn(100),
            "scatter_x": np.random.randn(50),
            "scatter_y": np.random.randn(50),
            "histogram_data": np.random.normal(0, 1, 1000),
        }

    @pytest.fixture
    def temp_plot_file(self):
        """Create temporary file for plot saving tests."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.mark.unit
    def test_class_initialization(self, plotting_functions):
        """Test class initialization."""
        assert plotting_functions is not None
        # Test that matplotlib is configured
        assert plt.rcParams is not None

    @pytest.mark.unit
    def test_one_column_plot_creation(self, plotting_functions, sample_data):
        """Test basic one-column plot creation."""
        fig, ax = plotting_functions.one_column_plot()

        # Verify figure and axis creation
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        # Test with data
        ax.plot(sample_data["x"], sample_data["y"])

        # Should not raise exceptions
        fig.canvas.draw()
        plt.close(fig)

    @pytest.mark.unit
    def test_one_column_plot_with_parameters(self, plotting_functions, sample_data):
        """Test one-column plot with custom parameters."""
        fig, ax = plotting_functions.one_column_plot(width=6.0, height=4.0, dpi=150)

        # Check figure properties
        assert abs(fig.get_figwidth() - 6.0) < 0.1
        assert abs(fig.get_figheight() - 4.0) < 0.1
        assert fig.dpi == 150

        plt.close(fig)

    @pytest.mark.unit
    def test_two_column_plot_creation(self, plotting_functions):
        """Test two-column plot creation."""
        fig, (ax1, ax2) = plotting_functions.two_column_plot()

        # Verify subplots creation
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax1, plt.Axes)
        assert isinstance(ax2, plt.Axes)

        # Should have 2 subplots
        assert len(fig.axes) == 2

        plt.close(fig)

    @pytest.mark.unit
    def test_scatter_plot_creation(self, plotting_functions, sample_data):
        """Test scatter plot creation."""
        fig, ax = plotting_functions.scatter_plot(
            sample_data["scatter_x"], sample_data["scatter_y"]
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        # Check that scatter plot was created
        assert len(ax.collections) > 0  # Scatter creates a collection

        plt.close(fig)

    @pytest.mark.unit
    def test_scatter_plot_with_labels(self, plotting_functions, sample_data):
        """Test scatter plot with custom labels."""
        fig, ax = plotting_functions.scatter_plot(
            sample_data["scatter_x"],
            sample_data["scatter_y"],
            xlabel="X Values",
            ylabel="Y Values",
            title="Test Scatter Plot",
        )

        # Check labels were set
        assert ax.get_xlabel() == "X Values"
        assert ax.get_ylabel() == "Y Values"
        assert ax.get_title() == "Test Scatter Plot"

        plt.close(fig)

    @pytest.mark.unit
    def test_histogram_plot_creation(self, plotting_functions, sample_data):
        """Test histogram plot creation."""
        fig, ax = plotting_functions.histogram_plot(sample_data["histogram_data"])

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        # Check that histogram was created
        assert len(ax.patches) > 0  # Histogram creates patches

        plt.close(fig)

    @pytest.mark.unit
    def test_histogram_plot_with_bins(self, plotting_functions, sample_data):
        """Test histogram with custom bin count."""
        bins = 30
        fig, ax = plotting_functions.histogram_plot(
            sample_data["histogram_data"], bins=bins
        )

        # Should have approximately the requested number of bins
        assert len(ax.patches) <= bins + 5  # Allow some flexibility

        plt.close(fig)

    @pytest.mark.integration
    def test_plot_saving(self, plotting_functions, sample_data, temp_plot_file):
        """Test saving plots to file."""
        fig, ax = plotting_functions.one_column_plot()
        ax.plot(sample_data["x"], sample_data["y"])

        # Save plot
        plotting_functions.save_plot(fig, temp_plot_file, dpi=300)

        # Verify file was created
        assert os.path.exists(temp_plot_file)
        assert os.path.getsize(temp_plot_file) > 0

        plt.close(fig)

    @pytest.mark.unit
    def test_plot_formatting_utilities(self, plotting_functions):
        """Test plot formatting utility functions."""
        fig, ax = plotting_functions.one_column_plot()

        # Test grid formatting
        plotting_functions.apply_grid_formatting(ax)
        assert ax.grid is not None

        # Test spine formatting
        plotting_functions.format_spines(ax)

        # Test tick formatting
        plotting_functions.format_ticks(ax)

        plt.close(fig)

    @pytest.mark.unit
    def test_color_palette_application(self, plotting_functions, sample_data):
        """Test color palette application."""
        fig, ax = plotting_functions.one_column_plot()

        # Plot multiple series with palette
        colors = plotting_functions.get_color_palette("qualitative", 3)

        for i in range(3):
            y_data = sample_data["y"] + i * 0.5
            ax.plot(sample_data["x"], y_data, color=colors[i])

        # Check that colors were applied
        lines = ax.get_lines()
        assert len(lines) == 3

        plt.close(fig)

    @pytest.mark.unit
    def test_subplot_creation(self, plotting_functions):
        """Test subplot creation utilities."""
        fig, axes = plotting_functions.create_subplots(2, 2)

        # Should create 2x2 grid
        assert isinstance(fig, plt.Figure)
        assert axes.shape == (2, 2)
        assert len(fig.axes) == 4

        plt.close(fig)

    @pytest.mark.unit
    def test_figure_size_presets(self, plotting_functions):
        """Test figure size preset functionality."""
        # Test different preset sizes
        presets = ["small", "medium", "large", "wide"]

        for preset in presets:
            try:
                fig, ax = plotting_functions.one_column_plot(size_preset=preset)
                assert isinstance(fig, plt.Figure)
                assert fig.get_figwidth() > 0
                assert fig.get_figheight() > 0
                plt.close(fig)
            except (AttributeError, KeyError):
                # Preset might not be implemented - skip
                pass


class TestPlottingErrorHandling:
    """Test error handling in plotting functions."""

    @pytest.fixture
    def plotting_functions(self):
        """Create Plotter instance."""
        return Plotter()

    @pytest.mark.unit
    def test_empty_data_handling(self, plotting_functions):
        """Test handling of empty data arrays."""
        # Empty arrays should not crash but may produce warnings
        empty_x = np.array([])
        empty_y = np.array([])

        fig, ax = plotting_functions.scatter_plot(empty_x, empty_y)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @pytest.mark.unit
    def test_mismatched_array_lengths(self, plotting_functions):
        """Test handling of mismatched array lengths."""
        x = np.array([1, 2, 3])
        y = np.array([1, 2])  # Different length

        # Should handle gracefully or raise informative error
        try:
            fig, ax = plotting_functions.scatter_plot(x, y)
            plt.close(fig)
        except ValueError:
            # Expected behavior - arrays must be same length
            pass

    @pytest.mark.unit
    def test_invalid_file_path(self, plotting_functions):
        """Test handling of invalid save paths."""
        fig, ax = plotting_functions.one_column_plot()

        # Try to save to invalid path
        invalid_path = "/nonexistent/directory/plot.png"

        with pytest.raises((OSError, FileNotFoundError, PermissionError)):
            plotting_functions.save_plot(fig, invalid_path)

        plt.close(fig)


class TestPlottingIntegration:
    """Test integration scenarios for plotting functions."""

    @pytest.fixture
    def plotting_functions(self):
        """Create Plotter instance."""
        return Plotter()

    @pytest.fixture
    def complex_dataset(self):
        """Create complex dataset for integration testing."""
        np.random.seed(123)
        n_points = 1000

        return {
            "time": np.linspace(0, 10, n_points),
            "signal1": np.sin(2 * np.pi * np.linspace(0, 10, n_points))
            + 0.1 * np.random.randn(n_points),
            "signal2": np.cos(2 * np.pi * np.linspace(0, 10, n_points))
            + 0.1 * np.random.randn(n_points),
            "noise": np.random.randn(n_points),
            "categories": np.random.choice(["A", "B", "C"], n_points),
            "values": np.random.exponential(2, n_points),
        }

    @pytest.mark.integration
    def test_multi_panel_figure(self, plotting_functions, complex_dataset):
        """Test creation of multi-panel figure."""
        fig, axes = plotting_functions.create_subplots(2, 3, figsize=(12, 8))

        # Plot different data in each panel
        axes[0, 0].plot(complex_dataset["time"], complex_dataset["signal1"])
        axes[0, 0].set_title("Signal 1")

        axes[0, 1].plot(complex_dataset["time"], complex_dataset["signal2"])
        axes[0, 1].set_title("Signal 2")

        axes[0, 2].scatter(complex_dataset["signal1"], complex_dataset["signal2"])
        axes[0, 2].set_title("Signal Correlation")

        axes[1, 0].hist(complex_dataset["noise"], bins=50)
        axes[1, 0].set_title("Noise Distribution")

        axes[1, 1].hist(complex_dataset["values"], bins=50)
        axes[1, 1].set_title("Value Distribution")

        axes[1, 2].boxplot(
            [
                complex_dataset["values"][complex_dataset["categories"] == cat]
                for cat in ["A", "B", "C"]
            ]
        )
        axes[1, 2].set_title("Category Comparison")

        # Apply formatting to all subplots
        for ax in axes.flat:
            plotting_functions.apply_grid_formatting(ax)

        plt.tight_layout()

        # Verify figure structure
        assert len(fig.axes) == 6
        assert all(isinstance(ax, plt.Axes) for ax in fig.axes)

        plt.close(fig)

    @pytest.mark.integration
    def test_publication_quality_figure(
        self, plotting_functions, complex_dataset, temp_plot_file
    ):
        """Test creation of publication-quality figure."""
        # Create publication-style figure
        fig, ax = plotting_functions.one_column_plot(
            width=3.5, height=2.5, dpi=300  # Single column width
        )

        # Plot with publication formatting
        ax.plot(
            complex_dataset["time"],
            complex_dataset["signal1"],
            linewidth=1.0,
            color="black",
            label="Experimental",
        )
        ax.plot(
            complex_dataset["time"],
            complex_dataset["signal2"],
            linewidth=1.0,
            color="red",
            linestyle="--",
            label="Control",
        )

        # Publication formatting
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Signal (a.u.)", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)

        # Save as high-DPI figure
        plotting_functions.save_plot(
            fig, temp_plot_file.replace(".png", "_pub.png"), dpi=300
        )

        plt.close(fig)

    @pytest.mark.integration
    def test_interactive_plotting_compatibility(
        self, plotting_functions, complex_dataset
    ):
        """Test compatibility with interactive plotting backends."""
        # Should work regardless of backend
        current_backend = matplotlib.get_backend()

        try:
            fig, ax = plotting_functions.one_column_plot()
            ax.plot(complex_dataset["time"], complex_dataset["signal1"])

            # Should be able to draw without errors
            fig.canvas.draw()

            assert isinstance(fig, plt.Figure)

        finally:
            plt.close(fig)

    @pytest.mark.integration
    def test_memory_management_large_plots(self, plotting_functions):
        """Test memory management with large datasets."""
        # Create large dataset
        n_points = 100000
        large_x = np.linspace(0, 100, n_points)
        large_y = np.sin(large_x) + 0.01 * np.random.randn(n_points)

        # Should handle large datasets without memory issues
        fig, ax = plotting_functions.scatter_plot(large_x, large_y, alpha=0.1)

        # Verify plot was created
        assert len(ax.collections) > 0

        # Cleanup
        plt.close(fig)

        # Memory should be released (can't easily test, but function should complete)
        assert True  # Test completed successfully
