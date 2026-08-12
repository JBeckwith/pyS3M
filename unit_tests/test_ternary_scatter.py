#!/usr/bin/env python3
"""
Test script for plot_ternary_scatter function in PlottingBase.

This demonstrates how to use plot_ternary_scatter with existing axes
from one_column_plot() or other figure creation methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path

from pyS3M.PlottingBase import PublicationPlotter

def test_basic_scatter(test_output_dir):
    """Test basic ternary scatter plot."""
    print("Testing basic ternary scatter plot...")

    # Create some sample RGB data
    np.random.seed(42)
    n_points = 100

    # Create data clustered around different regions
    R = np.random.beta(2, 5, n_points)
    G = np.random.beta(5, 2, n_points)
    B = 1 - R - G

    # Normalize (in case of negative B values)
    totals = R + G + B
    R = R / totals
    G = G / totals
    B = B / totals

    # Create figure with ternary projection
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={'projection': 'ternary'})

    # Create plotter and add scatter
    plotter = PublicationPlotter()
    scatter = plotter.plot_ternary_scatter(
        ax, R, G, B,
        color='red',
        size=30,
        alpha=0.6,
        label='Sample Data'
    )

    ax.legend()
    ax.set_title('Basic Ternary Scatter Plot')

    plt.tight_layout()
    output_path = test_output_dir / 'test_ternary_scatter_basic.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")
    plt.close()


def test_multi_panel(test_output_dir):
    """Test ternary scatter in a multi-panel figure."""
    print("\nTesting multi-panel figure with ternary scatter...")

    # Generate sample data
    np.random.seed(42)
    n_points = 150

    R = np.random.beta(3, 3, n_points)
    G = np.random.beta(3, 3, n_points)
    B = 1 - R - G

    # Normalize
    totals = R + G + B
    R = R / totals
    G = G / totals
    B = B / totals

    # Create multi-panel figure
    fig = plt.figure(figsize=(15, 5))

    # Regular plot
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.hist(R, bins=20, alpha=0.7, color='red', label='R')
    ax1.hist(G, bins=20, alpha=0.7, color='green', label='G')
    ax1.hist(B, bins=20, alpha=0.7, color='blue', label='B')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Count')
    ax1.set_title('RGB Distributions')
    ax1.legend()

    # Ternary scatter plot
    ax2 = fig.add_subplot(1, 3, 2, projection='ternary')
    plotter = PublicationPlotter()
    plotter.plot_ternary_scatter(
        ax2, R, G, B,
        color='purple',
        size=25,
        alpha=0.5,
        edgecolor='black',
        linewidth=0.5,
        label='RGB Data'
    )
    ax2.set_title('Ternary Scatter')
    ax2.legend()

    # Another regular plot
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(R, G, c=B, cmap='viridis', s=30, alpha=0.6)
    ax3.set_xlabel('R')
    ax3.set_ylabel('G')
    ax3.set_title('R vs G (colored by B)')
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label('B value')

    # Note: skip tight_layout when using colorbars to avoid layout engine conflicts
    output_path = test_output_dir / 'test_ternary_scatter_multipanel.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()


def test_with_plotter_methods(test_output_dir):
    """Test using ternary scatter with PublicationPlotter's one_column_plot."""
    print("\nTesting with PublicationPlotter.one_column_plot()...")

    # This demonstrates the challenge: one_column_plot creates regular axes
    # For ternary plots, we need to manually specify projection='ternary'

    # Generate sample data
    np.random.seed(123)
    n_points = 200

    # Create two clusters
    R1 = np.random.beta(8, 2, n_points // 2)
    G1 = np.random.beta(2, 8, n_points // 2)
    B1 = 1 - R1 - G1

    R2 = np.random.beta(2, 8, n_points // 2)
    G2 = np.random.beta(8, 2, n_points // 2)
    B2 = 1 - R2 - G2

    # Combine and normalize
    R = np.concatenate([R1, R2])
    G = np.concatenate([G1, G2])
    B = np.concatenate([B1, B2])

    totals = R + G + B
    R = R / totals
    G = G / totals
    B = B / totals

    # Create colors for two groups
    colors = np.array(['red'] * (n_points // 2) + ['blue'] * (n_points // 2))

    # Create figure manually with ternary projection
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={'projection': 'ternary'})

    # Add scatter points for each group
    plotter = PublicationPlotter()

    # Group 1
    plotter.plot_ternary_scatter(
        ax, R[:n_points//2], G[:n_points//2], B[:n_points//2],
        color='red',
        size=40,
        alpha=0.5,
        marker='o',
        label='Group 1'
    )

    # Group 2
    plotter.plot_ternary_scatter(
        ax, R[n_points//2:], G[n_points//2:], B[n_points//2:],
        color='blue',
        size=40,
        alpha=0.5,
        marker='s',
        label='Group 2'
    )

    ax.set_title('Two-Group Ternary Scatter')
    ax.legend()

    plt.tight_layout()
    output_path = test_output_dir / 'test_ternary_scatter_groups.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")
    plt.close()


def test_colormap_scatter(test_output_dir):
    """Test ternary scatter with colormap coloring."""
    print("\nTesting ternary scatter with colormap...")

    # Generate sample data
    np.random.seed(456)
    n_points = 300

    R = np.random.dirichlet([2, 5, 3], n_points)[:, 0]
    G = np.random.dirichlet([2, 5, 3], n_points)[:, 1]
    B = np.random.dirichlet([2, 5, 3], n_points)[:, 2]

    # Create a value to color by (e.g., distance from center)
    center = np.array([1/3, 1/3, 1/3])
    distances = np.sqrt((R - center[0])**2 + (G - center[1])**2 + (B - center[2])**2)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={'projection': 'ternary'})

    plotter = PublicationPlotter()
    scatter = plotter.plot_ternary_scatter(
        ax, R, G, B,
        color=distances,
        size=30,
        alpha=0.7,
        marker='o',
        cmap='plasma',  # This goes to **kwargs
        label='Data points'
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, fraction=0.05)
    cbar.set_label('Distance from Center', rotation=270, labelpad=20)

    ax.set_title('Ternary Scatter with Colormap')

    # Note: skip tight_layout when using colorbars to avoid layout engine conflicts
    output_path = test_output_dir / 'test_ternary_scatter_colormap.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    print("Testing plot_ternary_scatter function\n" + "="*50)

    try:
        import mpltern
        print("mpltern is installed ✓\n")
    except ImportError:
        print("ERROR: mpltern is not installed!")
        print("Install with: pip install mpltern")
        sys.exit(1)

    from pathlib import Path
    _output_dir = Path("/tmp")

    test_basic_scatter(_output_dir)
    test_multi_panel(_output_dir)
    test_with_plotter_methods(_output_dir)
    test_colormap_scatter(_output_dir)

    print("\n" + "="*50)
    print("All tests completed successfully!")
    print("\nGenerated files:")
    print(f"  - {_output_dir / 'test_ternary_scatter_basic.png'}")
    print(f"  - {_output_dir / 'test_ternary_scatter_multipanel.png'}")
    print(f"  - {_output_dir / 'test_ternary_scatter_groups.png'}")
    print(f"  - {_output_dir / 'test_ternary_scatter_colormap.png'}")
