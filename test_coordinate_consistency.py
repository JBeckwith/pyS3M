#!/usr/bin/env python3
"""
Test script to verify coordinate consistency between scatter plots and imshow extents.
"""
import numpy as np
import matplotlib.pyplot as plt

def test_coordinate_consistency():
    """Test coordinate mapping between scatter and imshow."""

    # Create test data with known positions
    np.random.seed(42)

    # Create a simple grid pattern to verify positioning
    x = np.array([10, 20, 30, 10, 20, 30, 10, 20, 30])
    y = np.array([10, 10, 10, 20, 20, 20, 30, 30, 30])

    # Add some random noise around the grid points
    x_noise = x + np.random.normal(0, 1, len(x))
    y_noise = y + np.random.normal(0, 1, len(y))

    print(f"Test data ranges:")
    print(f"  X: {x_noise.min():.2f} to {x_noise.max():.2f}")
    print(f"  Y: {y_noise.min():.2f} to {y_noise.max():.2f}")

    # Create a simple 2D histogram (simulating datashader aggregation)
    H, xedges, yedges = np.histogram2d(x_noise, y_noise, bins=20)

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Regular scatter plot
    ax1.scatter(x_noise, y_noise, c='red', s=50, alpha=0.7)
    ax1.set_xlim(x_noise.min()-2, x_noise.max()+2)
    ax1.set_ylim(y_noise.min()-2, y_noise.max()+2)
    ax1.set_title('Scatter Plot')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)

    # Plot 2: imshow with extent (simulating datashader)
    extent = [x_noise.min(), x_noise.max(), y_noise.min(), y_noise.max()]
    ax2.imshow(H.T, extent=extent, origin='lower', aspect='auto', cmap='hot')
    ax2.scatter(x_noise, y_noise, c='cyan', s=20, alpha=0.8, edgecolors='white', linewidths=0.5)
    ax2.set_title('imshow + scatter overlay')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    # Plot 3: Test different origin
    ax3.imshow(H.T, extent=extent, origin='upper', aspect='auto', cmap='hot')
    ax3.scatter(x_noise, y_noise, c='cyan', s=20, alpha=0.8, edgecolors='white', linewidths=0.5)
    ax3.set_title('imshow origin=upper (wrong)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')

    plt.tight_layout()
    plt.savefig('/home/jbeckwith/Documents/pCloud/Chemistry/Lee/Code/Python/pyBayerSMLM/coordinate_test.png', dpi=150)
    plt.show()

    print("✅ Coordinate consistency test completed")
    print(f"Check coordinate_test.png to verify alignment")

    # Summary of coordinate mapping rules
    print("\n📍 Coordinate Mapping Rules:")
    print("  scatter(x, y) - Direct data coordinates")
    print("  imshow(extent=[left, right, bottom, top], origin='lower')")
    print(f"  extent=[{extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f}, {extent[3]:.2f}]")
    print("  origin='lower' ensures y increases upward (matches scatter)")

    return True

if __name__ == "__main__":
    test_coordinate_consistency()