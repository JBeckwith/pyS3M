"""
Test multichannel overlay plotting functionality.

Tests the new multichannel_overlay_plot method in PlottingBase.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PlottingBase import PublicationPlotter, AnalysisPlotter


def test_basic_two_channel_overlay():
    """Test basic two-channel overlay with default parameters."""
    print("Testing basic two-channel overlay...")

    # Create synthetic rendered images (simulate SMLM data)
    np.random.seed(42)

    # Channel 1: More signal in left half
    img1 = np.zeros((512, 512))
    img1[:, :256] = np.random.poisson(50, size=(512, 256))
    img1[:, 256:] = np.random.poisson(10, size=(512, 256))

    # Channel 2: More signal in right half
    img2 = np.zeros((512, 512))
    img2[:, :256] = np.random.poisson(10, size=(512, 256))
    img2[:, 256:] = np.random.poisson(50, size=(512, 256))

    # Create plotter and figure
    plotter = PublicationPlotter()
    fig, ax = plotter.create_figure(figsize=(10, 10))

    # Create overlay
    plotter.multichannel_overlay_plot(
        ax,
        images=[img1, img2],
        cmaps=['cyan', 'yellow'],
        pixelsize=5.0,
        scalebarsize=1000,
        scalebarlabel='1 μm',
    )

    # Save output
    output_path = '/tmp/test_multichannel_basic.png'
    plotter.save_or_show(fig, save_path=output_path)

    # Verify file was created
    assert os.path.exists(output_path), "Output file not created"
    file_size = os.path.getsize(output_path)
    assert file_size > 1000, f"Output file too small ({file_size} bytes)"

    print(f"✓ Basic overlay test passed. Output: {output_path}")
    plt.close(fig)


def test_three_channel_overlay():
    """Test three-channel overlay."""
    print("Testing three-channel overlay...")

    np.random.seed(43)

    # Create three synthetic channels
    img1 = np.random.poisson(30, size=(256, 256)).astype(float)
    img2 = np.random.poisson(40, size=(256, 256)).astype(float)
    img3 = np.random.poisson(35, size=(256, 256)).astype(float)

    plotter = PublicationPlotter()
    fig, ax = plotter.create_figure(figsize=(10, 10))

    plotter.multichannel_overlay_plot(
        ax,
        images=[img1, img2, img3],
        cmaps=['cyan', 'yellow', 'magenta'],
        alphas=[0.6, 0.6, 0.6],
        pixelsize=5.0,
        scalebarsize=500,
        scalebarlabel='500 nm',
    )

    output_path = '/tmp/test_multichannel_three.png'
    plotter.save_or_show(fig, save_path=output_path)

    assert os.path.exists(output_path)
    print(f"✓ Three-channel overlay test passed. Output: {output_path}")
    plt.close(fig)


def test_with_colorbars():
    """Test overlay with colorbars enabled."""
    print("Testing overlay with colorbars...")

    np.random.seed(44)

    img1 = np.random.poisson(50, size=(256, 256)).astype(float)
    img2 = np.random.poisson(30, size=(256, 256)).astype(float)

    plotter = PublicationPlotter()
    fig, ax = plotter.create_figure(figsize=(12, 10))

    plotter.multichannel_overlay_plot(
        ax,
        images=[img1, img2],
        cmaps=['cyan', 'orange'],
        cbar='on',
        cbarlabels=['Channel 1', 'Channel 2'],
        pixelsize=5.0,
    )

    output_path = '/tmp/test_multichannel_colorbars.png'
    plotter.save_or_show(fig, save_path=output_path)

    assert os.path.exists(output_path)
    print(f"✓ Colorbar test passed. Output: {output_path}")
    plt.close(fig)


def test_custom_intensity_scaling():
    """Test with custom vmin/vmax values."""
    print("Testing custom intensity scaling...")

    np.random.seed(45)

    img1 = np.random.poisson(100, size=(256, 256)).astype(float)
    img2 = np.random.poisson(50, size=(256, 256)).astype(float)

    plotter = PublicationPlotter()
    fig, ax = plotter.create_figure(figsize=(10, 10))

    plotter.multichannel_overlay_plot(
        ax,
        images=[img1, img2],
        cmaps=['red', 'green'],
        vmins=[50, 20],
        vmaxs=[150, 80],
        sbar='off',  # No scale bar
    )

    output_path = '/tmp/test_multichannel_custom_scaling.png'
    plotter.save_or_show(fig, save_path=output_path)

    assert os.path.exists(output_path)
    print(f"✓ Custom scaling test passed. Output: {output_path}")
    plt.close(fig)


def test_white_background():
    """Test with white background instead of black."""
    print("Testing white background...")

    np.random.seed(46)

    img1 = np.random.poisson(40, size=(256, 256)).astype(float)
    img2 = np.random.poisson(35, size=(256, 256)).astype(float)

    plotter = PublicationPlotter()
    fig, ax = plotter.create_figure(figsize=(10, 10), facecolor='white')

    plotter.multichannel_overlay_plot(
        ax,
        images=[img1, img2],
        cmaps=['blue', 'red'],
        background_color='white',
        pixelsize=10.0,
        scalebarsize=2000,
        scalebarlabel='2 μm',
    )

    output_path = '/tmp/test_multichannel_white_bg.png'
    plotter.save_or_show(fig, save_path=output_path)

    assert os.path.exists(output_path)
    print(f"✓ White background test passed. Output: {output_path}")
    plt.close(fig)


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("Testing error handling...")

    img1 = np.random.poisson(50, size=(256, 256)).astype(float)
    img2 = np.random.poisson(30, size=(256, 256)).astype(float)
    img3_wrong_size = np.random.poisson(40, size=(128, 128)).astype(float)

    plotter = PublicationPlotter()
    fig, ax = plotter.create_figure()

    # Test 1: Single image (should fail)
    try:
        plotter.multichannel_overlay_plot(ax, images=[img1])
        assert False, "Should have raised ValueError for single image"
    except ValueError as e:
        assert "at least 2 images" in str(e)
        print("  ✓ Single image error caught correctly")

    # Test 2: Mismatched image sizes (should fail)
    try:
        plotter.multichannel_overlay_plot(ax, images=[img1, img3_wrong_size])
        assert False, "Should have raised ValueError for mismatched sizes"
    except ValueError as e:
        assert "doesn't match" in str(e)
        print("  ✓ Size mismatch error caught correctly")

    # Test 3: Wrong number of colormaps (should fail)
    try:
        plotter.multichannel_overlay_plot(
            ax, images=[img1, img2], cmaps=['cyan']
        )
        assert False, "Should have raised ValueError for wrong number of cmaps"
    except ValueError as e:
        assert "colormaps" in str(e)
        print("  ✓ Colormap count error caught correctly")

    # Test 4: Invalid alpha value (should fail)
    try:
        plotter.multichannel_overlay_plot(
            ax, images=[img1, img2], alphas=[0.5, 1.5]
        )
        assert False, "Should have raised ValueError for invalid alpha"
    except ValueError as e:
        assert "Alpha" in str(e)
        print("  ✓ Invalid alpha error caught correctly")

    plt.close(fig)
    print("✓ All error handling tests passed")


def test_analysis_plotter():
    """Test that AnalysisPlotter also works with multichannel overlay."""
    print("Testing with AnalysisPlotter...")

    np.random.seed(47)

    img1 = np.random.poisson(60, size=(256, 256)).astype(float)
    img2 = np.random.poisson(45, size=(256, 256)).astype(float)

    plotter = AnalysisPlotter()
    fig, ax = plotter.create_figure(figsize=(10, 10))

    plotter.multichannel_overlay_plot(
        ax,
        images=[img1, img2],
        cmaps=['magenta', 'green'],
        pixelsize=5.0,
    )

    output_path = '/tmp/test_multichannel_analysis.png'
    plotter.save_or_show(fig, save_path=output_path)

    assert os.path.exists(output_path)
    print(f"✓ AnalysisPlotter test passed. Output: {output_path}")
    plt.close(fig)


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running multichannel overlay plotting tests...")
    print("=" * 60)

    test_basic_two_channel_overlay()
    test_three_channel_overlay()
    test_with_colorbars()
    test_custom_intensity_scaling()
    test_white_background()
    test_error_handling()
    test_analysis_plotter()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nTest outputs saved to /tmp/test_multichannel_*.png")


if __name__ == "__main__":
    run_all_tests()
