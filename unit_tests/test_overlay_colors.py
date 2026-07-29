"""
Test and visualize color options for multichannel overlay plotting.

This script creates comparison images showing different color combinations
on dark backgrounds to help choose the best colors for visibility.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from pyS3M.PlottingBase import PublicationPlotter


def test_red_alternatives():
    """Compare red with brighter alternatives (pink, coral, orange, salmon, tomato)."""
    print("Testing red alternatives for dark background visibility...")

    np.random.seed(50)
    img = np.random.poisson(50, size=(256, 256)).astype(float)

    # Test different "red-ish" colors
    colors = ['red', 'pink', 'coral', 'orange', 'salmon', 'tomato', 'hotpink']

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor='black')
    axes = axes.flatten()

    plotter = PublicationPlotter()

    for i, color in enumerate(colors):
        ax = axes[i]
        ax.set_facecolor('black')

        plotter.multichannel_overlay_plot(
            ax,
            images=[img],
            cmaps=[color],
            sbar='off',
            background_color='black',
        )

        ax.set_title(color.upper(), color='white', fontsize=12, pad=10)

    # Hide unused subplot
    axes[-1].axis('off')
    axes[-1].set_facecolor('black')

    plt.tight_layout()
    output_path = '/tmp/color_comparison_red_alternatives.png'
    fig.savefig(output_path, facecolor='black', dpi=150)
    plt.close(fig)

    print(f"  Saved comparison: {output_path}")
    print("  Recommendation: 'pink', 'coral', or 'orange' are much brighter than 'red'\n")


def test_dual_channel_combinations():
    """Test recommended dual-channel color combinations."""
    print("Testing dual-channel color combinations...")

    np.random.seed(51)

    # Create two channels with different spatial patterns
    img1 = np.zeros((256, 256))
    img2 = np.zeros((256, 256))

    img1[:, :128] = np.random.poisson(60, size=(256, 128))
    img2[:, 128:] = np.random.poisson(60, size=(256, 128))

    combinations = [
        (['cyan', 'yellow'], 'Classic'),
        (['cyan', 'orange'], 'Cyan/Orange'),
        (['cyan', 'pink'], 'Cyan/Pink'),
        (['yellow', 'pink'], 'Yellow/Pink'),
        (['lime', 'pink'], 'Lime/Pink'),
        (['cyan', 'coral'], 'Cyan/Coral'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor='black')
    axes = axes.flatten()

    plotter = PublicationPlotter()

    for i, (cmaps, name) in enumerate(combinations):
        ax = axes[i]

        plotter.multichannel_overlay_plot(
            ax,
            images=[img1, img2],
            cmaps=cmaps,
            sbar='off',
            background_color='black',
        )

        ax.set_title(f"{name}: {cmaps[0]}/{cmaps[1]}", color='white', fontsize=11, pad=10)

    plt.tight_layout()
    output_path = '/tmp/color_comparison_dual_channel.png'
    fig.savefig(output_path, facecolor='black', dpi=150)
    plt.close(fig)

    print(f"  Saved comparison: {output_path}\n")


def test_three_channel_default():
    """Test the new default color scheme for 3 channels."""
    print("Testing new 3-channel default colors (cyan/yellow/pink)...")

    np.random.seed(52)

    # Three channels with different regions
    img1 = np.zeros((256, 256))
    img2 = np.zeros((256, 256))
    img3 = np.zeros((256, 256))

    img1[:85, :] = np.random.poisson(50, size=(85, 256))
    img2[85:170, :] = np.random.poisson(50, size=(85, 256))
    img3[170:, :] = np.random.poisson(50, size=(86, 256))

    plotter = PublicationPlotter()
    fig, ax = plotter.create_figure(figsize=(10, 10), facecolor='black')

    # Use new defaults (should be cyan/yellow/pink)
    plotter.multichannel_overlay_plot(
        ax,
        images=[img1, img2, img3],
        # cmaps defaults to ['cyan', 'yellow', 'pink', ...]
        sbar='off',
        background_color='black',
    )

    ax.set_title('New Default 3-Channel Colors\n(cyan/yellow/pink)',
                 color='white', fontsize=14, pad=15)

    output_path = '/tmp/color_comparison_three_channel_default.png'
    plotter.save_or_show(fig, save_path=output_path)

    print(f"  Saved: {output_path}")
    print("  New defaults are brighter and more visible!\n")


def test_all_available_colors():
    """Show all available predefined colors."""
    print("Generating palette of all available colors...")

    colors = [
        'cyan', 'yellow', 'pink', 'lime', 'orange', 'hotpink',
        'coral', 'salmon', 'tomato', 'green', 'magenta',
        'red', 'blue', 'deeppink'
    ]

    np.random.seed(53)
    img = np.random.poisson(60, size=(128, 128)).astype(float)

    fig, axes = plt.subplots(3, 5, figsize=(15, 9), facecolor='black')
    axes = axes.flatten()

    plotter = PublicationPlotter()

    for i, color in enumerate(colors):
        ax = axes[i]

        plotter.multichannel_overlay_plot(
            ax,
            images=[img],
            cmaps=[color],
            sbar='off',
            background_color='black',
        )

        # Determine if color is recommended
        recommended = color in ['cyan', 'yellow', 'pink', 'lime', 'orange',
                               'hotpink', 'coral', 'salmon', 'tomato']
        marker = '✓' if recommended else '○'

        ax.set_title(f"{marker} {color}", color='white', fontsize=9, pad=5)

    # Hide unused subplot
    axes[-1].axis('off')
    axes[-1].set_facecolor('black')

    plt.tight_layout()
    output_path = '/tmp/color_comparison_all_colors.png'
    fig.savefig(output_path, facecolor='black', dpi=150)
    plt.close(fig)

    print(f"  Saved color palette: {output_path}")
    print("  ✓ = Recommended (bright), ○ = Available but darker\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Color Comparison Tests for Multichannel Overlay")
    print("=" * 70)
    print()

    test_red_alternatives()
    test_dual_channel_combinations()
    test_three_channel_default()
    test_all_available_colors()

    print("=" * 70)
    print("All color comparison images generated!")
    print("Output files: /tmp/color_comparison_*.png")
    print("=" * 70)
    print()
    print("RECOMMENDATIONS:")
    print("  Best bright colors: cyan, yellow, orange, pink, lime")
    print("  Good alternatives: coral, salmon, hotpink, tomato")
    print("  Avoid on dark bg: red, blue, magenta (use brighter versions)")
