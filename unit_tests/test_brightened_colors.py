"""
Quick test to show the brightened red/pink colors on dark background.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')


from pyS3M.PlottingBase import PublicationPlotter

np.random.seed(100)
img1 = np.random.poisson(50, size=(256, 256)).astype(float)
img2 = np.zeros((256, 256))  # Empty second image to satisfy 2-image requirement

# Test the brightened red colors
colors = ['red', 'pink', 'coral', 'salmon', 'hotpink', 'orange']

plotter = PublicationPlotter()

for color in colors:
    fig, ax = plotter.create_figure(figsize=(8, 8), facecolor='black')

    plotter.multichannel_overlay_plot(
        ax,
        images=[img1, img2],  # Use two images
        cmaps=[color, 'cyan'],  # Second channel won't show (empty)
        sbar='off',
        background_color='black',
    )

    ax.set_title(f'{color.upper()} (now with more white)',
                 color='white', fontsize=14, pad=10)

    output_path = f'/tmp/test_brightened_{color}.png'
    plotter.save_or_show(fig, save_path=output_path)
    print(f"✓ {color}: {output_path}")

print("\nAll colors now have more white mixed in for better visibility!")
print("Compare especially: red, pink, coral - should be much more visible now")
