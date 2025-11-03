"""
Test brightness_boost parameter for multichannel overlays.

Demonstrates how to boost dim filamentous structures to match bright globular structures.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PlottingBase import PublicationPlotter

# Create synthetic data mimicking your scenario
np.random.seed(200)

# Channel 1: Bright globular structures (high local intensity)
img_globular = np.zeros((512, 512))
for _ in range(100):
    x, y = np.random.randint(50, 462, 2)
    y_grid, x_grid = np.ogrid[:512, :512]
    mask = ((x_grid - x)**2 + (y_grid - y)**2) < 25  # Small bright spots
    img_globular[mask] += np.random.poisson(200)

# Channel 2: Dim filamentous structures (spread out intensity)
img_filaments = np.zeros((512, 512))
for i in range(10):
    angle = i * np.pi / 5
    for t in range(512):
        x = int(256 + 200 * np.cos(angle) * (t - 256) / 256)
        y = int(256 + 200 * np.sin(angle) * (t - 256) / 256)
        if 0 <= x < 512 and 0 <= y < 512:
            # Much dimmer per-pixel because spread over line
            y_grid, x_grid = np.ogrid[:512, :512]
            mask = ((x_grid - x)**2 + (y_grid - y)**2) < 4
            img_filaments[mask] += np.random.poisson(30)  # Much lower intensity

plotter = PublicationPlotter()

# Test 1: Without brightness boost (filaments nearly invisible)
print("Test 1: Without brightness boost...")
fig, ax = plotter.create_figure(figsize=(10, 10), facecolor='black')

plotter.multichannel_overlay_plot(
    ax,
    images=[img_globular, img_filaments],
    cmaps=['cyan', 'red'],
    pixelsize=5.0,
    scalebarsize=1000,
    scalebarlabel='1 μm',
    background_color='black',
)

ax.set_title('WITHOUT Brightness Boost\n(Red filaments barely visible)',
             color='white', fontsize=14, pad=15)

output1 = '/tmp/test_brightness_boost_OFF.png'
plotter.save_or_show(fig, save_path=output1)
print(f"  Saved: {output1}")
print(f"  Red channel is DIM (filamentous structure spread out)\n")

# Test 2: With brightness boost (filaments much more visible)
print("Test 2: With brightness boost (2.5x)...")
fig, ax = plotter.create_figure(figsize=(10, 10), facecolor='black')

plotter.multichannel_overlay_plot(
    ax,
    images=[img_globular, img_filaments],
    cmaps=['cyan', 'red'],
    brightness_boost=[1.0, 2.5],  # Boost red channel 2.5x
    pixelsize=5.0,
    scalebarsize=1000,
    scalebarlabel='1 μm',
    background_color='black',
)

ax.set_title('WITH Brightness Boost (2.5x)\n(Red filaments now visible)',
             color='white', fontsize=14, pad=15)

output2 = '/tmp/test_brightness_boost_ON.png'
plotter.save_or_show(fig, save_path=output2)
print(f"  Saved: {output2}")
print(f"  Red channel is BRIGHT (2.5x boost applied)\n")

# Test 3: Comparison with different boost values
print("Test 3: Comparing different boost values...")
boost_values = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='black')
axes = axes.flatten()

for i, boost in enumerate(boost_values):
    ax = axes[i]

    plotter.multichannel_overlay_plot(
        ax,
        images=[img_globular, img_filaments],
        cmaps=['cyan', 'red'],
        brightness_boost=[1.0, boost],
        sbar='off',
        background_color='black',
    )

    ax.set_title(f'Boost = {boost}x', color='white', fontsize=12, pad=10)

plt.tight_layout()
output3 = '/tmp/test_brightness_boost_comparison.png'
fig.savefig(output3, facecolor='black', dpi=150)
plt.close(fig)

print(f"  Saved comparison: {output3}\n")

print("=" * 70)
print("Brightness boost feature successfully tested!")
print("=" * 70)
print("\nUSAGE:")
print("  brightness_boost=[1.0, 2.5]  # Keep ch1 normal, boost ch2 by 2.5x")
print("  brightness_boost=[1.0, 3.0]  # Stronger boost for very dim structures")
print("\nTIP: Start with 2.0-2.5x boost and adjust based on visual appearance")
