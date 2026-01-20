"""
Example: Using plot_ternary_scatter for Qdot RGB analysis

This demonstrates how to create ternary scatter plots for RGB data
from the Qdot wavelength shift analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from src.PlottingBase import PublicationPlotter

# Example: Simulate RGB data from wavelength-shifted Qdots
np.random.seed(42)

# Simulate RGB values for different wavelength shifts
# Each shift creates a cluster in RGB space
wavelength_shifts = [-10, -5, 0, 5, 10]
n_points_per_shift = 100

all_R = []
all_G = []
all_B = []
all_shifts = []

for shift in wavelength_shifts:
    # Simulate slight variations in RGB for each shift
    # These would come from your actual simulation data
    R_mean = 0.3 + shift * 0.02
    G_mean = 0.4 - shift * 0.01
    B_mean = 1.0 - R_mean - G_mean

    R = np.random.normal(R_mean, 0.05, n_points_per_shift)
    G = np.random.normal(G_mean, 0.05, n_points_per_shift)
    B = 1.0 - R - G

    # Normalize
    totals = R + G + B
    R = R / totals
    G = G / totals
    B = B / totals

    all_R.append(R)
    all_G.append(G)
    all_B.append(B)
    all_shifts.append([shift] * n_points_per_shift)

# Combine all data
R_all = np.concatenate(all_R)
G_all = np.concatenate(all_G)
B_all = np.concatenate(all_B)
shifts_all = np.concatenate(all_shifts)

# ============================================================================
# Example 1: Simple ternary scatter plot
# ============================================================================
print("Creating Example 1: Simple ternary scatter...")

fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={'projection': 'ternary'})

plotter = PublicationPlotter()
scatter = plotter.plot_ternary_scatter(
    ax, R_all, G_all, B_all,
    color='purple',
    size=15,
    alpha=0.4,
    label='All shifts'
)

ax.set_title('Qdot RGB Values - All Wavelength Shifts')
ax.legend()

plt.savefig('/tmp/example1_simple_ternary.png', dpi=200, bbox_inches='tight')
print("Saved to /tmp/example1_simple_ternary.png")
plt.close()

# ============================================================================
# Example 2: Color by wavelength shift
# ============================================================================
print("\nCreating Example 2: Colored by wavelength shift...")

fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={'projection': 'ternary'})

plotter = PublicationPlotter()
scatter = plotter.plot_ternary_scatter(
    ax, R_all, G_all, B_all,
    color=shifts_all,
    size=20,
    alpha=0.6,
    cmap='coolwarm',  # Colormap for wavelength shifts
    label='Wavelength shift'
)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, pad=0.1, fraction=0.05)
cbar.set_label('Wavelength Shift (nm)', rotation=270, labelpad=20)

ax.set_title('Qdot RGB Values Colored by Wavelength Shift')

plt.savefig('/tmp/example2_colored_by_shift.png', dpi=200, bbox_inches='tight')
print("Saved to /tmp/example2_colored_by_shift.png")
plt.close()

# ============================================================================
# Example 3: Individual groups with different colors/markers
# ============================================================================
print("\nCreating Example 3: Individual groups...")

fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={'projection': 'ternary'})

plotter = PublicationPlotter()
colors = ['blue', 'cyan', 'green', 'orange', 'red']
markers = ['o', 's', '^', 'v', 'D']

for i, shift in enumerate(wavelength_shifts):
    plotter.plot_ternary_scatter(
        ax, all_R[i], all_G[i], all_B[i],
        color=colors[i],
        size=30,
        alpha=0.6,
        marker=markers[i],
        edgecolor='black',
        linewidth=0.5,
        label=f'{shift:+.0f} nm'
    )

ax.set_title('Qdot RGB Values - Individual Wavelength Shifts')
ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1), title='Shift')

plt.savefig('/tmp/example3_individual_groups.png', dpi=200, bbox_inches='tight')
print("Saved to /tmp/example3_individual_groups.png")
plt.close()

# ============================================================================
# Example 4: Multi-panel figure with ternary plot
# ============================================================================
print("\nCreating Example 4: Multi-panel figure...")

fig = plt.figure(figsize=(15, 5))

# Panel 1: Histogram of shifts
ax1 = fig.add_subplot(1, 3, 1)
ax1.hist(shifts_all, bins=len(wavelength_shifts), edgecolor='black')
ax1.set_xlabel('Wavelength Shift (nm)')
ax1.set_ylabel('Count')
ax1.set_title('Distribution of Wavelength Shifts')
ax1.grid(True, alpha=0.3)

# Panel 2: Ternary scatter
ax2 = fig.add_subplot(1, 3, 2, projection='ternary')
plotter = PublicationPlotter()
scatter = plotter.plot_ternary_scatter(
    ax2, R_all, G_all, B_all,
    color=shifts_all,
    size=15,
    alpha=0.5,
    cmap='viridis'
)
cbar = plt.colorbar(scatter, ax=ax2, pad=0.1, fraction=0.05)
cbar.set_label('Shift (nm)', rotation=270, labelpad=15)
ax2.set_title('Ternary RGB Space')

# Panel 3: R vs G scatter colored by B
ax3 = fig.add_subplot(1, 3, 3)
sc = ax3.scatter(R_all, G_all, c=B_all, cmap='plasma', s=15, alpha=0.5)
ax3.set_xlabel('R')
ax3.set_ylabel('G')
ax3.set_title('R vs G (colored by B)')
cbar3 = plt.colorbar(sc, ax=ax3)
cbar3.set_label('B value')
ax3.grid(True, alpha=0.3)

plt.savefig('/tmp/example4_multipanel.png', dpi=200, bbox_inches='tight')
print("Saved to /tmp/example4_multipanel.png")
plt.close()

print("\n" + "="*70)
print("All examples completed!")
print("\nGenerated files:")
print("  /tmp/example1_simple_ternary.png")
print("  /tmp/example2_colored_by_shift.png")
print("  /tmp/example3_individual_groups.png")
print("  /tmp/example4_multipanel.png")
print("\nYou can now use plot_ternary_scatter() in your Qdot notebook!")
