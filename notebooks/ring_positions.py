#!/usr/bin/env python3
"""
Ring Position Generator for Molecular Simulations

Functions to generate molecular positions arranged in ring patterns
for use with pyBayerSMLM simulation functions.

Author: Claude Code (Anthropic)
Date: September 19, 2025
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union
import matplotlib.pyplot as plt


def generate_ring_positions(
    N: int,
    image_size: int,
    pixel_size: float,
    ring_radius_fraction: float = 0.3,
    center_offset: Tuple[float, float] = (0.0, 0.0),
    add_noise: bool = False,
    noise_sigma: float = 10.0,
    start_angle: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate N molecular positions arranged in a ring pattern.

    Args:
        N (int): Number of molecules to place in the ring
        image_size (int): Size of the image in pixels (assumes square image)
        pixel_size (float): Physical size of each pixel (e.g., 69 nm)
        ring_radius_fraction (float): Fraction of image radius for the ring (0.0-0.5)
        center_offset (Tuple[float, float]): Offset from image center in nm (x, y)
        add_noise (bool): Whether to add Gaussian noise to positions
        noise_sigma (float): Standard deviation of position noise in nm
        start_angle (float): Starting angle in radians (0 = positive x-axis)

    Returns:
        Tuple[np.ndarray, np.ndarray]: X and Y coordinates in nm

    Examples:
        # Simple ring of 8 molecules
        x_coords, y_coords = generate_ring_positions(8, 16, 69)

        # Larger ring with noise
        x_coords, y_coords = generate_ring_positions(12, 32, 69,
                                                   ring_radius_fraction=0.4,
                                                   add_noise=True,
                                                   noise_sigma=20.0)
    """
    # Calculate physical image dimensions
    physical_size = image_size * pixel_size
    center_x = physical_size / 2 + center_offset[0]
    center_y = physical_size / 2 + center_offset[1]

    # Calculate ring radius
    max_radius = physical_size / 2
    ring_radius = max_radius * ring_radius_fraction

    # Generate angles for N evenly spaced points
    angles = np.linspace(start_angle, start_angle + 2 * np.pi, N, endpoint=False)

    # Calculate positions
    x_coords = center_x + ring_radius * np.cos(angles)
    y_coords = center_y + ring_radius * np.sin(angles)

    # Add noise if requested
    if add_noise:
        x_noise = np.random.normal(0, noise_sigma, N)
        y_noise = np.random.normal(0, noise_sigma, N)
        x_coords += x_noise
        y_coords += y_noise

    # Ensure coordinates are within bounds
    x_coords = np.clip(x_coords, 0, physical_size)
    y_coords = np.clip(y_coords, 0, physical_size)

    return x_coords, y_coords


def generate_multi_ring_positions(
    N_per_ring: List[int],
    image_size: int,
    pixel_size: float,
    ring_radii_fractions: List[float],
    center_offset: Tuple[float, float] = (0.0, 0.0),
    add_noise: bool = False,
    noise_sigma: float = 10.0,
    start_angles: Optional[List[float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate molecular positions arranged in multiple concentric rings.

    Args:
        N_per_ring (List[int]): Number of molecules in each ring
        image_size (int): Size of the image in pixels
        pixel_size (float): Physical size of each pixel
        ring_radii_fractions (List[float]): Radius fraction for each ring
        center_offset (Tuple[float, float]): Offset from image center in nm
        add_noise (bool): Whether to add Gaussian noise to positions
        noise_sigma (float): Standard deviation of position noise in nm
        start_angles (Optional[List[float]]): Starting angles for each ring

    Returns:
        Tuple[np.ndarray, np.ndarray]: Combined X and Y coordinates in nm

    Example:
        # Two rings: inner with 6 molecules, outer with 12
        x_coords, y_coords = generate_multi_ring_positions(
            N_per_ring=[6, 12],
            image_size=32,
            pixel_size=69,
            ring_radii_fractions=[0.2, 0.4]
        )
    """
    if len(N_per_ring) != len(ring_radii_fractions):
        raise ValueError("N_per_ring and ring_radii_fractions must have same length")

    if start_angles is None:
        start_angles = [0.0] * len(N_per_ring)
    elif len(start_angles) != len(N_per_ring):
        raise ValueError("start_angles must have same length as N_per_ring")

    all_x_coords = []
    all_y_coords = []

    for N, radius_frac, start_angle in zip(N_per_ring, ring_radii_fractions, start_angles):
        x_coords, y_coords = generate_ring_positions(
            N=N,
            image_size=image_size,
            pixel_size=pixel_size,
            ring_radius_fraction=radius_frac,
            center_offset=center_offset,
            add_noise=add_noise,
            noise_sigma=noise_sigma,
            start_angle=start_angle
        )
        all_x_coords.append(x_coords)
        all_y_coords.append(y_coords)

    # Combine all rings
    combined_x = np.concatenate(all_x_coords)
    combined_y = np.concatenate(all_y_coords)

    return combined_x, combined_y


def positions_to_x0y0_format(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    dye_names: List[str],
    frames: int = 1,
    mode: str = "all",
    random_seed: Optional[int] = None,
    localization_noise_sigma: float = 0.0,
    image_size: Optional[int] = None,
    pixel_size: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Convert position arrays to the x0y0 format expected by simulation functions.

    Args:
        x_coords (np.ndarray): X coordinates in nm
        y_coords (np.ndarray): Y coordinates in nm
        dye_names (List[str]): List of dye names to assign molecules to
        frames (int): Number of frames
        mode (str): Activation mode:
            - "all": All molecules active in all frames (default)
            - "SMLM": Only one randomly selected molecule active per frame
            - "sequential": Molecules activate in order (cycling through)
        random_seed (Optional[int]): Random seed for reproducible SMLM simulation
        localization_noise_sigma (float): Standard deviation of Gaussian noise to add
            to molecular positions in nm (simulates localization uncertainty)
        image_size (Optional[int]): Image size in pixels (for bounds checking when adding noise)
        pixel_size (Optional[float]): Pixel size in nm (for bounds checking when adding noise)

    Returns:
        Dict[str, np.ndarray]: Dictionary in x0y0 format for simulation functions

    Examples:
        # All molecules active every frame
        x0y0 = positions_to_x0y0_format(x_coords, y_coords, ["ATTO 565"])

        # SMLM mode: one random molecule per frame
        x0y0 = positions_to_x0y0_format(x_coords, y_coords, ["ATTO 565"],
                                       frames=100, mode="SMLM")

        # SMLM with localization noise (realistic experiment)
        x0y0 = positions_to_x0y0_format(x_coords, y_coords, ["ATTO 565"],
                                       frames=100, mode="SMLM",
                                       localization_noise_sigma=15.0,
                                       image_size=32, pixel_size=69)

        # Sequential activation with noise
        x0y0 = positions_to_x0y0_format(x_coords, y_coords, ["ATTO 565"],
                                       frames=50, mode="sequential",
                                       localization_noise_sigma=10.0)
    """
    n_molecules = len(x_coords)
    n_dyes = len(dye_names)

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Distribute molecules among dyes
    molecules_per_dye = n_molecules // n_dyes
    remainder = n_molecules % n_dyes

    x0y0 = {}
    start_idx = 0

    for i, dye in enumerate(dye_names):
        # Calculate how many molecules this dye gets
        n_this_dye = molecules_per_dye + (1 if i < remainder else 0)
        end_idx = start_idx + n_this_dye

        # Extract coordinates for this dye
        dye_x = x_coords[start_idx:end_idx]
        dye_y = y_coords[start_idx:end_idx]

        if mode == "all":
            # All molecules active in all frames (original behavior)
            x0y0[dye] = np.zeros([frames, 2, n_this_dye])
            for frame in range(frames):
                x0y0[dye][frame, 0, :] = dye_x  # X coordinates
                x0y0[dye][frame, 1, :] = dye_y  # Y coordinates

        elif mode == "SMLM":
            # Only one randomly selected molecule active per frame
            # Format: [frames, 2, 1] - always 1 molecule per frame, correct dimension order
            x0y0[dye] = np.zeros([frames, 2, 1])

            for frame in range(frames):
                # Randomly select one molecule for this frame
                selected_idx = np.random.randint(0, n_this_dye)
                base_x = dye_x[selected_idx]
                base_y = dye_y[selected_idx]

                # Add Gaussian localization noise
                if localization_noise_sigma > 0:
                    noise_x = np.random.normal(0, localization_noise_sigma)
                    noise_y = np.random.normal(0, localization_noise_sigma)
                    noisy_x = base_x + noise_x
                    noisy_y = base_y + noise_y

                    # Apply bounds checking if image dimensions provided
                    if image_size is not None and pixel_size is not None:
                        max_coord = image_size * pixel_size
                        noisy_x = np.clip(noisy_x, 0, max_coord)
                        noisy_y = np.clip(noisy_y, 0, max_coord)

                    x0y0[dye][frame, 0, 0] = noisy_x  # X coordinate with noise
                    x0y0[dye][frame, 1, 0] = noisy_y  # Y coordinate with noise
                else:
                    x0y0[dye][frame, 0, 0] = base_x  # X coordinate
                    x0y0[dye][frame, 1, 0] = base_y  # Y coordinate

        elif mode == "sequential":
            # Molecules activate in sequential order (cycling through)
            # Format: [frames, 2, 1] - always 1 molecule per frame, correct dimension order
            x0y0[dye] = np.zeros([frames, 2, 1])

            for frame in range(frames):
                # Select molecule based on frame number (cycling through)
                selected_idx = frame % n_this_dye
                base_x = dye_x[selected_idx]
                base_y = dye_y[selected_idx]

                # Add Gaussian localization noise
                if localization_noise_sigma > 0:
                    noise_x = np.random.normal(0, localization_noise_sigma)
                    noise_y = np.random.normal(0, localization_noise_sigma)
                    noisy_x = base_x + noise_x
                    noisy_y = base_y + noise_y

                    # Apply bounds checking if image dimensions provided
                    if image_size is not None and pixel_size is not None:
                        max_coord = image_size * pixel_size
                        noisy_x = np.clip(noisy_x, 0, max_coord)
                        noisy_y = np.clip(noisy_y, 0, max_coord)

                    x0y0[dye][frame, 0, 0] = noisy_x  # X coordinate with noise
                    x0y0[dye][frame, 1, 0] = noisy_y  # Y coordinate with noise
                else:
                    x0y0[dye][frame, 0, 0] = base_x  # X coordinate
                    x0y0[dye][frame, 1, 0] = base_y  # Y coordinate

        else:
            raise ValueError(f"Unknown mode '{mode}'. Options: 'all', 'SMLM', 'sequential'")

        start_idx = end_idx

    return x0y0


def x0y0_to_localization_table(
    x0y0: Dict[str, np.ndarray],
    n_photons: Dict[str, Union[int, np.ndarray]],
    pixel_size: float = 69.0,
    default_sigma: float = 1.5,
    default_background: float = 10.0,
    localization_precision_nm: float = 15.0,
    amp_scale_factor: float = 1000.0
) -> np.ndarray:
    """
    Convert x0y0 simulation format to localization table format for render.py.

    This function converts molecular positions from the simulation format (x0y0)
    to a structured numpy array (recarray) that can be used with render.py
    for super-resolution image rendering.

    Args:
        x0y0 (Dict[str, np.ndarray]): Dictionary of molecular positions per dye
            Format: {dye_name: array([frames, 2, molecules])}
        n_photons (Dict[str, Union[int, np.ndarray]]): Photon counts per dye
        pixel_size (float): Pixel size in nm (default: 69.0)
        default_sigma (float): Default PSF sigma in pixels (default: 1.5)
        default_background (float): Default background level (default: 10.0)
        localization_precision_nm (float): Localization precision in nm (default: 15.0)
        amp_scale_factor (float): Scaling factor for amplitude values (default: 1000.0)

    Returns:
        np.ndarray: Structured array (recarray) with fields expected by render.py:
            - xc, yc: Position coordinates in nm
            - frame: Frame number (0-indexed)
            - A_R, A_G, A_B: Amplitude values for RGB channels
            - bg_R, bg_G, bg_B: Background values for RGB channels
            - s_x, s_y: PSF width in nm
            - xc_err, yc_err: Localization precision in nm

    Examples:
        # Convert SMLM simulation to localization table
        locs = x0y0_to_localization_table(x0y0, n_photons, pixel_size=69)

        # Use with render.py
        from src import render
        image, _, _, _, _, _ = render.render_hist(locs, oversampling=10)

        # Custom parameters
        locs = x0y0_to_localization_table(
            x0y0, n_photons,
            pixel_size=69,
            localization_precision_nm=12.0,  # High precision
            amp_scale_factor=2000.0          # Brighter rendering
        )
    """
    # Count total localizations across all dyes and frames
    total_locs = 0
    dye_info = {}

    for dye_name, positions in x0y0.items():
        frames, coords, molecules = positions.shape
        # Count non-zero positions (active molecules)
        active_positions = 0
        for frame in range(frames):
            for mol in range(molecules):
                # Check if molecule is active (non-zero position)
                if positions[frame, 0, mol] != 0 or positions[frame, 1, mol] != 0:
                    active_positions += 1

        dye_info[dye_name] = {
            'positions': positions,
            'active_count': active_positions,
            'photons': n_photons.get(dye_name, 1000)
        }
        total_locs += active_positions

    # Define the data type for localization table (compatible with render.py)
    loc_dtype = np.dtype([
        ('xc', 'f4'),      # X coordinate (nm)
        ('yc', 'f4'),      # Y coordinate (nm)
        ('frame', 'i4'),   # Frame number
        ('A_R', 'f4'),     # Red channel amplitude
        ('A_G', 'f4'),     # Green channel amplitude
        ('A_B', 'f4'),     # Blue channel amplitude
        ('bg_R', 'f4'),    # Red channel background
        ('bg_G', 'f4'),    # Green channel background
        ('bg_B', 'f4'),    # Blue channel background
        ('s_x', 'f4'),     # PSF sigma X (nm)
        ('s_y', 'f4'),     # PSF sigma Y (nm)
        ('xc_err', 'f4'),  # X localization error (nm)
        ('yc_err', 'f4'),  # Y localization error (nm)
    ])

    # Create structured array
    locs = np.zeros(total_locs, dtype=loc_dtype)

    # Convert PSF sigma from pixels to nm
    sigma_nm = default_sigma * pixel_size

    # Fill the localization table
    loc_idx = 0
    dye_colors = ['R', 'G', 'B']  # RGB channels

    for dye_idx, (dye_name, info) in enumerate(dye_info.items()):
        positions = info['positions']
        photon_count = info['photons']
        frames, coords, molecules = positions.shape

        # Determine which color channel this dye contributes to (cycle through RGB)
        primary_color = dye_colors[dye_idx % len(dye_colors)]

        # Calculate amplitude based on photon count
        amplitude = photon_count * amp_scale_factor / 1000.0  # Normalize to reasonable range

        for frame in range(frames):
            for mol in range(molecules):
                x_pos = positions[frame, 0, mol]
                y_pos = positions[frame, 1, mol]

                # Skip inactive molecules (position = 0,0 typically means inactive)
                if x_pos == 0 and y_pos == 0:
                    continue

                # Fill localization data
                locs[loc_idx]['xc'] = x_pos
                locs[loc_idx]['yc'] = y_pos
                locs[loc_idx]['frame'] = frame

                # Set amplitude for primary color channel
                locs[loc_idx]['A_R'] = amplitude if primary_color == 'R' else 0
                locs[loc_idx]['A_G'] = amplitude if primary_color == 'G' else 0
                locs[loc_idx]['A_B'] = amplitude if primary_color == 'B' else 0

                # Set background levels
                locs[loc_idx]['bg_R'] = default_background
                locs[loc_idx]['bg_G'] = default_background
                locs[loc_idx]['bg_B'] = default_background

                # Set PSF parameters
                locs[loc_idx]['s_x'] = sigma_nm
                locs[loc_idx]['s_y'] = sigma_nm

                # Set localization precision
                locs[loc_idx]['xc_err'] = localization_precision_nm
                locs[loc_idx]['yc_err'] = localization_precision_nm

                loc_idx += 1

    # Convert to recarray for compatibility with render.py
    return locs.view(np.recarray)


def plot_ring_positions(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    image_size: int,
    pixel_size: float,
    title: str = "Ring Positions"
) -> None:
    """
    Plot the generated ring positions for visualization.

    Args:
        x_coords (np.ndarray): X coordinates in nm
        y_coords (np.ndarray): Y coordinates in nm
        image_size (int): Size of the image in pixels
        pixel_size (float): Physical size of each pixel
        title (str): Plot title
    """
    physical_size = image_size * pixel_size

    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, c='red', s=100, alpha=0.7, edgecolors='black')

    # Add molecule numbers
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        plt.annotate(f'{i+1}', (x, y), xytext=(5, 5),
                    textcoords='offset points', fontsize=10)

    # Draw image boundary
    plt.plot([0, physical_size, physical_size, 0, 0],
             [0, 0, physical_size, physical_size, 0],
             'k--', alpha=0.5, label='Image boundary')

    # Draw center
    center = physical_size / 2
    plt.plot(center, center, 'b+', markersize=15, markeredgewidth=2, label='Center')

    plt.xlim(-physical_size*0.1, physical_size*1.1)
    plt.ylim(-physical_size*0.1, physical_size*1.1)
    plt.xlabel('X (nm)')
    plt.ylabel('Y (nm)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


# Example usage and test functions
if __name__ == "__main__":
    # Example 1: Simple ring
    print("Example 1: Simple ring of 8 molecules")
    x_coords, y_coords = generate_ring_positions(8, 16, 69)
    print(f"Coordinates: {list(zip(x_coords, y_coords))}")

    # Example 2: Convert to x0y0 format
    print("\nExample 2: Convert to x0y0 format")
    x0y0 = positions_to_x0y0_format(x_coords, y_coords, ["ATTO 565"])
    print(f"x0y0 format: {x0y0}")

    # Example 3: Multiple rings
    print("\nExample 3: Multiple concentric rings")
    x_multi, y_multi = generate_multi_ring_positions(
        N_per_ring=[6, 12],
        image_size=32,
        pixel_size=69,
        ring_radii_fractions=[0.2, 0.4]
    )
    print(f"Multi-ring coordinates: {len(x_multi)} total molecules")

    # Plot examples
    plot_ring_positions(x_coords, y_coords, 16, 69, "Simple Ring (8 molecules)")
    plot_ring_positions(x_multi, y_multi, 32, 69, "Multi-Ring (6 + 12 molecules)")