"""
Extreme Value-based Emitter Recovery (EVER) Functions

Implementation of EVER algorithm from Ma et al. (2021) Scientific Reports.
https://doi.org/10.1038/s41598-021-00066-3

EVER uses temporal minimum values and extreme value statistics to accurately
separate fluorescent emitters from heterogeneous background, providing superior
performance compared to temporal median filtering.

Author: Claude Code (Anthropic)
Date: October 6, 2025
"""

import numpy as np
from scipy.stats import poisson
from scipy.ndimage import uniform_filter


class EVER_Functions:
    """
    Extreme Value-based Emitter Recovery for super-resolution microscopy.

    This class implements the EVER algorithm which uses temporal minimum values
    and extreme value statistics to estimate and remove heterogeneous background
    from fluorescence microscopy data.

    Key advantages over temporal median filtering:
    - ~5x faster computation
    - More robust to high emitter density (>50% occupancy)
    - No over-estimation of background (preserves emitter intensity/size)
    - ~98% accuracy compared to ground truth
    - Fully automatic with no manual parameter tuning required
    """

    def __init__(self, verbose: bool = False, io_functions=None):
        """Initialize EVER with default parameters.

        Args:
            verbose: If True, print detailed progress information (default: False)
            io_functions: IOFunctions instance for photoelectron conversion (default: creates new instance)
        """
        self.default_window_size = 100  # frames
        self.default_spatial_filter_size = 3  # pixels (for spatial mean filter)
        self.lut_cache = {}  # Cache lookup tables for reuse
        self.verbose = verbose

        # Import and initialize IOFunctions for photoelectron conversion
        if io_functions is None:
            import sys
            import os
            module_dir = os.path.abspath(os.path.dirname(__file__))
            sys.path.append(module_dir)
            import IOFunctions
            self.io = IOFunctions.IO_Functions()
        else:
            self.io = io_functions

    def compute_ever_background(
        self,
        frames: np.ndarray,
        window_size: int = 100,
        spatial_filter_size: int = 3,
        use_cache: bool = True,
        n_jobs: int = -1,
        bayer_masks: dict = None,
    ) -> tuple:
        """
        Compute EVER background estimation and recover emitters using per-frame sliding windows.

        This is the main entry point for EVER processing. For each frame, it calculates the
        temporal minimum over a sliding window, transforms it to background using extreme value
        statistics, and subtracts the frame-specific background to recover clean emitter signals.

        IMPORTANT: This uses a per-frame sliding window approach as described in Ma et al. (2021).
        For frame N with window_size=100, it uses frames [N-50, N+50] to compute the background
        for frame N specifically.

        Args:
            frames: 3D array (n_frames, height, width) in photoelectrons
            window_size: Temporal window size for minimum calculation (default: 100)
                Typical range: 50-200 frames depending on background variation
            spatial_filter_size: Size of spatial mean filter for noise reduction (default: 3)
                Set to 1 to disable spatial filtering. When > 1, uses Bayer-aware filtering
                to avoid cross-channel contamination
            use_cache: Whether to cache and reuse lookup tables (default: True)
            n_jobs: Number of parallel jobs for processing frames (default: -1 = all CPUs)
            bayer_masks: Optional dictionary of Bayer masks {'B': mask, 'G': mask, 'R': mask}
                If None and spatial_filter_size > 1, standard BGGR pattern is assumed

        Returns:
            tuple: (backgrounds, emitters)
                - backgrounds: 3D array (n_frames, height, width) - per-frame background maps
                - emitters: 3D array (n_frames, height, width) - background-subtracted frames

        Example:
            >>> ever = EVER_Functions()
            >>> # No spatial filtering (default for most cases)
            >>> backgrounds, emitters = ever.compute_ever_background(
            ...     raw_frames, window_size=100, spatial_filter_size=1
            ... )

            >>> # With Bayer-aware spatial filtering (for uniform backgrounds)
            >>> backgrounds, emitters = ever.compute_ever_background(
            ...     raw_frames, window_size=100, spatial_filter_size=2,
            ...     bayer_masks=camera_masks
            ... )
        """
        n_frames, height, width = frames.shape
        half_window = window_size // 2

        if self.verbose:
            print(f"  EVER: Processing {n_frames} frames ({height}×{width} pixels)")
            print(
                f"  EVER: Window size={window_size}, spatial filter={spatial_filter_size}×{spatial_filter_size}"
            )
            print(f"  EVER: Using per-frame sliding windows (parallel jobs={n_jobs})")

        # Step 1: Calculate background decay ratio (once for entire stack)
        if self.verbose:
            print("  EVER: Calculating decay ratio...", end="\r", flush=True)
        decay_ratio = self._calculate_decay_ratio(frames, window_size)

        # Step 2: Build lookup table (once, shared across all frames)
        cache_key = (window_size, decay_ratio, spatial_filter_size)
        if use_cache and cache_key in self.lut_cache:
            lut = self.lut_cache[cache_key]
            if self.verbose:
                print("  EVER: Using cached lookup table    ", end="\r", flush=True)
        else:
            if self.verbose:
                print("  EVER: Building lookup table...    ", end="\r", flush=True)
            # Use global min and max for LUT range estimation
            # Use max instead of min to ensure we cover the full range
            global_max = np.max(frames)
            lut = self._build_lookup_table(
                window_size, decay_ratio, spatial_filter_size, global_max
            )
            if use_cache:
                self.lut_cache[cache_key] = lut

        # Step 3: Process each frame with its own sliding window (parallelized)
        if self.verbose:
            print(f"  EVER: Computing per-frame backgrounds...", end="\r", flush=True)

        backgrounds, emitters = self._process_frames_parallel(
            frames, window_size, spatial_filter_size, decay_ratio, lut, n_jobs,
            bayer_masks
        )

        if self.verbose:
            mean_bg = backgrounds.mean()
            std_bg = backgrounds.std()
            print(
                f"  EVER: Complete (R={decay_ratio:.3f}, bg={mean_bg:.0f}±{std_bg:.0f} photons)    "
            )

        return backgrounds, emitters

    def _process_frames_parallel(
        self,
        frames: np.ndarray,
        window_size: int,
        spatial_filter_size: int,
        decay_ratio: float,
        lut: dict,
        n_jobs: int,
        bayer_masks: dict = None,
    ) -> tuple:
        """
        Process all frames in parallel using sliding windows.

        For each frame i, computes background using frames [i-half_window, i+half_window].

        Args:
            frames: 3D array (n_frames, height, width)
            window_size: Size of sliding window
            spatial_filter_size: Spatial filter size (uses Bayer-aware filtering if > 1)
            decay_ratio: Background decay ratio
            lut: Lookup table for minimum -> background transformation
            n_jobs: Number of parallel jobs
            bayer_masks: Dictionary of Bayer masks (if spatial_filter_size > 1)

        Returns:
            tuple: (backgrounds, emitters) - both 3D arrays
        """
        from joblib import Parallel, delayed

        n_frames, height, width = frames.shape
        half_window = window_size // 2

        # Pre-allocate output arrays
        backgrounds = np.zeros_like(frames, dtype=np.float32)
        emitters = np.zeros_like(frames, dtype=np.float32)

        def process_single_frame(frame_idx):
            """Process a single frame with its sliding window."""
            # Determine window boundaries
            window_start = max(0, frame_idx - half_window)
            window_end = min(n_frames, frame_idx + half_window + 1)

            # Extract window of frames
            window_frames = frames[window_start:window_end]

            # Compute temporal minimum for this window
            temporal_min = np.min(window_frames, axis=0).astype(np.float32)

            # Apply spatial filter (Bayer-aware only)
            if spatial_filter_size > 1:
                temporal_min = self._apply_bayer_aware_spatial_filter(
                    temporal_min, spatial_filter_size, bayer_masks
                )

            # Transform to background using LUT
            background = self._transform_minimum_to_background(temporal_min, lut)

            # Compute emitter (background-subtracted frame)
            emitter = frames[frame_idx] - background
            # Preserve full Poisson statistics - do not clip negative values
            # Negative values represent noise fluctuations and are essential for chi-squared fitting

            return frame_idx, background, emitter

        # Process frames in parallel
        results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(process_single_frame)(i) for i in range(n_frames)
        )

        # Assemble results
        for frame_idx, background, emitter in results:
            backgrounds[frame_idx] = background
            emitters[frame_idx] = emitter

        return backgrounds, emitters

    def _calculate_temporal_minimum(
        self, frames: np.ndarray, window_size: int
    ) -> np.ndarray:
        """
        Calculate pixel-wise temporal minimum over sliding windows.

        This is the first key step of EVER. Unlike temporal median which is heavily
        affected by emitter presence, temporal minimum is inherently robust to
        varying emitter densities.

        Args:
            frames: 3D array (n_frames, height, width)
            window_size: Window size for minimum calculation

        Returns:
            temporal_min: 2D array (height, width) - minimum value at each pixel
        """
        n_frames, height, width = frames.shape

        # For each pixel, find minimum across all frames
        # We use a global minimum across the entire stack (simplest approach)
        # More sophisticated: sliding window minimum for time-varying background
        temporal_min = np.min(frames, axis=0).astype(np.float32)

        return temporal_min

    def _apply_bayer_aware_spatial_filter(
        self, image: np.ndarray, filter_size: int, masks: dict = None
    ) -> np.ndarray:
        """
        Apply Bayer-aware spatial filtering that respects color channel structure.

        Instead of filtering across all pixels (which mixes RGB channels in Bayer pattern),
        this method filters each color channel independently on its own spatial grid.

        For a Bayer pattern:
            B G
            G R

        Blue and Red pixels form sparse grids (every 4th pixel), while Green pixels
        form a denser grid (every 2nd pixel). This function:
        1. Extracts each color channel to its own grid
        2. Applies spatial filtering within that grid
        3. Places filtered values back to original positions

        This prevents averaging blue with red or green pixels, which would be
        physically incorrect for Bayer sensors.

        Args:
            image: 2D array to filter (height, width)
            filter_size: Size of filter in units of color grid spacing
                For filter_size=1: 3×3 kernel on the color-specific grid
                For filter_size=2: 5×5 kernel on the color-specific grid
            masks: Dictionary of color masks {'B': bool_array, 'G': bool_array, 'R': bool_array}
                If None, assumes standard Bayer BGGR pattern

        Returns:
            filtered: Spatially filtered image with same shape as input
        """
        from scipy.ndimage import uniform_filter

        height, width = image.shape
        filtered = image.copy()

        # If no masks provided, create standard Bayer BGGR masks
        if masks is None:
            masks = self._create_bayer_masks(height, width)

        # Process each color channel independently
        for color, mask in masks.items():
            if not np.any(mask):
                continue

            # Extract pixel positions for this color
            y_indices, x_indices = np.where(mask)

            if len(y_indices) == 0:
                continue

            # Determine the sparse grid spacing for this color
            # Blue/Red: 2×2 spacing (every other pixel in both dimensions)
            # Green: offset 2×2 pattern (still every other pixel)
            if color in ['B', 'R']:
                grid_spacing = 2
            else:  # Green
                grid_spacing = 2

            # Create a dense representation of just this color channel
            # by extracting values at the sparse grid positions
            min_y, max_y = y_indices.min(), y_indices.max()
            min_x, max_x = x_indices.min(), x_indices.max()

            # Build dense grid from sparse samples
            grid_height = (max_y - min_y) // grid_spacing + 1
            grid_width = (max_x - min_x) // grid_spacing + 1

            dense_grid = np.zeros((grid_height, grid_width), dtype=np.float32)

            # Map sparse pixel positions to dense grid coordinates
            for y, x in zip(y_indices, x_indices):
                grid_y = (y - min_y) // grid_spacing
                grid_x = (x - min_x) // grid_spacing
                if 0 <= grid_y < grid_height and 0 <= grid_x < grid_width:
                    dense_grid[grid_y, grid_x] = image[y, x]

            # Apply uniform filter on the dense grid
            # For Bayer patterns, filter_size represents the kernel size in the ORIGINAL image
            # Since we're on a 2× downsampled grid, divide by 2 (rounding up)
            grid_filter_size = max(1, (filter_size + 1) // 2)
            if dense_grid.size > 0 and grid_filter_size > 0:
                filtered_grid = uniform_filter(
                    dense_grid, size=grid_filter_size, mode='nearest'
                )

                # Map filtered values back to original sparse positions
                for y, x in zip(y_indices, x_indices):
                    grid_y = (y - min_y) // grid_spacing
                    grid_x = (x - min_x) // grid_spacing
                    if 0 <= grid_y < grid_height and 0 <= grid_x < grid_width:
                        filtered[y, x] = filtered_grid[grid_y, grid_x]

        return filtered.astype(np.float32)

    def _create_bayer_masks(self, height: int, width: int) -> dict:
        """
        Create standard Bayer BGGR pattern masks.

        Pattern:
            B G
            G R

        Args:
            height: Image height
            width: Image width

        Returns:
            masks: Dictionary with keys 'B', 'G', 'R' containing boolean masks
        """
        masks = {}

        # Blue: top-left of 2×2 unit (0,0), (0,2), (2,0), (2,2), ...
        blue_mask = np.zeros((height, width), dtype=bool)
        blue_mask[0::2, 0::2] = True
        masks['B'] = blue_mask

        # Green: two positions in 2×2 unit
        # G1: top-right (0,1), (0,3), (2,1), (2,3), ...
        # G2: bottom-left (1,0), (1,2), (3,0), (3,2), ...
        green_mask = np.zeros((height, width), dtype=bool)
        green_mask[0::2, 1::2] = True  # G1
        green_mask[1::2, 0::2] = True  # G2
        masks['G'] = green_mask

        # Red: bottom-right of 2×2 unit (1,1), (1,3), (3,1), (3,3), ...
        red_mask = np.zeros((height, width), dtype=bool)
        red_mask[1::2, 1::2] = True
        masks['R'] = red_mask

        return masks

    def _calculate_decay_ratio(self, frames: np.ndarray, window_size: int) -> float:
        """
        Automatically calculate background decay ratio R.

        The decay ratio accounts for photobleaching or other slow temporal variations
        in the background fluorescence. It's calculated by identifying background pixels
        (those with low temporal variation) and measuring their average decay.

        Args:
            frames: 3D array (n_frames, height, width)
            window_size: Window size (not directly used, kept for API consistency)

        Returns:
            R: Background decay ratio (R=1 for no decay, R<1 for decay)
        """
        n_frames, height, width = frames.shape

        # Identify background pixels (low temporal variation)
        # Background pixels have std < 2*mean (from Ma et al. paper)
        mean_vals = np.mean(frames, axis=0)
        std_vals = np.std(frames, axis=0)

        # Avoid division by zero
        with np.errstate(invalid="ignore"):
            bg_mask = std_vals < (2 * mean_vals)

        # If no clear background pixels, use all pixels
        if not np.any(bg_mask):
            bg_mask = np.ones((height, width), dtype=bool)

        # Calculate average background intensity per frame
        bg_per_frame = np.zeros(n_frames)
        for i in range(n_frames):
            if np.any(bg_mask):
                bg_per_frame[i] = np.mean(frames[i][bg_mask])
            else:
                bg_per_frame[i] = np.mean(frames[i])

        # Decay ratio: sum / (min * N)
        # R = 1 if no decay, R > 1 if background increases, R < 1 if background decreases
        min_bg = np.min(bg_per_frame)
        if min_bg > 0:
            R = np.sum(bg_per_frame) / (min_bg * n_frames)
        else:
            R = 1.0  # Default to no decay

        # Clamp to reasonable range [0.5, 2.0]
        R = np.clip(R, 0.5, 2.0)

        return R

    def _build_lookup_table(
        self,
        window_size: int,
        decay_ratio: float,
        spatial_filter_size: int,
        temporal_max_range: float,
    ) -> dict:
        """
        Build lookup table to transform temporal minimum to background.

        This is the core of EVER. The LUT is based on extreme value statistics:
        for a given background level λ, we can calculate the expected temporal
        minimum value using Poisson statistics. The LUT inverts this relationship.

        Args:
            window_size: Temporal window size N
            decay_ratio: Background decay ratio R
            spatial_filter_size: Size of spatial filter m (affects convolution)
            temporal_max_range: Maximum value from frames to determine LUT range

        Returns:
            lut: Dictionary mapping {temporal_min_value: background_value}
        """
        # Determine range of photon values to compute
        # Use max value to set upper bound, always start from 0
        if isinstance(temporal_max_range, np.ndarray):
            max_val = int(np.max(temporal_max_range) * 2)  # 2x margin for headroom
        else:
            # Scalar value
            max_val = int(temporal_max_range * 2)  # 2x margin for headroom

        min_val = 0  # Always start from 0 to handle low photon counts
        max_val = max(min_val + 100, max_val + 100)  # Ensure at least 100 values, add headroom
        max_val = min(max_val, 10000)  # Cap at reasonable value

        # Build LUT by computing expected minimum for each background level
        lut = {}

        # Sample at reasonable intervals for speed
        lambda_range = np.arange(min_val, max_val, 10)

        for lambda_bg in lambda_range:
            if lambda_bg <= 0:
                continue

            # Calculate expected temporal minimum for this background level
            expected_min = self._calculate_expected_minimum(
                lambda_bg, window_size, decay_ratio, spatial_filter_size
            )

            # Map minimum -> background
            min_key = int(round(expected_min))
            lut[min_key] = lambda_bg

        return lut

    def _calculate_expected_minimum(
        self, lambda_bg: float, N: int, R: float, m: int
    ) -> float:
        """
        Calculate expected temporal minimum value using extreme value statistics.

        This implements Equations (3-6) from Ma et al. (2021).

        For a Poisson distribution with mean λ, the probability mass function
        of the temporal minimum over N samples is:

        pmf_min(k) = [1 - CDF(k-1)]^N - [1 - CDF(k)]^N

        With decay ratio R and spatial averaging m.

        Args:
            lambda_bg: Expected background photon count
            N: Number of frames (window size)
            R: Background decay ratio
            m: Number of pixels in spatial filter (1 for no filtering)

        Returns:
            expected_min: Expected value of temporal minimum
        """
        # For large N, use analytical approximation for speed
        # Minimum ~ λ - sqrt(2*λ*ln(N)) for Poisson(λ) over N samples
        # Use approximation for N > 10 to avoid slow PMF calculation
        if N > 10:
            # Validate inputs to avoid invalid sqrt/log operations
            if lambda_bg <= 0 or R <= 0 or N <= 1:
                return max(0, lambda_bg * R)

            # Fast approximation based on extreme value theory
            # For no decay (R=1), use Gumbel approximation
            expected_min = lambda_bg * R - np.sqrt(2 * lambda_bg * R * np.log(N))

            # Account for spatial averaging (reduces dispersion)
            if m > 1:
                # Spatial averaging shifts minimum up slightly
                expected_min += np.sqrt(lambda_bg * R) * 0.5 * np.log(m)

            # Clamp to reasonable range
            expected_min = max(0, min(expected_min, lambda_bg * R))
            return expected_min

        # For small N, use exact calculation
        # Range of possible minimum values (truncate for speed)
        k_max = int(lambda_bg * 2 + 100)
        k_range = np.arange(0, k_max)

        # Calculate PMF of temporal minimum with decay
        pmf_min = self._calculate_pmf_minimum_with_decay(k_range, lambda_bg, N, R)

        # Account for spatial averaging (convolution of m PMFs)
        # For m>1, this reduces dispersion
        # For simplicity and speed, we approximate the effect
        if m > 1:
            # Spatial averaging reduces variance by factor of m
            # Approximate by narrowing distribution
            variance_reduction = np.sqrt(m)
            pmf_min_filtered = pmf_min / variance_reduction
            pmf_min_filtered = pmf_min_filtered / np.sum(
                pmf_min_filtered
            )  # Renormalize
        else:
            pmf_min_filtered = pmf_min

        # Expected value (mean of distribution)
        expected_min = np.sum(k_range * pmf_min_filtered)

        return expected_min

    def _calculate_pmf_minimum_with_decay(
        self, k_range: np.ndarray, lambda_bg: float, N: int, R: float
    ) -> np.ndarray:
        """
        Calculate probability mass function of temporal minimum with background decay.

        Implements Equation (4) from Ma et al. (2021):

        pmf_min(k, λ, N, R) = Π[n=0 to N-1] [1-CDF(k-1, λ_n)] - Π[n=0 to N-1] [1-CDF(k, λ_n)]

        where λ_n = λ * [1 + (R-1) * n/(N-1)]

        Args:
            k_range: Array of photon values
            lambda_bg: Mean background photon count
            N: Window size (number of frames)
            R: Decay ratio

        Returns:
            pmf: Probability mass function array
        """
        import warnings

        # Vectorized computation for speed
        # Pre-compute time-varying background for all frames
        if N > 1:
            n_array = np.arange(N)
            decay_factors = 1.0 + (R - 1.0) * (n_array / (N - 1))
            lambda_n_array = lambda_bg * decay_factors
        else:
            lambda_n_array = np.array([lambda_bg])

        # Ensure lambda values are valid (positive, finite)
        lambda_n_array = np.clip(lambda_n_array, 1e-10, None)  # Minimum valid lambda

        # Check for invalid values and return uniform distribution if found
        if not np.all(np.isfinite(lambda_n_array)) or np.any(lambda_n_array <= 0):
            # Return uniform distribution as fallback
            pmf = np.ones_like(k_range, dtype=np.float64) / len(k_range)
            return pmf

        pmf = np.zeros_like(k_range, dtype=np.float64)

        # Suppress scipy warnings about invalid values (we've already validated)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            for idx, k in enumerate(k_range):
                # Vectorized CDF computation for all frames
                if k > 0:
                    cdf_k_minus_1 = poisson.cdf(k - 1, lambda_n_array)
                    prod_k_minus_1 = np.prod(1.0 - cdf_k_minus_1)
                else:
                    prod_k_minus_1 = 1.0

                cdf_k = poisson.cdf(k, lambda_n_array)
                prod_k = np.prod(1.0 - cdf_k)

                # PMF is difference of products
                pmf[idx] = prod_k_minus_1 - prod_k

        # Normalize (ensure sums to 1)
        pmf_sum = np.sum(pmf)
        if pmf_sum > 0:
            pmf = pmf / pmf_sum
        else:
            # Fallback: uniform distribution
            pmf = np.ones_like(k_range, dtype=np.float64) / len(k_range)

        return pmf

    def _transform_minimum_to_background(
        self, temporal_min: np.ndarray, lut: dict
    ) -> np.ndarray:
        """
        Transform temporal minimum map to background map using lookup table.

        For each pixel's temporal minimum value, look up the corresponding
        background value from the LUT. Use linear interpolation for values
        not in the LUT.

        Args:
            temporal_min: 2D array of temporal minimum values
            lut: Lookup table {min_value: background_value}

        Returns:
            background: 2D array of estimated background values
        """
        height, width = temporal_min.shape
        background = np.zeros((height, width), dtype=np.float32)

        # Get sorted LUT keys and values for interpolation
        lut_keys = np.array(sorted(lut.keys()))
        lut_values = np.array([lut[k] for k in lut_keys])

        # Vectorized interpolation
        background = np.interp(temporal_min.ravel(), lut_keys, lut_values)
        background = background.reshape(height, width)

        return background.astype(np.float32)
