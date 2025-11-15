"""
Spatial-spectral refinement methods for channel unmixing.

This module extends the extract_SMs class with hierarchical spatial-spectral
refinement for improved channel unmixing in multi-color SMLM data.

Author: Claude/JSB
Date: 2025-11-14
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy.stats import multivariate_normal
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import KDTree


def unmix_channels_with_spatial_refinement(
    self,
    loc_data: pd.DataFrame,
    n_channels: int,
    channels_to_use: list = ['A_R', 'A_G'],

    # Initial spectral unmixing parameters
    confidence_threshold_initial: float = 0.95,
    gmm_fit_method: str = 'fixed',
    initial_guess_percentile: float = 50,
    initial_guess_scale: float = 0.5,

    # Spatial clustering parameters
    spatial_eps: Optional[float] = None,
    min_cluster_size: int = 10,
    spatial_method: str = 'DBSCAN',

    # Hierarchical refinement parameters
    confidence_threshold_clear: float = 0.80,
    confidence_threshold_overlap: float = 0.90,
    max_iterations: int = 5,
    min_new_assignments: int = 10,

    # Diagnostic parameters
    verbose: bool = True,
    plot_results: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform channel unmixing with iterative hierarchical spatial-spectral refinement.

    This function improves upon pure spectral unmixing by iteratively recovering
    localizations that were initially unassigned due to moderate spectral confidence,
    but are spatially coincident with confidently-assigned puncta.

    The key innovation is adaptive confidence thresholds based on spatial context:
    - Clear regions (near puncta from 1 channel): Lower threshold (0.80)
    - Overlap regions (near puncta from 2+ channels): Higher threshold (0.90)

    Args:
        loc_data: DataFrame with localization data (xc, yc, A_R, A_G, etc.)
        n_channels: Number of spectral channels (typically 2)
        channels_to_use: Spectral features for unmixing (e.g., ['A_R', 'A_G'])

        confidence_threshold_initial: High threshold for initial seed assignments (0.95)
        gmm_fit_method: 'fixed' (recommended), 'EM', or 'extreme_deconvolution'
        initial_guess_percentile: Percentile for initial covariance estimation (50)
        initial_guess_scale: Scaling factor for initial covariances (0.5)

        spatial_eps: DBSCAN epsilon (distance threshold for neighbors)
                     If None, auto-calculated as mean([median(xc_err), median(yc_err)])
        min_cluster_size: Minimum localizations to form a valid punctum (default: 10)
        spatial_method: 'DBSCAN' or 'HDBSCAN' for spatial clustering

        confidence_threshold_clear: Spectral threshold for clear regions (0.80)
        confidence_threshold_overlap: Spectral threshold for overlap regions (0.90)
        max_iterations: Maximum refinement iterations (5)
        min_new_assignments: Stop if fewer than this assigned per iteration (10)

        verbose: Print progress
        plot_results: Create diagnostic plots

    Returns:
        assigned_locs: DataFrame with added columns:
            - 'channel': Final channel assignment
            - 'assignment_stage': 'initial', 'refinement_iter_1', etc.
            - 'spatial_cluster_id': ID of spatial cluster (punctum)
            - 'is_spatial_overlap': Whether assigned in overlap region
            - 'nearest_punctum_distance': Spatial distance to nearest punctum

        metadata: Dict with refinement statistics

    Example:
        >>> assigned, meta = SM_E.unmix_channels_with_spatial_refinement(
        ...     loc_data,
        ...     n_channels=2,
        ...     channels_to_use=['A_R', 'A_G'],
        ...     verbose=True
        ... )
        >>> print(f"Recovered: {meta['n_recovered_total']} locs")
    """
    if verbose:
        print("=" * 80)
        print("Hierarchical Spatial-Spectral Channel Unmixing")
        print("=" * 80)
        print(f"Input: {len(loc_data):,} localizations")
        print(f"Channels: {n_channels}")
        print(f"Features: {channels_to_use}")
        print()

    # ===== STEP 1: Initial Conservative Spectral Unmixing =====
    if verbose:
        print("=" * 80)
        print("STEP 1: Initial Spectral Unmixing (Conservative Seeds)")
        print("=" * 80)

    assigned_initial, metadata = self.unmix_channels(
        loc_data,
        n_channels=n_channels,
        channels_to_use=channels_to_use,
        confidence_threshold=confidence_threshold_initial,
        gmm_fit_method=gmm_fit_method,
        initial_guess_percentile=initial_guess_percentile,
        initial_guess_scale=initial_guess_scale,
        covariance_type='full',
        outlier_rejection='mahalanobis',
        verbose=verbose,
        plot_results=False,  # Save plotting for the end
    )

    # Extract GMM parameters
    means = metadata['means']
    covariances = metadata['covariances']
    weights = metadata['weights']

    # Track assignment stage
    assigned_initial['assignment_stage'] = 'unassigned'
    assigned_initial.loc[assigned_initial['channel'] >= 0, 'assignment_stage'] = 'initial'

    n_assigned_initial = {k: (assigned_initial['channel'] == k).sum()
                          for k in range(n_channels)}
    n_unassigned_initial = (assigned_initial['channel'] == -1).sum()

    if verbose:
        print(f"\nInitial assignments (confidence ≥ {confidence_threshold_initial}):")
        for k in range(n_channels):
            print(f"  Channel {k}: {n_assigned_initial[k]:,} locs")
        print(f"  Unassigned: {n_unassigned_initial:,} locs")
        print()

    # ===== STEP 2: Spatial Clustering Per Channel =====
    if verbose:
        print("=" * 80)
        print("STEP 2: Spatial Clustering of Seeds (per channel)")
        print("=" * 80)

    # Auto-calculate spatial epsilon from seed localizations
    spatial_eps, puncta_per_channel, spatial_cluster_ids = self._cluster_seeds_spatially(
        assigned_initial,
        n_channels,
        spatial_eps=spatial_eps,
        min_cluster_size=min_cluster_size,
        spatial_method=spatial_method,
        verbose=verbose
    )

    assigned_initial['spatial_cluster_id'] = spatial_cluster_ids

    # ===== STEP 3: Hierarchical Iterative Refinement =====
    if verbose:
        print("=" * 80)
        print("STEP 3: Hierarchical Spatial-Spectral Refinement")
        print("=" * 80)
        print(f"Spectral thresholds:")
        print(f"  Clear regions (1 channel nearby):    {confidence_threshold_clear:.2f}")
        print(f"  Overlap regions (2+ channels nearby): {confidence_threshold_overlap:.2f}")
        print(f"(Higher threshold in overlap regions accounts for spatial ambiguity)")
        print()

    # Calculate posterior probabilities for ALL localizations
    X = loc_data[channels_to_use].values
    n_locs = len(X)

    posterior_probs, most_likely_channel, confidence_per_loc = self._calculate_posteriors(
        X, means, covariances, weights, n_channels
    )

    # Build spatial indices for fast queries
    puncta_kdtrees, puncta_members = self._build_puncta_kdtrees(
        assigned_initial, n_channels, verbose=verbose
    )

    # Iterative refinement
    assigned_current, assignments_per_iteration = self._iterative_spatial_spectral_refinement(
        assigned_initial,
        most_likely_channel,
        confidence_per_loc,
        puncta_kdtrees,
        n_channels,
        spatial_eps,
        confidence_threshold_clear,
        confidence_threshold_overlap,
        max_iterations,
        min_new_assignments,
        verbose=verbose
    )

    # ===== STEP 4: Final Statistics and Output =====
    n_assigned_final = {k: (assigned_current['channel'] == k).sum()
                        for k in range(n_channels)}
    n_unassigned_final = (assigned_current['channel'] == -1).sum()

    n_recovered = {k: n_assigned_final[k] - n_assigned_initial[k]
                   for k in range(n_channels)}
    n_recovered_total = sum(n_recovered.values())

    # Create final metadata
    metadata_final = {
        **metadata,  # Include initial GMM metadata
        'n_assigned_initial': n_assigned_initial,
        'n_assigned_final': n_assigned_final,
        'n_unassigned_initial': n_unassigned_initial,
        'n_unassigned_final': n_unassigned_final,
        'n_recovered': n_recovered,
        'n_recovered_total': n_recovered_total,
        'n_iterations': len(assignments_per_iteration),
        'assignments_per_iteration': assignments_per_iteration,
        'puncta_per_channel': puncta_per_channel,
        'spatial_eps': spatial_eps,
        'confidence_threshold_clear': confidence_threshold_clear,
        'confidence_threshold_overlap': confidence_threshold_overlap,
    }

    if verbose:
        print("=" * 80)
        print("Refinement Complete")
        print("=" * 80)
        print(f"\nFinal assignments:")
        for k in range(n_channels):
            print(f"  Channel {k}: {n_assigned_final[k]:,} locs (+{n_recovered[k]:,} from refinement)")
        print(f"  Unassigned: {n_unassigned_final:,} locs")
        print(f"\nTotal recovered: {n_recovered_total:,} locs ({100*n_recovered_total/len(loc_data):.2f}%)")
        print()

    # ===== STEP 5: Diagnostic Plotting =====
    if plot_results:
        self._plot_refinement_diagnostics(
            assigned_current,
            metadata_final,
            n_channels,
            channels_to_use
        )

    return assigned_current, metadata_final


# Helper methods follow...


def _cluster_seeds_spatially(
    self,
    assigned_initial: pd.DataFrame,
    n_channels: int,
    spatial_eps: Optional[float],
    min_cluster_size: int,
    spatial_method: str,
    verbose: bool
) -> Tuple[float, Dict[int, int], np.ndarray]:
    """
    Perform spatial clustering of seed localizations per channel.

    Returns:
        spatial_eps: Calculated or provided epsilon
        puncta_per_channel: Number of valid puncta per channel
        spatial_cluster_ids: Array of cluster IDs for each localization
    """
    # Auto-calculate spatial epsilon from conservatively-assigned seeds
    if spatial_eps is None:
        seed_mask = assigned_initial['channel'] >= 0
        seeds = assigned_initial[seed_mask]

        if 'xc_err' in seeds.columns and 'yc_err' in seeds.columns:
            median_xc_err = seeds['xc_err'].median()
            median_yc_err = seeds['yc_err'].median()
            spatial_eps = np.mean([median_xc_err, median_yc_err])
        else:
            spatial_eps = 1.0  # Default fallback

        if verbose:
            print(f"Auto-calculated spatial_eps = {spatial_eps:.4f} pixels")
            print(f"  (Based on mean of median errors from {len(seeds):,} seed localizations)")
            print(f"  median(xc_err) = {median_xc_err:.4f}, median(yc_err) = {median_yc_err:.4f}")
            print()

    # Perform spatial clustering for each channel
    spatial_cluster_ids = np.full(len(assigned_initial), -1, dtype=int)
    puncta_per_channel = {}

    for k in range(n_channels):
        channel_k_mask = (assigned_initial['channel'] == k)
        channel_k_locs = assigned_initial[channel_k_mask]

        if len(channel_k_locs) < min_cluster_size:
            if verbose:
                print(f"Channel {k}: Too few locs ({len(channel_k_locs)}), skipping")
            puncta_per_channel[k] = 0
            continue

        # Extract spatial coordinates
        X_spatial = np.vstack([channel_k_locs['xc'], channel_k_locs['yc']]).T

        # Spatial clustering
        if spatial_method == 'DBSCAN':
            clusterer = DBSCAN(eps=spatial_eps, min_samples=min_cluster_size)
        elif spatial_method == 'HDBSCAN':
            clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                               cluster_selection_epsilon=spatial_eps)
        else:
            raise ValueError(f"Unknown spatial_method: {spatial_method}")

        cluster_labels = clusterer.fit_predict(X_spatial)

        # Filter: Keep only puncta with >= min_cluster_size localizations
        unique_cluster_ids = np.unique(cluster_labels[cluster_labels >= 0])
        valid_puncta = []

        for cluster_id in unique_cluster_ids:
            cluster_size = np.sum(cluster_labels == cluster_id)
            if cluster_size >= min_cluster_size:
                valid_puncta.append(cluster_id)

        # Map valid cluster labels to unique global IDs
        # Format: channel_k * 100000 + cluster_id
        channel_k_indices = np.where(channel_k_mask)[0]
        for i, cluster_label in enumerate(cluster_labels):
            if cluster_label in valid_puncta:
                global_cluster_id = k * 100000 + cluster_label
                spatial_cluster_ids[channel_k_indices[i]] = global_cluster_id

        n_puncta_k = len(valid_puncta)
        puncta_per_channel[k] = n_puncta_k

        if verbose:
            n_raw_clusters = len(unique_cluster_ids)
            n_filtered = n_raw_clusters - n_puncta_k
            n_in_valid = sum(1 for lbl in cluster_labels if lbl in valid_puncta)

            print(f"Channel {k}: {n_puncta_k} valid puncta (≥{min_cluster_size} locs each)")
            if n_filtered > 0:
                print(f"  Filtered out {n_filtered} small clusters")
            print(f"  In valid puncta: {n_in_valid:,} locs")
            print(f"  Noise/small clusters: {len(channel_k_locs) - n_in_valid:,} locs")
            print()

    return spatial_eps, puncta_per_channel, spatial_cluster_ids


def _calculate_posteriors(
    self,
    X: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    weights: np.ndarray,
    n_channels: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate posterior probabilities for all localizations.

    Returns:
        posterior_probs: (n_locs, n_channels) array of posterior probabilities
        most_likely_channel: (n_locs,) array of most likely channel indices
        confidence_per_loc: (n_locs,) array of confidence values
    """
    n_locs = len(X)
    log_probs = np.zeros((n_locs, n_channels))

    for k in range(n_channels):
        mvn = multivariate_normal(mean=means[k], cov=covariances[k])
        log_probs[:, k] = mvn.logpdf(X) + np.log(weights[k])

    # Normalize to get posterior probabilities (log-sum-exp trick)
    log_probs_max = log_probs.max(axis=1, keepdims=True)
    probs = np.exp(log_probs - log_probs_max)
    posterior_probs = probs / probs.sum(axis=1, keepdims=True)

    # Most likely channel and confidence
    most_likely_channel = np.argmax(posterior_probs, axis=1)
    confidence_per_loc = posterior_probs[np.arange(n_locs), most_likely_channel]

    return posterior_probs, most_likely_channel, confidence_per_loc


def _build_puncta_kdtrees(
    self,
    assigned_initial: pd.DataFrame,
    n_channels: int,
    verbose: bool
) -> Tuple[Dict[int, KDTree], Dict[int, Dict]]:
    """
    Build KDTree spatial indices for each channel's puncta.

    Returns:
        puncta_kdtrees: {channel_k: KDTree of punctum centers}
        puncta_members: {channel_k: {punctum_id: [loc_indices]}}
    """
    puncta_kdtrees = {}
    puncta_members = {}

    for k in range(n_channels):
        channel_k_clustered = (assigned_initial['channel'] == k) & \
                             (assigned_initial['spatial_cluster_id'] >= 0)

        if not channel_k_clustered.any():
            continue

        # Get unique puncta for this channel
        puncta_ids_k = assigned_initial.loc[channel_k_clustered, 'spatial_cluster_id'].unique()

        # Calculate punctum centers
        punctum_centers = []
        puncta_members[k] = {}

        for punctum_id in puncta_ids_k:
            punctum_mask = (assigned_initial['spatial_cluster_id'] == punctum_id)
            punctum_locs = assigned_initial[punctum_mask]

            # Center = mean position
            center_x = punctum_locs['xc'].mean()
            center_y = punctum_locs['yc'].mean()
            punctum_centers.append([center_x, center_y])

            # Store member indices
            puncta_members[k][punctum_id] = np.where(punctum_mask)[0].tolist()

        # Build KDTree
        if len(punctum_centers) > 0:
            puncta_kdtrees[k] = KDTree(np.array(punctum_centers))
            if verbose:
                print(f"Channel {k}: Built KDTree with {len(punctum_centers)} punctum centers")

    if verbose:
        print()

    return puncta_kdtrees, puncta_members


def _iterative_spatial_spectral_refinement(
    self,
    assigned_initial: pd.DataFrame,
    most_likely_channel: np.ndarray,
    confidence_per_loc: np.ndarray,
    puncta_kdtrees: Dict[int, KDTree],
    n_channels: int,
    spatial_eps: float,
    confidence_threshold_clear: float,
    confidence_threshold_overlap: float,
    max_iterations: int,
    min_new_assignments: int,
    verbose: bool
) -> Tuple[pd.DataFrame, list]:
    """
    Perform hierarchical iterative spatial-spectral refinement.

    This is the core of the algorithm that implements adaptive thresholding
    based on spatial context (clear vs overlap regions).

    Returns:
        assigned_current: DataFrame with refined assignments
        assignments_per_iteration: List of assignment counts per iteration
    """
    assigned_current = assigned_initial.copy()
    assignments_per_iteration = []
    iteration = 0

    # Initialize fields
    if 'nearest_punctum_distance' not in assigned_current.columns:
        assigned_current['nearest_punctum_distance'] = np.nan
    if 'is_spatial_overlap' not in assigned_current.columns:
        assigned_current['is_spatial_overlap'] = False

    while iteration < max_iterations:
        iteration += 1

        # Get currently unassigned localizations
        unassigned_mask = (assigned_current['channel'] == -1)
        n_unassigned = unassigned_mask.sum()

        if n_unassigned == 0:
            if verbose:
                print(f"Iteration {iteration}: No unassigned locs remaining, stopping.")
            break

        if verbose:
            print(f"Iteration {iteration}: Testing {n_unassigned:,} unassigned locs...")

        # For each unassigned loc, apply hierarchical spatial-spectral test
        unassigned_indices = np.where(unassigned_mask)[0]
        new_assignments = {}  # loc_index → (channel, distance, is_overlap)

        for idx in unassigned_indices:
            loc_row = assigned_current.iloc[idx]
            loc_coords = np.array([[loc_row['xc'], loc_row['yc']]])

            # Get this loc's spectral preferences
            loc_most_likely_channel = most_likely_channel[idx]
            loc_confidence = confidence_per_loc[idx]

            # STAGE 1: Identify which channels have nearby puncta
            nearby_channels = []  # List of (channel, distance) tuples

            for k in range(n_channels):
                if k not in puncta_kdtrees:
                    continue  # No puncta for this channel

                # Query nearest punctum from channel k
                distances, indices = puncta_kdtrees[k].query(loc_coords, k=1)
                nearest_distance_k = distances[0, 0]

                if nearest_distance_k <= spatial_eps:
                    nearby_channels.append((k, nearest_distance_k))

            if len(nearby_channels) == 0:
                # Not near any puncta → cannot assign
                continue

            # STAGE 2: Determine spatial context (clear vs overlap)
            is_overlap = len(nearby_channels) > 1

            # STAGE 3: Apply appropriate spectral threshold
            if is_overlap:
                # OVERLAP REGION: Multiple channels have nearby puncta
                # Require HIGH spectral confidence
                required_confidence = confidence_threshold_overlap
            else:
                # CLEAR REGION: Only one channel has nearby puncta
                # Allow MODERATE spectral confidence
                required_confidence = confidence_threshold_clear

            # STAGE 4: Check if loc meets threshold
            if loc_confidence < required_confidence:
                continue  # Spectral confidence too low for this spatial context

            # STAGE 5: Verify most likely channel is among nearby channels
            k = loc_most_likely_channel
            nearby_channel_ids = [ch for ch, _ in nearby_channels]

            if k not in nearby_channel_ids:
                # Spectral preference doesn't match any nearby punctum
                continue

            # STAGE 6: Assign to most likely channel
            # Find distance to that channel's nearest punctum
            distance_to_k = next(dist for ch, dist in nearby_channels if ch == k)
            new_assignments[idx] = (k, distance_to_k, is_overlap)

        # Apply new assignments
        n_new = len(new_assignments)
        n_new_clear = sum(1 for _, (_, _, is_ov) in new_assignments.items() if not is_ov)
        n_new_overlap = sum(1 for _, (_, _, is_ov) in new_assignments.items() if is_ov)
        assignments_per_iteration.append(n_new)

        for idx, (channel_k, distance, is_overlap) in new_assignments.items():
            assigned_current.loc[idx, 'channel'] = channel_k
            assigned_current.loc[idx, 'assignment_stage'] = f'refinement_iter_{iteration}'
            assigned_current.loc[idx, 'nearest_punctum_distance'] = distance
            assigned_current.loc[idx, 'is_spatial_overlap'] = is_overlap

        if verbose:
            print(f"  Assigned {n_new:,} locs ({n_new_clear:,} clear + {n_new_overlap:,} overlap)")
            for k in range(n_channels):
                n_k_new = sum(1 for _, (ch, _, _) in new_assignments.items() if ch == k)
                n_k_clear = sum(1 for _, (ch, _, is_ov) in new_assignments.items() if ch == k and not is_ov)
                n_k_overlap = sum(1 for _, (ch, _, is_ov) in new_assignments.items() if ch == k and is_ov)
                print(f"    Channel {k}: +{n_k_new:,} locs ({n_k_clear:,} clear, {n_k_overlap:,} overlap)")

        # Check convergence
        if n_new < min_new_assignments:
            if verbose:
                print(f"  Convergence: Fewer than {min_new_assignments} new assignments, stopping.")
            break

        print()

    if verbose:
        print("=" * 80)

    return assigned_current, assignments_per_iteration


def _plot_refinement_diagnostics(
    self,
    assigned_current: pd.DataFrame,
    metadata: Dict,
    n_channels: int,
    channels_to_use: list
) -> None:
    """
    Create diagnostic plots for spatial-spectral refinement.
    """
    import matplotlib.pyplot as plt

    n_features = len(channels_to_use)
    assignments_per_iteration = metadata['assignments_per_iteration']
    n_assigned_initial = metadata['n_assigned_initial']
    n_assigned_final = metadata['n_assigned_final']
    n_recovered = metadata['n_recovered']

    # Plot 1: Refinement Progress
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Assignments per iteration
    ax = axes[0]
    iterations = list(range(1, len(assignments_per_iteration) + 1))
    ax.bar(iterations, assignments_per_iteration)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('New Assignments', fontsize=12)
    ax.set_title('Spatial-Spectral Refinement Progress', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Initial vs Final assignments (per channel)
    ax = axes[1]
    x = np.arange(n_channels)
    width = 0.35
    ax.bar(x - width/2, [n_assigned_initial[k] for k in range(n_channels)],
           width, label='Initial', alpha=0.7)
    ax.bar(x + width/2, [n_assigned_final[k] for k in range(n_channels)],
           width, label='Final', alpha=0.7)
    ax.set_xlabel('Channel', fontsize=12)
    ax.set_ylabel('Number of Localizations', fontsize=12)
    ax.set_title('Initial vs Final Assignments', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Ch {k}' for k in range(n_channels)])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Recovery rate per channel
    ax = axes[2]
    recovery_rates = [100 * n_recovered[k] / n_assigned_initial[k]
                      if n_assigned_initial[k] > 0 else 0
                      for k in range(n_channels)]
    ax.bar(range(n_channels), recovery_rates)
    ax.set_xlabel('Channel', fontsize=12)
    ax.set_ylabel('Recovery Rate (%)', fontsize=12)
    ax.set_title('Localizations Recovered by Refinement', fontsize=14)
    ax.set_xticks(range(n_channels))
    ax.set_xticklabels([f'Ch {k}' for k in range(n_channels)])
    ax.grid(True, alpha=0.3)

    _safe_tight_layout(fig)
    plt.show()

    # Plot 2: Spatial Distribution by Assignment Stage (if 2D)
    if n_features == 2:
        fig, axes = plt.subplots(1, n_channels + 1, figsize=(6*(n_channels+1), 5))

        for k in range(n_channels):
            ax = axes[k]

            # Plot initial assignments
            mask_initial = (assigned_current['channel'] == k) & \
                          (assigned_current['assignment_stage'] == 'initial')
            if mask_initial.any():
                ax.scatter(assigned_current.loc[mask_initial, 'xc'],
                          assigned_current.loc[mask_initial, 'yc'],
                          s=1, alpha=0.5, color=f'C{k}', label='Initial')

            # Plot refined assignments
            mask_refined = (assigned_current['channel'] == k) & \
                          (assigned_current['assignment_stage'].str.startswith('refinement'))
            if mask_refined.any():
                ax.scatter(assigned_current.loc[mask_refined, 'xc'],
                          assigned_current.loc[mask_refined, 'yc'],
                          s=1, alpha=0.8, color='gold', marker='x', label='Recovered')

            ax.set_title(f'Channel {k}\n(+{n_recovered[k]:,} recovered)', fontsize=12)
            ax.set_xlabel('x (pixels)', fontsize=11)
            ax.set_ylabel('y (pixels)', fontsize=11)
            ax.legend(fontsize=10)
            ax.set_aspect('equal')

        # Combined plot
        ax = axes[n_channels]
        for k in range(n_channels):
            mask_k = (assigned_current['channel'] == k)
            ax.scatter(assigned_current.loc[mask_k, 'xc'],
                      assigned_current.loc[mask_k, 'yc'],
                      s=1, alpha=0.5, color=f'C{k}', label=f'Channel {k}')

        # Unassigned
        mask_unassigned = (assigned_current['channel'] == -1)
        if mask_unassigned.any():
            ax.scatter(assigned_current.loc[mask_unassigned, 'xc'],
                      assigned_current.loc[mask_unassigned, 'yc'],
                      s=1, alpha=0.3, color='black', label='Unassigned')

        ax.set_title('All Channels (Final)', fontsize=12)
        ax.set_xlabel('x (pixels)', fontsize=11)
        ax.set_ylabel('y (pixels)', fontsize=11)
        ax.legend(fontsize=10)
        ax.set_aspect('equal')

        _safe_tight_layout(fig)
        plt.show()

