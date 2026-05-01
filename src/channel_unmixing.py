# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import re
import warnings
from typing import Tuple, Dict, Optional
from scipy.stats import multivariate_normal
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KDTree
import logging
logger = logging.getLogger(__name__)


try:
    from fast_hdbscan import HDBSCAN
    HDBSCAN_BACKEND = "fast_hdbscan"
except ImportError:
    from sklearn.cluster import HDBSCAN
    HDBSCAN_BACKEND = "sklearn"


def _safe_tight_layout(fig):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                              message='The figure layout has changed to tight',
                              category=UserWarning)
        try:
            fig.tight_layout()
        except Exception:
            pass


class ChannelUnmixingMixin:
    """Mixin providing channel-unmixing methods for extract_SMs."""
    def unmix_channels(
        self,
        loc_data,
        n_channels,
        channels_to_use=["A_R", "A_G"],
        confidence_threshold=0.95,
        false_positive_rate=None,
        initial_guess_method="histogram_peaks",
        gmm_fit_method="EM",
        covariance_type="full",
        max_iter=500,
        outlier_rejection="mahalanobis",
        mestimator_type="tukey",
        initial_guess_percentile=50,
        initial_guess_scale=0.7,
        verbose=True,
        plot_results=False,
    ):
        """
        Separate SMLM localizations into N channels based on RGB amplitude ratios.

        This function uses Gaussian Mixture Model (GMM) fitting to separate multi-color
        SMLM data into distinct channels based on spectral signatures (A_R, A_G, A_B).
        Assignments are confidence-based with optional outlier rejection.

        Args:
            loc_data (pd.DataFrame): Localization data with columns:
                - A_R, A_G, A_B: Normalized RGB amplitudes
                - A_R_err, A_G_err, A_B_err: Fitting uncertainties (optional)
                - xc, yc: Localization coordinates
                - frame: Frame number

            n_channels (int): Number of distinct color channels (2-5 typical)

            channels_to_use (list): Which amplitude channels to use for separation
                - ['A_R', 'A_G']: 2D separation (typical for 2-3 color)
                - ['A_R', 'A_G', 'A_B']: 3D separation (for >3 colors)
                - ['A_R']: 1D separation (single ratio)
                Note: Corresponding error columns (e.g., A_R_err, A_G_err) are ALWAYS
                used for weighting if available. Higher errors = lower trust.

            confidence_threshold (float): Minimum posterior probability for assignment (0-1)
                Higher = more conservative (fewer assignments, higher purity)

            false_positive_rate (float, optional): Maximum acceptable FPR for assignment
                If specified, calculates confidence threshold from analytical overlap

            initial_guess_method (str): Method for initial channel centers
                - 'histogram_peaks': Find peaks in 1D histograms (default)
                - 'kmeans': K-means clustering

            gmm_fit_method (str): GMM fitting algorithm
                - 'EM': Expectation-Maximization (auto-selects best method):
                    * If error columns present → pygmmis Extreme Deconvolution (recommended)
                    * If no error columns → sklearn EM
                - 'EM_weighted': EM with photon-based weighting (legacy)
                - 'fixed': Use initial guess without EM refinement (most conservative)

            covariance_type (str): GMM covariance structure
                - 'full': Full covariance (allows correlation, default)
                - 'tied': Same covariance for all components
                - 'diag': Diagonal (no correlation)
                - 'spherical': Single variance per component

            max_iter (int): Maximum GMM fitting iterations

            outlier_rejection (str): Outlier handling method
                - 'none': No outlier rejection (default)
                - 'mahalanobis': Hard threshold on Mahalanobis distance

            mestimator_type (str): If outlier_rejection='mestimator'
                - 'huber': Moderate robustness
                - 'tukey': Aggressive robustness

            initial_guess_percentile (float): Percentile for core region selection (0-100)
                Lower = tighter initial guess. Default: 50 (median)
                Try 25-30 for very conservative separation

            initial_guess_scale (float): Scaling factor for initial covariances
                Lower = tighter ellipses. Default: 0.7
                Try 0.4-0.5 to prevent EM from over-expanding

            verbose (bool): Print progress and diagnostics

            plot_results (bool): Create diagnostic plots

        Returns:
            assigned_locs (pd.DataFrame): Input data with added columns:
                - 'channel': Assigned channel (0 to n_channels-1, or -1 for unassigned)
                - 'channel_confidence': Posterior probability for assigned channel
                - 'channel_probability_0', ...: Posterior for each channel
                - 'mahalanobis_distance': Distance to assigned channel mean
                - 'is_outlier': Boolean flag for outliers

            metadata (dict): Diagnostic information:
                - 'means': Fitted channel means
                - 'covariances': Fitted covariances
                - 'weights': Fitted channel weights
                - 'converged': Whether GMM converged
                - 'n_assigned': Number per channel
                - 'n_unassigned': Number rejected
                - 'confusion_matrix': Expected confusion matrix (if FPR specified)

        Example:
            >>> # 2-color separation (ATTO655 + Cy3B)
            >>> assigned, metadata = SM_E.unmix_channels(
            ...     loc_data,
            ...     n_channels=2,
            ...     channels_to_use=['A_R', 'A_G'],
            ...     confidence_threshold=0.95,
            ...     verbose=True,
            ...     plot_results=True
            ... )
            >>> print(f"Channel 0: {metadata['n_assigned'][0]} locs")
            >>> print(f"Channel 1: {metadata['n_assigned'][1]} locs")
        """
        if verbose:
            logger.info("=" * 70)
            logger.info("Channel Unmixing")
            logger.info("=" * 70)
            logger.info(f"Input: {len(loc_data)} localizations")
            logger.info(f"Channels: {n_channels}")
            logger.info(f"Features: {channels_to_use}")
            logger.info()

        # ===== Phase 1: Input Validation and Preprocessing =====
        # Check required columns
        for col in channels_to_use:
            if col not in loc_data.columns:
                raise ValueError(f"Column '{col}' not found in loc_data")

        # Check for error columns and auto-generate if needed
        # Make a copy of loc_data to avoid modifying the original
        loc_data_work = loc_data.copy()

        error_cols = [f"{col}_err" for col in channels_to_use]
        missing_errors = [col for col in error_cols if col not in loc_data_work.columns]

        if missing_errors:
            if verbose:
                logger.info(f"Missing error columns: {missing_errors}")
                logger.info("Attempting to auto-generate errors...")

            for error_col in missing_errors:
                # Extract base column name (remove '_err' suffix)
                base_col = error_col.replace('_err', '')

                if base_col == 'photons':
                    # Poisson statistics: σ = sqrt(N)
                    loc_data_work[error_col] = np.sqrt(np.maximum(loc_data_work[base_col].values, 1))
                    if verbose:
                        mean_err = loc_data_work[error_col].mean()
                        logger.info(f"  {error_col}: Generated from Poisson statistics (mean={mean_err:.2f})")
                elif base_col in loc_data_work.columns:
                    # For other columns, use a small fraction of the value as error estimate
                    # This is a conservative guess - user should provide real errors if possible
                    loc_data_work[error_col] = loc_data_work[base_col].values * 0.05  # 5% relative error
                    if verbose:
                        mean_err = loc_data_work[error_col].mean()
                        logger.info(f"  {error_col}: Estimated as 5% of {base_col} (mean={mean_err:.4f})")
                        logger.info(f"    WARNING: Using estimated errors. Provide measured errors for better results.")
                else:
                    raise ValueError(f"Cannot auto-generate error for '{error_col}': column '{base_col}' not found")

        # Extract feature matrix
        X = loc_data_work[channels_to_use].values
        n_features = X.shape[1]

        if verbose:
            logger.info(f"Feature matrix: {X.shape}")
            logger.info(f"Feature ranges:")
            for i, col in enumerate(channels_to_use):
                logger.info(f"  {col}: [{X[:, i].min():.3f}, {X[:, i].max():.3f}], mean={X[:, i].mean():.3f}")
            logger.info()

        # ===== Phase 2: Initial Guess for Channel Means =====
        if verbose:
            logger.info(f"Finding initial channel means (method: {initial_guess_method})...")

        if n_features == 1:
            # 1D case
            initial_means = self._find_histogram_peaks_1d(
                X[:, 0], n_channels
            ).reshape(-1, 1)
        elif n_features == 2:
            # 2D case
            initial_means = self._find_initial_means_2d(
                X, n_channels, method=initial_guess_method
            )
        else:
            # 3D or higher - use k-means
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_channels, n_init=10, random_state=42)
            kmeans.fit(X)
            initial_means = kmeans.cluster_centers_

        if verbose:
            logger.info(f"Initial means:")
            for k in range(n_channels):
                mean_str = ", ".join(
                    [f"{channels_to_use[i]}={initial_means[k, i]:.3f}" for i in range(n_features)]
                )
                logger.info(f"  Channel {k}: {mean_str}")
            logger.info()

        # ===== Phase 2.5: Estimate Initial Covariances =====
        # Two-stage initialization: means from histograms, covariances from data
        if initial_guess_method == "histogram_peaks":
            if verbose:
                logger.info("Estimating initial covariances from core regions (conservative)...")

            # Extract error matrix if available (now works for any n_features)
            error_cols = [f"{col}_err" for col in channels_to_use]
            if all(col in loc_data_work.columns for col in error_cols):
                X_err = loc_data_work[error_cols].values
            else:
                X_err = None

            initial_covariances = self._estimate_initial_covariances_2d(
                X, initial_means, n_channels, X_err=X_err,
                use_core_region=True, percentile=initial_guess_percentile,
                scale=initial_guess_scale
            )

            if verbose:
                for k in range(n_channels):
                    det_k = np.linalg.det(initial_covariances[k])
                    # Calculate standard deviations along principal axes
                    eigvals = np.linalg.eigvalsh(initial_covariances[k])
                    sigma_str = ", ".join([f"σ{i+1}={np.sqrt(eigvals[i]):.3f}" for i in range(n_features)])
                    logger.info(f"  Channel {k}: det(cov)={det_k:.6f}, {sigma_str}")
                logger.info()

            # Create diagnostic plot showing initial guess (only for 2D)
            if plot_results and n_features == 2:
                self._plot_initial_guess_2d(
                    X, channels_to_use, initial_means, initial_covariances, n_channels
                )
            elif plot_results and n_features > 2:
                if verbose:
                    logger.info(f"  Skipping initial guess plot (only available for 2D, current: {n_features}D)")
        else:
            initial_covariances = None

        # ===== Phase 3: GMM Fitting =====
        # Extract errors - ALWAYS use them if available (now includes auto-generated ones)
        error_cols = [f"{col}_err" for col in channels_to_use]
        has_errors = all(col in loc_data_work.columns for col in error_cols)

        if has_errors:
            X_err = loc_data_work[error_cols].values
        else:
            X_err = None

        # Intelligent method selection: Use pygmmis if errors available, sklearn otherwise
        if gmm_fit_method == "EM":
            if has_errors:
                # Use pygmmis Extreme Deconvolution (theoretically optimal for per-point errors)
                actual_method = "extreme_deconvolution"
                if verbose:
                    logger.info(f"Fitting GMM (method: EM → Extreme Deconvolution, covariance: {covariance_type})...")
                    logger.info(f"  Auto-selected pygmmis (error columns detected)")
                    logger.info(f"  Mean errors: {X_err.mean(axis=0)}")
            else:
                # Use sklearn EM (no errors available)
                actual_method = "sklearn_EM"
                if verbose:
                    logger.info(f"Fitting GMM (method: EM → sklearn, covariance: {covariance_type})...")
                    logger.info("  No error columns found, using sklearn EM without error weighting")
        else:
            actual_method = gmm_fit_method
            if verbose:
                logger.info(f"Fitting GMM (method: {gmm_fit_method}, covariance: {covariance_type})...")
                if has_errors:
                    logger.info(f"  Error columns available (mean errors: {X_err.mean(axis=0)})")
                else:
                    logger.info("  No error columns found")

        if actual_method == "sklearn_EM":
            # Use sklearn GMM without error weighting (pure EM)
            from sklearn.mixture import GaussianMixture

            # Prepare precisions_init if we have initial covariances
            if initial_covariances is not None and covariance_type == "full":
                # sklearn uses precisions (inverse covariances) for initialization
                precisions_init = np.zeros_like(initial_covariances)
                for k in range(n_channels):
                    try:
                        precisions_init[k] = np.linalg.inv(initial_covariances[k])
                    except np.linalg.LinAlgError:
                        # Singular, use regularized version
                        cov_reg = initial_covariances[k] + np.eye(n_features) * 1e-3
                        precisions_init[k] = np.linalg.inv(cov_reg)
            else:
                precisions_init = None

            gmm = GaussianMixture(
                n_components=n_channels,
                covariance_type=covariance_type,
                max_iter=max_iter,
                n_init=1,
                means_init=initial_means,
                precisions_init=precisions_init,
                random_state=42,
            )
            gmm.fit(X)  # No replication - pure sklearn EM
            means = gmm.means_
            covariances = gmm.covariances_
            weights = gmm.weights_
            converged = gmm.converged_

        elif gmm_fit_method == "EM_weighted":
            # Use existing weighted EM implementation
            photons = loc_data_work["photons"].values if "photons" in loc_data_work.columns else None
            A_R = loc_data_work["A_R"].values if "A_R" in loc_data_work.columns else None
            A_G = loc_data_work["A_G"].values if "A_G" in loc_data_work.columns else None

            # ALWAYS use error columns if available (match channels_to_use)
            # NOTE: EM_weighted only supports 2D (legacy method, use 'EM' for N-D support)
            if has_errors and len(channels_to_use) == 2:
                # Extract the specific error columns for the channels being used
                if channels_to_use[0] == 'A_R':
                    sigma_dim0 = loc_data_work["A_R_err"].values
                elif channels_to_use[0] == 'A_G':
                    sigma_dim0 = loc_data_work["A_G_err"].values
                elif channels_to_use[0] == 'A_B':
                    sigma_dim0 = loc_data_work["A_B_err"].values
                elif channels_to_use[0] == 'photons':
                    sigma_dim0 = loc_data_work["photons_err"].values
                else:
                    sigma_dim0 = loc_data_work[f"{channels_to_use[0]}_err"].values

                if channels_to_use[1] == 'A_R':
                    sigma_dim1 = loc_data_work["A_R_err"].values
                elif channels_to_use[1] == 'A_G':
                    sigma_dim1 = loc_data_work["A_G_err"].values
                elif channels_to_use[1] == 'A_B':
                    sigma_dim1 = loc_data_work["A_B_err"].values
                elif channels_to_use[1] == 'photons':
                    sigma_dim1 = loc_data_work["photons_err"].values
                else:
                    sigma_dim1 = loc_data_work[f"{channels_to_use[1]}_err"].values

                # For compatibility with _fit_gmm_em (expects A_R and A_G)
                sigma_A_R = sigma_dim0
                sigma_A_G = sigma_dim1
            else:
                if len(channels_to_use) != 2:
                    raise ValueError("EM_weighted method only supports 2D (len(channels_to_use)==2). Use gmm_fit_method='EM' for N-D support.")
                sigma_A_R = None
                sigma_A_G = None

            means, covariances, weights, converged = self._fit_gmm_em(
                X=X,
                initial_means=initial_means,
                n_components=n_channels,
                covariance_type=covariance_type,
                max_iter=max_iter,
                photons=photons,
                A_R=A_R,
                A_G=A_G,
                has_error_columns=has_errors,
                sigma_A_R=sigma_A_R,
                sigma_A_G=sigma_A_G,
                verbose=False,
            )
            gmm = None  # Not using sklearn GMM object

        elif actual_method == "extreme_deconvolution":
            # Use pygmmis Extreme Deconvolution for proper error handling
            if not has_errors:
                raise ValueError(
                    "Extreme Deconvolution requires error columns (A_R_err, A_G_err, etc.). "
                    "This should not happen when auto-selected by gmm_fit_method='EM'."
                )

            means, covariances, weights, converged = self._fit_gmm_pygmmis(
                X=X,
                X_err=X_err,
                initial_means=initial_means,
                n_components=n_channels,
                max_iter=max_iter,
                verbose=verbose,
            )
            gmm = None  # Not using sklearn GMM object

        elif gmm_fit_method == "fixed":
            # Use initial guess without EM refinement (most conservative)
            # This prevents EM from expanding the Gaussians
            if initial_covariances is None:
                raise ValueError(
                    "gmm_fit_method='fixed' requires initial_guess_method='histogram_peaks' "
                    "to compute initial covariances. Either change initial_guess_method or "
                    "use a different gmm_fit_method."
                )

            means = initial_means
            covariances = initial_covariances

            # Validate all covariances are positive definite
            for k in range(n_channels):
                eigvals = np.linalg.eigvalsh(covariances[k])
                if np.min(eigvals) <= 0:
                    if verbose:
                        logger.info(f"  Warning: Channel {k} covariance not positive definite (min eigenvalue={np.min(eigvals):.2e})")
                        logger.info(f"           Adding regularization...")
                    # Add regularization
                    diag_mean = np.mean(np.diag(covariances[k]))
                    reg_amount = np.abs(np.min(eigvals)) + np.maximum(1e-6, diag_mean * 1e-3)
                    covariances[k] += np.eye(n_features) * reg_amount

            # Calculate weights by hard assignment
            from scipy.spatial.distance import cdist
            distances = cdist(X, initial_means, metric='euclidean')
            assignments = np.argmin(distances, axis=1)
            weights = np.array([np.sum(assignments == k) / len(X) for k in range(n_channels)])

            converged = True  # No iteration needed
            gmm = None

            if verbose:
                logger.info("Using fixed initial guess (no EM refinement)")

        else:
            raise ValueError(f"Unknown gmm_fit_method: {gmm_fit_method}")

        if verbose:
            status = "converged" if converged else "did not converge"
            logger.info(f"GMM fitting: {status}")
            logger.info(f"Fitted means:")
            for k in range(n_channels):
                mean_str = ", ".join(
                    [f"{channels_to_use[i]}={means[k, i]:.3f}" for i in range(n_features)]
                )
                weight_pct = weights[k] * 100
                logger.info(f"  Channel {k}: {mean_str} (weight: {weight_pct:.1f}%)")
            logger.info()

        # ===== Phase 4: Channel Assignment with Confidence =====
        if verbose:
            logger.info("Calculating posterior probabilities and assignments...")

        # Calculate posterior probabilities
        n_locs = len(X)
        log_probs = np.zeros((n_locs, n_channels))

        for k in range(n_channels):
            mvn = multivariate_normal(mean=means[k], cov=covariances[k])
            log_probs[:, k] = mvn.logpdf(X) + np.log(weights[k])

        # Normalize to get posteriors (log-sum-exp trick)
        log_probs_max = log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs - log_probs_max)
        posterior_probs = probs / probs.sum(axis=1, keepdims=True)

        # Most likely channel
        channel_assignments = np.argmax(posterior_probs, axis=1)

        # Confidence = posterior of assigned channel
        confidence = posterior_probs[np.arange(n_locs), channel_assignments]

        # Calculate analytical confusion matrix if FPR specified
        if false_positive_rate is not None:
            if verbose:
                logger.info(f"Calculating analytical FPR to determine confidence threshold (target: {false_positive_rate:.3f})...")

            stats = self.calculate_analytical_misidentification(
                means, covariances, weights, n_samples=10000, random_state=42
            )

            if verbose:
                logger.info(f"Analytical accuracy: {stats['overall_accuracy']:.3f}")
                logger.info(f"Confusion matrix:")
                logger.info(stats["confusion_matrix"])
                logger.info()

            # Use simple threshold based on FPR
            # Higher FPR tolerance → lower threshold → more assignments
            confidence_threshold = 1.0 - (false_positive_rate / n_channels)

            if verbose:
                logger.info(f"Setting confidence threshold to {confidence_threshold:.3f} (from FPR={false_positive_rate:.3f})")

        # Apply confidence threshold
        is_assigned = confidence >= confidence_threshold
        channel_assignments_filtered = channel_assignments.copy()
        channel_assignments_filtered[~is_assigned] = -1

        # ===== Phase 5: Outlier Detection =====
        is_outlier = np.zeros(n_locs, dtype=bool)

        if outlier_rejection == "mahalanobis":
            if verbose:
                logger.info("Applying Mahalanobis distance outlier rejection...")

            from scipy.stats import chi2

            # Calculate Mahalanobis distance to assigned channel
            mahalanobis_distances = np.zeros(n_locs)

            for k in range(n_channels):
                channel_k_mask = channel_assignments_filtered == k
                if not np.any(channel_k_mask):
                    continue

                X_k = X[channel_k_mask]
                try:
                    inv_cov_k = np.linalg.inv(covariances[k])
                except np.linalg.LinAlgError:
                    # Singular covariance, regularize
                    cov_reg = covariances[k] + 1e-6 * np.eye(n_features)
                    inv_cov_k = np.linalg.inv(cov_reg)

                diff_k = X_k - means[k]
                mahal_k = np.sqrt(np.sum(diff_k @ inv_cov_k * diff_k, axis=1))
                mahalanobis_distances[channel_k_mask] = mahal_k

            # Flag outliers (99.9% quantile of chi-squared)
            outlier_threshold = np.sqrt(chi2.ppf(0.999, df=n_features))
            is_outlier = mahalanobis_distances > outlier_threshold
            channel_assignments_filtered[is_outlier] = -1

            if verbose:
                n_outliers = is_outlier.sum()
                logger.info(f"  Outliers detected: {n_outliers} ({100*n_outliers/n_locs:.2f}%, threshold={outlier_threshold:.2f})")

        else:
            # Calculate Mahalanobis distance anyway for diagnostics
            mahalanobis_distances = np.zeros(n_locs)
            for k in range(n_channels):
                channel_k_mask = channel_assignments_filtered == k
                if not np.any(channel_k_mask):
                    continue

                X_k = X[channel_k_mask]
                try:
                    inv_cov_k = np.linalg.inv(covariances[k])
                except np.linalg.LinAlgError:
                    cov_reg = covariances[k] + 1e-6 * np.eye(n_features)
                    inv_cov_k = np.linalg.inv(cov_reg)

                diff_k = X_k - means[k]
                mahal_k = np.sqrt(np.sum(diff_k @ inv_cov_k * diff_k, axis=1))
                mahalanobis_distances[channel_k_mask] = mahal_k

        # ===== Phase 6: Create Output DataFrame =====
        assigned_locs = loc_data.copy()
        assigned_locs["channel"] = channel_assignments_filtered
        assigned_locs["channel_confidence"] = confidence
        assigned_locs["mahalanobis_distance"] = mahalanobis_distances
        assigned_locs["is_outlier"] = is_outlier

        # Add per-channel posterior probabilities
        for k in range(n_channels):
            assigned_locs[f"channel_probability_{k}"] = posterior_probs[:, k]

        # Create metadata
        n_assigned_per_channel = {
            k: np.sum(channel_assignments_filtered == k) for k in range(n_channels)
        }
        n_unassigned = np.sum(channel_assignments_filtered == -1)

        metadata = {
            "means": means,
            "covariances": covariances,
            "weights": weights,
            "converged": converged,
            "n_assigned": n_assigned_per_channel,
            "n_unassigned": n_unassigned,
            "initial_means": initial_means,
            "channels_used": channels_to_use,
            "confidence_threshold": confidence_threshold,
        }

        # Add confusion matrix if calculated
        if false_positive_rate is not None:
            metadata["confusion_matrix"] = stats["confusion_matrix"]
            metadata["assignment_purity"] = np.diag(stats["confusion_matrix"])
            metadata["false_positive_rates"] = 1.0 - np.diag(stats["confusion_matrix"])

        if verbose:
            logger.info("\n" + "=" * 70)
            logger.info("Unmixing Complete")
            logger.info("=" * 70)
            logger.info(f"Assignments:")
            for k in range(n_channels):
                n_k = n_assigned_per_channel[k]
                pct_k = 100 * n_k / n_locs
                logger.info(f"  Channel {k}: {n_k:,} ({pct_k:.1f}%)")
            pct_unassigned = 100 * n_unassigned / n_locs
            logger.info(f"  Unassigned: {n_unassigned:,} ({pct_unassigned:.1f}%)")
            logger.info()

        # ===== Phase 7: Diagnostic Plotting =====
        if plot_results:
            self._plot_unmixing_results(
                X, channels_to_use, channel_assignments_filtered, confidence,
                means, covariances, weights, n_channels, confidence_threshold,
                metadata
            )

        return assigned_locs, metadata

    def _plot_initial_guess_2d(self, X, channels_to_use, initial_means,
                                initial_covariances, n_channels, display: bool = True):
        """
        Plot 2D histogram with initial guess overlaid (means and 2σ ellipses).

        Args:
            X (np.ndarray): Data matrix, shape (n_samples, 2)
            channels_to_use (list): Channel names
            initial_means (np.ndarray): Initial means, shape (n_channels, 2)
            initial_covariances (np.ndarray): Initial covariances, shape (n_channels, 2, 2)
            n_channels (int): Number of channels
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from PlottingBase import PublicationPlotter

        # Use colors that don't conflict with channel names (avoid red/green for R/G channels)
        colors_ch = ['blue', 'orange', 'purple', 'cyan', 'magenta', 'brown'][:n_channels]

        plotter = PublicationPlotter()
        fig, ax = plotter.two_column_plot(nrows=1, ncols=1)  # Use default height (3 inches)

        # 2D histogram
        hist_2d, xedges, yedges = np.histogram2d(X[:, 0], X[:, 1], bins=100)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        im = ax.imshow(
            hist_2d.T, origin='lower', extent=extent,
            cmap='gray', aspect='auto', interpolation='nearest'
        )
        plt.colorbar(im, ax=ax, label='Count')

        # Overlay initial means and 2σ ellipses
        for k in range(n_channels):
            # Plot mean
            ax.scatter(
                initial_means[k, 0], initial_means[k, 1],
                s=200, marker='x', color=colors_ch[k],
                linewidths=4, label=f'Channel {k}', zorder=10
            )

            # Plot 2σ confidence ellipse
            eigvals, eigvecs = np.linalg.eigh(initial_covariances[k])
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            width, height = 2 * 2 * np.sqrt(eigvals)  # 2σ

            ellipse = Ellipse(
                initial_means[k], width, height, angle=angle,
                edgecolor=colors_ch[k], facecolor='none', linewidth=3, zorder=9
            )
            ax.add_patch(ellipse)

        ax.set_xlabel(channels_to_use[0])
        ax.set_ylabel(channels_to_use[1])
        ax.set_title('Initial Guess: 2D Histogram + Means + 2σ Ellipses', weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Don't use tight_layout() - it conflicts with colorbar layout engine
        fig.subplots_adjust(right=0.85)  # Make room for colorbar
        if display:
            plt.show()

    def _plot_unmixing_results(
        self, X, channels_to_use, assignments, confidence,
        means, covariances, weights, n_channels, confidence_threshold,
        metadata, display: bool = True
    ):
        """Create diagnostic plots for channel unmixing results."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from scipy.stats import norm
        from PlottingBase import PublicationPlotter

        n_features = X.shape[1]

        # Plot 1: 1D Histograms with GMM overlay
        plotter = PublicationPlotter()
        # Use 2.5 inches per panel (reasonable for stacked histograms)
        fig, axes = plotter.one_column_plot(npanels=n_features, height=2.5 * n_features)
        if n_features == 1:
            axes = [axes]

        # Use colors that don't conflict with channel names (avoid red/green for R/G channels)
        colors = ['blue', 'orange', 'purple', 'cyan', 'magenta', 'brown'][:n_channels]

        for i, channel_name in enumerate(channels_to_use):
            ax = axes[i]

            # Histogram of all data
            ax.hist(X[:, i], bins=500, alpha=0.3, color='gray', label='All data', density=True)

            # Histograms per assigned channel
            for k in range(n_channels):
                mask = assignments == k
                if mask.sum() > 0:
                    ax.hist(
                        X[mask, i], bins=200, alpha=0.5,
                        color=colors[k], label=f'Channel {k}', density=True
                    )

            # GMM components (marginal distributions)
            x_range = np.linspace(X[:, i].min(), X[:, i].max(), 1000)
            gmm_pdf = np.zeros_like(x_range)

            for k in range(n_channels):
                # Project to 1D (marginal)
                if n_features == 1:
                    mean_1d = means[k, 0]
                    var_1d = covariances[k, 0, 0]
                else:
                    mean_1d = means[k, i]
                    var_1d = covariances[k, i, i]

                pdf_k = weights[k] * norm.pdf(x_range, mean_1d, np.sqrt(var_1d))
                ax.plot(x_range, pdf_k, color=colors[k], linewidth=2, linestyle='--',
                       label=f'GMM Ch{k}')
                gmm_pdf += pdf_k

            ax.plot(x_range, gmm_pdf, 'k-', linewidth=2, label='GMM total')

            ax.set_xlabel(channel_name)
            ax.set_ylabel('Density')
            ax.set_title(f'{channel_name} Distribution with GMM Fit')
            ax.legend()
            ax.grid(True, alpha=0.3)

        _safe_tight_layout(fig)  # Use safe wrapper to avoid layout warnings
        if display:
            plt.show()

        # Plot 2: 2D Scatter (if 2D data)
        if n_features == 2:
            fig, axes = plotter.two_column_plot(nrows=1, ncols=2)  # Use default height (3 inches)

            # Left: GMM ellipses
            ax = axes[0]
            ax.scatter(X[:, 0], X[:, 1], s=1, alpha=0.2, c='gray', rasterized=True)

            for k in range(n_channels):
                # 2σ confidence ellipse
                eigvals, eigvecs = np.linalg.eigh(covariances[k])
                angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                width, height = 2 * 2 * np.sqrt(eigvals)

                ellipse = Ellipse(
                    means[k], width, height, angle=angle,
                    edgecolor=colors[k], facecolor='none', linewidth=3
                )
                ax.add_patch(ellipse)
                ax.scatter(
                    means[k, 0], means[k, 1], s=200, marker='x',
                    color=colors[k], linewidths=4, label=f'Channel {k}'
                )

            ax.set_xlabel(channels_to_use[0])
            ax.set_ylabel(channels_to_use[1])
            ax.set_title('GMM Fit (2σ ellipses)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Right: Assignments
            ax = axes[1]
            for k in range(n_channels):
                mask = assignments == k
                if mask.sum() > 0:
                    ax.scatter(
                        X[mask, 0], X[mask, 1], s=1, alpha=0.5,
                        color=colors[k], label=f'Ch {k} (n={mask.sum():,})',
                        rasterized=True
                    )

            # Unassigned
            unassigned_mask = assignments == -1
            if unassigned_mask.sum() > 0:
                ax.scatter(
                    X[unassigned_mask, 0], X[unassigned_mask, 1],
                    s=1, alpha=0.3, color='black',
                    label=f'Unassigned (n={unassigned_mask.sum():,})',
                    rasterized=True
                )

            ax.set_xlabel(channels_to_use[0])
            ax.set_ylabel(channels_to_use[1])
            ax.set_title(f'Assignments (threshold={confidence_threshold:.2f})')
            ax.legend()
            ax.grid(True, alpha=0.3)

            _safe_tight_layout(fig)  # Use safe wrapper to avoid layout warnings
            if display:
                plt.show()

        # Plot 3: Confidence histogram
        fig, ax = plotter.one_column_plot(npanels=1)  # Use default height (3 inches)

        for k in range(n_channels):
            mask = (assignments == k)
            if mask.sum() > 0:
                ax.hist(
                    confidence[mask], bins=100, alpha=0.6,
                    color=colors[k], label=f'Channel {k}'
                )

        ax.axvline(
            confidence_threshold, color='red', linestyle='--',
            linewidth=3, label=f'Threshold ({confidence_threshold:.2f})'
        )
        ax.set_xlabel('Assignment Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Assignment Confidences')
        ax.legend()
        ax.grid(True, alpha=0.3)
        _safe_tight_layout(fig)  # Use safe wrapper to avoid layout warnings
        if display:
            plt.show()

        # Plot 4: Confusion matrix (if available)
        if 'confusion_matrix' in metadata:
            fig, ax = plotter.one_column_plot(npanels=1, height=3.5)
            conf_mat = metadata['confusion_matrix']

            im = ax.imshow(conf_mat, cmap='Blues', vmin=0, vmax=1)

            # Annotate cells
            for i in range(n_channels):
                for j in range(n_channels):
                    text_color = 'white' if conf_mat[i, j] > 0.5 else 'black'
                    ax.text(
                        j, i, f"{conf_mat[i, j]:.3f}",
                        ha="center", va="center",
                        color=text_color, weight='bold'
                    )

            ax.set_xticks(np.arange(n_channels))
            ax.set_yticks(np.arange(n_channels))
            ax.set_xticklabels([f'Ch {k}' for k in range(n_channels)])
            ax.set_yticklabels([f'Ch {k}' for k in range(n_channels)])
            ax.set_xlabel('Predicted Channel')
            ax.set_ylabel('True Channel')
            ax.set_title('Expected Confusion Matrix (Analytical)')
            plt.colorbar(im, ax=ax, label='Probability')
            # Don't use tight_layout() - it conflicts with colorbar layout engine
            fig.subplots_adjust(right=0.85)  # Make room for colorbar
            if display:
                plt.show()

    # ========================================================================
    # Hierarchical Spatial-Spectral Refinement
    # ========================================================================
    # Added: 2025-11-14
    # Methods for iterative spatial-spectral channel unmixing with adaptive
    # thresholds based on spatial context (clear vs overlap regions)
    # ========================================================================

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

            spatial_eps: Scaling factor for spatial clustering epsilon (default: 1.0)
                         Actual epsilon = spatial_eps × mean([median(xc_err), median(yc_err)])
                         - spatial_eps=1.0: Use base scale (recommended)
                         - spatial_eps=2.0: Look for larger clusters (2× base scale)
                         - spatial_eps=0.5: Require tighter clustering (0.5× base scale)
                         If None, defaults to 1.0
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
            logger.info("=" * 80)
            logger.info("Hierarchical Spatial-Spectral Channel Unmixing")
            logger.info("=" * 80)
            logger.info(f"Input: {len(loc_data):,} localizations")
            logger.info(f"Channels: {n_channels}")
            logger.info(f"Features: {channels_to_use}")
            logger.info()

        # ===== STEP 1: Initial Conservative Spectral Unmixing =====
        if verbose:
            logger.info("=" * 80)
            logger.info("STEP 1: Initial Spectral Unmixing (Conservative Seeds)")
            logger.info("=" * 80)

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
        # assignment_stage: 0=unassigned, 1=initial, 2+=refinement_iteration
        assigned_initial['assignment_stage'] = 0
        assigned_initial.loc[assigned_initial['channel'] >= 0, 'assignment_stage'] = 1

        n_assigned_initial = {k: (assigned_initial['channel'] == k).sum()
                              for k in range(n_channels)}
        n_unassigned_initial = (assigned_initial['channel'] == -1).sum()

        if verbose:
            logger.info(f"\nInitial assignments (confidence ≥ {confidence_threshold_initial}):")
            for k in range(n_channels):
                logger.info(f"  Channel {k}: {n_assigned_initial[k]:,} locs")
            logger.info(f"  Unassigned: {n_unassigned_initial:,} locs")
            logger.info()

        # ===== STEP 2: Spatial Clustering Per Channel =====
        if verbose:
            logger.info("=" * 80)
            logger.info("STEP 2: Spatial Clustering of Seeds (per channel)")
            logger.info("=" * 80)

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

        # ===== STEP 2.5: Refine Spectral Model from Puncta (OPTIONAL ENHANCEMENT) =====
        n_puncta_total = sum(puncta_per_channel.values())

        if n_puncta_total >= n_channels:
            if verbose:
                logger.info("=" * 80)
                logger.info("STEP 2.5: Refine Spectral Model from Puncta")
                logger.info("=" * 80)

            # Refine spectral model using puncta-based statistics
            means, covariances, weights = self._refine_spectral_model_from_puncta(
                assigned_initial,
                channels_to_use,
                means,  # original means as fallback
                covariances,  # original covs as fallback
                weights,  # original weights as fallback
                n_channels,
                min_locs_per_channel=30,
                verbose=verbose
            )

            if verbose:
                logger.info()
        else:
            if verbose:
                logger.info("=" * 80)
                logger.info(f"STEP 2.5: Skipping spectral refinement (only {n_puncta_total} puncta)")
                logger.info("=" * 80)
                logger.info()

        # ===== STEP 3: Hierarchical Iterative Refinement =====
        if verbose:
            logger.info("=" * 80)
            logger.info("STEP 3: Hierarchical Spatial-Spectral Refinement")
            logger.info("=" * 80)
            logger.info(f"Spectral thresholds:")
            logger.info(f"  Clear regions (1 channel nearby):    {confidence_threshold_clear:.2f}")
            logger.info(f"  Overlap regions (2+ channels nearby): {confidence_threshold_overlap:.2f}")
            logger.info(f"(Higher threshold in overlap regions accounts for spatial ambiguity)")
            logger.info()

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
            logger.info("=" * 80)
            logger.info("Refinement Complete")
            logger.info("=" * 80)
            logger.info(f"\nFinal assignments:")
            for k in range(n_channels):
                logger.info(f"  Channel {k}: {n_assigned_final[k]:,} locs (+{n_recovered[k]:,} from refinement)")
            logger.info(f"  Unassigned: {n_unassigned_final:,} locs")
            logger.info(f"\nTotal recovered: {n_recovered_total:,} locs ({100*n_recovered_total/len(loc_data):.2f}%)")
            logger.info()

        # ===== STEP 5: Diagnostic Plotting =====
        if plot_results:
            self.plot_refinement_diagnostics(
                assigned_current,
                metadata_final,
                n_channels,
                channels_to_use
            )

        return assigned_current, metadata_final


    # Helper methods for spatial-spectral refinement
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
        # Calculate base spatial scale from conservatively-assigned seeds
        seed_mask = assigned_initial['channel'] >= 0
        seeds = assigned_initial[seed_mask]

        if 'xc_err' in seeds.columns and 'yc_err' in seeds.columns:
            median_xc_err = seeds['xc_err'].median()
            median_yc_err = seeds['yc_err'].median()
            base_scale = np.mean([median_xc_err, median_yc_err])
        else:
            median_xc_err = 1.0
            median_yc_err = 1.0
            base_scale = 1.0  # Default fallback

        # Apply scaling factor (spatial_eps as multiplier)
        if spatial_eps is None:
            spatial_eps = 1.0  # Default: 1x base scale

        epsilon_pixels = spatial_eps * base_scale

        if verbose:
            logger.info(f"Spatial clustering scale:")
            logger.info(f"  Base scale (mean of median errors) = {base_scale:.4f} pixels")
            logger.info(f"    median(xc_err) = {median_xc_err:.4f}, median(yc_err) = {median_yc_err:.4f}")
            logger.info(f"  spatial_eps multiplier = {spatial_eps:.2f}")
            logger.info(f"  Effective epsilon = {epsilon_pixels:.4f} pixels")
            logger.info()

        # Perform spatial clustering for each channel
        spatial_cluster_ids = np.full(len(assigned_initial), -1, dtype=int)
        puncta_per_channel = {}

        for k in range(n_channels):
            channel_k_mask = (assigned_initial['channel'] == k)
            channel_k_locs = assigned_initial[channel_k_mask]

            if len(channel_k_locs) < min_cluster_size:
                if verbose:
                    logger.info(f"Channel {k}: Too few locs ({len(channel_k_locs)}), skipping")
                puncta_per_channel[k] = 0
                continue

            # Extract spatial coordinates
            X_spatial = np.vstack([channel_k_locs['xc'], channel_k_locs['yc']]).T

            # Spatial clustering
            if spatial_method == 'DBSCAN':
                clusterer = DBSCAN(eps=epsilon_pixels, min_samples=min_cluster_size)
                method_info = "DBSCAN"
            elif spatial_method == 'HDBSCAN':
                clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                                   cluster_selection_epsilon=epsilon_pixels)
                method_info = f"HDBSCAN ({HDBSCAN_BACKEND})"
            else:
                raise ValueError(f"Unknown spatial_method: {spatial_method}")

            if verbose and k == 0:  # Only print once
                logger.info(f"  Clustering method: {method_info}")

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

                logger.info(f"Channel {k}: {n_puncta_k} valid puncta (≥{min_cluster_size} locs each)")
                if n_filtered > 0:
                    logger.info(f"  Filtered out {n_filtered} small clusters")
                logger.info(f"  In valid puncta: {n_in_valid:,} locs")
                logger.info(f"  Noise/small clusters: {len(channel_k_locs) - n_in_valid:,} locs")
                logger.info()

        return epsilon_pixels, puncta_per_channel, spatial_cluster_ids

    def _refine_spectral_model_from_puncta(
        self,
        assigned_current: pd.DataFrame,
        channels_to_use: list,
        original_means: np.ndarray,
        original_covs: np.ndarray,
        original_weights: np.ndarray,
        n_channels: int,
        min_locs_per_channel: int = 30,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Refine spectral means/covariances by averaging over spatially-clustered puncta.

        This method computes empirical statistics from locs in valid puncta (spatial_cluster_id >= 0),
        which are likely to be "pure" single-color signals. This produces sharper channel means
        and can improve assignment accuracy in subsequent iterations.

        Parameters
        ----------
        assigned_current : pd.DataFrame
            Current assignments with 'channel' and 'spatial_cluster_id' columns
        channels_to_use : list
            Spectral feature names (e.g., ['A_R', 'A_G'])
        original_means : np.ndarray
            Original GMM means (fallback if insufficient data)
        original_covs : np.ndarray
            Original GMM covariances (fallback)
        original_weights : np.ndarray
            Original GMM weights (fallback)
        n_channels : int
            Number of spectral channels
        min_locs_per_channel : int
            Minimum locs required per channel for stable statistics (default: 30)
        verbose : bool
            Print diagnostics

        Returns
        -------
        means : np.ndarray (n_channels, n_features)
            Refined channel means
        covariances : np.ndarray (n_channels, n_features, n_features)
            Refined covariances
        weights : np.ndarray (n_channels,)
            Refined channel weights
        """
        n_features = len(channels_to_use)
        refined_means = original_means.copy()
        refined_covs = original_covs.copy()
        refined_weights = np.zeros(n_channels)

        for k in range(n_channels):
            # Get all locs in valid puncta for this channel
            mask = (assigned_current['channel'] == k) & \
                   (assigned_current['spatial_cluster_id'] >= 0)

            n_locs_k = mask.sum()

            if n_locs_k < min_locs_per_channel:
                if verbose:
                    logger.info(f"  Channel {k}: Only {n_locs_k} locs in puncta (< {min_locs_per_channel}), keeping original GMM parameters")
                # Keep original GMM parameters (safety)
                refined_weights[k] = original_weights[k]
                continue

            # Extract spectral features
            X_k = assigned_current.loc[mask, channels_to_use].values

            # Compute refined statistics
            refined_means[k] = X_k.mean(axis=0)

            # Compute empirical covariance with regularization
            cov_k = np.cov(X_k, rowvar=False)

            # Ensure positive definiteness by adding small diagonal regularization
            # This prevents numerical issues with near-singular matrices
            eps = 1e-6
            cov_k += eps * np.eye(cov_k.shape[0])

            refined_covs[k] = cov_k
            refined_weights[k] = n_locs_k

            if verbose:
                logger.info(f"  Channel {k}: Refined from {n_locs_k:,} locs in puncta")
                logger.info(f"    Original mean: {original_means[k]}")
                logger.info(f"    Refined mean:  {refined_means[k]}")

        # Normalize weights
        if refined_weights.sum() > 0:
            refined_weights /= refined_weights.sum()
        else:
            refined_weights = original_weights.copy()

        return refined_means, refined_covs, refined_weights

    def _calculate_posteriors(
        self,
        X: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        weights: np.ndarray,
        n_channels: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate posterior probabilities for all localizations."""
        n_locs = len(X)
        log_probs = np.zeros((n_locs, n_channels))

        for k in range(n_channels):
            mvn = multivariate_normal(mean=means[k], cov=covariances[k])
            log_probs[:, k] = mvn.logpdf(X) + np.log(weights[k])

        # Normalize (log-sum-exp trick)
        log_probs_max = log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs - log_probs_max)
        posterior_probs = probs / probs.sum(axis=1, keepdims=True)

        most_likely_channel = np.argmax(posterior_probs, axis=1)
        confidence_per_loc = posterior_probs[np.arange(n_locs), most_likely_channel]

        return posterior_probs, most_likely_channel, confidence_per_loc

    def _build_puncta_kdtrees(
        self,
        assigned_initial: pd.DataFrame,
        n_channels: int,
        verbose: bool
    ) -> Tuple[Dict[int, KDTree], Dict[int, Dict]]:
        """Build KDTree spatial indices for each channel's puncta."""
        puncta_kdtrees = {}
        puncta_members = {}

        for k in range(n_channels):
            channel_k_clustered = (assigned_initial['channel'] == k) & \
                                 (assigned_initial['spatial_cluster_id'] >= 0)

            if not channel_k_clustered.any():
                continue

            puncta_ids_k = assigned_initial.loc[channel_k_clustered, 'spatial_cluster_id'].unique()

            punctum_centers = []
            puncta_members[k] = {}

            for punctum_id in puncta_ids_k:
                punctum_mask = (assigned_initial['spatial_cluster_id'] == punctum_id)
                punctum_locs = assigned_initial[punctum_mask]

                center_x = punctum_locs['xc'].mean()
                center_y = punctum_locs['yc'].mean()
                punctum_centers.append([center_x, center_y])

                puncta_members[k][punctum_id] = np.where(punctum_mask)[0].tolist()

            if len(punctum_centers) > 0:
                puncta_kdtrees[k] = KDTree(np.array(punctum_centers))
                if verbose:
                    logger.info(f"Channel {k}: Built KDTree with {len(punctum_centers)} punctum centers")

        if verbose:
            logger.info()

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
                    logger.info(f"Iteration {iteration}: No unassigned locs remaining, stopping.")
                break

            if verbose:
                logger.info(f"Iteration {iteration}: Testing {n_unassigned:,} unassigned locs...")

            # VECTORIZED APPROACH: Query all unassigned locs at once per channel
            # Performance: Instead of n_locs × n_channels individual queries,
            # we do n_channels vectorized queries (10-100× faster)
            unassigned_indices = np.where(unassigned_mask)[0]
            new_assignments = {}  # loc_index → (channel, distance, is_overlap)

            # Get coordinates of all unassigned locs
            unassigned_coords = assigned_current.loc[unassigned_mask, ['xc', 'yc']].values
            n_unassigned_locs = len(unassigned_indices)

            # Get spectral preferences for unassigned locs
            unassigned_most_likely = most_likely_channel[unassigned_indices]
            unassigned_confidence = confidence_per_loc[unassigned_indices]

            # STAGE 1: Vectorized KDTree queries for all channels
            # For each channel, find distance to nearest punctum for ALL unassigned locs
            nearest_distances = np.full((n_unassigned_locs, n_channels), np.inf)

            for k in range(n_channels):
                if k not in puncta_kdtrees:
                    continue  # No puncta for this channel, leave as inf

                # Query ALL unassigned locs at once (VECTORIZED)
                distances, _ = puncta_kdtrees[k].query(unassigned_coords, k=1)
                nearest_distances[:, k] = distances.ravel()

            # STAGE 2: Determine spatial context for each loc
            # Count how many channels have puncta within spatial_eps
            nearby_mask = (nearest_distances <= spatial_eps)  # (n_locs, n_channels)
            n_nearby_channels = nearby_mask.sum(axis=1)  # (n_locs,)

            # Identify overlap regions: >1 nearby channel
            is_overlap_per_loc = (n_nearby_channels > 1)
            is_clear_per_loc = (n_nearby_channels == 1)

            # STAGE 3: Apply adaptive spectral thresholds
            # Determine required confidence based on spatial context
            required_confidence = np.where(
                is_overlap_per_loc,
                confidence_threshold_overlap,  # Overlap: high threshold
                np.where(
                    is_clear_per_loc,
                    confidence_threshold_clear,  # Clear: moderate threshold
                    np.inf  # No nearby puncta: impossible threshold
                )
            )

            # STAGE 4: Check spectral confidence meets threshold
            passes_spectral_threshold = (unassigned_confidence >= required_confidence)

            # STAGE 5: Verify most likely channel has nearby punctum
            # For each loc, check if its spectral preference is among nearby channels
            loc_channel_nearby = nearby_mask[np.arange(n_unassigned_locs), unassigned_most_likely]

            # STAGE 6: Combine all criteria
            # Can assign if: passes spectral threshold AND preferred channel is nearby
            can_assign = passes_spectral_threshold & loc_channel_nearby

            # Build assignments for locs that pass all criteria
            for i, global_idx in enumerate(unassigned_indices):
                if not can_assign[i]:
                    continue

                k = unassigned_most_likely[i]
                distance_to_k = nearest_distances[i, k]
                is_overlap = is_overlap_per_loc[i]

                new_assignments[global_idx] = (k, distance_to_k, is_overlap)

            # Apply new assignments
            n_new = len(new_assignments)
            n_new_clear = sum(1 for _, (_, _, is_ov) in new_assignments.items() if not is_ov)
            n_new_overlap = sum(1 for _, (_, _, is_ov) in new_assignments.items() if is_ov)
            assignments_per_iteration.append(n_new)

            for idx, (channel_k, distance, is_overlap) in new_assignments.items():
                assigned_current.loc[idx, 'channel'] = channel_k
                assigned_current.loc[idx, 'assignment_stage'] = 1 + iteration  # 2, 3, 4, ... for iterations 1, 2, 3, ...
                assigned_current.loc[idx, 'nearest_punctum_distance'] = distance
                assigned_current.loc[idx, 'is_spatial_overlap'] = is_overlap

            if verbose:
                logger.info(f"  Assigned {n_new:,} locs ({n_new_clear:,} clear + {n_new_overlap:,} overlap)")
                for k in range(n_channels):
                    n_k_new = sum(1 for _, (ch, _, _) in new_assignments.items() if ch == k)
                    n_k_clear = sum(1 for _, (ch, _, is_ov) in new_assignments.items() if ch == k and not is_ov)
                    n_k_overlap = sum(1 for _, (ch, _, is_ov) in new_assignments.items() if ch == k and is_ov)
                    logger.info(f"    Channel {k}: +{n_k_new:,} locs ({n_k_clear:,} clear, {n_k_overlap:,} overlap)")

            # Check convergence
            if n_new < min_new_assignments:
                if verbose:
                    logger.info(f"  Convergence: Fewer than {min_new_assignments} new assignments, stopping.")
                break

            logger.info()

        if verbose:
            logger.info("=" * 80)

        return assigned_current, assignments_per_iteration


    def plot_refinement_diagnostics(
        self,
        assigned_current: pd.DataFrame,
        metadata: Dict,
        n_channels: int,
        channels_to_use: list,
        save_path: Optional[str] = None,
        display: bool = True,
    ) -> None:
        """
        Create diagnostic plots for spatial-spectral refinement.

        Parameters
        ----------
        assigned_current : pd.DataFrame
            DataFrame with final channel assignments
        metadata : Dict
            Dictionary containing refinement statistics:
                - assignments_per_iteration: List[int]
                - n_assigned_initial: Dict[int, int]
                - n_assigned_final: Dict[int, int]
                - n_recovered: Dict[int, int]
        n_channels : int
            Number of channels
        channels_to_use : list
            List of channel names used for unmixing
        save_path : Optional[str]
            If provided, save figure to this path
        """
        import matplotlib.pyplot as plt
        from PlottingBase import PublicationPlotter

        # Extract metadata
        assignments_per_iteration = metadata['assignments_per_iteration']
        n_assigned_initial = metadata['n_assigned_initial']
        n_assigned_final = metadata['n_assigned_final']
        n_recovered = metadata['n_recovered']

        # Create 3-panel summary figure
        plotter = PublicationPlotter()
        fig, axes = plotter.two_column_plot(nrows=1, ncols=3, height=4)
        
        # Panel 1: Assignments per iteration
        ax = axes[0]
        iterations = list(range(1, len(assignments_per_iteration) + 1))
        ax.bar(iterations, assignments_per_iteration, color='skyblue', alpha=0.7)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('New Assignments', fontsize=12)
        ax.set_title('Spatial-Spectral Refinement Progress', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Initial vs Final assignments per channel
        ax = axes[1]
        x = np.arange(n_channels)
        width = 0.35
        initial_counts = [n_assigned_initial[k] for k in range(n_channels)]
        final_counts = [n_assigned_final[k] for k in range(n_channels)]
        
        ax.bar(x - width/2, initial_counts, width, label='Initial',
               color='steelblue', alpha=0.7)
        ax.bar(x + width/2, final_counts, width, label='Final',
               color='coral', alpha=0.7)
        ax.set_xlabel('Channel', fontsize=12)
        ax.set_ylabel('Number of Localizations', fontsize=12)
        ax.set_title('Initial vs Final Assignments', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Ch {k}' for k in range(n_channels)])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Panel 3: Recovery rate per channel
        ax = axes[2]
        recovery_rates = [
            100 * n_recovered[k] / n_assigned_initial[k] if n_assigned_initial[k] > 0 else 0
            for k in range(n_channels)
        ]
        bars = ax.bar(range(n_channels), recovery_rates, color='seagreen', alpha=0.7)
        ax.set_xlabel('Channel', fontsize=12)
        ax.set_ylabel('Recovery Rate (%)', fontsize=12)
        ax.set_title('Localizations Recovered by Refinement', fontsize=14, fontweight='bold')
        ax.set_xticks(range(n_channels))
        ax.set_xticklabels([f'Ch {k}' for k in range(n_channels)])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, recovery_rates)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%\n(+{n_recovered[i]:,})',
                   ha='center', va='bottom', fontsize=10)
        
        _safe_tight_layout(fig)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved refinement diagnostics to: {save_path}")
        elif display:
            plt.show()

        # Create spatial distribution plot if 2D data
        n_features = len(channels_to_use)
        if n_features == 2 and 'xc' in assigned_current.columns and 'yc' in assigned_current.columns:
            self._plot_spatial_distribution(
                assigned_current, n_channels, n_recovered, save_path, display=display
            )

    def _plot_spatial_distribution(
        self,
        assigned_current: pd.DataFrame,
        n_channels: int,
        n_recovered: Dict[int, int],
        save_path: Optional[str] = None,
        display: bool = True,
    ) -> None:
        """
        Create spatial distribution plot showing initial vs recovered localizations.

        Parameters
        ----------
        assigned_current : pd.DataFrame
            DataFrame with final assignments
        n_channels : int
            Number of channels
        n_recovered : Dict[int, int]
            Number of recovered locs per channel
        save_path : Optional[str]
            Base path for saving (will append '_spatial')
        """
        import matplotlib.pyplot as plt
        from PlottingBase import PublicationPlotter

        # Create figure with n_channels + 1 subplots (one per channel + combined)
        plotter = PublicationPlotter()
        fig, axes = plotter.two_column_plot(
            nrows=1, ncols=n_channels + 1,
            width=6 * (n_channels + 1), height=5, big=True
        )
        
        # Standard matplotlib colors for channels
        channel_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

        # Plot each channel separately
        for k in range(n_channels):
            ax = axes[k]

            # Plot initial assignments (conservative)
            mask_initial = (assigned_current['channel'] == k) & \
                          (assigned_current['assignment_stage'] == 1)
            if mask_initial.any():
                ax.scatter(
                    assigned_current.loc[mask_initial, 'xc'],
                    assigned_current.loc[mask_initial, 'yc'],
                    s=1, alpha=0.5, color=channel_colors[k % len(channel_colors)],
                    label='Initial', rasterized=True
                )

            # Plot recovered assignments (refinement iterations)
            mask_refined = (assigned_current['channel'] == k) & \
                          (assigned_current['assignment_stage'] >= 2)
            if mask_refined.any():
                ax.scatter(
                    assigned_current.loc[mask_refined, 'xc'],
                    assigned_current.loc[mask_refined, 'yc'],
                    s=2, alpha=0.8, color='gold', marker='x',
                    label='Recovered', rasterized=True
                )
            
            ax.set_title(f'Channel {k}\n(+{n_recovered[k]:,} recovered)', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('x (pixels)', fontsize=11)
            ax.set_ylabel('y (pixels)', fontsize=11)
            ax.legend(fontsize=10, markerscale=3)
            ax.set_aspect('equal', adjustable='box')
        
        # Combined plot showing all channels
        ax = axes[n_channels]
        for k in range(n_channels):
            mask_k = (assigned_current['channel'] == k)
            if mask_k.any():
                ax.scatter(
                    assigned_current.loc[mask_k, 'xc'],
                    assigned_current.loc[mask_k, 'yc'],
                    s=1, alpha=0.5, color=channel_colors[k % len(channel_colors)],
                    label=f'Channel {k}', rasterized=True
                )
        
        # Show unassigned locs
        mask_unassigned = (assigned_current['channel'] == -1)
        if mask_unassigned.any():
            ax.scatter(
                assigned_current.loc[mask_unassigned, 'xc'],
                assigned_current.loc[mask_unassigned, 'yc'],
                s=1, alpha=0.3, color='black',
                label='Unassigned', rasterized=True
            )
        
        ax.set_title('All Channels (Final)', fontsize=12, fontweight='bold')
        ax.set_xlabel('x (pixels)', fontsize=11)
        ax.set_ylabel('y (pixels)', fontsize=11)
        ax.legend(fontsize=10, markerscale=3)
        ax.set_aspect('equal', adjustable='box')
        
        _safe_tight_layout(fig)

        if save_path:
            # Append '_spatial' to filename
            base, ext = os.path.splitext(save_path)
            spatial_path = f"{base}_spatial{ext}"
            fig.savefig(spatial_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved spatial distribution to: {spatial_path}")
        elif display:
            plt.show()

    def find_exemplar_dye_pair(
        self,
        sf_db,
        mean_0,
        mean_1,
        spectral_tol: float = 0.05,
        min_spatial_dist_nm: float = 500.0,
        max_spatial_dist_nm: float | None = None,
        min_photons: float = 2000.0,
        pixel_size: float = None,  # nm; None → self.pixel_size * 1000
        n_top: int = 10,
    ):
        """Find a co-localised pair of single-frame localisations representing two dye classes.

        Searches the **single-frame** database so that both localisations must
        appear in the same frame of the same FOV — i.e. they are simultaneously
        visible.  Candidates in each class are those whose (A_R, A_G) lies within
        `spectral_tol` of the class mean.  Pairs are ranked by spatial separation
        so the result can be used directly as an exemplar figure.

        Args:
            sf_db: Single-frame DataFrame as returned by extract_single_molecules_*.
                   Must contain: xc, yc, A_R, A_G, photons, frame, fov_index, molecular_index.
            mean_0: (A_R, A_G) mean for class 0 (e.g. from GMM fixed_means[0]).
            mean_1: (A_R, A_G) mean for class 1 (e.g. from GMM fixed_means[1]).
            spectral_tol: Maximum Euclidean distance in (A_R, A_G) space for a
                          localisation to be accepted as a candidate for a class.
                          Start with 0.05; relax if no pairs found.
            min_spatial_dist_nm: Minimum separation (nm) — excludes pairs that are
                                 too close to resolve in the raw image (default 500).
            max_spatial_dist_nm: Maximum separation (nm).  None means no upper limit.
            min_photons: Minimum photon count for a localisation to be considered
                         (default 2000).
            pixel_size: Camera pixel size in nm (default 69.0 nm for Ximea).
            n_top: Number of best pairs to return (ranked by spatial_dist_nm).

        Returns:
            pd.DataFrame with one row per candidate pair, sorted by
            spatial_dist_nm (closest first), containing:
              fov_index, frame, mol_0_idx, mol_1_idx,
              xc_0, yc_0, A_R_0, A_G_0, spec_dist_0,
              xc_1, yc_1, A_R_1, A_G_1, spec_dist_1,
              spatial_dist_nm, spectral_score
            Returns None if no pairs satisfy the constraints.
        """
        if pixel_size is None:
            pixel_size = self.pixel_size * 1000  # µm → nm

        import pandas as pd
        from scipy.spatial.distance import cdist

        mean_0 = np.asarray(mean_0, dtype=float)
        mean_1 = np.asarray(mean_1, dtype=float)

        sf_db = sf_db[sf_db['photons'] >= min_photons]

        ar = sf_db['A_R'].to_numpy()
        ag = sf_db['A_G'].to_numpy()

        spec_dist_0 = np.sqrt((ar - mean_0[0])**2 + (ag - mean_0[1])**2)
        spec_dist_1 = np.sqrt((ar - mean_1[0])**2 + (ag - mean_1[1])**2)

        cands_0 = sf_db[spec_dist_0 <= spectral_tol].copy()
        cands_0['spec_dist'] = spec_dist_0[spec_dist_0 <= spectral_tol]

        cands_1 = sf_db[spec_dist_1 <= spectral_tol].copy()
        cands_1['spec_dist'] = spec_dist_1[spec_dist_1 <= spectral_tol]

        if len(cands_0) == 0 or len(cands_1) == 0:
            logger.info(f"No candidates found within spectral_tol={spectral_tol:.3f}. " f"Class 0: {len(cands_0)} candidates, Class 1: {len(cands_1)} candidates. " "Try increasing spectral_tol.")
            return None

        fov_col = 'fov_index' if 'fov_index' in sf_db.columns else 'fov_name'

        # Group by (fov, frame) so both localisations must be in the same frame.
        common_groups = set(
            map(tuple, cands_0[[fov_col, 'frame']].drop_duplicates().values)
        ) & set(
            map(tuple, cands_1[[fov_col, 'frame']].drop_duplicates().values)
        )

        if len(common_groups) == 0:
            logger.info("No (FOV, frame) group contains candidates from both classes.")
            return None

        all_pairs = []
        for fov, frame in common_groups:
            f0 = cands_0[(cands_0[fov_col] == fov) & (cands_0['frame'] == frame)]
            f1 = cands_1[(cands_1[fov_col] == fov) & (cands_1['frame'] == frame)]

            xy_0 = f0[['xc', 'yc']].to_numpy() * pixel_size  # nm
            xy_1 = f1[['xc', 'yc']].to_numpy() * pixel_size

            dists_nm = cdist(xy_0, xy_1)  # (n0, n1)

            i_flat, j_flat = np.meshgrid(
                np.arange(len(f0)), np.arange(len(f1)), indexing='ij'
            )
            i_flat = i_flat.ravel()
            j_flat = j_flat.ravel()
            d_flat = dists_nm.ravel()

            keep = d_flat >= min_spatial_dist_nm
            if max_spatial_dist_nm is not None:
                keep &= d_flat <= max_spatial_dist_nm
            i_flat, j_flat, d_flat = i_flat[keep], j_flat[keep], d_flat[keep]

            if len(d_flat) == 0:
                continue

            mol0 = f0.iloc[i_flat]
            mol1 = f1.iloc[j_flat]

            sd0 = mol0['spec_dist'].to_numpy()
            sd1 = mol1['spec_dist'].to_numpy()

            pairs = pd.DataFrame({
                fov_col:           fov,
                'frame':           frame,
                'mol_0_idx':       mol0['molecular_index'].to_numpy(),
                'xc_0':            mol0['xc'].to_numpy(),
                'yc_0':            mol0['yc'].to_numpy(),
                'A_R_0':           mol0['A_R'].to_numpy(),
                'A_G_0':           mol0['A_G'].to_numpy(),
                'spec_dist_0':     sd0,
                'mol_1_idx':       mol1['molecular_index'].to_numpy(),
                'xc_1':            mol1['xc'].to_numpy(),
                'yc_1':            mol1['yc'].to_numpy(),
                'A_R_1':           mol1['A_R'].to_numpy(),
                'A_G_1':           mol1['A_G'].to_numpy(),
                'spec_dist_1':     sd1,
                'spatial_dist_nm': d_flat,
                'spectral_score':  sd0 + sd1,
            })
            all_pairs.append(pairs)

        if not all_pairs:
            logger.info("No pairs found within spatial distance constraints.")
            return None

        result = pd.concat(all_pairs, ignore_index=True)
        # A molecule within spectral_tol of both means would appear in both cands_0 and
        # cands_1, producing a self-pair.  Remove any row where mol_0 == mol_1.
        result = result[result['mol_0_idx'] != result['mol_1_idx']]
        if len(result) == 0:
            logger.info("All pairs were self-pairs (same molecule in both classes). " "Try reducing spectral_tol.")
            return None
        result = result.sort_values('spatial_dist_nm').head(n_top).reset_index(drop=True)

        logger.info(f"Found {len(result)} candidate pairs from {len(common_groups)} (FOV, frame) group(s).")
        logger.info(f"Best pair: FOV={result.iloc[0][fov_col]}, frame={result.iloc[0]['frame']}, " f"dist={result.iloc[0]['spatial_dist_nm']:.0f} nm, " f"mol_0 A_R={result.iloc[0]['A_R_0']:.3f} A_G={result.iloc[0]['A_G_0']:.3f}, " f"mol_1 A_R={result.iloc[0]['A_R_1']:.3f} A_G={result.iloc[0]['A_G_1']:.3f}")
        return result

    def get_exemplar_crop(
        self,
        pair_row,
        data_folder: str,
        crop_size_px: int = 30,
    ):
        """Load the raw TIFF for the FOV in pair_row and return a single-frame crop.

        The TIFF is located by sorting all TIFFs in data_folder and picking the
        one at position fov_index.  The specific frame stored in pair_row['frame']
        is extracted directly — no projection — so the crop shows exactly what the
        camera captured when both molecules were localised.

        Args:
            pair_row: A single row (pd.Series) from find_exemplar_dye_pair,
                      typically result.iloc[0] for the best pair.
            data_folder: Directory containing the raw TIFF files.
            crop_size_px: Half-width of the square crop in camera pixels.
                          The returned image is (2*crop_size_px) × (2*crop_size_px).

        Returns:
            crop: 2D np.ndarray — the raw single-frame crop.
            pair_info: pd.DataFrame (single row) — the input pair_row with extra
                       columns: xc_0_crop, yc_0_crop, xc_1_crop, yc_1_crop
                       giving molecule positions relative to the crop origin.
        """
        import glob

        fov_index = int(pair_row['fov_index'])
        frame_index = int(pair_row['frame'])

        # Collect and sort all TIFFs in the folder, then pick the Nth one.
        tif_files = sorted(
            f for f in glob.glob(os.path.join(data_folder, "*.tif*"))
            if not f.endswith('.h5')
        )

        if len(tif_files) == 0:
            raise FileNotFoundError(f"No TIFF files found in '{data_folder}'.")
        if fov_index >= len(tif_files):
            raise IndexError(
                f"fov_index={fov_index} but only {len(tif_files)} TIFFs found in "
                f"'{data_folder}'."
            )
        tif_path = tif_files[fov_index]
        logger.info(f"Loading FOV {fov_index}, frame {frame_index}: {os.path.basename(tif_path)}")

        # Load a single frame directly via IO.read_tiff
        projected = self.io.read_tiff(tif_path, frame=frame_index)

        # Crop centre: midpoint of the two molecule positions (camera pixels)
        cx = int(round(0.5 * (pair_row['xc_0'] + pair_row['xc_1'])))
        cy = int(round(0.5 * (pair_row['yc_0'] + pair_row['yc_1'])))

        H, W = projected.shape
        x0 = max(cx - crop_size_px, 0)
        x1 = min(cx + crop_size_px, W)
        y0 = max(cy - crop_size_px, 0)
        y1 = min(cy + crop_size_px, H)

        crop = projected[y0:y1, x0:x1]

        # Molecule positions relative to crop origin
        pair_info = pair_row.to_frame().T.copy().reset_index(drop=True)
        pair_info['xc_0'] = pair_row['xc_0'] - x0
        pair_info['yc_0'] = pair_row['yc_0'] - y0
        pair_info['xc_1'] = pair_row['xc_1'] - x0
        pair_info['yc_1'] = pair_row['yc_1'] - y0
        pair_info['crop_x0_px'] = x0
        pair_info['crop_y0_px'] = y0

        return crop, pair_info
