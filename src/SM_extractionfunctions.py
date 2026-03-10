# -*- coding: utf-8 -*-
"""
This class contains functions pertaining to analysis of images,
relating to the bayerSMLM concept.
jsb92, 2024/01/02
"""
import numpy as np
import pandas as pd
import os
import sys
import gc
import re
from typing import Tuple, Dict, Optional

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)
import IOFunctions

import postprocess
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KDTree
from scipy.stats import multivariate_normal
import warnings

# Try to import fast_hdbscan (multicore-optimized), fall back to sklearn
try:
    from fast_hdbscan import HDBSCAN
    HDBSCAN_BACKEND = "fast_hdbscan"
except ImportError:
    from sklearn.cluster import HDBSCAN
    HDBSCAN_BACKEND = "sklearn"


def _safe_tight_layout(fig):
    """Apply tight_layout while suppressing UserWarning about layout changes.

    This suppresses the "The figure layout has changed to tight" warning that
    occurs when tight_layout() is called on figures that already have automatic
    layout management enabled.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                              message='The figure layout has changed to tight',
                              category=UserWarning)
        try:
            fig.tight_layout()
        except Exception:
            # If tight_layout fails for any reason, just continue
            pass


class extract_SMs:
    def __init__(self, io_functions=None) -> None:
        """Single molecule extraction functions for clustering localizations into single molecules.

        Args:
            io_functions: IO functions instance (default: creates new instance)
        """
        # Dependency injection with sensible defaults
        self.io = (
            io_functions if io_functions is not None else IOFunctions.IO_Functions()
        )

    def _load_localisation_files(self, loc_data):
        """Load localisations from HDF5 file paths if needed.

        Args:
            loc_data: Either a pd.DataFrame (returned as-is) or a list/array of
                      HDF5 file paths (concatenated and returned as pd.DataFrame).

        Returns:
            pd.DataFrame of localisation data.
        """
        if isinstance(loc_data, pd.DataFrame):
            return loc_data
        # Treat as iterable of file paths
        dfs = []
        for f in loc_data:
            dfs.append(pd.read_hdf(str(f), key="data"))
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def filter_quality_localisations(
        self,
        loc_data,
        chi_val=None,
        max_localisation_error=1.0,
        max_colour_error=0.15,
        min_sigma=(75./69),
        max_sigma=(160./69),
        max_sigma_error=(40./69),
        min_photons=500,
        max_photons=None,
    ):
        """
        Apply quality filters to localisation data.

        Args:
            loc_data (pd.DataFrame): Localization data to filter
            chi_val (float, optional): Chi-squared threshold. If None, uses median.
            max_localization_error (float): Maximum localisation precision in pixels
            min_photons (int): Minimum total photon count
            max_photons (int): Maximum total photon count

        Returns:
            pd.DataFrame: Filtered localisation data
        """
        # Calculate chi-squared threshold if not provided
        if chi_val is None:
            chi_val = np.median(loc_data["chi_sqr"])

        # Apply quality filters
        filtered_data = loc_data[loc_data["chi_sqr"] < chi_val].copy()
        filtered_data = filtered_data[filtered_data["xc_err"] < max_localisation_error]
        filtered_data = filtered_data[filtered_data["yc_err"] < max_localisation_error]
        for key in ['A_R_err', 'A_G_err', 'A_B_err', 'bg_R_err', 'bg_G_err', 'bg_B_err']:
            filtered_data = filtered_data[filtered_data[key] < max_colour_error]
        for key in ['s_x_err', 's_y_err']:
            filtered_data = filtered_data[filtered_data[key] < max_sigma_error]
        for key in ['s_x', 's_y']:
            filtered_data = filtered_data[filtered_data[key] < max_sigma]
            filtered_data = filtered_data[filtered_data[key] > min_sigma]

        # Add photons column using centralized method and apply photon count filters
        if "photons" not in filtered_data.columns:
            filtered_data = self.io._add_photon_columns(filtered_data, normalise=True)
        filtered_data = filtered_data[filtered_data["photons"] < max_photons]
        filtered_data = filtered_data[filtered_data["photons"] > min_photons]

        return filtered_data.reset_index(drop=True)

    def average_parameters(self, data, dbscan_labels):
        """
        Average parameters for clustered single molecule localizations.

        Args:
            data (pd.DataFrame): Localization data with fitting parameters
            dbscan_labels (np.array): Cluster labels from DBSCAN/HDBSCAN

        Returns:
            pd.DataFrame: Averaged parameters per single molecule
        """
        labels = np.sort(np.unique(dbscan_labels))
        labels = labels[labels > -1]
        dict_obj = {}
        dict_obj["photons"] = np.zeros(len(labels))
        dict_obj["frames"] = np.zeros(len(labels))
        for column in np.array(data.columns):
            if column == "index":
                continue
            else:
                dict_obj[column] = np.zeros(len(labels))

        for label in labels:
            for column in np.array(data.columns):
                if column == "index":
                    continue
                elif column in ["A_B", "A_G", "A_R"]:
                    dict_obj[column][label] = np.average(
                        data[column][dbscan_labels == label],
                        weights=data[column + "_err"][dbscan_labels == label] ** -2,
                    )
                else:
                    dict_obj[column][label] = np.mean(
                        data[column][dbscan_labels == label]
                    )
            dict_obj["frames"][label] = len(data[column][dbscan_labels == label])
            # Use existing photons column if available, otherwise calculate manually
            if "photons" in data.columns:
                dict_obj["photons"][label] = np.sum(
                    data["photons"][dbscan_labels == label]
                )
            else:
                dict_obj["photons"][label] = (
                    np.average(
                        data["A_B"][dbscan_labels == label],
                        weights=data["A_B_err"][dbscan_labels == label] ** -2,
                    )
                    + np.average(
                        data["A_G"][dbscan_labels == label],
                        weights=data["A_G_err"][dbscan_labels == label] ** -2,
                    )
                    + np.average(
                        data["A_R"][dbscan_labels == label],
                        weights=data["A_R_err"][dbscan_labels == label] ** -2,
                    )
                )
        df = pd.DataFrame.from_dict(dict_obj)
        # Normalise photon fractions using centralised IOFunctions method
        return df

    def collect_traces(self, data, dbscan_labels, image_stack, image_size=12):
        """
        Collect intensity traces for clustered single molecules.

        Args:
            data (pd.DataFrame): Localization data with positions
            dbscan_labels (np.array): Cluster labels from DBSCAN/HDBSCAN
            image_stack (np.array): Full image stack [frames, x, y]
            image_size (int): Size of extraction window around molecule

        Returns:
            tuple: (locations, trace_matrix) where locations are [x,y] positions
                   and trace_matrix is [molecules, frames] intensity traces
        """
        labels = np.sort(np.unique(dbscan_labels))
        labels = labels[labels > -1]
        trace_matrix = np.zeros([len(labels), image_stack.shape[0]])
        locations = np.zeros([2, len(labels)])

        for i, label in enumerate(labels):
            locations[0, i] = np.nanmean(data["xc"][dbscan_labels == label].to_numpy())
            locations[1, i] = np.nanmean(data["yc"][dbscan_labels == label].to_numpy())
            xmin = int(locations[0, i]) - int(image_size / 2)
            xmax = int(locations[0, i]) + int(image_size / 2)
            ymin = int(locations[1, i]) - int(image_size / 2)
            ymax = int(locations[1, i]) + int(image_size / 2)
            # Extract ROI using correct indexing [frame, y, x]
            trace_matrix[i, :] = np.sum(
                np.sum(image_stack[:, ymin:ymax, xmin:xmax], axis=-1), axis=-1
            )
            print(
                "Summed trace {}/{}".format(i + 1, len(labels)),
                end="\r",
                flush=True,
            )

        return locations, trace_matrix

    def extract_single_molecules_HDBSCAN(
        self,
        loc_data,
        min_cluster_size=10,
        chi_val=None,
        max_localisation_error=1.0,
        max_colour_error=0.15,
        min_sigma=(75./69),
        max_sigma=(160./69),
        max_sigma_error=(40./69),
        min_photons=500,
        max_photons=None,
    ):
        """
        Extract single molecules from multiple localization files by clustering.

        Args:
            loc_data (pd.DataFrame): Localization data to process
            chi_val (float, optional): Chi-squared threshold for filtering. Defaults to median.

        Returns:
            tuple: (single_molecule_database, single_frame_database) as DataFrames
                   single_frame_database includes molecular_index column and excludes unassigned localizations
        """

        molecular_index_offset = 0

        loc_data = self._load_localisation_files(loc_data)

        loc_data = self.filter_quality_localisations(
            loc_data=loc_data, chi_val=chi_val, max_localisation_error=max_localisation_error,
            min_photons=min_photons, max_photons=max_photons, max_colour_error=max_colour_error,
            min_sigma=min_sigma, max_sigma=max_sigma, max_sigma_error=max_sigma_error
        )

        # Check if we have any data left after filtering
        if len(loc_data) == 0:
            print("Warning: No localizations remaining after filtering. Returning empty databases.")
            return pd.DataFrame(), pd.DataFrame()

        # Check if we have enough points for clustering
        # HDBSCAN needs at least min_cluster_size points to work
        if len(loc_data) < min_cluster_size:
            print(f"Warning: Only {len(loc_data)} localizations remaining after filtering, "
                  f"but min_cluster_size={min_cluster_size}. Need at least {min_cluster_size} points. "
                  f"Returning empty databases.")
            return pd.DataFrame(), pd.DataFrame()

        X = np.vstack([loc_data["xc"], loc_data["yc"]]).T
        loc_precision = 0.5 * (
            np.mean(loc_data["xc_err"]) + np.mean(loc_data["yc_err"])
        )

        print(f"Using {HDBSCAN_BACKEND} for HDBSCAN clustering")
        hdb = HDBSCAN(
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=loc_precision,
        )
        hdb.fit(X)

        # Filter out unassigned localizations (label = -1)
        assigned_mask = hdb.labels_ >= 0
        loc_data_assigned = loc_data[assigned_mask].copy()
        labels_assigned = hdb.labels_[assigned_mask]

        # Add molecular index column (offset by previous files)
        loc_data_assigned["molecular_index"] = labels_assigned + molecular_index_offset

        # Create single molecule database for this file
        df = self.average_parameters(loc_data_assigned, labels_assigned)
        df["molecular_index"] = df.index + molecular_index_offset

        single_frame_database = loc_data_assigned
        single_molecule_database = df

        return single_molecule_database, single_frame_database

    def extract_single_molecules_DBSCAN(
        self,
        loc_data,
        min_cluster_size=10,
        chi_val=None,
        max_localisation_error=1.0,
        max_colour_error=0.15,
        min_sigma=(75./69),
        max_sigma=(160./69),
        max_sigma_error=(40./69),
        min_photons=500,
        max_photons=None,
        epsilon_multiplier=1.0,
    ):
        """
        Extract single molecules from multiple localization files by clustering.

        Args:
            localisation_files (list): List of HDF5 localization file paths
            chi_val (float, optional): Chi-squared threshold for filtering. Defaults to median.
            epsilon_multiplier (float): Multiplicative factor for epsilon (clustering distance).
                                       epsilon = avg_localization_error * epsilon_multiplier.
                                       Defaults to 1.0.

        Returns:
            tuple: (single_molecule_database, single_frame_database) as DataFrames
                   single_frame_database includes molecular_index column and excludes unassigned localizations
        """

        molecular_index_offset = 0

        loc_data = self._load_localisation_files(loc_data)

        loc_data = self.filter_quality_localisations(
            loc_data=loc_data, chi_val=chi_val, max_localisation_error=max_localisation_error,
            min_photons=min_photons, max_photons=max_photons, max_colour_error=max_colour_error,
            min_sigma=min_sigma, max_sigma=max_sigma, max_sigma_error=max_sigma_error
        )

        # Check if we have any data left after filtering
        if len(loc_data) == 0:
            print("Warning: No localizations remaining after filtering. Returning empty databases.")
            return pd.DataFrame(), pd.DataFrame()

        # Check if we have enough points for clustering
        # DBSCAN needs at least min_samples points to work
        if len(loc_data) < min_cluster_size:
            print(f"Warning: Only {len(loc_data)} localizations remaining after filtering, "
                  f"but min_cluster_size={min_cluster_size}. Need at least {min_cluster_size} points. "
                  f"Returning empty databases.")
            return pd.DataFrame(), pd.DataFrame()

        X = np.vstack([loc_data["xc"], loc_data["yc"]]).T
        loc_precision = 0.5 * (
            np.mean(loc_data["xc_err"]) + np.mean(loc_data["yc_err"])
        )
        if loc_precision <= 0:
            raise ValueError(
                f"loc_precision = {loc_precision:.6f} (computed from mean xc_err / yc_err). "
                "Error columns are zero or NaN — this usually means chi_sqr was very small "
                "(bright spots cause pcov * chisqr → 0) or errors were not saved. "
                "Check that ImageAnalysisFunctions.calculate_errors returns non-zero values "
                "for your data, or pass epsilon_multiplier with an explicit eps value."
            )
        hdb = DBSCAN(
            min_samples=min_cluster_size,
            eps=loc_precision * epsilon_multiplier,
        )
        hdb.fit(X)

        # Filter out unassigned localizations (label = -1)
        assigned_mask = hdb.labels_ >= 0
        loc_data_assigned = loc_data[assigned_mask].copy()
        labels_assigned = hdb.labels_[assigned_mask]

        # Add molecular index column (offset by previous files)
        loc_data_assigned["molecular_index"] = labels_assigned + molecular_index_offset

        # Create single molecule database for this file
        df = self.average_parameters(loc_data_assigned, labels_assigned)
        df["molecular_index"] = df.index + molecular_index_offset

        single_frame_database = loc_data_assigned
        single_molecule_database = df

        return single_molecule_database, single_frame_database

    def extract_single_molecules_linked(
        self,
        loc_data,
        max_distance=1.0,
        max_frames=10,
        chi_val=None,
        max_localisation_error=1.0,
        max_colour_error=0.15,
        min_sigma=(75./69),
        max_sigma=(160./69),
        max_sigma_error=(40./69),
        min_photons=500,
        max_photons=None,
    ):
        """
        Extract single molecules from multiple localization files using temporal linking.

        Uses postprocess.py get_link_groups function to link localizations across frames
        based on spatial proximity and temporal continuity.

        Args:
            localisation_files (list): List of HDF5 localization file paths
            max_distance (float): Maximum distance for linking localizations (pixels)
            max_frames (int): Maximum frame gap for linking
            chi_val (float, optional): Chi-squared threshold for filtering. Defaults to median.
            max_localization_error (float): Maximum localization precision in pixels
            min_photons (int): Minimum total photon count
            max_photons (int): Maximum total photon count

        Returns:
            tuple: (single_molecule_database, single_frame_database) as DataFrames
                   single_frame_database includes molecular_index column and excludes unlinked localizations
        """
        molecular_index_offset = 0

        loc_data = self.filter_quality_localisations(
            loc_data=loc_data, chi_val=chi_val, max_localisation_error=max_localisation_error, 
            min_photons=min_photons, max_photons=max_photons, max_colour_error=max_colour_error,
            min_sigma=min_sigma, max_sigma=max_sigma, max_sigma_error=max_sigma_error
        )

        # Convert to numpy record array for postprocess.py compatibility and sort by frame
        loc_data_sorted = loc_data.sort_values("frame")
        loc_array = loc_data_sorted.to_records(index=False)

        # Create group array (all localizations belong to same group for single molecule analysis)
        group = np.zeros(len(loc_array), dtype=np.int32)

        # Use postprocess linking function
        link_groups = postprocess.get_link_groups(
            loc_array, max_distance, max_frames, group
        )

        # Filter out unlinked localizations (group = -1 typically means no link)
        linked_mask = link_groups >= 0
        loc_data_linked = loc_data_sorted[linked_mask].copy()
        link_groups_linked = link_groups[linked_mask]

        # Add molecular index column (offset by previous files)
        loc_data_linked["molecular_index"] = link_groups_linked + molecular_index_offset

        # Create single molecule database for this file
        df = self.average_parameters(loc_data_linked, link_groups_linked)
        df["molecular_index"] = df.index + molecular_index_offset

        single_frame_database = loc_data_linked
        single_molecule_database = df

        # Update offset for next file
        molecular_index_offset += len(df)

        return single_molecule_database, single_frame_database

    # -----------------------------------------------------------------
    # Spectral-assisted LAP linking
    # -----------------------------------------------------------------

    def spectral_lap_link(
        self,
        loc_data,
        max_distance=1.0,
        max_dark_time=1,
        w_spatial=1.0,
        w_spectral=0.5,
        spectral_tol=0.15,
        spectral_columns=("A_R", "A_G", "A_B"),
        verbose=False,
    ):
        """Link localisations across frames using LAP with spectral cost.

        Builds a cost matrix combining spatial distance and spectral (colour)
        distance for each frame, then solves the Linear Assignment Problem
        (Hungarian algorithm) for globally optimal frame-to-frame assignments.
        Active tracks persist across dark frames up to ``max_dark_time``.

        The LAP formulation and augmented cost matrix for birth/death follow
        Jaqaman et al. (2008) Nat. Methods 5, 695-702. Spectral distance in
        normalised colour space is added to the spatial cost, extending the
        approach to multicolour single-molecule data. The cost function is:

            cost = w_spatial * (d_spatial / max_distance)**2
                 + w_spectral * (d_spectral / spectral_tol)**2

        References:
            Jaqaman, K. et al. (2008) Robust single-particle tracking in
                live-cell time-lapse sequences. Nat. Methods 5, 695-702.
            Crocker, J. C. & Grier, D. G. (1996) Methods of digital video
                microscopy for colloidal studies. J. Colloid Interface Sci.
                179, 298-310.
            Chenouard, N. et al. (2014) Objective comparison of particle
                tracking methods. Nat. Methods 11, 281-289.
            Serge, A. et al. (2008) Dynamic multiple-target tracing to probe
                spatiotemporal cartography of cell membranes. Nat. Methods
                5, 687-694.

        Args:
            loc_data (pd.DataFrame): Localisations sorted by frame. Must
                contain columns ``frame``, ``xc``, ``yc`` and the columns
                listed in *spectral_columns*.
            max_distance (float): Hard spatial cutoff in pixels. Pairs
                further apart than this are never linked.
            max_dark_time (int): Maximum frame gap allowed between
                consecutive localisations in a track.
            w_spatial (float): Weight for the spatial cost term.
            w_spectral (float): Weight for the spectral cost term.
            spectral_tol (float): Normalisation scale for spectral
                distance (analogous to ``max_distance`` for spatial).
            spectral_columns (tuple): Column names for the spectral
                channels used in the cost function.
            verbose (bool): Print progress every 500 frames.

        Returns:
            np.ndarray: Integer array of length ``len(loc_data)`` where
            each element is a track ID (``>= 0``) or ``-1`` for
            unlinked localisations. Compatible with
            ``average_parameters()`` and ``link_loc_groups()``.
        """
        from scipy.optimize import linear_sum_assignment

        frames_arr = loc_data["frame"].to_numpy()
        xc = loc_data["xc"].to_numpy()
        yc = loc_data["yc"].to_numpy()

        # Build normalised spectral vectors
        spec_cols = [loc_data[c].to_numpy() for c in spectral_columns]
        spectra = np.column_stack(spec_cols)  # (N, n_channels)
        spec_sum = spectra.sum(axis=1, keepdims=True)
        spec_sum[spec_sum == 0] = 1.0  # avoid division by zero
        spectra_norm = spectra / spec_sum

        n_locs = len(loc_data)
        link_group = np.full(n_locs, -1, dtype=np.int32)
        unique_frames = np.unique(frames_arr)

        # Cutoff cost for birth/death dummies (at threshold boundary)
        cutoff = w_spatial + w_spectral

        # Active tracks: list of dicts with track state
        # Each: {'id': int, 'last_frame': int, 'last_x': float,
        #        'last_y': float, 'last_spec': array}
        active_tracks = []
        next_track_id = 0

        for fi, current_frame in enumerate(unique_frames):
            if verbose and fi % 500 == 0 and fi > 0:
                print(f"  Frame {fi}/{len(unique_frames)}, "
                      f"{next_track_id} tracks so far")

            # Indices of localisations in this frame
            frame_mask = frames_arr == current_frame
            cur_indices = np.where(frame_mask)[0]
            n_cur = len(cur_indices)

            if n_cur == 0:
                continue

            cur_x = xc[cur_indices]
            cur_y = yc[cur_indices]
            cur_spec = spectra_norm[cur_indices]

            # Prune expired tracks
            active_tracks = [
                t for t in active_tracks
                if t["last_frame"] >= current_frame - max_dark_time - 1
            ]
            n_active = len(active_tracks)

            if n_active == 0:
                # No active tracks — all current locs start new tracks
                for j, idx in enumerate(cur_indices):
                    link_group[idx] = next_track_id
                    active_tracks.append({
                        "id": next_track_id,
                        "last_frame": current_frame,
                        "last_x": cur_x[j],
                        "last_y": cur_y[j],
                        "last_spec": cur_spec[j],
                    })
                    next_track_id += 1
                continue

            # --- Build cost matrix (n_active x n_cur) ---
            # Vectorised spatial distances
            trk_x = np.array([t["last_x"] for t in active_tracks])
            trk_y = np.array([t["last_y"] for t in active_tracks])
            trk_spec = np.array([t["last_spec"] for t in active_tracks])

            dx = trk_x[:, None] - cur_x[None, :]  # (n_active, n_cur)
            dy = trk_y[:, None] - cur_y[None, :]
            d_spatial = np.sqrt(dx**2 + dy**2)

            # Spectral distance (Euclidean in normalised RGB space)
            d_spectral = np.sqrt(
                ((trk_spec[:, None, :] - cur_spec[None, :, :]) ** 2).sum(axis=2)
            )

            cost = (w_spatial * (d_spatial / max_distance) ** 2
                    + w_spectral * (d_spectral / spectral_tol) ** 2)

            # Hard spatial cutoff
            cost[d_spatial > max_distance] = 1e9

            # --- Augment for birth/death (Jaqaman 2008) ---
            # Build (n_active + n_cur) x (n_active + n_cur) matrix
            dim = n_active + n_cur
            aug = np.full((dim, dim), 1e9)

            # Upper-left: real assignment costs
            aug[:n_active, :n_cur] = cost

            # Upper-right: track death (diagonal, cost = cutoff)
            for i in range(n_active):
                aug[i, n_cur + i] = cutoff

            # Lower-left: track birth (diagonal, cost = cutoff)
            for j in range(n_cur):
                aug[n_active + j, j] = cutoff

            # Lower-right: dummy-to-dummy (zero cost, transpose of upper-left)
            aug[n_active:, n_cur:] = 0.0

            # --- Solve LAP ---
            row_ind, col_ind = linear_sum_assignment(aug)

            # Process assignments
            assigned_cur = set()
            matched_tracks = set()

            for r, c in zip(row_ind, col_ind):
                if r < n_active and c < n_cur:
                    # Real assignment: track r -> current loc c
                    if aug[r, c] < cutoff:
                        trk = active_tracks[r]
                        idx = cur_indices[c]
                        link_group[idx] = trk["id"]
                        trk["last_frame"] = current_frame
                        trk["last_x"] = cur_x[c]
                        trk["last_y"] = cur_y[c]
                        trk["last_spec"] = cur_spec[c]
                        assigned_cur.add(c)
                        matched_tracks.add(r)

            # Unassigned current locs → new tracks
            for j in range(n_cur):
                if j not in assigned_cur:
                    idx = cur_indices[j]
                    link_group[idx] = next_track_id
                    active_tracks.append({
                        "id": next_track_id,
                        "last_frame": current_frame,
                        "last_x": cur_x[j],
                        "last_y": cur_y[j],
                        "last_spec": cur_spec[j],
                    })
                    next_track_id += 1

        if verbose:
            n_tracks = len(np.unique(link_group[link_group >= 0]))
            n_linked = np.sum(link_group >= 0)
            print(f"  Linking complete: {n_tracks} tracks from "
                  f"{n_linked}/{n_locs} localisations")

        return link_group

    def extract_single_molecules_spectral_lap(
        self,
        loc_data,
        max_distance=1.0,
        max_dark_time=1,
        w_spatial=1.0,
        w_spectral=0.5,
        spectral_tol=0.15,
        spectral_columns=("A_R", "A_G", "A_B"),
        min_frames=3,
        chi_val=None,
        max_localisation_error=1.0,
        max_colour_error=0.15,
        min_sigma=(75.0 / 69),
        max_sigma=(160.0 / 69),
        max_sigma_error=(40.0 / 69),
        min_photons=500,
        max_photons=None,
        verbose=False,
    ):
        """Extract single molecules using spectral-assisted LAP linking.

        Same interface and output format as
        ``extract_single_molecules_linked()`` but uses a Linear Assignment
        Problem (Hungarian algorithm) with combined spatial + spectral cost
        instead of greedy nearest-neighbour linking (Jaqaman et al. 2008,
        Nat. Methods 5, 695-702). This produces globally optimal
        frame-to-frame assignments and uses colour information to
        disambiguate molecules that are spatially close.

        Args:
            loc_data (pd.DataFrame): Localisation data with columns
                ``frame``, ``xc``, ``yc``, ``A_R``, ``A_G``, ``A_B``, etc.
            max_distance (float): Maximum linking distance in pixels.
            max_dark_time (int): Maximum frame gap for linking.
            w_spatial (float): Weight for spatial cost term.
            w_spectral (float): Weight for spectral cost term.
            spectral_tol (float): Spectral distance tolerance.
            spectral_columns (tuple): Column names for spectral channels.
            min_frames (int): Minimum number of localisations per track.
                Tracks shorter than this are discarded.
            chi_val, max_localisation_error, max_colour_error, min_sigma,
            max_sigma, max_sigma_error, min_photons, max_photons:
                Quality filter parameters (see ``filter_quality_localisations``).
            verbose (bool): Print progress messages.

        Returns:
            tuple: ``(single_molecule_database, single_frame_database)``
                as pandas DataFrames, same format as
                ``extract_single_molecules_linked()``.
        """
        loc_data = self.filter_quality_localisations(
            loc_data=loc_data,
            chi_val=chi_val,
            max_localisation_error=max_localisation_error,
            min_photons=min_photons,
            max_photons=max_photons,
            max_colour_error=max_colour_error,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            max_sigma_error=max_sigma_error,
        )

        loc_data_sorted = loc_data.sort_values("frame").copy()

        if verbose:
            print(f"Spectral LAP linking: {len(loc_data_sorted)} localisations, "
                  f"max_distance={max_distance}, max_dark_time={max_dark_time}")

        link_groups = self.spectral_lap_link(
            loc_data_sorted,
            max_distance=max_distance,
            max_dark_time=max_dark_time,
            w_spatial=w_spatial,
            w_spectral=w_spectral,
            spectral_tol=spectral_tol,
            spectral_columns=spectral_columns,
            verbose=verbose,
        )

        # Filter out unlinked localisations
        linked_mask = link_groups >= 0
        loc_data_linked = loc_data_sorted[linked_mask].copy()
        link_groups_linked = link_groups[linked_mask]

        # Re-number groups contiguously from 0
        unique_ids = np.unique(link_groups_linked)
        id_map = {old: new for new, old in enumerate(unique_ids)}
        link_groups_linked = np.array(
            [id_map[g] for g in link_groups_linked], dtype=np.int32
        )

        loc_data_linked["molecular_index"] = link_groups_linked

        # Aggregate into single molecule database
        single_molecule_database = self.average_parameters(
            loc_data_linked, link_groups_linked
        )
        single_molecule_database["molecular_index"] = np.arange(
            len(single_molecule_database)
        )

        # Remove short tracks
        if min_frames > 1:
            keep_mask = single_molecule_database["frames"] >= min_frames
            removed_ids = set(
                single_molecule_database.loc[~keep_mask, "molecular_index"]
            )
            single_molecule_database = single_molecule_database[keep_mask].reset_index(drop=True)
            single_molecule_database["molecular_index"] = np.arange(
                len(single_molecule_database)
            )
            loc_data_linked = loc_data_linked[
                ~loc_data_linked["molecular_index"].isin(removed_ids)
            ].copy()

        if verbose:
            n_tracks = len(single_molecule_database)
            n_locs_linked = len(loc_data_linked)
            print(f"Result: {n_tracks} molecules from "
                  f"{n_locs_linked} linked localisations "
                  f"(min_frames={min_frames})")

        return single_molecule_database, loc_data_linked

    def _extract_fov_name(self, filepath):
        """
        Extract FOV identifier from filename.

        Looks for 'Pos' followed by digits in the filename.

        Args:
            filepath (str): Full path to localization file

        Returns:
            str: FOV name (e.g., "Pos0", "Pos15") or None if pattern not found

        Examples:
            >>> _extract_fov_name("/path/to/Pos15_undrifted_locs.h5")
            "Pos15"
            >>> _extract_fov_name("/path/to/Pos0_data.h5")
            "Pos0"
            >>> _extract_fov_name("/path/to/nopattern.h5")
            None
        """
        # Extract filename from path
        filename = os.path.basename(filepath)

        # Search for Pos followed by digits
        match = re.search(r"Pos\d+", filename)

        if match:
            return match.group(0)
        else:
            return None

    def extract_single_molecules_batch(
        self,
        localisation_files,
        clustering_method="HDBSCAN",
        min_cluster_size=10,
        chi_val=None,
        max_localisation_error=1.0,
        max_colour_error=0.15,
        min_sigma=(75./69),
        max_sigma=(160./69),
        max_sigma_error=(40./69),
        min_photons=500,
        max_photons=1e6,
        max_distance=0.5,
        max_frames=10,
        epsilon_multiplier=1.0,
        verbose=True,
    ):
        """
        Extract single molecules from multiple localization files (FOVs) with tracking.

        Processes multiple FOV files and combines them into unified databases with
        FOV tracking columns. Each molecule gets a globally unique molecular_index.

        Args:
            localisation_files (list): List of HDF5 localization file paths
            clustering_method (str): "HDBSCAN", "DBSCAN", or "linked"
            min_cluster_size (int): Minimum cluster size for HDBSCAN/DBSCAN
            chi_val (float, optional): Chi-squared threshold. Defaults to median.
            max_localization_error (float): Maximum localization precision in pixels
            min_photons (int): Minimum total photon count
            max_photons (float): Maximum total photon count
            max_distance (float): Maximum distance for linked method (pixels)
            max_frames (int): Maximum frame gap for linked method
            epsilon_multiplier (float): Multiplicative factor for epsilon in DBSCAN clustering.
                                       Only used when clustering_method="DBSCAN". Defaults to 1.0.
            verbose (bool): Print progress information

        Returns:
            tuple: (single_molecule_database, single_frame_database)
                Both DataFrames include fov_index and fov_name columns.
                molecular_index is globally unique across all FOVs.

        Example:
            >>> sm_db, sf_db = SM_E.extract_single_molecules_batch(
            ...     localisation_files,
            ...     clustering_method="HDBSCAN",
            ...     verbose=True
            ... )
            >>> # Filter by FOV
            >>> fov0_molecules = sm_db[sm_db['fov_name'] == 'Pos0']
        """
        if not isinstance(localisation_files, (list, np.ndarray)):
            raise ValueError("localisation_files must be a list or array of file paths")

        molecular_index_offset = 0
        all_molecule_dbs = []
        all_frame_dbs = []

        for fov_idx, loc_file in enumerate(localisation_files):
            if verbose:
                print(
                    f"Processing FOV {fov_idx + 1}/{len(localisation_files)} ({os.path.basename(loc_file)})...",
                    end="",
                    flush=True,
                )

            # Load localization data
            loc_data = pd.read_hdf(loc_file)

            if verbose:
                print(f" {len(loc_data)} localizations...", end="", flush=True)

            # Extract FOV name from filename
            fov_name = self._extract_fov_name(loc_file)
            if fov_name is None:
                fov_name = f"fov_{fov_idx}"
                if verbose:
                    print(f" (no Pos pattern, using {fov_name})...", end="", flush=True)

            # Apply chosen clustering method
            if clustering_method.upper() == "HDBSCAN":
                sm_db, sf_db = self.extract_single_molecules_HDBSCAN(
                    loc_data,
                    min_cluster_size=min_cluster_size,
                    chi_val=chi_val,
                    max_localisation_error=max_localisation_error,
                    max_colour_error=max_colour_error,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    max_sigma_error=max_sigma_error,
                    min_photons=min_photons,
                    max_photons=max_photons,
                )
            elif clustering_method.upper() == "DBSCAN":
                sm_db, sf_db = self.extract_single_molecules_DBSCAN(
                    loc_data,
                    min_cluster_size=min_cluster_size,
                    chi_val=chi_val,
                    max_localisation_error=max_localisation_error,
                    max_colour_error=max_colour_error,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    max_sigma_error=max_sigma_error,
                    min_photons=min_photons,
                    max_photons=max_photons,
                    epsilon_multiplier=epsilon_multiplier,
                )
            elif clustering_method.lower() == "linked":
                sm_db, sf_db = self.extract_single_molecules_linked(
                    loc_data,
                    max_distance=max_distance,
                    max_frames=max_frames,
                    chi_val=chi_val,
                    max_localisation_error=max_localisation_error,
                    max_colour_error=max_colour_error,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    max_sigma_error=max_sigma_error,
                    min_photons=min_photons,
                    max_photons=max_photons,
                )
            else:
                raise ValueError(
                    f"Unknown clustering_method: {clustering_method}. "
                    f"Choose from 'HDBSCAN', 'DBSCAN', or 'linked'"
                )

            if verbose:
                print(f" Found {len(sm_db)} molecules")

            # Skip empty results (from FOVs with insufficient data)
            if len(sm_db) == 0:
                continue

            # Add FOV tracking columns
            sm_db["fov_index"] = fov_idx
            sm_db["fov_name"] = fov_name
            sf_db["fov_index"] = fov_idx
            sf_db["fov_name"] = fov_name

            # Update molecular_index to be globally unique
            sm_db["molecular_index"] = sm_db["molecular_index"] + molecular_index_offset
            sf_db["molecular_index"] = sf_db["molecular_index"] + molecular_index_offset

            # Append to master lists
            all_molecule_dbs.append(sm_db)
            all_frame_dbs.append(sf_db)

            # Update offset for next FOV
            molecular_index_offset += len(sm_db)

        # Combine all FOVs
        if verbose:
            print("\nCombining databases...")

        # Handle case where all FOVs had insufficient data
        if len(all_molecule_dbs) == 0:
            if verbose:
                print("Warning: No molecules found in any FOV. Returning empty databases.")
            return pd.DataFrame(), pd.DataFrame()

        single_molecule_database = pd.concat(all_molecule_dbs, ignore_index=True)
        single_frame_database = pd.concat(all_frame_dbs, ignore_index=True)

        if verbose:
            print(f"Complete! Summary:")
            print(f"  - Total FOVs: {len(localisation_files)}")
            print(f"  - Total molecules: {len(single_molecule_database)}")
            print(f"  - Total localizations: {len(single_frame_database)}")

        return single_molecule_database, single_frame_database

    def build_photon_accumulation_database(
        self, single_frame_database, verbose=True
    ):
        """
        Build cumulative photon accumulation database for precision analysis.

        For each molecule, creates a cumulative sum trace showing how A_R, A_G, A_B
        evolve as more frames are accumulated. This enables analysis of parameter
        precision vs. photon count.

        Args:
            single_frame_database (pd.DataFrame): Output from extract_single_molecules_*
                Must contain: molecular_index, frame, A_R, A_G, A_B, photons,
                A_R_err, A_G_err, A_B_err, xc, yc
            verbose (bool): Print progress information

        Returns:
            pd.DataFrame: Photon accumulation database with columns:
                - molecular_index: Unique molecule ID
                - fov_index: FOV identifier (if present in input)
                - fov_name: FOV name (if present in input)
                - frames_accumulated: Number of frames in cumsum (1, 2, 3, ...)
                - photons_accumulated: Total photons accumulated
                - A_R, A_G, A_B: Normalized cumulative RGB (sum = 1.0)
                - A_R_err, A_G_err, A_B_err: Propagated errors
                - xc_mean, yc_mean: Average position
                - xc_std, yc_std: Position standard deviation
                - s_x_mean, s_y_mean: Average PSF widths (if available)

        Example:
            >>> photon_db = SM_E.build_photon_accumulation_database(single_frame_database)
            >>> # Get molecules with 1000-1100 accumulated photons
            >>> filtered = photon_db[
            ...     (photon_db['photons_accumulated'] >= 1000) &
            ...     (photon_db['photons_accumulated'] < 1100)
            ... ]
        """
        # Check required columns
        required_cols = ["molecular_index", "frame", "A_R", "A_G", "A_B", "photons"]
        missing_cols = [col for col in required_cols if col not in single_frame_database.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Get unique molecules
        unique_molecules = np.sort(single_frame_database["molecular_index"].unique())

        if verbose:
            print(f"Building photon accumulation database for {len(unique_molecules)} molecules...")

        # Storage for accumulation data
        accumulation_records = []

        for mol_idx in unique_molecules:
            # Extract all frames for this molecule
            mol_data = single_frame_database[
                single_frame_database["molecular_index"] == mol_idx
            ].copy()

            # Sort by frame number (chronological order)
            mol_data = mol_data.sort_values("frame")

            # Get FOV information if available
            fov_index = mol_data["fov_index"].iloc[0] if "fov_index" in mol_data.columns else None
            fov_name = mol_data["fov_name"].iloc[0] if "fov_name" in mol_data.columns else None

            n_frames = len(mol_data)

            if verbose and (mol_idx % 100 == 0 or mol_idx == unique_molecules[0]):
                print(
                    f"  Molecule {mol_idx + 1}/{len(unique_molecules)} ({n_frames} frames)",
                    end="\r",
                    flush=True,
                )

            # Build cumulative arrays using inverse-variance weighted averaging
            # IMPORTANT: A_R, A_G, A_B are already normalized per-frame (sum to 1.0)
            # We use inverse-variance weighting: frames with lower error contribute more
            cumsum_photons = np.cumsum(mol_data["photons"].values)
            frames_range = np.arange(1, n_frames + 1)

            if "A_R_err" in mol_data.columns:
                # Inverse-variance weighted cumulative average
                # weight_i = 1 / σ_i²
                # weighted_mean = Σ(w_i * x_i) / Σ(w_i)
                # error = 1 / sqrt(Σ(w_i))

                A_R_vals = mol_data["A_R"].values
                A_G_vals = mol_data["A_G"].values
                A_B_vals = mol_data["A_B"].values

                A_R_errs = mol_data["A_R_err"].values
                A_G_errs = mol_data["A_G_err"].values
                A_B_errs = mol_data["A_B_err"].values

                # Avoid division by zero - use small epsilon
                eps = 1e-12
                A_R_weights = 1.0 / (A_R_errs**2 + eps)
                A_G_weights = 1.0 / (A_G_errs**2 + eps)
                A_B_weights = 1.0 / (A_B_errs**2 + eps)

                # Cumulative weighted sums
                cumsum_A_R_weighted = np.cumsum(A_R_weights * A_R_vals)
                cumsum_A_G_weighted = np.cumsum(A_G_weights * A_G_vals)
                cumsum_A_B_weighted = np.cumsum(A_B_weights * A_B_vals)

                cumsum_A_R_weights = np.cumsum(A_R_weights)
                cumsum_A_G_weights = np.cumsum(A_G_weights)
                cumsum_A_B_weights = np.cumsum(A_B_weights)

                # Weighted mean at each accumulation step
                A_R_mean = cumsum_A_R_weighted / cumsum_A_R_weights
                A_G_mean = cumsum_A_G_weighted / cumsum_A_G_weights
                A_B_mean = cumsum_A_B_weighted / cumsum_A_B_weights

                # Error of weighted mean: 1 / sqrt(sum of weights)
                A_R_mean_err = 1.0 / np.sqrt(cumsum_A_R_weights)
                A_G_mean_err = 1.0 / np.sqrt(cumsum_A_G_weights)
                A_B_mean_err = 1.0 / np.sqrt(cumsum_A_B_weights)

            else:
                # Fallback: simple cumulative average if no errors available
                cumsum_A_R = np.cumsum(mol_data["A_R"].values)
                cumsum_A_G = np.cumsum(mol_data["A_G"].values)
                cumsum_A_B = np.cumsum(mol_data["A_B"].values)

                A_R_mean = cumsum_A_R / frames_range
                A_G_mean = cumsum_A_G / frames_range
                A_B_mean = cumsum_A_B / frames_range

                A_R_mean_err = np.zeros(n_frames)
                A_G_mean_err = np.zeros(n_frames)
                A_B_mean_err = np.zeros(n_frames)

            # Calculate position statistics
            xc_cumsum = np.cumsum(mol_data["xc"].values)
            yc_cumsum = np.cumsum(mol_data["yc"].values)

            xc_mean = xc_cumsum / frames_range
            yc_mean = yc_cumsum / frames_range

            # Calculate cumulative std dev
            # Using Welford's method for numerical stability
            xc_std = np.zeros(n_frames)
            yc_std = np.zeros(n_frames)
            for i in range(n_frames):
                if i > 0:
                    xc_std[i] = np.std(mol_data["xc"].values[: i + 1])
                    yc_std[i] = np.std(mol_data["yc"].values[: i + 1])

            # PSF widths if available
            if "s_x" in mol_data.columns:
                s_x_cumsum = np.cumsum(mol_data["s_x"].values)
                s_y_cumsum = np.cumsum(mol_data["s_y"].values)
                s_x_mean = s_x_cumsum / frames_range
                s_y_mean = s_y_cumsum / frames_range
            else:
                s_x_mean = np.zeros(n_frames)
                s_y_mean = np.zeros(n_frames)

            # Create records for each accumulated frame
            for i in range(n_frames):
                record = {
                    "molecular_index": mol_idx,
                    "frames_accumulated": i + 1,
                    "photons_accumulated": cumsum_photons[i],
                    "A_R": A_R_mean[i],
                    "A_G": A_G_mean[i],
                    "A_B": A_B_mean[i],
                    "A_R_err": A_R_mean_err[i],
                    "A_G_err": A_G_mean_err[i],
                    "A_B_err": A_B_mean_err[i],
                    "xc_mean": xc_mean[i],
                    "yc_mean": yc_mean[i],
                    "xc_std": xc_std[i],
                    "yc_std": yc_std[i],
                    "s_x_mean": s_x_mean[i],
                    "s_y_mean": s_y_mean[i],
                }

                # Add FOV info if available
                if fov_index is not None:
                    record["fov_index"] = fov_index
                if fov_name is not None:
                    record["fov_name"] = fov_name

                accumulation_records.append(record)

        if verbose:
            print("\n  Converting to DataFrame...")

        # Convert to DataFrame
        photon_accumulation_db = pd.DataFrame(accumulation_records)

        if verbose:
            total_rows = len(photon_accumulation_db)
            avg_frames_per_mol = total_rows / len(unique_molecules)
            print(f"Complete! Photon accumulation database:")
            print(f"  - Total rows: {total_rows}")
            print(f"  - Average frames per molecule: {avg_frames_per_mol:.1f}")

        return photon_accumulation_db

    def analyse_multi_fov_dataset(
        self,
        localisation_files,
        clustering_method="HDBSCAN",
        build_accumulation=True,
        min_cluster_size=10,
        chi_val=None,
        max_localisation_error=1.0,
        max_colour_error=0.15,
        min_sigma=(75./69),
        max_sigma=(160./69),
        max_sigma_error=(40./69),
        min_photons=500,
        max_photons=1e6,
        max_distance=0.5,
        max_frames=10,
        epsilon_multiplier=0.5,
        output_folder=None,
        output_prefix="analysis",
        verbose=True,
    ):
        """
        Complete workflow: Extract single molecules from multiple FOVs and build databases.

        High-level wrapper that processes multiple FOV localization files, extracts
        single molecules, and optionally builds photon accumulation database.

        Args:
            localisation_files (list): List of HDF5 localization file paths
            clustering_method (str): "HDBSCAN", "DBSCAN", or "linked"
            build_accumulation (bool): Build photon accumulation database?
            min_cluster_size (int): Minimum cluster size for HDBSCAN/DBSCAN
            chi_val (float, optional): Chi-squared threshold. Defaults to median.
            max_localization_error (float): Maximum localization precision in pixels
            min_photons (int): Minimum total photon count
            max_photons (float): Maximum total photon count
            max_distance (float): Maximum distance for linked method (pixels)
            max_frames (int): Maximum frame gap for linked method
            epsilon_multiplier (float): Multiplicative factor for epsilon in DBSCAN clustering.
                                       Only used when clustering_method="DBSCAN". Defaults to 0.5.
            output_folder (str, optional): If provided, save databases to this folder
            output_prefix (str): Prefix for output filenames
            verbose (bool): Print progress information

        Returns:
            tuple: If build_accumulation=True: (single_molecule_db, single_frame_db, photon_accumulation_db)
                   If build_accumulation=False: (single_molecule_db, single_frame_db)

        Example:
            >>> sm_db, sf_db, pa_db = SM_E.analyze_multi_fov_dataset(
            ...     localisation_files,
            ...     clustering_method="HDBSCAN",
            ...     build_accumulation=True,
            ...     output_folder="./analysis_output",
            ...     output_prefix="dye_mixture"
            ... )
            >>> # Save to disk automatically creates:
            >>> # - dye_mixture_single_molecules.h5
            >>> # - dye_mixture_single_frames.h5
            >>> # - dye_mixture_photon_accumulation.h5
        """
        if verbose:
            print("=" * 60)
            print("Multi-FOV Single Molecule Analysis")
            print("=" * 60)

        # Step 1: Extract single molecules from all FOVs
        single_molecule_db, single_frame_db = self.extract_single_molecules_batch(
            localisation_files=localisation_files,
            clustering_method=clustering_method,
            min_cluster_size=min_cluster_size,
            chi_val=chi_val,
            max_localisation_error=max_localisation_error,
            max_colour_error=max_colour_error,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            max_sigma_error=max_sigma_error,
            min_photons=min_photons,
            max_photons=max_photons,
            max_distance=max_distance,
            max_frames=max_frames,
            epsilon_multiplier=epsilon_multiplier,
            verbose=verbose,
        )

        # Step 2: Build photon accumulation database if requested
        photon_accumulation_db = None
        if build_accumulation:
            if verbose:
                print()
            photon_accumulation_db = self.build_photon_accumulation_database(
                single_frame_db, verbose=verbose
            )

        # Step 3: Save to disk if output folder specified
        if output_folder is not None:
            if verbose:
                print(f"\nSaving databases to {output_folder}...")

            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)

            # Save single molecule database
            # Note: Use pandas to_hdf directly since single_molecule_db doesn't have frame column
            sm_path = os.path.join(output_folder, f"{output_prefix}_single_molecules.h5")
            single_molecule_db.to_hdf(sm_path, key="data", mode="w", format="table")
            if verbose:
                print(f"  Saved: {os.path.basename(sm_path)}")

            # Save single frame database
            sf_path = os.path.join(output_folder, f"{output_prefix}_single_frames.h5")
            self.io._write_h5_database(
                single_frame_db, sf_path, normalise_photons=False, append=False, verbose=verbose
            )
            if verbose:
                print(f"  Saved: {os.path.basename(sf_path)}")

            # Save photon accumulation database if built
            if photon_accumulation_db is not None:
                pa_path = os.path.join(
                    output_folder, f"{output_prefix}_photon_accumulation.h5"
                )
                # Note: Use pandas to_hdf directly - this database doesn't have traditional frame column
                photon_accumulation_db.to_hdf(pa_path, key="data", mode="w", format="table")
                if verbose:
                    print(f"  Saved: {os.path.basename(pa_path)}")

        if verbose:
            print("\n" + "=" * 60)
            print("Analysis Complete!")
            print("=" * 60)

        # Return databases
        if build_accumulation:
            return single_molecule_db, single_frame_db, photon_accumulation_db
        else:
            return single_molecule_db, single_frame_db

    def _fit_gmm_mle(
        self,
        X,
        initial_means,
        n_components,
        covariance_type="full",
        max_iter=500,
        verbose=False,
    ):
        """
        Fit Gaussian Mixture Model using Maximum Likelihood Estimation (MLE).

        Uses scipy.optimize to directly minimize negative log-likelihood.
        This is more robust than EM when dealing with weighted data.

        Args:
            X (np.ndarray): Data points, shape (n_samples, n_features)
            initial_means (np.ndarray): Initial mean positions, shape (n_components, n_features)
            n_components (int): Number of mixture components
            covariance_type (str): Type of covariance ('full', 'tied', 'diag', 'spherical')
            max_iter (int): Maximum optimization iterations
            verbose (bool): Print optimization progress

        Returns:
            tuple: (means, covariances, weights, converged)
                - means (np.ndarray): Fitted means, shape (n_components, n_features)
                - covariances (np.ndarray): Fitted covariances, shape (n_components, n_features, n_features)
                - weights (np.ndarray): Component weights, shape (n_components,)
                - converged (bool): Whether optimization succeeded
        """
        from scipy.optimize import minimize

        def pack_params(means, covariances, weights):
            """Pack GMM parameters into flat array."""
            params = []
            # Means
            params.extend(means.flatten())
            # Covariances (upper triangular elements)
            for k in range(n_components):
                cov = covariances[k]
                params.extend([cov[0, 0], cov[0, 1], cov[1, 1]])
            # Weights (n-1 to maintain sum=1 constraint)
            params.extend(weights[:-1])
            return np.array(params)

        def unpack_params(params):
            """Unpack flat array into GMM parameters."""
            idx = 0
            # Means
            means = params[:n_components * 2].reshape(n_components, 2)
            idx += n_components * 2
            # Covariances
            covariances = []
            for k in range(n_components):
                cov = np.array([
                    [params[idx], params[idx + 1]],
                    [params[idx + 1], params[idx + 2]]
                ])
                covariances.append(cov)
                idx += 3
            # Weights
            weights = np.zeros(n_components)
            weights[:-1] = params[idx:]
            weights[-1] = 1.0 - weights[:-1].sum()
            return means, np.array(covariances), weights

        def negative_log_likelihood(params):
            """Negative log-likelihood for minimization."""
            try:
                means, covariances, weights = unpack_params(params)

                # Check validity
                for k in range(n_components):
                    eigvals = np.linalg.eigvalsh(covariances[k])
                    if np.any(eigvals <= 0):
                        return 1e10
                if np.any(weights <= 0) or np.any(weights >= 1):
                    return 1e10

                # Calculate log-likelihood
                log_probs = np.zeros((len(X), n_components))
                for k in range(n_components):
                    mvn = multivariate_normal(mean=means[k], cov=covariances[k])
                    log_probs[:, k] = mvn.logpdf(X) + np.log(weights[k])

                log_probs_max = log_probs.max(axis=1, keepdims=True)
                log_likelihood = np.sum(log_probs_max + np.log(np.exp(log_probs - log_probs_max).sum(axis=1)))

                return -log_likelihood

            except (np.linalg.LinAlgError, ValueError):
                return 1e10

        # Initialize with sklearn GMM
        gmm_init = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=50,
            n_init=1,
            means_init=initial_means,
        )
        gmm_init.fit(X)

        # Pack initial parameters
        params_init = pack_params(gmm_init.means_, gmm_init.covariances_, gmm_init.weights_)

        if verbose:
            print(f"  Running MLE optimization (L-BFGS-B)...")

        # Optimize with MLE
        result = minimize(
            negative_log_likelihood,
            params_init,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-6}
        )

        # Unpack optimized parameters
        means_opt, covariances_opt, weights_opt = unpack_params(result.x)

        return means_opt, covariances_opt, weights_opt, result.success

    def _fit_gmm_em(
        self,
        X,
        initial_means,
        n_components,
        covariance_type="full",
        max_iter=100,
        n_reweighting_iterations=2,
        photons=None,
        A_R=None,
        A_G=None,
        has_error_columns=False,
        sigma_A_R=None,
        sigma_A_G=None,
        verbose=False,
    ):
        """
        Fit Gaussian Mixture Model using Expectation-Maximization (EM) with iterative re-weighting.

        This approach normalizes weights within each component to handle population imbalance better.

        Args:
            X (np.ndarray): Data points, shape (n_samples, n_features)
            initial_means (np.ndarray): Initial mean positions, shape (n_components, n_features)
            n_components (int): Number of mixture components
            covariance_type (str): Type of covariance ('full', 'tied', 'diag', 'spherical')
            max_iter (int): Maximum EM iterations
            n_reweighting_iterations (int): Number of re-weighting iterations (default: 2)
            photons (np.ndarray): Photon counts for weighting (optional)
            A_R (np.ndarray): A_R values for uncertainty calculation (optional)
            A_G (np.ndarray): A_G values for uncertainty calculation (optional)
            has_error_columns (bool): Whether error columns are available
            sigma_A_R (np.ndarray): A_R errors (optional)
            sigma_A_G (np.ndarray): A_G errors (optional)
            verbose (bool): Print progress

        Returns:
            tuple: (means, covariances, weights, converged)
                - means (np.ndarray): Fitted means, shape (n_components, n_features)
                - covariances (np.ndarray): Fitted covariances, shape (n_components, n_features, n_features)
                - weights (np.ndarray): Component weights, shape (n_components,)
                - converged (bool): Whether EM converged
        """
        # Initial fit using histogram-based initialization
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            n_init=1,
            means_init=initial_means,
        )
        gmm.fit(X)

        # Iterative re-weighting
        for iteration in range(n_reweighting_iterations):
            if verbose:
                print(f"  Re-weighting iteration {iteration + 1}/{n_reweighting_iterations}...")

            # Get component assignments on original data
            labels = gmm.predict(X)

            # Build within-component weighted dataset
            X_reweighted_list = []
            for k in range(n_components):
                component_mask = labels == k
                n_in_component = component_mask.sum()

                if n_in_component == 0:
                    continue

                component_data = X[component_mask]

                # Apply error-based weighting within this component only
                if photons is not None and A_R is not None and A_G is not None:
                    component_photons = photons[component_mask]
                    component_A_R = A_R[component_mask]
                    component_A_G = A_G[component_mask]

                    # Calculate uncertainty for this component
                    if has_error_columns and sigma_A_R is not None and sigma_A_G is not None:
                        component_sigma_A_R = sigma_A_R[component_mask]
                        component_sigma_A_G = sigma_A_G[component_mask]
                        sigma_combined_k = np.sqrt(component_sigma_A_R**2 + component_sigma_A_G**2)
                    else:
                        sigma_A_R_k = np.sqrt(component_A_R * (1 - component_A_R) / (component_photons + 1))
                        sigma_A_G_k = np.sqrt(component_A_G * (1 - component_A_G) / (component_photons + 1))
                        sigma_combined_k = np.sqrt(sigma_A_R_k**2 + sigma_A_G_k**2)

                    # Weight = 1/sigma
                    weights_k = 1.0 / (sigma_combined_k + 1e-10)
                    weights_k = weights_k / weights_k.mean()

                    # Convert to integer replication counts
                    rep_counts_k = np.round(weights_k * 10).astype(int)
                    rep_counts_k = np.maximum(rep_counts_k, 1)

                    X_reweighted_list.append(np.repeat(component_data, rep_counts_k, axis=0))
                else:
                    # No photon weighting - just use the data as is
                    X_reweighted_list.append(component_data)

            # Combine all components
            if len(X_reweighted_list) > 0:
                X_balanced = np.vstack(X_reweighted_list)
            else:
                X_balanced = X

            # Refit GMM with balanced weights
            gmm.fit(X_balanced)

        return gmm.means_, gmm.covariances_, gmm.weights_, gmm.converged_

    def _fit_gmm_pygmmis(
        self,
        X,
        X_err,
        initial_means,
        n_components,
        max_iter=100,
        verbose=False,
    ):
        """
        Fit Gaussian Mixture Model using pygmmis Extreme Deconvolution.

        This method properly handles per-point measurement uncertainties by deconvolving
        measurement noise from the intrinsic distribution. This is the theoretically correct
        approach for SMLM data where each localization has its own fitting uncertainty.

        Unlike point replication methods, Extreme Deconvolution:
        - Treats measurement covariances as part of the model
        - Deconvolves noise to recover the true error-free distribution
        - Scales to millions of points without replication overhead
        - Provides proper uncertainty quantification

        Args:
            X (np.ndarray): Data points, shape (n_samples, n_features)
            X_err (np.ndarray): Per-point uncertainties, shape (n_samples, n_features)
                These should be the standard errors (sigma), not variances.
            initial_means (np.ndarray): Initial mean positions, shape (n_components, n_features)
            n_components (int): Number of mixture components
            max_iter (int): Maximum iterations for extreme deconvolution (default: 100)
            verbose (bool): Print progress

        Returns:
            tuple: (means, covariances, weights, converged)
                - means (np.ndarray): Fitted means, shape (n_components, n_features)
                - covariances (np.ndarray): Fitted covariances, shape (n_components, n_features, n_features)
                - weights (np.ndarray): Component weights, shape (n_components,)
                - converged (bool): Whether fitting converged

        References:
            Bovy, Hogg & Roweis (2011) "Extreme deconvolution: Inferring complete
            distribution functions from noisy, heterogeneous and incomplete observations"
            https://github.com/pmelchior/pygmmis
        """
        try:
            import pygmmis
        except ImportError:
            raise ImportError(
                "pygmmis is required for extreme deconvolution fitting. "
                "Install with: pip install pygmmis"
            )

        n_samples, n_features = X.shape

        if verbose:
            print(f"  Extreme Deconvolution fitting with pygmmis...")
            print(f"    Data: {n_samples} points, {n_features} features")
            print(f"    Components: {n_components}")
            print(f"    Mean errors: {X_err.mean(axis=0)}")

        # Prepare per-point covariance matrices (diagonal, since A_R and A_G errors are independent)
        covar = np.zeros((n_samples, n_features, n_features))
        for i in range(n_samples):
            # Convert standard errors to variances (sigma^2)
            covar[i] = np.diag(X_err[i]**2)

        # Initialize GMM with k-means or provided means
        gmm = pygmmis.GMM(K=n_components, D=n_features)

        # Set initial means
        gmm.mean = initial_means.copy()

        # Initialize covariances using simple empirical estimate
        # Assign each point to nearest mean and estimate covariance
        from scipy.spatial.distance import cdist
        distances = cdist(X, initial_means)
        labels = np.argmin(distances, axis=1)

        initial_covariances = np.zeros((n_components, n_features, n_features))
        initial_weights = np.zeros(n_components)

        for k in range(n_components):
            mask = labels == k
            n_k = mask.sum()

            if n_k > n_features:  # Need enough points to estimate covariance
                X_k = X[mask]
                # Empirical covariance
                diff = X_k - initial_means[k]
                cov_k = (diff.T @ diff) / n_k
                # Add regularization to ensure positive definite
                cov_k += np.eye(n_features) * 1e-4
                initial_covariances[k] = cov_k
                initial_weights[k] = n_k / n_samples
            else:
                # Not enough points, use identity
                initial_covariances[k] = np.eye(n_features) * 0.01
                initial_weights[k] = 1.0 / n_components

        # Ensure weights sum to 1
        initial_weights /= initial_weights.sum()

        gmm.covar = initial_covariances
        gmm.amp = initial_weights

        if verbose:
            print(f"    Initial weights: {gmm.amp}")
            print(f"    Running extreme deconvolution (max_iter={max_iter})...")

        # Run extreme deconvolution
        # pygmmis returns log-likelihood and component assignments
        try:
            logL, U = pygmmis.fit(
                gmm,
                data=X,
                covar=covar,
                init_method='none',   # Use our provided initialization
                w=1e-6,              # Covariance regularization (small value)
                cutoff=5.0,          # Mahalanobis distance cutoff for outliers
                maxiter=max_iter,
                tol=1e-3,            # Convergence tolerance
            )

            converged = True  # pygmmis doesn't explicitly report convergence

            if verbose:
                print(f"    Final log-likelihood: {logL:.2f}")
                print(f"    Final weights: {gmm.amp}")

        except Exception as e:
            if verbose:
                print(f"    Warning: Extreme deconvolution failed: {e}")
                print(f"    Returning initial parameters")
            converged = False

        # Extract results
        means = gmm.mean.copy()
        covariances = gmm.covar.copy()
        weights = gmm.amp.copy()

        return means, covariances, weights, converged

    def extract_reference_means(
        self,
        data_db,
        reference_photon_threshold=None,
        n_components=2,
        covariance_type="full",
        fit_type="MLE",
        random_state=42,
        verbose=True,
    ):
        """
        Extract reference means from molecule data using GMM (analytical approach).

        Fits a Gaussian Mixture Model to establish reference mean positions for each dye
        population. These fixed means are then used with photon-dependent covariance fitting
        to analytically calculate misidentification rates.

        This is the first step in the analytical approach:
        1. Extract means from reference data (this function)
        2. Fit covariances at each photon level with fixed means
        3. Analytically calculate overlap/error rates from distributions

        Two modes of operation:
        A. **Photon accumulation database + threshold**: Use highest-photon data only
           - Pass photon_accumulation_db with reference_photon_threshold
           - Extracts molecules reaching threshold for stable mean estimates

        B. **Single molecule database**: Use all molecules (no threshold)
           - Pass single_molecule_database with reference_photon_threshold=None
           - Uses averaged RGB values from all molecules

        Args:
            data_db (pd.DataFrame): Either:
                - Photon accumulation database (with 'photons_accumulated' column)
                - Single molecule database (with 'A_R', 'A_G', 'A_B' columns)
            reference_photon_threshold (float, optional): Minimum photons for reference.
                - If provided: Filters accumulation DB to molecules >= threshold
                - If None: Uses all molecules (assumes single molecule database)
                (default: None - use all molecules)
            n_components (int): Number of Gaussian components (default: 2)
            covariance_type (str): GMM covariance type - "full", "tied", "diag", "spherical"
                (default: "full" allows correlation between A_R and A_G)
            fit_type (str): Fitting algorithm - "MLE" or "EM" (default: "MLE")
                - "MLE": Direct maximum likelihood via scipy.optimize (more robust)
                - "EM": Iterative EM with within-component re-weighting (faster, handles imbalance)
            random_state (int): Random seed for reproducibility
            verbose (bool): Print progress and statistics

        Returns:
            tuple: (reference_means, reference_db, gmm_model)
                - reference_means (np.ndarray): Shape (n_components, 2) - fixed means for [A_R, A_G]
                - reference_db (pd.DataFrame): Reference molecules with assignments:
                    - molecular_index: Unique molecule ID
                    - true_label: 0 or 1 (e.g., 0=Red, 1=Green)
                    - A_R_ref, A_G_ref, A_B_ref: Reference RGB values
                    - posterior_prob_0, posterior_prob_1: Posterior probabilities
                    - photons (if from single molecule DB) or max_photons (if from accumulation DB)
                    - fov_index, fov_name: FOV tracking (if available)
                - gmm_model (GaussianMixture): Fitted GMM (for reference only)

        Examples:
            >>> # Mode A: High-photon data from accumulation database
            >>> means, ref_db, gmm = SM_E.extract_reference_means(
            ...     pa_db,
            ...     reference_photon_threshold=200000,
            ...     verbose=True
            ... )

            >>> # Mode B: All molecules from single molecule database
            >>> means, ref_db, gmm = SM_E.extract_reference_means(
            ...     sm_db,
            ...     reference_photon_threshold=None,
            ...     verbose=True
            ... )

            >>> print("Fixed mean positions:")
            >>> print(f"  Component 0: A_R={means[0,0]:.3f}, A_G={means[0,1]:.3f}")
            >>> print(f"  Component 1: A_R={means[1,0]:.3f}, A_G={means[1,1]:.3f}")
        """
        if verbose:
            print("=" * 60)
            print("Extracting Reference Means (Analytical Approach)")
            print("=" * 60)

        # Detect database type and extract reference data accordingly
        is_photon_accumulation_db = "photons_accumulated" in data_db.columns

        if is_photon_accumulation_db:
            # Mode A: Photon accumulation database with threshold
            if reference_photon_threshold is None:
                raise ValueError(
                    "reference_photon_threshold must be provided when using photon accumulation database. "
                    "To use all molecules, pass a single molecule database instead."
                )

            if verbose:
                print("Mode: Photon Accumulation Database")
                print(f"Using highest-photon data (threshold: {reference_photon_threshold:,.0f})")

            # Get maximum photons accumulated for each molecule
            max_photons_per_mol = (
                data_db.groupby("molecular_index")["photons_accumulated"]
                .max()
                .reset_index()
            )
            max_photons_per_mol.columns = ["molecular_index", "max_photons"]

            if verbose:
                print(f"Total molecules in database: {len(max_photons_per_mol)}")

            # Filter molecules that reach reference threshold
            qualified_molecules = max_photons_per_mol[
                max_photons_per_mol["max_photons"] >= reference_photon_threshold
            ]

            if verbose:
                n_qualified = len(qualified_molecules)
                n_total = len(max_photons_per_mol)
                pct_qualified = 100 * n_qualified / n_total
                print(f"Molecules reaching threshold: {n_qualified}/{n_total} ({pct_qualified:.1f}%)")

            if len(qualified_molecules) == 0:
                raise ValueError(
                    f"No molecules reach photon threshold {reference_photon_threshold}. "
                    f"Maximum photons in dataset: {max_photons_per_mol['max_photons'].max():.0f}"
                )

            # Get data at maximum photons for each qualified molecule
            reference_data = []
            for mol_idx in qualified_molecules["molecular_index"]:
                mol_data = data_db[data_db["molecular_index"] == mol_idx]
                # Get row with maximum photons
                max_row = mol_data.loc[mol_data["photons_accumulated"].idxmax()]
                reference_data.append(max_row)

            reference_df = pd.DataFrame(reference_data).reset_index(drop=True)
            photon_column = "photons_accumulated"  # For later reference

        else:
            # Mode B: Single molecule database
            if verbose:
                print("Mode: Single Molecule Database")
                if reference_photon_threshold is not None:
                    print(f"Filtering molecules with photons >= {reference_photon_threshold:,.0f}")
                else:
                    print("Using all molecules (no photon threshold)")

            # Check required columns
            required_cols = ["A_R", "A_G", "A_B"]
            missing_cols = [col for col in required_cols if col not in data_db.columns]
            if missing_cols:
                raise ValueError(
                    f"Single molecule database missing required columns: {missing_cols}. "
                    f"Available columns: {list(data_db.columns)}"
                )

            # Apply photon threshold if provided
            if reference_photon_threshold is not None:
                if "photons" not in data_db.columns:
                    raise ValueError(
                        "reference_photon_threshold provided but 'photons' column not found in database. "
                        "Either provide a database with 'photons' column or set reference_photon_threshold=None."
                    )
                reference_df = data_db[data_db["photons"] >= reference_photon_threshold].copy()

                if verbose:
                    n_qualified = len(reference_df)
                    n_total = len(data_db)
                    pct_qualified = 100 * n_qualified / n_total if n_total > 0 else 0
                    print(f"Total molecules in database: {n_total}")
                    print(f"Molecules passing threshold: {n_qualified}/{n_total} ({pct_qualified:.1f}%)")

                if len(reference_df) == 0:
                    raise ValueError(
                        f"No molecules have photons >= {reference_photon_threshold}. "
                        f"Maximum photons in dataset: {data_db['photons'].max():.0f}"
                    )
            else:
                reference_df = data_db.copy()
                if verbose:
                    print(f"Total molecules in database: {len(reference_df)}")

            photon_column = "photons" if "photons" in reference_df.columns else None

        if verbose:
            print(f"\nFitting {n_components}-component Gaussian Mixture Model...")
            print(f"  Covariance type: {covariance_type}")
            print(f"  Features: A_R, A_G")

        # Prepare data for GMM: (A_R, A_G) coordinates
        X = reference_df[["A_R", "A_G"]].values

        # Apply error-based weighting if available
        # Weight by inverse of uncertainty in A_R and A_G measurements
        # We do this by replicating high-precision samples more times

        # Check if error columns exist in the database
        has_error_columns = "A_R_err" in reference_df.columns and "A_G_err" in reference_df.columns

        if has_error_columns:
            # Use pre-calculated errors from database
            sigma_A_R = reference_df["A_R_err"].values
            sigma_A_G = reference_df["A_G_err"].values

            # Combined uncertainty (quadrature sum for 2D Gaussian)
            sigma_combined = np.sqrt(sigma_A_R**2 + sigma_A_G**2)

            # Weight = 1/sigma (inverse of uncertainty)
            weights = 1.0 / (sigma_combined + 1e-10)  # Small constant to avoid infinity
            weights = weights / weights.mean()  # Normalize

            # Store for later use in EM
            photons = reference_df[photon_column].values if photon_column is not None else None
            A_R = reference_df["A_R"].values
            A_G = reference_df["A_G"].values
            A_B = reference_df["A_B"].values if "A_B" in reference_df.columns else None

        elif photon_column is not None:
            # Fall back to calculating uncertainty from photon statistics
            photons = reference_df[photon_column].values

            # Get amplitude ratios
            A_R = reference_df["A_R"].values
            A_G = reference_df["A_G"].values
            A_B = reference_df["A_B"].values

            # Uncertainty in A_R = R/(R+G+B) using error propagation
            # σ(A_R) ≈ sqrt(A_R*(1-A_R)/total_photons) for Poisson statistics

            sigma_A_R = np.sqrt(A_R * (1 - A_R) / (photons + 1))  # +1 to avoid division by zero
            sigma_A_G = np.sqrt(A_G * (1 - A_G) / (photons + 1))

            # Combined uncertainty (quadrature sum for 2D Gaussian)
            sigma_combined = np.sqrt(sigma_A_R**2 + sigma_A_G**2)

            # Weight = 1/sigma (inverse of uncertainty)
            weights = 1.0 / (sigma_combined + 1e-10)  # Small constant to avoid infinity
            weights = weights / weights.mean()  # Normalize
        else:
            # No weighting information available
            weights = None
            photons = None
            A_R = None
            A_G = None
            A_B = None
            sigma_combined = None

        if weights is not None:

            # Convert weights to integer replication counts
            # Use factor of 5 for MLE (faster) and factor of 10 for EM
            # MLE is more expensive per iteration so we use fewer replications
            replication_factor = 5 if fit_type.upper() == "MLE" else 10
            replication_counts = np.round(weights * replication_factor).astype(int)
            replication_counts = np.maximum(replication_counts, 1)  # At least 1 copy

            # Create weighted dataset by replicating samples
            X_weighted = np.repeat(X, replication_counts, axis=0)

            # For MLE, cap total samples at 3000 to keep optimization tractable
            if fit_type.upper() == "MLE" and len(X_weighted) > 3000:
                # Downsample while preserving weight distribution
                downsample_ratio = 3000 / len(X_weighted)
                new_rep_counts = np.maximum(np.round(replication_counts * downsample_ratio).astype(int), 1)
                X_weighted = np.repeat(X, new_rep_counts, axis=0)
                if verbose:
                    print(f"  MLE optimization: downsampled to {len(X_weighted)} samples for tractability")

            if verbose:
                if has_error_columns:
                    print(f"  Using error-weighted GMM fitting (1/σ weighting from A_R_err, A_G_err columns)")
                else:
                    print(f"  Using error-weighted GMM fitting (1/σ weighting calculated from photon statistics)")
                if photons is not None:
                    print(f"    Photon range: {photons.min():.0f} - {photons.max():.0f}")
                print(f"    Uncertainty range: σ={sigma_combined.min():.4f} - {sigma_combined.max():.4f}")
                print(f"    Weight range: {weights.min():.2f} - {weights.max():.2f}")
                print(f"    Replication range: {replication_counts.min()} - {replication_counts.max()} copies")
                print(f"    Original samples: {len(X)}, Weighted samples: {len(X_weighted)}")

            X_fit = X_weighted
        else:
            if verbose:
                print(f"  Using uniform weights (no photon column available)")
            X_fit = X

        # Validate fit_type parameter
        if fit_type.upper() not in ["MLE", "EM"]:
            raise ValueError(f"fit_type must be 'MLE' or 'EM', got '{fit_type}'")

        # Initialize means using histogram-based peak finding
        # This gives better initial guesses than k-means++
        from scipy.signal import find_peaks

        def fit_histogram_gaussians(data, n_peaks=2):
            """Fit Gaussians to histogram peaks to get initial guesses."""
            # Create histogram
            hist, bin_edges = np.histogram(data, bins=50, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Find peaks in histogram
            peaks, properties = find_peaks(hist, height=0, distance=5)

            if len(peaks) >= n_peaks:
                # Sort by height and take top n_peaks
                peak_heights = hist[peaks]
                top_peak_indices = np.argsort(peak_heights)[-n_peaks:]
                peak_positions = bin_centers[peaks[top_peak_indices]]
                return np.sort(peak_positions)
            else:
                # Fallback: use quantiles
                return np.quantile(data, np.linspace(0.2, 0.8, n_peaks))

        # Get initial means from histograms
        A_R_peaks = fit_histogram_gaussians(X[:, 0], n_components)
        A_G_peaks = fit_histogram_gaussians(X[:, 1], n_components)

        # Combine to form initial 2D means
        # Match peaks: if A_R high, A_G should be low (and vice versa)
        initial_means = np.zeros((n_components, 2))
        if n_components == 2:
            # Sort A_R peaks ascending, A_G peaks descending
            A_R_sorted = np.sort(A_R_peaks)
            A_G_sorted = np.sort(A_G_peaks)[::-1]
            initial_means[:, 0] = A_R_sorted
            initial_means[:, 1] = A_G_sorted
        else:
            # For n_components != 2, use a simpler approach
            initial_means[:, 0] = A_R_peaks
            initial_means[:, 1] = A_G_peaks

        if verbose:
            print(f"  Histogram-based initialization:")
            for i in range(n_components):
                print(f"    Component {i}: A_R={initial_means[i, 0]:.4f}, A_G={initial_means[i, 1]:.4f}")

        # Fit GMM using selected method
        if fit_type.upper() == "MLE":
            # Use helper function for MLE fitting
            means_opt, covariances_opt, weights_opt, converged = self._fit_gmm_mle(
                X_fit,
                initial_means,
                n_components,
                covariance_type=covariance_type,
                max_iter=500,
                verbose=verbose,
            )

            # Create GMM object with optimized parameters for compatibility
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                random_state=random_state,
            )
            gmm.means_ = means_opt
            gmm.covariances_ = covariances_opt
            gmm.weights_ = weights_opt
            gmm.converged_ = converged

            # Compute precisions_cholesky_ for sklearn compatibility
            # This is needed for predict() and predict_proba() methods
            gmm.precisions_cholesky_ = np.empty((n_components, 2, 2))
            for k in range(n_components):
                cov_chol = np.linalg.cholesky(covariances_opt[k])
                gmm.precisions_cholesky_[k] = np.linalg.solve(cov_chol, np.eye(2)).T

            # Predict labels and posterior probabilities on original unweighted data
            labels = gmm.predict(X)
            posteriors = gmm.predict_proba(X)

            if verbose:
                print(f"  Converged: {gmm.converged_}")
                print(f"  BIC: {gmm.bic(X):.2f}")
                print(f"  AIC: {gmm.aic(X):.2f}")
                print("\nGMM Component Parameters:")
                for i in range(n_components):
                    print(f"  Component {i}:")
                    print(f"    Mean A_R: {gmm.means_[i, 0]:.4f}")
                    print(f"    Mean A_G: {gmm.means_[i, 1]:.4f}")
                    print(f"    Weight: {gmm.weights_[i]:.4f}")
                    n_assigned = np.sum(labels == i)
                    print(f"    Molecules assigned: {n_assigned} ({100*n_assigned/len(labels):.1f}%)")

        elif fit_type.upper() == "EM":
            # Use helper function for EM fitting
            means_opt, covariances_opt, weights_opt, converged = self._fit_gmm_em(
                X,
                initial_means,
                n_components,
                covariance_type=covariance_type,
                max_iter=100,
                n_reweighting_iterations=2,
                photons=photons,
                A_R=A_R,
                A_G=A_G,
                has_error_columns=has_error_columns,
                sigma_A_R=sigma_A_R if has_error_columns else None,
                sigma_A_G=sigma_A_G if has_error_columns else None,
                verbose=verbose,
            )

            # Create GMM object with optimized parameters for compatibility
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                random_state=random_state,
            )
            gmm.means_ = means_opt
            gmm.covariances_ = covariances_opt
            gmm.weights_ = weights_opt
            gmm.converged_ = converged

            # Compute precisions_cholesky_ for sklearn compatibility
            gmm.precisions_cholesky_ = np.empty((n_components, 2, 2))
            for k in range(n_components):
                cov_chol = np.linalg.cholesky(covariances_opt[k])
                gmm.precisions_cholesky_[k] = np.linalg.solve(cov_chol, np.eye(2)).T

            # Predict labels and posterior probabilities on original data
            labels = gmm.predict(X)
            posteriors = gmm.predict_proba(X)

            if verbose:
                print(f"  Converged: {gmm.converged_}")
                print(f"  BIC: {gmm.bic(X):.2f}")
                print(f"  AIC: {gmm.aic(X):.2f}")
                print("\nGMM Component Parameters:")
                for i in range(n_components):
                    print(f"  Component {i}:")
                    print(f"    Mean A_R: {gmm.means_[i, 0]:.4f}")
                    print(f"    Mean A_G: {gmm.means_[i, 1]:.4f}")
                    print(f"    Weight: {gmm.weights_[i]:.4f}")
                    n_assigned = np.sum(labels == i)
                    print(f"    Molecules assigned: {n_assigned} ({100*n_assigned/len(labels):.1f}%)")


        # Build reference database - handle both modes
        ref_db_dict = {
            "true_label": labels,
            "A_R_ref": reference_df["A_R"].values,
            "A_G_ref": reference_df["A_G"].values,
            "A_B_ref": reference_df["A_B"].values,
            "posterior_prob_0": posteriors[:, 0],
            "posterior_prob_1": posteriors[:, 1],
        }

        # Add molecular_index if available
        if "molecular_index" in reference_df.columns:
            ref_db_dict["molecular_index"] = reference_df["molecular_index"].values

        # Add photon information based on database type
        if photon_column is not None:
            if is_photon_accumulation_db:
                ref_db_dict["max_photons"] = reference_df[photon_column].values
            else:
                ref_db_dict["photons"] = reference_df[photon_column].values

        reference_db = pd.DataFrame(ref_db_dict)

        # Add FOV tracking if available
        if "fov_index" in reference_df.columns:
            reference_db["fov_index"] = reference_df["fov_index"].values
        if "fov_name" in reference_df.columns:
            reference_db["fov_name"] = reference_df["fov_name"].values

        if verbose:
            print("\n" + "=" * 60)
            print("Reference Means Extraction Complete!")
            print("=" * 60)
            print(f"\nExtracted {n_components} fixed mean positions:")
            for i in range(n_components):
                print(f"  Component {i}: A_R={gmm.means_[i, 0]:.4f}, A_G={gmm.means_[i, 1]:.4f}")

            # Plot histograms with fitted means
            try:
                from PlottingBase import AnalysisPlotter

                plotter = AnalysisPlotter()
                fig, (ax1, ax2) = plotter.two_column_plot(nrows=1, ncols=2, height=3)

                # Define colors for each component
                colors = ['red', 'green', 'blue', 'orange', 'purple'][:n_components]

                # Plot A_R histogram
                ax1.hist(X[:, 0], bins=50, alpha=0.3, color='gray', label='All data')
                for i in range(n_components):
                    component_mask = labels == i
                    ax1.hist(X[component_mask, 0], bins=50, alpha=0.5,
                            color=colors[i], label=f'Component {i}')
                    ax1.axvline(gmm.means_[i, 0], color=colors[i], linestyle='--',
                              linewidth=2, label=f'Mean {i}: {gmm.means_[i, 0]:.3f}')
                ax1.set_xlabel('A_R')
                ax1.set_ylabel('Count')
                ax1.set_title('A_R Distribution with Fitted Means')
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)

                # Plot A_G histogram
                ax2.hist(X[:, 1], bins=50, alpha=0.3, color='gray', label='All data')
                for i in range(n_components):
                    component_mask = labels == i
                    ax2.hist(X[component_mask, 1], bins=50, alpha=0.5,
                            color=colors[i], label=f'Component {i}')
                    ax2.axvline(gmm.means_[i, 1], color=colors[i], linestyle='--',
                              linewidth=2, label=f'Mean {i}: {gmm.means_[i, 1]:.3f}')
                ax2.set_xlabel('A_G')
                ax2.set_ylabel('Count')
                ax2.set_title('A_G Distribution with Fitted Means')
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)

                _safe_tight_layout(fig)
                plotter.save_or_show(fig, save_path=None)  # Show only

                print("\n  (Close the plot window to continue)")

            except ImportError:
                print("\n  (Plotting skipped - PlottingBase not available)")
            except Exception as e:
                print(f"\n  (Plotting skipped - error: {e})")

        return gmm.means_, reference_db, gmm

    def fit_covariances_fixed_means(
        self,
        X,
        fixed_means,
        fit_type="EM",
        max_iter=100,
        tol=1e-6,
        verbose=False,
    ):
        """
        Fit covariance matrices with fixed means using either MLE or EM algorithm.

        This is the core of the analytical approach: given fixed mean positions
        from high-photon data, fit covariances at a specific photon level to
        characterize measurement uncertainty.

        Args:
            X (np.ndarray): Data points, shape (n_samples, n_features)
                Typically [:, 0] = A_R, [:, 1] = A_G
            fixed_means (np.ndarray): Fixed mean positions, shape (n_components, n_features)
            fit_type (str): Fitting method - "MLE" or "EM" (default: "EM")
                - "MLE": Direct maximum likelihood optimization (more robust)
                - "EM": Expectation-Maximization algorithm (faster)
            max_iter (int): Maximum iterations (default: 100)
            tol (float): Convergence tolerance for log-likelihood (default: 1e-6)
            verbose (bool): Print iteration progress

        Returns:
            tuple: (covariances, weights, converged)
                - covariances (np.ndarray): Shape (n_components, n_features, n_features)
                - weights (np.ndarray): Component weights, shape (n_components,)
                - converged (bool): Whether fitting converged

        Example:
            >>> # Get reference means from high-photon data
            >>> means, ref_db, _ = SM_E.extract_reference_means(pa_db, threshold=200000)
            >>>
            >>> # Fit covariances at lower photon level using MLE
            >>> low_photon_data = pa_db[pa_db['photons_accumulated'].between(5000, 6000)]
            >>> X = low_photon_data[['A_R', 'A_G']].values
            >>> cov, weights, converged = SM_E.fit_covariances_fixed_means(X, means, fit_type="MLE")
        """
        n_samples, n_features = X.shape
        n_components = len(fixed_means)

        if fit_type.upper() == "MLE":
            # Use MLE optimization for fixed means
            from scipy.optimize import minimize

            def pack_params_fixed_means(covariances, weights):
                """Pack covariances and weights (means are fixed)."""
                params = []
                # Covariances (upper triangular elements)
                for k in range(n_components):
                    cov = covariances[k]
                    params.extend([cov[0, 0], cov[0, 1], cov[1, 1]])
                # Weights (n-1 to maintain sum=1 constraint)
                params.extend(weights[:-1])
                return np.array(params)

            def unpack_params_fixed_means(params):
                """Unpack covariances and weights."""
                idx = 0
                # Covariances
                covariances = []
                for k in range(n_components):
                    cov = np.array([
                        [params[idx], params[idx + 1]],
                        [params[idx + 1], params[idx + 2]]
                    ])
                    covariances.append(cov)
                    idx += 3
                # Weights
                weights = np.zeros(n_components)
                weights[:-1] = params[idx:]
                weights[-1] = 1.0 - weights[:-1].sum()
                return np.array(covariances), weights

            def negative_log_likelihood_fixed(params):
                """Negative log-likelihood with fixed means."""
                try:
                    covariances, weights = unpack_params_fixed_means(params)

                    # Check validity
                    for k in range(n_components):
                        eigvals = np.linalg.eigvalsh(covariances[k])
                        if np.any(eigvals <= 0):
                            return 1e10
                    if np.any(weights <= 0) or np.any(weights >= 1):
                        return 1e10

                    # Calculate log-likelihood with fixed means
                    log_probs = np.zeros((len(X), n_components))
                    for k in range(n_components):
                        mvn = multivariate_normal(mean=fixed_means[k], cov=covariances[k])
                        log_probs[:, k] = mvn.logpdf(X) + np.log(weights[k])

                    log_probs_max = log_probs.max(axis=1, keepdims=True)
                    log_likelihood = np.sum(log_probs_max + np.log(np.exp(log_probs - log_probs_max).sum(axis=1)))

                    return -log_likelihood

                except (np.linalg.LinAlgError, ValueError):
                    return 1e10

            # Initialize covariances and weights
            covariances_init = np.array([np.eye(n_features) for _ in range(n_components)])
            weights_init = np.ones(n_components) / n_components

            params_init = pack_params_fixed_means(covariances_init, weights_init)

            if verbose:
                print(f"  Running MLE optimization with fixed means...")

            # Optimize
            result = minimize(
                negative_log_likelihood_fixed,
                params_init,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': tol}
            )

            covariances, weights = unpack_params_fixed_means(result.x)
            converged = result.success

            return covariances, weights, converged

        elif fit_type.upper() == "EM":
            # EM algorithm for fitting covariances with fixed means
            # Initialize covariances and weights
            covariances = [np.eye(n_features) for _ in range(n_components)]
            weights = np.ones(n_components) / n_components

            log_likelihood_old = -np.inf
            converged = False

            for iteration in range(max_iter):
                # E-step: Calculate responsibilities
                log_probs = np.zeros((n_samples, n_components))

                for k in range(n_components):
                    try:
                        mvn = multivariate_normal(mean=fixed_means[k], cov=covariances[k])
                        log_probs[:, k] = mvn.logpdf(X) + np.log(weights[k])
                    except np.linalg.LinAlgError:
                        # Singular covariance - add regularization
                        cov_reg = covariances[k] + 1e-6 * np.eye(n_features)
                        mvn = multivariate_normal(mean=fixed_means[k], cov=cov_reg)
                        log_probs[:, k] = mvn.logpdf(X) + np.log(weights[k])

                # Normalize to get responsibilities (log-sum-exp trick for stability)
                log_probs_max = log_probs.max(axis=1, keepdims=True)
                probs = np.exp(log_probs - log_probs_max)
                responsibilities = probs / probs.sum(axis=1, keepdims=True)

                # Calculate log-likelihood
                log_likelihood = np.sum(np.log(probs.sum(axis=1)) + log_probs_max.flatten())

                # Check convergence
                if abs(log_likelihood - log_likelihood_old) < tol:
                    converged = True
                    if verbose:
                        print(f"  Converged at iteration {iteration+1}")
                    break
                log_likelihood_old = log_likelihood

                # M-step: Update weights and covariances (NOT means!)
                weights = responsibilities.mean(axis=0)

                for k in range(n_components):
                    # Weighted covariance around fixed mean
                    centered = X - fixed_means[k]
                    weighted = responsibilities[:, k:k+1] * centered

                    cov_k = (weighted.T @ centered) / responsibilities[:, k].sum()

                    # Regularize to ensure positive definite
                    min_eig = np.linalg.eigvalsh(cov_k)[0]
                    if min_eig < 1e-6:
                        cov_k += (1e-6 - min_eig) * np.eye(n_features)

                    covariances[k] = cov_k

            if not converged and verbose:
                print(f"  Warning: Did not converge after {max_iter} iterations")

            return np.array(covariances), weights, converged

    def fit_covariances_fixed_means_mestimator(
        self,
        X,
        fixed_means,
        reference_covariances=None,
        estimator_type="tukey",
        max_iter=20,
        tol=1e-4,
        verbose=False,
    ):
        """
        Robustly fit covariances using M-estimators with iterative re-weighting.

        This method handles outliers in multimodal mixture data by:
        1. Hard-assigning points to nearest component
        2. Using M-estimator weight functions to down-weight outliers
        3. Iteratively re-fitting covariances with weighted data

        M-estimators provide soft rejection of outliers (gradual down-weighting)
        rather than hard thresholding, making them robust while retaining efficiency.

        Args:
            X (np.ndarray): Data matrix, shape (n_samples, n_features)
            fixed_means (np.ndarray): Fixed component means, shape (n_components, n_features)
            reference_covariances (np.ndarray, optional): Reference covariances from high-photon
                data for comparison/diagnostics, shape (n_components, n_features, n_features)
            estimator_type (str): Type of M-estimator to use:
                - "huber": Huber loss (moderate robustness, c=1.345)
                - "tukey": Tukey bisquare (aggressive robustness, c=4.685) [default]
            max_iter (int): Maximum number of iterations for re-weighting (default 20)
            tol (float): Convergence tolerance on covariance change (default 1e-4)
            verbose (bool): Print diagnostic information

        Returns:
            covariances (np.ndarray): Robust covariances, shape (n_components, n_features, n_features)
            weights (np.ndarray): Component weights based on assigned counts, shape (n_components,)
            point_weights (list): Per-point weights for each component (for diagnostics)

        Example:
            >>> # Get reference means from high-photon fit
            >>> ref_means, ref_db, gmm = SM_E.extract_reference_means(high_photon_data)
            >>> ref_covs = gmm.covariances_
            >>>
            >>> # Robustly fit at lower photon level using M-estimators
            >>> low_photon_data = pa_db[pa_db['photons'].between(5000, 6000)]
            >>> X = low_photon_data[['A_R', 'A_G']].values
            >>> cov, weights, pt_wts = SM_E.fit_covariances_fixed_means_mestimator(
            ...     X, ref_means, reference_covariances=ref_covs, estimator_type="tukey"
            ... )
        """
        from scipy.spatial.distance import cdist
        from scipy.stats import norm

        n_samples, n_features = X.shape
        n_components = len(fixed_means)

        if verbose:
            print(f"  M-estimator robust fitting (type={estimator_type}, max_iter={max_iter})")

        # Hard assignment to nearest component (Euclidean distance)
        distances = cdist(X, fixed_means, metric='euclidean')
        assignments = np.argmin(distances, axis=1)

        if verbose:
            for k in range(n_components):
                n_k = (assignments == k).sum()
                print(f"  Component {k}: {n_k} points assigned")

        # M-estimator weight functions
        def huber_weight(r, c=1.345):
            """Huber weight function (moderate robustness)"""
            return np.where(np.abs(r) <= c, 1.0, c / np.abs(r))

        def tukey_weight(r, c=4.685):
            """Tukey bisquare weight function (aggressive robustness)"""
            return np.where(np.abs(r) <= c, (1 - (r/c)**2)**2, 0.0)

        weight_func = tukey_weight if estimator_type == "tukey" else huber_weight

        # Initialize covariances with standard MLE per component
        covariances = []
        for k in range(n_components):
            X_k = X[assignments == k]
            if len(X_k) > n_features:
                diff = X_k - fixed_means[k]
                cov_k = (diff.T @ diff) / len(X_k)
            else:
                # Not enough points
                if reference_covariances is not None:
                    cov_k = reference_covariances[k]
                else:
                    cov_k = np.eye(n_features) * 0.001
            covariances.append(cov_k)

        covariances = np.array(covariances)

        # Iterative re-weighting
        all_point_weights = []

        for iteration in range(max_iter):
            old_covs = covariances.copy()
            iteration_weights = []

            for k in range(n_components):
                X_k = X[assignments == k]
                n_k = len(X_k)

                if n_k <= n_features:
                    # Too few points, keep current covariance
                    iteration_weights.append(np.ones(n_k))
                    continue

                # Calculate Mahalanobis distances
                try:
                    inv_cov = np.linalg.inv(covariances[k])
                except np.linalg.LinAlgError:
                    # Singular matrix, regularize
                    covariances[k] += np.eye(n_features) * 1e-6
                    inv_cov = np.linalg.inv(covariances[k])

                diff = X_k - fixed_means[k]
                mahal = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

                # Normalize by robust scale estimate (median absolute deviation)
                scale = np.median(mahal) / np.sqrt(norm.ppf(0.75))  # Chi distribution with df=n_features
                if scale < 1e-10:  # Avoid division by zero
                    scale = 1.0
                standardized = mahal / scale

                # Calculate M-estimator weights
                weights_k = weight_func(standardized)

                # Re-fit covariance with weights
                weighted_diff = diff * np.sqrt(weights_k[:, np.newaxis])
                weight_sum = weights_k.sum()

                if weight_sum > 0:
                    covariances[k] = (weighted_diff.T @ weighted_diff) / weight_sum

                iteration_weights.append(weights_k)

                if verbose and iteration == max_iter - 1:  # Print on last iteration
                    n_downweighted = (weights_k < 0.5).sum()
                    pct = 100 * n_downweighted / len(weights_k)
                    det_k = np.linalg.det(covariances[k])
                    print(f"  Component {k}: {n_downweighted}/{len(weights_k)} heavily downweighted (<0.5 weight, {pct:.1f}%)")
                    print(f"    Covariance determinant: {det_k:.6f}")

                    if reference_covariances is not None:
                        det_ref = np.linalg.det(reference_covariances[k])
                        ratio = det_k / det_ref
                        print(f"    Ratio to reference: {ratio:.2f}x")

            all_point_weights = iteration_weights

            # Check convergence (Frobenius norm of covariance change)
            max_change = np.max([np.linalg.norm(covariances[k] - old_covs[k], 'fro')
                                 for k in range(n_components)])

            if verbose and (iteration == 0 or iteration == max_iter - 1 or max_change < tol):
                print(f"  Iteration {iteration + 1}: max covariance change = {max_change:.6f}")

            if max_change < tol:
                if verbose:
                    print(f"  Converged at iteration {iteration + 1}")
                break

        # Compute final component weights (based on number of assigned points)
        weights = np.array([np.sum(assignments == k) for k in range(n_components)], dtype=float)
        weights /= weights.sum()

        return covariances, weights, all_point_weights

    def calculate_analytical_misidentification(
        self,
        fixed_means,
        covariances,
        weights,
        n_samples=10000,
        random_state=42,
    ):
        """
        Analytically calculate misidentification rates from Gaussian overlap.

        For 2-component case: Uses Monte Carlo integration to calculate the probability
        that a sample from component i is classified as component j.

        Method:
        1. Generate synthetic samples from each component
        2. Classify using Bayes decision rule (posterior probabilities)
        3. Calculate confusion matrix and error rates

        This is more accurate than empirical measurement because it characterizes
        the noise model itself, not specific noisy measurements.

        Args:
            fixed_means (np.ndarray): Mean positions, shape (n_components, n_features)
            covariances (np.ndarray): Covariance matrices, shape (n_components, n_features, n_features)
            weights (np.ndarray): Component weights, shape (n_components,)
            n_samples (int): Number of Monte Carlo samples per component (default: 10,000)
            random_state (int): Random seed for reproducibility

        Returns:
            dict: Misidentification statistics with keys:
                - 'confusion_matrix': np.ndarray of shape (n_components, n_components)
                    Entry [i, j] = P(classified as j | true component i)
                - 'accuracy_per_component': np.ndarray of shape (n_components,)
                    Probability of correct classification for each component
                - 'overall_accuracy': float
                    Weighted average accuracy
                - 'error_rate_per_component': np.ndarray of shape (n_components,)
                    Probability of misclassification for each component
                - 'overall_error_rate': float
                    Weighted average error rate

        Example:
            >>> # After fitting covariances at specific photon level
            >>> cov, weights, _ = SM_E.fit_covariances_fixed_means(X, fixed_means)
            >>> stats = SM_E.calculate_analytical_misidentification(
            ...     fixed_means, cov, weights, n_samples=10000
            ... )
            >>> print(f"Overall accuracy: {stats['overall_accuracy']:.3f}")
            >>> print(f"Confusion matrix:\n{stats['confusion_matrix']}")
        """
        np.random.seed(random_state)

        n_components = len(fixed_means)
        n_features = fixed_means.shape[1]

        # Generate synthetic samples from each component
        all_samples = []
        all_true_labels = []

        for k in range(n_components):
            # Generate samples from component k
            samples = np.random.multivariate_normal(
                mean=fixed_means[k],
                cov=covariances[k],
                size=n_samples
            )
            all_samples.append(samples)
            all_true_labels.append(np.full(n_samples, k))

        # Combine all samples
        X = np.vstack(all_samples)
        y_true = np.concatenate(all_true_labels)

        # Classify using Bayes decision rule
        log_probs = np.zeros((len(X), n_components))

        for k in range(n_components):
            mvn = multivariate_normal(mean=fixed_means[k], cov=covariances[k])
            log_probs[:, k] = mvn.logpdf(X) + np.log(weights[k])

        # Predict labels (argmax of posterior)
        y_pred = np.argmax(log_probs, axis=1)

        # Calculate confusion matrix
        # Entry [i, j] = number of samples from component i classified as j
        conf_matrix_counts = np.zeros((n_components, n_components), dtype=int)
        for i in range(n_components):
            mask = y_true == i
            for j in range(n_components):
                conf_matrix_counts[i, j] = np.sum(y_pred[mask] == j)

        # Normalize to get probabilities (rows sum to 1)
        conf_matrix = conf_matrix_counts.astype(float) / n_samples

        # Calculate accuracies
        accuracy_per_component = np.diag(conf_matrix)
        overall_accuracy = np.average(accuracy_per_component, weights=weights)

        # Calculate error rates
        error_rate_per_component = 1.0 - accuracy_per_component
        overall_error_rate = 1.0 - overall_accuracy

        return {
            'confusion_matrix': conf_matrix,
            'accuracy_per_component': accuracy_per_component,
            'overall_accuracy': overall_accuracy,
            'error_rate_per_component': error_rate_per_component,
            'overall_error_rate': overall_error_rate,
        }

    def analyze_photon_dependent_misidentification_analytical(
        self,
        photon_accumulation_db,
        fixed_means,
        reference_db,
        photon_bins,
        reference_covariances=None,
        use_earliest_entry=True,
        n_mc_samples=10000,
        estimator_type="tukey",
        max_iter=20,
        verbose=True,
    ):
        """
        Analytically analyze misidentification rates across photon bins using robust M-estimator fitting.

        Fits covariances at each photon level with fixed means using M-estimators (Huber or Tukey),
        then analytically calculates misidentification rates from distribution overlap using Monte Carlo
        integration.

        This approach:
        1. Separates signal (means - dye properties) from noise (covariances - measurement uncertainty)
        2. Robustly characterizes the noise model at each photon level (soft down-weighting of outliers)
        3. Provides stable and interpretable error rate predictions

        Workflow:
        1. For each photon bin:
           a. Extract data at that photon level
           b. Robustly fit covariances with fixed means using M-estimators
           c. Analytically calculate misidentification from overlap
        2. Return summary of error rates vs photon count

        Args:
            photon_accumulation_db (pd.DataFrame): Photon accumulation database
            fixed_means (np.ndarray): Fixed mean positions from extract_reference_means(),
                shape (n_components, 2) for [A_R, A_G]
            reference_db (pd.DataFrame): Reference molecules from extract_reference_means()
            photon_bins (array-like): Photon bin edges (e.g., [1000, 2000, 5000, 10000])
            reference_covariances (np.ndarray, optional): Reference covariances from high-photon fit
                for comparison/diagnostics. Can be obtained from gmm.covariances_ after extract_reference_means()
            use_earliest_entry (bool): If True, use earliest crossing into each bin.
                If False, use midpoint of bin. (default: True)
            n_mc_samples (int): Monte Carlo samples for analytical error calculation (default: 10,000)
            estimator_type (str): M-estimator type: "huber" (moderate) or "tukey" (aggressive, default)
            max_iter (int): Maximum iterations for M-estimator re-weighting (default: 20)
            verbose (bool): Print progress and statistics

        Returns:
            pd.DataFrame: Summary database with columns:
                - photon_bin_min, photon_bin_max: Bin edges
                - n_molecules: Number of molecules in bin
                - converged: Whether fitting converged
                - overall_accuracy: Predicted classification accuracy (analytical)
                - overall_error_rate: Predicted error rate (analytical)
                - component_0_accuracy, component_1_accuracy: Per-component accuracies
                - component_0_error_rate, component_1_error_rate: Per-component error rates
                - confusion_matrix_00, confusion_matrix_01, etc.: Full confusion matrix
                - cov_0_AR_AR, cov_0_AR_AG, cov_0_AG_AG: Fitted covariance components for component 0
                - cov_1_AR_AR, cov_1_AR_AG, cov_1_AG_AG: Fitted covariance components for component 1
                - weight_0, weight_1: Fitted component weights

        Example:
            >>> # Extract fixed means from high-photon data
            >>> means, ref_db, gmm = SM_E.extract_reference_means(pa_db, threshold=200000)
            >>> ref_covs = gmm.covariances_  # Get reference covariances
            >>>
            >>> # Analyze error rates across photon levels using M-estimators
            >>> photon_bins = np.logspace(3, 5, 11)  # 1k to 100k photons
            >>> summary = SM_E.analyze_photon_dependent_misidentification_analytical(
            ...     pa_db, means, ref_db, photon_bins,
            ...     reference_covariances=ref_covs,
            ...     estimator_type="tukey",  # Aggressive outlier down-weighting
            ...     verbose=True
            ... )
            >>>
            >>> # Plot results
            >>> plt.plot(summary['photon_bin_min'], summary['overall_accuracy'])
            >>> plt.xlabel('Photons')
            >>> plt.ylabel('Predicted Accuracy')
        """
        if verbose:
            print("=" * 60)
            print("Analytical Misidentification Analysis")
            print("=" * 60)
            print(f"Photon bins: {len(photon_bins)-1} bins")
            print(f"  Range: {photon_bins[0]:,.0f} - {photon_bins[-1]:,.0f} photons")
            print(f"Reference molecules: {len(reference_db)}")
            print(f"Fixed means: {fixed_means.shape[0]} components")
            print()

        # Filter photon accumulation data to only include reference molecules
        reference_mol_ids = set(reference_db["molecular_index"].values)
        pa_filtered = photon_accumulation_db[
            photon_accumulation_db["molecular_index"].isin(reference_mol_ids)
        ]

        if verbose:
            print(f"Photon accumulation rows (reference molecules only): {len(pa_filtered)}")

        # Storage for summary results
        all_summaries = []
        n_components = len(fixed_means)

        # Process each photon bin
        for i in range(len(photon_bins) - 1):
            bin_min = photon_bins[i]
            bin_max = photon_bins[i + 1]

            if verbose:
                print(f"\nBin {i+1}/{len(photon_bins)-1}: [{bin_min:,.0f}, {bin_max:,.0f}) photons...")

            # Get molecules in this bin
            bin_data = pa_filtered[
                (pa_filtered["photons_accumulated"] >= bin_min)
                & (pa_filtered["photons_accumulated"] < bin_max)
            ]

            if len(bin_data) == 0:
                if verbose:
                    print(f"  No molecules in this bin, skipping...")
                continue

            # Get one row per molecule (earliest entry or midpoint)
            if use_earliest_entry:
                bin_molecules = (
                    bin_data.sort_values("photons_accumulated")
                    .groupby("molecular_index")
                    .first()
                    .reset_index()
                )
            else:
                bin_midpoint = (bin_min + bin_max) / 2
                bin_data["dist_to_midpoint"] = np.abs(
                    bin_data["photons_accumulated"] - bin_midpoint
                )
                bin_molecules = (
                    bin_data.sort_values("dist_to_midpoint")
                    .groupby("molecular_index")
                    .first()
                    .reset_index()
                )
                bin_molecules = bin_molecules.drop(columns=["dist_to_midpoint"])

            n_molecules = len(bin_molecules)
            if verbose:
                print(f"  Molecules in bin: {n_molecules}")

            # Extract A_R, A_G data
            X = bin_molecules[["A_R", "A_G"]].values

            # Step 1: Robustly fit covariances with fixed means using M-estimators
            if verbose:
                print(f"  Fitting covariances (M-estimator: {estimator_type})...")
            covariances, weights, point_weights = self.fit_covariances_fixed_means_mestimator(
                X, fixed_means,
                reference_covariances=reference_covariances,
                estimator_type=estimator_type,
                max_iter=max_iter,
                verbose=False
            )
            converged = True  # M-estimators converge if they finish iterating

            if verbose:
                status = "converged" if converged else "did not converge"
                print(f"  Covariance fitting: {status}")

            # Step 2: Analytically calculate misidentification
            if verbose:
                print(f"  Calculating analytical error rates...")
            stats = self.calculate_analytical_misidentification(
                fixed_means, covariances, weights, n_samples=n_mc_samples, random_state=42
            )

            # Plot distributions for this bin if verbose
            if verbose:
                try:
                    from PlottingBase import AnalysisPlotter

                    # Predict labels for molecules in this bin
                    log_probs = np.zeros((len(X), n_components))
                    for k in range(n_components):
                        mvn = multivariate_normal(mean=fixed_means[k], cov=covariances[k])
                        log_probs[:, k] = mvn.logpdf(X) + np.log(weights[k])
                    labels = np.argmax(log_probs, axis=1)

                    plotter = AnalysisPlotter()
                    fig, (ax1, ax2) = plotter.two_column_plot(nrows=1, ncols=2, height=3)

                    colors = ['red', 'green', 'blue', 'orange', 'purple'][:n_components]

                    # Plot A_R histogram
                    ax1.hist(X[:, 0], bins=30, alpha=0.3, color='gray', label='All data')
                    for k in range(n_components):
                        component_mask = labels == k
                        if component_mask.sum() > 0:
                            ax1.hist(X[component_mask, 0], bins=30, alpha=0.5,
                                    color=colors[k], label=f'Component {k}')
                        ax1.axvline(fixed_means[k, 0], color=colors[k], linestyle='--',
                                  linewidth=2, label=f'Mean {k}: {fixed_means[k, 0]:.3f}')
                    ax1.set_xlabel('A_R')
                    ax1.set_ylabel('Count')
                    ax1.set_title(f'Bin {i+1}: [{bin_min:.0f}, {bin_max:.0f}) photons\nA_R Distribution (Acc: {stats["overall_accuracy"]:.3f})')
                    ax1.legend(fontsize=8)
                    ax1.grid(True, alpha=0.3)

                    # Plot A_G histogram
                    ax2.hist(X[:, 1], bins=30, alpha=0.3, color='gray', label='All data')
                    for k in range(n_components):
                        component_mask = labels == k
                        if component_mask.sum() > 0:
                            ax2.hist(X[component_mask, 1], bins=30, alpha=0.5,
                                    color=colors[k], label=f'Component {k}')
                        ax2.axvline(fixed_means[k, 1], color=colors[k], linestyle='--',
                                  linewidth=2, label=f'Mean {k}: {fixed_means[k, 1]:.3f}')
                    ax2.set_xlabel('A_G')
                    ax2.set_ylabel('Count')
                    ax2.set_title(f'A_G Distribution\nσ²(0)={covariances[0,0,0]+covariances[0,1,1]:.4f}, σ²(1)={covariances[1,0,0]+covariances[1,1,1]:.4f}')
                    ax2.legend(fontsize=8)
                    ax2.grid(True, alpha=0.3)

                    _safe_tight_layout(fig)
                    plotter.save_or_show(fig, save_path=None)

                    print(f"  (Close plot to continue to next bin)")

                except Exception as e:
                    print(f"  (Plotting skipped for this bin - error: {e})")

            # Build summary row
            summary_row = {
                "photon_bin_min": bin_min,
                "photon_bin_max": bin_max,
                "n_molecules": n_molecules,
                "converged": converged,
                "overall_accuracy": stats['overall_accuracy'],
                "overall_error_rate": stats['overall_error_rate'],
            }

            # Add per-component accuracies and error rates
            for k in range(n_components):
                summary_row[f'component_{k}_accuracy'] = stats['accuracy_per_component'][k]
                summary_row[f'component_{k}_error_rate'] = stats['error_rate_per_component'][k]

            # Add confusion matrix elements
            for k1 in range(n_components):
                for k2 in range(n_components):
                    summary_row[f'confusion_matrix_{k1}{k2}'] = stats['confusion_matrix'][k1, k2]

            # Add covariance elements for each component
            for k in range(n_components):
                cov = covariances[k]
                summary_row[f'cov_{k}_AR_AR'] = cov[0, 0]
                summary_row[f'cov_{k}_AR_AG'] = cov[0, 1]
                summary_row[f'cov_{k}_AG_AG'] = cov[1, 1]
                summary_row[f'weight_{k}'] = weights[k]

            if verbose:
                print(f"  Overall accuracy (analytical): {stats['overall_accuracy']:.3f}")
                print(f"  Component accuracies: {stats['accuracy_per_component']}")

            all_summaries.append(summary_row)

        # Combine results
        if len(all_summaries) > 0:
            summary_db = pd.DataFrame(all_summaries)

            if verbose:
                print("\n" + "=" * 60)
                print("Analytical Analysis Complete!")
                print("=" * 60)
                print(f"Summary database: {len(summary_db)} bins")
                print(
                    f"Overall accuracy range: {summary_db['overall_accuracy'].min():.3f} - {summary_db['overall_accuracy'].max():.3f}"
                )
        else:
            summary_db = pd.DataFrame()
            if verbose:
                print("\n" + "=" * 60)
                print("No results generated (no molecules in bins)")
                print("=" * 60)

        return summary_db

    def _find_histogram_peaks_1d(self, data, n_peaks, bins=50):
        """
        Find peaks in 1D histogram for initial channel guess.

        Uses the same approach as extract_reference_means() for consistency.

        Args:
            data (np.ndarray): 1D array of values
            n_peaks (int): Number of peaks to find
            bins (int): Number of histogram bins (default: 50, matches extract_reference_means)

        Returns:
            np.ndarray: Peak positions, shape (n_peaks,)
        """
        from scipy.signal import find_peaks

        # Create histogram with density normalization
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find peaks in histogram (distance=5 prevents very close peaks)
        peaks, properties = find_peaks(hist, height=0, distance=5)

        if len(peaks) >= n_peaks:
            # Sort by height and take top n_peaks
            peak_heights = hist[peaks]
            top_peak_indices = np.argsort(peak_heights)[-n_peaks:]
            peak_positions = bin_centers[peaks[top_peak_indices]]
            return np.sort(peak_positions)
        else:
            # Fallback: use quantiles
            return np.quantile(data, np.linspace(0.2, 0.8, n_peaks))

    def _find_initial_means_2d(self, X, n_channels, method="histogram_peaks"):
        """
        Find initial channel means in 2D feature space.

        Uses the same approach as extract_reference_means() for consistency.
        For 2-channel case, assumes anticorrelated colors (high A_R → low A_G).

        Args:
            X (np.ndarray): Data matrix, shape (n_samples, 2)
            n_channels (int): Number of channels
            method (str): Method for finding means

        Returns:
            np.ndarray: Initial means, shape (n_channels, 2)
        """
        if method == "histogram_peaks":
            # Find peaks in each dimension separately
            peaks_dim0 = self._find_histogram_peaks_1d(X[:, 0], n_channels)
            peaks_dim1 = self._find_histogram_peaks_1d(X[:, 1], n_channels)

            # Combine to form initial 2D means
            # Match peaks: if dim0 high, dim1 should be low (and vice versa)
            # This assumes anticorrelated channels (typical for multicolor SMLM)
            initial_means = np.zeros((n_channels, 2))

            if n_channels == 2:
                # Sort dim0 ascending, dim1 descending to pair anticorrelated peaks
                dim0_sorted = np.sort(peaks_dim0)
                dim1_sorted = np.sort(peaks_dim1)[::-1]
                initial_means[:, 0] = dim0_sorted
                initial_means[:, 1] = dim1_sorted
            else:
                # For n_channels != 2, use simpler approach (no assumed correlation)
                initial_means[:, 0] = peaks_dim0
                initial_means[:, 1] = peaks_dim1

            return initial_means

        elif method == "kmeans":
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_channels, n_init=10, random_state=42)
            kmeans.fit(X)
            return kmeans.cluster_centers_

        else:
            raise ValueError(f"Unknown method: {method}")

    def _estimate_initial_covariances_2d(self, X, initial_means, n_channels,
                                          X_err=None, use_core_region=True,
                                          percentile=50, scale=0.7):
        """
        Estimate initial covariances conservatively from core regions around means.

        Strategy:
        1. Hard assign points to nearest mean
        2. Take only the CORE points (e.g., 50th percentile by distance) for each component
        3. Calculate robust covariance from core region
        4. Optionally incorporate fitting errors
        5. Scale down by factor (default 0.7) to prevent EM from over-expanding

        This prevents the initial guess from being too broad by focusing on the
        well-separated core of each distribution.

        Args:
            X (np.ndarray): Data matrix, shape (n_samples, n_features)
            initial_means (np.ndarray): Initial means, shape (n_channels, n_features)
            n_channels (int): Number of channels
            X_err (np.ndarray, optional): Error matrix, shape (n_samples, n_features)
            use_core_region (bool): If True, use only core percentile of each component
            percentile (float): Percentile threshold for core region (default: 50)
            scale (float): Scaling factor for covariances (default: 0.7)

        Returns:
            np.ndarray: Initial covariances, shape (n_channels, n_features, n_features)
        """
        from scipy.spatial.distance import cdist

        n_samples, n_features = X.shape

        # Hard assignment: assign each point to nearest mean
        distances = cdist(X, initial_means, metric='euclidean')
        assignments = np.argmin(distances, axis=1)

        # Calculate sample covariance for each component
        initial_covariances = np.zeros((n_channels, n_features, n_features))

        for k in range(n_channels):
            mask = assignments == k
            n_assigned = mask.sum()

            if n_assigned > 20:  # Need reasonable number of points
                X_k = X[mask]

                # Use only core region for more conservative estimate
                if use_core_region and n_assigned > 50:
                    # Calculate distances from mean
                    dists_k = np.linalg.norm(X_k - initial_means[k], axis=1)
                    # Take only the closest percentile of points
                    threshold = np.percentile(dists_k, percentile)
                    core_mask = dists_k <= threshold
                    X_k_core = X_k[core_mask]
                else:
                    X_k_core = X_k

                if len(X_k_core) > 5:
                    # Calculate robust sample covariance from core
                    centered = X_k_core - initial_means[k]
                    cov_k = (centered.T @ centered) / len(X_k_core)

                    # Ensure symmetry (numerical precision can break this)
                    cov_k = (cov_k + cov_k.T) / 2.0

                    # If we have error information, incorporate it
                    if X_err is not None:
                        # Average measurement error for this component
                        X_err_k = X_err[mask]
                        if use_core_region and n_assigned > 50:
                            X_err_k = X_err_k[core_mask]

                        # Mean squared error (diagonal covariance contribution)
                        mean_err_sq = np.mean(X_err_k**2, axis=0)
                        err_cov = np.diag(mean_err_sq)

                        # Add measurement error to intrinsic spread
                        # But cap it so errors don't dominate
                        err_cov_capped = np.minimum(err_cov, cov_k * 0.5)
                        cov_k = cov_k + err_cov_capped

                    # Scale down covariance to be conservative (prevents overly broad initial guess)
                    cov_k = cov_k * scale

                    # Add regularization to ensure positive definite
                    # Use scale-dependent regularization based on diagonal variance
                    diag_var = np.diag(cov_k)
                    reg_scale = np.maximum(1e-6, np.mean(diag_var) * 1e-3)
                    cov_k += np.eye(n_features) * reg_scale

                    # Final validation: ensure positive definite
                    eigvals = np.linalg.eigvalsh(cov_k)
                    if np.min(eigvals) <= 0:
                        # Add stronger regularization
                        cov_k += np.eye(n_features) * (np.abs(np.min(eigvals)) + 1e-4)

                    initial_covariances[k] = cov_k
                else:
                    # Not enough core points, use small isotropic
                    initial_covariances[k] = np.eye(n_features) * 0.005
            else:
                # Very few points assigned, use small isotropic covariance
                initial_covariances[k] = np.eye(n_features) * 0.005

        return initial_covariances

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
            print("=" * 70)
            print("Channel Unmixing")
            print("=" * 70)
            print(f"Input: {len(loc_data)} localizations")
            print(f"Channels: {n_channels}")
            print(f"Features: {channels_to_use}")
            print()

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
                print(f"Missing error columns: {missing_errors}")
                print("Attempting to auto-generate errors...")

            for error_col in missing_errors:
                # Extract base column name (remove '_err' suffix)
                base_col = error_col.replace('_err', '')

                if base_col == 'photons':
                    # Poisson statistics: σ = sqrt(N)
                    loc_data_work[error_col] = np.sqrt(np.maximum(loc_data_work[base_col].values, 1))
                    if verbose:
                        mean_err = loc_data_work[error_col].mean()
                        print(f"  {error_col}: Generated from Poisson statistics (mean={mean_err:.2f})")
                elif base_col in loc_data_work.columns:
                    # For other columns, use a small fraction of the value as error estimate
                    # This is a conservative guess - user should provide real errors if possible
                    loc_data_work[error_col] = loc_data_work[base_col].values * 0.05  # 5% relative error
                    if verbose:
                        mean_err = loc_data_work[error_col].mean()
                        print(f"  {error_col}: Estimated as 5% of {base_col} (mean={mean_err:.4f})")
                        print(f"    WARNING: Using estimated errors. Provide measured errors for better results.")
                else:
                    raise ValueError(f"Cannot auto-generate error for '{error_col}': column '{base_col}' not found")

        # Extract feature matrix
        X = loc_data_work[channels_to_use].values
        n_features = X.shape[1]

        if verbose:
            print(f"Feature matrix: {X.shape}")
            print(f"Feature ranges:")
            for i, col in enumerate(channels_to_use):
                print(
                    f"  {col}: [{X[:, i].min():.3f}, {X[:, i].max():.3f}], mean={X[:, i].mean():.3f}"
                )
            print()

        # ===== Phase 2: Initial Guess for Channel Means =====
        if verbose:
            print(f"Finding initial channel means (method: {initial_guess_method})...")

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
            print(f"Initial means:")
            for k in range(n_channels):
                mean_str = ", ".join(
                    [f"{channels_to_use[i]}={initial_means[k, i]:.3f}" for i in range(n_features)]
                )
                print(f"  Channel {k}: {mean_str}")
            print()

        # ===== Phase 2.5: Estimate Initial Covariances =====
        # Two-stage initialization: means from histograms, covariances from data
        if initial_guess_method == "histogram_peaks":
            if verbose:
                print("Estimating initial covariances from core regions (conservative)...")

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
                    print(f"  Channel {k}: det(cov)={det_k:.6f}, {sigma_str}")
                print()

            # Create diagnostic plot showing initial guess (only for 2D)
            if plot_results and n_features == 2:
                self._plot_initial_guess_2d(
                    X, channels_to_use, initial_means, initial_covariances, n_channels
                )
            elif plot_results and n_features > 2:
                if verbose:
                    print(f"  Skipping initial guess plot (only available for 2D, current: {n_features}D)")
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
                    print(f"Fitting GMM (method: EM → Extreme Deconvolution, covariance: {covariance_type})...")
                    print(f"  Auto-selected pygmmis (error columns detected)")
                    print(f"  Mean errors: {X_err.mean(axis=0)}")
            else:
                # Use sklearn EM (no errors available)
                actual_method = "sklearn_EM"
                if verbose:
                    print(f"Fitting GMM (method: EM → sklearn, covariance: {covariance_type})...")
                    print("  No error columns found, using sklearn EM without error weighting")
        else:
            actual_method = gmm_fit_method
            if verbose:
                print(f"Fitting GMM (method: {gmm_fit_method}, covariance: {covariance_type})...")
                if has_errors:
                    print(f"  Error columns available (mean errors: {X_err.mean(axis=0)})")
                else:
                    print("  No error columns found")

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
                        print(f"  Warning: Channel {k} covariance not positive definite (min eigenvalue={np.min(eigvals):.2e})")
                        print(f"           Adding regularization...")
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
                print("Using fixed initial guess (no EM refinement)")

        else:
            raise ValueError(f"Unknown gmm_fit_method: {gmm_fit_method}")

        if verbose:
            status = "converged" if converged else "did not converge"
            print(f"GMM fitting: {status}")
            print(f"Fitted means:")
            for k in range(n_channels):
                mean_str = ", ".join(
                    [f"{channels_to_use[i]}={means[k, i]:.3f}" for i in range(n_features)]
                )
                weight_pct = weights[k] * 100
                print(f"  Channel {k}: {mean_str} (weight: {weight_pct:.1f}%)")
            print()

        # ===== Phase 4: Channel Assignment with Confidence =====
        if verbose:
            print("Calculating posterior probabilities and assignments...")

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
                print(
                    f"Calculating analytical FPR to determine confidence threshold (target: {false_positive_rate:.3f})..."
                )

            stats = self.calculate_analytical_misidentification(
                means, covariances, weights, n_samples=10000, random_state=42
            )

            if verbose:
                print(f"Analytical accuracy: {stats['overall_accuracy']:.3f}")
                print(f"Confusion matrix:")
                print(stats["confusion_matrix"])
                print()

            # Use simple threshold based on FPR
            # Higher FPR tolerance → lower threshold → more assignments
            confidence_threshold = 1.0 - (false_positive_rate / n_channels)

            if verbose:
                print(
                    f"Setting confidence threshold to {confidence_threshold:.3f} (from FPR={false_positive_rate:.3f})"
                )

        # Apply confidence threshold
        is_assigned = confidence >= confidence_threshold
        channel_assignments_filtered = channel_assignments.copy()
        channel_assignments_filtered[~is_assigned] = -1

        # ===== Phase 5: Outlier Detection =====
        is_outlier = np.zeros(n_locs, dtype=bool)

        if outlier_rejection == "mahalanobis":
            if verbose:
                print("Applying Mahalanobis distance outlier rejection...")

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
                print(
                    f"  Outliers detected: {n_outliers} ({100*n_outliers/n_locs:.2f}%, threshold={outlier_threshold:.2f})"
                )

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
            print("\n" + "=" * 70)
            print("Unmixing Complete")
            print("=" * 70)
            print(f"Assignments:")
            for k in range(n_channels):
                n_k = n_assigned_per_channel[k]
                pct_k = 100 * n_k / n_locs
                print(f"  Channel {k}: {n_k:,} ({pct_k:.1f}%)")
            pct_unassigned = 100 * n_unassigned / n_locs
            print(f"  Unassigned: {n_unassigned:,} ({pct_unassigned:.1f}%)")
            print()

        # ===== Phase 7: Diagnostic Plotting =====
        if plot_results:
            self._plot_unmixing_results(
                X, channels_to_use, channel_assignments_filtered, confidence,
                means, covariances, weights, n_channels, confidence_threshold,
                metadata
            )

        return assigned_locs, metadata

    def _plot_initial_guess_2d(self, X, channels_to_use, initial_means,
                                initial_covariances, n_channels):
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
        plt.show()

    def _plot_unmixing_results(
        self, X, channels_to_use, assignments, confidence,
        means, covariances, weights, n_channels, confidence_threshold,
        metadata
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
        # assignment_stage: 0=unassigned, 1=initial, 2+=refinement_iteration
        assigned_initial['assignment_stage'] = 0
        assigned_initial.loc[assigned_initial['channel'] >= 0, 'assignment_stage'] = 1

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

        # ===== STEP 2.5: Refine Spectral Model from Puncta (OPTIONAL ENHANCEMENT) =====
        n_puncta_total = sum(puncta_per_channel.values())

        if n_puncta_total >= n_channels:
            if verbose:
                print("=" * 80)
                print("STEP 2.5: Refine Spectral Model from Puncta")
                print("=" * 80)

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
                print()
        else:
            if verbose:
                print("=" * 80)
                print(f"STEP 2.5: Skipping spectral refinement (only {n_puncta_total} puncta)")
                print("=" * 80)
                print()

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
            print(f"Spatial clustering scale:")
            print(f"  Base scale (mean of median errors) = {base_scale:.4f} pixels")
            print(f"    median(xc_err) = {median_xc_err:.4f}, median(yc_err) = {median_yc_err:.4f}")
            print(f"  spatial_eps multiplier = {spatial_eps:.2f}")
            print(f"  Effective epsilon = {epsilon_pixels:.4f} pixels")
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
                clusterer = DBSCAN(eps=epsilon_pixels, min_samples=min_cluster_size)
                method_info = "DBSCAN"
            elif spatial_method == 'HDBSCAN':
                clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                                   cluster_selection_epsilon=epsilon_pixels)
                method_info = f"HDBSCAN ({HDBSCAN_BACKEND})"
            else:
                raise ValueError(f"Unknown spatial_method: {spatial_method}")

            if verbose and k == 0:  # Only print once
                print(f"  Clustering method: {method_info}")

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
                    print(f"  Channel {k}: Only {n_locs_k} locs in puncta (< {min_locs_per_channel}), keeping original GMM parameters")
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
                print(f"  Channel {k}: Refined from {n_locs_k:,} locs in puncta")
                print(f"    Original mean: {original_means[k]}")
                print(f"    Refined mean:  {refined_means[k]}")

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


    def plot_refinement_diagnostics(
        self,
        assigned_current: pd.DataFrame,
        metadata: Dict,
        n_channels: int,
        channels_to_use: list,
        save_path: Optional[str] = None,
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
            print(f"Saved refinement diagnostics to: {save_path}")
        else:
            plt.show()
        
        # Create spatial distribution plot if 2D data
        n_features = len(channels_to_use)
        if n_features == 2 and 'xc' in assigned_current.columns and 'yc' in assigned_current.columns:
            self._plot_spatial_distribution(
                assigned_current, n_channels, n_recovered, save_path
            )

    def _plot_spatial_distribution(
        self,
        assigned_current: pd.DataFrame,
        n_channels: int,
        n_recovered: Dict[int, int],
        save_path: Optional[str] = None,
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
            print(f"Saved spatial distribution to: {spatial_path}")
        else:
            plt.show()
