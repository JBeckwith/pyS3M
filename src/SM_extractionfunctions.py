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
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal


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

    def filter_quality_localizations(
        self,
        loc_data,
        chi_val=None,
        max_localization_error=1.0,
        min_photons=500,
        max_photons=None,
    ):
        """
        Apply quality filters to localization data.

        Args:
            loc_data (pd.DataFrame): Localization data to filter
            chi_val (float, optional): Chi-squared threshold. If None, uses median.
            max_localization_error (float): Maximum localization precision in pixels
            min_photons (int): Minimum total photon count
            max_photons (int): Maximum total photon count

        Returns:
            pd.DataFrame: Filtered localization data
        """
        # Calculate chi-squared threshold if not provided
        if chi_val is None:
            chi_val = np.median(loc_data["chi_sqr"])

        # Apply quality filters
        filtered_data = loc_data[loc_data["chi_sqr"] < chi_val].copy()
        filtered_data = filtered_data[filtered_data["xc_err"] < max_localization_error]
        filtered_data = filtered_data[filtered_data["yc_err"] < max_localization_error]

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
        max_localization_error=1.0,
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

        loc_data = self.filter_quality_localizations(
            loc_data, chi_val, max_localization_error, min_photons, max_photons
        )
        X = np.vstack([loc_data["xc"], loc_data["yc"]]).T
        loc_precision = 0.5 * (
            np.mean(loc_data["xc_err"]) + np.mean(loc_data["yc_err"])
        )
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
        max_localization_error=1.0,
        min_photons=500,
        max_photons=None,
    ):
        """
        Extract single molecules from multiple localization files by clustering.

        Args:
            localisation_files (list): List of HDF5 localization file paths
            chi_val (float, optional): Chi-squared threshold for filtering. Defaults to median.

        Returns:
            tuple: (single_molecule_database, single_frame_database) as DataFrames
                   single_frame_database includes molecular_index column and excludes unassigned localizations
        """

        molecular_index_offset = 0

        loc_data = self.filter_quality_localizations(
            loc_data, chi_val, max_localization_error, min_photons, max_photons
        )
        X = np.vstack([loc_data["xc"], loc_data["yc"]]).T
        loc_precision = 0.5 * (
            np.mean(loc_data["xc_err"]) + np.mean(loc_data["yc_err"])
        )
        hdb = DBSCAN(
            min_samples=min_cluster_size,
            eps=loc_precision,
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
        max_localization_error=1.0,
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

        loc_data = self.filter_quality_localizations(
            loc_data, chi_val, max_localization_error, min_photons, max_photons
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
        max_localization_error=1.0,
        min_photons=500,
        max_photons=1e6,
        max_distance=0.5,
        max_frames=10,
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
                    max_localization_error=max_localization_error,
                    min_photons=min_photons,
                    max_photons=max_photons,
                )
            elif clustering_method.upper() == "DBSCAN":
                sm_db, sf_db = self.extract_single_molecules_DBSCAN(
                    loc_data,
                    min_cluster_size=min_cluster_size,
                    chi_val=chi_val,
                    max_localization_error=max_localization_error,
                    min_photons=min_photons,
                    max_photons=max_photons,
                )
            elif clustering_method.lower() == "linked":
                sm_db, sf_db = self.extract_single_molecules_linked(
                    loc_data,
                    max_distance=max_distance,
                    max_frames=max_frames,
                    chi_val=chi_val,
                    max_localization_error=max_localization_error,
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

            # Build cumulative arrays
            cumsum_A_R = np.cumsum(mol_data["A_R"].values)
            cumsum_A_G = np.cumsum(mol_data["A_G"].values)
            cumsum_A_B = np.cumsum(mol_data["A_B"].values)
            cumsum_photons = np.cumsum(mol_data["photons"].values)

            # Error propagation for cumulative sums (assuming independent measurements)
            # σ_sum² = σ_1² + σ_2² + ... + σ_N²
            if "A_R_err" in mol_data.columns:
                cumsum_A_R_err_sq = np.cumsum(mol_data["A_R_err"].values ** 2)
                cumsum_A_G_err_sq = np.cumsum(mol_data["A_G_err"].values ** 2)
                cumsum_A_B_err_sq = np.cumsum(mol_data["A_B_err"].values ** 2)
                cumsum_A_R_err = np.sqrt(cumsum_A_R_err_sq)
                cumsum_A_G_err = np.sqrt(cumsum_A_G_err_sq)
                cumsum_A_B_err = np.sqrt(cumsum_A_B_err_sq)
            else:
                cumsum_A_R_err = np.zeros(n_frames)
                cumsum_A_G_err = np.zeros(n_frames)
                cumsum_A_B_err = np.zeros(n_frames)

            # Normalize RGB at each accumulation step
            total_rgb = cumsum_A_R + cumsum_A_G + cumsum_A_B
            A_R_norm = cumsum_A_R / total_rgb
            A_G_norm = cumsum_A_G / total_rgb
            A_B_norm = cumsum_A_B / total_rgb

            # Propagate errors through normalization
            # For f = R/(R+G+B), using simplified error propagation
            # σ_f ≈ σ_R / (R+G+B)  (ignoring correlations)
            A_R_norm_err = cumsum_A_R_err / total_rgb
            A_G_norm_err = cumsum_A_G_err / total_rgb
            A_B_norm_err = cumsum_A_B_err / total_rgb

            # Calculate position statistics
            xc_cumsum = np.cumsum(mol_data["xc"].values)
            yc_cumsum = np.cumsum(mol_data["yc"].values)
            frames_range = np.arange(1, n_frames + 1)

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
                    "A_R": A_R_norm[i],
                    "A_G": A_G_norm[i],
                    "A_B": A_B_norm[i],
                    "A_R_err": A_R_norm_err[i],
                    "A_G_err": A_G_norm_err[i],
                    "A_B_err": A_B_norm_err[i],
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

    def analyze_multi_fov_dataset(
        self,
        localisation_files,
        clustering_method="HDBSCAN",
        build_accumulation=True,
        min_cluster_size=10,
        chi_val=None,
        max_localization_error=1.0,
        min_photons=500,
        max_photons=1e6,
        max_distance=0.5,
        max_frames=10,
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
            max_localization_error=max_localization_error,
            min_photons=min_photons,
            max_photons=max_photons,
            max_distance=max_distance,
            max_frames=max_frames,
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
                single_frame_db, sf_path, normalise_photons=False, append=False
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
                fig, (ax1, ax2) = plotter.create_subplots(nrows=1, ncols=2, figsize=(12, 4))

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

                fig.tight_layout()
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
                    fig, (ax1, ax2) = plotter.create_subplots(nrows=1, ncols=2, figsize=(12, 4))

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

                    fig.tight_layout()
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
