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

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)
import IOFunctions

import postprocess as _postprocess
from sklearn.cluster import DBSCAN, HDBSCAN


class extract_SMs:
    def __init__(self, io_functions=None) -> None:
        """Single molecule extraction functions for clustering localizations into single molecules.
        
        Args:
            io_functions: IO functions instance (default: creates new instance)
        """
        # Dependency injection with sensible defaults
        self.io = io_functions if io_functions is not None else IOFunctions.IO_Functions()

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
            trace_matrix[i, :] = np.sum(
                np.sum(image_stack[:, xmin:xmax, ymin:ymax], axis=-1), axis=-1
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
        link_groups = _postprocess.get_link_groups(
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
