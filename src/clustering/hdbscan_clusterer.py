# -*- coding: utf-8 -*-
"""
clustering/hdbscan_clusterer.py

HDBSCAN-based single-molecule extraction mixin.
Extracted from SM_extractionfunctions.py.
"""
import numpy as np
import pandas as pd

from Constants import FilteringConstants
from ._config import ClusteringConfig
import logging
logger = logging.getLogger(__name__)


# Prefer the multicore-optimised fast_hdbscan; fall back to sklearn.
try:
    from fast_hdbscan import HDBSCAN
    HDBSCAN_BACKEND = "fast_hdbscan"
except ImportError:
    from sklearn.cluster import HDBSCAN
    HDBSCAN_BACKEND = "sklearn"


class HDBSCANMixin:
    """Mixin supplying HDBSCAN-based single-molecule extraction."""

    def extract_single_molecules_HDBSCAN(
        self,
        loc_data,
        min_cluster_size=10,
        criteria=None,
        chi_val=None,
        max_localisation_error=FilteringConstants.MAX_LOCALISATION_ERROR_PX,
        max_colour_error=FilteringConstants.MAX_COLOUR_ERROR,
        min_sigma=None,
        max_sigma=None,
        max_sigma_error=None,
        min_photons=FilteringConstants.MIN_PHOTONS,
        max_photons=None,
        start_frame=0,
        config: ClusteringConfig = None,
    ):
        """
        Extract single molecules from localisation data by HDBSCAN clustering.

        Args:
            loc_data (pd.DataFrame): Localisation data to process.
            min_cluster_size (int): Minimum cluster size for HDBSCAN.
            criteria (FilteringCriteria, optional): Quality filter bundle.
            chi_val, max_localisation_error, max_colour_error, min_sigma,
            max_sigma, max_sigma_error, min_photons, max_photons:
                Quality filter parameters (see filter_quality_localisations).
            start_frame (int): Discard localisations before this frame.

        Returns:
            tuple: (single_molecule_database, single_frame_database) as DataFrames.
                   single_frame_database includes molecular_index column and
                   excludes unassigned localisations.
        """
        if config is not None:
            min_cluster_size = config.min_cluster_size
            start_frame      = config.start_frame

        molecular_index_offset = 0

        loc_data = self._load_localisation_files(loc_data, start_frame=start_frame)

        loc_data = self.filter_quality_localisations(
            loc_data=loc_data, criteria=criteria,
            chi_val=chi_val, max_localisation_error=max_localisation_error,
            min_photons=min_photons, max_photons=max_photons,
            max_colour_error=max_colour_error,
            min_sigma=min_sigma, max_sigma=max_sigma,
            max_sigma_error=max_sigma_error,
        )

        if len(loc_data) == 0:
            logger.warning("Warning: No localizations remaining after filtering. Returning empty databases.")
            return pd.DataFrame(), pd.DataFrame()

        if len(loc_data) < min_cluster_size:
            logger.warning(f"Warning: Only {len(loc_data)} localizations remaining after filtering, " f"but min_cluster_size={min_cluster_size}. Need at least {min_cluster_size} points. " f"Returning empty databases.")
            return pd.DataFrame(), pd.DataFrame()

        X = np.vstack([loc_data["xc"], loc_data["yc"]]).T
        loc_precision = 0.5 * (
            np.mean(loc_data["xc_err"]) + np.mean(loc_data["yc_err"])
        )

        logger.info(f"Using {HDBSCAN_BACKEND} for HDBSCAN clustering")
        hdb = HDBSCAN(
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=loc_precision,
        )
        hdb.fit(X)

        assigned_mask = hdb.labels_ >= 0
        loc_data_assigned = loc_data[assigned_mask].copy()
        labels_assigned = hdb.labels_[assigned_mask]

        loc_data_assigned["molecular_index"] = labels_assigned + molecular_index_offset

        df = self.average_parameters(loc_data_assigned, labels_assigned)
        df["molecular_index"] = df.index + molecular_index_offset

        return df, loc_data_assigned
