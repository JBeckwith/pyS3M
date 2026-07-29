# -*- coding: utf-8 -*-
"""
clustering/hdbscan_clusterer.py

HDBSCAN-based single-molecule extraction mixin.
Extracted from SM_extractionfunctions.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pyS3M.Constants import FilteringConstants, FilteringCriteria
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
        loc_data: pd.DataFrame,
        min_cluster_size: int = 10,
        criteria: FilteringCriteria = None,
        chi_val: float | None = None,
        max_localisation_error: float = FilteringConstants.MAX_LOCALISATION_ERROR_PX,
        max_colour_error: float = FilteringConstants.MAX_COLOUR_ERROR,
        min_sigma: float | None = None,
        max_sigma: float | None = None,
        max_sigma_error: float | None = None,
        min_photons: float = FilteringConstants.MIN_PHOTONS,
        max_photons: float | None = None,
        start_frame: int = 0,
        config: ClusteringConfig = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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

        loc_data = self._prepare_locs(
            loc_data, start_frame, criteria, chi_val,
            max_localisation_error, max_colour_error,
            min_sigma, max_sigma, max_sigma_error,
            min_photons, max_photons,
        )

        if not self._check_min_locs(loc_data, min_cluster_size):
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

        return self._finish_clustering(loc_data, hdb.labels_)
