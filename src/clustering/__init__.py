"""
clustering — clustering subpackage for pyBayerSMLM.

Provides algorithm-specific mixins that are composed into the main
extract_SMs class in SM_extractionfunctions.py.  Each mixin assumes
the host class supplies:
    - self.filter_quality_localisations(...)
    - self.average_parameters(data, labels)
    - self._load_localisation_files(loc_data, start_frame)
    - self.io   (IOFunctions.IO_Functions instance)
    - self.pixel_size (µm)
"""

from ._config import ClusteringConfig
from .hdbscan_clusterer import HDBSCANMixin
from .dbscan_clusterer import DBSCANMixin
from .linked_clusterer import LinkedMixin
from .batch import BatchMixin

__all__ = [
    "ClusteringConfig",
    "HDBSCANMixin",
    "DBSCANMixin",
    "LinkedMixin",
    "BatchMixin",
]
