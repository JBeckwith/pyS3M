# -*- coding: utf-8 -*-
"""
This class contains helper functions pertaining to analysis of images based on their
radiality, relating to the RASP concept.
jsb92, 2024/01/02
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import fnmatch

import numpy as np
import polars as pl
from numpy.typing import NDArray


class Helper_Functions:
    """Helper functions for image analysis and data processing.

    Provides utility functions for database cleaning, file operations,
    and radiality analysis related to the RASP (Radiality Analysis of
    Single-molecule Positions) concept.
    """

    def __init__(self) -> None:
        """Initialize Helper_Functions class."""
        pass

    def clean_database(self, database: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
        """

        clean_database function replaces columns that are not filename (assumed last)
        with floats

        Args:
            database (pl.DataFrame): database in question
            colunms (list): columns

        Returns:
            database (pl.DataFrame): cleaned database
        """
        for i, column in enumerate(columns[:-1]):
            database = database.replace_column(
                i,
                pl.Series(column, np.array(database[column].to_numpy(), dtype="float")),
            )
        return database

    def file_search(self, folder: Path | str, string1: str, string2: str) -> list[str]:
        """
        Search for files containing 'string1' in their names within 'folder',
        and then filter the results to include only those containing 'string2'.

        Args:
            folder (str): The directory to search for files.
            string1 (str): The first string to search for in the filenames.
            string2 (str): The second string to filter the filenames containing string1.

        Returns:
            file_list (list): A sorted list of file paths matching the search criteria.
        """
        import re

        def sorted_alphanumeric(data):
            convert = lambda text: int(text) if text.isdigit() else text.lower()
            alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
            return sorted(data, key=alphanum_key)

        # Get a list of all files containing 'string1' in their names within 'folder'
        file_list = [
            str(p)
            for p in Path(folder).rglob("*" + string1 + "*")
            if p.is_file()
        ]
        file_list = np.sort([e for e in file_list if string2 in e])
        return sorted_alphanumeric(file_list)

    def crop_calibration_maps(self, maps_dict: dict[str, NDArray[np.float32]], start_x: int, start_y: int, width: int, height: int) -> dict[str, NDArray[np.float32]]:
        """Crop all calibration maps to ROI using correct numpy indexing [y, x].

        Args:
            maps_dict (dict): Dictionary of calibration maps (gain_map, offset_map, etc.)
            start_x (int): Starting x coordinate (column)
            start_y (int): Starting y coordinate (row)
            width (int): Width of ROI (x-dimension)
            height (int): Height of ROI (y-dimension)

        Returns:
            dict: Dictionary with same keys, but maps cropped to ROI
        """
        return {
            key: arr[start_y : start_y + height, start_x : start_x + width]
            for key, arr in maps_dict.items()
        }

    def calculate_roi_bounds(
        self, xcentre: float, ycentre: float, roi_size: int, width: int, height: int, min_roi_size: int = 4
    ) -> tuple[int, int, int, int] | None:
        """Calculate square ROI boundaries within image bounds.

        Computes xmin, xmax, ymin, ymax for a square ROI centered at (xcentre, ycentre),
        ensuring the ROI stays within image boundaries and is perfectly square.

        Args:
            xcentre (float): Center x coordinate (column)
            ycentre (float): Center y coordinate (row)
            roi_size (int): Desired ROI size (pixels)
            width (int): Image width (pixels)
            height (int): Image height (pixels)
            min_roi_size (int): Minimum acceptable ROI size (default: 4)

        Returns:
            tuple or None: (xmin, xmax, ymin, ymax) if ROI is valid, None otherwise
                          Returns None if ROI is not square or smaller than min_roi_size
        """
        xmin = max(0, int(xcentre - roi_size / 2))
        xmax = min(int(xcentre + roi_size / 2), width)
        ymin = max(0, int(ycentre - roi_size / 2))
        ymax = min(int(ycentre + roi_size / 2), height)

        # Check if ROI is square
        roi_width = xmax - xmin
        roi_height = ymax - ymin
        if roi_width != roi_height:
            return None

        # Check if ROI is large enough
        if roi_width < min_roi_size or roi_height < min_roi_size:
            return None

        return xmin, xmax, ymin, ymax

    def calculate_parallel_chunks(
        self, total_items: int, max_workers: int = 60, worker_ratio: float = 0.9, tasks_per_worker: int = 100
    ) -> tuple[int, int, list[int], NDArray[np.intp]]:
        """Calculate optimal chunk distribution for parallel processing.

        Distributes items across parallel workers with load balancing to ensure
        efficient parallel execution without overwhelming the system.

        Args:
            total_items (int): Total number of items to process
            max_workers (int): Maximum number of worker processes (default: 60)
            worker_ratio (float): Fraction of CPU cores to use (default: 0.9)
            tasks_per_worker (int): Number of tasks per worker for load balancing (default: 100)

        Returns:
            tuple: (n_workers, n_tasks, items_per_task, start_indices)
                - n_workers (int): Number of worker processes to use
                - n_tasks (int): Total number of tasks to create
                - items_per_task (list): Number of items for each task (with load balancing)
                - start_indices (np.ndarray): Starting index for each task

        Example:
            >>> helper = Helper_Functions()
            >>> n_workers, n_tasks, items_per_task, start_indices = helper.calculate_parallel_chunks(1000)
            >>> # Process items in parallel using these chunks
        """
        import multiprocessing

        # Calculate number of workers (limit to avoid system overload)
        n_workers = min(
            max_workers, max(1, int(worker_ratio * multiprocessing.cpu_count()))
        )

        # Calculate number of tasks (more tasks than workers for load balancing)
        n_tasks = min(tasks_per_worker * n_workers, total_items)

        # Distribute items across tasks with load balancing
        # Tasks that get extra items: first (total_items % n_tasks) tasks
        items_per_task = [
            (
                int(total_items / n_tasks + 1)
                if i < total_items % n_tasks
                else int(total_items / n_tasks)
            )
            for i in range(n_tasks)
        ]

        # Calculate starting indices for each task
        start_indices = np.cumsum([0] + items_per_task[:-1])

        return n_workers, n_tasks, items_per_task, start_indices

    def load_metadata_roi(self, image_folder: Path | str, io_functions: Any, use_fallback: bool = True) -> tuple[int, int, int | None, int | None]:
        """
        Load ROI information from metadata files.

        Searches for metadata files in the image folder and loads ROI parameters.
        Optionally falls back to default values if no metadata is found.

        Args:
            image_folder (str): Path to folder containing metadata files
            io_functions: IOFunctions instance for reading metadata
            use_fallback (bool): If True, return (0, 0, None, None) when no metadata found.
                                 If False, assumes metadata exists (will raise error if missing)

        Returns:
            tuple: (start_x, start_y, width, height)
                   If use_fallback=True and no metadata: returns (0, 0, None, None)
                   Otherwise: returns actual ROI values from metadata
        """
        metadatafiles = self.file_search(image_folder, "metadata", "")

        if metadatafiles:
            # Load ROI from first metadata file
            start_x, start_y, width, height = io_functions.metadata_reader_imageJ(
                metadatafiles[0]
            )
            return start_x, start_y, width, height
        elif use_fallback:
            # No metadata found, use default (full image)
            return 0, 0, None, None
        else:
            # No metadata and no fallback allowed
            raise FileNotFoundError(f"No metadata files found in {image_folder}")

    def format_elapsed_time(self, elapsed_seconds: float) -> tuple[float, str]:
        """
        Format elapsed time in seconds to human-readable format.

        Converts elapsed time to appropriate units (seconds, minutes, or hours)
        based on magnitude.

        Args:
            elapsed_seconds (float): Elapsed time in seconds

        Returns:
            tuple: (elapsed_display, timestring)
                   elapsed_display (float): Time value in appropriate units
                   timestring (str): Unit string ("s", "min", or "hours")

        Examples:
            >>> format_elapsed_time(45.3)
            (45.3, "s")
            >>> format_elapsed_time(180.5)
            (3.008, "min")
            >>> format_elapsed_time(7320.0)
            (2.033, "hours")
        """
        from pyS3M.Constants import CalibrationConstants

        if elapsed_seconds > CalibrationConstants.TIME_DISPLAY_THRESHOLD_HOURS:
            elapsed_display = (
                elapsed_seconds / CalibrationConstants.TIME_DISPLAY_THRESHOLD_HOURS
            )
            timestring = "hours"
        elif elapsed_seconds > CalibrationConstants.TIME_DISPLAY_THRESHOLD_MINUTES:
            elapsed_display = (
                elapsed_seconds / CalibrationConstants.TIME_DISPLAY_THRESHOLD_MINUTES
            )
            timestring = "min"
        else:
            elapsed_display = elapsed_seconds
            timestring = "s"

        return elapsed_display, timestring
