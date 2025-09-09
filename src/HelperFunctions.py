# -*- coding: utf-8 -*-
"""
This class contains helper functions pertaining to analysis of images based on their
radiality, relating to the RASP concept.
jsb92, 2024/01/02
"""
import numpy as np
import polars as pl
import os
import fnmatch


class Helper_Functions:
    """Helper functions for image analysis and data processing.

    Provides utility functions for database cleaning, file operations,
    and radiality analysis related to the RASP (Radiality Analysis of
    Single-molecule Positions) concept.
    """

    def __init__(self):
        """Initialize Helper_Functions class."""
        pass

    def clean_database(self, database, columns):
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

    def file_search(self, folder, string1, string2):
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
            alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
            return sorted(data, key=alphanum_key)
        # Get a list of all files containing 'string1' in their names within 'folder'
        file_list = [
            os.path.join(dirpath, f)
            for dirpath, dirnames, files in os.walk(folder)
            for f in fnmatch.filter(files, "*" + string1 + "*")
        ]
        file_list = np.sort([e for e in file_list if string2 in e])
        return sorted_alphanumeric(file_list)
