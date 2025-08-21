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

IO = IOFunctions.IO_Functions()

import HelperFunctions

H_F = HelperFunctions.Helper_Functions()

from src import PlottingFunctions

plotter = PlottingFunctions.Plotter()

from src import postprocess as _postprocess
from sklearn.cluster import DBSCAN, HDBSCAN

class extract_SMs:
    def __init__(self) -> None:
        """Single molecule extraction functions for clustering localizations into single molecules."""
        pass

    def filter_quality_localizations(self, loc_data, chi_val=None, max_localization_error=1.0, 
                                    min_photons=500, max_photons=None):
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
            chi_val = np.median(loc_data['chi_sqr'])
        
        # Apply quality filters
        filtered_data = loc_data[loc_data['chi_sqr'] < chi_val].copy()
        filtered_data = filtered_data[filtered_data['xc_err'] < max_localization_error]
        filtered_data = filtered_data[filtered_data['yc_err'] < max_localization_error]
        
        # Add photons column using centralized method and apply photon count filters
        filtered_data = IO._add_photon_columns(filtered_data, normalize=False)
        filtered_data = filtered_data[filtered_data['photons'] < max_photons]
        filtered_data = filtered_data[filtered_data['photons'] > min_photons]
        
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
        dict_obj['photons'] = np.zeros(len(labels))
        dict_obj['frames'] = np.zeros(len(labels))
        for column in np.array(data.columns):
            if column == 'index':
                continue
            else:
                dict_obj[column] = np.zeros(len(labels))
        
        for label in labels:
            for column in np.array(data.columns):
                if column == 'index':
                    continue
                elif column in ['A_B', 'A_G', 'A_R']:
                    dict_obj[column][label] = np.sum(data[column][dbscan_labels == label])
                else:
                    dict_obj[column][label] = np.mean(data[column][dbscan_labels == label])
            dict_obj['frames'][label] = len(data[column][dbscan_labels == label])
            # Use existing photons column if available, otherwise calculate manually
            if 'photons' in data.columns:
                dict_obj['photons'][label] = np.sum(data['photons'][dbscan_labels == label])
            else:
                dict_obj['photons'][label] = np.sum(data['A_B'][dbscan_labels == label]) + np.sum(data['A_G'][dbscan_labels == label]) + np.sum(data['A_R'][dbscan_labels == label])
        df = pd.DataFrame.from_dict(dict_obj)
        # Normalize photon fractions using centralized IOFunctions method
        df = IO._add_photon_columns(df, normalize=True)
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
            locations[0, i] = np.nanmean(data['xc'][dbscan_labels == label].to_numpy())
            locations[1, i] = np.nanmean(data['yc'][dbscan_labels == label].to_numpy())
            xmin = int(locations[0, i])-int(image_size/2)
            xmax = int(locations[0, i])+int(image_size/2)
            ymin = int(locations[1, i])-int(image_size/2)
            ymax = int(locations[1, i])+int(image_size/2)
            trace_matrix[i, :] = np.sum(np.sum(image_stack[:, xmin:xmax, ymin:ymax], axis=-1), axis=-1)
            print(
                    "Summed trace {}/{}".format(
                        i + 1, len(labels)
                    ),
                    end="\r",
                    flush=True,
                )

        return locations, trace_matrix

    def extract_single_molecules(self, localisation_files, chi_val=None):
        """
        Extract single molecules from multiple localization files by clustering.
        
        Args:
            localisation_files (list): List of HDF5 localization file paths
            chi_val (float, optional): Chi-squared threshold for filtering. Defaults to median.
            
        Returns:
            tuple: (single_molecule_database, single_frame_database) as DataFrames
                   single_frame_database includes molecular_index column and excludes unassigned localizations
        """
        columns = ['xc', 'yc', 's_x', 's_y', 'bg_B', 'bg_G', 'bg_R', 'A_B', 'A_G', 'A_R',
           'chi_sqr', 'frame', 'xc_err', 'yc_err', 's_x_err', 's_y_err',
           'bg_B_err', 'bg_G_err', 'bg_R_err', 'A_B_err', 'A_G_err', 'A_R_err']

        single_frame_database_list = []
        single_molecule_database_list = []
        molecular_index_offset = 0

        for i, file in enumerate(localisation_files):
            loc_data = pd.read_hdf(file, columns=columns)
            loc_data = self.filter_quality_localizations(loc_data, chi_val)
            X = np.vstack([loc_data['xc'], loc_data['yc']]).T
            loc_precision = 0.5*(np.mean(loc_data['xc_err']) + np.mean(loc_data['yc_err']))
            hdb = HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=loc_precision)
            hdb.fit(X)
            
            # Filter out unassigned localizations (label = -1)
            assigned_mask = hdb.labels_ >= 0
            loc_data_assigned = loc_data[assigned_mask].copy()
            labels_assigned = hdb.labels_[assigned_mask]
            
            # Add molecular index column (offset by previous files)
            loc_data_assigned['molecular_index'] = labels_assigned + molecular_index_offset
            
            # Create single molecule database for this file
            df = self.average_parameters(loc_data_assigned, labels_assigned)
            df['molecular_index'] = df.index + molecular_index_offset
            
            # Normalize photon fractions for assigned localizations using centralized IOFunctions method
            loc_data_assigned = IO._add_photon_columns(loc_data_assigned, normalize=True)
            
            single_frame_database_list.append(loc_data_assigned)
            single_molecule_database_list.append(df)
            
            # Update offset for next file
            molecular_index_offset += len(df)

        # Concatenate all data
        single_frame_database = pd.concat(single_frame_database_list, ignore_index=True)
        single_molecule_database = pd.concat(single_molecule_database_list, ignore_index=True)
        
        return single_molecule_database, single_frame_database

    def extract_single_molecule_traces(self, localisation_file, smoothing_function, gain, offset, rqe, readnoise, chi_val=None):
        """
        Extract single molecule traces including intensity time series from image data.
        
        Args:
            localisation_file (str): Path to HDF5 localization file
            smoothing_function: Smoothing function object for image processing
            gain (np.array): Camera gain map
            offset (np.array): Camera offset map  
            rqe (np.array): Relative quantum efficiency map
            readnoise (np.array): Read noise map
            chi_val (float, optional): Chi-squared threshold. Defaults to median.
            
        Returns:
            tuple: (locations, trace_matrix, image_data) where locations are molecule positions,
                   trace_matrix contains intensity traces, and image_data is the full stack
        """
        columns = ['xc', 'yc', 's_x', 's_y', 'bg_B', 'bg_G', 'bg_R', 'A_B', 'A_G', 'A_R',
           'chi_sqr', 'frame', 'xc_err', 'yc_err', 's_x_err', 's_y_err',
           'bg_B_err', 'bg_G_err', 'bg_R_err', 'A_B_err', 'A_G_err', 'A_R_err']

        image_file = localisation_file.split('.')[0]+'.ome.tif'
        metadata = localisation_file.split('.')[0]+'_metadata.txt'
        x_coord, y_coord, width, height = IO.metadata_reader_imageJ(metadata)
        image_data, _, _ = IO.read_tiff_tophotoelectrons(image_file, smoothing_function, gain_map=gain[x_coord:x_coord+width, y_coord:y_coord+height], offset_map=offset[x_coord:x_coord+width, y_coord:y_coord+height], rqe=rqe[x_coord:x_coord+width, y_coord:y_coord+height], read_noise=readnoise[x_coord:x_coord+width, y_coord:y_coord+height], frame=np.arange(750))
        loc_data = pd.read_hdf(localisation_file, columns=columns)
        loc_data = self.filter_quality_localizations(loc_data, chi_val)
        X = np.vstack([loc_data['xc'], loc_data['yc']]).T
        loc_precision = 0.5*(np.mean(loc_data['xc_err']) + np.mean(loc_data['yc_err']))
        hdb = DBSCAN(min_samples=10, eps=loc_precision)
        hdb.fit(X)
        locations, trace_matrix = self.collect_traces(loc_data, hdb.labels_, image_data)
        return locations, trace_matrix, image_data
