#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  # import numerical python
import matplotlib.pyplot as plt  # import plotting functions
import seaborn as sns  # import nicer plotting functions
import polars as pl  # import polars to import data
import tifffile as tiff
from tifffile import imwrite, imread
from copy import deepcopy
import os
import time
import pandas as pd
from datetime import datetime, timedelta

from IPython.display import clear_output

import sys
sys.path.append("..")

from src import IOFunctions

IO = IOFunctions.IO_Functions()

from src import Multicolour_Simulation_Functions

MSF = Multicolour_Simulation_Functions.MultiC_Sim_Funcs()

from src import PlottingFunctions

plotter = PlottingFunctions.Plotter()

from src import ImageAnalysisFunctions

I_AF = ImageAnalysisFunctions.Image_Analysis_Functions()

from src import sCMOSFunctions

sCMOS = sCMOSFunctions.sCMOS_Functions()

from src import PSFFunctions

PSF = PSFFunctions.PSF_Functions()

from src import SpectralFunctions

S_F = SpectralFunctions.Spectral_Funcs()

from src import MaskFunctions

M_F = MaskFunctions.Mask_Functions()

from src import SpotDetectionFunctions

SD_F = SpotDetectionFunctions.SpotDetection_Functions()

from src import SR_Functions

SupRes_F = SR_Functions.SuperRes_Functions()

from io import BytesIO
from PIL import Image
import HelperFunctions
import shutil

H_F = HelperFunctions.Helper_Functions()


# In[2]:


data_folder = '../Camera_Calibrations/Ximea_Camera'
gain_map = IO.read_tiff(os.path.join(data_folder, "gain.tif"))
offset_map = IO.read_tiff(os.path.join(data_folder, "offset.tif"))
variance = IO.read_tiff(os.path.join(data_folder, "variance.tif"))
read_noise = IO.read_tiff(os.path.join(data_folder, "readnoise.tif"))
rqe = IO.read_tiff(os.path.join(data_folder, "rqe.tif"))
R, G, B, wavelength = S_F.getpixelefficiency()

pixel_QYs = np.vstack([B, G, R])
camera_parameters = {}
camera_parameters["pixel_QYs"] = pixel_QYs
camera_parameters["pixel_order"] = ['B', 'G', 'R']
camera_parameters["pixel_order_indices"] = [0, 1, 2]


# In[3]:


def copy_file_to_scratch(file, new_folder):
    try:
        os.makedirs(new_folder, exist_ok=True)
        new_file = os.path.join(new_folder, os.path.split(file)[-1])
        shutil.copyfile(file, new_file)
    except:
        time.sleep(10)
        copy_file_to_scratch(file, new_folder)


# In[4]:


def copy_file_from_scratch(file, new_file):
    try:
        os.makedirs(os.path.split(new_file)[0], exist_ok=True)
        shutil.copyfile(file, new_file)
    except:
        time.sleep(10)
        copy_file_from_scratch(file, new_file)


# In[5]:


def copy_folder_to_scratch(files, new_folder):
    for file in files:
        copy_file_to_scratch(file, new_folder)


# In[6]:


def delete_folder(folder):
    try:
        shutil.rmtree(folder)
    except:
        time.sleep(10)
        delete_folder(folder)


# In[7]:


def copy_from_scratch(new_folder, folder, filetype='.h5'):
    files = np.sort([x for x in os.listdir(new_folder) if filetype in x])
    for file in files:
        copy_file_from_scratch(os.path.join(new_folder, file), os.path.join(folder, file))


# In[8]:


def should_analyse_folder(folder_path, cutoff_time=None):
    """
    Check if folder should be analysed based on .h5 file timestamps.
    
    Args:
        folder_path (str): Path to the folder to check
        cutoff_time (datetime, optional): Cutoff time. If None, uses 10am on August 26th, 2025
        
    Returns:
        bool: True if folder should be analysed (no .h5 files or all .h5 files are from before cutoff_time)
    """
    if cutoff_time is None:
        # Set cutoff to 10am on August 26th, 2025
        cutoff_time = datetime(2025, 8, 26, 10, 0, 0)
    
    # Get all .h5 files in the folder
    h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    
    # If no .h5 files exist, proceed with analysis
    if not h5_files:
        print(f"No .h5 files found in {folder_path} - proceeding with analysis")
        return True
    
    # Check timestamp of each .h5 file
    for h5_file in h5_files:
        file_path = os.path.join(folder_path, h5_file)
        # Get file modification time
        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
        
        if file_mtime >= cutoff_time:
            print(f"Found .h5 file {h5_file} from {file_mtime} (after {cutoff_time}) - skipping analysis")
            return False
    
    print(f"All .h5 files in {folder_path} are from before {cutoff_time} - proceeding with analysis")
    return True


# In[9]:


import types
smoothing_function = types.SimpleNamespace()
smoothing_function.args = {"sigma" :  1.5}
smoothing_function.extent =  1.5
smoothing_function.smoothing_function = sCMOS.gaussian_filter_stack
smoothing_function.data_arg = "image"


# In[ ]:


image_folder = '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250819_TetraspeckCalibration'



dye_folders = np.array(['/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250819_TetraspeckCalibration',
                        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250717 BiotinDyes/ATTO488_50PM_PCA_PCD',
                        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250725 biotinylated dyes/ATTO514_50pM_PCAPCDTx',
                        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250725 biotinylated dyes/ATTO520_50pM_PCAPCDTx',
                        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250725 biotinylated dyes/ATTORho6G_50pM_PCAPCDTx',
                        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto565_PCA_PCD_Tx_50pMDye',
                        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto620_PCA_PCD_Tx_50pMDye',
                        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250711 Biotinylated Dyes/Atto633_PCA_PCD_Tx_100pMDye',
                        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto647N_PCA_PCD_Tx_20pMDye',
                        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/29250717 BiotinDyes/ATTO655_50PM _PCA_PCD',
                        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/29250717 BiotinDyes/ATTO700_50PM _PCA_PCD',
                        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250609_dyes/data',
                        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto594_PCA_PCD_Tx_50pMDye'])


# In[ ]:


for starting_directory in dye_folders:
    lowest_dirs = list()
    for root,dirs,files in os.walk(starting_directory):
        if not dirs:
            lowest_dirs.append(root)
    for image_folder in np.sort(lowest_dirs):
        # Check if folder should be analysed based on .h5 file timestamps
        if not should_analyse_folder(image_folder):
            continue
            
        new_folder = os.path.join('/scratch2/jsb92', os.path.split(image_folder)[-1])
        files_in_folder = [os.path.join(image_folder, x) for x in os.listdir(image_folder) if '.h5' not in x]
        copy_folder_to_scratch(files_in_folder, new_folder)
        SupRes_F.fit_SM_data(
                new_folder,
                smoothing_function,
                gain_map,
                offset_map,
                rqe,
                read_noise,
                variance=variance,
                pfa=1e-4,
                ROI_size=12,
                peak_wavelength=0.6,
                NA=1.49,
                pixel_size=0.069,
                image_type=".tif",
            )
        copy_from_scratch(new_folder, image_folder, filetype='.h5')
        delete_folder(new_folder)


# In[ ]:


image_folder = '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell3_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1'

# Check if folder should be analysed based on .h5 file timestamps
if should_analyse_folder(image_folder):
    new_folder = os.path.join('/scratch2/jsb92', os.path.split(image_folder)[-1])
    files_in_folder = [os.path.join(image_folder, x) for x in os.listdir(image_folder) if '.h5' not in x]
    copy_folder_to_scratch(files_in_folder, new_folder)
    SupRes_F.fit_imaging_data(
            new_folder,
            smoothing_function,
            gain_map,
            offset_map,
            rqe,
            read_noise,
            variance=variance,
            pfa=1e-4,
            ROI_size=12,
            peak_wavelength=0.647,
            NA=1.49,
            pixel_size=0.069,
            image_type=".tif",
        )
    copy_from_scratch(new_folder, image_folder, filetype='.h5')
    delete_folder(new_folder)


# In[ ]:


image_folder = '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell4_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1'

# Check if folder should be analysed based on .h5 file timestamps
if should_analyse_folder(image_folder):
    new_folder = os.path.join('/scratch2/jsb92', os.path.split(image_folder)[-1])
    files_in_folder = [os.path.join(image_folder, x) for x in os.listdir(image_folder) if '.h5' not in x]
    copy_folder_to_scratch(files_in_folder, new_folder)
    SupRes_F.fit_imaging_data(
            new_folder,
            smoothing_function,
            gain_map,
            offset_map,
            rqe,
            read_noise,
            variance=variance,
            pfa=1e-4,
            ROI_size=12,
            peak_wavelength=0.647,
            NA=1.49,
            pixel_size=0.069,
            image_type=".tif",
        )
    copy_from_scratch(new_folder, image_folder, filetype='.h5')
    delete_folder(new_folder)


# In[9]:


image_folder = '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell2_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1'

# Check if folder should be analysed based on .h5 file timestamps
if should_analyse_folder(image_folder):
    new_folder = os.path.join('/scratch2/jsb92', os.path.split(image_folder)[-1])
    files_in_folder = [os.path.join(image_folder, x) for x in os.listdir(image_folder) if '.h5' not in x]
    copy_folder_to_scratch(files_in_folder, new_folder)
    SupRes_F.fit_imaging_data(
            new_folder,
            smoothing_function,
            gain_map,
            offset_map,
            rqe,
            read_noise,
            variance=variance,
            pfa=1e-4,
            ROI_size=12,
            peak_wavelength=0.647,
            NA=1.49,
            pixel_size=0.069,
            image_type=".tif",
        )
    copy_from_scratch(new_folder, image_folder, filetype='.h5')
    delete_folder(new_folder)


# In[10]:


image_folder = '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell1_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_2'

# Check if folder should be analysed based on .h5 file timestamps
if should_analyse_folder(image_folder):
    new_folder = os.path.join('/scratch2/jsb92', os.path.split(image_folder)[-1])
    files_in_folder = [os.path.join(image_folder, x) for x in os.listdir(image_folder) if '.h5' not in x]
    copy_folder_to_scratch(files_in_folder, new_folder)
    SupRes_F.fit_imaging_data(
            new_folder,
            smoothing_function,
            gain_map,
            offset_map,
            rqe,
            read_noise,
            variance=variance,
            pfa=1e-4,
            ROI_size=12,
            peak_wavelength=0.647,
            NA=1.49,
            pixel_size=0.069,
            image_type=".tif",
        )
    copy_from_scratch(new_folder, image_folder, filetype='.h5')
    delete_folder(new_folder)


# In[11]:


folders = np.array(['/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250717_Origami/F1F2F3F4Cy3B500pM/10perc561_LP561_BP586-64_1',
                    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250717_Origami/F1F2F3F4Cy3B500pM_LowConcOrigami/10perc561_LP561_BP586-64_1',
                    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250514_DNANanoruler/data/DNANanoRuler_10perc561_30mW488_50mW638/F1CF640CF550R_F2ATTO488AF647_F3ATTO565ATTO655_F4Cy3BCF488A_MultiNotch_488LP_758SP_1',
                    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250514_DNANanoruler/data/DNANanoRuler_10perc561_30mW488_50mW638/F1CF640CF550R_F2ATTO488AF647_F3ATTO565ATTO655_F4Cy3BCF488A_MultiNotch_488LP_758SP_1nM_1',
                    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250730 single colour origami/AlexaFluor647_2nM_strands/30mWboth638_NF_785SP_488LP_1',
                    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250730 single colour origami/CF488A_2nM_strands/20mW488_NF_785SP_488LP_1',
                    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250730 single colour origami/CF550R_2nM_strands_adjusteddichroic/30p561_NF_785SP_488LP_1',
                    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250730 single colour origami/CF640R_2nM_strands/30mWboth638_NF_785SP_488LP_1',
                    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250723 DNA Origami/FourColour_F1AF647_F2ATTO565_F3Cy3B_F4ATTO655_500pMEach/15percent_561_40mWEach_638_NotchFilter_785SP_1',
                    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250723 DNA Origami/FourColour_F1AF647_F2ATTO565_F3Cy3B_F4ATTO655_500pMEach/15percent_561_100mWEach_638_NotchFilter_785SP_1',
                    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250723 DNA Origami/FourColour_F1AF647_F2ATTO565_F3Cy3B_F4CF488A_500pMEach/30mW_488_15percent_561_100mWEach_638_NotchFilter_785SP_1',
                    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250723 DNA Origami/FourColour_F1CF550R_F2ATTO565_F3Cy3B_F4CF488A_500pMEach/30mW_488_15percent_561_NotchFilter_785SP_1',
                    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250716_iPSCJamesEvans/40mW488_30perc561_50mW638_NF_488LP_785SP_1',
                    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250716_iPSCJamesEvans/250pMCy3B_250pM565_250pMCF550_250pM647/20perc561_40mW638_NF_488LP_785SP_1'])


# In[12]:


for image_folder in folders:
    # Check if folder should be analysed based on .h5 file timestamps
    if not should_analyse_folder(image_folder):
        continue
        
    new_folder = os.path.join('/scratch2/jsb92', os.path.split(image_folder)[-1])
    files_in_folder = [os.path.join(image_folder, x) for x in os.listdir(image_folder) if '.h5' not in x]
    copy_folder_to_scratch(files_in_folder, new_folder)
    SupRes_F.fit_imaging_data(
            new_folder,
            smoothing_function,
            gain_map,
            offset_map,
            rqe,
            read_noise,
            variance=variance,
            pfa=1e-4,
            ROI_size=12,
            peak_wavelength=0.55,
            NA=1.49,
            pixel_size=0.069,
            image_type=".tif",
        )
    copy_from_scratch(new_folder, image_folder, filetype='.h5')
    delete_folder(new_folder)


# In[9]:


starting_directory = '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250414_CellPAINT/data'
lowest_dirs = list()
for root,dirs,files in os.walk(starting_directory):
    if not dirs:
        lowest_dirs.append(root)


# In[10]:


lowest_dirs = np.sort([x for x in lowest_dirs if 'WL_image' not in x])
lowest_dirs = np.sort([x for x in lowest_dirs if 'WL_Image' not in x])


# In[ ]:


for image_folder in np.sort(lowest_dirs):
    # Check if folder should be analysed based on .h5 file timestamps
    if not should_analyse_folder(image_folder):
        continue
        
    new_folder = os.path.join('/scratch2/jsb92', os.path.split(image_folder)[-1])
    files_in_folder = [os.path.join(image_folder, x) for x in os.listdir(image_folder) if '.h5' not in x]
    copy_folder_to_scratch(files_in_folder, new_folder)
    SupRes_F.fit_imaging_data(
            new_folder,
            smoothing_function,
            gain_map,
            offset_map,
            rqe,
            read_noise,
            variance=variance,
            pfa=1e-4,
            ROI_size=12,
            peak_wavelength=0.6,
            NA=1.49,
            pixel_size=0.069,
            image_type=".tif",
        )
    copy_from_scratch(new_folder, image_folder, filetype='.h5')
    delete_folder(new_folder)


# In[ ]:


starting_directory = '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250404_Ximea_AsynNRThX/data'
lowest_dirs = list()
for root,dirs,files in os.walk(starting_directory):
    if not dirs:
        lowest_dirs.append(root)


# In[ ]:


for image_folder in np.sort(lowest_dirs):
    # Check if folder should be analysed based on .h5 file timestamps
    if not should_analyse_folder(image_folder):
        continue
        
    new_folder = os.path.join('/scratch2/jsb92', os.path.split(image_folder)[-1])
    files_in_folder = [os.path.join(image_folder, x) for x in os.listdir(image_folder)]
    copy_folder_to_scratch(files_in_folder, new_folder)
    SupRes_F.fit_imaging_data(
            new_folder,
            smoothing_function,
            gain_map,
            offset_map,
            rqe,
            read_noise,
            variance=variance,
            pfa=1e-4,
            ROI_size=12,
            peak_wavelength=0.6,
            NA=1.49,
            pixel_size=0.069,
            image_type=".tif",
        )
    copy_from_scratch(new_folder, image_folder, filetype='.h5')
    delete_folder(new_folder)


# In[ ]:


folder = '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250818_DNAOrigami'
lowest_dirs = list()
for root,dirs,files in os.walk(starting_directory):
    if not dirs:
        lowest_dirs.append(root)


# In[ ]:


for image_folder in np.sort(lowest_dirs):
    # Check if folder should be analysed based on .h5 file timestamps
    if not should_analyse_folder(image_folder):
        continue
        
    new_folder = os.path.join('/scratch2/jsb92', os.path.split(image_folder)[-1])
    files_in_folder = [os.path.join(image_folder, x) for x in os.listdir(image_folder) if '.h5' not in x]
    copy_folder_to_scratch(files_in_folder, new_folder)
    SupRes_F.fit_imaging_data(
            new_folder,
            smoothing_function,
            gain_map,
            offset_map,
            rqe,
            read_noise,
            variance=variance,
            pfa=1e-4,
            ROI_size=12,
            peak_wavelength=0.6,
            NA=1.49,
            pixel_size=0.069,
            image_type=".tif",
        )
    copy_from_scratch(new_folder, image_folder, filetype='.h5')
    delete_folder(new_folder)


# In[ ]:




