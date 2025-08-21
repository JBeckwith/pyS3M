# -*- coding: utf-8 -*-
"""
This class contains functions pertaining to IO of files for POLCAM.
jsb92, 2024/01/02
"""
import json
import os
import tifffile
from tifffile import imread, imwrite
import numpy as np
import pandas as pd
import polars as pl
import sys
from copy import copy

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)


class IO_Functions:
    """File I/O operations for microscopy data and analysis results.

    Provides functionality for reading and writing various file formats
    used in single-molecule localization microscopy analysis.
    """

    def __init__(self):
        """Initialize IO_Functions class."""
        pass

    def _write_h5_database(self, df, filepath, append=False, normalize_photons=True):
        if df.shape[0] > 0:
            # Add photon columns if amplitude columns are present
            df = self._add_photon_columns(df, normalize=normalize_photons)

            if append and os.path.isfile(filepath):
                df.to_hdf(filepath, key="data", append=True, mode="r+", format="table")
            else:
                df.to_hdf(filepath, key="data", format="table")

    def _add_photon_columns(self, df, normalize=True):
        """
        Automatically add photon columns and optionally normalize amplitude/background data.

        Args:
            df (pd.DataFrame): Input dataframe
            normalize (bool): Whether to normalize A_B/A_G/A_R and bg_B/bg_G/bg_R columns

        Returns:
            pd.DataFrame: Dataframe with photon columns added and optionally normalized
        """
        df = df.copy()  # Avoid modifying original dataframe

        # Add total photons column and normalize A_B, A_G, A_R if they exist
        if all(col in df.columns for col in ["A_B", "A_G", "A_R"]):
            df["photons"] = df["A_B"] + df["A_G"] + df["A_R"]

            if normalize:
                # Avoid division by zero
                mask = df["photons"] > 0
                df.loc[mask, "A_B"] = df.loc[mask, "A_B"] / df.loc[mask, "photons"]
                df.loc[mask, "A_G"] = df.loc[mask, "A_G"] / df.loc[mask, "photons"]
                df.loc[mask, "A_R"] = df.loc[mask, "A_R"] / df.loc[mask, "photons"]

        # Add background photons column and normalize bg_B, bg_G, bg_R if they exist
        if all(col in df.columns for col in ["bg_B", "bg_G", "bg_R"]):
            df["background_photons"] = df["bg_B"] + df["bg_G"] + df["bg_R"]

            if normalize:
                # Avoid division by zero
                mask = df["background_photons"] > 0
                df.loc[mask, "bg_B"] = (
                    df.loc[mask, "bg_B"] / df.loc[mask, "background_photons"]
                )
                df.loc[mask, "bg_G"] = (
                    df.loc[mask, "bg_G"] / df.loc[mask, "background_photons"]
                )
                df.loc[mask, "bg_R"] = (
                    df.loc[mask, "bg_R"] / df.loc[mask, "background_photons"]
                )

        return df

    def _write_csv_dataframe(self, df, filepath, append=False, normalize_photons=False):
        if df.shape[0] > 0:
            # Add photon columns if amplitude columns are present
            df = self._add_photon_columns(df, normalize=normalize_photons)

            if append and os.path.isfile(filepath):
                with open(filepath, mode="ab") as f:
                    df.write_csv(f, include_header=False)
            else:
                df.write_csv(filepath)

    def read_json(self, filename, encoding="ISO-8859-1"):
        """
        read data from a JSON file.

        Args:
            filename (str): The name of the JSON file to load.

        Returns:
            data (dict): The loaded JSON data.
        """
        try:
            with open(filename, "r", encoding=encoding) as file:
                data = json.load(file)
        except:
            with open(filename, "r") as file:
                data = json.load(file)
        return data

    def get_num_pages_in_TIF(self, filename):
        """
        Loads metadata from a json file.

        Args:
            filename (str): The name of the tif file to load.

        Returns:
            n_pages (int): number of frames in TIFF file.
        """
        return len(tifffile.TiffFile(filename).pages)

    def metadata_reader_imageJ(self, filename):
        """
        Loads metadata from an imageJ json file.
        NB ImageJ starts its ROIs at (0,0), like Python

        Args:
            filename (str): The name of the json file to load.

        Returns:
            x_coord (int): starting x_coord pixel.
            y_coord (int): starting y_coord pixel.
            width (int): width
            height (int): height
        """
        data = self.read_json(filename)
        key = np.sort([x for x in data.keys() if "FrameKey" in x])[0]
        metadatadict = data[key]
        ROI = metadatadict["ROI"].split("-")
        x_coord = int(ROI[1])
        y_coord = int(ROI[0])
        width = int(ROI[3])
        height = int(ROI[2])
        return x_coord, y_coord, width, height

    def metadata_nframes_reader_imageJ(self, filename):
        """
        Loads metadata from an imageJ json file.
        NB ImageJ starts its ROIs at (0,0), like Python

        Args:
            filename (str): The name of the json file to load.

        Returns:
            n_frames (int): intended n_frames.
        """
        data = self.read_json(filename)
        n_frames = int(data["Summary"]["IntendedDimensions"]["time"])
        return n_frames

    def metadata_reader_Thorlabs(self, filename):
        """
        Loads metadata from a json file.

        Args:
            filename (str): The name of the json file to load.

        Returns:
            x_coord (int): starting x_coord pixel.
            y_coord (int): starting y_coord pixel.
            width (int): width
            height (int): height
        """
        data = self.read_json(filename)
        x_coord = int(data["ROIOriginX_pixels"])
        y_coord = int(data["ROIOriginY_pixels"])
        width = int(data["ROIWidth_pixels"])
        height = int(data["ROIHeight_pixels"])
        return x_coord, y_coord, width, height

    def make_directory(self, directory_path):
        """
        Creates a directory if it doesn't exist.

        Args:
            directory_path (str): The path of the directory to be created.
        """
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def write_json(self, data, file_name):
        """
        Saves data to a JSON file.

        Args:
            data (dict): The data to be saved in JSON format.
            file_name (str): The name of the JSON file.
        """
        with open(file_name, "w") as json_file:
            json.dump(data, json_file, indent=4)

    def read_tiff(self, file_path, frame=None, dtype="double", nchannels=1):
        """
        Read a TIFF file using the skimage library.

        Args:
            file_path (str): The path to the TIFF file to be read.
            frame (int): if not None, loads a single frame

        Returns:
            image (numpy.ndarray): The image data from the TIFF file.
        """
        if isinstance(frame, type(None)):
            image = np.asarray(
                imread(file_path, is_ome=False, is_mmstack=False, is_imagej=False),
                dtype=dtype,
            )
        else:
            if hasattr(frame, "__len__"):
                image = np.asarray(
                    imread(
                        file_path,
                        key=frame,
                        is_ome=False,
                        is_mmstack=False,
                        is_imagej=False,
                    ),
                    dtype=dtype,
                )
            else:
                image = np.asarray(
                    imread(
                        file_path,
                        key=int(frame),
                        is_ome=False,
                        is_mmstack=False,
                        is_imagej=False,
                    ),
                    dtype=dtype,
                )
        return image

    def read_tiff_tophotoelectrons(
        self,
        file_path,
        smoothing_function,
        dtype="double",
        gain_map=1.0,
        offset_map=0.0,
        rqe=1.0,
        read_noise=1.0,
        frame=None,
        hot_pixel_threshold=20,
    ):
        """
        Read a TIFF file using the skimage library.
        Use camera parameters to convert output to photoelectrons
        This uses the formula (eqn 3) of Lin et al (Lin, R., Clowsley, A. H.,
        Jayasinghe, I. D., Baddeley, D. & Soeller, C.
        Opt. Express, 25, 11701–11716 (2017))

        Args:
            file_path (str): The path to the TIFF file to be read.
            smoothing_function (type): function, args to smooth data
            dtype (str): data type to read out
            gain_map (matrix, or float): gain map. Assumes units of ADU/photoelectrons
            offset_map (matrix, or float): offset map. Assumes units of ADU
            rqe (matrix, or float): relative quantum yield map.
            read_noise (matrix, or float): read noise map of the camera
            frame (int, optional): if not None, loads a single frame

        Returns:
            image (numpy.ndarray): The image data from the TIFF file.
            smoothed_data (np.ndarray): Smoothed data for use in initial guesses etc.
            weights_map (np.ndarray): Weights for fitting of data.
        """
        # Use skimage's imread function to read the TIFF file
        # specifying the 'tifffile' plugin explicitly
        data = self.read_tiff(file_path, dtype=dtype, frame=frame)

        if type(gain_map) is not float:

            if data.shape[-2:] != gain_map.shape:
                print(
                    "Gain and offset map not compatible with image dimensions. Defaulting to gain of 1 and offset of 0."
                )
                gain_map = 1.0
                offset_map = 0.0

        if type(gain_map) is not float:
            if len(data.shape) > 2:
                data = np.divide(
                    np.divide(
                        np.subtract(data, offset_map[np.newaxis, :, :]),
                        gain_map[np.newaxis, :, :],
                    ),
                    rqe[np.newaxis, :, :],
                )
            else:
                data = np.divide(
                    np.divide(np.subtract(data, offset_map), gain_map), rqe
                )
        else:
            data = np.divide(np.divide(np.subtract(data, offset_map), gain_map), rqe)

        smoothed_data = copy(data)

        smoothing_args = smoothing_function.args
        smoothing_args[smoothing_function.data_arg] = smoothed_data
        smoothed_data = smoothing_function.smoothing_function(**smoothing_args)

        error_data = copy(smoothed_data)
        error_data[error_data < 0] = 0
        error_data = error_data + 1
        if type(read_noise) is not float:
            if len(data.shape) > 2:
                error_map = np.add(error_data, np.square(read_noise[np.newaxis, :, :]))
            else:
                error_map = np.add(error_data, np.square(read_noise))
        else:
            error_map = np.add(error_data, np.square(read_noise))
        weights_map = np.power(error_map, -1)
        if type(read_noise) is not float:
            hot_pixels = read_noise > hot_pixel_threshold
            if len(data.shape) > 2:
                hot_pixels = np.tile(hot_pixels, (data.shape[0], 1, 1))
            weights_map[hot_pixels] = 1e-8
        return data, smoothed_data, weights_map

    def write_tiff(self, volume, file_path, bit="double", pixel_size=0.11):
        """
        Write a TIFF file using the skimage library.

        Args:
            volume (numpy.ndarray): The volume data to be saved as a TIFF file.
            file_path (str): The path where the TIFF file will be saved.
            bit (int): Bit-depth for the saved TIFF file (default is 16).

        Notes:
            The function uses skimage's imsave to save the volume as a TIFF file.
            The plugin is set to 'tifffile' and photometric to 'minisblack'.
            Additional metadata specifying the software as 'Python' is included.
        """
        xamount = str(volume.shape[-2])
        yamount = str(volume.shape[-1])

        description = "ImageJ=1.54f\nunit=micron\nmin=" + xamount + "\nmax=" + yamount

        pixel_unit = int(1e6 / pixel_size)

        extra_tags = [
            ("ImageDescription", "s", 1, description, True),
            ("XResolution", "i", 2, (pixel_unit, 1000000), True),
            ("YResolution", "i", 2, (pixel_unit, 1000000), True),
            ("ResolutionUnit", "i", 1, True),
        ]

        imwrite(
            file_path,
            np.asarray(volume, dtype=bit),
            extratags=extra_tags,
        )

    def save_simulation_results(
        self,
        save_folder,
        starting_flag,
        default_params,
        n_photon_space,
        fit_RMSE_mean,
        fit_RMSE_std,
        pixel_size,
        NA,
        background_photons,
        fit_function_name,
        smoothing_function_name,
        smoothing_function_extent,
        dye,
    ):
        """
        Saves simulation analysis.

        Args:
            save_folder (string): Folder to save data to.
            starting_flat (string): Starting flag of data saving.
            default_params (array): Array of default parameters to be saved.
            n_photon_space (np.1darray): 1d array of different photon values to save.
            fit_RMSE_mean (np.2darray): 2d array of fit RMSEs to save
            fit_RMSE_std (np.2darray): 2d array of fit stds to save
            pixel_size (float): pixel size of simulations
            NA (float): NA of simulations
            background_photons (float): background of simulations
            fit_function_name (object): fit function name to save
            smoothing_function_name (str): smoothing function name to save
            smoothing_function_extent (int): default extent of smoothing function
            dye (string): dye string
        """
        parameters_to_save = list(
            np.concatenate([np.array(["n_photons"]), default_params])
        )
        means = pl.DataFrame(
            data=np.vstack([n_photon_space, fit_RMSE_mean]).T,
            schema=parameters_to_save,
        )
        stds = pl.DataFrame(
            data=np.vstack([n_photon_space, fit_RMSE_std]).T,
            schema=parameters_to_save,
        )
        if int(pixel_size) == pixel_size:
            px_save = str(int(pixel_size))
        else:
            px_save = str(np.around(pixel_size, 1)).replace(".", "p")
        if int(NA) == NA:
            NA_save = str(int(NA))
        else:
            NA_save = str(np.around(NA, 2)).replace(".", "p")
        if int(background_photons) == background_photons:
            b_save = str(int(background_photons))
        else:
            b_save = str(np.around(background_photons, 2)).replace(".", "p")
        if int(smoothing_function_extent) == smoothing_function_extent:
            sf_e_save = str(int(smoothing_function_extent)).zfill(6)
        else:
            sf_e_save = (
                str(np.around(smoothing_function_extent, 2)).replace(".", "p").zfill(6)
            )

        dyestr = dye.replace("/", "-")
        means.write_csv(
            os.path.join(
                save_folder,
                starting_flag
                + str(fit_function_name)
                + "_smoothingfunction_"
                + str(smoothing_function_name)
                + "_smoothingextent_"
                + sf_e_save
                + "_"
                + dyestr
                + "_"
                + px_save
                + "_pixelsize_"
                + NA_save
                + "_NA_"
                + b_save
                + "_background_RMSE_mean_bootstrapping.csv",
            )
        )
        stds.write_csv(
            os.path.join(
                save_folder,
                starting_flag
                + str(fit_function_name)
                + "_smoothingfunction_"
                + str(smoothing_function_name)
                + "_smoothingextent_"
                + sf_e_save
                + "_"
                + dyestr
                + "_"
                + px_save
                + "_pixelsize_"
                + NA_save
                + "_NA_"
                + b_save
                + "_background_RMSE_std_bootstrapping.csv",
            )
        )
        return

    def save_simulation_results_pixelsize(
        self,
        save_folder,
        starting_flag,
        default_params,
        pixel_size_space,
        fit_RMSE_mean,
        fit_RMSE_std,
        n_photon,
        NA,
        fit_function_name,
        error_type,
        dye,
    ):
        """
        Saves simulation analysis.

        Args:
            save_folder (string): Folder to save data to.
            starting_flat (string): Starting flag of data saving.
            default_params (array): Array of default parameters to be saved.
            n_photon_space (np.1darray): 1d array of different photon values to save.
            fit_RMSE_mean (np.2darray): 2d array of fit RMSEs to save
            fit_RMSE_std (np.2darray): 2d array of fit stds to save
            pixel_size (float): pixel size of simulations
            NA (float): NA of simulations
            fit_function_name (object): fit function name to save
            error_type (object): error type to save
            dye (string): dye string
        """
        parameters_to_save = list(
            np.concatenate([np.array(["pixel_size_nm"]), default_params])
        )
        means = pl.DataFrame(
            data=np.vstack([pixel_size_space, fit_RMSE_mean]).T,
            schema=parameters_to_save,
        )
        stds = pl.DataFrame(
            data=np.vstack([pixel_size_space, fit_RMSE_std]).T,
            schema=parameters_to_save,
        )
        if int(n_photon) == n_photon:
            px_save = str(int(n_photon))
        else:
            px_save = str(np.around(n_photon, 1)).replace(".", "p")
        if int(NA) == NA:
            NA_save = str(int(NA))
        else:
            NA_save = str(np.around(NA, 2)).replace(".", "p")
        means.write_csv(
            os.path.join(
                save_folder,
                starting_flag
                + str(fit_function_name)
                + "_error_"
                + str(error_type)
                + "_"
                + dye
                + "_"
                + px_save
                + "_nphoton_"
                + NA_save
                + "_NA_RMSE_mean_bootstrapping.csv",
            )
        )
        stds.write_csv(
            os.path.join(
                save_folder,
                starting_flag
                + str(fit_function_name)
                + "_error_"
                + str(error_type)
                + "_"
                + dye
                + "_"
                + px_save
                + "_nphoton_"
                + NA_save
                + "_NA_RMSE_std_bootstrapping.csv",
            )
        )
        return
