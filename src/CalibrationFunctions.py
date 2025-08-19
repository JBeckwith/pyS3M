#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:27:38 2024

@author: jbeckwith
"""

import numpy as np
import os
import sys
import time
import scipy
import scipy.ndimage

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)
import IOFunctions

IO = IOFunctions.IO_Functions()

import MaskFunctions


class Calibration_Functions:
    """Camera calibration routines for sCMOS cameras and Bayer filter patterns.

    Provides functionality for calibrating camera parameters, processing
    gain/offset/variance maps, and handling Bayer pattern configurations.
    """

    def __init__(self, mosaic_unit=None, high_memory=False):
        """Initialize Calibration_Functions class.

        Args:
            mosaic_unit: Optional custom Bayer mosaic pattern.
                        Defaults to standard [["B", "G"], ["G", "R"]] pattern.
            high_memory: Whether to use high-memory processing mode.
        """
        self.high_memory = high_memory
        if isinstance(mosaic_unit, type(None)):
            self.mosaic_unit = np.array([["B", "G"], ["G", "R"]])
        else:
            self.mosaic_unit = mosaic_unit
        self.Mask = MaskFunctions.Mask_Functions()

    def filesearch(self, directory, string1, string2):
        files = os.listdir(directory)
        files = np.sort([x for x in files if string1 in x])
        files = np.sort([x for x in files if string2 in x])
        return files

    def calibrate_multicolour_camera(self, directory, imtype=".tif"):
        """
        Calibrates multicolour camera.

        Args:
            directory (string): Folder containing tifs
            imtype (string): image type to read in

        Returns:
            offset (np.2darray): offset matrix
            variance (np.2darray): variance matrix
            gain (np.2darray): gain matrix
            read_noise (np.2darray): read noise matrix
            r_QE (np.2darray): relative QE matrix
        """
        colours = np.unique(self.mosaic_unit)

        directories = os.listdir(directory)
        colour_directories = np.sort([x for x in directories if x in colours])
        dark_directory = np.sort([x for x in directories if "dark" in x])
        try:
            if len(dark_directory) != 1:
                raise Exception("Incorrect number of dark folders; should be 1")
        except Exception as error:
            print("Caught this error: " + repr(error))
            return
        dark_directory = str(dark_directory[0])

        n_powermatrix = np.zeros(len(colour_directories))
        for i, colour in enumerate(colour_directories):
            files = np.unique(os.listdir(os.path.join(directory, colour)))
            files = np.sort([x for x in files if imtype in x])
            intensity_strings = np.sort(
                np.unique([x.split("Intensity_")[1].split("_")[0] for x in files])
            )
            intensity_strings = np.asarray(
                np.full(len(intensity_strings), "Intensity_")
                + np.asarray(intensity_strings, dtype="object"),
                dtype=str,
            )
            n_powermatrix[i] = len(intensity_strings)

        try:
            if n_powermatrix.ptp() != 0:
                raise Exception(
                    "Incorrect number of intensity values for each colour; should be equal"
                )
        except Exception as error:
            print("Caught this error: " + repr(error))
            return

        n_powers = int(np.unique(n_powermatrix))
        # calculate dark offset and variance for all colours
        offset = self.calculate_offset(os.path.join(directory, dark_directory), "dark")
        variance = self.calculate_variance(
            offset, os.path.join(directory, dark_directory), "dark"
        )

        offset_intensities = np.zeros([offset.shape[0], offset.shape[1], n_powers])
        variance_intensities = np.zeros(
            [variance.shape[0], variance.shape[1], n_powers]
        )

        masks = self.Mask.get_masks(
            mosaic_unit=self.mosaic_unit, size_x=offset.shape[0], size_y=offset.shape[1]
        )

        for i, intensity in enumerate(intensity_strings):
            for colour in colour_directories:
                offset_intensities[:, :, i] = offset_intensities[:, :, i] + np.asarray(
                    np.multiply(
                        self.calculate_offset(
                            os.path.join(directory, colour), intensity
                        ),
                        masks[str(colour)],
                    ),
                    dtype=float,
                )
                variance_intensities[:, :, i] = variance_intensities[
                    :, :, i
                ] + np.asarray(
                    np.multiply(
                        self.calculate_variance(
                            offset_intensities[:, :, i],
                            os.path.join(directory, colour),
                            intensity,
                        ),
                        masks[str(colour)],
                    ),
                    dtype=float,
                )

        A = np.subtract(variance_intensities, variance[:, :, np.newaxis])
        B = np.subtract(offset_intensities, offset[:, :, np.newaxis])

        A_filepath = os.path.join(directory, "A.tif")
        B_filepath = os.path.join(directory, "B.tif")
        IO.write_tiff(
            np.swapaxes(np.swapaxes(A, -1, 0), -1, -2),
            A_filepath,
            bit=float,
            pixel_size=3.45,
        )
        IO.write_tiff(
            np.swapaxes(np.swapaxes(B, -1, 0), -1, -2),
            B_filepath,
            bit=float,
            pixel_size=3.45,
        )

        gain = np.zeros_like(offset)

        for i in np.arange(gain.shape[0]):
            for j in np.arange(gain.shape[1]):
                Ai = A[i, j, :]
                Bi = B[i, j, :]
                gain[i, j] = np.asarray(
                    np.dot(
                        np.linalg.pinv(np.matrix(np.dot(Bi, Bi.T))), np.dot(Bi, Ai.T)
                    ),
                    dtype=float,
                )[0][0]

        readnoise = np.divide(np.sqrt(variance), gain)
        rqe = self.calculate_rqe(offset_intensities[:, :, -1], offset, gain)

        offset_file_path = os.path.join(directory, "offset.tif")
        gain_file_path = os.path.join(directory, "gain.tif")
        variance_file_path = os.path.join(directory, "variance.tif")
        readnoise_file_path = os.path.join(directory, "readnoise.tif")
        rqe_file_path = os.path.join(directory, "rqe.tif")

        # write tiffs
        IO.write_tiff(offset, offset_file_path, bit=float, pixel_size=3.45)
        IO.write_tiff(variance, variance_file_path, bit=float, pixel_size=3.45)
        IO.write_tiff(gain, gain_file_path, bit=float, pixel_size=3.45)
        IO.write_tiff(readnoise, readnoise_file_path, bit=float, pixel_size=3.45)
        IO.write_tiff(rqe, rqe_file_path, bit=float, pixel_size=3.45)

        print(
            "The average offset is {:.3f} +- {:.3f} ADU counts".format(
                np.nanmean(offset), np.nanstd(offset)
            ),
            end="\r",
        )
        print(
            "The average variance is {:.3f} +- {:.3f} ADU counts".format(
                np.nanmean(variance), np.nanstd(variance)
            ),
            end="\r",
        )
        print(
            "The average gain is {:.3f} +- {:.3f} ADU counts/photoelectron".format(
                np.nanmean(gain), np.nanstd(gain)
            ),
            end="\r",
        )
        print(
            "The average read noise is {:.3f} +- {:.3f} photoelectrons".format(
                np.nanmean(readnoise), np.nanstd(readnoise)
            ),
            end="\r",
        )
        print(
            "The median read noise is {:.3f} photoelectrons".format(
                np.nanmedian(readnoise)
            ),
            end="\r",
        )
        print(
            "The RMS read noise is {:.3f} photoelectrons".format(
                np.sqrt(np.nanmean(np.square(readnoise)))
            ),
            end="\r",
        )
        return offset, variance, gain, readnoise, rqe

    def calculate_rqe(self, intensity_image, offset, gain):
        """
        Calibrates relative QE. Given an intensity image, an offset and a gain, calculate a relative QE.

        Args:
            intensity_image (np.2darray): Intensity image
            offset (np.2darray): 2d matrix of offset
            gain (np.2darray): 2d matrix of gain

        Returns:
            rqe (np.2darray): rqe matrix
        """
        corrected_image = np.divide(np.subtract(intensity_image, offset), gain)
        smoothed_image = scipy.ndimage.uniform_filter(
            corrected_image, size=10, mode="nearest"
        )
        rqe = np.divide(corrected_image, smoothed_image)
        return rqe

    def calculate_offset(self, directory, intensity_string, imtype=".tif"):
        """
        Calibrates offset. Given a directory, looks for a particular intensity
        string and loads these images. Then gets an offset.

        Args:
            directory (string): Folder containing tifs
            intensity_string (string): Intensity string
            imtype (string): image type to read in

        Returns:
            offset (np.2darray): offset matrix
        """
        filelist = self.filesearch(directory, imtype, intensity_string)

        frame0_shape = IO.read_tiff(os.path.join(directory, filelist[0]), 0).shape
        width = frame0_shape[0]
        height = frame0_shape[1]
        framesCounter = 0
        offset = np.zeros([width, height])
        if directory.split("/")[-1] == intensity_string:
            print(
                "Starting offset analysis of " + intensity_string,
                end="\r",
                flush=True,
            )
        else:
            print(
                "Starting offset analysis of "
                + directory.split("/")[-1]
                + " "
                + intensity_string,
                end="\r",
                flush=True,
            )
        start = time.time()
        for i, file in enumerate(filelist):
            if self.high_memory == True:
                image = IO.read_tiff(os.path.join(directory, file))
                if len(image.shape) == 2:
                    n_frames = 1
                else:
                    n_frames = image.shape[-1]
                if n_frames == 1:
                    offset = np.add(offset, image)
                else:
                    offset = np.add(offset, np.sum(image, axis=-1))
            else:
                n_frames = 0
                finished = 0
                while finished == 0:
                    try:
                        frame = IO.read_tiff(os.path.join(directory, file), n_frames)
                        n_frames += 1
                        offset = np.add(offset, frame)
                    except:
                        finished = 1
            framesCounter = framesCounter + n_frames
            elapsed = time.time() - start
            if elapsed > 60:
                elapsed_display = elapsed / 60
                timestring = "min"
            elif elapsed > np.square(60):
                elapsed_display = elapsed / np.square(60)
                timestring = "hours"
            else:
                elapsed_display = elapsed
                timestring = "s"

            if directory.split("/")[-1] == intensity_string:
                print(
                    "Analysed offset of "
                    + intensity_string
                    + " image {}/{}    Time elapsed: {:.3f} {}".format(
                        i + 1, len(filelist), elapsed_display, timestring
                    ),
                    end="\r",
                    flush=True,
                )
            else:
                print(
                    "Analysed offset of "
                    + directory.split("/")[-1]
                    + " "
                    + intensity_string
                    + " image {}/{}    Time elapsed: {:.3f} {}".format(
                        i + 1, len(filelist), elapsed_display, timestring
                    ),
                    end="\r",
                    flush=True,
                )
        offset = offset / framesCounter
        return offset

    def calculate_variance(self, offset, directory, intensity_string, imtype=".tif"):
        """
        Calibrates variance. Given a directory, looks for a particular intensity
        string and loads these images. Then gets an offset.

        Args:
            offset (np.2darray): 2d matrix of offset
            directory (string): Folder containing tifs
            intensity_string (string): Intensity string
            imtype (string): image type to read in

        Returns:
            variance (np.2darray): variance matrix
        """
        filelist = self.filesearch(directory, imtype, intensity_string)

        framesCounter = 0
        offset_sq = np.square(offset)
        variance = np.zeros_like(offset_sq)

        start = time.time()
        for i, file in enumerate(filelist):
            if self.high_memory == True:
                image = IO.read_tiff(os.path.join(directory, file))
                if len(image.shape) == 2:
                    n_frames = 1
                else:
                    n_frames = image.shape[-1]
                if n_frames == 1:
                    variance = np.add(
                        variance, np.subtract(np.square(image), offset_sq)
                    )
                else:
                    variance = np.add(
                        variance,
                        np.sum(
                            np.subtract(np.square(image), offset_sq[np.newaxis, :, :]),
                            axis=-1,
                        ),
                    )
            else:
                n_frames = 0
                finished = 0
                while finished == 0:
                    try:
                        frame = IO.read_tiff(os.path.join(directory, file), n_frames)
                        n_frames += 1
                        variance = np.add(
                            variance, np.subtract(np.square(frame), offset_sq)
                        )
                    except:
                        finished = 1
            framesCounter = framesCounter + n_frames

            elapsed = time.time() - start
            if elapsed > 60:
                elapsed_display = elapsed / 60
                timestring = "min"
            elif elapsed > np.square(60):
                elapsed_display = elapsed / np.square(60)
                timestring = "hours"
            else:
                elapsed_display = elapsed
                timestring = "s"
            print(
                "Analysed variance of "
                + directory.split("/")[-1]
                + " "
                + intensity_string
                + " image {}/{}    Time elapsed: {:.3f} {}".format(
                    i + 1, len(filelist), elapsed_display, timestring
                ),
                end="\r",
                flush=True,
            )
        variance = variance / framesCounter
        return variance
