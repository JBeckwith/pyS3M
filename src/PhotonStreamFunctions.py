# -*- coding: utf-8 -*-
"""
This class contains functions pertaining to photon generation
jsb92
"""
import numpy as np
import sys, os

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)


class Photon_Stream_Functions:
    """Photon generation and streaming simulation functions.

    Provides functionality for simulating photon streams, PAINT experiments,
    and time-resolved single-molecule behavior for SMLM applications.
    """

    def __init__(self):
        """Initialize Photon_Stream_Functions class."""
        pass

    def PAINT_array_generator(
        self,
        save_name,
        camera_calibration,
        wavelength,
        absolute_QYs,
        dyes,
        dye_names,
        kon=37e6,
        conc=500e-12,
        bright_time=300e-3,
        frames=50000,
        frametime=100e-3,
        linker_size=20,
        photonrate=16000,
        photonratestd=8000,
        photonbudget=1500000,
        pixel_size=69,
        grid_n=3,
        NA=1.49,
        background_photons=0,
        gen_trace_function=True,
    ):
        """
        Multicolour-PAINT experiment generator
        Given an array of protein spots which are labelled in different
        colours, simulate grids of these DNAs

        Args:
            save_name_tiff (str): name to save files
            camera_calibration (dict): dictionary of gain, offset, variance and rqe. Image size will be derived from this.
            wavelength (np.1darray): wavelengths in same units as object_locs
            absolute_QYs (np.2darray): pixel_type by wavelength array, ordered in same way as mosaic unit
            dyes (np.2darray): 2d array of N dyes * emission; should be normalised such that sum is 1
            dye_names (dict): dictionary of dye names
            kon (np.float): on rate of DNA docking-imager interaction
            conc (np.float): concentration of imager strand
            bright_time (np.float): mean bright time in s
            framers_perround (int): number of frames per exchange-PAINT round
            frametime (np.float): time per frame in s
            DNA_array (np.1darray): array of where n_DNAs are labelled
                in PAINT experiment.
            linker_size (np.float): how big a linker unit is in same units as pixels
            photonrate (int): rate of photon flux per single molecule
            photonratestd (int): std of photon flux
            photonbudget (int): amount of photons a single molecule can emit pre-bleaching
            pixel_size (np.float): pixel size of camera used
            grid_n (int): x and y dimensions of grid (i.e. will simulate n by n grid of oligomers)
            wavelength (np.float): imaging wavelength in micron
            NA (np.float): numerical aperture of scope
            background_photons (np.float): background photons per pixel
            gen_trace_function (boolean): if True, uses "physical model" of on/off rates
                                        if False, just ensures spot on in every frame

        Returns:
            spot_locations (pd.DataFrame): spot locations in x and y (same unit as pixel size)
            input_parameters (xr.DataArray): input data parameters for the simulation

        """
        try:
            if len(wavelength) != absolute_QYs.shape[1]:
                raise Exception("absolute_QYs not defined at all wavelengths.")
            if len(wavelength) != dyes.shape[1]:
                raise Exception("dyes not defined at all wavelengths.")
            if dyes.shape[0] != int(np.square(grid_n)):
                raise Exception("each individual grid position does not have a dye.")
            test_len = np.zeros(dyes.shape[0])
            correct_vals = np.ones(len(test_len))
            for i in np.arange(dyes.shape[0]):
                test_len[i] = np.sum(dyes[i, :])
            if not np.all(np.isclose(test_len, correct_vals)):
                raise Exception("dye emission spectra are not normalised.")
        except Exception as error:
            print("Caught this error: " + repr(error))
            return
        import sys
        import os

        module_dir = os.path.dirname(__file__)
        sys.path.append(module_dir)
        import Multicolour_Simulation_Functions

        MSF = Multicolour_Simulation_Functions.MultiC_Sim_Funcs()

        import IOFunctions

        IO = IOFunctions.IO_Functions()

        import pandas as pd

        to_save = {
            "dye_names": list(dye_names),
            "kon/s": kon,
            "conc/M": conc,
            "bright_time/s": bright_time,
            "frames": frames,
            "frametime/s": frametime,
            "linker_size/nm": float(linker_size),
            "photonrate": photonrate,
            "pixel_size/nm": pixel_size,
            "grid_number": grid_n,
            "NA": NA,
            "background_photons": background_photons,
        }

        IO.write_json(to_save, save_name + "_inputparameters.json")

        h = camera_calibration["gain"].shape[0]
        w = camera_calibration["gain"].shape[1]

        taud = 1 / (kon * conc)  # in seconds

        x_centres = np.add(
            (
                np.arange(0, linker_size * grid_n, linker_size)
                - np.mean(np.arange(0, linker_size * grid_n, linker_size))
            ),
            np.divide(np.multiply(pixel_size, w), 2),
        )
        y_centres = np.add(
            (
                np.arange(0, linker_size * grid_n, linker_size)
                - np.mean(np.arange(0, linker_size * grid_n, linker_size))
            ),
            np.divide(np.multiply(pixel_size, h), 2),
        )
        x0, y0 = np.meshgrid(x_centres, y_centres)
        x0 = x0.ravel()
        y0 = y0.ravel()
        x0y0 = {}
        n_photons = {}

        for i, dye in enumerate(dye_names):
            x0y0[dye] = np.array([[x0[i]], [y0[i]]])
            if gen_trace_function == True:
                n_photons[dye] = np.asarray(
                    np.squeeze(
                        self.PAINT_gen(
                            taud,
                            bright_time,
                            frames,
                            frametime,
                            photonrate,
                            photonbudget,
                        )
                    ),
                    dtype=int,
                )
        if gen_trace_function == False:
            on_trace = np.random.choice(np.arange(len(list(dye_names))), size=frames)
            for i, dye in enumerate(dye_names):
                n_photons[dye] = np.zeros(frames)
                n_photons[dye][on_trace == i] = np.random.poisson(
                    lam=photonrate, size=frames
                )[on_trace == i]

        spot_locations = pd.DataFrame(
            np.vstack([x0.ravel(), y0.ravel()]).T, columns=["x_nm", "y_nm"]
        )

        image_stack = MSF.gen_camera_image_stack(
            camera_calibration,
            wavelength,
            absolute_QYs,
            dyes,
            n_photons,
            x0y0,
            background_photons=background_photons,
            NA=NA,
            pixel_size=pixel_size,
        )

        IO.write_tiff(
            image_stack, save_name + "_imagestack.tif", bit=np.uint16
        )  # write images

        spot_locations.to_csv(save_name + "_spotlocations.csv", index=False)
        return image_stack, spot_locations

    def PAINT_gen_ontrace(self, meandark, meanbright, frames, time, n_traces):
        """
        PAINT experiment generator:
        Generates on and off-traces for given parameters.
        Calculates the number of Photons in each frame for a binding site.
        Function cribbed from picasso (https://github.com/jungmannlab/picasso)

        Args:
            meandark (np.float): mean dark time (in same units as time, typically s)
            meanbright (np.float): mean bright time (in same units as time, typically s)
            frames (int): number of frames to simulate over
            time (np.float): amount of time per frame (typically in units of s)
            ntracks (int): number of PAINT experiments to simulate

        Returns:
            on_trace (np.1darray): list of frames and fractional on-time
        """
        meanlocs = 4 * int(
            np.ceil(frames * time / (meandark + meanbright))
        )  # This is an estimate for the total number of binding events
        if meanlocs < 10:
            meanlocs = meanlocs * 10

        dark_times = np.random.exponential(meandark, size=(meanlocs, n_traces))
        bright_times = np.random.exponential(meanbright, size=(meanlocs, n_traces))

        t = np.linspace(time, time * frames, frames)
        on_trace = np.zeros([n_traces, len(t)])

        p = 0.5
        coin_toss = np.random.binomial(n=1, p=p, size=n_traces)
        for n in np.arange(n_traces):
            if coin_toss[n] == 0:
                events = np.vstack((dark_times[:, n], bright_times[:, n])).reshape(
                    (-1,), order="F"
                )  # Interweave dark_times and bright_times [dt,bt,dt,bt..]
                eventsum = np.cumsum(events)
                maxloc = np.argmax(
                    eventsum > (frames * time)
                )  # Find the first event that exceeds the total integration time
                events = events[: maxloc + 1]
                for i in np.arange(0, len(events)):
                    if np.mod(i, 2) == 1:
                        if i > 0:
                            on_steps = (eventsum[i - 1] <= t) & (t <= eventsum[i])
                        else:
                            on_steps = eventsum[i] >= t - time

                        on_trace[n, on_steps] = 1
                        if len(np.where(on_steps)[0]) > 0:
                            on_trace[n, np.where(on_steps)[0][-1]] = (
                                np.abs(t[on_steps][-1] - eventsum[i]) / time
                            )
            else:
                events = np.vstack((bright_times[:, n], dark_times[:, n])).reshape(
                    (-1,), order="F"
                )  # Interweave bright_times and dark_times [bt,dt,bt,dt..]
                eventsum = np.cumsum(events)
                maxloc = np.argmax(
                    eventsum > (frames * time)
                )  # Find the first event that exceeds the total integration time
                events = events[: maxloc + 1]
                eventsum = eventsum[: maxloc + 1]
                for i in np.arange(0, len(events)):
                    if np.mod(i, 2) == 0:
                        if i > 0:
                            on_steps = (eventsum[i - 1] <= t) & (t <= eventsum[i])
                        else:
                            on_steps = eventsum[i] >= t - time

                        on_trace[n, on_steps] = 1
                        if len(np.where(on_steps)[0]) > 0:
                            on_trace[n, np.where(on_steps)[0][-1]] = (
                                np.abs(t[on_steps][-1] - eventsum[i]) / time
                            )
            if on_trace[n, -1] > 1:
                on_trace[n, -1] = 1
        return on_trace

    # keep working on
    def PAINT_gen(
        self, meandark, meanbright, frames, time, photonrate, photonbudget, n_traces=1
    ):
        """
        PAINT experiment generator:
        Generates on and off-traces for given parameters.
        Calculates the number of Photons in each frame for a binding site.
        Function cribbed from picasso (https://github.com/jungmannlab/picasso)

        Args:
            meandark (np.float): mean dark time (in same units as time, typically s)
            meanbright (np.float): mean bright time (in same units as time, typically s)
            frames (int): number of frames to simulate over
            time (np.float): amount of time per frame (typically in units of s)
            photonrate (int): rate of photon flux per single molecule
            photonbudget (int): amount of photons a single molecule can emit pre-bleaching

        Returns:
            photonsinframe (np.1darray): list of frames vs number of photons
            timetrace (np.1darray): array of on/off times
            spotkinetics (list): list of
                [on_events sum(photonsinframe > 0) meandark meanbright]
        """
        on_trace = self.PAINT_gen_ontrace(meandark, meanbright, frames, time, n_traces)

        photonsinframe = np.random.poisson(
            photonrate * on_trace
        )  # get initial photon counts

        return photonsinframe

    def photontiminggenerator_dt(self, Ij, nphotons):
        """
        generates differences in photon arrival times for set number of photons.
        see equations 1 and 2 of Watkins, L. P.; Yang, H.
        J. Phys. Chem. B 2005, 109 (1), 617–628.
        https://doi.org/10.1021/jp0467548.

        Args:
            Ij (float): photon counts per second.
            nphotons (int): number of photons to simulate.

        Returns:
            photontimediff (np.1darray): array of photon arrival time differences in s
        """
        photontimediff = np.random.exponential(1 / Ij, nphotons)
        return photontimediff

    def photongenerator_dt_nt(self, Ij, deltaT):
        """
        generates differences in photon arrival times for set time interval.
        see equations 1 and 2 of Watkins, L. P.; Yang, H.
        J. Phys. Chem. B 2005, 109 (1), 617–628.
        https://doi.org/10.1021/jp0467548.

        Args:
            Ij (float): photon counts per second.
            deltaT (float): time interval in seconds.

        Returns:
            photontimediff (np.1darray): array of photon arrival time differences in s
        """
        nphotons = int(Ij * deltaT)
        # get MLE of number of photons we'd expect
        photontimediff = np.random.exponential(1 / Ij, nphotons * 2)
        # generate enough photon times so we know we've hit our time
        photontimediff = photontimediff[np.cumsum(photontimediff) < deltaT]
        # prune our number of photons such that we only have as many as we expect
        return photontimediff

    def photongenerator_CP(self, Ij=10.0e3, deltaT=1.0):
        """
        generates photon time stamps for a range of intensity levels for
        a range of time steps.
        see equations 1 and 2 of Watkins, L. P.; Yang, H.
        J. Phys. Chem. B 2005, 109 (1), 617–628.
        https://doi.org/10.1021/jp0467548.

        Args:
            Ij (np.1darray): array of photon counts per second.
            deltaT (np.1darray): array of time intervals in second.
                                must have same length as Ij

        Returns:
            photontimes (np.1darray): array of photon arrival times in s
        """
        try:
            test0 = isinstance(Ij, float)
            if test0 == True:
                if ~isinstance(deltaT, float):
                    raise Exception("Ij is a float, but deltaT is not.")
                else:
                    if len(Ij) != len(deltaT):
                        raise Exception("Ij and deltaT must be the same lengths.")
        except Exception as error:
            print("Caught this error: " + repr(error))
            return
        if ~isinstance(Ij, float):
            for i, Ival in enumerate(Ij):
                if i == 0:
                    photontimes = np.cumsum(self.photongenerator_dt_nt(Ival, deltaT[i]))
                else:
                    photontimes = np.hstack(
                        [
                            photontimes,
                            np.sum(deltaT[:i])
                            + np.cumsum(self.photongenerator_dt_nt(Ival, deltaT[i])),
                        ]
                    )
        else:
            nphotons = Ij * deltaT
            # get MLE of number of photons we'd expect
            photontimediff = np.random.exponential(1 / Ij, nphotons * 2)
            # generate enough photon times so we know we've hit our time
            photontimediff = photontimediff[np.cumsum(photontimediff) < deltaT]
            # prune our number of photons such that we only have as many as we expect
            photontimes = np.cumsum(photontimediff)
        return photontimes
