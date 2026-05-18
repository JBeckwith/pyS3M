# Backup of functions removed from src/CalibrationFunctions.py
# Removed: 2026-04-10
# Reason: calculate_variance is superseded by calculate_offset_and_variance(),
#         which computes both offset and variance in a single pass over the files,
#         halving I/O. No callers of calculate_variance exist in src/ or notebooks/.
#         calculate_offset and _process_calibration_files are NOT removed here
#         because calculate_offset is still called in notebooks/calibration/sCMOS_Testing.ipynb.

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

        offset_sq = np.square(offset)
        variance = np.zeros_like(offset_sq)

        # Define processing functions for variance calculation
        def process_single(acc, frame):
            return np.add(acc, np.subtract(np.square(frame), offset_sq))

        def process_multi(acc, image):
            return np.add(
                acc,
                np.sum(
                    np.subtract(np.square(image), offset_sq[:, :, np.newaxis]),
                    axis=-1,
                ),
            )

        # Process all files
        variance, framesCounter = self._process_calibration_files(
            directory,
            intensity_string,
            filelist,
            variance,
            "variance",
            process_single,
            process_multi,
        )

        variance = variance / framesCounter
        return variance
