#!/usr/bin/env python3
"""
Single Folder Analysis Script for pyBayerSMLM

Processes one folder and exits - called by batch_analysis.sh for complete isolation.
Each invocation gets a fresh Python interpreter to prevent memory leaks.

Usage:
    python3 single_folder_analysis.py <type> <scratch_folder_path> <original_folder_path> <wavelength> <pfa> <sigma> <fraction_true> <use_variance_aware_demosaic>

    type: 'sm' or 'imaging'
    scratch_folder_path: full path to scratch folder (for processing)
    original_folder_path: full path to original folder (for .h5 output)
    wavelength: peak wavelength (e.g., 0.55, 0.638, 0.647)
    pfa: false alarm probability (e.g., 1e-4)
    sigma: sigma parameter (e.g., 1.5)
    fraction_true: fraction true parameter (e.g., 0.2)
    use_variance_aware_demosaic: use variance-aware demosaicing (true/false)

Created for pyBayerSMLM super-resolution microscopy analysis pipeline.
"""

import sys
import os
import gc
import glob
import traceback


def main():
    if len(sys.argv) != 9:
        print(
            "Usage: python3 single_folder_analysis.py <type> <scratch_folder_path> <original_folder_path> <wavelength> <pfa> <sigma> <fraction_true> <use_variance_aware_demosaic>"
        )
        print("  type: 'sm' or 'imaging'")
        print("  scratch_folder_path: full path to scratch folder (for processing)")
        print("  original_folder_path: full path to original folder (for .h5 output)")
        print("  wavelength: peak wavelength (e.g., 0.55, 0.638, 0.647)")
        print("  pfa: false alarm probability (e.g., 1e-4)")
        print("  sigma: sigma parameter (e.g., 1.5)")
        print("  fraction_true: fraction true parameter (e.g., 0.2)")
        print("  use_variance_aware_demosaic: use variance-aware demosaicing (true/false)")
        sys.exit(1)

    folder_type = sys.argv[1]
    scratch_folder_path = sys.argv[2]
    original_folder_path = sys.argv[3]
    peak_wavelength = float(sys.argv[4])
    pfa = float(sys.argv[5])
    sigma = float(sys.argv[6])
    fraction_true = float(sys.argv[7])
    use_variance_aware_demosaic = sys.argv[8].lower() in ('true', '1', 'yes', 'on')

    print(
        f"Using threshold parameters: pfa={pfa}, sigma={sigma}, fraction_true={fraction_true}"
    )
    print(f"Variance-aware demosaicing: {use_variance_aware_demosaic}")

    print(f"=== pyBayerSMLM Single Folder Analysis ===")
    print(f"Processing: {scratch_folder_path}")
    print(f"Original folder: {original_folder_path}")
    print(f"Type: {folder_type}, Wavelength: {peak_wavelength}")
    print(
        f"Threshold Parameters: pfa={pfa}, sigma={sigma}, fraction_true={fraction_true}"
    )
    print(f"Use variance-aware demosaicing: {use_variance_aware_demosaic}")
    print(f"Started: {os.popen('date').read().strip()}")
    print()

    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    src_dir = os.path.join(project_root, "src")

    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    try:
        # Basic validation
        if not os.path.exists(scratch_folder_path):
            print(f"ERROR: Scratch folder does not exist: {scratch_folder_path}")
            sys.exit(1)

        if not os.path.isdir(scratch_folder_path):
            print(f"ERROR: Scratch path is not a directory: {scratch_folder_path}")
            sys.exit(1)

        if not os.path.exists(original_folder_path):
            print(f"ERROR: Original folder does not exist: {original_folder_path}")
            sys.exit(1)

        if not os.path.isdir(original_folder_path):
            print(f"ERROR: Original path is not a directory: {original_folder_path}")
            sys.exit(1)

        # Check for required files in scratch folder
        tif_files = [f for f in os.listdir(scratch_folder_path) if f.endswith(".tif")]
        if not tif_files:
            print(f"SKIP: No .tif files found in {scratch_folder_path}")
            sys.exit(0)  # Not an error, just skip

        metadata_files = [
            f for f in os.listdir(scratch_folder_path) if "metadata" in f.lower()
        ]
        if not metadata_files:
            print(f"SKIP: No metadata files found in {scratch_folder_path}")
            sys.exit(0)  # Not an error, just skip

        print(
            f"Found {len(tif_files)} .tif files and {len(metadata_files)} metadata files"
        )

        # Clean the .h5 file that will be replaced
        h5_filename = "Localisations.h5"
        h5_file_to_remove = os.path.join(original_folder_path, h5_filename)
        if os.path.exists(h5_file_to_remove):
            print(f"Removing existing {h5_filename} from original folder...")
            try:
                os.remove(h5_file_to_remove)
                print(f"Removed: {h5_filename}")
            except Exception as e:
                print(f"Failed to remove {h5_filename}: {e}")

        # Import modules (do this late to avoid import overhead for skipped folders)
        print("Importing modules...")
        import IOFunctions

        # Optimize TIFF reading performance with aggressive memory mapping
        print("Configuring memory-efficient TIFF reading...")
        # Set environment variables for faster TIFF reading
        os.environ["TIFFFILE_NUM_THREADS"] = (
            "8"  # Use multiple threads for TIFF reading
        )
        os.environ["OMP_NUM_THREADS"] = "8"  # Optimize OpenMP for image processing

        # Force all IOFunctions to use memory mapping to avoid loading entire stacks into RAM
        original_read_tiff = None

        def patch_io_functions():
            import IOFunctions

            global original_read_tiff
            io_instance = IOFunctions.IO_Functions()
            original_read_tiff = io_instance.read_tiff

            def memory_mapped_read_tiff(
                file_path, frame=None, dtype="float32", memmap=True
            ):
                # Force memory mapping to be always True for large files
                return original_read_tiff(
                    file_path, frame=frame, dtype=dtype, memmap=True
                )

            # Patch the instance method
            io_instance.read_tiff = memory_mapped_read_tiff.__get__(
                io_instance, IOFunctions.IO_Functions
            )
            return io_instance

        print("Memory mapping optimization configured")
        import sCMOSFunctions
        import SpectralFunctions
        import MaskFunctions
        import SpotDetectionFunctions
        import SR_Functions
        import ImageAnalysisFunctions
        import HelperFunctions
        import types

        # Initialize functions
        print("Initializing functions...")
        functions = {
            "IO": IOFunctions.IO_Functions(),
            "sCMOS": sCMOSFunctions.sCMOS_Functions(),
            "S_F": SpectralFunctions.Spectral_Funcs(),
            "M_F": MaskFunctions.Mask_Functions(),
            "SD_F": SpotDetectionFunctions.SpotDetection_Functions(),
            "SupRes_F": SR_Functions.SuperRes_Functions(),
            "I_AF": ImageAnalysisFunctions.Image_Analysis_Functions(),
            "H_F": HelperFunctions.Helper_Functions(),
        }

        # Load camera parameters
        print("Loading camera parameters...")
        data_folder = os.path.join(project_root, "Camera_Calibrations", "Ximea_Camera")
        camera_data = {
            "gain": functions["IO"].read_tiff(os.path.join(data_folder, "gain.tif")),
            "offset": functions["IO"].read_tiff(
                os.path.join(data_folder, "offset.tif")
            ),
            "variance": functions["IO"].read_tiff(
                os.path.join(data_folder, "variance.tif")
            ),
            "readnoise": functions["IO"].read_tiff(
                os.path.join(data_folder, "readnoise.tif")
            ),
            "rqe": functions["IO"].read_tiff(os.path.join(data_folder, "rqe.tif")),
        }

        # Setup smoothing function
        smoothing_function = types.SimpleNamespace()
        smoothing_function.args = {"sigma": 1.5}
        smoothing_function.extent = 1.5
        smoothing_function.smoothing_function = functions["sCMOS"].gaussian_filter_stack
        smoothing_function.data_arg = "image"

        print("Setup complete, starting analysis...")

        # Process data using SR_Functions
        SupRes_F = functions["SupRes_F"]

        if folder_type == "sm":
            # SM data processing - process files from scratch, .h5 files will be created there
            # Note: fit_SM_data() does NOT support temporal_median_mode/ever_window parameters
            print(f"Processing as SM data with wavelength {peak_wavelength}")
            SupRes_F.fit_SM_data(
                scratch_folder_path,  # Process files from scratch
                smoothing_function,
                camera_data["gain"],
                camera_data["offset"],
                camera_data["rqe"],
                camera_data["readnoise"],
                variance=camera_data["variance"],
                pfa=pfa,
                ROI_size=16,
                peak_wavelength=peak_wavelength,
                NA=1.49,
                pixel_size=0.069,
                sigma=sigma,
                fraction_true=fraction_true,
                image_type=".tif",
                use_variance_aware_demosaic=use_variance_aware_demosaic,
            )
            print("SM data processing completed")
        else:
            # Imaging data processing - process files from scratch, .h5 files will be created there
            print(f"Processing as imaging data with wavelength {peak_wavelength}")
            SupRes_F.fit_imaging_data(
                scratch_folder_path,  # Process files from scratch
                smoothing_function,
                camera_data["gain"],
                camera_data["offset"],
                camera_data["rqe"],
                camera_data["readnoise"],
                variance=camera_data["variance"],
                pfa=pfa,
                ROI_size=16,
                peak_wavelength=peak_wavelength,
                NA=1.49,
                pixel_size=0.069,
                sigma=sigma,
                fraction_true=fraction_true,
                image_type=".tif",
                use_variance_aware_demosaic=use_variance_aware_demosaic,
            )
            print("Imaging data processing completed")

        # Move .h5 files from scratch folder to original folder
        scratch_h5_files = glob.glob(os.path.join(scratch_folder_path, "*.h5"))
        if scratch_h5_files:
            print(
                f"Moving {len(scratch_h5_files)} .h5 files from scratch to original folder..."
            )
            for h5_file in scratch_h5_files:
                filename = os.path.basename(h5_file)
                dest_file = os.path.join(original_folder_path, filename)
                try:
                    # Use move instead of copy to avoid leaving files in scratch
                    import shutil

                    shutil.move(h5_file, dest_file)
                    print(f"Moved: {filename}")
                except Exception as e:
                    print(f"Failed to move {filename}: {e}")

        # Check results in original folder
        final_h5_files = glob.glob(os.path.join(original_folder_path, "*.h5"))
        if final_h5_files:
            print(f"SUCCESS: Created {len(final_h5_files)} .h5 files")
            for h5_file in final_h5_files:
                print(f"Created: {os.path.basename(h5_file)}")
            sys.exit(0)
        else:
            print("WARNING: No .h5 files were created")
            sys.exit(0)  # Not necessarily an error

    except Exception as e:
        print(f"ERROR: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        print(f"Full traceback:")
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Force cleanup
        try:
            # Close any matplotlib figures that might be open
            import matplotlib.pyplot as plt

            plt.close("all")
        except:
            pass

        # Force garbage collection
        gc.collect()
        print("Cleanup completed")


if __name__ == "__main__":
    # Set environment to reduce noise
    os.environ["NUMEXPR_MAX_THREADS"] = "24"

    main()
