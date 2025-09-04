#!/usr/bin/env python3
"""
Single Folder Analysis Script for pyBayerSMLM

Processes one folder and exits - called by batch_analysis.sh for complete isolation.
Each invocation gets a fresh Python interpreter to prevent memory leaks.

Usage:
    python3 single_folder_analysis.py <type> <folder_path> <wavelength> <pfa> <sigma> <fraction_true>
    
    type: 'sm' or 'imaging'
    folder_path: full path to folder to process  
    wavelength: peak wavelength (e.g., 0.55, 0.638, 0.647)
    pfa: false alarm probability (e.g., 1e-4)
    sigma: sigma parameter (e.g., 1.5)
    fraction_true: fraction true parameter (e.g., 0.15)

Created for pyBayerSMLM super-resolution microscopy analysis pipeline.
"""

import sys
import os
import gc
import glob
import traceback


def main():
    # Check arguments - now expects 7 arguments (including script name)
    if len(sys.argv) != 7:
        print(
            "Usage: python3 single_folder_analysis.py <type> <folder_path> <wavelength> <pfa> <sigma> <fraction_true>"
        )
        print("  type: 'sm' or 'imaging'")
        print("  folder_path: full path to folder to process")
        print("  wavelength: peak wavelength (e.g., 0.55, 0.638, 0.647)")
        print("  pfa: false alarm probability (e.g., 1e-4)")
        print("  sigma: sigma parameter (e.g., 1.5)")
        print("  fraction_true: fraction true parameter (e.g., 0.15)")
        sys.exit(1)

    folder_type = sys.argv[1]
    folder_path = sys.argv[2]
    peak_wavelength = float(sys.argv[3])
    pfa = float(sys.argv[4])
    sigma = float(sys.argv[5])
    fraction_true = float(sys.argv[6])

    print(
        f"Using threshold parameters: pfa={pfa}, sigma={sigma}, fraction_true={fraction_true}"
    )

    print(f"=== pyBayerSMLM Single Folder Analysis ===")
    print(f"Processing: {folder_path}")
    print(f"Type: {folder_type}, Wavelength: {peak_wavelength}")
    print(
        f"Threshold Parameters: pfa={pfa}, sigma={sigma}, fraction_true={fraction_true}"
    )
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
        if not os.path.exists(folder_path):
            print(f"ERROR: Folder does not exist: {folder_path}")
            sys.exit(1)

        if not os.path.isdir(folder_path):
            print(f"ERROR: Path is not a directory: {folder_path}")
            sys.exit(1)

        # Check for required files
        tif_files = [f for f in os.listdir(folder_path) if f.endswith(".tif")]
        if not tif_files:
            print(f"SKIP: No .tif files found in {folder_path}")
            sys.exit(0)  # Not an error, just skip

        metadata_files = [f for f in os.listdir(folder_path) if "metadata" in f.lower()]
        if not metadata_files:
            print(f"SKIP: No metadata files found in {folder_path}")
            sys.exit(0)  # Not an error, just skip

        print(
            f"Found {len(tif_files)} .tif files and {len(metadata_files)} metadata files"
        )

        # Clean existing .h5 files
        h5_files = glob.glob(os.path.join(folder_path, "*.h5"))
        if h5_files:
            print(f"Removing {len(h5_files)} existing .h5 files...")
            for h5_file in h5_files:
                try:
                    os.remove(h5_file)
                    print(f"Removed: {os.path.basename(h5_file)}")
                except Exception as e:
                    print(f"Failed to remove {h5_file}: {e}")

        # Import modules (do this late to avoid import overhead for skipped folders)
        print("Importing modules...")
        import IOFunctions
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
            # SM data processing
            print(f"Processing as SM data with wavelength {peak_wavelength}")
            SupRes_F.fit_SM_data(
                folder_path,
                smoothing_function,
                camera_data["gain"],
                camera_data["offset"],
                camera_data["rqe"],
                camera_data["readnoise"],
                variance=camera_data["variance"],
                pfa=pfa,
                ROI_size=12,
                peak_wavelength=peak_wavelength,
                NA=1.49,
                pixel_size=0.069,
                sigma=sigma,
                fraction_true=fraction_true,
                image_type=".tif",
            )
            print("SM data processing completed")
        else:
            # Imaging data processing
            print(f"Processing as imaging data with wavelength {peak_wavelength}")
            SupRes_F.fit_imaging_data(
                folder_path,
                smoothing_function,
                camera_data["gain"],
                camera_data["offset"],
                camera_data["rqe"],
                camera_data["readnoise"],
                variance=camera_data["variance"],
                pfa=pfa,
                ROI_size=12,
                peak_wavelength=peak_wavelength,
                NA=1.49,
                pixel_size=0.069,
                sigma=sigma,
                fraction_true=fraction_true,
                image_type=".tif",
            )
            print("Imaging data processing completed")

        # Check results
        final_h5_files = glob.glob(os.path.join(folder_path, "*.h5"))
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
