#!/usr/bin/env python3
"""
Single Folder Interactive Analysis
Pure Python script for threshold tuning + analysis of a single folder

Usage:
    python single_folder_interactive_analysis.py "/path/to/folder"

This script:
1. Loads test frames from the specified folder
2. Allows interactive tuning of threshold parameters
3. Runs the full analysis using optimized parameters
"""

import os
import sys
import argparse
import subprocess
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append("../../src")

try:
    from interactive_threshold_tuner import InteractiveThresholdTuner
    from IOFunctions import IO_Functions
    from HelperFunctions import Helper_Functions
except ImportError as e:
    print(f"Error importing pyS3M modules: {e}")
    print("Please ensure the virtual environment is activated:")
    print("source /home/jbeckwith/.virtualenvs/pyS3M/bin/activate")
    sys.exit(1)


class SingleFolderAnalyzer:
    """Single folder analysis with interactive threshold tuning"""

    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)
        self.folder_name = self.folder_path.name
        self.log_file = f"single_folder_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Initialize helper functions
        self.iof = IO_Functions()
        self.hf = Helper_Functions()

        # Analysis script path
        self.analysis_script = Path(__file__).parent / "single_folder_analysis.py"

        # Initialize log
        self._init_log()

    def _init_log(self):
        """Initialize log file"""
        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"pyS3M Single Folder Analysis - {datetime.now()}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Folder: {self.folder_path}\n")
            f.write(f"Script: {Path(__file__).name}\n")
            f.write(f"Log File: {self.log_file}\n")
            f.write("=" * 80 + "\n\n")

    def log_message(self, message):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")

    def console_message(self, message):
        """Print to console and log"""
        print(message)
        self.log_message(message)

    def validate_folder(self):
        """Validate folder exists and contains TIFF files"""
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")

        if not self.folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {self.folder_path}")

        # Check for TIFF files
        tiff_files = list(self.folder_path.glob("*.tif")) + list(self.folder_path.glob("*.tiff"))
        if not tiff_files:
            raise ValueError(f"No TIFF files found in: {self.folder_path}")

        self.console_message(f"✓ Folder validated: {len(tiff_files)} TIFF files found")
        return len(tiff_files)

    def determine_folder_type(self):
        """Determine folder type and wavelength from path"""
        folder_path_str = str(self.folder_path)

        # Default values
        folder_type = "imaging"
        wavelength = 0.6

        # Check folder path patterns
        if any(pattern in folder_path_str for pattern in ["HeLa", "HILO"]):
            folder_type = "imaging"
            wavelength = 0.647
            message = f"Detected HeLa/HILO data - using wavelength: {wavelength}μm"
        elif any(pattern in folder_path_str for pattern in ["SM", "STORM", "Dye", "biotinylated"]):
            folder_type = "sm"
            wavelength = 0.638
            message = f"Detected SM/dye data - using wavelength: {wavelength}μm"
        elif any(pattern in folder_path_str for pattern in ["Origami", "DNA", "iPSC", "PAINT"]):
            folder_type = "imaging"
            wavelength = 0.55
            message = f"Detected DNA Origami/iPSC/PAINT data - using wavelength: {wavelength}μm"
        else:
            message = f"Using default parameters - type: {folder_type}, wavelength: {wavelength}μm"

        self.console_message(message)
        return folder_type, wavelength

    def run_threshold_tuning(self):
        """Run interactive threshold tuning"""
        self.console_message("")
        self.console_message("=" * 80)
        self.console_message("STEP 1: INTERACTIVE THRESHOLD TUNING")
        self.console_message("=" * 80)

        # Determine folder type and wavelength
        folder_type, wavelength = self.determine_folder_type()

        self.log_message(f"Running threshold tuning: type={folder_type}, wavelength={wavelength}")

        # Create modified tuner for single folder
        tuner = SingleFolderTuner(str(self.folder_path), folder_type, wavelength)

        try:
            result = tuner.run_single_folder()

            if result:
                self.console_message("✓ Threshold tuning completed successfully")
                self.log_message("Threshold tuning completed successfully")
                return result
            else:
                self.console_message("✗ Threshold tuning failed or was cancelled")
                self.log_message("Threshold tuning failed or was cancelled")
                return None

        except Exception as e:
            self.console_message(f"✗ Error during threshold tuning: {e}")
            self.log_message(f"Error during threshold tuning: {e}")
            return None

    def run_analysis(self, parameters):
        """Run analysis with optimized parameters"""
        self.console_message("")
        self.console_message("=" * 80)
        self.console_message("STEP 2: RUNNING ANALYSIS WITH OPTIMIZED PARAMETERS")
        self.console_message("=" * 80)

        # Extract parameters
        pfa = parameters['pfa']
        sigma = parameters['sigma']
        fraction_true = parameters['fraction_true']
        wavelength = parameters['wavelength']
        use_variance_aware = parameters.get('use_variance_aware', True)
        folder_type = parameters['folder_type']

        self.console_message("Using optimized parameters:")
        self.console_message(f"  PFA: {pfa}")
        self.console_message(f"  Sigma: {sigma}")
        self.console_message(f"  Fraction True: {fraction_true}")
        self.console_message(f"  Wavelength: {wavelength}")
        self.console_message(f"  Variance-aware demosaicing: {use_variance_aware}")
        self.console_message(f"  Folder Type: {folder_type}")

        self.log_message(f"Starting analysis with optimized parameters: pfa={pfa}, sigma={sigma}, fraction_true={fraction_true}, wavelength={wavelength}, variance_aware={use_variance_aware}")

        # Check if analysis script exists
        if not self.analysis_script.exists():
            raise FileNotFoundError(f"Analysis script not found: {self.analysis_script}")

        # Prepare command
        cmd = [
            "python3",
            str(self.analysis_script),
            folder_type,
            str(self.folder_path),  # scratch folder (same as original for single folder)
            str(self.folder_path),  # original folder
            str(wavelength),
            str(pfa),
            str(sigma),
            str(fraction_true),
            str(use_variance_aware).lower()
        ]

        self.console_message("")
        self.console_message(f"Running analysis on: {self.folder_name}")
        self.console_message("This may take some time depending on the dataset size...")

        try:
            # Run the analysis with real-time output
            self.log_message("Starting analysis subprocess...")

            with open(self.log_file, 'a') as log_f:
                log_f.write("\n" + "=" * 40 + " ANALYSIS OUTPUT " + "=" * 40 + "\n")
                log_f.flush()

                # Use Popen for real-time output
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                # Read output in real-time
                for line in process.stdout:
                    print(line.rstrip())  # Print to console
                    log_f.write(line)     # Write to log
                    log_f.flush()

                # Wait for completion with extended timeout (4 hours)
                try:
                    return_code = process.wait(timeout=14400)  # 4 hour timeout
                except subprocess.TimeoutExpired:
                    process.kill()
                    self.console_message("✗ Analysis timed out (>4 hours)")
                    self.log_message("Analysis timed out after 4 hours")
                    return False

                log_f.write("\n" + "=" * 80 + "\n")

            if return_code == 0:
                self.console_message("✓ Analysis completed successfully")
                self.log_message("Analysis completed successfully")

                # Check for generated .h5 files
                h5_files = list(self.folder_path.glob("*.h5"))
                if h5_files:
                    self.console_message(f"✓ Generated {len(h5_files)} .h5 result file(s)")
                    for h5_file in h5_files:
                        self.console_message(f"  - {h5_file.name}")
                    self.log_message(f"Generated {len(h5_files)} .h5 result files")
                else:
                    self.console_message("⚠ No .h5 files generated (analysis may not have produced results)")
                    self.log_message("WARNING: No .h5 files generated after analysis")

                return True
            else:
                self.console_message(f"✗ Analysis failed with exit code {return_code}")
                self.console_message("Check the log file for detailed error information")
                self.log_message(f"Analysis failed with exit code {return_code}")
                return False
        except Exception as e:
            self.console_message(f"✗ Error running analysis: {e}")
            self.log_message(f"Error running analysis: {e}")
            return False

    def run(self):
        """Main execution flow"""
        self.console_message(f"Starting single folder analysis for: {self.folder_name}")

        try:
            # Validate folder
            self.validate_folder()

            # Step 1: Interactive threshold tuning
            parameters = self.run_threshold_tuning()

            if parameters:
                # Step 2: Analysis with optimized parameters
                if self.run_analysis(parameters):
                    self.console_message("")
                    self.console_message("🎉 Single folder analysis completed successfully!")
                    self.console_message(f"Results saved in: {self.folder_path}")
                    self.console_message(f"Log file: {self.log_file}")
                    return True
                else:
                    self.console_message("")
                    self.console_message("⚠️ Analysis failed. Check the log for details.")
                    self.console_message(f"Log file: {self.log_file}")
                    return False
            else:
                self.console_message("")
                self.console_message("⚠️ Threshold tuning failed or was cancelled.")
                self.console_message("Cannot proceed with analysis.")
                self.console_message(f"Log file: {self.log_file}")
                return False

        except Exception as e:
            self.console_message(f"✗ Error: {e}")
            self.log_message(f"Error: {e}")
            return False


class SingleFolderTuner(InteractiveThresholdTuner):
    """Modified tuner for single folder analysis"""

    def __init__(self, folder_path, folder_type, default_wavelength):
        super().__init__()
        self.single_folder_path = folder_path
        self.single_folder_type = folder_type
        self.single_default_wavelength = default_wavelength

    def run_single_folder(self):
        """Run threshold tuning for a single folder"""
        print("=" * 80)
        print("pyS3M Single Folder Threshold Tuner")
        print("=" * 80)
        print(f"Folder: {self.single_folder_path}")
        print(f"Type: {self.single_folder_type}")
        print(f"Default wavelength: {self.single_default_wavelength}")
        print("=" * 80)

        if not os.path.isdir(self.single_folder_path):
            print(f"ERROR: Folder not found: {self.single_folder_path}")
            return None

        # Run interactive parameter tuning
        result = self.interactive_parameter_tuning(
            self.single_folder_path,
            self.single_folder_type,
            self.single_default_wavelength
        )

        if result == "quit":
            print("Quitting by user request")
            return None
        elif result is not None:
            print(f"\n✓ Parameters optimized for {os.path.basename(self.single_folder_path)}")
            return result
        else:
            print(f"✗ Parameter tuning skipped for {os.path.basename(self.single_folder_path)}")
            return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Single folder threshold tuning and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python single_folder_interactive_analysis.py "/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250915 DNA_PAINT_Slides/PAINT_40_RY/20p_561_60_638_both_NF_785SP_1"

This script will:
1. Load test frames and allow interactive parameter tuning with live preview
2. Run full analysis using the optimized parameters
3. Generate .h5 result files in the original folder
"""
    )

    parser.add_argument(
        "folder_path",
        help="Path to folder to analyze"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Create and run analyzer
    analyzer = SingleFolderAnalyzer(args.folder_path)
    success = analyzer.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()