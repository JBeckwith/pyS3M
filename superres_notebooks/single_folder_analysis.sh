#!/bin/bash

# Single Folder Analysis Script - Interactive threshold tuning + analysis
# Usage: ./single_folder_analysis.sh "path/to/folder"
#
# This script:
# 1. Loads test frames from the specified folder
# 2. Allows interactive tuning of threshold parameters
# 3. Runs the full analysis using optimized parameters
#
# Created for pyBayerSMLM super-resolution analysis pipeline

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THRESHOLD_TUNER="$SCRIPT_DIR/interactive_threshold_tuner.py"
ANALYSIS_SCRIPT="$SCRIPT_DIR/single_folder_analysis.py"
LOG_FILE="single_folder_analysis_$(date +%Y%m%d_%H%M%S).log"

# Check arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <folder_path>"
    echo ""
    echo "Example: $0 '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250715_HollidayJunctions/60pM_HollidayJunction_50mMMgCl2/40perc561_NF_SP785_30ms_1'"
    echo ""
    echo "This script will:"
    echo "  1. Load test frames and allow interactive parameter tuning"
    echo "  2. Run full analysis using the optimized parameters"
    exit 1
fi

FOLDER_PATH="$1"
FOLDER_NAME=$(basename "$FOLDER_PATH")

# Initialize log file
{
    echo "============================================================"
    echo "pyBayerSMLM Single Folder Analysis - $(date)"
    echo "============================================================"
    echo "Folder: $FOLDER_PATH"
    echo "Script: $(basename "$0")"
    echo "Location: $SCRIPT_DIR"
    echo "Log File: $LOG_FILE"
    echo "============================================================"
    echo
} | tee "$LOG_FILE"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

console_message() {
    echo "$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Check if virtual environment is activated
check_virtualenv() {
    if [[ "$VIRTUAL_ENV" != *"pyBayerSMLM"* ]]; then
        console_message "WARNING: pyBayerSMLM virtual environment may not be activated"
        console_message "Please activate it with:"
        console_message "  source /home/jbeckwith/.virtualenvs/pyBayerSMLM/bin/activate"
        echo
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            console_message "Aborted by user"
            exit 1
        fi
    else
        log_message "pyBayerSMLM virtual environment detected: $VIRTUAL_ENV"
    fi
}

# Check if folder exists and has TIFF files
validate_folder() {
    if [ ! -d "$FOLDER_PATH" ]; then
        console_message "ERROR: Folder not found: $FOLDER_PATH"
        exit 1
    fi

    # Check for TIFF files
    local tiff_count=$(find "$FOLDER_PATH" -name "*.tif" -o -name "*.tiff" -type f | wc -l)
    if [ $tiff_count -eq 0 ]; then
        console_message "ERROR: No TIFF files found in: $FOLDER_PATH"
        exit 1
    fi

    console_message "✓ Folder validated: $tiff_count TIFF files found"
    log_message "Folder validation passed: $tiff_count TIFF files in $FOLDER_PATH"
}

# Check if required scripts exist
validate_scripts() {
    if [ ! -f "$THRESHOLD_TUNER" ]; then
        console_message "ERROR: Threshold tuner script not found: $THRESHOLD_TUNER"
        exit 1
    fi

    if [ ! -f "$ANALYSIS_SCRIPT" ]; then
        console_message "ERROR: Analysis script not found: $ANALYSIS_SCRIPT"
        exit 1
    fi

    console_message "✓ Required scripts found"
    log_message "Script validation passed"
}

# Determine folder type and default parameters based on path
determine_folder_type() {
    local folder_path="$1"
    local folder_type="imaging"
    local wavelength="0.6"

    # Check folder path patterns to determine type and wavelength
    if [[ "$folder_path" == *"HeLa"* ]] || [[ "$folder_path" == *"HILO"* ]]; then
        folder_type="imaging"
        wavelength="0.647"
        console_message "Detected HeLa/HILO data - using wavelength: ${wavelength}μm"
    elif [[ "$folder_path" == *"SM"* ]] || [[ "$folder_path" == *"STORM"* ]] || [[ "$folder_path" == *"Dye"* ]] || [[ "$folder_path" == *"biotinylated"* ]]; then
        folder_type="sm"
        wavelength="0.638"
        console_message "Detected SM/dye data - using wavelength: ${wavelength}μm"
    elif [[ "$folder_path" == *"Origami"* ]] || [[ "$folder_path" == *"DNA"* ]] || [[ "$folder_path" == *"iPSC"* ]]; then
        folder_type="imaging"
        wavelength="0.55"
        console_message "Detected DNA Origami/iPSC data - using wavelength: ${wavelength}μm"
    else
        console_message "Using default parameters - type: $folder_type, wavelength: ${wavelength}μm"
    fi

    echo "$folder_type $wavelength"
}

# Run interactive threshold tuning for single folder
run_threshold_tuning() {
    console_message ""
    console_message "============================================================"
    console_message "STEP 1: INTERACTIVE THRESHOLD TUNING"
    console_message "============================================================"

    log_message "Starting interactive threshold tuning for: $FOLDER_PATH"

    # Create a temporary threshold tuner script for single folder
    local temp_tuner="$SCRIPT_DIR/temp_single_folder_tuner.py"

    # Generate a modified version of the threshold tuner for single folder
    cat > "$temp_tuner" << 'EOF'
#!/usr/bin/env python3
"""
Single Folder Interactive Threshold Tuner
Modified version of interactive_threshold_tuner.py for single folder analysis
"""

import os
import sys
import argparse

# Add src to path for imports
sys.path.append("../src")

# Import the main tuner class
from interactive_threshold_tuner import InteractiveThresholdTuner

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
        print("pyBayerSMLM Single Folder Threshold Tuner")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single folder threshold tuning")
    parser.add_argument("folder_path", help="Path to folder to analyze")
    parser.add_argument("folder_type", help="Folder type (sm, imaging)")
    parser.add_argument("wavelength", type=float, help="Default wavelength")

    args = parser.parse_args()

    tuner = SingleFolderTuner(args.folder_path, args.folder_type, args.wavelength)
    result = tuner.run_single_folder()

    if result:
        # Save parameters to temporary file for analysis script
        import json
        temp_params_file = "/tmp/single_folder_params.json"
        with open(temp_params_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Parameters saved to: {temp_params_file}")
        sys.exit(0)
    else:
        print("No parameters to save")
        sys.exit(1)
EOF

    # Make the temporary script executable
    chmod +x "$temp_tuner"

    # Get folder type and wavelength
    local folder_info=($(determine_folder_type "$FOLDER_PATH"))
    local folder_type="${folder_info[0]}"
    local wavelength="${folder_info[1]}"

    log_message "Running threshold tuning: type=$folder_type, wavelength=$wavelength"

    # Run the threshold tuning
    if python3 "$temp_tuner" "$FOLDER_PATH" "$folder_type" "$wavelength" 2>&1 | tee -a "$LOG_FILE"; then
        console_message "✓ Threshold tuning completed successfully"
        log_message "Threshold tuning completed successfully"

        # Check if parameters were saved
        if [ -f "/tmp/single_folder_params.json" ]; then
            console_message "✓ Optimized parameters saved"
            log_message "Optimized parameters saved to /tmp/single_folder_params.json"
            rm -f "$temp_tuner"  # Cleanup
            return 0
        else
            console_message "✗ No parameters were saved"
            log_message "No parameters were saved from threshold tuning"
            rm -f "$temp_tuner"  # Cleanup
            return 1
        fi
    else
        console_message "✗ Threshold tuning failed"
        log_message "Threshold tuning failed"
        rm -f "$temp_tuner"  # Cleanup
        return 1
    fi
}

# Run analysis with optimized parameters
run_analysis() {
    console_message ""
    console_message "============================================================"
    console_message "STEP 2: RUNNING ANALYSIS WITH OPTIMIZED PARAMETERS"
    console_message "============================================================"

    # Load optimized parameters
    local params_file="/tmp/single_folder_params.json"
    if [ ! -f "$params_file" ]; then
        console_message "ERROR: Optimized parameters file not found: $params_file"
        return 1
    fi

    # Extract parameters from JSON file
    local pfa=$(python3 -c "import json; data=json.load(open('$params_file')); print(data['pfa'])")
    local sigma=$(python3 -c "import json; data=json.load(open('$params_file')); print(data['sigma'])")
    local fraction_true=$(python3 -c "import json; data=json.load(open('$params_file')); print(data['fraction_true'])")
    local wavelength=$(python3 -c "import json; data=json.load(open('$params_file')); print(data['wavelength'])")
    local use_variance_aware=$(python3 -c "import json; data=json.load(open('$params_file')); print(str(data.get('use_variance_aware', True)).lower())")
    local folder_type=$(python3 -c "import json; data=json.load(open('$params_file')); print(data['folder_type'])")

    console_message "Using optimized parameters:"
    console_message "  PFA: $pfa"
    console_message "  Sigma: $sigma"
    console_message "  Fraction True: $fraction_true"
    console_message "  Wavelength: $wavelength"
    console_message "  Variance-aware demosaicing: $use_variance_aware"
    console_message "  Folder Type: $folder_type"

    log_message "Starting analysis with optimized parameters: pfa=$pfa, sigma=$sigma, fraction_true=$fraction_true, wavelength=$wavelength, variance_aware=$use_variance_aware"

    # Run the analysis
    console_message ""
    console_message "Running analysis on: $FOLDER_NAME"
    console_message "This may take some time depending on the dataset size..."

    if python3 "$ANALYSIS_SCRIPT" "$folder_type" "$FOLDER_PATH" "$FOLDER_PATH" "$wavelength" "$pfa" "$sigma" "$fraction_true" "$use_variance_aware" 2>&1 | tee -a "$LOG_FILE"; then
        console_message "✓ Analysis completed successfully"
        log_message "Analysis completed successfully"

        # Check for generated .h5 files
        local h5_count=$(find "$FOLDER_PATH" -name "*.h5" -type f | wc -l)
        if [ $h5_count -gt 0 ]; then
            console_message "✓ Generated $h5_count .h5 result file(s)"
            log_message "Generated $h5_count .h5 result files"
        else
            console_message "⚠ No .h5 files generated (analysis may not have produced results)"
            log_message "WARNING: No .h5 files generated after analysis"
        fi

        return 0
    else
        console_message "✗ Analysis failed"
        log_message "Analysis failed"
        return 1
    fi
}

# Cleanup function
cleanup() {
    # Remove temporary files
    rm -f "/tmp/single_folder_params.json"
    log_message "Cleanup completed"
}

# Main execution flow
main() {
    console_message "Starting single folder analysis for: $FOLDER_NAME"

    # Validations
    check_virtualenv
    validate_folder
    validate_scripts

    # Step 1: Interactive threshold tuning
    if run_threshold_tuning; then
        # Step 2: Analysis with optimized parameters
        if run_analysis; then
            console_message ""
            console_message "🎉 Single folder analysis completed successfully!"
            console_message "Results saved in: $FOLDER_PATH"
            console_message "Log file: $LOG_FILE"
        else
            console_message ""
            console_message "⚠️ Analysis failed. Check the log for details."
            console_message "Log file: $LOG_FILE"
        fi
    else
        console_message ""
        console_message "⚠️ Threshold tuning failed or was cancelled."
        console_message "Cannot proceed with analysis."
        console_message "Log file: $LOG_FILE"
    fi

    # Cleanup
    cleanup
}

# Set up trap for cleanup on exit
trap cleanup EXIT

# Run main function
main