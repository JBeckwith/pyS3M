#!/bin/bash

# Nile Red Analysis Script - Process a single folder with configurable parameters
# Takes a folder path as input and processes it using parameters from threshold_parameters.txt
# Created for pyBayerSMLM super-resolution analysis pipeline
# Based on batch_analysis.sh but simplified for single-folder processing

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/single_folder_analysis.py"
THRESHOLD_PARAMS_FILE="$SCRIPT_DIR/nile_red_threshold_parameters.txt"

# Check arguments
if [ $# -lt 1 ]; then
    echo "ERROR: Missing required argument"
    echo ""
    echo "Usage:"
    echo "  $0 <folder_path>"
    echo ""
    echo "Example:"
    echo "  $0 /scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20251128_SAureus/data"
    echo ""
    echo "This script processes Nile Red imaging data using parameters from:"
    echo "  nile_red_threshold_parameters.txt"
    echo ""
    echo "To generate the parameter file, first run:"
    echo "  python NileRedAnalysisTuner.py <folder_path>"
    exit 1
fi

FOLDER_PATH="$1"

# Verify folder exists
if [ ! -d "$FOLDER_PATH" ]; then
    echo "ERROR: Folder not found: $FOLDER_PATH"
    exit 1
fi

# Check for threshold parameters file
if [ ! -f "$THRESHOLD_PARAMS_FILE" ]; then
    echo "ERROR: threshold parameters file not found: $THRESHOLD_PARAMS_FILE"
    echo ""
    echo "Please run the interactive threshold tuner first:"
    echo "  python NileRedAnalysisTuner.py $FOLDER_PATH"
    echo ""
    echo "This will generate the required parameter file."
    exit 1
fi

# Create log file
LOG_FILE="nile_red_analysis_$(basename "$FOLDER_PATH")_$(date +%Y%m%d_%H%M%S).log"

# Initialize log file with header
{
    echo "============================================================"
    echo "Nile Red Analysis - $(date)"
    echo "============================================================"
    echo "Script: $(basename "$0")"
    echo "Location: $SCRIPT_DIR"
    echo "Python Script: $PYTHON_SCRIPT"
    echo "Folder: $FOLDER_PATH"
    echo "Log File: $LOG_FILE"
    echo "Threshold Parameters: $THRESHOLD_PARAMS_FILE"
    echo "============================================================"
    echo
} | tee "$LOG_FILE"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Function for console output (also logs)
console_message() {
    echo "$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Function to get threshold parameters for the folder
get_threshold_params() {
    local folder_path="$1"
    local default_pfa="1e-3"
    local default_sigma="1.5"
    local default_fraction_true="0.25"
    local default_wavelength="0.650"
    local default_use_variance_aware="true"
    local default_ever_mode="0"  # Default to NONE (EVER off)
    local default_ever_window="500"

    # Look for exact match first
    local params_line=$(grep -v "^#" "$THRESHOLD_PARAMS_FILE" | grep "^$folder_path|" | head -1)

    if [ -n "$params_line" ]; then
        # Extract parameters (format: folder|pfa|sigma|fraction_true|wavelength|use_variance_aware|ever_mode|ever_window)
        local pfa=$(echo "$params_line" | cut -d'|' -f2)
        local sigma=$(echo "$params_line" | cut -d'|' -f3)
        local fraction_true=$(echo "$params_line" | cut -d'|' -f4)
        local wavelength=$(echo "$params_line" | cut -d'|' -f5)
        local use_variance_aware=$(echo "$params_line" | cut -d'|' -f6)
        local ever_mode=$(echo "$params_line" | cut -d'|' -f7)
        local ever_window=$(echo "$params_line" | cut -d'|' -f8)
        echo "$pfa $sigma $fraction_true $wavelength $use_variance_aware $ever_mode $ever_window"
        log_message "Using parameters for $folder_path: pfa=$pfa, sigma=$sigma, fraction_true=$fraction_true, wavelength=$wavelength, variance_aware=$use_variance_aware, ever_mode=$ever_mode, ever_window=$ever_window"
    else
        # Use defaults
        echo "$default_pfa $default_sigma $default_fraction_true $default_wavelength $default_use_variance_aware $default_ever_mode $default_ever_window"
        log_message "Using default parameters for $folder_path"
    fi
}

# Get parameters
PARAMS=($(get_threshold_params "$FOLDER_PATH"))
PFA="${PARAMS[0]}"
SIGMA="${PARAMS[1]}"
FRACTION_TRUE="${PARAMS[2]}"
WAVELENGTH="${PARAMS[3]}"
USE_VARIANCE_AWARE="${PARAMS[4]}"
EVER_MODE="${PARAMS[5]}"
EVER_WINDOW="${PARAMS[6]}"

console_message "Processing folder: $(basename "$FOLDER_PATH")"
console_message "Parameters:"
console_message "  PFA: $PFA"
console_message "  Sigma: $SIGMA"
console_message "  Fraction True: $FRACTION_TRUE"
console_message "  Wavelength: $WAVELENGTH"
console_message "  Variance-aware demosaicing: $USE_VARIANCE_AWARE"
console_message "  EVER mode: $EVER_MODE"
console_message "  EVER window: $EVER_WINDOW"
console_message ""

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    console_message "ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Run analysis
console_message "Starting analysis..."
log_message "Running: python3 $PYTHON_SCRIPT imaging $FOLDER_PATH $FOLDER_PATH $WAVELENGTH $PFA $SIGMA $FRACTION_TRUE $USE_VARIANCE_AWARE $EVER_MODE $EVER_WINDOW"

if python3 "$PYTHON_SCRIPT" "imaging" "$FOLDER_PATH" "$FOLDER_PATH" "$WAVELENGTH" "$PFA" "$SIGMA" "$FRACTION_TRUE" "$USE_VARIANCE_AWARE" "$EVER_MODE" "$EVER_WINDOW" >> "$LOG_FILE" 2>&1; then
    console_message "✅ Analysis completed successfully"

    # Check for output files
    H5_COUNT=$(find "$FOLDER_PATH" -name "*.h5" -type f | wc -l)
    console_message "Generated $H5_COUNT .h5 files"
    exit_code=0
else
    console_message "❌ Analysis failed"
    console_message "Check log file for details: $LOG_FILE"
    exit_code=1
fi

console_message "Log saved to: $LOG_FILE"
exit $exit_code
