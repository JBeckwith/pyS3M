#!/bin/bash

# Nile Red Analysis Script - Process folders from threshold parameters file
# Reads folder paths and parameters from nile_red_threshold_parameters.txt
# Created for pyBayerSMLM super-resolution analysis pipeline

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/single_folder_analysis.py"
THRESHOLD_PARAMS_FILE="$SCRIPT_DIR/nile_red_threshold_parameters.txt"

# Check for threshold parameters file
if [ ! -f "$THRESHOLD_PARAMS_FILE" ]; then
    echo "ERROR: threshold parameters file not found: $THRESHOLD_PARAMS_FILE"
    echo ""
    echo "Please run the interactive threshold tuner first:"
    echo "  python NileRedAnalysisTuner.py <folder_path>"
    echo ""
    echo "This will generate the required parameter file."
    exit 1
fi

# Create log file
LOG_FILE="nile_red_analysis_$(date +%Y%m%d_%H%M%S).log"

# Initialize log file with header
{
    echo "============================================================"
    echo "Nile Red Analysis - $(date)"
    echo "============================================================"
    echo "Script: $(basename "$0")"
    echo "Location: $SCRIPT_DIR"
    echo "Python Script: $PYTHON_SCRIPT"
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

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    console_message "ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Count total folders to process
TOTAL_FOLDERS=$(grep -v "^#" "$THRESHOLD_PARAMS_FILE" | grep -v "^$" | wc -l)

if [ "$TOTAL_FOLDERS" -eq 0 ]; then
    console_message "ERROR: No folders found in $THRESHOLD_PARAMS_FILE"
    console_message ""
    console_message "The parameter file appears to be empty or contains only comments."
    console_message "Please run the threshold tuner to generate parameters:"
    console_message "  python NileRedAnalysisTuner.py <folder_path>"
    exit 1
fi

console_message "Found $TOTAL_FOLDERS folder(s) to process"
console_message ""

# Process each folder
FOLDER_COUNT=0
SUCCESS_COUNT=0
FAIL_COUNT=0

while IFS='|' read -r folder_path pfa sigma fraction_true wavelength use_variance_aware ever_mode ever_window; do
    # Skip comments and empty lines
    [[ "$folder_path" =~ ^#.*$ ]] && continue
    [[ -z "$folder_path" ]] && continue

    FOLDER_COUNT=$((FOLDER_COUNT + 1))

    console_message "============================================================"
    console_message "Processing folder $FOLDER_COUNT/$TOTAL_FOLDERS"
    console_message "============================================================"
    console_message "Folder: $folder_path"
    console_message "Parameters:"
    console_message "  PFA: $pfa"
    console_message "  Sigma: $sigma"
    console_message "  Fraction True: $fraction_true"
    console_message "  Wavelength: $wavelength"
    console_message "  Variance-aware demosaicing: $use_variance_aware"
    console_message "  EVER mode: $ever_mode"
    console_message "  EVER window: $ever_window"
    console_message ""

    # Verify folder exists
    if [ ! -d "$folder_path" ]; then
        console_message "⚠️  WARNING: Folder not found: $folder_path"
        console_message "Skipping..."
        console_message ""
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # Run analysis
    console_message "Starting analysis..."
    log_message "Running: python3 $PYTHON_SCRIPT imaging $folder_path $folder_path $wavelength $pfa $sigma $fraction_true $use_variance_aware $ever_mode $ever_window"

    if python3 "$PYTHON_SCRIPT" "imaging" "$folder_path" "$folder_path" "$wavelength" "$pfa" "$sigma" "$fraction_true" "$use_variance_aware" "$ever_mode" "$ever_window" >> "$LOG_FILE" 2>&1; then
        console_message "✅ Analysis completed successfully"

        # Check for output files
        H5_COUNT=$(find "$folder_path" -name "*.h5" -type f 2>/dev/null | wc -l)
        console_message "Generated $H5_COUNT .h5 files"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        console_message "❌ Analysis failed"
        console_message "Check log file for details: $LOG_FILE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    console_message ""

done < "$THRESHOLD_PARAMS_FILE"

# Summary
console_message "============================================================"
console_message "Analysis Complete - $(date)"
console_message "============================================================"
console_message "Total folders: $TOTAL_FOLDERS"
console_message "Successfully processed: $SUCCESS_COUNT"
console_message "Failed: $FAIL_COUNT"
console_message ""
console_message "Log saved to: $LOG_FILE"

# Exit with appropriate code
if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
else
    exit 0
fi
