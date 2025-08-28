#!/bin/bash

# Batch Analysis Script - Process folders individually with Python
# Each folder gets its own isolated Python process to prevent memory leaks
# Created for pyBayerSMLM super-resolution analysis pipeline

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/single_folder_analysis.py"
LOG_FILE="batch_analysis_$(date +%Y%m%d_%H%M%S).log"

# Initialize log file with header
{
    echo "============================================================"
    echo "pyBayerSMLM Batch Analysis - $(date)"
    echo "============================================================"
    echo "Script: $(basename "$0")"
    echo "Location: $SCRIPT_DIR"
    echo "Python Script: $PYTHON_SCRIPT"
    echo "Log File: $LOG_FILE"
    echo "============================================================"
    echo
} | tee "$LOG_FILE"

# Function to log with timestamp (detailed to log, summary to console)
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Function for console output (also logs)
console_message() {
    echo "$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Define all folder lists exactly from MemorySafe script
declare -a SM_DATA_DIRS=(
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250717_BiotinDyes/ATTO488_50PM_PCA_PCD'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/29250717_BiotinDyes/ATTO655_50PM _PCA_PCD'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/29250717_BiotinDyes/ATTO700_50PM _PCA_PCD'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250725 biotinylated dyes/ATTO514_50pM_PCAPCDTx'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250725 biotinylated dyes/ATTO520_50pM_PCAPCDTx'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250725 biotinylated dyes/ATTORho6G_50pM_PCAPCDTx'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto565_PCA_PCD_Tx_50pMDye'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto620_PCA_PCD_Tx_50pMDye'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250711 Biotinylated Dyes/Atto633_PCA_PCD_Tx_100pMDye'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto647N_PCA_PCD_Tx_20pMDye'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250609_dyes/data'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto594_PCA_PCD_Tx_50pMDye'
)

declare -a HELA_FOLDERS=(
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell3_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell4_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell2_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell1_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_2'
)

declare -a IMAGING_FOLDERS=(
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250717_Origami/F1F2F3F4Cy3B500pM/10perc561_LP561_BP586-64_1'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250717_Origami/F1F2F3F4Cy3B500pM_LowConcOrigami/10perc561_LP561_BP586-64_1'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250514_DNANanoruler/data/DNANanoRuler_10perc561_30mW488_50mW638/F1CF640CF550R_F2ATTO488AF647_F3ATTO565ATTO655_F4Cy3BCF488A_MultiNotch_488LP_758SP_1'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250514_DNANanoruler/data/DNANanoRuler_10perc561_30mW488_50mW638/F1CF640CF550R_F2ATTO488AF647_F3ATTO565ATTO655_F4Cy3BCF488A_MultiNotch_488LP_758SP_1nM_1'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250730 single colour origami/AlexaFluor647_2nM_strands/30mWboth638_NF_785SP_488LP_1'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250730 single colour origami/CF488A_2nM_strands/20mW488_NF_785SP_488LP_1'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250730 single colour origami/CF550R_2nM_strands_adjusteddichroic/30p561_NF_785SP_488LP_1'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250730 single colour origami/CF640R_2nM_strands/30mWboth638_NF_785SP_488LP_1'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250723 DNA Origami/FourColour_F1AF647_F2ATTO565_F3Cy3B_F4ATTO655_500pMEach/15percent_561_40mWEach_638_NotchFilter_785SP_1'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250723 DNA Origami/FourColour_F1AF647_F2ATTO565_F3Cy3B_F4ATTO655_500pMEach/15percent_561_100mWEach_638_NotchFilter_785SP_1'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250723 DNA Origami/FourColour_F1AF647_F2ATTO565_F3Cy3B_F4CF488A_500pMEach/30mW_488_15percent_561_100mWEach_638_NotchFilter_785SP_1'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250723 DNA Origami/FourColour_F1CF550R_F2ATTO565_F3Cy3B_F4CF488A_500pMEach/30mW_488_15percent_561_NotchFilter_785SP_1'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250716_iPSCJamesEvans/40mW488_30perc561_50mW638_NF_488LP_785SP_1'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250716_iPSCJamesEvans/250pMCy3B_250pM565_250pMCF550_250pM647/20perc561_40mW638_NF_488LP_785SP_1'
)

declare -a HIERARCHICAL_DIRS=(
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250414_CellPAINT/data'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250404_Ximea_AsynNRThX/data'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250818_DNAOrigami'
)

# Counters for tracking - fix subshell issue with temp files
COUNTER_DIR="/tmp/batch_counters_$$"
mkdir -p "$COUNTER_DIR"
echo "0" > "$COUNTER_DIR/total"
echo "0" > "$COUNTER_DIR/success"
echo "0" > "$COUNTER_DIR/error"
echo "0" > "$COUNTER_DIR/skip"

# Helper functions for counters
increment_counter() {
    local counter_file="$COUNTER_DIR/$1"
    local current=$(cat "$counter_file")
    echo $((current + 1)) > "$counter_file"
}

get_counter() {
    cat "$COUNTER_DIR/$1"
}

# Function to process single folder
process_folder() {
    local folder_type="$1"
    local folder_path="$2"
    local wavelength="$3"
    local folder_name=$(basename "$folder_path")
    
    increment_counter "total"
    local current_total=$(get_counter "total")
    
    # Console progress update
    echo -n "[$current_total] Processing: $folder_name... "
    
    # Detailed logging
    {
        echo
        echo "----------------------------------------"
        echo "Processing Folder: $folder_path"
        echo "Type: $folder_type"
        echo "Wavelength: $wavelength"
        echo "Started: $(date)"
        echo "----------------------------------------"
    } >> "$LOG_FILE"
    
    # Check if Python script exists
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        echo "ERROR - Script not found"
        log_message "ERROR: Python script not found: $PYTHON_SCRIPT"
        increment_counter "error"
        return 1
    fi
    
    # Run Python script for single folder (isolated process)
    if python3 "$PYTHON_SCRIPT" "$folder_type" "$folder_path" "$wavelength" >> "$LOG_FILE" 2>&1; then
        echo "✅ SUCCESS"
        log_message "SUCCESS: Processing completed for $folder_path"
        increment_counter "success"
        return 0
    else
        echo "❌ ERROR"
        log_message "ERROR: Processing failed for $folder_path"
        increment_counter "error"
        return 1
    fi
}

# Function to discover and process hierarchical folders  
process_hierarchical() {
    local base_dir="$1"
    local folder_type="$2"
    local wavelength="$3"
    
    if [ ! -d "$base_dir" ]; then
        log_message "WARNING: Directory not found: $base_dir"
        return 0
    fi
    
    log_message "Scanning hierarchical directory: $base_dir"
    
    # Use os.walk equivalent - find all directories and check if they're leaves
    find "$base_dir" -type d | while read -r folder; do
        # Skip the base directory itself
        if [ "$folder" = "$base_dir" ]; then
            continue
        fi
        
        # Check if this directory contains any subdirectories
        has_subdirs=$(find "$folder" -maxdepth 1 -type d ! -path "$folder" | head -1)
        
        # If no subdirectories found, it's a leaf directory
        if [ -z "$has_subdirs" ]; then
            log_message "Found leaf directory: $folder"
            process_folder "$folder_type" "$folder" "$wavelength"
        fi
    done
}

console_message "Starting folder discovery and processing..."

# Process SM data hierarchical directories
console_message "Processing SM data directories (${#SM_DATA_DIRS[@]} base directories)..."
for base_dir in "${SM_DATA_DIRS[@]}"; do
    process_hierarchical "$base_dir" "sm" "0.638"
done

# Process HeLa folders directly (647nm wavelength)
console_message "Processing HeLa imaging folders (${#HELA_FOLDERS[@]} folders)..."
for folder in "${HELA_FOLDERS[@]}"; do
    if [ -d "$folder" ]; then
        process_folder "imaging" "$folder" "0.647"
    else
        log_message "WARNING: HeLa folder not found: $folder"
    fi
done

# Process imaging folders directly (550nm default)
console_message "Processing general imaging folders (${#IMAGING_FOLDERS[@]} folders)..."
for folder in "${IMAGING_FOLDERS[@]}"; do
    if [ -d "$folder" ]; then
        process_folder "imaging" "$folder" "0.55"
    else
        log_message "WARNING: Imaging folder not found: $folder"
    fi
done

# Process hierarchical imaging directories
console_message "Processing hierarchical imaging directories (${#HIERARCHICAL_DIRS[@]} base directories)..."
for base_dir in "${HIERARCHICAL_DIRS[@]}"; do
    process_hierarchical "$base_dir" "imaging" "0.55"
done

echo  # New line after progress output

# Final summary
TOTAL_FOLDERS=$(get_counter "total")
SUCCESS_COUNT=$(get_counter "success")
ERROR_COUNT=$(get_counter "error")
SKIP_COUNT=$(get_counter "skip")

# Console summary
{
    echo "============================================================"
    echo "BATCH ANALYSIS COMPLETE - $(date)"
    echo "============================================================"
    echo "Total folders processed: $TOTAL_FOLDERS"
    echo "Successful: $SUCCESS_COUNT"
    echo "Errors: $ERROR_COUNT"
    echo "Skipped: $SKIP_COUNT"
    echo "Detailed log saved to: $LOG_FILE"
    echo "============================================================"
} | tee -a "$LOG_FILE"

# Detailed final log entry
{
    echo
    echo "==================== FINAL SUMMARY ===================="
    echo "Analysis completed: $(date)"
    echo "Total processing time: $SECONDS seconds"
    echo "Success rate: $(( SUCCESS_COUNT * 100 / (TOTAL_FOLDERS == 0 ? 1 : TOTAL_FOLDERS) ))%"
    echo "======================================================="
} >> "$LOG_FILE"

# Cleanup
rm -rf "$COUNTER_DIR"

if [ "$ERROR_COUNT" -eq 0 ] && [ "$TOTAL_FOLDERS" -gt 0 ]; then
    console_message "🎉 All folders processed successfully!"
    exit 0
elif [ "$TOTAL_FOLDERS" -eq 0 ]; then
    console_message "⚠️  No folders found to process. Check directory paths."
    exit 1
else
    console_message "⚠️  $ERROR_COUNT/$TOTAL_FOLDERS folders had errors. Check the log for details."
    exit 1
fi