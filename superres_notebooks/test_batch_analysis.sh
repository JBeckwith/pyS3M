#!/bin/bash

# Test Batch Analysis Script - Works locally for demonstration
# This version creates test folders and processes them to show the concept

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/single_folder_analysis.py"
LOG_FILE="test_batch_analysis_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================" | tee -a "$LOG_FILE"
echo "Starting TEST Batch Analysis - $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Create test folder structure
TEST_BASE="/tmp/smlm_test"
rm -rf "$TEST_BASE"
mkdir -p "$TEST_BASE"

log_message "Creating test folder structure in $TEST_BASE"

# Create some test folders with required files
mkdir -p "$TEST_BASE/sm_data/dye1" "$TEST_BASE/sm_data/dye2"
mkdir -p "$TEST_BASE/imaging/cell1" "$TEST_BASE/imaging/cell2" "$TEST_BASE/imaging/cell3"
mkdir -p "$TEST_BASE/hierarchical/exp1/session1" "$TEST_BASE/hierarchical/exp1/session2"
mkdir -p "$TEST_BASE/hierarchical/exp2/session1"

# Add required files to each test folder
for folder in "$TEST_BASE"/sm_data/*/; do
    touch "$folder/test.tif" "$folder/metadata.txt"
done

for folder in "$TEST_BASE"/imaging/*/; do
    touch "$folder/test.tif" "$folder/metadata.txt"
done

for folder in "$TEST_BASE"/hierarchical/exp*/session*/; do
    touch "$folder/test.tif" "$folder/metadata.txt"
done

log_message "Test structure created"

# Counters - fix subshell issue by using temporary files
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
    
    increment_counter "total"
    
    log_message "Processing: $folder_path (type: $folder_type, wavelength: $wavelength)"
    
    # Check if Python script exists
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        log_message "ERROR: Python script not found: $PYTHON_SCRIPT"
        increment_counter "error"
        return 1
    fi
    
    # Run Python script for single folder (isolated process)
    if python3 "$PYTHON_SCRIPT" "$folder_type" "$folder_path" "$wavelength" >> "$LOG_FILE" 2>&1; then
        log_message "✅ SUCCESS: $folder_path"
        increment_counter "success"
        return 0
    else
        log_message "❌ ERROR: $folder_path"
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
    
    # Find leaf directories (directories with no subdirectories)
    find "$base_dir" -type d | while read -r folder; do
        # Skip the base directory itself
        if [ "$folder" = "$base_dir" ]; then
            continue
        fi
        
        # Check if this directory contains any subdirectories
        subdirs_count=$(find "$folder" -maxdepth 1 -type d ! -path "$folder" | wc -l)
        
        # If no subdirectories found, it's a leaf directory
        if [ "$subdirs_count" -eq 0 ]; then
            log_message "Found leaf directory: $folder"
            process_folder "$folder_type" "$folder" "$wavelength"
        fi
    done
}

log_message "Discovering and processing folders..."

# Process SM data hierarchical directories  
log_message "Processing SM data directories..."
process_hierarchical "$TEST_BASE/sm_data" "sm" "0.638"

# Process imaging folders directly
log_message "Processing imaging directories..."
for folder in "$TEST_BASE"/imaging/*/; do
    if [ -d "$folder" ]; then
        process_folder "imaging" "$folder" "0.55"
    fi
done

# Process hierarchical imaging directories
log_message "Processing hierarchical imaging directories..."  
process_hierarchical "$TEST_BASE/hierarchical" "imaging" "0.55"

# Final summary
TOTAL_FOLDERS=$(get_counter "total")
SUCCESS_COUNT=$(get_counter "success")  
ERROR_COUNT=$(get_counter "error")
SKIP_COUNT=$(get_counter "skip")

echo "============================================================" | tee -a "$LOG_FILE"
echo "TEST BATCH ANALYSIS COMPLETE - $(date)" | tee -a "$LOG_FILE" 
echo "============================================================" | tee -a "$LOG_FILE"
log_message "Total folders processed: $TOTAL_FOLDERS"
log_message "Successful: $SUCCESS_COUNT"
log_message "Errors: $ERROR_COUNT" 
log_message "Skipped: $SKIP_COUNT"
log_message "Log saved to: $LOG_FILE"

# Cleanup
rm -rf "$COUNTER_DIR"
rm -rf "$TEST_BASE"

if [ "$ERROR_COUNT" -eq 0 ] && [ "$TOTAL_FOLDERS" -gt 0 ]; then
    log_message "🎉 Test completed successfully! Found and processed $TOTAL_FOLDERS folders."
    exit 0
else
    log_message "⚠️  Test had issues: $ERROR_COUNT errors out of $TOTAL_FOLDERS folders."
    exit 1
fi