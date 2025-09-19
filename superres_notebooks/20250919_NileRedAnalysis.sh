#!/bin/bash

# Nile Red Analysis Script - Process specific bacteria data folders
# Adapted from batch_analysis.sh for 20250918/20250919 Bacteria with Sina experiments
# Each folder gets its own isolated Python process to prevent memory leaks
# Created for pyBayerSMLM super-resolution analysis pipeline
# Enhanced with swap usage minimization and memory monitoring
# Uses variance-aware demosaicing settings from interactive threshold tuner

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/single_folder_analysis.py"
LOG_FILE="nile_red_analysis_$(date +%Y%m%d_%H%M%S).log"
THRESHOLD_PARAMS_FILE="$SCRIPT_DIR/20250919_nile_red_threshold_parameters.txt"

# Memory management configuration - more aggressive settings
MEMORY_CHECK_INTERVAL=3  # seconds between memory checks (more frequent)
MAX_SWAP_USAGE_MB=1024   # Maximum swap usage before pausing (1GB - more conservative)
MAX_MEMORY_USAGE_PCT=75  # Maximum RAM usage percentage before pausing (more conservative)
COOLDOWN_TIME=20         # seconds to wait when memory usage is high (shorter)

# Check for threshold parameters file
check_threshold_params() {
    if [ ! -f "$THRESHOLD_PARAMS_FILE" ]; then
        echo "ERROR: 20250919_nile_red_threshold_parameters.txt not found!"
        echo "Please run the interactive threshold tuner first:"
        echo "  source /home/jbeckwith/.virtualenvs/pyBayerSMLM/bin/activate"
        echo "  python superres_notebooks/20250919_NileRedAnalysisTuner.py"
        echo ""
        echo "The interactive threshold tuner generates optimized parameters"
        echo "for spot detection that are required for batch analysis."
        log_message "ERROR: 20250919_nile_red_threshold_parameters.txt missing - 20250919_NileRedAnalysisTuner must be run first"
        exit 1
    fi

    # Validate file format (check for pipe-delimited content with 5 fields: folder|pfa|sigma|fraction_true|wavelength)
    if ! grep -q "|" "$THRESHOLD_PARAMS_FILE"; then
        echo "WARNING: 20250919_nile_red_threshold_parameters.txt format may be invalid (no pipe delimiters found)"
        log_message "WARNING: 20250919_nile_red_threshold_parameters.txt format validation failed - no pipe delimiters"
    fi

    # Check if format has new parameters (sigma and fraction_true)
    local sample_line=$(grep -v "^#" "$THRESHOLD_PARAMS_FILE" | head -1)
    if [ -n "$sample_line" ]; then
        local field_count=$(echo "$sample_line" | tr '|' '\n' | wc -l)
        if [ "$field_count" -eq 4 ]; then
            echo "INFO: Using legacy format (folder|pfa|perc_threshold|wavelength)"
            log_message "INFO: Legacy 20250919_nile_red_threshold_parameters.txt format detected (4 fields)"
        elif [ "$field_count" -eq 5 ]; then
            echo "INFO: Using enhanced format (folder|pfa|sigma|fraction_true|wavelength)"
            log_message "INFO: Enhanced 20250919_nile_red_threshold_parameters.txt format detected (5 fields)"
        elif [ "$field_count" -eq 6 ]; then
            echo "INFO: Using full format (folder|pfa|sigma|fraction_true|wavelength|use_variance_aware)"
            log_message "INFO: Full 20250919_nile_red_threshold_parameters.txt format detected (6 fields)"
        else
            echo "WARNING: Unexpected 20250919_nile_red_threshold_parameters.txt format ($field_count fields)"
            log_message "WARNING: Unexpected 20250919_nile_red_threshold_parameters.txt format with $field_count fields"
        fi
    fi

    local param_count=$(grep -v "^#" "$THRESHOLD_PARAMS_FILE" | wc -l)
    echo "Found threshold parameters for $param_count folders"
    log_message "Loaded threshold parameters for $param_count folders from $THRESHOLD_PARAMS_FILE"
}

# Function to get threshold parameters for a folder
get_threshold_params() {
    local folder_path="$1"
    local default_pfa="1e-4"
    local default_sigma="1.5"
    local default_fraction_true="0.2"
    local default_wavelength="$2"
    local default_use_variance_aware="true"

    # Look for exact match first
    local params_line=$(grep -v "^#" "$THRESHOLD_PARAMS_FILE" | grep "^$folder_path|" | head -1)

    if [ -n "$params_line" ]; then
        # Check format based on field count
        local field_count=$(echo "$params_line" | tr '|' '\n' | wc -l)

        if [ "$field_count" -eq 6 ]; then
            # Full format: folder_path|pfa|sigma|fraction_true|wavelength|use_variance_aware
            local pfa=$(echo "$params_line" | cut -d'|' -f2)
            local sigma=$(echo "$params_line" | cut -d'|' -f3)
            local fraction_true=$(echo "$params_line" | cut -d'|' -f4)
            local wavelength=$(echo "$params_line" | cut -d'|' -f5)
            local use_variance_aware=$(echo "$params_line" | cut -d'|' -f6)
            echo "$pfa $sigma $fraction_true $wavelength $use_variance_aware"
            log_message "Using full parameters for $folder_path: pfa=$pfa, sigma=$sigma, fraction_true=$fraction_true, wavelength=$wavelength, variance_aware=$use_variance_aware"
        elif [ "$field_count" -eq 5 ]; then
            # Enhanced format: folder_path|pfa|sigma|fraction_true|wavelength
            local pfa=$(echo "$params_line" | cut -d'|' -f2)
            local sigma=$(echo "$params_line" | cut -d'|' -f3)
            local fraction_true=$(echo "$params_line" | cut -d'|' -f4)
            local wavelength=$(echo "$params_line" | cut -d'|' -f5)
            echo "$pfa $sigma $fraction_true $wavelength $default_use_variance_aware"
            log_message "Using enhanced parameters for $folder_path: pfa=$pfa, sigma=$sigma, fraction_true=$fraction_true, wavelength=$wavelength, variance_aware=$default_use_variance_aware (default)"
        elif [ "$field_count" -eq 4 ]; then
            # Legacy format: folder_path|pfa|perc_threshold|wavelength (convert perc_threshold to defaults)
            local pfa=$(echo "$params_line" | cut -d'|' -f2)
            local perc_threshold=$(echo "$params_line" | cut -d'|' -f3)  # Ignored in new format
            local wavelength=$(echo "$params_line" | cut -d'|' -f4)
            echo "$pfa $default_sigma $default_fraction_true $wavelength $default_use_variance_aware"
            log_message "Using legacy parameters for $folder_path: pfa=$pfa, sigma=$default_sigma (default), fraction_true=$default_fraction_true (default), wavelength=$wavelength, variance_aware=$default_use_variance_aware (default)"
        else
            # Invalid format, use defaults
            echo "$default_pfa $default_sigma $default_fraction_true $default_wavelength $default_use_variance_aware"
            log_message "Invalid parameter format for $folder_path, using defaults: pfa=$default_pfa, sigma=$default_sigma, fraction_true=$default_fraction_true, wavelength=$default_wavelength, variance_aware=$default_use_variance_aware"
        fi
    else
        # Use defaults
        echo "$default_pfa $default_sigma $default_fraction_true $default_wavelength $default_use_variance_aware"
        log_message "Using default parameters for $folder_path: pfa=$default_pfa, sigma=$default_sigma, fraction_true=$default_fraction_true, wavelength=$default_wavelength, variance_aware=$default_use_variance_aware"
    fi
}

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

# Function to log with timestamp (detailed to log, summary to console)
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Function for console output (also logs)
console_message() {
    echo "$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Check for nocache availability to reduce filesystem cache pressure (after log functions defined)
NOCACHE_CMD=""
if command -v nocache >/dev/null 2>&1; then
    NOCACHE_CMD="nocache"
    log_message "nocache command available - will use for file operations"
else
    log_message "nocache command not available - using standard file operations"
fi

# Define Nile Red analysis folder lists
declare -a NILE_RED_FOLDERS=(
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250918_BacteriawithSina/NR4A'
    '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250919_BacteriawithSina/NR4A'
)

# Counters for tracking - fix subshell issue with temp files
COUNTER_DIR="/tmp/nile_red_counters_$$"
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

# Memory monitoring functions
get_memory_stats() {
    # Get memory statistics in a single call to reduce overhead
    local mem_info=$(free -m)
    local swap_info=$(echo "$mem_info" | grep "Swap:")
    local mem_info_line=$(echo "$mem_info" | grep "Mem:")

    # Extract values
    local total_ram=$(echo "$mem_info_line" | awk '{print $2}')
    local used_ram=$(echo "$mem_info_line" | awk '{print $3}')
    local total_swap=$(echo "$swap_info" | awk '{print $2}')
    local used_swap=$(echo "$swap_info" | awk '{print $3}')

    # Calculate percentages
    local ram_usage_pct=0
    if [ "$total_ram" -gt 0 ]; then
        ram_usage_pct=$(( (used_ram * 100) / total_ram ))
    fi

    # Return values separated by spaces
    echo "$used_swap $total_swap $ram_usage_pct $total_ram $used_ram"
}

check_memory_pressure() {
    local stats=($(get_memory_stats))
    local used_swap_mb=${stats[0]}
    local total_swap_mb=${stats[1]}
    local ram_usage_pct=${stats[2]}
    local total_ram_mb=${stats[3]}
    local used_ram_mb=${stats[4]}

    # Log current memory status
    log_message "Memory: ${used_ram_mb}MB/${total_ram_mb}MB RAM (${ram_usage_pct}%), ${used_swap_mb}MB/${total_swap_mb}MB swap"

    # Check if memory pressure is too high
    local memory_pressure=false
    local pressure_reasons=""

    if [ "$used_swap_mb" -gt "$MAX_SWAP_USAGE_MB" ]; then
        memory_pressure=true
        pressure_reasons="swap usage ${used_swap_mb}MB > ${MAX_SWAP_USAGE_MB}MB"
    fi

    if [ "$ram_usage_pct" -gt "$MAX_MEMORY_USAGE_PCT" ]; then
        memory_pressure=true
        if [ -n "$pressure_reasons" ]; then
            pressure_reasons="$pressure_reasons, "
        fi
        pressure_reasons="${pressure_reasons}RAM usage ${ram_usage_pct}% > ${MAX_MEMORY_USAGE_PCT}%"
    fi

    if [ "$memory_pressure" = true ]; then
        log_message "WARNING: High memory pressure detected ($pressure_reasons)"
        return 1
    fi

    return 0
}

wait_for_memory_relief() {
    log_message "Waiting for memory pressure to decrease..."
    echo -n "⏳ High memory usage, waiting for relief... "

    local wait_count=0
    while ! check_memory_pressure >/dev/null 2>&1; do
        sleep "$MEMORY_CHECK_INTERVAL"
        wait_count=$((wait_count + MEMORY_CHECK_INTERVAL))

        # Show progress dots
        echo -n "."

        # Aggressive memory cleanup every 15 seconds
        if [ $((wait_count % 15)) -eq 0 ]; then
            echo -n " [cleanup] "

            # Force filesystem sync to reduce buffer cache
            sync

            # Force Python garbage collection for any running processes
            pkill -SIGUSR1 python3 2>/dev/null || true

            # Clear user-space caches that we can control using nocache
            # Clear any temporary files older than 1 hour in common temp locations
            if [ -n "$NOCACHE_CMD" ]; then
                $NOCACHE_CMD find /tmp -user "$(whoami)" -type f -mmin +60 -delete 2>/dev/null || true
                $NOCACHE_CMD find /scratch2/jsb92 -name "*.tmp" -mmin +30 -delete 2>/dev/null || true
            else
                find /tmp -user "$(whoami)" -type f -mmin +60 -delete 2>/dev/null || true
                find /scratch2/jsb92 -name "*.tmp" -mmin +30 -delete 2>/dev/null || true
            fi

            # Force malloc to return memory to system
            export MALLOC_TRIM_THRESHOLD_=65536

            log_message "Performed aggressive cleanup cycle (${wait_count}s elapsed)"
        fi

        # More frequent lighter cleanup every 6 seconds
        if [ $((wait_count % 6)) -eq 0 ]; then
            sync
        fi

        # Timeout after 8 minutes (reduced from 10 minutes)
        if [ "$wait_count" -gt 480 ]; then
            echo " ⏰ TIMEOUT"
            log_message "WARNING: Memory pressure timeout after 8 minutes, continuing anyway"
            break
        fi
    done

    echo " ✅ READY"
    log_message "Memory pressure relieved, continuing processing"
}

optimize_system_memory() {
    log_message "Optimizing user-space memory settings for batch processing"

    # Only do user-space optimizations that don't require root
    sync

    log_message "User-space memory optimization complete"
}

# Selective copy with retry logic - only copy files needed for analysis
# Copies .tif/.tiff files (for analysis) and essential metadata (NO .h5 files - date check done earlier)
selective_copy() {
    local src="$1"
    local dst="$2"
    local max_attempts=5
    local wait_time=10
    local folder_name=$(basename "$src")
    local target_dir="$dst/$folder_name"

    # Create target directory
    mkdir -p "$target_dir" || return 1

    for attempt in $(seq 1 $max_attempts); do
        log_message "Selective copy attempt $attempt/$max_attempts: copying only TIFF files and metadata"

        local copy_success=true

        # Copy TIFF files (needed for analysis) using nocache to reduce cache pressure
        find "$src" -name "*.tif" -o -name "*.tiff" -type f | while read -r tiff_file; do
            local rel_path=$(realpath --relative-to="$src" "$tiff_file")
            local dest_file="$target_dir/$rel_path"
            local dest_dir=$(dirname "$dest_file")
            mkdir -p "$dest_dir"
            if ! $NOCACHE_CMD cp "$tiff_file" "$dest_file" 2>> "$LOG_FILE"; then
                log_message "Failed to copy TIFF file: $tiff_file"
                copy_success=false
            fi
        done

        # Copy essential metadata files (small files) using nocache
        find "$src" -name "*.txt" -o -name "*.json" -o -name "*.csv" -o -name "*.xml" -type f -size -1M | while read -r meta_file; do
            local rel_path=$(realpath --relative-to="$src" "$meta_file")
            local dest_file="$target_dir/$rel_path"
            local dest_dir=$(dirname "$dest_file")
            mkdir -p "$dest_dir"
            $NOCACHE_CMD cp "$meta_file" "$dest_file" 2>> "$LOG_FILE" || true  # Non-critical files
        done

        if [ "$copy_success" = true ]; then
            log_message "Selective copy successful on attempt $attempt"
            return 0
        else
            log_message "Selective copy failed on attempt $attempt, waiting ${wait_time}s before retry"
            if [ $attempt -lt $max_attempts ]; then
                sleep $wait_time
                wait_time=$((wait_time * 2))  # Exponential backoff
                # Clean up partial copy before retry
                rm -rf "$target_dir" 2>/dev/null || true
                mkdir -p "$target_dir"
            fi
        fi
    done

    log_message "ERROR: Selective copy failed after $max_attempts attempts"
    return 1
}

# Copy .h5 files back with retry logic
copy_results_back() {
    local scratch_folder="$1"
    local original_folder="$2"
    local max_attempts=3
    local wait_time=5

    # First, clean up any .h5.backup files that might exist in scratch folder
    log_message "Cleaning up .h5.backup files from scratch folder"
    local scratch_backup_files=($(find "$scratch_folder" -name "*.h5.backup" -type f 2>/dev/null))
    for backup_file in "${scratch_backup_files[@]}"; do
        if rm "$backup_file" 2>> "$LOG_FILE"; then
            log_message "Removed scratch .h5.backup file: $(basename "$backup_file")"
        else
            log_message "WARNING: Could not remove scratch .h5.backup file: $(basename "$backup_file")"
        fi
    done

    # Find all .h5 files in scratch folder
    local h5_files=($(find "$scratch_folder" -name "*.h5" -type f))

    if [ ${#h5_files[@]} -eq 0 ]; then
        log_message "No .h5 files found to copy back"
        return 0
    fi

    log_message "Found ${#h5_files[@]} .h5 files to copy back"

    # Clean up existing .h5 and .h5.backup files in the destination folder first
    log_message "Cleaning up existing .h5 and .h5.backup files in destination"

    local existing_h5_files=($(find "$original_folder" -maxdepth 1 -name "*.h5" -type f 2>/dev/null))
    local existing_backup_files=($(find "$original_folder" -maxdepth 1 -name "*.h5.backup" -type f 2>/dev/null))

    # Remove existing .h5 files
    for existing_file in "${existing_h5_files[@]}"; do
        if rm "$existing_file" 2>> "$LOG_FILE"; then
            log_message "Removed existing .h5 file: $(basename "$existing_file")"
        else
            log_message "WARNING: Could not remove existing .h5 file: $(basename "$existing_file")"
        fi
    done

    # Remove existing .h5.backup files
    for backup_file in "${existing_backup_files[@]}"; do
        if rm "$backup_file" 2>> "$LOG_FILE"; then
            log_message "Removed existing .h5.backup file: $(basename "$backup_file")"
        else
            log_message "WARNING: Could not remove existing .h5.backup file: $(basename "$backup_file")"
        fi
    done

    for h5_file in "${h5_files[@]}"; do
        local filename=$(basename "$h5_file")
        local dest_file="$original_folder/$filename"

        for attempt in $(seq 1 $max_attempts); do
            log_message "Copying back attempt $attempt/$max_attempts: $filename"

            if cp "$h5_file" "$dest_file" 2>> "$LOG_FILE"; then
                log_message "Successfully copied back: $filename"
                break
            else
                log_message "Failed to copy back $filename on attempt $attempt"
                if [ $attempt -lt $max_attempts ]; then
                    sleep $wait_time
                fi
            fi
        done

        # Verify copy was successful
        if [ -f "$dest_file" ]; then
            log_message "Verified: $filename exists in original location"
        else
            log_message "ERROR: Failed to copy back $filename after $max_attempts attempts"
        fi
    done
}

# Safe cleanup with verification
safe_cleanup() {
    local scratch_folder="$1"

    # Verify we're cleaning up the right directory (safety check)
    if [[ "$scratch_folder" != /scratch2/jsb92/* ]]; then
        log_message "ERROR: Refusing to cleanup directory outside /scratch2/jsb92: $scratch_folder"
        return 1
    fi

    if [ -d "$scratch_folder" ]; then
        log_message "Cleaning up scratch directory: $scratch_folder"
        if rm -rf "$scratch_folder" 2>> "$LOG_FILE"; then
            log_message "Successfully cleaned up: $scratch_folder"
        else
            log_message "WARNING: Failed to cleanup: $scratch_folder"
        fi
    else
        log_message "Scratch directory already cleaned or doesn't exist: $scratch_folder"
    fi
}

# Function to process single folder with scratch workflow
process_folder() {
    local folder_type="$1"
    local folder_path="$2"
    local wavelength="$3"
    local folder_name=$(basename "$folder_path")
    local scratch_folder="/scratch2/jsb92/$folder_name"

    # Get threshold parameters for this folder
    local threshold_params=($(get_threshold_params "$folder_path" "$wavelength"))
    local pfa="${threshold_params[0]}"
    local sigma="${threshold_params[1]}"
    local fraction_true="${threshold_params[2]}"
    local param_wavelength="${threshold_params[3]}"
    local use_variance_aware="${threshold_params[4]:-true}"  # Default to true if not specified

    # Use parameter wavelength if available, fallback to passed wavelength
    if [ "$param_wavelength" != "$wavelength" ] && [ -n "$param_wavelength" ]; then
        wavelength="$param_wavelength"
        log_message "Using wavelength from threshold parameters: $wavelength (overriding default $3)"
    fi

    increment_counter "total"
    local current_total=$(get_counter "total")

    # Check memory pressure before processing
    if ! check_memory_pressure; then
        wait_for_memory_relief
    fi

    # Console progress update
    echo -n "[$current_total] Processing: $folder_name... "

    # Detailed logging
    {
        echo
        echo "========================================"
        echo "Processing Folder: $folder_path"
        echo "Scratch Location: $scratch_folder"
        echo "Type: $folder_type"
        echo "Wavelength: $wavelength"
        echo "PFA: $pfa"
        echo "Sigma: $sigma"
        echo "Fraction True: $fraction_true"
        echo "Variance-aware demosaicing: $use_variance_aware"
        echo "Started: $(date)"
        echo "========================================"
    } >> "$LOG_FILE"

    # Check if Python script exists
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        echo "ERROR - Script not found"
        log_message "ERROR: Python script not found: $PYTHON_SCRIPT"
        increment_counter "error"
        return 1
    fi

    # Check if original folder exists
    if [ ! -d "$folder_path" ]; then
        echo "ERROR - Folder not found"
        log_message "ERROR: Original folder not found: $folder_path"
        increment_counter "error"
        return 1
    fi

    # Ensure scratch base directory exists
    if [ ! -d "/scratch2/jsb92" ]; then
        log_message "Creating scratch base directory: /scratch2/jsb92"
        if ! mkdir -p "/scratch2/jsb92" 2>> "$LOG_FILE"; then
            echo "ERROR - Cannot create scratch base"
            log_message "ERROR: Cannot create scratch base directory"
            increment_counter "error"
            return 1
        fi
    fi

    # Clean up any existing scratch folder with same name
    if [ -d "$scratch_folder" ]; then
        log_message "Removing existing scratch folder: $scratch_folder"
        safe_cleanup "$scratch_folder"
    fi

    # Step 1: Copy folder to scratch with retry logic and memory monitoring
    log_message "STEP 1: Copying folder to scratch"

    # Check memory before large copy operation
    if ! check_memory_pressure; then
        wait_for_memory_relief
    fi

    if ! selective_copy "$folder_path" "/scratch2/jsb92"; then
        echo "ERROR - Copy to scratch failed"
        log_message "ERROR: Failed to selectively copy folder to scratch after all retry attempts"
        increment_counter "error"
        return 1
    fi

    # Verify scratch folder exists and has content
    if [ ! -d "$scratch_folder" ]; then
        echo "ERROR - Scratch folder missing"
        log_message "ERROR: Scratch folder does not exist after copy: $scratch_folder"
        increment_counter "error"
        return 1
    fi

    local file_count=$(find "$scratch_folder" -type f | wc -l)
    log_message "Scratch folder created with $file_count files"

    # Force aggressive memory cleanup before analysis
    log_message "Forcing memory cleanup before analysis"
    sync

    # Step 2: Run analysis on scratch folder with memory monitoring
    log_message "STEP 2: Running analysis on scratch folder"
    local analysis_success=false

    # Check memory before intensive analysis
    if ! check_memory_pressure; then
        wait_for_memory_relief
    fi

    # Run Python analysis with explicit garbage collection and memory monitoring
    # Don't use nocache for analysis - we want TIFF files cached for faster processing
    export PYTHONHASHSEED=0
    export MALLOC_TRIM_THRESHOLD_=65536

    # Use variance-aware demosaicing setting from threshold parameters file
    if python3 "$PYTHON_SCRIPT" "$folder_type" "$scratch_folder" "$folder_path" "$wavelength" "$pfa" "$sigma" "$fraction_true" "$use_variance_aware" >> "$LOG_FILE" 2>&1; then
        log_message "SUCCESS: Analysis completed on scratch folder"
        analysis_success=true
        # Force immediate garbage collection after successful analysis
        sync
    else
        log_message "ERROR: Analysis failed on scratch folder"
        analysis_success=false
    fi

    # Force memory cleanup after analysis regardless of success/failure
    log_message "Forcing post-analysis memory cleanup"
    sync

    # Step 3: Analysis generates new .h5 files directly in original location (no copying back needed)
    if [ "$analysis_success" = true ]; then
        log_message "STEP 3: Analysis completed - new .h5 files generated in original location"

        # Force memory cleanup after successful analysis
        sync

        # Verify new results exist in original location
        local h5_count=$(find "$folder_path" -name "*.h5" -type f | wc -l)
        if [ $h5_count -gt 0 ]; then
            echo "✅ SUCCESS ($h5_count .h5 files generated)"
            log_message "SUCCESS: Processing completed, $h5_count new .h5 files generated"
            increment_counter "success"
        else
            echo "⚠️  PARTIAL SUCCESS (no .h5 files generated)"
            log_message "WARNING: Analysis completed but no .h5 files generated"
            increment_counter "success"  # Still count as success since analysis ran
        fi
    else
        echo "❌ ERROR"
        log_message "ERROR: Analysis failed"
        increment_counter "error"
    fi

    # Step 4: Cleanup scratch folder using nocache to avoid caching deleted files
    log_message "STEP 4: Cleaning up scratch folder"
    # Export the log function and variables needed by safe_cleanup
    if [ -n "$NOCACHE_CMD" ]; then
        $NOCACHE_CMD bash -c "
        LOG_FILE='$LOG_FILE'
        log_message() { echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$1\" >> \"\$LOG_FILE\"; }
        $(declare -f safe_cleanup)
        safe_cleanup '$scratch_folder'
        " || safe_cleanup "$scratch_folder"
    else
        safe_cleanup "$scratch_folder"
    fi

    # Force final memory cleanup after each folder to prevent accumulation
    sync

    log_message "Folder processing complete: $folder_path"

    # Return based on analysis success
    if [ "$analysis_success" = true ]; then
        return 0
    else
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

    # Find directories that should be processed - either leaf directories OR directories with .tif files
    local processing_dirs=()

    # First, check if the base directory itself has .tif files
    base_has_tif_files=$(find "$base_dir" -maxdepth 1 \( -name "*.tif" -o -name "*.tiff" \) -type f | head -1)

    if [ -n "$base_has_tif_files" ]; then
        log_message "Found .tif files in base directory: $base_dir"
        processing_dirs+=("$base_dir")
    fi

    # Then scan all subdirectories
    while IFS= read -r -d '' folder; do
        # Skip the base directory itself (already checked above)
        if [ "$folder" = "$base_dir" ]; then
            continue
        fi

        # Check if this directory contains any subdirectories
        has_subdirs=$(find "$folder" -maxdepth 1 -type d ! -path "$folder" | head -1)

        # Check if this directory contains .tif or .tiff files
        has_tif_files=$(find "$folder" -maxdepth 1 \( -name "*.tif" -o -name "*.tiff" \) -type f | head -1)

        # Process directory if:
        # 1. It's a leaf directory (no subdirectories), OR
        # 2. It has .tif/.tiff files in it
        if [ -z "$has_subdirs" ] || [ -n "$has_tif_files" ]; then
            if [ -z "$has_subdirs" ]; then
                log_message "Found leaf directory: $folder"
            else
                log_message "Found directory with .tif files: $folder"
            fi
            processing_dirs+=("$folder")
        fi
    done < <(find "$base_dir" -type d -print0)

    # Process directories
    if [ ${#processing_dirs[@]} -gt 0 ]; then
        log_message "Found ${#processing_dirs[@]} directories to process in $base_dir"

        # Process directories
        for folder in "${processing_dirs[@]}"; do
            process_folder "$folder_type" "$folder" "$wavelength"
        done
    else
        log_message "No processable directories found in $base_dir"
    fi
}

# Initialize system for swap minimization
optimize_system_memory

# Check threshold parameters file after all functions are defined
check_threshold_params

console_message "Starting Nile Red analysis folder processing..."

# Process Nile Red folders hierarchically - 700nm default wavelength (near-infrared region typical for Nile Red)
console_message "Processing Nile Red folders (${#NILE_RED_FOLDERS[@]} base directories)..."
for base_dir in "${NILE_RED_FOLDERS[@]}"; do
    process_hierarchical "$base_dir" "imaging" "0.700"
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
    echo "NILE RED ANALYSIS COMPLETE - $(date)"
    echo "============================================================"
    echo "Total folders processed: $TOTAL_FOLDERS"
    echo "Successful: $SUCCESS_COUNT"
    echo "Errors: $ERROR_COUNT"
    echo "Skipped: $SKIP_COUNT"
    echo "Detailed log saved to: $LOG_FILE"
    echo "============================================================"
} | tee -a "$LOG_FILE"

# Final memory status
FINAL_MEMORY_STATS=($(get_memory_stats))
FINAL_SWAP_MB=${FINAL_MEMORY_STATS[0]}
FINAL_RAM_PCT=${FINAL_MEMORY_STATS[2]}

# Detailed final log entry
{
    echo
    echo "==================== FINAL SUMMARY ===================="
    echo "Analysis completed: $(date)"
    echo "Total processing time: $SECONDS seconds"
    echo "Success rate: $(( SUCCESS_COUNT * 100 / (TOTAL_FOLDERS == 0 ? 1 : TOTAL_FOLDERS) ))%"
    echo "Final memory usage: ${FINAL_RAM_PCT}% RAM, ${FINAL_SWAP_MB}MB swap"
    echo "Maximum swap usage: ${MAX_SWAP_USAGE_MB}MB (threshold)"
    echo "Maximum RAM usage: ${MAX_MEMORY_USAGE_PCT}% (threshold)"
    echo "======================================================="
} >> "$LOG_FILE"

# Cleanup
rm -rf "$COUNTER_DIR"

if [ "$ERROR_COUNT" -eq 0 ] && [ "$TOTAL_FOLDERS" -gt 0 ]; then
    console_message "🎉 All Nile Red folders processed successfully!"
    exit 0
elif [ "$TOTAL_FOLDERS" -eq 0 ]; then
    console_message "⚠️  No Nile Red folders found to process. Check directory paths."
    exit 1
else
    console_message "⚠️  $ERROR_COUNT/$TOTAL_FOLDERS folders had errors. Check the log for details."
    exit 1
fi