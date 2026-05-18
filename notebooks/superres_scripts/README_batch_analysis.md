# pyS3M Batch Analysis Scripts

## Overview

This directory contains the final, optimized batch analysis solution for processing large numbers of SMLM folders without memory leaks or crashes.

## Scripts

### `batch_analysis.sh` ⭐ **RECOMMENDED**
Main batch processing script that discovers and processes all folders using complete process isolation.

**Features:**
- ✅ **Complete process isolation** - each folder gets fresh Python interpreter
- ✅ **Memory leak prevention** - no accumulation between folders  
- ✅ **Crash resilience** - one folder failure doesn't affect others
- ✅ **Clean logging** - detailed logs with progress tracking
- ✅ **Exact folder paths** - uses paths from MemorySafe script

**Usage:**
```bash
cd /path/to/superres_notebooks
./batch_analysis.sh
```

### `single_folder_analysis.py`
Individual folder processor called by the bash script. Processes one folder and exits completely.

**Features:**
- Validates folder contents (.tif files, metadata)  
- Cleans existing .h5 files before processing
- Loads camera parameters and configures analysis
- Calls SR_Functions for SM or imaging data processing
- Complete resource cleanup on exit

## Folder Processing Logic

The script processes folders in this order:

1. **SM Data Hierarchies** (13 base directories) - wavelength 0.638nm
   - Walks directory trees to find leaf folders with data files
   
2. **HeLa Imaging Folders** (4 direct folders) - wavelength 0.647nm  
   - Direct folder processing for HeLa cell data
   
3. **General Imaging Folders** (14 direct folders) - wavelength 0.55nm
   - Origami, DNA ruler, and other imaging experiments
   
4. **Hierarchical Imaging** (3 base directories) - wavelength 0.55nm
   - Complex directory structures requiring tree walking

## Output

### Console Output
Clean, concise progress updates:
```
[1] Processing: TetraspeckCalibration... ✅ SUCCESS
[2] Processing: ATTO488_50PM_PCA_PCD... ❌ ERROR  
[3] Processing: Cell3_HILO_190mW_638... ✅ SUCCESS
```

### Log File
Detailed timestamped log saved to `batch_analysis_YYYYMMDD_HHMMSS.log`:
- Full folder paths and processing details
- Python script output and error messages  
- Processing time and success rates
- Final summary with statistics

## Architecture Benefits

### vs. Single Large Python Script:
- ❌ **Old way**: Memory accumulates, eventual crashes, hard to debug
- ✅ **New way**: Fresh process per folder, isolated failures, easy debugging

### vs. Memory-Safe Python Script:  
- ❌ **Old way**: Complex resource management, still potential for leaks
- ✅ **New way**: OS handles cleanup automatically, impossible to have leaks

### Process Isolation Details:
1. Bash discovers folder → Calls `python3 single_folder_analysis.py`
2. Python starts fresh → Imports modules → Processes folder → Exits completely  
3. OS cleans up ALL resources → Bash continues to next folder
4. Repeat with completely clean slate

## Legacy Scripts

### `All_Analysis_OneBook_Direct_DEPRECATED.py`
❌ **DEPRECATED** - Original attempt at direct processing. Superseded by bash+python solution.

### `All_Analysis_OneBook_MemorySafe.py`  
✅ **Still functional** - Memory-safe single Python script with file copying to scratch directories. Use only if bash+python approach has issues.

## Troubleshooting

### No folders found
- Check that `/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/` paths exist
- Run on analysis PC where data is stored, not local development machine

### Python import errors
- Ensure virtual environment is activated: `workon pyS3M`
- Check that all required packages are installed: `pip install -r requirements.txt`

### Permission errors
- Ensure bash script is executable: `chmod +x batch_analysis.sh`
- Check read/write permissions on data directories

### Individual folder failures
- Check log file for detailed Python error messages
- Test individual folder with: `python3 single_folder_analysis.py imaging "/path/to/folder" 0.55`

## Performance

Expected processing time varies by folder size:
- Small folders (~100 images): 2-5 minutes  
- Large folders (~10,000 images): 30-60 minutes
- Total batch: Several hours depending on data volume

Memory usage per folder peaks around 8-16GB during processing, then drops to zero between folders.