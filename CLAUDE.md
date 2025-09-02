# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

pyBayerSMLM is a Python package for multicolour single-molecule localization microscopy (SMLM) analysis, specifically designed for Bayer filter-equipped cameras. The package simulates camera images, performs demosaicing, and analyzes fluorescent dye molecules for super-resolution microscopy applications.

## Installation and Dependencies

**Virtual Environment Setup:**
The project uses a dedicated Python virtual environment located at `/home/jbeckwith/.virtualenvs/pyBayerSMLM/`. 

To activate the virtual environment:
```bash
source ~/.bashrc                          # Load virtualenvwrapper if needed
workon pyBayerSMLM                       # Activate the project's virtual environment

# Alternative direct activation:
source /home/jbeckwith/.virtualenvs/pyBayerSMLM/bin/activate
```

**For Claude Code:** **CRITICAL - ALWAYS activate the virtual environment before running ANY Python commands:**
```bash
# Method 1: Activate environment first (REQUIRED for interactive scripts)
source /home/jbeckwith/.virtualenvs/pyBayerSMLM/bin/activate
python script.py

# Method 2: Direct python path (preferred for non-interactive scripts)
/home/jbeckwith/.virtualenvs/pyBayerSMLM/bin/python script.py
```

**IMPORTANT:** The virtual environment must be activated for:
- Interactive scripts requiring tkinter/matplotlib GUI
- Scripts importing pyBayerSMLM modules
- Any Python command execution in this project
- Installing additional packages with pip

Always run `source /home/jbeckwith/.virtualenvs/pyBayerSMLM/bin/activate` before Python operations.

Install dependencies using:
```bash
pip install -r requirements.txt
```

The package requires Python 3.10+ and includes dependencies for:
- Scientific computing (numpy, scipy, pandas, polars)
- Image processing (scikit-image, opencv-python, colour-demosaicing, tifffile)
- Machine learning (scikit-learn, numba for performance)
- Visualization (matplotlib, seaborn, plotly)
- Microscopy analysis (fpbase for fluorophore data)
- Jupyter notebooks for interactive analysis

## Core Architecture

### Main Source Modules (`src/`)

The codebase follows a modular architecture with specialized function classes:

**Core Analysis Functions:**
- `Multicolour_Simulation_Functions.py` - Main simulation class (`MultiC_Sim_Funcs`) for camera image generation and fitting analysis
  - **REFACTORED**: `Multicolour_Simulation_Functions_refactor.py` - Improved version with strategy pattern, consolidated duplicate methods, ~40% code reduction
- `ImageAnalysisFunctions.py` - Image fitting and analysis utilities
  - **REFACTORED**: Complete strategy pattern implementation, 19 duplicate functions → 2 unified methods, ~60% code reduction
- `SpectralFunctions.py` - Fluorophore spectra and filter calculations
  - **REFACTORED**: Improved with strategy pattern for dye/filter handling, database query optimization, comprehensive type hints
- `PSFFunctions.py` - Point spread function modeling and photon simulation
- `sCMOSFunctions.py` - Camera calibration and Bayer pattern demosaicing

**Utilities:**
- `IOFunctions.py` - File I/O operations for microscopy data
  - **REFACTORED (August 28, 2025)**: Memory-efficient image processing workflow with 4 new functions for ROI-based photoelectron conversion, reducing peak memory from 4x to 1x file size
  - **HDF5 COMPATIBILITY FIX (August 28, 2025)**: Frame columns automatically converted to int32 to handle large frame numbers and prevent dtype mismatch errors in batch processing
- `DriftCorrectionFunctions.py` - Drift correction methods for SMLM data
  - **ENHANCED (August 28, 2025)**: Automatic fiducial detection integrated with configurable parameters (threshold, box size, frame requirements), comprehensive testing framework, and unified workflow via `undrift_with_fiducial_detection()` convenience function
- `CalibrationFunctions.py` - Camera calibration routines
- `PlottingFunctions.py` - Visualization functions
  - **REFACTORED**: Google-style docstrings, DRY principles, constants organization
- `SpotDetectionFunctions.py` - Fluorescent spot identification
- `MaskFunctions.py` - Spatial masking operations
- `SR_Functions.py` - Super-resolution analysis workflows
  - **REFACTORED (August 28, 2025)**: Updated to use memory-efficient ROI processing workflow in 3 main analysis functions

**Legacy Integration:**
- `lib.py`, `localise.py`, `postprocess.py`, `render.py` - Adapted from Picasso SMLM package
- `gaussoptfuncs.py`, `imageprocess.py` - Gaussian fitting and image processing
- `aim.py` - Additional analysis functions

### Key Class Structure

Most modules implement function classes that organize related operations:
```python
# Example pattern used throughout
class PSF_Functions:
    def gen_camera_image_stack(self, ...): pass
    def sigma_PSF(self, ...): pass
    def gen_spatial_PSF(self, ...): pass
```

### Data Flow Architecture

1. **Camera Calibration** - Load/generate gain, offset, variance maps for sCMOS cameras
2. **Spectral Analysis** - Calculate dye efficiencies across Bayer filter pixels
3. **Image Simulation** - Generate realistic camera images with noise modeling
4. **Demosaicing** - Convert Bayer-filtered images to RGB channels
5. **Localization** - Fit Gaussian models to identify molecule positions and colors
6. **Analysis** - Statistical analysis and visualization of results

## Working with Notebooks

The repository contains 39 curated Jupyter notebooks organized by application:

- `notebooks/` - General analysis and testing notebooks (18 notebooks)
- `superres_notebooks/` - Super-resolution microscopy analysis (11 notebooks)  
- `figure_notebooks/` - Publication figure generation (5 notebooks)
- `FRET_notebooks/` - FRET analysis workflows (2 notebooks)
- `single_dye_experiment_notebooks/` - Single molecule experiments (1 notebook)
- `Lanthanide_nanoparticles_notebooks/` - Specialized nanoparticle analysis (1 notebook)
- `demosaic_example/` - Demosaicing examples (1 notebook)

**✅ Recent Cleanup (August 21, 2025):** Streamlined from 105 files to 39 notebooks by removing:
- 55 `.ipynb_checkpoints` directories
- 9 redundant testing variants  
- 2 conflicted/duplicate files

Start with `notebooks/Basic_Camera_Example.ipynb` to understand the core workflow.

## Key Development Patterns

**Spelling Convention:** **ALWAYS use British English spelling** throughout the codebase for consistency and professionalism:
- "optimise" (not "optimize"), "normalise" (not "normalize"), "centre" (not "center"), "colour" (not "color"), "analyse" (not "analyze")
- External library compatibility preserved (matplotlib.colors, scipy.optimize retain American spelling)
- This standard applies to all code, comments, docstrings, and documentation

**Function Organization:** Each module typically implements a class with related methods rather than standalone functions.
- **Recent Improvements**: Anti-pattern empty `__init__` methods being replaced with proper initialisation
- **Strategy Pattern**: Used in refactored modules for handling different data types and processing methods

**Data Handling:** Heavy use of numpy/pandas for numerical data, with polars for performance-critical operations.
- **Database Integration**: DuckDB for spectral data with optimised query patterns
- **Type Safety**: Comprehensive type hints being added across refactored modules

**Parallelisation:** Uses multiprocessing for image analysis tasks (see `ImageAnalysisFunctions.py`).
- **CRITICAL**: ProcessPoolExecutor instances must use context managers to prevent resource leaks
- **Memory leak patterns identified**: ProcessPoolExecutor, ThreadPoolExecutor, and matplotlib figure cleanup issues

**File I/O:** Custom formats for microscopy data, with TIFF support for images and CSV/HDF5 for tabular data.
- **Error Handling**: Improved with specific exceptions and input validation

## Camera Calibration

Camera parameters are stored as dictionaries containing:
```python
camera_parameters = {
    'gain': np.ndarray,      # Pixel gain map
    'offset': np.ndarray,    # Pixel offset map  
    'variance': np.ndarray,  # Pixel variance map
    'readnoise': float,      # Read noise level
    'rqe': np.ndarray,      # Relative quantum efficiency
    'masks': dict,          # Bayer filter masks {'B': ..., 'G': ..., 'R': ...}
    'pixel_QYs': np.ndarray, # Quantum yields vs wavelength
    'pixel_order': list,    # Color channel order
    'pixel_order_indices': dict # Channel index mapping
}
```

Camera calibrations are stored in `Camera_Calibrations/` directory.

## Threshold Parameter Optimization Workflow

The codebase provides a complete system for optimizing spot detection parameters across datasets:

### **Interactive Parameter Tuning**
```bash
# Activate virtual environment (required)
source /home/jbeckwith/.virtualenvs/pyBayerSMLM/bin/activate

# Run interactive threshold tuner
python superres_notebooks/interactive_threshold_tuner.py
```

**Features:**
- Real-time spot detection preview with parameter adjustment
- Automatic dataset discovery matching batch analysis folder lists
- GUI interface with tkinter or graceful fallback to file-based mode
- Generates `threshold_parameters.txt` with format: `folder_path|pfa|perc_threshold|wavelength`

### **Batch Analysis Integration**
```bash
# Run batch analysis with optimized parameters
bash superres_notebooks/batch_analysis.sh
```

**Workflow:**
1. `batch_analysis.sh` validates `threshold_parameters.txt` exists
2. For each folder, custom parameters are loaded or defaults applied
3. `single_folder_analysis.py` receives 5 arguments: `type folder wavelength pfa perc_threshold`
4. `SR_Functions.py` applies parameters to `fit_SM_data()` and `fit_imaging_data()`
5. All parameters logged for full traceability

**Parameter Flow:**
- **Interactive Tuner** → `threshold_parameters.txt` → **Batch Analysis** → **SR Functions** → **Spot Detection**
- Graceful fallback to defaults (pfa=1e-4, perc_threshold=98) for unlisted folders
- Parameter validation with clear error messages and instructions

## Testing and Development

No formal test suite exists - testing is primarily done through Jupyter notebooks. When implementing new features, create corresponding test notebooks following existing patterns in `notebooks/`.

**Python Test Scripts:** All Python test scripts (not unit tests) should be placed in the `claude/` directory. This includes performance tests, validation scripts, and standalone testing utilities.

**Drift Correction Testing Framework (August 28, 2025):** Comprehensive testing system implemented in `unit_tests/test_drift_correction.py` covering:
- RCC, AIM, and fiducial drift correction methods
- Automatic fiducial detection with configurable parameters
- Parameter validation and error handling
- Backward compatibility functions
- HDF5 compatibility testing in `test_hdf5_fix.py` for large frame number handling

**Analysis Scripts:** Large-scale analysis scripts are located in `superres_notebooks/`. Available batch processing scripts:
- `batch_analysis.sh` + `single_folder_analysis.py` - **RECOMMENDED**: Modern bash+python batch processing system with:
  - Complete threshold parameter integration from `interactive_threshold_tuner.py`
  - Memory-efficient scratch disk workflow with automatic cleanup
  - Per-folder parameter customization with fallback to defaults
  - Comprehensive logging and error handling with resource monitoring
- `interactive_threshold_tuner.py` - **NEW**: Interactive parameter optimization tool:
  - GUI-based threshold tuning with real-time spot detection preview
  - Automatic folder discovery matching batch analysis datasets
  - Generates `threshold_parameters.txt` for batch processing integration
  - Graceful fallback to file-based mode for headless environments
- `All_Analysis_OneBook_MemorySafe.py` - **LEGACY**: Sequential processing with comprehensive resource cleanup
- `All_Analysis_OneBook_Debug.py` - **DIAGNOSTIC**: Maximum logging version for troubleshooting
- `All_Analysis_OneBook.py` - **AVOID**: Original script causes terminal exits due to memory issues

**Project TODO:** See `claude/TODO.md` for comprehensive analysis of completed refactoring work and remaining high-priority tasks.

For simulation testing, use the pattern:
1. Generate synthetic data with known parameters
2. Run analysis pipeline
3. Compare results to ground truth
4. Validate statistical performance across parameter ranges

## Data Formats

- **Images:** TIFF format (multi-frame supported)
- **Localizations:** CSV with standard SMLM columns (xc, yc, photons, etc.)
- **HDF5 Tables:** Localization data with automatic dtype compatibility (frame columns converted to int32 for large frame numbers)
- **Spectra:** CSV files in `Spectra/` directory
- **Results:** CSV for tabular data, NPY for arrays

## Performance Considerations

- Use `numba.jit` decorators for performance-critical loops
- Leverage `multiprocessing` for parallel image analysis
- Consider memory usage when processing large image stacks
- Use `polars` instead of `pandas` for large datasets where available

## Memory Management (Critical)

**Memory Leak Prevention:**
- **ProcessPoolExecutor**: Always use context managers (`with ProcessPoolExecutor() as executor:`)
- **ThreadPoolExecutor**: Use context managers instead of manual `.shutdown()` calls
- **Matplotlib**: Set backend to 'Agg' for batch processing, always call `plt.close()` after figures
- **Large arrays**: Explicitly delete temporary arrays and call `gc.collect()` in intensive loops
- **Resource monitoring**: Use `psutil` to monitor memory usage during large-scale processing

**Identified Memory Leak Locations:**
- `ImageAnalysisFunctions.py:1155` - ProcessPoolExecutor without context manager
- `SpotDetectionFunctions.py:137` - ProcessPoolExecutor without context manager  
- `aim.py:200` - ThreadPoolExecutor with vulnerable manual cleanup
- `imageprocess.py:108` - Matplotlib figures without cleanup
- `postprocess.py:1151` - Matplotlib figures without cleanup

## Progress Bar Integration

The codebase uses a unified progress bar system through `ProgressUtils.py` providing:
- **Clean context managers**: `clean_progress_bar()` for guaranteed cleanup
- **Specialized functions**: `fitting_progress_bar()`, `analysis_progress_bar()`, etc.
- **Jupyter notebook support**: Automatic detection of notebook environment
- **Global control**: Environment variable `PYBAYERSMLM_NO_PROGRESS=1` to disable
- **Error handling**: Graceful degradation and proper exception handling

**Usage Pattern:**
```python
import ProgressUtils

# For iteration with automatic cleanup
with ProgressUtils.clean_progress_bar(range(n), desc="Processing") as pbar:
    for item in pbar:
        # Process item
        pass

# For manual updates
with ProgressUtils.fitting_progress_bar(total=n) as pbar:
    for i in range(n):
        # Process item
        pbar.update(1)
```

## Recent Major Refactoring (August 15, 2025)

### **Latest Updates (August 28, 2025)**

**Memory-Efficient Image Processing Refactor [COMPLETED - August 28, 2025]:**
- ✅ **IOFunctions.py enhancement**: Added 4 new functions for memory-efficient ROI processing
  - `convert_to_photoelectrons()` - Raw ADU to photoelectron conversion
  - `apply_smoothing()` - Data smoothing operations
  - `generate_weights()` - Weights map generation for fitting
  - `process_roi_to_photoelectrons()` - **Core unified ROI processing pipeline**
- ✅ **SR_Functions.py workflow update**: Updated 3 main analysis functions to use ROI-based processing
  - `example_spots_singleframe()` - Single frame analysis with on-demand plotting data
  - `fit_SM_data()` - Multi-frame batch analysis 
  - `fit_imaging_data()` - Cross-file analysis
- ✅ **Memory optimization achieved**: Peak memory usage reduced from 4x file size to 1x file size
- ✅ **Processing efficiency**: Only detected ROIs converted to photoelectrons/smoothed/weights
- ✅ **HDF5 append bug fix**: Corrected `dropna(axis=1, how='all')` → `dropna(axis=0, how='all')` to preserve column structure
- ✅ **Batch analysis restored**: Added complete SM dataset list (13 datasets) including Tetraspeck calibration and biotinylated dyes

**Previous Updates (August 27, 2025)**

**Memory Leak Analysis and Diagnosis [COMPLETED - August 27, 2025]:**
- ✅ **Comprehensive codebase analysis**: Identified critical memory leak patterns causing terminal crashes
- ✅ **ProcessPoolExecutor leaks found**: 2 instances without context managers in core analysis functions
- ✅ **ThreadPoolExecutor leaks found**: 1 instance with vulnerable manual cleanup in drift correction
- ✅ **Matplotlib memory issues**: Multiple figures created without explicit cleanup in batch processing
- ✅ **Diagnostic scripts created**: Step-by-step testing, debug logging, and memory-safe versions
- ✅ **Memory-safe rewrite**: `All_Analysis_OneBook_MemorySafe.py` addresses all identified leak patterns
- ✅ **Resource monitoring**: Added memory usage tracking and automatic cleanup systems

**All_Analysis_OneBook Script Evolution:**
- ✅ **Original issue identified**: ProcessPoolExecutor and matplotlib memory leaks causing terminal crashes  
- ✅ **Memory-safe version created**: Eliminates multiprocessing leaks, uses sequential processing with cleanup
- ✅ **Debug version created**: Maximum logging, PID tracking, system monitoring for crash diagnosis
- ✅ **Diagnostic testing**: Step-by-step crash source identification system

**Previous Updates (August 26, 2025):**
- ✅ **IOFunctions TIFF optimizations**: 58-62% speed improvement with memory mapping
- ✅ **Progress bar integration fixes**: Resolved `'_GeneratorContextManager' object is not iterable` errors
- ✅ **DriftCorrectionFunctions.py**: Corrected context manager usage in RCC and AIM methods
- ✅ **aim.py**: Fixed both 2D and 3D drift correction progress bar integration

### **Completed Code Improvements**

**PlottingFunctions.py [COMPLETED]:**
- ✅ **Google-style docstrings**: Complete documentation for all methods
- ✅ **Streamlined code**: Removed unused imports and parameters  
- ✅ **Fixed width/height handling**: Proper aspect ratio support in `one_column_plot()`
- ✅ **DRY principles**: Extracted font size logic, grid setup, and rcParams configuration
- ✅ **Type hints**: Complete type annotations throughout
- ✅ **Constants organization**: Magic numbers moved to `PlotConstants` class

**SpectralFunctions.py [COMPLETED]:**
- ✅ **Strategy pattern**: `SpectrumProcessor` ABC with `DyeSpectrumProcessor`/`FilterSpectrumProcessor`
- ✅ **Database optimization**: `DatabaseQueryHandler` class with safe parameterized queries
- ✅ **Type safety**: `SpectralDataType` enum replacing boolean conditionals
- ✅ **Code consolidation**: 81-line duplicate method eliminated through clean architecture
- ✅ **Performance**: Memory-optimized queries and resource management
- ✅ **Documentation**: Comprehensive Google-style docstrings with scientific references
- ✅ **Backward compatibility**: Legacy methods preserved through compatibility layer

**ImageAnalysisFunctions.py [COMPLETED]:**
- ✅ **Strategy pattern**: Complete implementation with `FittingStrategy` enum and 5 processor classes
- ✅ **Code elimination**: 19 duplicate fitting functions → 2 unified methods (`fit_puncta_method`, `fit_puncta_parallel_method`)
- ✅ **Parameter validation**: Dataclass-based `FittingParameters` with comprehensive type checking
- ✅ **Performance optimization**: Vectorized operations, memory pooling, parallel processing improvements
- ✅ **Statistical accuracy**: Correct chi-squared calculations integrated from original implementation
- ✅ **API modernization**: Clean, consistent interface with proper error handling
- ✅ **Backwards compatibility removal**: 139-line compatibility layer eliminated after updating all calling code
- ✅ **Documentation**: Comprehensive Google-style docstrings with mathematical formulations
- ✅ **Type safety**: Complete type annotations throughout all methods and classes

### **Architecture Patterns Established**
These patterns are now proven and ready for application to remaining modules:
- **Strategy Pattern**: For handling method variations and data type processing
- **Handler Classes**: For resource management (database connections, file I/O)  
- **ABC Patterns**: For extensible processor architectures
- **Constants Classes**: For organizing magic numbers and configuration
- **Dataclass Validation**: For parameter validation and type safety
- **Enum-based Type Safety**: Replacing boolean conditionals and string comparisons
- **Memory Optimization**: Object pooling and vectorized operations for performance
- **Clean API Design**: Unified interfaces with proper error handling and documentation
- **Memory-Safe Patterns**: Context managers for all resource-intensive operations (ProcessPoolExecutor, file I/O, figure handling)

### **Next Priority Targets**
1. **Memory leak fixes (HIGH PRIORITY)**: 
   - Fix ProcessPoolExecutor leaks in `ImageAnalysisFunctions.py:1155` and `SpotDetectionFunctions.py:137`
   - Fix ThreadPoolExecutor leak in `aim.py:200`  
   - Add matplotlib cleanup in `imageprocess.py:108` and `postprocess.py:1151`
2. **Legacy cleanup**: Remove SpectralFunctions_Old.py and PlottingFunctions_Old.py (1,925 lines total)
3. **postprocess.py**: Highest complexity (239), large function decomposition needed  
4. **Final error handling**: Fix remaining 4 bare `except:` clauses in CalibrationFunctions.py, IOFunctions.py, and lib.py

### **Completed Achievements (August 15, 2025)**
- ✅ **Code Reduction**: 19 duplicate functions eliminated from ImageAnalysisFunctions.py
- ✅ **Anti-pattern Fixes**: All 9 empty `__init__` methods replaced with proper docstrings
- ✅ **API Integration**: All calling code updated in SR_Functions.py and Multicolour_Simulation_Functions.py
- ✅ **Backwards Compatibility**: Clean removal of 139-line compatibility layer
- ✅ **Database Optimization**: Fixed DuckDB parameter binding issues in SpectralFunctions.py
- ✅ **Strategy Pattern**: Successfully applied across SpectralFunctions.py, PlottingFunctions.py, and ImageAnalysisFunctions.py

### **API Integration Issues Resolved (August 18, 2025)**

**Issue:** After completing the ImageAnalysisFunctions.py refactoring, integration testing revealed API compatibility problems with calling modules using incorrect parameter passing patterns.

**Root Cause:** Calling modules (SR_Functions.py, Multicolour_Simulation_Functions.py) were passing `masks_tofit` as positional argument instead of using the required `masks=` keyword argument from the new unified API.

**Resolution Applied:**
- ✅ **SR_Functions.py**: Updated all 3 calls to `fit_puncta_parallel_method()` to use `masks=masks_tofit`
- ✅ **Multicolour_Simulation_Functions.py**: Confirmed all 3 calls already use correct `masks=masks_tofit` format
- ✅ **API Consistency**: Both modules now use identical, correct interface patterns
- ✅ **Parameter Flow Fixed**: Background photon distribution corrected from 40 per channel → 40 total
- ✅ **Chi-squared Calculations**: Verified correct calculation and storage in fitting results
- ✅ **Parameter Squaring**: Fixed double-squaring issues, parameters now squared once after error calculation

**Technical Details:**
```python
# OLD (incorrect) - positional masks argument
fit_results, fit_errors = I_AF.fit_puncta_parallel_method(
    puncta_tofit, smoothed_puncta_tofit, weights_tofit, 
    relative_coords, planes, FittingStrategy.STANDARD, 
    masks_tofit  # ERROR: masks as positional argument
)

# NEW (correct) - keyword masks argument  
fit_results, fit_errors = I_AF.fit_puncta_parallel_method(
    puncta_tofit, smoothed_puncta_tofit, weights_tofit,
    relative_coords, planes, FittingStrategy.STANDARD,
    masks=masks_tofit  # CORRECT: masks as keyword argument
)
```

**Result:** ImageAnalysisFunctions.py refactored interface is now fully integrated across the codebase with consistent API usage and proper parameter flow. All fitting operations now use the modernized strategy pattern interface correctly.

## Analysis Script Modernization [COMPLETED - August 27, 2025]

### **All_Analysis_OneBook Script Evolution**

The original `All_Analysis_OneBook.py` script had critical memory management and error handling issues causing terminal exits. Two improved versions are now available:

#### **All_Analysis_OneBook_Optimized.py** ⭐ **RECOMMENDED**
**Created:** August 27, 2025  
**Features:**
- **28-core parallelization**: Uses all available cores for file operations
- **Memory management**: `gc.collect()` after each folder processing prevents memory exhaustion
- **Clean terminal output**: Single-line status updates with carriage return, no terminal spam
- **Proper logging**: Timestamped `.txt` log files with immediate flush statements
- **Essential recursion preserved**: Server communication retry loops maintained for network reliability
- **Compact organization**: Consolidated, well-structured code eliminating repetition

#### **All_Analysis_OneBook_Fixed.py**
**Features:**
- **Memory management**: Forced garbage collection and resource cleanup
- **Safe retry logic**: Maximum attempts with proper error boundaries (removes infinite recursion)
- **Progress tracking**: Comprehensive logging with success/failure statistics
- **Configuration-driven**: Type-safe dataclass approach

#### **Usage:**
```bash
# Recommended (optimized with 28-core parallelization)
python superres_notebooks/All_Analysis_OneBook_Optimized.py  # ⭐ Best performance

# Alternative (memory-safe but no parallelization)  
python superres_notebooks/All_Analysis_OneBook_Fixed.py  # ✅ Safe but slower

# Avoid (problematic)
python superres_notebooks/All_Analysis_OneBook.py  # ❌ Crashes terminal
```

## Interactive Threshold Tuning System [COMPLETED - September 1, 2025]

### **Batch Analysis Parameter Optimization**

The interactive threshold tuner helps determine optimal spot detection parameters (`pfa` and `perc_threshold`) for each dataset in your batch analysis workflow.

**Location:** `superres_notebooks/interactive_threshold_tuner.py`

**Key Features:**
- **Automatic folder discovery**: Uses exact same folder lists as `batch_analysis.sh`
- **Interactive/file-based display**: Auto-detects tkinter availability, graceful fallback
- **Professional visualization**: Integrates `PlottingFunctions.Plotter.image_scatter_plot()`
- **Parameter tuning**: Real-time adjustment of PFA (probability of false alarm) and percentile thresholds
- **Batch integration**: Outputs `threshold_parameters.txt` for direct use by `batch_analysis.sh`

#### **Usage Workflow:**

```bash
# CRITICAL: Always activate virtual environment first
source /home/jbeckwith/.virtualenvs/pyBayerSMLM/bin/activate

# Run interactive threshold tuner before batch analysis
python superres_notebooks/interactive_threshold_tuner.py

# Then run batch analysis with optimized parameters
./superres_notebooks/batch_analysis.sh
```

#### **Interactive Process:**
1. **Folder Discovery**: Automatically finds all folders from batch analysis workflow (SM data, HeLa, imaging, hierarchical)
2. **Frame Loading**: Loads one representative TIFF frame from each folder
3. **Parameter Testing**: Interactive menu to adjust:
   - `pfa` (probability of false alarm, default: 1e-4)
   - `perc_threshold` (percentile threshold, default: 98%)
   - `wavelength` if needed
4. **Real-time Visualization**: Side-by-side comparison of original image vs detected spots
5. **Parameter Storage**: Save optimized parameters for each folder
6. **Batch Output**: Generate `threshold_parameters.txt` for batch processing

#### **Output Format:**
```
# threshold_parameters.txt
folder_path|pfa|perc_threshold|wavelength
/path/to/TetraspeckCalibration|1e-04|98.0|0.638
/path/to/ATTO488_data|1e-05|95.0|0.638
/path/to/HeLa_Cell1|5e-05|96.5|0.647
```

#### **tkinter Dependency Handling:**
- **Interactive Mode**: Full GUI display if tkinter available
- **File Mode**: Saves detection preview images as PNG files if tkinter unavailable
- **Installation**: `sudo apt-get install python3-tk` to enable interactive mode

#### **Integration with Batch Analysis:**
The threshold tuner is designed to run **before** `batch_analysis.sh`, providing folder-specific parameters that optimize spot detection for each dataset's characteristics.

## American to British Spelling Standardisation [COMPLETED - August 22, 2025]

### **Complete Spelling Conversion**

The entire codebase has been systematically converted from American to British English spellings while preserving external API compatibility. This ensures consistent professional presentation and eliminates spelling confusion throughout the project.

#### **Key Conversions Applied:**
- "optimize" → "optimise" (and variants: optimization → optimisation)
- "normalize" → "normalise" (and variants: normalization → normalisation)  
- "center" → "centre" (and variants: centered → centred, centering → centring)
- "color" → "colour" (in internal code only)
- "analyze" → "analyse"

#### **Function Names Updated:**
- `optimize_matrix_symmetry()` → `optimise_matrix_symmetry()` (MaskFunctions.py)
- `_sum_and_center_of_mass()` → `_sum_and_centre_of_mass()` (gaussoptfuncs.py)
- Parameter names: `normalize_photons` → `normalise_photons`, `normalize` → `normalise` (IOFunctions.py)

#### **External Library Compatibility Preserved:**
- matplotlib functions (e.g., `plt.colorbar()`, `matplotlib.colors`) retain American spelling
- scipy imports (e.g., `scipy.optimize`) maintain American spelling
- All external API calls remain unchanged to ensure library compatibility

#### **Files Modified (15 total):**
Core modules updated include MaskFunctions.py, IOFunctions.py, SpotDetectionFunctions.py, Multicolour_Simulation_Functions.py, and associated test files. All function calls have been consistently updated throughout the codebase.

#### **Benefits:**
- **Consistency:** Unified British spelling convention throughout the project
- **Maintainability:** Clear spelling standards for future development
- **Professionalism:** Consistent with British English scientific publications
- **Zero Breaking Changes:** All functionality preserved with external library compatibility maintained

## Global Object Instantiation Anti-Pattern Analysis [IDENTIFIED - September 2025]

### **Anti-Pattern Identified**
The codebase contains 18 global object instantiations across 7 core modules, creating tight coupling and testing difficulties:

```python
# Current anti-pattern (problematic):
import IOFunctions
import PSFFunctions
IO = IOFunctions.IO_Functions()          # Global - tight coupling
PSF = PSFFunctions.PSF_Functions()       # Global - hard to test
```

### **Files Affected:**
- **SR_Functions.py**: 5 global objects (IO, H_F, M_F, I_AF, SD_F)
- **Multicolour_Simulation_Functions.py**: 3 global objects (IO, PSF, I_AF)
- **Toy_Model_Functions.py**: 5 global objects (IO, PSF, MSF, Mask, S_F)
- **SpotDetectionFunctions.py**: 2 global objects (PSF, sCMOS)
- **sCMOSFunctions.py, CalibrationFunctions.py**: 1 global object each (IO)

### **Proposed Solution - Dependency Injection:**
```python
# Dependency injection with sensible defaults (best practice):
import IOFunctions
import PSFFunctions

class SpotDetection_Functions:
    def __init__(self, psf_functions=None, io_functions=None):
        self.psf = psf_functions or PSFFunctions.PSF_Functions()
        self.io = io_functions or IOFunctions.IO_Functions()
    
    def some_method(self):
        return self.psf.sigma_PSF(...)  # Uses injected dependency
```

### **Benefits:**
- **Testability**: Enable mock injection for unit testing
- **Modularity**: Reduce tight coupling between modules
- **Flexibility**: Allow custom implementations of dependencies
- **Performance**: Lazy loading and instance sharing opportunities
- **Maintainability**: Explicit dependency relationships

### **Implementation Priority:**
1. **High Impact, Low Complexity**: SpotDetectionFunctions.py, sCMOSFunctions.py (2-4 hours)
2. **Medium Impact**: CalibrationFunctions.py, SM_extractionfunctions.py (4-6 hours)  
3. **High Complexity**: SR_Functions.py, Multicolour_Simulation_Functions.py (8-12 hours)

**Analysis complete** - Detailed refactoring plan available in `claude/global_instantiation_refactor_plan.py`