# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

pyBayerSMLM is a Python package for multicolour single-molecule localization microscopy (SMLM) analysis, specifically designed for Bayer filter-equipped cameras. The package simulates camera images, performs demosaicing, and analyzes fluorescent dye molecules for super-resolution microscopy applications.

## Installation and Dependencies

**Virtual Environment Setup:**
```bash
workon pyBayerSMLM  # Activate the project's virtual environment
```

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
- `CalibrationFunctions.py` - Camera calibration routines
- `PlottingFunctions.py` - Visualization functions
  - **REFACTORED**: Google-style docstrings, DRY principles, constants organization
- `SpotDetectionFunctions.py` - Fluorescent spot identification
- `MaskFunctions.py` - Spatial masking operations

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

**Function Organization:** Each module typically implements a class with related methods rather than standalone functions.
- **Recent Improvements**: Anti-pattern empty `__init__` methods being replaced with proper initialization
- **Strategy Pattern**: Used in refactored modules for handling different data types and processing methods

**Data Handling:** Heavy use of numpy/pandas for numerical data, with polars for performance-critical operations.
- **Database Integration**: DuckDB for spectral data with optimized query patterns
- **Type Safety**: Comprehensive type hints being added across refactored modules

**Parallelization:** Uses multiprocessing for image analysis tasks (see `ImageAnalysisFunctions.py`).

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

## Testing and Development

No formal test suite exists - testing is primarily done through Jupyter notebooks. When implementing new features, create corresponding test notebooks following existing patterns in `notebooks/`.

**Python Test Scripts:** All Python test scripts (not unit tests) should be placed in the `claude/` directory. This includes performance tests, validation scripts, and standalone testing utilities.

**Analysis Scripts:** Large-scale analysis scripts are located in `superres_notebooks/`. Available batch processing scripts:
- `All_Analysis_OneBook_Optimized.py` - **RECOMMENDED**: 28-core parallelization, memory cleanup, proper logging with flush statements, preserves essential server communication recursion
- `All_Analysis_OneBook_Fixed.py` - Memory-safe processing with comprehensive logging and progress tracking  
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
- **Spectra:** CSV files in `Spectra/` directory
- **Results:** CSV for tabular data, NPY for arrays

## Performance Considerations

- Use `numba.jit` decorators for performance-critical loops
- Leverage `multiprocessing` for parallel image analysis
- Consider memory usage when processing large image stacks
- Use `polars` instead of `pandas` for large datasets where available

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

### **Latest Updates (August 27, 2025)**

**All_Analysis_OneBook Script Rewrite [COMPLETED]:**
- ✅ **Terminal exit issue resolved**: Complete memory-safe rewrite eliminates crashes
- ✅ **Memory management**: Forced garbage collection and resource cleanup after each folder
- ✅ **Terminal output optimized**: Single-line status updates with carriage return, proper flush-only logging
- ✅ **Robust error handling**: Replaced dangerous infinite recursion with safe retry logic
- ✅ **Progress tracking**: Comprehensive logging with real-time statistics and success rates
- ✅ **Configuration-driven**: Type-safe dataclass approach for easy parameter management
- ✅ **Production ready**: Handles large-scale dataset processing without system resource exhaustion

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

### **Next Priority Targets**
1. **Legacy cleanup**: Remove SpectralFunctions_Old.py and PlottingFunctions_Old.py (1,925 lines total)
2. **postprocess.py**: Highest complexity (239), large function decomposition needed  
3. **Final error handling**: Fix remaining 4 bare `except:` clauses in CalibrationFunctions.py, IOFunctions.py, and lib.py

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