# Spot Detection Validation Framework

## Overview

This framework provides comprehensive quantitative validation of spot detection functions to measure false positive and false negative rates using simulated ground truth data.

## Key Features

✅ **Ground Truth Simulation**: Creates 10×10 grids of puncta with known positions  
✅ **Realistic Imaging Pipeline**: Full camera simulation with noise, gain, offset, variance  
✅ **Bootstrap Sampling**: Tests across different camera regions for robust statistics  
✅ **Bayer Processing Comparison**: Tests both `bayer_image=True/False` conditions  
✅ **Performance Metrics**: Precision, recall, F1-score, false positive/negative rates  
✅ **Automated Analysis**: Complete pipeline from simulation to metrics calculation  

## Files Created

### Core Framework
- **`SpotDetectionValidation.py`** - Main validation framework class
- **`test_spot_detection_validation.py`** - Test script for framework validation

### Key Classes

#### `ValidationConfig`
Configuration dataclass with parameters:
```python
- grid_size: int = 10           # 10×10 puncta grid
- grid_spacing_microns: float = 1.0  # 1 μm spacing between puncta
- image_size_pixels: int = 200  # Camera image size
- n_photons_range: (1000, 5000) # Photon count range per punctum
- n_bootstrap: int = 50         # Number of bootstrap samples
- pfa: float = 1e-4            # Probability of false alarm
- test_bayer_processing: bool = True     # Test both Bayer conditions
- detection_tolerance_nm: float = 150.0  # True positive threshold
```

#### `SpotDetectionValidator`
Main validation class with methods:
- `load_camera_calibration()` - Load realistic camera parameters
- `generate_grid_positions()` - Create ground truth 10×10 grid
- `simulate_camera_image()` - Generate realistic Bayer images with noise
- `detect_spots()` - Apply spot detection algorithms
- `evaluate_detection_performance()` - Calculate metrics
- `run_validation_bootstrap()` - Complete bootstrap analysis

#### `DetectionMetrics` 
Results dataclass containing:
- True positives, false positives, false negatives
- Precision, recall, F1-score
- True positive rate, false positive rate
- Detected and ground truth positions

## Usage

### Basic Usage
```python
from SpotDetectionValidation import SpotDetectionValidator, ValidationConfig

# Configure validation parameters
config = ValidationConfig(
    grid_size=10,
    n_bootstrap=100,
    pfa=1e-4,
    detection_tolerance_nm=150.0
)

# Set up paths
camera_path = "Camera_Calibrations/Ximea_Camera"
save_folder = "validation_results/spot_detection"

# Run validation
validator = SpotDetectionValidator(config)
results = validator.run_validation_bootstrap(camera_path, save_folder)
```

### Test Framework
```bash
# Test the framework first
python src/test_spot_detection_validation.py

# Run full validation
python src/SpotDetectionValidation.py
```

## Validation Pipeline

### 1. **Ground Truth Generation**
- Creates 10×10 grid of puncta positions
- 1 μm spacing (approximately diffraction-limited separation)
- Positions centered in field of view
- Known coordinates for performance evaluation

### 2. **Camera Simulation**
- Loads realistic sCMOS camera calibrations (gain, offset, variance, RQE)
- Simulates realistic Bayer-filtered images
- Adds Poisson photon noise and camera read noise
- Uses ATTO550 dye spectrum as representative fluorophore

### 3. **Bootstrap Sampling**
- Samples different regions of camera sensor (realistic noise variation)
- Varies photon counts per punctum (signal strength testing)
- Multiple independent trials for statistical robustness

### 4. **Spot Detection**
- Converts simulated images to photoelectron counts (as with real data)
- Applies parallel spot detection with configurable PFA
- Uses matched filtering and CFAR detection algorithms

### 5. **Performance Evaluation**
- Matches detected spots to ground truth within tolerance
- Calculates comprehensive performance metrics
- Identifies systematic biases and failure modes

## Output Files

The framework generates several output files:

### Results Files
- **`spot_detection_validation_results.csv`** - Detailed per-sample results
- **`spot_detection_validation_summary.csv`** - Summary statistics
- **`spot_detection_validation_plots.png`** - Performance visualization plots

### Summary Report
Console output includes Bayer processing comparison:
```
======================================================================
SPOT DETECTION VALIDATION SUMMARY
======================================================================
Total bootstrap samples: 200
Successful samples: 196
Bayer processing comparison: ENABLED

📊 BAYER PROCESSING COMPARISON:

  🔶 WITH Bayer Averaging (bayer_image=True):
    Precision: 0.945 ± 0.082
    Recall: 0.892 ± 0.106
    F1-Score: 0.915 ± 0.089
    False Positive Rate: 0.0234 ± 0.0189
    Median False Positives: 2.0
    Median False Negatives: 8.0

  🔷 WITHOUT Bayer Averaging (bayer_image=False):
    Precision: 0.923 ± 0.095
    Recall: 0.885 ± 0.112
    F1-Score: 0.901 ± 0.096
    False Positive Rate: 0.0312 ± 0.0245
    Median False Positives: 3.0
    Median False Negatives: 9.0

  📈 PERFORMANCE DIFFERENCES:
    Precision difference (True - False): +0.022
    Recall difference (True - False): +0.007
    False Positive Rate difference (True - False): -0.0078
    → Bayer averaging REDUCES false positives by 0.0078
```

## Key Validation Metrics

### False Positive Rate Assessment
- **Target**: <1% false positive rate for high-precision SMLM
- **Measurement**: Ratio of incorrect detections to total detections
- **Impact**: False positives create artifact localizations

### False Negative Rate Assessment  
- **Target**: <5% false negative rate to maintain detection completeness
- **Measurement**: Fraction of ground truth spots missed
- **Impact**: False negatives reduce localization density

### Detection Precision vs SNR
- Tests performance across photon count range (1000-5000 photons)
- Evaluates degradation at low signal levels
- Identifies optimal detection thresholds

## Technical Implementation

### Simulation Pipeline
1. **Spectral Data**: Uses ATTO550 emission spectrum and camera QE curves
2. **PSF Modeling**: Airy disk PSF with realistic NA=1.49, 69nm pixels
3. **Noise Modeling**: Poisson photon noise + Gaussian read noise + sensor variance
4. **Bayer Processing**: Full Bayer filter simulation with demosaicing

### Detection Pipeline  
1. **Photoelectron Conversion**: (counts - offset) / (gain × RQE)
2. **Matched Filtering**: PSF-matched filtering for optimal SNR
3. **CFAR Detection**: Constant false alarm rate detection
4. **Local Maxima**: Non-maximum suppression for spot isolation

### Performance Evaluation
- **Spatial Matching**: Hungarian algorithm for optimal spot assignment
- **Distance Threshold**: 150nm default (≈2× localization precision)
- **Statistical Robustness**: Bootstrap sampling for confidence intervals

## Configuration Recommendations

### For Low False Positives (High Precision)
```python
config = ValidationConfig(
    pfa=1e-5,  # Stricter false alarm probability
    detection_tolerance_nm=100.0,  # Tighter matching
    n_bootstrap=200  # More samples for robust statistics
)
```

### For High Sensitivity (Low False Negatives)  
```python
config = ValidationConfig(
    pfa=1e-3,  # Relaxed false alarm probability
    n_photons_range=(800, 2000),  # Lower photon counts
    detection_tolerance_nm=200.0  # More generous matching
)
```

### For Computational Efficiency
```python
config = ValidationConfig(
    grid_size=5,  # Smaller grid (25 spots)
    image_size_pixels=100,  # Smaller images
    n_bootstrap=20  # Fewer samples
)
```

## Expected Performance Benchmarks

Based on typical SMLM spot detection performance:

| Metric | Good Performance | Acceptable | Needs Improvement |
|--------|------------------|------------|-------------------|
| Precision | >95% | 90-95% | <90% |
| Recall | >90% | 80-90% | <80% |
| False Positive Rate | <1% | 1-3% | >3% |
| F1-Score | >92% | 85-92% | <85% |

## Troubleshooting

### Common Issues

**High False Positive Rate**
- Increase PFA threshold (more stringent)
- Check background subtraction
- Verify camera calibration quality

**High False Negative Rate**  
- Decrease PFA threshold (less stringent)
- Check PSF model accuracy
- Verify photon count simulation

**Inconsistent Results**
- Increase bootstrap sample count
- Check camera region sampling strategy
- Verify ground truth grid positioning

### Debug Mode
Enable detailed debugging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Applications

### Algorithm Development
- Compare different detection algorithms
- Optimize detection parameters
- Benchmark performance improvements

### Quality Assurance
- Validate detection performance before experiments
- Monitor algorithm degradation over time
- Generate performance reports for publications

### Parameter Optimization
- Find optimal PFA thresholds
- Balance precision vs recall
- Optimize for specific experimental conditions

## Bayer Processing Comparison Analysis

### The `bayer_image` Parameter

The SpotDetectionFunctions now includes a `bayer_image` Boolean parameter that controls how Bayer-filtered images are processed:

- **`bayer_image=True`**: Applies Bayer averaging before spot detection (demosaicing/binning)
- **`bayer_image=False`**: Performs spot detection directly on raw Bayer-filtered image

### Expected Performance Differences

#### **Bayer Averaging (`bayer_image=True`) - Typically Better:**
- **Reduced Noise**: Averaging neighboring pixels reduces photon noise
- **Lower False Positives**: Smoother images reduce spurious detections
- **Higher Precision**: More reliable spot identification
- **Trade-off**: Slight reduction in spatial resolution

#### **No Bayer Averaging (`bayer_image=False`) - Raw Performance:**
- **Full Resolution**: Maintains original pixel resolution
- **Higher Noise**: More susceptible to individual pixel variations
- **More False Positives**: Noise spikes can trigger false detections
- **Trade-off**: Better localization precision but lower reliability

### Validation Results Interpretation

#### **When Bayer Averaging Helps:**
- **High Noise Conditions**: Low photon counts, high camera noise
- **Dense Spot Fields**: Reduces background false positives
- **Precision-Critical Applications**: When false positives are costly

#### **When Raw Processing May Be Better:**
- **High Signal-to-Noise**: Clean images with abundant photons
- **Sparse Spot Fields**: Few background distractions
- **Resolution-Critical Applications**: When every pixel matters

### Using Results to Optimize Detection

The validation framework quantifies the trade-offs:

1. **Compare False Positive Rates**: How much does Bayer averaging reduce FP?
2. **Compare Precision/Recall**: What's the detection sensitivity trade-off?
3. **Analyze by Photon Count**: Does the benefit change with signal strength?
4. **Consider Your Application**: Which metric matters most for your experiments?

### Example Decision Framework

```python
# If false positive rate difference > 0.01 and precision difference > 0.02:
#   Use bayer_image=True for high precision
# Elif recall difference < -0.05:
#   Use bayer_image=False to avoid missing spots  
# Else:
#   Use bayer_image=True as default (typically more robust)
```

This framework provides the quantitative foundation needed to ensure robust spot detection with well-characterized false positive and false negative rates for single-molecule localization microscopy applications, including understanding the impact of Bayer processing choices on detection performance.