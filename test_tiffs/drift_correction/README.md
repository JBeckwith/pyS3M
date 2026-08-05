# drift_correction

Simulated dye(s): ATTO 647N
Recommended Peak λ for fitting (FittingPanel "Peak λ"): 0.685 µm
  — mean of the simulated dyes' average emission wavelengths, the single shared
  value the renderer used for PSF sigma; matching it here keeps the fit's PSF-sigma
  expectation consistent with what was actually simulated.
Modality: PAINT
Candidate density: 0.3 /µm² (pool size scales with density x area x n_frames — see pattern_source.pool_size_for_density)
N frames: 300
N candidates (ground-truth pool size): 610

Injected linear drift trajectory: (0,0) -> (1000, 500) nm over the movie (ground_truth/injected_drift_nm.npy, shape (n_frames, 2), [dx, dy] in nm). Ground truth xc_nm/yc_nm are the undrifted reference positions drift correction should recover.

In the Drift Correction panel, use Segmentation = 5 (not the panel's default 100). AIM's roi_r (default 60 nm) is the search radius used to match the same molecule between segments — it must exceed the drift accumulated within one segment. At this fixture's drift rate (~3.3/1.7 nm/frame x/y), segmentation=5 accumulates ~17/8 nm per segment (well inside the default 60 nm roi_r); segmentation=100 accumulates ~333/167 nm per segment (far past it), so AIM loses track and the reported drift collapses to ~100 nm total instead of the real ~1000/500 nm — confirmed directly (segmentation=5 recovers a 703/461 nm span; segmentation=30/100 recover only 72-119 nm). This isn't a bug — any real dataset needs roi_r scaled to segmentation x drift rate the same way.
