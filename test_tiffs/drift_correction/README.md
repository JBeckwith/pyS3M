# drift_correction

Simulated dye(s): Gold nanoparticle (561 nm elastic scatter)
Recommended Peak λ for fitting (FittingPanel "Peak λ"): 0.561 µm
  — mean of the simulated dyes' average emission wavelengths, the single shared
  value the renderer used for PSF sigma; matching it here keeps the fit's PSF-sigma
  expectation consistent with what was actually simulated.
Modality: PAINT
Candidate density: 0.015 /µm² (pool size scales with density x area x n_frames — see pattern_source.pool_size_for_density)
N frames: 300
N candidates (ground-truth pool size): 30
Minimum candidate separation: 1000 nm centre-to-centre, enforced across the whole candidate pool (not just per-frame ON subsets, which this conservatively subsumes) — see pattern_source.sample_n_positions_in_mask.

Emitters simulate gold-nanoparticle fiducials, not a fluorescent dye: non-blinking, non-bleaching elastic (Rayleigh/Mie) point scatterers at an effective illumination wavelength of 561 nm (no Stokes shift — the 'dye' the pipeline sees is a narrow synthetic spectrum centred there, not a database lookup). on_rate=1.0/off_rate=0.0 keeps every candidate ON for the whole movie, matching real fiducial usage. ~10,000 photons/frame (uniform 9000-11000, i.e. shot-noise-level frame-to-frame variation only, no stochastic blinking) — much brighter than the dye fixtures' 1000-10000 photon range, matching how real gold-NP fiducials read out.

Injected linear drift trajectory: (0,0) -> (1000, 500) nm over the movie (ground_truth/injected_drift_nm.npy, shape (n_frames, 2), [dx, dy] in nm). Ground truth xc_nm/yc_nm are the undrifted reference positions drift correction should recover.

In the Drift Correction panel, use Segmentation = 5 (not the panel's default 100). AIM's roi_r (default 60 nm) is the search radius used to match the same molecule between segments — it must exceed the drift accumulated within one segment. At this fixture's drift rate (~3.3/1.7 nm/frame x/y), segmentation=5 accumulates ~17/8 nm per segment (well inside the default 60 nm roi_r); segmentation=100 accumulates ~333/167 nm per segment (far past it), so AIM loses track and the reported drift collapses to ~100 nm total instead of the real ~1000/500 nm — confirmed directly (segmentation=5 recovers a 703/461 nm span; segmentation=30/100 recover only 72-119 nm). This isn't a bug — any real dataset needs roi_r scaled to segmentation x drift rate the same way.
