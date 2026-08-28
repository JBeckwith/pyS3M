# clustering

Simulated dye(s): ATTO 647N
Recommended Peak λ for fitting (FittingPanel "Peak λ"): 0.685 µm
  — mean of the simulated dyes' average emission wavelengths, the single shared
  value the renderer used for PSF sigma; matching it here keeps the fit's PSF-sigma
  expectation consistent with what was actually simulated.
Modality: PAINT
Candidate density: 0.02 /µm² (pool size scales with density x area x n_frames — see pattern_source.pool_size_for_density)
N frames: 400
N candidates (ground-truth pool size): 15
Minimum candidate separation: 1000 nm centre-to-centre, enforced across the whole candidate pool (not just per-frame ON subsets, which this conservatively subsumes) — see pattern_source.sample_n_positions_in_mask.

Mean n_frames_on per candidate: 58.9 (repeat-visit blinking).
