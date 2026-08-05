# channel_unmixing

Simulated dye(s): Cy3B, Alexa Fluor 647
Recommended Peak λ for fitting (FittingPanel "Peak λ"): 0.644 µm
  — mean of the simulated dyes' average emission wavelengths, the single shared
  value the renderer used for PSF sigma; matching it here keeps the fit's PSF-sigma
  expectation consistent with what was actually simulated.
Modality: PAINT
Candidate density: 0.02 /µm² (pool size scales with density x area x n_frames — see pattern_source.pool_size_for_density)
N frames: 400
N candidates (ground-truth pool size): 40

Per-dye candidate split: {'Cy3B': 20, 'Alexa Fluor 647': 20}. Both dyes share one Peak λ during fitting (gen_camera_image_stack uses one shared PSF sigma for the whole simulated stack) — the two dyes differ in Bayer-channel colour split (A_R/A_G/A_B), not PSF size, which is what unmix_channels actually separates on.

