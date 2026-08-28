# nile_red_example

Simulated dye(s): Nile Red (620 nm environment), Nile Red (640 nm environment)
Recommended Peak λ for fitting (FittingPanel "Peak λ"): 0.629 µm
  — mean of the simulated dyes' average emission wavelengths, the single shared
  value the renderer used for PSF sigma; matching it here keeps the fit's PSF-sigma
  expectation consistent with what was actually simulated.
Modality: PAINT
Candidate density: 0.05 /µm² (pool size scales with density x area x n_frames — see pattern_source.pool_size_for_density)
N frames: 60000
N candidates (ground-truth pool size): 5760
Minimum candidate separation: none (disabled) — candidates deliberately dense/overlapping for this fixture, see below.

**No raw TIFF stack** — unlike every other fixture, this one only ships the already fit+clustered localisation table (nile_red_example_localisations.h5, 41,056 rows, HDF5 key "data" — load directly via the Nile Red panel's "Load Localisations", or `IOFunctions.IO_Functions().read_h5_database(...)`). Statistically equivalent to a real ~60,000-frame, ~0.05 localisations/µm² PAINT movie (density_per_um2=0.05 × n_frames=60000 preserves that same product) — simulated, fit, and clustered through the real camera + spot-fitting pipeline (real per-localisation noise, not synthetic), then the raw TIFF was discarded to keep this fixture's size reasonable.

Per-environment candidate split (ground truth): {'Nile Red (620 nm environment)': 3603, 'Nile Red (640 nm environment)': 2157}. Two genuinely different Nile Red environments were injected, resolved so the Nile Red panel's pixelated fit recovers a centre-of-mass wavelength of 620 nm / 640 nm respectively (not the recommended Peak λ above, which is just their mean and is NOT the right thing to fit a preview PSF with here). Use the Nile Red panel's filters (default: semrock-di03-r514-t1-25x36, semrock-ff01-515-lp, semrock-ff01-650-200) — this fixture was simulated with exactly those, so fitting with a different filter set will not recover 620/640 nm as cleanly. Load Localisations, then Run Nile Red Fit — expect two clear peaks near 620/640 nm in the histogram and two visually distinct circle regions in the wavelength map.

Deliberately dense/overlapping coverage (min_separation_nm=0, unlike the other fixtures' 1000 nm minimum) — the pixelated fit needs several localisations per 50 nm grid cell almost everywhere inside each circle, not isolated single molecules.
