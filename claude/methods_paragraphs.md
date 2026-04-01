# Methods Paragraphs

## Single-dye SMLM analysis pipeline

```latex
\subsection*{Single-molecule localisation and drift correction}

Raw Bayer-mosaic frames were converted to photoelectron images by subtracting
the per-pixel offset map and dividing by the gain map, both obtained from
camera calibration (see \ref{note:Camera_Calibration}). Candidate
single-molecule puncta were identified in each frame using a Matched Filter
followed by a Cell-Averaging Constant False Alarm Rate (CA-CFAR) threshold
set at a per-pixel false-alarm probability $P_\mathrm{FA} = 10^{-3}$, implemented
following Hekrdla \ea\cite{hekrdlaOptimized2025Nat.Commun.} Each candidate was
extracted as a $16 \times 16$\,pixel region of interest from the raw Bayer
image, and fitted by Levenberg--Marquardt weighted least squares to a
multi-channel Gaussian point-spread function model in which the background
and per-channel amplitudes $(A_\mathrm{R}, A_\mathrm{G}, A_\mathrm{B})$ enter
as their square roots to enforce positivity. Fitting was performed in two
stages: an initial fit with weights derived from a Gaussian-smoothed
($\sigma = 1.5$\,pixels) copy of the ROI, followed by a second fit
warm-started from the first using model-predicted Poisson weights.
Localisations were retained if the reduced chi-squared $\chi^2_\nu$ was below
the median value for the acquisition, the fitted PSF width $\sigma_{xy}$
lay within $75$--$160$\,nm, the per-axis localisation uncertainty
$\delta x, \delta y < 1$\,pixel, and the total detected photon count exceeded
$500$\,photons.

Stage drift was corrected using the Adaptive Intersection Maximisation (AIM)
algorithm\cite{maDriftfree2024Sci.Adv.} The temporal segmentation was chosen
to match the localisation density of each dataset (typically 10--50\,frames).
The intersection distance and search-region radius were fixed at 20\,nm
and 60\,nm respectively across all datasets. For super-resolution experiments
(\textit{i.e.} all data presented in Fig.\,\ref{fig:Single-Resolution_Applications}),
repeated detections of the same blinking emitter were linked across frames:
localisations separated by less than the mean per-axis localisation precision
in the image plane and by no more than two consecutive dark frames were
consolidated into a single event, with photon counts summed and spectral
fractions, PSF widths, and $\chi^2_\nu$ averaged over contributing frames;
events whose first or last frame coincided with the start or end of the
acquisition were discarded as their on-times were indeterminate. For
single-molecule experiments (\textit{i.e.} all data presented in
Fig.\,\ref{fig:Single-Molecule_Applications}), localisations from the same
physical emitter were grouped into single-molecule identities by Hierarchical
Density-Based Spatial Clustering of Applications with Noise
(HDBSCAN)\cite{mcinnesHdbscan2017JOSS} applied to the two-dimensional
localisation coordinates, with cluster-selection distance $\varepsilon$ set
to the mean per-axis localisation precision and a minimum cluster size of
10\,localisations; localisations assigned noise labels ($-1$) were discarded,
and photon-weighted means of the position, colour fractions, and PSF width
were taken as the representative single-molecule observables.
```

---

## FRET post-hoc change-point analysis

```latex
\subsection*{FRET single-molecule analysis}

Holliday junction FRET data were processed using a dedicated pipeline that
separates spot detection, photobleaching identification, and spectral fitting.
Candidate puncta were identified on the variance-aware demosaiced sum of the
first 50\,frames of each acquisition; summing increases the signal-to-noise
ratio for detection whilst the variance map is scaled accordingly
($\sigma^2_\mathrm{sum} = N\,\sigma^2_\mathrm{single}$), preserving the
statistical validity of the CA-CFAR threshold. A $12 \times 12$\,pixel ROI
was extracted for each detected punctum.

Before fitting the PSF in every frame, the photobleaching transition was
located using the PELT change-point algorithm\cite{truongSelective2020SignalProcessing}
applied to the one-dimensional total-intensity trace (sum of photoelectrons
within the ROI) across the full acquisition. The noise level $\hat{\sigma}$
was estimated from the last 100\,frames of each trace, which correspond to
the post-bleach background; the penalty was set to $\beta = n\,\hat{\sigma}^2$,
where $n$ is the trace length, and a minimum segment length of 5\,frames was
imposed. Puncta for which no change point was detected (i.e.\ no bleaching
step) were discarded. The Gaussian PSF model was then fitted independently at
every frame from the start of the acquisition up to the identified bleaching
transition, yielding per-frame position, PSF width, photon count, and spectral
fractions $(A_\mathrm{R}, A_\mathrm{G}, A_\mathrm{B})$.

Localisations in which all three colour fractions lay within 0.01 of the
fitter's initial value of $\nicefrac{1}{3}$ were subsequently discarded as
unconverged fits. The FRET ratio at each frame was computed as
$A_\mathrm{R}/A_\mathrm{G}$; no additional spectral correction was applied
beyond the per-pixel quantum efficiency weighting implicit in the fitting model.

FRET state transitions were then identified by a second application of PELT,
now operating jointly on the two-dimensional signal
$[A_\mathrm{R}(t),\, A_\mathrm{G}(t)]$. Joint detection exploits the
constraint $A_\mathrm{R} + A_\mathrm{G} + A_\mathrm{B} = 1$, under which a
FRET transition shifts $A_\mathrm{R}$ and $A_\mathrm{G}$ in anti-correlated
directions; a coherent change in both channels is detected with greater
sensitivity than either channel alone. An $\ell_2$ cost function was used.
The penalty was set to $\beta = \log(n)\,d\,\hat{\sigma}^2$, where $n$ is
the number of pre-bleach frames, $d = 2$ is the signal dimension, and
$\hat{\sigma}^2$ is the mean per-channel variance of the trace. This
Bayesian Information Criterion (BIC) scaling is necessary because spectral
fractions are bounded on $[0,\,1]$ and their variance is orders of magnitude
smaller than that of photon-count traces; applying an unscaled penalty would
suppress all but the largest transitions. A minimum segment length of
25\,frames was imposed. Puncta for which no FRET transition was detected were
excluded from further analysis; all change-point searches were parallelised
across puncta.
```
