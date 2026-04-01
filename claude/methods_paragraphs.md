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
\subsection*{FRET post-hoc change-point analysis}

Per-punctum spectral fraction time series $(A_\mathrm{R}(t), A_\mathrm{G}(t))$
were extracted from the localisation database produced by \texttt{fit\_SM\_data}.
Localisations in which all three colour fractions lay within 0.01 of the
fitter's initial value of $\nicefrac{1}{3}$ were discarded as unconverged fits.
The FRET ratio at each frame was computed as $A_\mathrm{R}/A_\mathrm{G}$; no
additional spectral correction was applied beyond the per-pixel quantum
efficiency weighting implicit in the fitting model.

Change points in the FRET state were detected jointly on the two-dimensional
signal $[A_\mathrm{R}(t),\, A_\mathrm{G}(t)]$ using the Pruned Exact Linear
Time (PELT) algorithm\cite{truongSelective2020SignalProcessing} as implemented
in the \texttt{ruptures} package. Joint detection on both channels
simultaneously exploits the constraint $A_\mathrm{R} + A_\mathrm{G} +
A_\mathrm{B} = 1$, under which a FRET transition shifts $A_\mathrm{R}$ and
$A_\mathrm{G}$ in anti-correlated directions; a change point that moves both
channels coherently is detected with greater sensitivity than one identified
from either channel alone. An $\ell_2$ cost function was used. The penalty
was set to $\beta = \log(n)\,d\,\hat{\sigma}^2$, where $n$ is the number of
frames in the punctum trace, $d = 2$ is the signal dimension, and
$\hat{\sigma}^2$ is the mean per-channel variance estimated from the trace
itself. This Bayesian Information Criterion (BIC) scaling avoids the
over-penalisation that arises when applying a penalty calibrated for photon
counts to the bounded spectral fractions, whose variance is typically orders
of magnitude smaller. A minimum segment length of 25\,frames was imposed to
suppress spurious detections from shot noise. Puncta for which no change
point was found were excluded from further analysis; all detection was
performed in parallel across puncta.
```
