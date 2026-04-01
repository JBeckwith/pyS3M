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
and 60\,nm respectively across all datasets. Drift-corrected localisations were grouped into
single-molecule identities by Hierarchical Density-Based Spatial Clustering
of Applications with Noise (HDBSCAN)\cite{mcinnesHdbscan2017JOSS} applied to
the two-dimensional localisation coordinates. The minimum cluster size was set
to 10\,localisations, and the cluster-selection distance $\varepsilon$ was set
to the mean per-axis localisation precision of the dataset, calculated from the
fitted position uncertainties. Localisations assigned noise labels ($-1$) were
discarded. Per-molecule average position, colour fractions, PSF width, and
photon count were computed as photon-weighted means over all member
localisations.
```
