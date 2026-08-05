# -*- coding: utf-8 -*-
"""
pattern_source
~~~~~~~~~~~~~~

Turn a user-supplied pattern image (e.g. a grid, or flat-coloured shapes) into
per-dye/per-colour boolean masks describing where synthetic single-molecule
localisations should be rendered, for image-driven STORM/PAINT simulation.

A single-colour image (foreground colour vs. background) yields one mask; a
multicolour image (several distinct flat foreground colours, e.g. one ring per
dye) yields one mask per detected colour, so a multicolour input drives a
multi-dye simulation.

Each foreground colour's mask is obtained via an Otsu threshold
(``skimage.filters.threshold_otsu``, the same call pattern already used by
``postprocess.segment_locs_by_rendered_image``) applied to that colour's
per-pixel alpha/coverage score — this turns the soft, anti-aliased edge of a
drawn shape into a principled binary "solidly inside this colour's region"
mask, rather than an arbitrary alpha>0 cutoff.

:author: jsb92, 2026-08-05
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from skimage.filters import threshold_otsu


RGBTuple = Tuple[int, int, int]

#: Assumed physical span of a pattern image (e.g. the 512x512 example
#: patterns), in microns. Keeps two things in check: (1) the pattern's own
#: pixel grid stays well below the diffraction limit (~200-300 nm), so
#: candidate positions are sampled at a finer grid than the eventual PSF can
#: resolve, not quantised to it; (2) the resulting camera-pixel frame stays
#: small (a 10 um FOV at a 69-71.5 nm camera pixel size is ~140-145 camera
#: pixels across) — treating each pattern pixel as one camera pixel instead
#: would imply an unrealistically large FOV and correspondingly oversized
#: TIFF stacks.
DEFAULT_PATTERN_FOV_UM = 10.0


def default_pattern_pixel_size_nm(
    image_width_px: int, fov_um: float = DEFAULT_PATTERN_FOV_UM,
) -> float:
    """Physical size of one pattern-image pixel, in nm, assuming the image
    spans *fov_um* microns.

    Args:
        image_width_px: Pattern image width, in pixels.
        fov_um: Assumed physical width the image spans, in microns.

    Returns:
        Pixel size in nm.
    """
    return (fov_um * 1000.0) / image_width_px


def image_fov_camera_pixels(
    image: Union[str, Path, np.ndarray], camera_pixel_size_nm: float,
) -> Tuple[int, int]:
    """Camera-pixel field-of-view size a pattern image implies.

    Callers that need to pre-crop calibration maps before
    :func:`simulate_acquisition` (which expects them already cropped) use
    this to compute the crop size first — kept in sync with the identical
    computation :func:`simulate_acquisition` does internally.

    Args:
        image: Path to a pattern image, or an already-loaded RGB/RGBA array.
        camera_pixel_size_nm: Camera pixel size, in nm.

    Returns:
        ``(width, height)`` in camera pixels.
    """
    image = _as_rgba(image)
    pixel_size_nm = default_pattern_pixel_size_nm(image.shape[1])
    width = max(1, round(image.shape[1] * pixel_size_nm / camera_pixel_size_nm))
    height = max(1, round(image.shape[0] * pixel_size_nm / camera_pixel_size_nm))
    return width, height


def load_pattern_image(path: Union[str, Path]) -> np.ndarray:
    """Load a pattern image as an (H, W, 4) uint8 RGBA array.

    Args:
        path: Path to a PNG (or any format Pillow can open).

    Returns:
        RGBA image array, dtype uint8.
    """
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im.convert("RGBA"))


def _as_rgba(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    if isinstance(image, (str, Path)):
        return load_pattern_image(image)
    image = np.asarray(image)
    if image.ndim == 2:
        # Grayscale: broadcast to RGB, fully opaque.
        rgb = np.stack([image] * 3, axis=-1)
        alpha = np.full(image.shape, 255, dtype=np.uint8)
        return np.dstack([rgb, alpha]).astype(np.uint8)
    if image.shape[-1] == 3:
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
        return np.dstack([image, alpha]).astype(np.uint8)
    return image.astype(np.uint8)


def detect_palette(
    image: Union[str, Path, np.ndarray],
    background: Optional[RGBTuple] = None,
    min_fraction: float = 5e-4,
    merge_distance: float = 24.0,
) -> Tuple[RGBTuple, List[RGBTuple]]:
    """Identify the background colour and distinct foreground colours in *image*.

    Args:
        image: Path to a pattern image, or an already-loaded RGB/RGBA array.
        background: Known background RGB colour. If ``None``, taken as the
            single most common RGB colour in the image (the usual case for a
            flat-colour canvas, e.g. white).
        min_fraction: Minimum fraction of total pixels a colour must cover to
            be counted as a genuine foreground colour, rather than an
            anti-aliasing/noise artefact.
        merge_distance: Candidate foreground colours within this Euclidean RGB
            distance of an already-kept colour are merged into it (by pixel
            count) rather than kept as a separate colour — compression/export
            can otherwise split what is visually one flat colour into several
            near-duplicate RGB values (e.g. ``(128, 0, 128)`` vs.
            ``(129, 0, 129)``).

    Returns:
        ``(background_rgb, foreground_rgbs)`` — background as an
        ``(r, g, b)`` tuple, foreground colours as a list of the same,
        ordered by descending pixel count.
    """
    rgba = _as_rgba(image)
    rgb = rgba[..., :3].reshape(-1, 3)
    n_total = rgb.shape[0]

    colours, counts = np.unique(rgb, axis=0, return_counts=True)
    order = np.argsort(-counts)
    colours, counts = colours[order], counts[order]

    if background is None:
        background = tuple(int(v) for v in colours[0])
    bg = np.array(background)

    foreground: List[RGBTuple] = []
    kept = np.empty((0, 3))
    for c, n in zip(colours, counts):
        if np.array_equal(c, bg) or n / n_total < min_fraction:
            continue
        if kept.shape[0] > 0:
            dist = np.linalg.norm(kept - c, axis=1)
            if dist.min() < merge_distance:
                continue  # near-duplicate of an already-kept, more-common colour
        foreground.append(tuple(int(v) for v in c))
        kept = np.vstack([kept, c])

    return background, foreground


def mask_from_image(
    image: Union[str, Path, np.ndarray],
    background: Optional[RGBTuple] = None,
    min_fraction: float = 5e-4,
) -> Dict[RGBTuple, np.ndarray]:
    """Otsu-threshold *image* into one boolean mask per foreground colour.

    Each non-background pixel is assigned to its nearest foreground colour
    (by Euclidean RGB distance); that colour's per-pixel alpha value (its
    coverage/opacity) then forms a scalar map which is Otsu-thresholded to a
    boolean mask, so anti-aliased edges are resolved to a clean "this pixel is
    solidly part of that colour's shape" region rather than an arbitrary
    alpha>0 cutoff.

    Args:
        image: Path to a pattern image, or an already-loaded RGB/RGBA array.
        background: Known background RGB colour, forwarded to
            :func:`detect_palette`.
        min_fraction: Forwarded to :func:`detect_palette`.

    Returns:
        Dict mapping each foreground ``(r, g, b)`` colour to a boolean mask of
        shape ``(H, W)``, ``True`` where localisations of that colour should
        be rendered. Empty dict if no foreground colour is found.
    """
    rgba = _as_rgba(image)
    rgb = rgba[..., :3].astype(np.int16)
    alpha = rgba[..., 3].astype(np.float64)

    background, foreground = detect_palette(rgba, background=background, min_fraction=min_fraction)
    if not foreground:
        return {}

    bg = np.array(background, dtype=np.int16)
    is_background = np.all(rgb == bg, axis=-1)

    palette = np.array(foreground, dtype=np.int16)  # (n_colours, 3)
    # Nearest-palette-colour assignment for every non-background pixel.
    diffs = rgb[None, ...] - palette[:, None, None, :]  # (n_colours, H, W, 3)
    dist2 = np.sum(diffs.astype(np.int64) ** 2, axis=-1)  # (n_colours, H, W)
    nearest = np.argmin(dist2, axis=0)  # (H, W)

    masks: Dict[RGBTuple, np.ndarray] = {}
    for i, colour in enumerate(foreground):
        score = np.where((nearest == i) & (~is_background) & (alpha > 0), alpha, 0.0)
        if not np.any(score > 0):
            masks[colour] = np.zeros(score.shape, dtype=bool)
            continue
        try:
            threshold = threshold_otsu(score)
        except ValueError:
            threshold = 0.0
        masks[colour] = score > threshold

    return masks


def sample_n_positions_in_mask(
    mask: np.ndarray,
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw exactly *n* candidate emitter positions from the ``True`` region
    of *mask* (with replacement — positions are independent draws, so the
    same mask pixel can be hit more than once).

    Positions are pixel indices into *mask* plus a uniform sub-pixel jitter,
    so multiple candidates landing in the same mask pixel still separate out
    rather than stacking exactly on a lattice.

    Args:
        mask: Boolean array, ``True`` where localisations may be placed —
            typically one colour's mask from :func:`mask_from_image`.
        n: Number of positions to draw.
        rng: Random generator. A fresh :func:`numpy.random.default_rng` is
            used if omitted.

    Returns:
        ``(2, n)`` float array of ``[x, y]`` positions in *mask* pixel
        coordinates (fractional). Shape ``(2, 0)`` if the mask is empty or
        ``n <= 0``.
    """
    if rng is None:
        rng = np.random.default_rng()

    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or n <= 0:
        return np.zeros((2, 0))

    idx = rng.integers(0, len(xs), size=n)
    jitter = rng.uniform(-0.5, 0.5, size=(n, 2))
    x = xs[idx].astype(np.float64) + jitter[:, 0]
    y = ys[idx].astype(np.float64) + jitter[:, 1]
    return np.vstack([x, y])


def sample_positions_in_mask(
    mask: np.ndarray,
    density_per_um2: float,
    pixel_size_nm: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw candidate emitter positions from the ``True`` region of *mask*.

    The expected count is ``density_per_um2 * mask_area_um2``, actually drawn
    is Poisson about that mean (matching the random areal density of a real
    fluorophore/PAINT-probe labelling reaction, rather than a fixed count).

    Args:
        mask: Boolean array, ``True`` where localisations may be placed —
            typically one colour's mask from :func:`mask_from_image`.
        density_per_um2: Target areal density of candidate positions.
        pixel_size_nm: Physical size of one *mask* pixel, in nm — sets the
            mask's area in µm² together with ``mask.sum()``. Defaults to
            :func:`default_pattern_pixel_size_nm` (mask spans
            :data:`DEFAULT_PATTERN_FOV_UM`) if omitted.
        rng: Random generator. A fresh :func:`numpy.random.default_rng` is
            used if omitted.

    Returns:
        ``(2, N)`` float array of ``[x, y]`` positions in *mask* pixel
        coordinates (fractional). Shape ``(2, 0)`` if the mask is empty or no
        candidates are drawn.
    """
    if rng is None:
        rng = np.random.default_rng()
    if pixel_size_nm is None:
        pixel_size_nm = default_pattern_pixel_size_nm(mask.shape[1])

    n_px = int(np.count_nonzero(mask))
    area_per_px_um2 = (pixel_size_nm / 1000.0) ** 2
    expected_n = density_per_um2 * n_px * area_per_px_um2
    n = int(rng.poisson(expected_n)) if expected_n > 0 else 0
    return sample_n_positions_in_mask(mask, n, rng)


def sample_positions_per_colour(
    masks: Dict[RGBTuple, np.ndarray],
    density_per_um2: float,
    pixel_size_nm: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[RGBTuple, np.ndarray]:
    """Apply :func:`sample_positions_in_mask` to every colour in *masks*.

    A single *rng* is shared across colours (advanced between draws) so that
    calling this twice with the same seeded generator does not repeat one
    colour's draw.

    Args:
        masks: Per-colour boolean masks, as returned by :func:`mask_from_image`.
        density_per_um2: Forwarded to :func:`sample_positions_in_mask`.
        pixel_size_nm: Forwarded to :func:`sample_positions_in_mask`.
        rng: Random generator, shared across colours.

    Returns:
        Dict mapping each colour to its ``(2, N)`` position array.
    """
    if rng is None:
        rng = np.random.default_rng()
    return {
        colour: sample_positions_in_mask(mask, density_per_um2, pixel_size_nm, rng)
        for colour, mask in masks.items()
    }


def plot_sampled_positions(
    image: Union[str, Path, np.ndarray],
    positions_by_colour: Dict[RGBTuple, np.ndarray],
    colour_names: Optional[Dict[RGBTuple, str]] = None,
    point_size: float = 3.0,
):
    """Diagnostic figure: candidate localisation positions scattered over the
    source pattern, coloured per dye.

    Args:
        image: Path to a pattern image, or an already-loaded RGB/RGBA array.
        positions_by_colour: Per-colour ``(2, N)`` position arrays, as
            returned by :func:`sample_positions_per_colour`.
        colour_names: Optional ``{rgb: dye name}`` labels for the legend.
        point_size: Scatter marker size.

    Returns:
        A single-axes ``matplotlib.figure.Figure``.
    """
    from matplotlib.figure import Figure

    rgba = _as_rgba(image)
    alpha = rgba[..., 3:4].astype(np.float64) / 255.0
    rgb = rgba[..., :3].astype(np.float64)
    display_rgb = (rgb * alpha + 255.0 * (1 - alpha)).astype(np.uint8)

    fig = Figure(figsize=(6, 6), dpi=100, layout="constrained")
    ax = fig.add_subplot(111)
    ax.imshow(display_rgb)

    for colour, pos in positions_by_colour.items():
        if pos.shape[1] == 0:
            continue
        hexcode = "#%02x%02x%02x" % colour
        label = colour_names.get(colour, hexcode) if colour_names else hexcode
        ax.scatter(
            pos[0], pos[1], s=point_size, color=[c / 255.0 for c in colour],
            edgecolors="none", label=f"{label} (n={pos.shape[1]:,})",
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Sampled candidate localisations")
    ax.legend(fontsize=8, loc="upper right", markerscale=3)
    return fig


def plot_pattern_masks(
    image: Union[str, Path, np.ndarray],
    masks: Dict[RGBTuple, np.ndarray],
    colour_names: Optional[Dict[RGBTuple, str]] = None,
):
    """Diagnostic figure: the source pattern alongside where each colour's
    localisations will be rendered.

    Args:
        image: Path to a pattern image, or an already-loaded RGB/RGBA array.
        masks: Per-colour boolean masks, as returned by :func:`mask_from_image`.
        colour_names: Optional ``{rgb: dye name}`` labels for subplot titles.

    Returns:
        A ``matplotlib.figure.Figure`` with one source-image panel plus one
        panel per colour's mask.
    """
    from matplotlib.figure import Figure

    rgba = _as_rgba(image)
    # Composite over white for display, same convention as the source canvas.
    alpha = rgba[..., 3:4].astype(np.float64) / 255.0
    rgb = rgba[..., :3].astype(np.float64)
    display_rgb = (rgb * alpha + 255.0 * (1 - alpha)).astype(np.uint8)

    n_panels = 1 + len(masks)
    fig = Figure(figsize=(4 * n_panels, 4.2), dpi=100, layout="constrained")
    axes = fig.subplots(1, n_panels)
    if n_panels == 1:
        axes = [axes]

    axes[0].imshow(display_rgb)
    axes[0].set_title("Source pattern")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for ax, (colour, mask) in zip(axes[1:], masks.items()):
        hexcode = "#%02x%02x%02x" % colour
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = (*[c / 255.0 for c in colour], 1.0)
        ax.imshow(display_rgb)
        ax.imshow(overlay)
        label = colour_names.get(colour, hexcode) if colour_names else hexcode
        ax.set_title(f"{label}\n(n={int(mask.sum()):,} px)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    return fig


def duty_cycle(on_rate: float, off_rate: float) -> float:
    """Steady-state fraction of time a candidate spends ON under the
    two-state Markov model :func:`blinking_state_schedule` uses.

    Args:
        on_rate: Per-frame probability an OFF candidate turns ON.
        off_rate: Per-frame probability an ON candidate turns OFF.

    Returns:
        ``on_rate / (on_rate + off_rate)``, or 0.0 if both are 0.
    """
    return on_rate / (on_rate + off_rate) if (on_rate + off_rate) > 0 else 0.0


def pool_size_for_density(
    density_per_um2: float, mask_area_um2: float, n_frames: int,
) -> int:
    """Candidate pool size for a movie of *n_frames*: any point of the mask
    could in principle be sampled by a molecule in any frame, at the given
    density, so the pool of distinct candidates that could ever appear over
    the whole movie scales with ``density_per_um2 * mask_area_um2 *
    n_frames`` — not with the (much smaller) instantaneous per-frame count.

    ``on_rate``/``off_rate`` stay fixed, user-set constants — unaffected by
    pool size — passed to :func:`blinking_state_schedule` as before; it's
    the number of currently-on candidates that shrinks over a STORM movie
    (as more of the pool photobleaches and stops being eligible), not the
    per-candidate activation probability itself.

    Args:
        density_per_um2: Target density of candidate positions.
        mask_area_um2: Area of the mask this pool is drawn from.
        n_frames: Number of frames in the movie.

    Returns:
        Pool size (at least 1).
    """
    return max(1, round(density_per_um2 * mask_area_um2 * n_frames))


def blinking_state_schedule(
    n_candidates: int,
    n_frames: int,
    on_rate: float = 0.01,
    off_rate: float = 0.5,
    modality: str = "STORM",
    bleach_after_cycles: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Per-frame ON/OFF state from a simple two-state Markov blinking model.

    Each frame, an OFF candidate turns ON with probability *on_rate*; an ON
    candidate turns OFF with probability *off_rate*. Defaults
    (``on_rate=0.01``, ``off_rate=0.5``) give a duty cycle of ~2% (most
    candidates dark most of the time, matching real STORM/PAINT label
    kinetics) with a mean on-time of ``1/off_rate`` = 2 frames.

    *modality* controls the key STORM/PAINT difference:

    - ``"STORM"``: photobleaching. Each candidate is limited to
      *bleach_after_cycles* ON events (a fixed, finite dye survives only a
      handful of on/off cycles before permanent photobleaching); once used
      up, it can finish its current ON period but never turns on again.
    - ``"PAINT"``: no bleaching. Transient probes are continuously exchanged
      from an effectively infinite solution reservoir, so the two-state
      process runs unmodified for the whole acquisition.

    Note: this returns *state* only, not photon counts — the renderer this
    feeds (``simulation.multicolour.gen_camera_image_stack``) shares one
    photon count across every simultaneously-on candidate of a given dye in
    a frame (see :func:`per_frame_photon_budget`), rather than drawing an
    independent brightness per molecule per frame.

    Args:
        n_candidates: Number of candidate emitter positions.
        n_frames: Number of frames to simulate.
        on_rate: Per-frame probability an OFF candidate turns ON.
        off_rate: Per-frame probability an ON candidate turns OFF.
        modality: ``"STORM"`` (bleaches) or ``"PAINT"`` (does not bleach).
        bleach_after_cycles: STORM only — number of ON events before a
            candidate photobleaches permanently.
        rng: Random generator. A fresh :func:`numpy.random.default_rng` is
            used if omitted.

    Returns:
        ``(n_frames, n_candidates)`` boolean array, ``True`` where ON.
    """
    if rng is None:
        rng = np.random.default_rng()
    if modality not in ("STORM", "PAINT"):
        raise ValueError(f"modality must be 'STORM' or 'PAINT', got {modality!r}")
    if n_candidates == 0:
        return np.zeros((n_frames, 0), dtype=bool)

    dc = duty_cycle(on_rate, off_rate)
    state = rng.random(n_candidates) < dc  # steady-state initial condition

    bleaches = modality == "STORM"
    cycles_used = state.astype(np.int64)  # candidates starting ON have used cycle 1
    bleached = np.zeros(n_candidates, dtype=bool)

    schedule = np.zeros((n_frames, n_candidates), dtype=bool)
    for t in range(n_frames):
        can_turn_on = ~state if not bleaches else (~state & ~bleached)
        turn_on = can_turn_on & (rng.random(n_candidates) < on_rate)
        turn_off = state & (rng.random(n_candidates) < off_rate)
        state = (state | turn_on) & (~turn_off)

        if bleaches:
            cycles_used += turn_on.astype(np.int64)
            bleached |= cycles_used >= bleach_after_cycles

        schedule[t] = state

    return schedule


def build_x0y0_track(
    positions_nm: np.ndarray,
    on_state: np.ndarray,
    off_canvas_nm: float = 1.0e7,
) -> np.ndarray:
    """Combine static candidate positions with a per-frame ON/OFF schedule
    into the per-frame position track ``gen_camera_image_stack`` expects.

    ``gen_camera_image_stack`` has no notion of a candidate being absent from
    a frame (its per-frame photon count is one value shared by every position
    present that frame — see :func:`per_frame_photon_budget`), so an OFF
    candidate is represented by moving its position far outside the field of
    view for that frame instead: its PSF crop window then falls entirely
    outside the image and contributes nothing
    (``PSFFunctions.gen_spatial_PSF_fast`` skips crop windows that don't
    intersect the frame at all), which is safe and requires no special-casing
    downstream.

    Args:
        positions_nm: ``(2, N)`` static ``[x, y]`` positions in nm, as
            returned by :func:`sample_positions_in_mask` after converting
            from pattern-pixel to nm units.
        on_state: ``(n_frames, N)`` boolean ON/OFF schedule, as returned by
            :func:`blinking_state_schedule`.
        off_canvas_nm: Position (nm) substituted on both axes for OFF frames.
            Default (1 cm) is far outside any realistic field of view.

    Returns:
        ``(n_frames, 2, N)`` float array.
    """
    _, n = on_state.shape
    if positions_nm.shape != (2, n):
        raise ValueError(
            f"positions_nm shape {positions_nm.shape} does not match "
            f"on_state's {n} candidates"
        )
    return np.where(on_state[:, None, :], positions_nm[None, :, :], off_canvas_nm)


def per_frame_photon_budget(
    n_frames: int,
    photon_range: Tuple[float, float] = (500.0, 2000.0),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw one photon count per frame, shared by every candidate of a dye
    that is ON that frame (the granularity ``gen_camera_image_stack``
    actually supports — see :func:`build_x0y0_track`).

    Args:
        n_frames: Number of frames.
        photon_range: ``(min, max)`` photon count, drawn uniformly per frame.
        rng: Random generator. A fresh :func:`numpy.random.default_rng` is
            used if omitted.

    Returns:
        ``(n_frames,)`` float array.
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(photon_range[0], photon_range[1], size=n_frames)


def simulate_acquisition(
    image: Union[str, Path, np.ndarray],
    colour_to_dye: Dict[RGBTuple, str],
    camera: str,
    pixel_size_um: float,
    gain_map: np.ndarray,
    offset_map: np.ndarray,
    variance_map: np.ndarray,
    rqe_map: np.ndarray,
    n_frames: int,
    density_per_um2: float,
    modality: str = "STORM",
    on_rate: float = 0.01,
    off_rate: float = 0.5,
    bleach_after_cycles: int = 5,
    photon_range: Tuple[float, float] = (1000.0, 10000.0),
    background_photons: float = 5.0,
    na: float = 1.49,
    drift_nm: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
):
    """Render a synthetic STORM/PAINT frame stack from a pattern image.

    Pure simulation — no file I/O, no GUI/pipeline dependency — so it can be
    called from the GUI (``MainWindow._simulate_pattern_acquisition``, which
    additionally writes the result to disk) or standalone tooling (e.g.
    ``claude/generate_test_fixtures.py``) alike.

    Any point of each colour's Otsu mask (:func:`mask_from_image`) could be
    sampled by a molecule in any frame at *density_per_um2*, so the candidate
    pool scales with ``density_per_um2 * mask_area_um2 * n_frames``
    (:func:`pool_size_for_density`) — not with the much smaller instantaneous
    per-frame active count. ``on_rate``/``off_rate`` are fixed, independent of
    pool size; it's the number of currently-on candidates that shrinks over a
    STORM movie as the pool progressively bleaches out, not the per-candidate
    activation probability.

    Args:
        image: Path to a pattern image, or an already-loaded RGB/RGBA array.
        colour_to_dye: ``{(r, g, b): dye_name}`` — which dye each detected
            pattern colour represents. Two colours may map to the same dye
            (their candidate pools are merged).
        camera: Camera name (``"ximea"`` or ``"zwo"``) — sets the Bayer
            mosaic layout.
        pixel_size_um: Camera pixel size, in µm.
        gain_map, offset_map, variance_map, rqe_map: Calibration maps,
            already cropped to the pattern's implied field of view (see
            *width*/*height* in the return value — crop to those before
            calling, e.g. ``gain_map[:height, :width]``... but since the
            field-of-view size depends on the *image*, in practice call
            :func:`default_pattern_pixel_size_nm` and compute width/height
            first, as ``MainWindow._simulate_pattern_acquisition`` does).
        n_frames: Number of frames to simulate.
        density_per_um2: Candidate density (see above).
        modality: ``"STORM"`` (bleaches) or ``"PAINT"`` (does not bleach).
        on_rate: Per-frame probability an OFF candidate turns ON.
        off_rate: Per-frame probability an ON candidate turns OFF.
        bleach_after_cycles: STORM only — ON events before permanent bleach.
        photon_range: ``(min, max)`` photon count per frame.
        background_photons: Background photons per pixel.
        na: Numerical aperture.
        drift_nm: Optional ``(n_frames, 2)`` ``[dx, dy]`` trajectory (nm),
            added uniformly to every candidate's position track — for
            generating drift-correction test data. Ground truth positions
            (the return value's ``xc_nm``/``yc_nm``) stay at the undrifted
            reference position, which is what drift correction should
            recover.
        rng: Random generator. A fresh :func:`numpy.random.default_rng` is
            used if omitted.

    Returns:
        ``(bayer_stack, ground_truth_df, width, height, average_emission_wavelength_nm)``
        — the raw Bayer frame stack, a DataFrame with one row per candidate
        (``dye``, ``colour``, ``xc_nm``, ``yc_nm``, ``n_frames_on``), the
        camera-pixel field-of-view size the calibration maps must be cropped
        to (compute these first if pre-cropping the calibration args), and
        the mean of the selected dyes' average emission wavelengths (nm) —
        the single shared value the renderer used for PSF sigma (see the
        note above), and correspondingly the value to set as "Peak λ" in
        ``FittingConfig``/``FittingPanel`` so a downstream fit's PSF-sigma
        expectation matches what was actually simulated.

    Raises:
        ValueError: If every colour's mask is empty (nothing to simulate).
    """
    import pandas as pd
    from pyS3M.simulation.multicolour import MultiC_Sim_Funcs_Refactored
    import pyS3M.SpectralFunctions as SpectralFunctions
    import pyS3M.MaskFunctions as MaskFunctions
    import pyS3M.CameraDefaults as CameraDefaults

    if rng is None:
        rng = np.random.default_rng()

    image = _as_rgba(image)
    pixel_size_nm = default_pattern_pixel_size_nm(image.shape[1])
    masks = mask_from_image(image)
    mask_area_um2 = {
        colour: float(mask.sum()) * (pixel_size_nm / 1000.0) ** 2
        for colour, mask in masks.items()
    }

    positions_by_colour = {
        colour: sample_n_positions_in_mask(
            mask, pool_size_for_density(density_per_um2, mask_area_um2[colour], n_frames), rng,
        )
        for colour, mask in masks.items()
    }

    cam_cfg = CameraDefaults.get_camera_config(camera)
    camera_pixel_size_nm = pixel_size_um * 1000.0
    fov_x_nm = image.shape[1] * pixel_size_nm
    fov_y_nm = image.shape[0] * pixel_size_nm
    width = max(1, round(fov_x_nm / camera_pixel_size_nm))
    height = max(1, round(fov_y_nm / camera_pixel_size_nm))

    masks_cam = MaskFunctions.Mask_Functions().get_masks(
        mosaic_unit=cam_cfg.mosaic_unit, size_x=gain_map.shape[0], size_y=gain_map.shape[1],
    )

    spectral = SpectralFunctions.Spectral_Funcs()
    R_qe, G_qe, B_qe, wl = spectral.getpixelefficiency()
    pixel_QYs = np.vstack([B_qe, G_qe, R_qe])
    pixel_order = ["B", "G", "R"]

    camera_calibration = {
        "gain": gain_map, "offset": offset_map, "variance": variance_map,
        "rqe": rqe_map, "pixel_order": pixel_order, "pixel_QYs": pixel_QYs,
        "masks": masks_cam,
    }

    dye_to_colours: Dict[str, List[RGBTuple]] = {}
    for colour, dye in colour_to_dye.items():
        dye_to_colours.setdefault(dye, []).append(colour)

    x0y0, n_photons = {}, {}
    dye_efficiency_rows, avg_wavelengths, gt_rows = [], [], []
    for dye, colours_for_dye in dye_to_colours.items():
        pos_chunks, on_chunks = [], []
        for colour in colours_for_dye:
            pos_px = positions_by_colour.get(colour, np.zeros((2, 0)))
            if pos_px.shape[1] == 0:
                continue
            pos_nm = pos_px * pixel_size_nm
            on_state = blinking_state_schedule(
                pos_px.shape[1], n_frames, on_rate, off_rate, modality,
                bleach_after_cycles, rng,
            )
            pos_chunks.append(pos_nm)
            on_chunks.append(on_state)
            for k in range(pos_px.shape[1]):
                gt_rows.append({
                    "dye": dye, "colour": "#%02x%02x%02x" % colour,
                    "xc_nm": pos_nm[0, k], "yc_nm": pos_nm[1, k],
                    "n_frames_on": int(on_state[:, k].sum()),
                })
        if not pos_chunks:
            continue
        combined_pos_nm = np.concatenate(pos_chunks, axis=1)
        combined_on = np.concatenate(on_chunks, axis=1)
        track = build_x0y0_track(combined_pos_nm, combined_on)
        if drift_nm is not None:
            track = track + np.asarray(drift_nm)[:, :, None]
        x0y0[dye] = track
        n_photons[dye] = per_frame_photon_budget(n_frames, photon_range, rng)
        avg_wl, fracs = spectral.get_pixel_fractions_dye_and_filters(
            dyes=[dye], filters=None, wavelength=wl, pixel_QYs=pixel_QYs, normalized=True,
        )
        dye_efficiency_rows.append(fracs)
        avg_wavelengths.append(float(avg_wl))

    if not x0y0:
        breakdown = "; ".join(
            f"{colour_to_dye.get(c, '?')} (#{c[0]:02x}{c[1]:02x}{c[2]:02x}): "
            f"mask area {mask_area_um2.get(c, 0.0):.2f} µm²"
            for c in colour_to_dye
        )
        raise ValueError(
            f"No candidate localisations were sampled — {breakdown}. This colour's Otsu "
            "mask is empty (zero pixels), so there is nowhere in the pattern image to "
            "place it — check the image and the colour→dye assignment, or try a "
            "different pattern."
        )

    dye_pixel_efficiency = np.array(dye_efficiency_rows)
    average_emission_wavelength = float(np.mean(avg_wavelengths))

    sim = MultiC_Sim_Funcs_Refactored(camera=camera, pixel_size=pixel_size_um, mosaic_unit=cam_cfg.mosaic_unit)
    bayer_stack, _, _ = sim.gen_camera_image_stack(
        camera_calibration=camera_calibration,
        wavelength=wl,
        average_emission_wavelengths=average_emission_wavelength,
        dye_pixel_efficiency=dye_pixel_efficiency,
        n_photons=n_photons,
        x0y0=x0y0,
        background_photons=background_photons,
        NA=na,
    )

    ground_truth = pd.DataFrame(gt_rows)
    return bayer_stack, ground_truth, width, height, average_emission_wavelength
