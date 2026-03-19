"""
reorganise_calibration.py / notebook cells
===========================================
Reorganises raw Ximea camera calibration data into the directory layout
expected by CalibrationFunctions.calibrate_multicolour_camera.

Source layout (one exposure-time folder, e.g. 50_ms/):
    50_ms/
        R/
            DiaLamp_200/
                DiaLamp_200_MMStack_Default_1.ome.tif ...
            DiaLamp_400/ ...
        G/  B/  (same sub-structure)
        dark/
            dark_MMStack_Default_1.ome.tif ...

Target layout  (files MOVED in-place, no data copied):
    50_ms/
        R/
            Intensity_01_MMStack_Default_1.ome.tif   (was DiaLamp_200/...)
            Intensity_02_MMStack_Default_1.ome.tif   (was DiaLamp_400/...)
            ...
        G/  B/  (same)
        dark/
            dark_MMStack_Default_1.ome.tif  (unchanged — CalibrationFunctions.filesearch filters by "dark")

Paste each ### CELL ### block into a separate notebook cell.
Run Cell 1 (dry run) first, inspect the output, then run Cell 2 to execute.
"""

# =============================================================================
# ### CELL 1 — configuration + dry run ###
# =============================================================================

import re
from pathlib import Path

# ── edit these two lines ─────────────────────────────────────────────────────
SOURCE = Path("/run/user/1000/gvfs/smb-share:server=intelliflash-mgmt-b.ch.private.cam.ac.uk,share=sycamore_asap_server/ASAP_Members_Other_Imaging_Data/JSB/20260211_XimeaCalibration_Test/50_ms")
# SOURCE is also the destination: files are renamed/moved inside the same tree.
# ─────────────────────────────────────────────────────────────────────────────


def _dial_lamp_value(folder_name):
    m = re.match(r"DiaLamp_(\d+)$", folder_name)
    if m is None:
        raise ValueError(f"Unexpected folder name: {folder_name!r}")
    return int(m.group(1))


def _frame_sort_key(filename):
    m = re.search(r"_(\d+)\.ome\.tif$", filename)
    return int(m.group(1)) if m else 0


def _plan_colour_channel(src_channel_dir):
    """Return list of (src_path, dst_path) for one colour channel."""
    subdirs = sorted(
        [d for d in src_channel_dir.iterdir()
         if d.is_dir() and re.match(r"DiaLamp_\d+$", d.name)],
        key=lambda d: _dial_lamp_value(d.name),
    )
    ops = []
    for idx, subdir in enumerate(subdirs, start=1):
        label = f"Intensity_{idx:02d}"
        for src_file in sorted(subdir.glob("*.ome.tif"),
                               key=lambda f: _frame_sort_key(f.name)):
            m = re.match(r"DiaLamp_\d+_(MMStack_Default.*\.ome\.tif)$",
                         src_file.name)
            suffix = m.group(1) if m else src_file.name
            dst_file = src_channel_dir / f"{label}_{suffix}"
            ops.append((src_file, dst_file))
    return ops


def _plan_dark_channel(src_dark_dir):
    """Return list of (src_path, dst_path) for the dark channel.

    CalibrationFunctions.filesearch filters dark files by the string "dark",
    so the filename must keep the dark_ prefix.  The files are already named
    dark_MMStack_Default_N.ome.tif and need no renaming — this returns an
    empty list (nothing to do for the dark channel).
    """
    return []


def build_plan(source):
    plan = {}
    for channel in ("R", "G", "B"):
        ch_dir = source / channel
        if ch_dir.is_dir():
            plan[channel] = _plan_colour_channel(ch_dir)
        else:
            print(f"[WARNING] {ch_dir} not found, skipping")
    dark_dir = source / "dark"
    if dark_dir.is_dir():
        plan["dark"] = _plan_dark_channel(dark_dir)
    else:
        print(f"[WARNING] {dark_dir} not found, skipping")
    return plan


# --- dry run ---
_plan = build_plan(SOURCE)
for channel, ops in _plan.items():
    print(f"\n{channel}: {len(ops)} files")
    for src, dst in ops[:3]:          # show first 3 per channel
        print(f"  {src.parent.name}/{src.name}")
        print(f"    → {dst.name}")
    if len(ops) > 3:
        print(f"  ... and {len(ops) - 3} more")

total = sum(len(v) for v in _plan.values())
print(f"\nTotal: {total} moves planned.  Run Cell 2 to execute.")


# =============================================================================
# ### CELL 2 — execute moves ###
# =============================================================================

# Run this cell only after inspecting Cell 1 output.

for channel, ops in _plan.items():
    print(f"Moving {channel} ({len(ops)} files)...")
    for src, dst in ops:
        src.rename(dst)          # same filesystem → instant rename, no data copy
    # Remove now-empty DiaLamp_* subdirectories
    if channel != "dark":
        ch_dir = SOURCE / channel
        for d in ch_dir.iterdir():
            if d.is_dir() and re.match(r"DiaLamp_\d+$", d.name):
                try:
                    d.rmdir()    # only succeeds if empty
                except OSError:
                    print(f"  [WARNING] Could not remove {d} (not empty?)")
    print(f"  done.")

print("\nAll done.")
