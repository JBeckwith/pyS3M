"""
reorganise_calibration.py
=========================
Reorganises raw Ximea camera calibration data into the directory layout
expected by CalibrationFunctions.calibrate_multicolour_camera.

Source layout (one exposure-time folder, e.g. 50_ms/):
    50_ms/
        R/
            DiaLamp_200/
                DiaLamp_200_MMStack_Default_1.ome.tif
                DiaLamp_200_MMStack_Default_2.ome.tif
                ...
            DiaLamp_400/
                DiaLamp_400_MMStack_Default_1.ome.tif
                ...
        G/  (same sub-structure)
        B/  (same sub-structure)
        dark/
            dark_MMStack_Default_1.ome.tif
            ...

Target layout:
    <dest>/
        R/
            Intensity_01_MMStack_Default_1.ome.tif   (was DiaLamp_200/DiaLamp_200_MMStack_Default_1.ome.tif)
            Intensity_02_MMStack_Default_1.ome.tif   (was DiaLamp_400/...)
            ...
        G/
            Intensity_01_MMStack_Default_1.ome.tif
            ...
        B/
            ...
        dark/
            Intensity_01_MMStack_Default_1.ome.tif   (was dark/dark_MMStack_Default_1.ome.tif)

Usage
-----
    python reorganise_calibration.py \\
        --source /path/to/50_ms \\
        --dest   /path/to/calibration_organised

By default the script runs in **dry-run** mode and only prints what it would
copy.  Pass --execute to actually copy the files.
"""

import argparse
import re
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def dial_lamp_value(folder_name: str) -> int:
    """Return the numeric part of a DiaLamp_XXX folder name."""
    m = re.match(r"DiaLamp_(\d+)$", folder_name)
    if m is None:
        raise ValueError(f"Unexpected folder name: {folder_name!r}")
    return int(m.group(1))


def reorganise_colour_channel(
    src_channel_dir: Path,
    dst_channel_dir: Path,
    dry_run: bool,
) -> None:
    """
    Reorganise one colour channel (R, G, or B).

    Finds all DiaLamp_XXX sub-directories, sorts them numerically by lamp
    intensity, then copies every *.ome.tif inside each sub-directory to the
    destination directory with a new name:
        Intensity_<ZZ>_MMStack_Default_<N>.ome.tif
    where ZZ is the zero-padded index (01, 02, …) and N is the original
    frame index.
    """
    # Collect DiaLamp subdirs
    subdirs = sorted(
        [d for d in src_channel_dir.iterdir()
         if d.is_dir() and re.match(r"DiaLamp_\d+$", d.name)],
        key=lambda d: dial_lamp_value(d.name),
    )

    if not subdirs:
        print(f"  [WARNING] No DiaLamp_* subdirs found in {src_channel_dir}")
        return

    print(f"  {src_channel_dir.name}: {len(subdirs)} intensity levels "
          f"({', '.join(d.name for d in subdirs)})")

    for idx, subdir in enumerate(subdirs, start=1):
        intensity_label = f"Intensity_{idx:02d}"
        tif_files = sorted(
            subdir.glob("*.ome.tif"),
            key=lambda f: _frame_sort_key(f.name),
        )
        if not tif_files:
            print(f"    [WARNING] No .ome.tif files in {subdir}")
            continue

        for src_file in tif_files:
            # Extract the frame suffix: DiaLamp_200_MMStack_Default_3.ome.tif
            # → MMStack_Default_3.ome.tif
            m = re.match(r"DiaLamp_\d+_(MMStack_Default.*\.ome\.tif)$",
                         src_file.name)
            if m is None:
                # Fallback: keep everything after the first underscore group
                suffix = src_file.name
            else:
                suffix = m.group(1)

            dst_name = f"{intensity_label}_{suffix}"
            dst_file = dst_channel_dir / dst_name

            if dry_run:
                print(f"    [DRY-RUN] {src_file.relative_to(src_channel_dir.parent.parent)}"
                      f"  →  {dst_file.relative_to(dst_channel_dir.parent)}")
            else:
                dst_channel_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                print(f"    Copied → {dst_file.name}")


def reorganise_dark_channel(
    src_dark_dir: Path,
    dst_dark_dir: Path,
    dry_run: bool,
) -> None:
    """
    Reorganise the dark channel.

    The dark directory is flat (no sub-directories).  All dark frames are
    considered a single intensity level (Intensity_01).
    """
    tif_files = sorted(
        src_dark_dir.glob("*.ome.tif"),
        key=lambda f: _frame_sort_key(f.name),
    )

    if not tif_files:
        print(f"  [WARNING] No .ome.tif files in {src_dark_dir}")
        return

    print(f"  dark: {len(tif_files)} frames → Intensity_01")

    for src_file in tif_files:
        # dark_MMStack_Default_3.ome.tif → Intensity_01_MMStack_Default_3.ome.tif
        m = re.match(r"dark_(MMStack_Default.*\.ome\.tif)$", src_file.name)
        if m is None:
            suffix = src_file.name
        else:
            suffix = m.group(1)

        dst_name = f"Intensity_01_{suffix}"
        dst_file = dst_dark_dir / dst_name

        if dry_run:
            print(f"    [DRY-RUN] dark/{src_file.name}  →  dark/{dst_name}")
        else:
            dst_dark_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            print(f"    Copied → {dst_file.name}")


def _frame_sort_key(filename: str) -> int:
    """Sort .ome.tif files by their trailing integer index."""
    m = re.search(r"_(\d+)\.ome\.tif$", filename)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reorganise Ximea calibration data for CalibrationFunctions."
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to the source exposure-time folder (e.g. /path/to/50_ms).",
    )
    parser.add_argument(
        "--dest", required=True,
        help="Path to the destination folder (will be created if absent).",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually copy files.  Without this flag the script is a dry run.",
    )
    args = parser.parse_args()

    src = Path(args.source)
    dst = Path(args.dest)
    dry_run = not args.execute

    if not src.is_dir():
        raise SystemExit(f"Source directory not found: {src}")

    mode = "DRY-RUN (pass --execute to copy)" if dry_run else "EXECUTE"
    print(f"Mode : {mode}")
    print(f"Source : {src}")
    print(f"Dest   : {dst}")
    print()

    # Colour channels
    for channel in ("R", "G", "B"):
        src_ch = src / channel
        dst_ch = dst / channel
        if not src_ch.is_dir():
            print(f"[WARNING] Channel directory not found, skipping: {src_ch}")
            continue
        print(f"Processing channel: {channel}")
        reorganise_colour_channel(src_ch, dst_ch, dry_run)
        print()

    # Dark channel
    src_dark = src / "dark"
    dst_dark = dst / "dark"
    if src_dark.is_dir():
        print("Processing channel: dark")
        reorganise_dark_channel(src_dark, dst_dark, dry_run)
        print()
    else:
        print(f"[WARNING] Dark directory not found, skipping: {src_dark}")

    if dry_run:
        print("Dry run complete.  Run with --execute to copy files.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
