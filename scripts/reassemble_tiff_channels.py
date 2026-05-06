#!/usr/bin/env python3
"""Reassemble split multi-channel, multi-position TIFFs into ImageJ hyperstacks.

Handles files saved by acquisition software with naming convention:
    <prefix>_T{position}_C{channel}.tif

Each input file is a z-stack. Output is one ImageJ-compatible hyperstack per
position with shape (Z, C, Y, X), or a single file with all positions
(T, Z, C, Y, X).

Usage:
    python reassemble_tiff_channels.py /path/to/tiff/folder
    python reassemble_tiff_channels.py /path/to/tiff/folder --z-spacing 0.5
    python reassemble_tiff_channels.py /path/to/tiff/folder --per-position
    python reassemble_tiff_channels.py /path/to/tiff/folder --pixel-size 0.069
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile



def find_and_group_tiffs(input_dir: Path) -> dict:
    """Find TIFF files and group by (position, channel).

    Args:
        input_dir: Directory containing the split TIFF files.

    Returns:
        Nested dict: {position_idx: {channel_idx: file_path}}
    """
    pattern = re.compile(r'_T(\d+)_C(\d+)\.tiff?$', re.IGNORECASE)

    groups = defaultdict(dict)
    for f in sorted(input_dir.iterdir()):
        if not f.is_file():
            continue
        m = pattern.search(f.name)
        if m:
            t_idx = int(m.group(1))
            c_idx = int(m.group(2))
            groups[t_idx][c_idx] = f

    return dict(groups)


def reassemble(
    input_dir: str,
    output_path: str = None,
    z_spacing_um: float = 0.5,
    pixel_size_um: float = 0.11,
    per_position: bool = False,
):
    """Reassemble split TIFFs into ImageJ hyperstack(s).

    Args:
        input_dir: Directory containing the split TIFF files.
        output_path: Output file path. If None, uses input_dir/reassembled.tif.
            When per_position=True, this is used as a stem (e.g. stem_T0.tif).
        z_spacing_um: Z-plane spacing in microns (default: 0.5).
        pixel_size_um: XY pixel size in microns (default: 0.069).
        per_position: If True, write one file per position instead of a single
            combined file.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    groups = find_and_group_tiffs(input_dir)
    if not groups:
        print(f"Error: no files matching *_T{{n}}_C{{n}}.tif found in {input_dir}")
        sys.exit(1)

    positions = sorted(groups.keys())
    # Determine channel count from first position
    channels = sorted(groups[positions[0]].keys())
    n_positions = len(positions)
    n_channels = len(channels)

    print(f"Found {n_positions} position(s), {n_channels} channel(s)")
    for t in positions:
        chans = sorted(groups[t].keys())
        print(f"  T{t}: channels {chans}")
        for c in chans:
            print(f"    C{c}: {groups[t][c].name}")

    # Read first file to get shape and dtype
    first_file = groups[positions[0]][channels[0]]
    sample = tifffile.imread(str(first_file))
    if sample.ndim == 2:
        # Single z-plane
        n_z, h, w = 1, sample.shape[0], sample.shape[1]
    elif sample.ndim == 3:
        n_z, h, w = sample.shape
    else:
        print(f"Error: unexpected shape {sample.shape} in {first_file.name}")
        sys.exit(1)

    dtype = sample.dtype
    print(f"\nStack dimensions: {n_z} Z x {h} H x {w} W, dtype={dtype}")
    print(f"Z spacing: {z_spacing_um} um, pixel size: {pixel_size_um} um")

    # Resolution in pixels per micron (what ImageJ expects)
    resolution = (1.0 / pixel_size_um, 1.0 / pixel_size_um)

    if per_position:
        # Write one file per position
        if output_path is None:
            stem = input_dir / "reassembled"
        else:
            stem = Path(output_path).with_suffix("")

        for t in positions:
            # Shape: (Z, C, Y, X) — ImageJ requires TZCYX order
            volume = np.zeros((n_z, n_channels, h, w), dtype=dtype)

            for ci, c in enumerate(channels):
                if c not in groups[t]:
                    print(f"  Warning: T{t} missing channel C{c}, filling with zeros")
                    continue
                stack = tifffile.imread(str(groups[t][c]))
                if stack.ndim == 2:
                    stack = stack[np.newaxis, :, :]
                volume[:, ci] = stack

            out_file = f"{stem}_T{t}.tif"
            tifffile.imwrite(
                out_file,
                volume,
                imagej=True,
                resolution=resolution,
                metadata={
                    'spacing': z_spacing_um,
                    'unit': 'um',
                    'axes': 'ZCYX',
                },
            )
            print(f"  Wrote T{t}: {out_file}  shape={volume.shape}")

        print(f"\nDone. {n_positions} files written.")

    else:
        # Single combined file: shape (T, Z, C, Y, X) — ImageJ requires TZCYX order
        if output_path is None:
            output_path = str(input_dir / "reassembled.tif")

        volume = np.zeros((n_positions, n_z, n_channels, h, w), dtype=dtype)

        for ti, t in enumerate(positions):
            for ci, c in enumerate(channels):
                if c not in groups[t]:
                    print(f"  Warning: T{t} missing channel C{c}, filling with zeros")
                    continue
                stack = tifffile.imread(str(groups[t][c]))
                if stack.ndim == 2:
                    stack = stack[np.newaxis, :, :]
                volume[ti, :, ci] = stack

        tifffile.imwrite(
            output_path,
            volume,
            imagej=True,
            resolution=resolution,
            metadata={
                'spacing': z_spacing_um,
                'unit': 'um',
                'axes': 'TZCYX',
            },
        )
        print(f"\nWrote: {output_path}")
        print(f"Shape: {volume.shape} (T={n_positions}, Z={n_z}, "
              f"C={n_channels}, Y={h}, X={w})")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Reassemble split TIFF channels/positions into ImageJ hyperstacks."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing the split TIFF files (*_T{n}_C{n}.tif)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path (default: input_dir/reassembled.tif)"
    )
    parser.add_argument(
        "--z-spacing",
        type=float,
        default=0.5,
        help="Z-plane spacing in microns (default: 0.5)"
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.069,
        help="XY pixel size in microns (default: 0.069)"
    )
    parser.add_argument(
        "--per-position",
        action="store_true",
        help="Write one file per position instead of a single combined file"
    )

    args = parser.parse_args()

    reassemble(
        input_dir=args.input_dir,
        output_path=args.output,
        z_spacing_um=args.z_spacing,
        pixel_size_um=args.pixel_size,
        per_position=args.per_position,
    )


if __name__ == "__main__":
    main()
