"""
Debug script for drift correction frame mismatch issue.

This script adds diagnostic output to understand why apply_validated_fiducial_drift_correction
is dropping all data (0 localizations, 0 frames with fiducials).
"""

import sys
sys.path.insert(0, 'src')
import numpy as np

def diagnose_drift_correction_issue(locs, validated_fiducials):
    """
    Diagnose why drift correction drops all data.

    Parameters
    ----------
    locs : np.recarray
        Full localisation dataset
    validated_fiducials : List[np.recarray]
        List of validated fiducial clusters
    """
    print("=" * 80)
    print("DRIFT CORRECTION DIAGNOSTIC")
    print("=" * 80)

    # Check locs dataset
    print("\n1. Main localisation dataset (locs):")
    print(f"   - Total localisations: {len(locs)}")
    print(f"   - Frame field type: {locs.frame.dtype}")
    print(f"   - Frame range: {locs.frame.min():.1f} to {locs.frame.max():.1f}")
    print(f"   - Unique frames: {len(np.unique(locs.frame))}")
    print(f"   - First 5 frames: {np.unique(locs.frame)[:5]}")
    print(f"   - Last 5 frames: {np.unique(locs.frame)[-5:]}")

    # Check validated_fiducials
    print(f"\n2. Validated fiducials:")
    print(f"   - Number of fiducial clusters: {len(validated_fiducials)}")

    all_fiducial_frames = []
    for i, fiducial_cluster in enumerate(validated_fiducials):
        if len(fiducial_cluster) > 0:
            frames = np.asarray(fiducial_cluster.frame)
            all_fiducial_frames.extend(frames)
            print(f"\n   Cluster {i}:")
            print(f"     - Localisations: {len(fiducial_cluster)}")
            print(f"     - Frame field type: {fiducial_cluster.frame.dtype}")
            print(f"     - Frame range: {frames.min():.1f} to {frames.max():.1f}")
            print(f"     - Unique frames: {len(np.unique(frames))}")
            if len(frames) != len(np.unique(frames)):
                print(f"     - ⚠️ WARNING: Duplicate frames detected!")

    all_fiducial_frames = np.array(all_fiducial_frames)
    print(f"\n3. All fiducial frames combined:")
    print(f"   - Total fiducial localisations: {len(all_fiducial_frames)}")
    print(f"   - Frame range: {all_fiducial_frames.min():.1f} to {all_fiducial_frames.max():.1f}")
    print(f"   - Unique frames: {len(np.unique(all_fiducial_frames))}")

    # Check overlap
    print(f"\n4. Frame overlap analysis:")
    locs_frames = np.unique(locs.frame)
    fiducial_frames = np.unique(all_fiducial_frames)

    # Check for type mismatch
    print(f"   - locs frame dtype: {locs_frames.dtype}")
    print(f"   - fiducial frame dtype: {fiducial_frames.dtype}")

    # Check if fiducial frames are in locs
    frames_in_locs = np.isin(fiducial_frames, locs_frames)
    n_matching = np.sum(frames_in_locs)
    n_total = len(fiducial_frames)

    print(f"   - Fiducial frames that exist in locs: {n_matching}/{n_total}")
    print(f"   - Match percentage: {100.0 * n_matching / n_total:.1f}%")

    if n_matching < n_total:
        print(f"\n   ⚠️ PROBLEM FOUND: {n_total - n_matching} fiducial frames not in locs!")
        missing_frames = fiducial_frames[~frames_in_locs]
        print(f"   - Missing frame examples: {missing_frames[:10]}")

    # Check frame_to_idx mapping
    print(f"\n5. Frame mapping test:")
    unique_frames = np.unique(locs.frame)
    frame_to_idx = {frame: i for i, frame in enumerate(unique_frames)}

    print(f"   - frame_to_idx dictionary size: {len(frame_to_idx)}")
    print(f"   - First 5 entries: {list(frame_to_idx.items())[:5]}")

    # Try mapping fiducial frames
    mapping_errors = []
    for frame in fiducial_frames[:100]:  # Test first 100 frames
        try:
            idx = frame_to_idx[frame]
        except KeyError:
            mapping_errors.append(frame)

    if mapping_errors:
        print(f"\n   ⚠️ MAPPING ERRORS: {len(mapping_errors)} frames cannot be mapped!")
        print(f"   - Error examples: {mapping_errors[:10]}")
        print(f"   - These frames are in fiducials but not in locs")
    else:
        print(f"   ✅ All tested fiducial frames can be mapped")

    # Additional checks
    print(f"\n6. Data type compatibility:")
    print(f"   - Can convert fiducial frames to int? {all_fiducial_frames.astype(int)[:5]}")
    print(f"   - Can convert locs frames to int? {locs_frames.astype(int)[:5]}")

    # Check if casting causes mismatch
    fiducial_frames_int = fiducial_frames.astype(int)
    locs_frames_int = locs_frames.astype(int)
    frames_in_locs_after_cast = np.isin(fiducial_frames_int, locs_frames_int)
    n_matching_after_cast = np.sum(frames_in_locs_after_cast)

    print(f"\n   After int casting:")
    print(f"   - Matching frames: {n_matching_after_cast}/{n_total}")

    if n_matching_after_cast > n_matching:
        print(f"   ⚠️ Type casting improved match! This is likely a type mismatch bug.")

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


# Export the function
if __name__ == "__main__":
    print("This is a diagnostic module.")
    print("Import and call: diagnose_drift_correction_issue(locs, validated_fiducials)")
