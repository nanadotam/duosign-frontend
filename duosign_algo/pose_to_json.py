#!/usr/bin/env python3
"""
================================================================================
Pose to JSON Converter (Kalidokit-Compatible)
================================================================================

Converts .pose files (NumPy compressed archives) to JSON format for use
in JavaScript/TypeScript applications.

This script is designed for Kalidokit-compatible pose files with 543 landmarks.

Usage:
    python pose_to_json_kalidokit.py --input_dir ./poses_kalidokit --output_dir ../../../public/poses

Author: Nana Kwaku Amoako
================================================================================
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from datetime import datetime


# Kalidokit-compatible landmark layout
LANDMARK_RANGES = {
    "pose": (0, 33),
    "face": (33, 501),
    "left_hand": (501, 522),
    "right_hand": (522, 543),
}

TOTAL_LANDMARKS = 543


def load_pose_file(pose_path: Path) -> Dict[str, Any]:
    """Load a .pose file and return its contents."""
    data = np.load(str(pose_path), allow_pickle=True)
    return {key: data[key] for key in data.keys()}


def convert_pose_to_json(pose_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert pose data to JSON-serializable format.

    Optimizations:
    - Rounds coordinates to 6 decimal places (sufficient precision)
    - Converts NaN to null for JSON compatibility
    - Includes metadata for frontend validation
    """
    landmarks = pose_data["landmarks"]  # Shape: (T, 543, 3)
    confidence = pose_data["confidence"]  # Shape: (T, 543)

    frame_count = int(pose_data["frame_count"])
    fps = float(pose_data["fps"])
    source_video = str(pose_data.get("source_video", "unknown"))

    # Convert landmarks to list format with NaN handling
    landmarks_list = []
    confidence_list = []

    for frame_idx in range(frame_count):
        frame_landmarks = []
        frame_confidence = []

        for lm_idx in range(landmarks.shape[1]):
            lm = landmarks[frame_idx, lm_idx]
            conf = confidence[frame_idx, lm_idx]

            # Handle NaN values
            if np.isnan(lm).any():
                frame_landmarks.append(None)
            else:
                # Round to 6 decimal places for reasonable file size
                frame_landmarks.append([
                    round(float(lm[0]), 6),
                    round(float(lm[1]), 6),
                    round(float(lm[2]), 6)
                ])

            if np.isnan(conf):
                frame_confidence.append(None)
            else:
                frame_confidence.append(round(float(conf), 4))

        landmarks_list.append(frame_landmarks)
        confidence_list.append(frame_confidence)

    return {
        "landmarks": landmarks_list,
        "confidence": confidence_list,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "source_video": source_video,
        "format_version": "2.0-kalidokit",
        "kalidokit_compatible": True,
        "landmark_layout": {k: list(v) for k, v in LANDMARK_RANGES.items()},
        "total_landmarks": TOTAL_LANDMARKS,
    }


def convert_all_poses(input_dir: Path, output_dir: Path, verbose: bool = True) -> Dict[str, Any]:
    """
    Convert all .pose files in input_dir to JSON files in output_dir.

    Returns statistics about the conversion.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pose_files = sorted(input_dir.glob("*.pose"))

    stats = {
        "total": len(pose_files),
        "converted": 0,
        "failed": 0,
        "files": [],
        "errors": []
    }

    if verbose:
        print(f"Converting {len(pose_files)} pose files to JSON...")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print("-" * 60)

    for pose_file in pose_files:
        try:
            # Load pose data
            pose_data = load_pose_file(pose_file)

            # Convert to JSON format
            json_data = convert_pose_to_json(pose_data)

            # Save as JSON
            output_file = output_dir / f"{pose_file.stem}.json"
            with open(output_file, 'w') as f:
                json.dump(json_data, f)

            file_size = output_file.stat().st_size / 1024  # KB

            stats["converted"] += 1
            stats["files"].append({
                "name": pose_file.stem,
                "frames": json_data["frame_count"],
                "size_kb": round(file_size, 2)
            })

            if verbose:
                print(f"  ✓ {pose_file.stem}: {json_data['frame_count']} frames, {file_size:.1f} KB")

        except Exception as e:
            stats["failed"] += 1
            stats["errors"].append({
                "file": str(pose_file),
                "error": str(e)
            })
            if verbose:
                print(f"  ✗ {pose_file.stem}: {e}")

    if verbose:
        print("-" * 60)
        print(f"Conversion complete: {stats['converted']}/{stats['total']} files")
        if stats["failed"] > 0:
            print(f"Failed: {stats['failed']} files")

    # Save conversion summary
    summary_path = output_dir / "_conversion_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            **stats,
            "timestamp": datetime.now().isoformat(),
            "format": "kalidokit-compatible",
            "landmarks_per_frame": TOTAL_LANDMARKS,
        }, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert Kalidokit-compatible .pose files to JSON"
    )
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        required=True,
        help="Directory containing .pose files"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Directory for output JSON files"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output"
    )

    args = parser.parse_args()

    stats = convert_all_poses(
        Path(args.input_dir),
        Path(args.output_dir),
        verbose=not args.quiet
    )

    if stats["failed"] > 0:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()
