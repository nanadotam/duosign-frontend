#!/usr/bin/env python3
"""
Complete Pose V3 Conversion Pipeline
====================================

This module orchestrates the complete conversion from .pose files to .pose-v3 format,
combining all processing steps: filtering, normalization, quaternion conversion.

Pipeline stages:
    1. Load .pose file (landmarks + confidence)
    2. Apply 1€ filter for temporal smoothing
    3. Normalize skeleton scale
    4. Compute quaternions + velocities
    5. Save as .pose-v3 JSON

Author: Nana Kwaku Amoako
Date: 2026-01-31
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .filters import LandmarkFilter
from .quaternion_solver import process_landmark_sequence
from .skeleton_normalizer import normalize_landmarks


def convert_pose_to_v3(
    pose_path: Path,
    output_path: Path,
    filter_config: Optional[Dict[str, float]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convert a .pose file to .pose-v3 format with quaternions.
    
    This is the main entry point for the conversion pipeline. It orchestrates
    all processing steps and produces a quaternion-native pose file ready for
    frontend consumption.
    
    Pipeline:
    1. Load .pose file (landmarks + confidence)
    2. Apply 1€ filter to landmarks (adaptive temporal smoothing)
    3. Normalize skeleton scale (shoulder-width based)
    4. Compute quaternions + velocities (scipy-based rotation)
    5. Save as .pose-v3 JSON (compact, quaternion-native format)
    
    Args:
        pose_path (Path): Input .pose file path
        output_path (Path): Output .pose-v3 file path
        filter_config (Optional[Dict]): 1€ filter parameters
            - min_cutoff (float): Jitter reduction (default: 1.0)
            - beta (float): Lag reduction (default: 0.007)
            - d_cutoff (float): Derivative smoothing (default: 1.0)
        verbose (bool): Print progress messages (default: True)
    
    Returns:
        Dict[str, Any]: Conversion statistics
            - input_file: Input file path
            - output_file: Output file path
            - frame_count: Number of frames processed
            - fps: Frame rate
            - output_size_kb: Output file size in KB
            - processing_time_sec: Total processing time
    
    Raises:
        FileNotFoundError: If pose_path does not exist
        ValueError: If pose file has invalid format
    
    Example:
        >>> from pathlib import Path
        >>> 
        >>> # Convert with default settings
        >>> stats = convert_pose_to_v3(
        ...     pose_path=Path("public/poses/hello.pose"),
        ...     output_path=Path("public/poses_v3/hello.json")
        ... )
        >>> 
        >>> print(f"Converted {stats['frame_count']} frames")
        >>> print(f"Output size: {stats['output_size_kb']} KB")
        >>> 
        >>> # Convert with custom filter settings (for fingerspelling)
        >>> stats = convert_pose_to_v3(
        ...     pose_path=Path("public/poses/alphabet_a.pose"),
        ...     output_path=Path("public/poses_v3/alphabet_a.json"),
        ...     filter_config={
        ...         "min_cutoff": 0.5,  # More smoothing
        ...         "beta": 0.001       # Less lag reduction
        ...     }
        ... )
    """
    import time
    start_time = time.time()
    
    # Default filter configuration (balanced for dynamic signs)
    if filter_config is None:
        filter_config = {
            "min_cutoff": 1.0,
            "beta": 0.007,
            "d_cutoff": 1.0
        }
    
    # Validate input file exists
    pose_path = Path(pose_path)
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_path}")
    
    # Step 1: Load .pose file
    if verbose:
        print(f"[1/5] Loading {pose_path.name}...")
    
    try:
        data = np.load(str(pose_path), allow_pickle=True)
        landmarks = data["landmarks"]  # Shape: (T, 523, 3)
        confidence = data["confidence"]  # Shape: (T, 523)
        fps = float(data["fps"])
        frame_count = int(data["frame_count"])
        source_video = str(data.get("source_video", "unknown"))
    except Exception as e:
        raise ValueError(f"Failed to load pose file: {e}")
    
    # Validate data shapes
    if landmarks.ndim != 3 or landmarks.shape[1] != 523 or landmarks.shape[2] != 3:
        raise ValueError(f"Invalid landmarks shape: {landmarks.shape}, expected (T, 523, 3)")
    if confidence.shape != (landmarks.shape[0], 523):
        raise ValueError(f"Invalid confidence shape: {confidence.shape}")
    
    # Step 2: Apply 1€ filter
    if verbose:
        print(f"[2/5] Applying 1€ filter (min_cutoff={filter_config['min_cutoff']}, beta={filter_config['beta']})...")
    
    landmark_filter = LandmarkFilter(fps=fps, **filter_config)
    filtered_landmarks = landmark_filter.filter_landmarks(landmarks, confidence)
    
    # Step 3: Normalize skeleton scale
    if verbose:
        print("[3/5] Normalizing skeleton scale...")
    
    normalized_landmarks, norm_metadata = normalize_landmarks(
        filtered_landmarks,
        reference_scale=1.0
    )
    
    if verbose:
        print(f"      Original shoulder width: {norm_metadata['original_shoulder_width']:.3f}")
        print(f"      Scale factor: {norm_metadata['scale_factor']:.3f}")
    
    # Step 4: Compute quaternions and velocities
    if verbose:
        print("[4/5] Computing quaternions and velocities...")
    
    rotations_seq, velocities_seq, confidences_seq = process_landmark_sequence(
        normalized_landmarks,
        confidence,
        fps
    )
    
    # Step 5: Build .pose-v3 format
    if verbose:
        print("[5/5] Building output format and saving...")
    
    frames = []
    for t in range(frame_count):
        frame = {
            "rotations": {
                bone: quat.tolist()  # Convert numpy array to list for JSON
                for bone, quat in rotations_seq[t].items()
            },
            "velocities": {
                bone: float(vel)  # Convert numpy float to Python float
                for bone, vel in velocities_seq[t].items()
            },
            "confidences": {
                bone: float(conf)  # Convert numpy float to Python float
                for bone, conf in confidences_seq[t].items()
            }
        }
        frames.append(frame)
    
    output_data = {
        "format_version": "3.0-quaternion",
        "fps": float(fps),  # Convert to Python float
        "frame_count": int(frame_count),  # Convert to Python int
        "source_video": source_video,
        "frames": frames,
        "skeleton_info": {
            "source": "mediapipe",
            "normalization": {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in norm_metadata.items()
            },
            "bone_count": len(rotations_seq[0])
        },
        "filter_config": {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in filter_config.items()
        },
        "metadata": {
            "converted_at": datetime.now().isoformat(),
            "converter_version": "3.0.0"
        }
    }
    
    # Save as JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=None, separators=(',', ':'))  # Compact JSON
    
    # Compute statistics
    processing_time = time.time() - start_time
    file_size_kb = output_path.stat().st_size / 1024
    
    if verbose:
        print(f"\n✓ Conversion complete!")
        print(f"  Output: {output_path}")
        print(f"  Size: {file_size_kb:.2f} KB")
        print(f"  Processing time: {processing_time:.2f}s")
    
    return {
        "input_file": str(pose_path),
        "output_file": str(output_path),
        "frame_count": frame_count,
        "fps": fps,
        "output_size_kb": round(file_size_kb, 2),
        "processing_time_sec": round(processing_time, 2)
    }


def batch_convert_poses(
    input_dir: Path,
    output_dir: Path,
    filter_config: Optional[Dict[str, float]] = None,
    skip_existing: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Batch convert all .pose files in a directory to .pose-v3 format.
    
    Args:
        input_dir (Path): Directory containing .pose files
        output_dir (Path): Directory for output .pose-v3 files
        filter_config (Optional[Dict]): 1€ filter parameters
        skip_existing (bool): Skip files that already exist in output_dir
        verbose (bool): Print progress messages
    
    Returns:
        Dict[str, Any]: Batch conversion statistics
            - total: Total number of .pose files found
            - converted: Number successfully converted
            - skipped: Number skipped (already exist)
            - failed: Number that failed
            - files: List of per-file statistics
            - errors: List of error messages
    
    Example:
        >>> from pathlib import Path
        >>> 
        >>> # Batch convert all poses
        >>> stats = batch_convert_poses(
        ...     input_dir=Path("pose_extraction/poses_kalidokit"),
        ...     output_dir=Path("pose_processing/poses_v3")
        ... )
        >>> 
        >>> print(f"Converted {stats['converted']}/{stats['total']} files")
        >>> print(f"Total size: {sum(f['output_size_kb'] for f in stats['files']):.2f} KB")
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find all .pose files
    pose_files = sorted(input_dir.glob("*.pose"))
    
    stats = {
        "total": len(pose_files),
        "converted": 0,
        "skipped": 0,
        "failed": 0,
        "files": [],
        "errors": []
    }
    
    if verbose:
        print(f"Found {len(pose_files)} .pose files in {input_dir}")
        print(f"Output directory: {output_dir}")
        print("-" * 60)
    
    for pose_file in pose_files:
        output_file = output_dir / f"{pose_file.stem}.json"
        
        # Skip if already exists
        if skip_existing and output_file.exists():
            stats["skipped"] += 1
            if verbose:
                print(f"⊘ {pose_file.stem}: Already exists, skipping")
            continue
        
        try:
            # Convert file
            file_stats = convert_pose_to_v3(
                pose_file,
                output_file,
                filter_config=filter_config,
                verbose=False  # Suppress per-file output in batch mode
            )
            
            stats["converted"] += 1
            stats["files"].append(file_stats)
            
            if verbose:
                print(f"✓ {pose_file.stem}: {file_stats['frame_count']} frames, {file_stats['output_size_kb']} KB")
        
        except Exception as e:
            stats["failed"] += 1
            stats["errors"].append({
                "file": str(pose_file),
                "error": str(e)
            })
            
            if verbose:
                print(f"✗ {pose_file.stem}: {e}")
    
    if verbose:
        print("-" * 60)
        print(f"Batch conversion complete:")
        print(f"  Converted: {stats['converted']}")
        print(f"  Skipped: {stats['skipped']}")
        print(f"  Failed: {stats['failed']}")
        
        if stats["converted"] > 0:
            total_size = sum(f["output_size_kb"] for f in stats["files"])
            print(f"  Total output size: {total_size:.2f} KB")
    
    # Save summary
    summary_path = output_dir / "_conversion_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            **stats,
            "timestamp": datetime.now().isoformat(),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "filter_config": filter_config or {}
        }, f, indent=2)
    
    if verbose:
        print(f"  Summary saved to: {summary_path}")
    
    return stats


# Example usage
if __name__ == "__main__":
    """
    Example: Convert a single pose file with custom settings.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert .pose files to .pose-v3 format with quaternions"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input .pose file or directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output .pose-v3 file or directory"
    )
    parser.add_argument(
        "--min-cutoff",
        type=float,
        default=1.0,
        help="1€ filter min cutoff (default: 1.0)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.007,
        help="1€ filter beta (default: 0.007)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch convert all .pose files in input directory"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output"
    )
    
    args = parser.parse_args()
    
    filter_config = {
        "min_cutoff": args.min_cutoff,
        "beta": args.beta,
        "d_cutoff": 1.0
    }
    
    if args.batch:
        # Batch conversion
        stats = batch_convert_poses(
            Path(args.input),
            Path(args.output),
            filter_config=filter_config,
            verbose=not args.quiet
        )
        
        if stats["failed"] > 0:
            exit(1)
    else:
        # Single file conversion
        try:
            stats = convert_pose_to_v3(
                Path(args.input),
                Path(args.output),
                filter_config=filter_config,
                verbose=not args.quiet
            )
        except Exception as e:
            print(f"Error: {e}")
            exit(1)
