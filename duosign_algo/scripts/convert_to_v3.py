#!/usr/bin/env python3
"""
Batch Conversion Script for .pose to .pose-v3
=============================================

CLI script to batch convert existing .pose files to .pose-v3 format.

Usage:
    python scripts/convert_to_v3.py \\
        --input_dir ./pose_extraction/poses_kalidokit \\
        --output_dir ./pose_processing/poses_v3 \\
        --num_workers 4

Author: Nana Kwaku Amoako
Date: 2026-01-31
"""

import argparse
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pose_processing.pose_v3_converter import convert_pose_to_v3, batch_convert_poses


def convert_single_file_wrapper(
    pose_file: Path,
    output_dir: Path,
    filter_config: dict
) -> dict:
    """
    Wrapper for parallel processing.
    
    Args:
        pose_file: Input .pose file
        output_dir: Output directory
        filter_config: 1€ filter configuration
    
    Returns:
        Dict with conversion statistics or error info
    """
    output_file = output_dir / f"{pose_file.stem}.json"
    
    try:
        stats = convert_pose_to_v3(
            pose_file,
            output_file,
            filter_config=filter_config,
            verbose=False
        )
        return {"status": "success", **stats}
    except Exception as e:
        return {
            "status": "failed",
            "input_file": str(pose_file),
            "error": str(e)
        }


def main():
    """Main entry point for batch conversion script."""
    parser = argparse.ArgumentParser(
        description="Batch convert .pose files to .pose-v3 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all poses with default settings
  python scripts/convert_to_v3.py \\
      --input_dir pose_extraction/poses_kalidokit \\
      --output_dir pose_processing/poses_v3

  # Convert with custom filter settings (for fingerspelling)
  python scripts/convert_to_v3.py \\
      --input_dir pose_extraction/poses_kalidokit \\
      --output_dir pose_processing/poses_v3 \\
      --min-cutoff 0.5 \\
      --beta 0.001

  # Use parallel processing with 8 workers
  python scripts/convert_to_v3.py \\
      --input_dir pose_extraction/poses_kalidokit \\
      --output_dir pose_processing/poses_v3 \\
      --num-workers 8
        """
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
        help="Directory for output .pose-v3 files"
    )
    parser.add_argument(
        "--min-cutoff",
        type=float,
        default=1.0,
        help="1€ filter min cutoff frequency (default: 1.0 Hz)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.007,
        help="1€ filter beta coefficient (default: 0.007)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, use 0 for auto)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip files that already exist (default: True)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output"
    )
    
    args = parser.parse_args()
    
    # Validate directories
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter configuration
    filter_config = {
        "min_cutoff": args.min_cutoff,
        "beta": args.beta,
        "d_cutoff": 1.0
    }
    
    # Determine number of workers
    num_workers = args.num_workers
    if num_workers == 0:
        num_workers = cpu_count()
    
    if not args.quiet:
        print(f"DuoSign Pose V3 Batch Converter")
        print(f"=" * 60)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Filter config: min_cutoff={filter_config['min_cutoff']}, beta={filter_config['beta']}")
        print(f"Workers: {num_workers}")
        print(f"=" * 60)
    
    # Use batch converter (handles parallelization internally if needed)
    stats = batch_convert_poses(
        input_dir,
        output_dir,
        filter_config=filter_config,
        skip_existing=args.skip_existing,
        verbose=not args.quiet
    )
    
    # Exit with error code if any conversions failed
    if stats["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
