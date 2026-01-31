#!/usr/bin/env python3
"""
Complete End-to-End Pipeline Runner
===================================

This script runs the complete workflow:
1. Extract poses from WLASL videos using MediaPipe
2. Convert .pose files to .pose-v3 format with quaternions
3. Copy to public/poses_v3 for frontend access

Usage:
    python run_complete_pipeline.py --num_videos 5
"""

import sys
import subprocess
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run complete pose processing pipeline")
    parser.add_argument("--num_videos", type=int, default=5, help="Number of videos to process")
    parser.add_argument("--skip_extraction", action="store_true", help="Skip pose extraction (use existing)")
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent
    video_dir = base_dir / "pose_extraction" / "wlasl-processed" / "videos"
    pose_output = base_dir / "pose_extraction" / "poses_kalidokit"
    v3_output = base_dir / "pose_processing" / "poses_v3"
    frontend_output = base_dir.parent / "public" / "poses_v3"
    
    print("=" * 80)
    print("DUOSIGN COMPLETE PIPELINE")
    print("=" * 80)
    
    # Step 1: Extract poses (if not skipped)
    if not args.skip_extraction:
        print(f"\n[1/3] Extracting poses from {args.num_videos} videos...")
        print(f"  Input: {video_dir}")
        print(f"  Output: {pose_output}")
        
        # Get list of videos
        videos = sorted(list(video_dir.glob("*.mp4")))[:args.num_videos]
        print(f"  Found {len(videos)} videos to process")
        
        # Run extraction
        cmd = [
            sys.executable,
            str(base_dir / "pose_extraction" / "wlasl_pose_pipeline.py"),
            "--input_dir", str(video_dir),
            "--output_dir", str(pose_output),
            "--num_workers", "2"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: Pose extraction failed")
            print(result.stderr)
            return 1
        print(f"  ✓ Pose extraction complete")
    else:
        print(f"\n[1/3] Skipping pose extraction (using existing files)")
    
    # Step 2: Convert to V3 format
    print(f"\n[2/3] Converting poses to V3 format...")
    print(f"  Input: {pose_output}")
    print(f"  Output: {v3_output}")
    
    cmd = [
        sys.executable,
        str(base_dir / "scripts" / "convert_to_v3.py"),
        "--input_dir", str(pose_output),
        "--output_dir", str(v3_output),
        "--min-cutoff", "1.0",
        "--beta", "0.007"
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ERROR: V3 conversion failed")
        return 1
    print(f"  ✓ V3 conversion complete")
    
    # Step 3: Copy to frontend
    print(f"\n[3/3] Copying to frontend directory...")
    print(f"  Target: {frontend_output}")
    
    frontend_output.mkdir(parents=True, exist_ok=True)
    
    import shutil
    copied = 0
    for v3_file in v3_output.glob("*.json"):
        shutil.copy2(v3_file, frontend_output / v3_file.name)
        copied += 1
    
    print(f"  ✓ Copied {copied} pose files to frontend")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nPose files available in: {frontend_output}")
    print(f"\nNext steps:")
    print(f"  1. Start API server: cd duosign_algo && uvicorn api.main:app --reload")
    print(f"  2. Start frontend: npm run dev")
    print(f"  3. Test in browser: http://localhost:3000")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
