#!/usr/bin/env python3
"""
Video to V3 Pose Converter for DuoSign
======================================

Uses MediaPipe 0.10+ Tasks API to extract poses and convert to V3 JSON.

VRM Humanoid Bone Names:
- spine, chest, neck, head
- leftUpperArm, leftLowerArm, leftHand
- rightUpperArm, rightLowerArm, rightHand
"""

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from scipy.spatial.transform import Rotation as R
import argparse
import sys

# MediaPipe 0.10+ Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Pose landmark indices (MediaPipe 33-point model)
class PL:  # Pose Landmarks
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_HIP = 23
    RIGHT_HIP = 24


def extract_landmarks(video_path: Path, model_path: Path) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Extract pose landmarks from video using MediaPipe Tasks API."""
    
    # Check model exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}\nDownload from: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    all_landmarks = []
    all_confidence = []
    
    # Create pose landmarker
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        output_segmentation_masks=False
    )
    
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to MediaPipe Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            # Detect
            result = landmarker.detect(mp_image)
            
            # Extract landmarks
            frame_lm = np.full((33, 3), np.nan, dtype=np.float32)
            frame_conf = np.full((33,), 0.0, dtype=np.float32)
            
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                for i, lm in enumerate(result.pose_landmarks[0]):
                    frame_lm[i] = [lm.x, lm.y, lm.z]
                    frame_conf[i] = lm.visibility if hasattr(lm, 'visibility') else 1.0
            
            all_landmarks.append(frame_lm)
            all_confidence.append(frame_conf)
            frame_idx += 1
    
    cap.release()
    
    if not all_landmarks:
        raise RuntimeError(f"No frames extracted from: {video_path}")
    
    return (
        np.stack(all_landmarks, axis=0),
        np.stack(all_confidence, axis=0),
        fps,
        len(all_landmarks)
    )


def compute_quaternion(parent: np.ndarray, child: np.ndarray, ref: np.ndarray = None) -> np.ndarray:
    """Compute rotation quaternion from reference axis to bone direction."""
    if ref is None:
        ref = np.array([0, -1, 0])
    
    bone = child - parent
    bone_len = np.linalg.norm(bone)
    
    if bone_len < 1e-6:
        return np.array([0, 0, 0, 1])
    
    bone_dir = bone / bone_len
    ref_dir = ref / np.linalg.norm(ref)
    
    dot = np.clip(np.dot(ref_dir, bone_dir), -1, 1)
    
    if dot > 0.9999:
        return np.array([0, 0, 0, 1])
    if dot < -0.9999:
        return np.array([1, 0, 0, 0])
    
    axis = np.cross(ref_dir, bone_dir)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(dot)
    
    rot = R.from_rotvec(axis * angle)
    return rot.as_quat()


def frame_to_quaternions(lm: np.ndarray, conf: np.ndarray) -> Dict:
    """Convert frame landmarks to VRM bone quaternions."""
    
    def safe_quat(p_idx: int, c_idx: int) -> Tuple[List, float]:
        p, c = lm[p_idx], lm[c_idx]
        if np.isnan(p).any() or np.isnan(c).any():
            return [0.0, 0.0, 0.0, 1.0], 0.0
        q = compute_quaternion(p, c)
        cf = min(conf[p_idx], conf[c_idx])
        return q.tolist(), float(cf) if not np.isnan(cf) else 0.0
    
    rots = {}
    confs = {}
    vels = {}
    
    # Spine
    lh, rh = lm[PL.LEFT_HIP], lm[PL.RIGHT_HIP]
    ls, rs = lm[PL.LEFT_SHOULDER], lm[PL.RIGHT_SHOULDER]
    if not (np.isnan(lh).any() or np.isnan(rh).any() or np.isnan(ls).any() or np.isnan(rs).any()):
        hip_c = (lh + rh) / 2
        shoulder_c = (ls + rs) / 2
        q = compute_quaternion(hip_c, shoulder_c, np.array([0, 1, 0]))
        rots["spine"] = q.tolist()
        confs["spine"] = 0.8
    else:
        rots["spine"] = [0.0, 0.0, 0.0, 1.0]
        confs["spine"] = 0.0
    
    # Neck
    nose = lm[PL.NOSE]
    if not np.isnan(nose).any() and not np.isnan(ls).any() and not np.isnan(rs).any():
        shoulder_c = (ls + rs) / 2
        q = compute_quaternion(shoulder_c, nose, np.array([0, 1, 0]))
        rots["neck"] = q.tolist()
        confs["neck"] = float(conf[PL.NOSE])
    else:
        rots["neck"] = [0.0, 0.0, 0.0, 1.0]
        confs["neck"] = 0.0
    
    # Head
    rots["head"] = [0.0, 0.0, 0.0, 1.0]
    confs["head"] = 0.5
    
    # Arms
    for side, sh, el, wr, idx in [
        ("left", PL.LEFT_SHOULDER, PL.LEFT_ELBOW, PL.LEFT_WRIST, PL.LEFT_INDEX),
        ("right", PL.RIGHT_SHOULDER, PL.RIGHT_ELBOW, PL.RIGHT_WRIST, PL.RIGHT_INDEX)
    ]:
        q, c = safe_quat(sh, el)
        rots[f"{side}UpperArm"] = q
        confs[f"{side}UpperArm"] = c
        
        q, c = safe_quat(el, wr)
        rots[f"{side}LowerArm"] = q
        confs[f"{side}LowerArm"] = c
        
        q, c = safe_quat(wr, idx)
        rots[f"{side}Hand"] = q
        confs[f"{side}Hand"] = c
    
    # Velocities (zero baseline)
    for bone in rots:
        vels[bone] = 0.0
    
    return {"rotations": rots, "velocities": vels, "confidences": confs}


def convert_video(video_path: Path, output_path: Path, model_path: Path, verbose: bool = True) -> Dict:
    """Convert video to V3 pose format."""
    import time
    start = time.time()
    
    if verbose:
        print(f"  {video_path.name}...", end=" ", flush=True)
    
    # Extract
    lm, conf, fps, nframes = extract_landmarks(video_path, model_path)
    
    # Convert frames
    frames = [frame_to_quaternions(lm[t], conf[t]) for t in range(nframes)]
    
    # Build output
    output = {
        "format_version": "3.0-quaternion",
        "fps": float(fps),
        "frame_count": nframes,
        "source_video": video_path.name,
        "frames": frames,
        "skeleton_info": {
            "source": "mediapipe",
            "normalization": {"original_shoulder_width": 0.3, "scale_factor": 1.0, "reference_scale": 1.0},
            "bone_count": len(frames[0]["rotations"]) if frames else 0
        },
        "filter_config": {"min_cutoff": 1.0, "beta": 0.007, "d_cutoff": 1.0},
        "metadata": {"converted_at": datetime.now().isoformat(), "converter_version": "3.1.0"}
    }
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, separators=(',', ':'))
    
    elapsed = time.time() - start
    size = output_path.stat().st_size / 1024
    
    if verbose:
        print(f"✓ {nframes}f, {size:.1f}KB, {elapsed:.1f}s")
    
    return {"video": video_path.name, "frames": nframes, "size_kb": round(size, 2), "time_sec": round(elapsed, 2)}


def main():
    parser = argparse.ArgumentParser(description="Extract poses → V3")
    parser.add_argument("--video_dir", "-v", required=True, help="Video directory")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory")
    parser.add_argument("--model", "-m", default="pose_landmarker_full.task", help="Model path")
    parser.add_argument("--num_videos", "-n", type=int, default=5, help="Videos to process")
    parser.add_argument("--frontend_dir", "-f", default=None, help="Copy to frontend")
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model)
    
    if not video_dir.exists():
        print(f"Error: {video_dir} not found")
        sys.exit(1)
    
    videos = sorted(video_dir.glob("*.mp4"))[:args.num_videos]
    
    print(f"\n{'='*50}")
    print(f"DuoSign Pose Extractor")
    print(f"{'='*50}")
    print(f"Videos: {len(videos)} | Model: {model_path.name}")
    print(f"{'='*50}\n")
    
    stats = []
    for v in videos:
        try:
            out = output_dir / f"{v.stem}.json"
            stats.append(convert_video(v, out, model_path))
        except Exception as e:
            print(f"  {v.name}: ✗ {e}")
    
    print(f"\n{'='*50}")
    print(f"Done! {len(stats)}/{len(videos)} converted")
    if stats:
        print(f"Frames: {sum(s['frames'] for s in stats)} | Size: {sum(s['size_kb'] for s in stats):.1f}KB")
    
    if args.frontend_dir:
        import shutil
        fd = Path(args.frontend_dir)
        fd.mkdir(parents=True, exist_ok=True)
        for f in output_dir.glob("*.json"):
            if not f.name.startswith("_"):
                shutil.copy2(f, fd / f.name)
        print(f"Copied to: {fd}")
    
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
