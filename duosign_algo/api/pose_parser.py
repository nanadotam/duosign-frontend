#!/usr/bin/env python3
"""
Pose Parser for .pose Files
============================

Parses NumPy .npz pose files from the WLASL dataset and converts them
to a frontend-consumable JSON format for skeleton rendering.

The .pose files contain:
    - landmarks: (frames, 543, 3) array of x, y, z coordinates
    - confidence: (frames, 543) array of detection confidences
    - presence_mask: (frames, 543) boolean array for detection mask
    - landmark_layout: dict mapping body region to [start, end] indices

Author: Nana Kwaku Amoako
Date: 2026-02-06
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


# ============================================================================
# Constants
# ============================================================================

# MediaPipe landmark counts
POSE_LANDMARK_COUNT = 33
FACE_LANDMARK_COUNT = 468
HAND_LANDMARK_COUNT = 21

# Skeleton connections for pose landmarks (MediaPipe BlazePose)
POSE_CONNECTIONS = [
    # Torso
    (11, 12),  # Left shoulder - Right shoulder
    (11, 23),  # Left shoulder - Left hip
    (12, 24),  # Right shoulder - Right hip
    (23, 24),  # Left hip - Right hip
    # Left arm
    (11, 13),  # Left shoulder - Left elbow
    (13, 15),  # Left elbow - Left wrist
    # Right arm
    (12, 14),  # Right shoulder - Right elbow
    (14, 16),  # Right elbow - Right wrist
    # Face outline (simplified)
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left side
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right side
    # Shoulders to ears
    (11, 7), (12, 8),
]

# Hand connections (MediaPipe Hands)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
]


# ============================================================================
# Parser Classes
# ============================================================================

class PoseParseError(Exception):
    """Raised when pose file parsing fails."""
    pass


class PoseParser:
    """
    Parser for .pose files (NumPy .npz format).
    
    Attributes:
        poses_dir: Directory containing .pose files
        mapping: Gloss to video mapping from selection
    """
    
    def __init__(self, poses_dir: Path, mapping_file: Optional[Path] = None):
        """
        Initialize parser.
        
        Args:
            poses_dir: Directory containing selected .pose files
            mapping_file: Path to gloss_video_mapping.json (optional)
        """
        self.poses_dir = Path(poses_dir)
        self.mapping: Dict[str, Any] = {}
        
        if mapping_file and mapping_file.exists():
            with open(mapping_file, 'r') as f:
                self.mapping = json.load(f)
    
    def get_pose_path(self, gloss: str) -> Path:
        """
        Get path to .pose file for a gloss.
        
        Args:
            gloss: Gloss name (e.g., 'hello')
        
        Returns:
            Path to .pose file
        
        Raises:
            PoseParseError: If gloss not found
        """
        # Normalize gloss name (lowercase, spaces to underscores)
        normalized = gloss.lower().replace(' ', '_')
        pose_path = self.poses_dir / f"{normalized}.pose"
        
        if pose_path.exists():
            return pose_path
        
        # Try original name
        pose_path = self.poses_dir / f"{gloss}.pose"
        if pose_path.exists():
            return pose_path
        
        raise PoseParseError(f"Pose file not found for gloss: {gloss}")
    
    def parse(self, gloss: str) -> Dict[str, Any]:
        """
        Parse a .pose file and return landmark data.
        
        Args:
            gloss: Gloss name to parse
        
        Returns:
            Dict with landmark data for frontend rendering:
            {
                "gloss": str,
                "fps": float,
                "frame_count": int,
                "frames": [...],
                "connections": {...}
            }
        """
        pose_path = self.get_pose_path(gloss)
        
        try:
            data = np.load(pose_path, allow_pickle=True)
        except Exception as e:
            raise PoseParseError(f"Failed to load pose file: {e}")
        
        # Extract arrays
        landmarks = data['landmarks']  # (frames, 543, 3)
        confidence = data['confidence']  # (frames, 543)
        
        # Get layout (index ranges for each body region)
        layout_raw = data['landmark_layout']
        if hasattr(layout_raw, 'item'):
            layout = layout_raw.item()
        else:
            layout = dict(layout_raw)
        
        frame_count = landmarks.shape[0]
        fps = float(data.get('fps', 30.0)) if 'fps' in data.files else 30.0
        
        # Parse landmark indices from layout
        pose_start, pose_end = self._parse_range(layout.get('pose', '0:33'))
        face_start, face_end = self._parse_range(layout.get('face', '33:501'))
        left_hand_start, left_hand_end = self._parse_range(layout.get('left_hand', '501:522'))
        right_hand_start, right_hand_end = self._parse_range(layout.get('right_hand', '522:543'))
        
        # Build frames array
        frames = []
        for i in range(frame_count):
            frame_landmarks = landmarks[i]
            frame_confidence = confidence[i]
            
            # Extract landmarks for each region (cleaned of NaN)
            pose_lm = self._clean_landmarks(frame_landmarks[pose_start:pose_end])
            face_lm = self._clean_landmarks(frame_landmarks[face_start:face_end])
            left_hand_lm = self._clean_landmarks(frame_landmarks[left_hand_start:left_hand_end])
            right_hand_lm = self._clean_landmarks(frame_landmarks[right_hand_start:right_hand_end])
            
            # Calculate per-region confidence (safe against NaN)
            pose_conf = self._safe_mean(frame_confidence[pose_start:pose_end])
            face_conf = self._safe_mean(frame_confidence[face_start:face_end])
            left_hand_conf = self._safe_mean(frame_confidence[left_hand_start:left_hand_end])
            right_hand_conf = self._safe_mean(frame_confidence[right_hand_start:right_hand_end])
            
            frames.append({
                "pose_landmarks": pose_lm,
                "face_landmarks": face_lm,
                "left_hand_landmarks": left_hand_lm,
                "right_hand_landmarks": right_hand_lm,
                "confidences": {
                    "pose": round(pose_conf, 3),
                    "face": round(face_conf, 3),
                    "left_hand": round(left_hand_conf, 3),
                    "right_hand": round(right_hand_conf, 3),
                }
            })
        
        return {
            "gloss": gloss,
            "fps": fps,
            "frame_count": frame_count,
            "frames": frames,
            "connections": {
                "pose": POSE_CONNECTIONS,
                "hand": HAND_CONNECTIONS,
            }
        }
    
    def parse_debug(self, gloss: str) -> Dict[str, Any]:
        """
        Parse a .pose file with detailed debug metrics.
        
        Args:
            gloss: Gloss name to parse
        
        Returns:
            Dict with detailed debug information
        """
        pose_path = self.get_pose_path(gloss)
        
        try:
            data = np.load(pose_path, allow_pickle=True)
        except Exception as e:
            raise PoseParseError(f"Failed to load pose file: {e}")
        
        landmarks = data['landmarks']
        confidence = data['confidence']
        
        # Get layout
        layout_raw = data['landmark_layout']
        if hasattr(layout_raw, 'item'):
            layout = layout_raw.item()
        else:
            layout = dict(layout_raw)
        
        frame_count = landmarks.shape[0]
        fps = float(data.get('fps', 30.0)) if 'fps' in data.files else 30.0
        
        # Parse ranges
        pose_start, pose_end = self._parse_range(layout.get('pose', '0:33'))
        face_start, face_end = self._parse_range(layout.get('face', '33:501'))
        left_hand_start, left_hand_end = self._parse_range(layout.get('left_hand', '501:522'))
        right_hand_start, right_hand_end = self._parse_range(layout.get('right_hand', '522:543'))
        
        # Calculate per-frame metrics
        frame_metrics = []
        for i in range(frame_count):
            frame_conf = confidence[i]
            
            # Detection rates (confidence > 0.5 means detected)
            pose_detected = np.mean(frame_conf[pose_start:pose_end] > 0.5)
            face_detected = np.mean(frame_conf[face_start:face_end] > 0.5)
            left_hand_detected = np.mean(frame_conf[left_hand_start:left_hand_end] > 0.5)
            right_hand_detected = np.mean(frame_conf[right_hand_start:right_hand_end] > 0.5)
            
            frame_metrics.append({
                "frame_index": i,
                "pose_detection": self._safe_mean(np.array(frame_conf[pose_start:pose_end] > 0.5, dtype=float)),
                "face_detection": self._safe_mean(np.array(frame_conf[face_start:face_end] > 0.5, dtype=float)),
                "left_hand_detection": self._safe_mean(np.array(frame_conf[left_hand_start:left_hand_end] > 0.5, dtype=float)),
                "right_hand_detection": self._safe_mean(np.array(frame_conf[right_hand_start:right_hand_end] > 0.5, dtype=float)),
                "overall_confidence": self._safe_mean(frame_conf),
            })
        
        # Aggregate stats
        all_pose_conf = confidence[:, pose_start:pose_end]
        all_face_conf = confidence[:, face_start:face_end]
        all_left_hand_conf = confidence[:, left_hand_start:left_hand_end]
        all_right_hand_conf = confidence[:, right_hand_start:right_hand_end]
        
        return {
            "gloss": gloss,
            "file_path": str(pose_path.name),
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": round(frame_count / fps, 2),
            "landmark_counts": {
                "pose": pose_end - pose_start,
                "face": face_end - face_start,
                "left_hand": left_hand_end - left_hand_start,
                "right_hand": right_hand_end - right_hand_start,
                "total": landmarks.shape[1],
            },
            "detection_rates": {
                "pose": self._safe_mean(np.array(all_pose_conf > 0.5, dtype=float)),
                "face": self._safe_mean(np.array(all_face_conf > 0.5, dtype=float)),
                "left_hand": self._safe_mean(np.array(all_left_hand_conf > 0.5, dtype=float)),
                "right_hand": self._safe_mean(np.array(all_right_hand_conf > 0.5, dtype=float)),
            },
            "confidence_stats": {
                "mean": self._safe_mean(confidence),
                "std": round(float(np.nanstd(confidence)), 3) if np.isfinite(np.nanstd(confidence)) else 0.0,
                "min": round(float(np.nanmin(confidence)), 3) if np.isfinite(np.nanmin(confidence)) else 0.0,
                "max": round(float(np.nanmax(confidence)), 3) if np.isfinite(np.nanmax(confidence)) else 0.0,
            },
            "frame_metrics": frame_metrics,
        }
    
    def list_available_glosses(self) -> List[str]:
        """List all available glosses in poses directory."""
        if not self.poses_dir.exists():
            return []
        
        glosses = []
        for f in self.poses_dir.glob("*.pose"):
            # Skip non-gloss files
            if f.name.startswith("_"):
                continue
            glosses.append(f.stem)
        
        return sorted(glosses)
    
    def _parse_range(self, range_str: str) -> Tuple[int, int]:
        """Parse a range string like '0:33' or [0, 33] into (start, end)."""
        if isinstance(range_str, (list, tuple)):
            return int(range_str[0]), int(range_str[1])
        if ':' in str(range_str):
            parts = str(range_str).split(':')
            return int(parts[0]), int(parts[1])
        # Fallback
        return 0, int(range_str)
    
    def _safe_mean(self, arr: np.ndarray) -> float:
        """Calculate mean, returning 0.0 if result is NaN or array is empty."""
        if arr.size == 0:
            return 0.0
        val = float(np.nanmean(arr))
        return val if np.isfinite(val) else 0.0
    
    def _clean_landmarks(self, arr: np.ndarray) -> List[List[float]]:
        """Convert landmarks to list, replacing NaN with 0.0."""
        cleaned = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return cleaned.tolist()


# ============================================================================
# Singleton Instance
# ============================================================================

_parser_instance: Optional[PoseParser] = None


def get_parser() -> PoseParser:
    """Get or create singleton parser instance."""
    global _parser_instance

    if _parser_instance is None:
        # Default paths
        api_dir = Path(__file__).parent
        algo_dir = api_dir.parent
        poses_dir = algo_dir / "selected_poses"

        if not poses_dir.exists():
            # Fallback: try relative to current working directory
            poses_dir = Path("duosign_algo/selected_poses").resolve()

        mapping_file = poses_dir / "gloss_video_mapping.json"

        _parser_instance = PoseParser(poses_dir, mapping_file)

    return _parser_instance


# ============================================================================
# CLI for Testing
# ============================================================================

if __name__ == "__main__":
    import sys
    
    parser = get_parser()
    
    if len(sys.argv) > 1:
        gloss = sys.argv[1]
        try:
            result = parser.parse(gloss)
            print(f"Parsed {gloss}: {result['frame_count']} frames at {result['fps']} fps")
            print(f"First frame pose landmarks: {len(result['frames'][0]['pose_landmarks'])} points")
        except PoseParseError as e:
            print(f"Error: {e}")
    else:
        glosses = parser.list_available_glosses()
        print(f"Available glosses ({len(glosses)}): {glosses[:10]}...")
