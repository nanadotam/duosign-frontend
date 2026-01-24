#!/usr/bin/env python3
"""
================================================================================
WLASL Pose Extraction Pipeline
================================================================================

This pipeline extracts pose landmarks from WLASL dataset videos using MediaPipe
Holistic. Each video produces exactly one .pose file containing a fixed-shape
landmark tensor (T × 523 × 4) with NaN padding for missing detections.

Author: Nana Kwaku Amoako
Based on methodology from: Moryossef et al. (2023) - "An Open-Source Gloss-Based
Baseline for Spoken to Signed Language Translation"

Usage:
    # Desktop
    python wlasl_pose_pipeline.py --input_dir /path/to/videos --output_dir /path/to/output --num_workers 8
    
    # Colab
    # !python wlasl_pose_pipeline.py --input_dir /content/drive/MyDrive/WLASL/videos --output_dir /content/drive/MyDrive/pose_out --num_workers 4

================================================================================
"""

import os
import sys
import json
import logging
import argparse
import traceback
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Generator, Any
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

import cv2
import numpy as np

# Lazy import for tqdm to handle Colab vs local
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable

# MediaPipe import with version detection
try:
    import mediapipe as mp
    MP_VERSION = tuple(map(int, mp.__version__.split('.')[:2]))
    
    # MediaPipe 0.10+ uses new Tasks API
    if MP_VERSION >= (0, 10):
        USE_NEW_API = True
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe import Image as MPImage
    else:
        # Legacy API (0.9 and below)
        USE_NEW_API = False
        
except ImportError:
    print("ERROR: MediaPipe not installed. Run: pip install mediapipe")
    sys.exit(1)



# ==============================================================================
# SECTION 1: CONFIGURATION & CONSTANTS
# ==============================================================================
"""
This section defines all constants and configuration parameters for the pipeline.
These values are based on the methodology analysis from the spoken-to-signed-translation
codebase and ensure compatibility with downstream model training.
"""

# Landmark counts (after reduction from MediaPipe's raw output)
LANDMARK_COUNTS = {
    "pose": 13,           # Upper body only (reduced from 33)
    "face": 468,          # Full face mesh
    "left_hand": 21,      # Full hand
    "right_hand": 21,     # Full hand
}
TOTAL_LANDMARKS = sum(LANDMARK_COUNTS.values())  # 523

# Landmark index ranges in the final tensor
LANDMARK_RANGES = {
    "pose": (0, 13),              # Indices 0-12
    "face": (13, 481),            # Indices 13-480
    "left_hand": (481, 502),      # Indices 481-501
    "right_hand": (502, 523),     # Indices 502-522
}

# Upper body indices from MediaPipe's 33-point BlazePose model
# We extract only these for sign language (legs not needed)
UPPER_BODY_INDICES = [
    0,   # nose
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    17,  # left_pinky (hand connection point)
    18,  # right_pinky (hand connection point)
    19,  # left_index (hand connection point)
    20,  # right_index (hand connection point)
    21,  # left_thumb (hand connection point)
    22,  # right_thumb (hand connection point)
]

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

# Pose file format version
POSE_FORMAT_VERSION = "1.0"


@dataclass
class PipelineConfig:
    """
    Configuration parameters for the pose extraction pipeline.
    
    Attributes:
        input_dir: Directory containing input video files
        output_dir: Directory for output .pose files
        num_workers: Number of parallel worker processes
        model_complexity: MediaPipe model complexity (0=lite, 1=full, 2=heavy)
        min_detection_confidence: Minimum detection confidence threshold
        min_tracking_confidence: Minimum tracking confidence threshold
        skip_existing: Skip videos that already have valid .pose files
        log_level: Logging verbosity level
        recursive: Search input_dir recursively for videos
    """
    input_dir: str
    output_dir: str
    num_workers: int = 4
    model_complexity: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    skip_existing: bool = True
    log_level: str = "INFO"
    recursive: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        return asdict(self)


# ==============================================================================
# SECTION 2: LOGGING SETUP
# ==============================================================================
"""
Configures logging for the pipeline with both console and file output.
"""

def setup_logging(log_level: str, output_dir: Path) -> logging.Logger:
    """
    Set up logging with console and file handlers.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        output_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("wlasl_pose")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (detailed logs)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "pipeline.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


# ==============================================================================
# SECTION 3: VIDEO DECODER
# ==============================================================================
"""
Handles video file decoding with streaming frame iteration to minimize RAM usage.
Each video is processed frame-by-frame without loading all frames into memory.
"""

@dataclass
class VideoMetadata:
    """
    Metadata extracted from a video file.
    
    Attributes:
        source_path: Original video file path
        fps: Frames per second
        width: Frame width in pixels
        height: Frame height in pixels
        frame_count: Total number of frames
        duration_sec: Duration in seconds
    """
    source_path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_sec: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class VideoDecoder:
    """
    Decodes video files and yields frames as RGB numpy arrays.
    
    This class implements streaming frame iteration to minimize RAM usage.
    Only one frame is held in memory at a time per video.
    
    Example:
        decoder = VideoDecoder()
        metadata, frame_generator = decoder.decode(video_path)
        for frame in frame_generator:
            process(frame)
    """
    
    def __init__(self):
        """Initialize the video decoder."""
        self.logger = logging.getLogger("wlasl_pose.decoder")
    
    def decode(self, video_path: Path) -> Tuple[VideoMetadata, Generator[np.ndarray, None, None]]:
        """
        Decode a video file and return metadata with a frame generator.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (VideoMetadata, frame_generator)
            
        Raises:
            VideoDecodeError: If video cannot be opened or read
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise VideoDecodeError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise VideoDecodeError(f"Cannot open video: {video_path}")
        
        # Extract metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Handle edge cases
        if fps <= 0:
            fps = 30.0  # Default fallback
            self.logger.warning(f"Invalid FPS for {video_path}, defaulting to 30")
        
        duration_sec = frame_count / fps if fps > 0 else 0
        
        metadata = VideoMetadata(
            source_path=str(video_path),
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_sec=duration_sec
        )
        
        # Create frame generator
        def frame_generator():
            """Generate RGB frames from video."""
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                # Convert BGR (OpenCV default) to RGB (MediaPipe requirement)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield frame_rgb
            cap.release()
        
        return metadata, frame_generator()
    
    def get_metadata_only(self, video_path: Path) -> VideoMetadata:
        """
        Get video metadata without decoding frames.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoMetadata object
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoDecodeError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        
        return VideoMetadata(
            source_path=str(video_path),
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_sec=frame_count / fps if fps > 0 else 0
        )


class VideoDecodeError(Exception):
    """Raised when video decoding fails."""
    pass


# ==============================================================================
# SECTION 4: LANDMARK EXTRACTOR
# ==============================================================================
"""
Extracts landmarks from video frames using MediaPipe Holistic.

The extractor produces a fixed-shape tensor (T × 523 × 4) where:
- T = number of frames
- 523 = total landmarks (13 pose + 468 face + 21 left hand + 21 right hand)
- 4 = (x, y, z, confidence)

Missing landmarks are filled with NaN to maintain shape consistency.
"""

class LandmarkExtractor:
    """
    Extracts pose, face, and hand landmarks from video frames.
    
    Uses MediaPipe Holistic for unified detection of all body parts.
    Produces fixed-shape tensors with NaN padding for missing detections.
    
    Supports both MediaPipe 0.10+ (Tasks API) and legacy 0.9 (Solutions API).
    
    Example:
        extractor = LandmarkExtractor(model_complexity=2)
        landmarks = extractor.extract_from_frames(frame_generator)
        # landmarks.shape = (T, 523, 4)
    """
    
    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the landmark extractor.
        
        Args:
            model_complexity: MediaPipe model complexity (0=lite, 1=full, 2=heavy)
            min_detection_confidence: Minimum detection confidence [0, 1]
            min_tracking_confidence: Minimum tracking confidence [0, 1]
        """
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.logger = logging.getLogger("wlasl_pose.extractor")
        
        self._holistic = None
        self._pose_landmarker = None
        self._hand_landmarker = None
        self._face_landmarker = None
        self._use_new_api = USE_NEW_API
    
    def _init_legacy_holistic(self):
        """Initialize MediaPipe Holistic (legacy API for 0.9 and below)."""
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            smooth_landmarks=False,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
    
    def _init_new_api_landmarkers(self):
        """
        Initialize MediaPipe Tasks (new API for 0.10+).
        
        Note: The new API doesn't have a unified Holistic model, so we use
        individual detectors for pose, hands, and face, then combine results.
        """
        # For 0.10+, we use a simpler approach with just the pose landmarker
        # which includes pose and some hand detection
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            HandLandmarker,
            HandLandmarkerOptions,
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode
        )
        
        # Pose Landmarker
        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self._get_model_path("pose_landmarker_lite.task")
            ),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=False
        )
        self._pose_landmarker = PoseLandmarker.create_from_options(pose_options)
        
        # Hand Landmarker
        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self._get_model_path("hand_landmarker.task")
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self._hand_landmarker = HandLandmarker.create_from_options(hand_options)
        
        # Face Landmarker
        face_options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self._get_model_path("face_landmarker.task")
            ),
            running_mode=RunningMode.VIDEO,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self._face_landmarker = FaceLandmarker.create_from_options(face_options)
    
    def _get_model_path(self, model_name: str) -> str:
        """Get path to MediaPipe model file."""
        import urllib.request
        import tempfile
        
        model_urls = {
            "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        }
        
        # Cache models in temp directory
        cache_dir = Path(tempfile.gettempdir()) / "mediapipe_models"
        cache_dir.mkdir(exist_ok=True)
        model_path = cache_dir / model_name
        
        if not model_path.exists():
            self.logger.info(f"Downloading MediaPipe model: {model_name}")
            urllib.request.urlretrieve(model_urls[model_name], model_path)
        
        return str(model_path)
    
    def extract_from_frames(
        self,
        frames: Generator[np.ndarray, None, None]
    ) -> np.ndarray:
        """
        Extract landmarks from a sequence of frames.
        
        Args:
            frames: Generator yielding RGB frames
            
        Returns:
            Landmark tensor of shape (T, 523, 4) where 4 = (x, y, z, confidence)
        """
        all_landmarks = []
        timestamp_ms = 0
        
        for frame in frames:
            if self._use_new_api:
                frame_landmarks = self._extract_frame_landmarks_new_api(frame, timestamp_ms)
            else:
                if self._holistic is None:
                    self._init_legacy_holistic()
                frame_landmarks = self._extract_frame_landmarks_legacy(frame)
            
            all_landmarks.append(frame_landmarks)
            timestamp_ms += 33  # Approximate 30fps
        
        if not all_landmarks:
            return np.empty((0, TOTAL_LANDMARKS, 4), dtype=np.float32)
        
        return np.stack(all_landmarks, axis=0)
    
    def _extract_frame_landmarks_legacy(self, frame: np.ndarray) -> np.ndarray:
        """Extract landmarks using legacy Solutions API."""
        landmarks = np.full((TOTAL_LANDMARKS, 4), np.nan, dtype=np.float32)
        results = self._holistic.process(frame)
        
        # POSE LANDMARKS
        if results.pose_landmarks is not None:
            for i, mp_idx in enumerate(UPPER_BODY_INDICES):
                lm = results.pose_landmarks.landmark[mp_idx]
                landmarks[i] = [lm.x, lm.y, lm.z, lm.visibility]
        
        # FACE LANDMARKS
        if results.face_landmarks is not None:
            for i, lm in enumerate(results.face_landmarks.landmark):
                idx = LANDMARK_RANGES["face"][0] + i
                landmarks[idx] = [lm.x, lm.y, lm.z, 1.0]
        
        # LEFT HAND LANDMARKS
        if results.left_hand_landmarks is not None:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                idx = LANDMARK_RANGES["left_hand"][0] + i
                landmarks[idx] = [lm.x, lm.y, lm.z, 1.0]
        
        # RIGHT HAND LANDMARKS
        if results.right_hand_landmarks is not None:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                idx = LANDMARK_RANGES["right_hand"][0] + i
                landmarks[idx] = [lm.x, lm.y, lm.z, 1.0]
        
        return landmarks
    
    def _extract_frame_landmarks_new_api(
        self, 
        frame: np.ndarray, 
        timestamp_ms: int
    ) -> np.ndarray:
        """Extract landmarks using new Tasks API (MediaPipe 0.10+)."""
        # Initialize landmarkers if needed
        if self._pose_landmarker is None:
            try:
                self._init_new_api_landmarkers()
            except Exception as e:
                self.logger.warning(f"Failed to init new API, falling back to simple detection: {e}")
                return self._extract_simple_landmarks(frame)
        
        landmarks = np.full((TOTAL_LANDMARKS, 4), np.nan, dtype=np.float32)
        
        # Convert frame to MediaPipe Image
        mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=frame)
        
        try:
            # POSE LANDMARKS
            pose_result = self._pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            if pose_result.pose_landmarks:
                pose_lms = pose_result.pose_landmarks[0]
                for i, mp_idx in enumerate(UPPER_BODY_INDICES):
                    if mp_idx < len(pose_lms):
                        lm = pose_lms[mp_idx]
                        landmarks[i] = [lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, 'visibility') else 1.0]
            
            # HAND LANDMARKS
            hand_result = self._hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            if hand_result.hand_landmarks:
                for h_idx, (hand_lms, handedness) in enumerate(zip(
                    hand_result.hand_landmarks, 
                    hand_result.handedness
                )):
                    hand_type = handedness[0].category_name.lower()
                    if hand_type == "left":
                        base_idx = LANDMARK_RANGES["left_hand"][0]
                    else:
                        base_idx = LANDMARK_RANGES["right_hand"][0]
                    
                    for i, lm in enumerate(hand_lms):
                        landmarks[base_idx + i] = [lm.x, lm.y, lm.z, 1.0]
            
            # FACE LANDMARKS
            face_result = self._face_landmarker.detect_for_video(mp_image, timestamp_ms)
            if face_result.face_landmarks:
                face_lms = face_result.face_landmarks[0]
                for i, lm in enumerate(face_lms):
                    if i < 468:  # Ensure we don't exceed expected count
                        idx = LANDMARK_RANGES["face"][0] + i
                        landmarks[idx] = [lm.x, lm.y, lm.z, 1.0]
        except Exception as e:
            self.logger.debug(f"Frame detection error: {e}")
        
        return landmarks
    
    def _extract_simple_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Fallback simple extraction without holistic model."""
        # Return empty landmarks as fallback
        return np.full((TOTAL_LANDMARKS, 4), np.nan, dtype=np.float32)
    
    def close(self):
        """Release MediaPipe resources."""
        if self._holistic is not None:
            self._holistic.close()
            self._holistic = None
        if self._pose_landmarker is not None:
            self._pose_landmarker.close()
            self._pose_landmarker = None
        if self._hand_landmarker is not None:
            self._hand_landmarker.close()
            self._hand_landmarker = None
        if self._face_landmarker is not None:
            self._face_landmarker.close()
            self._face_landmarker = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False



# ==============================================================================
# SECTION 5: POSE SERIALIZER
# ==============================================================================
"""
Handles serialization of pose data to .pose files (compressed NumPy archives).

Features:
- Atomic writes (temp file + rename) for crash safety
- Validation of written files
- Metadata inclusion for auditability
"""

class PoseSerializer:
    """
    Serializes pose landmarks to .pose files.
    
    Output format is a compressed NumPy archive containing:
    - landmarks: (T, 523, 3) float32 - x, y, z coordinates
    - confidence: (T, 523) float32 - detection confidence
    - presence_mask: (T, 523) bool - True where landmark detected
    - fps: float - video frame rate
    - frame_count: int - number of frames
    - source_video: str - original video path
    - landmark_layout: dict - index ranges per component
    - version: str - format version
    
    Example:
        serializer = PoseSerializer()
        serializer.save(landmarks, metadata, output_path)
        
        # Load later:
        data = np.load(output_path, allow_pickle=True)
    """
    
    def __init__(self):
        """Initialize the pose serializer."""
        self.logger = logging.getLogger("wlasl_pose.serializer")
    
    def save(
        self,
        landmarks: np.ndarray,
        metadata: VideoMetadata,
        output_path: Path
    ) -> Path:
        """
        Save pose landmarks to a .pose file.
        
        Args:
            landmarks: Landmark tensor of shape (T, 523, 4)
            metadata: Video metadata
            output_path: Output file path (should end in .pose)
            
        Returns:
            Path to the saved file
            
        Raises:
            SerializationError: If saving fails
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate input shape
        if landmarks.ndim != 3 or landmarks.shape[1] != TOTAL_LANDMARKS or landmarks.shape[2] != 4:
            raise SerializationError(
                f"Invalid landmark shape: {landmarks.shape}, expected (T, {TOTAL_LANDMARKS}, 4)"
            )
        
        # Separate coordinates and confidence
        coords = landmarks[:, :, :3]      # (T, 523, 3) - x, y, z
        confidence = landmarks[:, :, 3]   # (T, 523) - confidence
        
        # Compute presence mask (True where detected, False where NaN)
        presence_mask = ~np.isnan(confidence)
        
        # Prepare data dictionary
        pose_data = {
            "landmarks": coords.astype(np.float32),
            "confidence": confidence.astype(np.float32),
            "presence_mask": presence_mask.astype(bool),
            "fps": np.float32(metadata.fps),
            "frame_count": np.int32(landmarks.shape[0]),
            "source_video": str(metadata.source_path),
            "landmark_layout": LANDMARK_RANGES,
            "version": POSE_FORMAT_VERSION
        }
        
        # Atomic write: save to temp file, then rename
        # Note: np.savez_compressed adds .npz extension automatically if not present
        temp_path = output_path.with_suffix(".tmp")
        temp_path_with_npz = Path(str(temp_path) + ".npz") if not str(temp_path).endswith(".npz") else temp_path
        
        try:
            np.savez_compressed(str(temp_path), **pose_data)
            
            # np.savez_compressed adds .npz, so the actual file is temp_path + .npz
            actual_temp_path = temp_path_with_npz if temp_path_with_npz.exists() else temp_path
            if not actual_temp_path.exists():
                # Try with .npz appended
                actual_temp_path = Path(str(temp_path) + ".npz")
            
            # Rename atomically (on POSIX systems)
            actual_temp_path.rename(output_path)
            
            self.logger.debug(f"Saved pose file: {output_path}")
            return output_path
            
        except Exception as e:
            # Clean up temp file on failure
            for p in [temp_path, temp_path_with_npz, Path(str(temp_path) + ".npz")]:
                if p.exists():
                    p.unlink()
            raise SerializationError(f"Failed to save pose file: {e}")
    
    @staticmethod
    def validate(pose_path: Path) -> bool:
        """
        Validate a .pose file for integrity.
        
        Args:
            pose_path: Path to the .pose file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            pose_path = Path(pose_path)
            
            if not pose_path.exists():
                return False
            
            # Try to load the file
            data = np.load(str(pose_path), allow_pickle=True)
            
            # Check required keys
            required_keys = {"landmarks", "confidence", "fps", "frame_count"}
            if not required_keys.issubset(set(data.keys())):
                return False
            
            landmarks = data["landmarks"]
            confidence = data["confidence"]
            
            # Check shapes
            if landmarks.ndim != 3:
                return False
            if landmarks.shape[1] != TOTAL_LANDMARKS:
                return False
            if landmarks.shape[2] != 3:
                return False
            if confidence.shape != landmarks.shape[:2]:
                return False
            
            # Check frame count > 0
            if data["frame_count"] == 0:
                return False
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def load(pose_path: Path) -> Dict[str, Any]:
        """
        Load a .pose file.
        
        Args:
            pose_path: Path to the .pose file
            
        Returns:
            Dictionary containing pose data
        """
        data = np.load(str(pose_path), allow_pickle=True)
        return {key: data[key] for key in data.keys()}


class SerializationError(Exception):
    """Raised when pose serialization fails."""
    pass


# ==============================================================================
# SECTION 6: MANIFEST & FAILURE LOGGING
# ==============================================================================
"""
Handles logging of successful extractions and failures for auditability.

Creates JSONL (JSON Lines) files for easy parsing and analysis.
"""

class ManifestLogger:
    """
    Logs successful pose extractions and failures to JSONL files.
    
    Creates two files:
    - manifest.jsonl: Successful extractions with metadata
    - failures.jsonl: Failed extractions with error details
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize the manifest logger.
        
        Args:
            output_dir: Directory for log files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self.failures_path = self.output_dir / "failures.jsonl"
        
        self.logger = logging.getLogger("wlasl_pose.manifest")
    
    def log_success(
        self,
        video_path: Path,
        pose_path: Path,
        metadata: VideoMetadata,
        processing_time_sec: float
    ):
        """
        Log a successful pose extraction.
        
        Args:
            video_path: Source video path
            pose_path: Output pose file path
            metadata: Video metadata
            processing_time_sec: Time taken to process
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "video_path": str(video_path),
            "pose_path": str(pose_path),
            "fps": metadata.fps,
            "frame_count": metadata.frame_count,
            "duration_sec": metadata.duration_sec,
            "processing_time_sec": round(processing_time_sec, 3)
        }
        
        self._append_jsonl(self.manifest_path, entry)
    
    def log_failure(
        self,
        video_path: Path,
        error: Exception,
        error_type: str = None
    ):
        """
        Log a failed pose extraction.
        
        Args:
            video_path: Source video path
            error: The exception that occurred
            error_type: Optional error classification
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "status": "failure",
            "video_path": str(video_path),
            "error_type": error_type or type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc()
        }
        
        self._append_jsonl(self.failures_path, entry)
        self.logger.warning(f"Failed to process {video_path}: {error}")
    
    def _append_jsonl(self, path: Path, entry: Dict):
        """Append a JSON entry to a JSONL file (thread-safe via append mode)."""
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ==============================================================================
# SECTION 7: PIPELINE ORCHESTRATOR
# ==============================================================================
"""
Main pipeline orchestrator that ties all components together.

Features:
- Multiprocessing for parallel video processing
- Progress tracking with tqdm
- Resumability (skips existing valid outputs)
- Configuration logging
"""

class PoseExtractionPipeline:
    """
    Orchestrates the pose extraction pipeline for batch processing.
    
    Handles:
    - Video discovery
    - Parallel processing with multiprocessing
    - Progress tracking
    - Manifest/failure logging
    - Resumability
    
    Example:
        config = PipelineConfig(
            input_dir="/path/to/videos",
            output_dir="/path/to/output",
            num_workers=8
        )
        pipeline = PoseExtractionPipeline(config)
        stats = pipeline.run()
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.input_dir = Path(config.input_dir)
        self.output_dir = Path(config.output_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = setup_logging(config.log_level, self.output_dir)
        
        # Save configuration
        self._save_config()
        
        # Initialize manifest logger
        self.manifest = ManifestLogger(self.output_dir)
    
    def _save_config(self):
        """Save pipeline configuration to JSON."""
        config_path = self.output_dir / "config.json"
        config_dict = self.config.to_dict()
        config_dict["run_timestamp"] = datetime.now().isoformat()
        config_dict["python_version"] = sys.version
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Configuration saved to {config_path}")
    
    def discover_videos(self) -> List[Path]:
        """
        Discover all video files in the input directory.
        
        Returns:
            List of video file paths
        """
        videos = []
        
        if self.config.recursive:
            # Recursive search
            for ext in VIDEO_EXTENSIONS:
                videos.extend(self.input_dir.rglob(f"*{ext}"))
        else:
            # Non-recursive search
            for ext in VIDEO_EXTENSIONS:
                videos.extend(self.input_dir.glob(f"*{ext}"))
        
        videos = sorted(videos)
        self.logger.info(f"Discovered {len(videos)} video files")
        
        return videos
    
    def get_output_path(self, video_path: Path) -> Path:
        """
        Compute output .pose file path for a video.
        
        Preserves relative directory structure if recursive mode is enabled.
        
        Args:
            video_path: Input video path
            
        Returns:
            Output .pose file path
        """
        if self.config.recursive:
            # Preserve relative structure
            relative = video_path.relative_to(self.input_dir)
            output_path = self.output_dir / relative.with_suffix(".pose")
        else:
            # Flat output
            output_path = self.output_dir / f"{video_path.stem}.pose"
        
        return output_path
    
    def filter_pending(self, videos: List[Path]) -> List[Path]:
        """
        Filter videos to only those needing processing.
        
        Skips videos with existing valid .pose files when skip_existing is True.
        
        Args:
            videos: List of video paths
            
        Returns:
            Filtered list of videos needing processing
        """
        if not self.config.skip_existing:
            return videos
        
        pending = []
        skipped = 0
        
        for video in videos:
            pose_path = self.get_output_path(video)
            
            if pose_path.exists():
                if PoseSerializer.validate(pose_path):
                    skipped += 1
                    continue
                else:
                    # Invalid file, remove and reprocess
                    self.logger.warning(f"Removing invalid pose file: {pose_path}")
                    pose_path.unlink()
            
            pending.append(video)
        
        if skipped > 0:
            self.logger.info(f"Skipping {skipped} already processed videos")
        
        return pending
    
    def run(self) -> Dict[str, int]:
        """
        Run the pipeline on all videos.
        
        Returns:
            Statistics dictionary with counts
        """
        start_time = datetime.now()
        
        # Discover videos
        all_videos = self.discover_videos()
        
        if not all_videos:
            self.logger.warning("No video files found in input directory")
            return {"total": 0, "processed": 0, "skipped": 0, "failed": 0}
        
        # Filter pending
        pending_videos = self.filter_pending(all_videos)
        skipped_count = len(all_videos) - len(pending_videos)
        
        self.logger.info(f"Processing {len(pending_videos)} videos with {self.config.num_workers} workers")
        
        # Process videos
        success_count = 0
        failure_count = 0
        
        if self.config.num_workers == 1:
            # Single-threaded processing
            for video in tqdm(pending_videos, desc="Extracting poses"):
                try:
                    self._process_single_video(video)
                    success_count += 1
                except Exception as e:
                    self.manifest.log_failure(video, e)
                    failure_count += 1
        else:
            # Multiprocessing
            # Prepare worker function with config
            worker_fn = partial(
                _worker_process_video,
                config=self.config,
                output_dir=self.output_dir
            )
            
            with Pool(processes=self.config.num_workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(worker_fn, pending_videos),
                    total=len(pending_videos),
                    desc="Extracting poses"
                ))
            
            # Count results
            for result in results:
                if result["status"] == "success":
                    success_count += 1
                else:
                    failure_count += 1
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        
        stats = {
            "total": len(all_videos),
            "processed": success_count,
            "skipped": skipped_count,
            "failed": failure_count,
            "elapsed_seconds": round(elapsed, 2)
        }
        
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info(f"  Total videos: {stats['total']}")
        self.logger.info(f"  Processed:    {stats['processed']}")
        self.logger.info(f"  Skipped:      {stats['skipped']}")
        self.logger.info(f"  Failed:       {stats['failed']}")
        self.logger.info(f"  Elapsed:      {stats['elapsed_seconds']:.1f}s")
        self.logger.info("=" * 60)
        
        # Save summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def _process_single_video(self, video_path: Path):
        """
        Process a single video (used in single-threaded mode).
        
        Args:
            video_path: Path to the video file
        """
        import time
        start = time.time()
        
        output_path = self.get_output_path(video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Decode video
        decoder = VideoDecoder()
        metadata, frames = decoder.decode(video_path)
        
        # Extract landmarks
        with LandmarkExtractor(
            model_complexity=self.config.model_complexity,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        ) as extractor:
            landmarks = extractor.extract_from_frames(frames)
        
        if landmarks.shape[0] == 0:
            raise VideoDecodeError("No frames extracted from video")
        
        # Serialize
        serializer = PoseSerializer()
        serializer.save(landmarks, metadata, output_path)
        
        elapsed = time.time() - start
        self.manifest.log_success(video_path, output_path, metadata, elapsed)


def _worker_process_video(video_path: Path, config: PipelineConfig, output_dir: Path) -> Dict:
    """
    Worker function for multiprocessing.
    
    This function runs in a separate process and must be picklable.
    
    Args:
        video_path: Path to the video file
        config: Pipeline configuration
        output_dir: Output directory
        
    Returns:
        Result dictionary with status and metadata
    """
    import time
    
    try:
        start = time.time()
        
        # Compute output path
        if config.recursive:
            input_dir = Path(config.input_dir)
            relative = video_path.relative_to(input_dir)
            output_path = output_dir / relative.with_suffix(".pose")
        else:
            output_path = output_dir / f"{video_path.stem}.pose"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Decode video
        decoder = VideoDecoder()
        metadata, frames = decoder.decode(video_path)
        
        # Extract landmarks
        with LandmarkExtractor(
            model_complexity=config.model_complexity,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence
        ) as extractor:
            landmarks = extractor.extract_from_frames(frames)
        
        if landmarks.shape[0] == 0:
            raise VideoDecodeError("No frames extracted from video")
        
        # Serialize
        serializer = PoseSerializer()
        serializer.save(landmarks, metadata, output_path)
        
        elapsed = time.time() - start
        
        # Log success
        manifest = ManifestLogger(output_dir)
        manifest.log_success(video_path, output_path, metadata, elapsed)
        
        return {
            "status": "success",
            "video": str(video_path),
            "output": str(output_path),
            "frames": landmarks.shape[0],
            "elapsed": elapsed
        }
        
    except Exception as e:
        # Log failure
        manifest = ManifestLogger(output_dir)
        manifest.log_failure(video_path, e)
        
        return {
            "status": "failure",
            "video": str(video_path),
            "error": str(e)
        }


# ==============================================================================
# SECTION 8: CLI INTERFACE
# ==============================================================================
"""
Command-line interface for running the pipeline.

Supports both desktop and Colab usage with the same interface.
"""

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="WLASL Pose Extraction Pipeline - Extract pose landmarks from sign language videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python wlasl_pose_pipeline.py --input_dir ./videos --output_dir ./poses
    
    # With custom settings
    python wlasl_pose_pipeline.py --input_dir ./videos --output_dir ./poses --num_workers 8 --model_complexity 2
    
    # Resume interrupted run
    python wlasl_pose_pipeline.py --input_dir ./videos --output_dir ./poses --skip_existing
    
    # Recursive directory search
    python wlasl_pose_pipeline.py --input_dir ./videos --output_dir ./poses --recursive
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        required=True,
        help="Directory containing input video files"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Directory for output .pose files"
    )
    
    # Processing options
    parser.add_argument(
        "--num_workers", "-w",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4)"
    )
    parser.add_argument(
        "--model_complexity", "-m",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="MediaPipe model complexity: 0=lite, 1=full, 2=heavy (default: 2)"
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.5,
        help="Minimum tracking confidence threshold (default: 0.5)"
    )
    
    # Behavior options
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip videos with existing valid .pose files (default: True)"
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_false",
        dest="skip_existing",
        help="Reprocess all videos even if .pose files exist"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=False,
        help="Search input directory recursively for videos"
    )
    
    # Logging
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity level (default: INFO)"
    )
    
    return parser.parse_args()


def main():
    """
    Main entry point for the pipeline.
    """
    args = parse_args()
    
    # Create configuration
    config = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        skip_existing=args.skip_existing,
        log_level=args.log_level,
        recursive=args.recursive
    )
    
    # Print banner
    print("=" * 60)
    print("WLASL POSE EXTRACTION PIPELINE")
    print("=" * 60)
    print(f"Input directory:  {config.input_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Workers:          {config.num_workers}")
    print(f"Model complexity: {config.model_complexity}")
    print(f"Skip existing:    {config.skip_existing}")
    print("=" * 60)
    print()
    
    # Run pipeline
    pipeline = PoseExtractionPipeline(config)
    stats = pipeline.run()
    
    # Return exit code based on failures
    if stats["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


# ==============================================================================
# SECTION 9: UTILITY FUNCTIONS
# ==============================================================================
"""
Additional utility functions for working with pose files.
"""

def visualize_pose(pose_path: Path, frame_idx: int = 0, save_path: Optional[Path] = None):
    """
    Visualize landmarks from a .pose file.
    
    Args:
        pose_path: Path to the .pose file
        frame_idx: Frame index to visualize
        save_path: Optional path to save the visualization
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization: pip install matplotlib")
        return
    
    data = PoseSerializer.load(pose_path)
    landmarks = data["landmarks"]
    confidence = data["confidence"]
    
    if frame_idx >= landmarks.shape[0]:
        print(f"Frame {frame_idx} out of range (max: {landmarks.shape[0] - 1})")
        return
    
    frame_landmarks = landmarks[frame_idx]
    frame_confidence = confidence[frame_idx]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Color by component
    colors = []
    for i in range(TOTAL_LANDMARKS):
        if i < LANDMARK_RANGES["pose"][1]:
            colors.append("blue")  # Pose
        elif i < LANDMARK_RANGES["face"][1]:
            colors.append("green")  # Face
        elif i < LANDMARK_RANGES["left_hand"][1]:
            colors.append("orange")  # Left hand
        else:
            colors.append("red")  # Right hand
    
    # Filter out NaN values
    valid = ~np.isnan(frame_landmarks[:, 0])
    
    ax.scatter(
        frame_landmarks[valid, 0],
        -frame_landmarks[valid, 1],  # Flip Y for correct orientation
        c=[colors[i] for i in range(TOTAL_LANDMARKS) if valid[i]],
        alpha=[frame_confidence[i] if not np.isnan(frame_confidence[i]) else 0.3 
               for i in range(TOTAL_LANDMARKS) if valid[i]],
        s=10
    )
    
    ax.set_title(f"Frame {frame_idx} - {pose_path.name}")
    ax.set_xlabel("X (normalized)")
    ax.set_ylabel("Y (normalized)")
    ax.axis("equal")
    ax.legend(["Pose (blue)", "Face (green)", "Left Hand (orange)", "Right Hand (red)"])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_pose_info(pose_path: Path):
    """
    Print information about a .pose file.
    
    Args:
        pose_path: Path to the .pose file
    """
    if not PoseSerializer.validate(pose_path):
        print(f"Invalid pose file: {pose_path}")
        return
    
    data = PoseSerializer.load(pose_path)
    
    print(f"Pose file: {pose_path}")
    print(f"  Version:      {data.get('version', 'N/A')}")
    print(f"  Source video: {data.get('source_video', 'N/A')}")
    print(f"  Frame count:  {data['frame_count']}")
    print(f"  FPS:          {data['fps']:.2f}")
    print(f"  Landmarks:    {data['landmarks'].shape}")
    print(f"  Duration:     {data['frame_count'] / data['fps']:.2f}s")
    
    # Detection statistics
    presence = data.get("presence_mask", ~np.isnan(data["confidence"]))
    
    for component, (start, end) in LANDMARK_RANGES.items():
        component_presence = presence[:, start:end]
        detection_rate = component_presence.any(axis=1).sum() / presence.shape[0] * 100
        print(f"  {component.capitalize()} detection rate: {detection_rate:.1f}%")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
