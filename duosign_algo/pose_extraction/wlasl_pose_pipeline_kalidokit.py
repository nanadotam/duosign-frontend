#!/usr/bin/env python3
"""
================================================================================
WLASL Pose Extraction Pipeline - KALIDOKIT COMPATIBLE VERSION
================================================================================

This pipeline extracts pose landmarks from WLASL dataset videos using MediaPipe
Holistic. This version extracts FULL 33-point pose landmarks for compatibility
with Kalidokit avatar rigging.

Key differences from original:
- Extracts ALL 33 pose landmarks (not reduced 13 upper body)
- Total landmarks: 543 (33 pose + 468 face + 21 left hand + 21 right hand)
- Compatible with Kalidokit.Pose.solve() which expects standard MediaPipe format

Author: Nana Kwaku Amoako
Modified for Kalidokit compatibility

Usage:
    python wlasl_pose_pipeline_kalidokit.py --input_dir /path/to/videos --output_dir /path/to/output --num_workers 4

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
    def tqdm(iterable, **kwargs):
        return iterable

# MediaPipe import with version detection
try:
    import mediapipe as mp
    MP_VERSION = tuple(map(int, mp.__version__.split('.')[:2]))

    if MP_VERSION >= (0, 10):
        USE_NEW_API = True
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe import Image as MPImage
    else:
        USE_NEW_API = False

except ImportError:
    print("ERROR: MediaPipe not installed. Run: pip install mediapipe")
    sys.exit(1)


# ==============================================================================
# SECTION 1: CONFIGURATION & CONSTANTS (KALIDOKIT COMPATIBLE)
# ==============================================================================
"""
KALIDOKIT COMPATIBLE LANDMARK FORMAT

Kalidokit.Pose.solve() expects the standard MediaPipe 33-point pose format:
https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

MediaPipe Pose Landmarks (33 points):
    0: nose
    1: left_eye_inner
    2: left_eye
    3: left_eye_outer
    4: right_eye_inner
    5: right_eye
    6: right_eye_outer
    7: left_ear
    8: right_ear
    9: mouth_left
    10: mouth_right
    11: left_shoulder
    12: right_shoulder
    13: left_elbow
    14: right_elbow
    15: left_wrist
    16: right_wrist
    17: left_pinky
    18: right_pinky
    19: left_index
    20: right_index
    21: left_thumb
    22: right_thumb
    23: left_hip
    24: right_hip
    25: left_knee
    26: right_knee
    27: left_ankle
    28: right_ankle
    29: left_heel
    30: right_heel
    31: left_foot_index
    32: right_foot_index
"""

# Landmark counts - FULL MediaPipe format for Kalidokit
LANDMARK_COUNTS = {
    "pose": 33,           # FULL 33-point MediaPipe pose (for Kalidokit)
    "face": 468,          # Full face mesh
    "left_hand": 21,      # Full hand
    "right_hand": 21,     # Full hand
}
TOTAL_LANDMARKS = sum(LANDMARK_COUNTS.values())  # 543

# Landmark index ranges in the final tensor
LANDMARK_RANGES = {
    "pose": (0, 33),              # Indices 0-32 (all 33 pose landmarks)
    "face": (33, 501),            # Indices 33-500 (468 face landmarks)
    "left_hand": (501, 522),      # Indices 501-521 (21 left hand landmarks)
    "right_hand": (522, 543),     # Indices 522-542 (21 right hand landmarks)
}

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

# Pose file format version - updated for Kalidokit compatibility
POSE_FORMAT_VERSION = "2.0-kalidokit"


@dataclass
class PipelineConfig:
    """Configuration parameters for the pose extraction pipeline."""
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
        return asdict(self)


# ==============================================================================
# SECTION 2: LOGGING SETUP
# ==============================================================================

def setup_logging(log_level: str, output_dir: Path) -> logging.Logger:
    """Set up logging with console and file handlers."""
    logger = logging.getLogger("wlasl_pose_kalidokit")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "pipeline_kalidokit.log"
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

@dataclass
class VideoMetadata:
    """Metadata extracted from a video file."""
    source_path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VideoDecoder:
    """Decodes video files and yields frames as RGB numpy arrays."""

    def __init__(self):
        self.logger = logging.getLogger("wlasl_pose_kalidokit.decoder")

    def decode(self, video_path: Path) -> Tuple[VideoMetadata, Generator[np.ndarray, None, None]]:
        """Decode a video file and return metadata with a frame generator."""
        video_path = Path(video_path)

        if not video_path.exists():
            raise VideoDecodeError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise VideoDecodeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 30.0
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

        def frame_generator():
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield frame_rgb
            cap.release()

        return metadata, frame_generator()


class VideoDecodeError(Exception):
    """Raised when video decoding fails."""
    pass


# ==============================================================================
# SECTION 4: LANDMARK EXTRACTOR (KALIDOKIT COMPATIBLE)
# ==============================================================================
"""
Extracts FULL 33-point pose landmarks for Kalidokit compatibility.

Output tensor shape: (T × 543 × 4) where:
- T = number of frames
- 543 = total landmarks (33 pose + 468 face + 21 left hand + 21 right hand)
- 4 = (x, y, z, confidence/visibility)
"""

class LandmarkExtractor:
    """
    Extracts pose, face, and hand landmarks from video frames.

    This version extracts FULL 33-point MediaPipe pose landmarks
    for Kalidokit compatibility.
    """

    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.logger = logging.getLogger("wlasl_pose_kalidokit.extractor")

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
            smooth_landmarks=True,  # Enable smoothing for better results
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

    def _init_new_api_landmarkers(self):
        """Initialize MediaPipe Tasks (new API for 0.10+)."""
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

        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self._get_model_path("pose_landmarker_full.task")
            ),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_segmentation_masks=False
        )
        self._pose_landmarker = PoseLandmarker.create_from_options(pose_options)

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
            "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
            "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        }

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
        """Extract landmarks from a sequence of frames."""
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
        """
        Extract landmarks using legacy Solutions API.

        KALIDOKIT COMPATIBLE: Extracts ALL 33 pose landmarks.
        """
        landmarks = np.full((TOTAL_LANDMARKS, 4), np.nan, dtype=np.float32)
        results = self._holistic.process(frame)

        # POSE LANDMARKS - Extract ALL 33 points for Kalidokit
        if results.pose_landmarks is not None:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                if i < 33:  # Ensure we only take 33 landmarks
                    landmarks[i] = [lm.x, lm.y, lm.z, lm.visibility]

        # FACE LANDMARKS
        if results.face_landmarks is not None:
            for i, lm in enumerate(results.face_landmarks.landmark):
                if i < 468:
                    idx = LANDMARK_RANGES["face"][0] + i
                    landmarks[idx] = [lm.x, lm.y, lm.z, 1.0]

        # LEFT HAND LANDMARKS
        if results.left_hand_landmarks is not None:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                if i < 21:
                    idx = LANDMARK_RANGES["left_hand"][0] + i
                    landmarks[idx] = [lm.x, lm.y, lm.z, 1.0]

        # RIGHT HAND LANDMARKS
        if results.right_hand_landmarks is not None:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                if i < 21:
                    idx = LANDMARK_RANGES["right_hand"][0] + i
                    landmarks[idx] = [lm.x, lm.y, lm.z, 1.0]

        return landmarks

    def _extract_frame_landmarks_new_api(
        self,
        frame: np.ndarray,
        timestamp_ms: int
    ) -> np.ndarray:
        """Extract landmarks using new Tasks API (MediaPipe 0.10+)."""
        if self._pose_landmarker is None:
            try:
                self._init_new_api_landmarkers()
            except Exception as e:
                self.logger.warning(f"Failed to init new API, falling back to legacy: {e}")
                self._use_new_api = False
                self._init_legacy_holistic()
                return self._extract_frame_landmarks_legacy(frame)

        landmarks = np.full((TOTAL_LANDMARKS, 4), np.nan, dtype=np.float32)
        mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=frame)

        try:
            # POSE LANDMARKS - ALL 33 points
            pose_result = self._pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            if pose_result.pose_landmarks:
                pose_lms = pose_result.pose_landmarks[0]
                for i, lm in enumerate(pose_lms):
                    if i < 33:
                        visibility = lm.visibility if hasattr(lm, 'visibility') else 1.0
                        landmarks[i] = [lm.x, lm.y, lm.z, visibility]

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
                        if i < 21:
                            landmarks[base_idx + i] = [lm.x, lm.y, lm.z, 1.0]

            # FACE LANDMARKS
            face_result = self._face_landmarker.detect_for_video(mp_image, timestamp_ms)
            if face_result.face_landmarks:
                face_lms = face_result.face_landmarks[0]
                for i, lm in enumerate(face_lms):
                    if i < 468:
                        idx = LANDMARK_RANGES["face"][0] + i
                        landmarks[idx] = [lm.x, lm.y, lm.z, 1.0]

        except Exception as e:
            self.logger.debug(f"Frame detection error: {e}")

        return landmarks

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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ==============================================================================
# SECTION 5: POSE SERIALIZER
# ==============================================================================

class PoseSerializer:
    """
    Serializes pose landmarks to .pose files.

    Output format includes metadata for Kalidokit compatibility:
    - landmarks: (T, 543, 3) float32 - x, y, z coordinates
    - confidence: (T, 543) float32 - detection confidence
    - kalidokit_compatible: bool - flag indicating format version
    """

    def __init__(self):
        self.logger = logging.getLogger("wlasl_pose_kalidokit.serializer")

    def save(
        self,
        landmarks: np.ndarray,
        metadata: VideoMetadata,
        output_path: Path
    ) -> Path:
        """Save pose landmarks to a .pose file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if landmarks.ndim != 3 or landmarks.shape[1] != TOTAL_LANDMARKS or landmarks.shape[2] != 4:
            raise SerializationError(
                f"Invalid landmark shape: {landmarks.shape}, expected (T, {TOTAL_LANDMARKS}, 4)"
            )

        coords = landmarks[:, :, :3]
        confidence = landmarks[:, :, 3]
        presence_mask = ~np.isnan(confidence)

        pose_data = {
            "landmarks": coords.astype(np.float32),
            "confidence": confidence.astype(np.float32),
            "presence_mask": presence_mask.astype(bool),
            "fps": np.float32(metadata.fps),
            "frame_count": np.int32(landmarks.shape[0]),
            "source_video": str(metadata.source_path),
            "landmark_layout": LANDMARK_RANGES,
            "landmark_counts": LANDMARK_COUNTS,
            "version": POSE_FORMAT_VERSION,
            "kalidokit_compatible": True,  # Flag for frontend
        }

        temp_path = output_path.with_suffix(".tmp")
        temp_path_with_npz = Path(str(temp_path) + ".npz") if not str(temp_path).endswith(".npz") else temp_path

        try:
            np.savez_compressed(str(temp_path), **pose_data)

            actual_temp_path = temp_path_with_npz if temp_path_with_npz.exists() else temp_path
            if not actual_temp_path.exists():
                actual_temp_path = Path(str(temp_path) + ".npz")

            actual_temp_path.rename(output_path)

            self.logger.debug(f"Saved Kalidokit-compatible pose file: {output_path}")
            return output_path

        except Exception as e:
            for p in [temp_path, temp_path_with_npz, Path(str(temp_path) + ".npz")]:
                if p.exists():
                    p.unlink()
            raise SerializationError(f"Failed to save pose file: {e}")

    @staticmethod
    def validate(pose_path: Path) -> bool:
        """Validate a .pose file for integrity."""
        try:
            pose_path = Path(pose_path)

            if not pose_path.exists():
                return False

            data = np.load(str(pose_path), allow_pickle=True)

            required_keys = {"landmarks", "confidence", "fps", "frame_count"}
            if not required_keys.issubset(set(data.keys())):
                return False

            landmarks = data["landmarks"]
            confidence = data["confidence"]

            if landmarks.ndim != 3:
                return False
            if landmarks.shape[1] != TOTAL_LANDMARKS:
                return False
            if landmarks.shape[2] != 3:
                return False
            if confidence.shape != landmarks.shape[:2]:
                return False
            if data["frame_count"] == 0:
                return False

            return True

        except Exception:
            return False

    @staticmethod
    def load(pose_path: Path) -> Dict[str, Any]:
        """Load a .pose file."""
        data = np.load(str(pose_path), allow_pickle=True)
        return {key: data[key] for key in data.keys()}


class SerializationError(Exception):
    """Raised when pose serialization fails."""
    pass


# ==============================================================================
# SECTION 6: MANIFEST & FAILURE LOGGING
# ==============================================================================

class ManifestLogger:
    """Logs successful pose extractions and failures to JSONL files."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.output_dir / "manifest_kalidokit.jsonl"
        self.failures_path = self.output_dir / "failures_kalidokit.jsonl"

        self.logger = logging.getLogger("wlasl_pose_kalidokit.manifest")

    def log_success(
        self,
        video_path: Path,
        pose_path: Path,
        metadata: VideoMetadata,
        processing_time_sec: float
    ):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "video_path": str(video_path),
            "pose_path": str(pose_path),
            "fps": metadata.fps,
            "frame_count": metadata.frame_count,
            "duration_sec": metadata.duration_sec,
            "processing_time_sec": round(processing_time_sec, 3),
            "format": "kalidokit-compatible",
            "total_landmarks": TOTAL_LANDMARKS,
        }

        self._append_jsonl(self.manifest_path, entry)

    def log_failure(
        self,
        video_path: Path,
        error: Exception,
        error_type: str = None
    ):
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
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ==============================================================================
# SECTION 7: PIPELINE ORCHESTRATOR
# ==============================================================================

class PoseExtractionPipeline:
    """Orchestrates the pose extraction pipeline for batch processing."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.input_dir = Path(config.input_dir)
        self.output_dir = Path(config.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logging(config.log_level, self.output_dir)
        self._save_config()
        self.manifest = ManifestLogger(self.output_dir)

    def _save_config(self):
        config_path = self.output_dir / "config_kalidokit.json"
        config_dict = self.config.to_dict()
        config_dict["run_timestamp"] = datetime.now().isoformat()
        config_dict["python_version"] = sys.version
        config_dict["format"] = "kalidokit-compatible"
        config_dict["total_landmarks"] = TOTAL_LANDMARKS
        config_dict["landmark_layout"] = {k: list(v) for k, v in LANDMARK_RANGES.items()}

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        self.logger.info(f"Configuration saved to {config_path}")

    def discover_videos(self) -> List[Path]:
        videos = []

        if self.config.recursive:
            for ext in VIDEO_EXTENSIONS:
                videos.extend(self.input_dir.rglob(f"*{ext}"))
        else:
            for ext in VIDEO_EXTENSIONS:
                videos.extend(self.input_dir.glob(f"*{ext}"))

        videos = sorted(videos)
        self.logger.info(f"Discovered {len(videos)} video files")

        return videos

    def get_output_path(self, video_path: Path) -> Path:
        if self.config.recursive:
            relative = video_path.relative_to(self.input_dir)
            output_path = self.output_dir / relative.with_suffix(".pose")
        else:
            output_path = self.output_dir / f"{video_path.stem}.pose"

        return output_path

    def filter_pending(self, videos: List[Path]) -> List[Path]:
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
                    self.logger.warning(f"Removing invalid pose file: {pose_path}")
                    pose_path.unlink()

            pending.append(video)

        if skipped > 0:
            self.logger.info(f"Skipping {skipped} already processed videos")

        return pending

    def run(self) -> Dict[str, int]:
        start_time = datetime.now()

        all_videos = self.discover_videos()

        if not all_videos:
            self.logger.warning("No video files found in input directory")
            return {"total": 0, "processed": 0, "skipped": 0, "failed": 0}

        pending_videos = self.filter_pending(all_videos)
        skipped_count = len(all_videos) - len(pending_videos)

        self.logger.info(f"Processing {len(pending_videos)} videos with {self.config.num_workers} workers")
        self.logger.info(f"Output format: Kalidokit-compatible ({TOTAL_LANDMARKS} landmarks)")

        success_count = 0
        failure_count = 0

        if self.config.num_workers == 1:
            for video in tqdm(pending_videos, desc="Extracting poses (Kalidokit)"):
                try:
                    self._process_single_video(video)
                    success_count += 1
                except Exception as e:
                    self.manifest.log_failure(video, e)
                    failure_count += 1
        else:
            worker_fn = partial(
                _worker_process_video,
                config=self.config,
                output_dir=self.output_dir
            )

            with Pool(processes=self.config.num_workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(worker_fn, pending_videos),
                    total=len(pending_videos),
                    desc="Extracting poses (Kalidokit)"
                ))

            for result in results:
                if result["status"] == "success":
                    success_count += 1
                else:
                    failure_count += 1

        elapsed = (datetime.now() - start_time).total_seconds()

        stats = {
            "total": len(all_videos),
            "processed": success_count,
            "skipped": skipped_count,
            "failed": failure_count,
            "elapsed_seconds": round(elapsed, 2),
            "format": "kalidokit-compatible",
            "landmarks_per_frame": TOTAL_LANDMARKS,
        }

        self.logger.info("=" * 60)
        self.logger.info("PIPELINE COMPLETE (Kalidokit-compatible format)")
        self.logger.info(f"  Total videos: {stats['total']}")
        self.logger.info(f"  Processed:    {stats['processed']}")
        self.logger.info(f"  Skipped:      {stats['skipped']}")
        self.logger.info(f"  Failed:       {stats['failed']}")
        self.logger.info(f"  Elapsed:      {stats['elapsed_seconds']:.1f}s")
        self.logger.info(f"  Format:       {TOTAL_LANDMARKS} landmarks/frame")
        self.logger.info("=" * 60)

        summary_path = self.output_dir / "summary_kalidokit.json"
        with open(summary_path, "w") as f:
            json.dump(stats, f, indent=2)

        return stats

    def _process_single_video(self, video_path: Path):
        import time
        start = time.time()

        output_path = self.get_output_path(video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        decoder = VideoDecoder()
        metadata, frames = decoder.decode(video_path)

        with LandmarkExtractor(
            model_complexity=self.config.model_complexity,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        ) as extractor:
            landmarks = extractor.extract_from_frames(frames)

        if landmarks.shape[0] == 0:
            raise VideoDecodeError("No frames extracted from video")

        serializer = PoseSerializer()
        serializer.save(landmarks, metadata, output_path)

        elapsed = time.time() - start
        self.manifest.log_success(video_path, output_path, metadata, elapsed)


def _worker_process_video(video_path: Path, config: PipelineConfig, output_dir: Path) -> Dict:
    """Worker function for multiprocessing."""
    import time

    try:
        start = time.time()

        if config.recursive:
            input_dir = Path(config.input_dir)
            relative = video_path.relative_to(input_dir)
            output_path = output_dir / relative.with_suffix(".pose")
        else:
            output_path = output_dir / f"{video_path.stem}.pose"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        decoder = VideoDecoder()
        metadata, frames = decoder.decode(video_path)

        with LandmarkExtractor(
            model_complexity=config.model_complexity,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence
        ) as extractor:
            landmarks = extractor.extract_from_frames(frames)

        if landmarks.shape[0] == 0:
            raise VideoDecodeError("No frames extracted from video")

        serializer = PoseSerializer()
        serializer.save(landmarks, metadata, output_path)

        elapsed = time.time() - start

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WLASL Pose Extraction Pipeline (Kalidokit-compatible) - Extract FULL 33-point pose landmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python wlasl_pose_pipeline_kalidokit.py --input_dir ./videos --output_dir ./poses

    # Process test subset
    python wlasl_pose_pipeline_kalidokit.py --input_dir ./wlasl-processed/test_subset/videos --output_dir ./poses_kalidokit --num_workers 4

    # Force reprocess all
    python wlasl_pose_pipeline_kalidokit.py --input_dir ./videos --output_dir ./poses --no_skip_existing
        """
    )

    parser.add_argument("--input_dir", "-i", type=str, required=True,
                        help="Directory containing input video files")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Directory for output .pose files")
    parser.add_argument("--num_workers", "-w", type=int, default=1,
                        help="Number of parallel worker processes (default: 1)")
    parser.add_argument("--model_complexity", "-m", type=int, choices=[0, 1, 2], default=2,
                        help="MediaPipe model complexity: 0=lite, 1=full, 2=heavy (default: 2)")
    parser.add_argument("--min_detection_confidence", type=float, default=0.5,
                        help="Minimum detection confidence threshold (default: 0.5)")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5,
                        help="Minimum tracking confidence threshold (default: 0.5)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip videos with existing valid .pose files (default: True)")
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing",
                        help="Reprocess all videos even if .pose files exist")
    parser.add_argument("--recursive", "-r", action="store_true", default=False,
                        help="Search input directory recursively for videos")
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Logging verbosity level (default: INFO)")

    return parser.parse_args()


def main():
    args = parse_args()

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

    print("=" * 60)
    print("WLASL POSE EXTRACTION PIPELINE (KALIDOKIT-COMPATIBLE)")
    print("=" * 60)
    print(f"Input directory:  {config.input_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Workers:          {config.num_workers}")
    print(f"Model complexity: {config.model_complexity}")
    print(f"Skip existing:    {config.skip_existing}")
    print(f"Landmarks/frame:  {TOTAL_LANDMARKS} (33 pose + 468 face + 42 hands)")
    print("=" * 60)
    print()

    pipeline = PoseExtractionPipeline(config)
    stats = pipeline.run()

    if stats["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
