"""
Pose Extractor from input videos
This program was authored by Nana Kwaku Amoako
Based on the work of Moryossef et al. (2023) for their
work on https://sign.mt 
"""

#depracated

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MediaPipePoseExtractor:
    """
    Extract poses from videos using MediaPipe Holistic
    """

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,     # 0=lite, 1=full, 2=heavy
            # static_image_mode=False,
            # enable_segmentation=True,
            # enable_tracking=True,
        )

        # MediaPipe Holistic Components
        self.component_names = [
            'POSE_LANDMARKS',       # 33 Body landmarks
            'LEFT_HAND_LANDMARKS',  # 21 Left hand landmarks
            'RIGHT_HAND_LANDMARKS', # 21 Right hand landmarks
            'FACE_LANDMARKS',       # 468 Face landmarks
        ]


    def extract_poses(self, video_path: Path) -> Dict:
        """
        Extract poses from a single video file

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary of pose landmarks: 'landmarks', 'fps', 'frame_count'
        """
        if not video_path.exists():
            logger.error(f"Video file does not exist: {video_path}")
            return None

        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

