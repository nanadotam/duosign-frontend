"""
Pose Processing Package
=======================

Advanced pose processing algorithms for sign language avatar animation.

Modules:
    - filters: 1€ filter and temporal smoothing
    - quaternion_solver: Landmark-to-quaternion conversion
    - skeleton_normalizer: Scale normalization
    - pose_v3_converter: Complete conversion pipeline
"""

__version__ = "3.0.0"
__author__ = "Nana Kwaku Amoako"

from .filters import OneEuroFilter, LandmarkFilter
from .quaternion_solver import (
    landmarks_to_bone_quaternions,
    process_landmark_sequence,
    BONE_DEFINITIONS
)
from .skeleton_normalizer import normalize_landmarks
from .pose_v3_converter import convert_pose_to_v3

__all__ = [
    "OneEuroFilter",
    "LandmarkFilter",
    "landmarks_to_bone_quaternions",
    "process_landmark_sequence",
    "BONE_DEFINITIONS",
    "normalize_landmarks",
    "convert_pose_to_v3",
]
