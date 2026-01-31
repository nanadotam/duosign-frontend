#!/usr/bin/env python3
"""
Quaternion-Based Skeletal Solver for MediaPipe Landmarks
========================================================

This module converts 3D landmark positions to bone rotations represented as
unit quaternions. It bypasses Euler angle intermediate representation to
eliminate gimbal lock and enable smooth interpolation.

Key Features:
    - Direct landmark-to-quaternion conversion using scipy
    - Angular velocity calculation for adaptive smoothing
    - Robust handling of edge cases (180° rotations, missing landmarks)
    - Native quaternion output (scalar-last format for THREE.js)

Author: Nana Kwaku Amoako
Date: 2026-01-31
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


# MediaPipe Pose landmark indices (33-point model)
class PoseLandmark:
    """
    MediaPipe Pose landmark indices for the 33-point BlazePose model.
    
    Reference:
        https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    """
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


@dataclass
class BoneDefinition:
    """
    Defines a bone by its start and end landmark indices.
    
    Attributes:
        name (str): VRM bone name (e.g., 'leftUpperArm')
        start_idx (int): MediaPipe landmark index for bone start (proximal joint)
        end_idx (int): MediaPipe landmark index for bone end (distal joint)
        rest_direction (np.ndarray): Default direction vector in T-pose (unit vector)
    
    Example:
        >>> # Define left upper arm bone
        >>> left_upper_arm = BoneDefinition(
        ...     name="leftUpperArm",
        ...     start_idx=PoseLandmark.LEFT_SHOULDER,
        ...     end_idx=PoseLandmark.LEFT_ELBOW,
        ...     rest_direction=np.array([-1, 0, 0])  # Points left in T-pose
        ... )
    """
    name: str
    start_idx: int
    end_idx: int
    rest_direction: np.ndarray  # Unit vector in T-pose


# Bone hierarchy for upper body (sign language focus)
# T-pose assumption: arms extended horizontally, palms down, facing forward
BONE_DEFINITIONS = [
    # Spine and head
    BoneDefinition(
        "spine", 
        PoseLandmark.LEFT_HIP, 
        PoseLandmark.LEFT_SHOULDER, 
        np.array([0, 1, 0])  # Upward in T-pose
    ),
    BoneDefinition(
        "neck", 
        PoseLandmark.LEFT_SHOULDER, 
        PoseLandmark.NOSE,
        np.array([0, 1, 0])  # Upward in T-pose
    ),
    BoneDefinition(
        "head", 
        PoseLandmark.NOSE, 
        PoseLandmark.LEFT_EYE,
        np.array([0, 1, 0])  # Upward in T-pose
    ),
    
    # Left arm
    BoneDefinition(
        "leftUpperArm", 
        PoseLandmark.LEFT_SHOULDER, 
        PoseLandmark.LEFT_ELBOW,
        np.array([-1, 0, 0])  # Points left in T-pose
    ),
    BoneDefinition(
        "leftLowerArm", 
        PoseLandmark.LEFT_ELBOW, 
        PoseLandmark.LEFT_WRIST,
        np.array([-1, 0, 0])  # Points left in T-pose
    ),
    BoneDefinition(
        "leftHand", 
        PoseLandmark.LEFT_WRIST, 
        PoseLandmark.LEFT_INDEX,
        np.array([-1, 0, 0])  # Points left in T-pose
    ),
    
    # Right arm (mirrored)
    BoneDefinition(
        "rightUpperArm", 
        PoseLandmark.RIGHT_SHOULDER, 
        PoseLandmark.RIGHT_ELBOW,
        np.array([1, 0, 0])  # Points right in T-pose
    ),
    BoneDefinition(
        "rightLowerArm", 
        PoseLandmark.RIGHT_ELBOW, 
        PoseLandmark.RIGHT_WRIST,
        np.array([1, 0, 0])  # Points right in T-pose
    ),
    BoneDefinition(
        "rightHand", 
        PoseLandmark.RIGHT_WRIST, 
        PoseLandmark.RIGHT_INDEX,
        np.array([1, 0, 0])  # Points right in T-pose
    ),
]


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        v (np.ndarray): Input vector of any dimension
    
    Returns:
        np.ndarray: Unit vector in same direction, or zero vector if input is zero
    
    Example:
        >>> v = np.array([3, 4, 0])
        >>> normalize_vector(v)
        array([0.6, 0.8, 0.])
    """
    norm = np.linalg.norm(v)
    if norm < 1e-8:  # Avoid division by zero
        return np.zeros_like(v)
    return v / norm


def compute_bone_rotation(
    current_direction: np.ndarray,
    rest_direction: np.ndarray
) -> np.ndarray:
    """
    Compute quaternion rotation from rest pose to current pose.
    
    Uses scipy's Rotation.align_vectors which implements the Kabsch algorithm
    for robust rotation computation. This handles edge cases like 180° rotations
    better than manual quaternion construction.
    
    Args:
        current_direction (np.ndarray): Current bone direction (unit vector)
        rest_direction (np.ndarray): Rest pose bone direction (unit vector)
    
    Returns:
        np.ndarray: Quaternion [x, y, z, w] representing rotation
            (scalar-last format for THREE.js compatibility)
    
    Example:
        >>> # Compute rotation from T-pose (left) to overhead (up)
        >>> rest = np.array([-1, 0, 0])  # Left
        >>> current = np.array([0, 1, 0])  # Up
        >>> quat = compute_bone_rotation(current, rest)
        >>> # quat represents 90° rotation around z-axis
    """
    # Normalize inputs (ensure unit vectors)
    current_direction = normalize_vector(current_direction)
    rest_direction = normalize_vector(rest_direction)
    
    # Handle degenerate case: zero-length vectors
    if np.linalg.norm(current_direction) < 1e-8 or np.linalg.norm(rest_direction) < 1e-8:
        return np.array([0, 0, 0, 1])  # Identity quaternion (no rotation)
    
    # Scipy's align_vectors finds optimal rotation from rest to current
    # It uses the Kabsch algorithm internally for robustness
    rotation, _ = Rotation.align_vectors(
        [current_direction],  # Target vectors
        [rest_direction]      # Source vectors
    )
    
    # Return quaternion in scalar-last format (x, y, z, w) for THREE.js
    # Scipy uses scalar-last by default
    return rotation.as_quat()


def compute_angular_velocity(
    q_prev: np.ndarray,
    q_curr: np.ndarray,
    dt: float
) -> float:
    """
    Compute angular velocity between two quaternions.
    
    This is used for velocity-adaptive smoothing in the frontend renderer.
    
    Formula:
        ω = 2 * arccos(|q_prev · q_curr|) / dt
    
    Args:
        q_prev (np.ndarray): Previous quaternion [x, y, z, w]
        q_curr (np.ndarray): Current quaternion [x, y, z, w]
        dt (float): Time delta in seconds
    
    Returns:
        float: Angular velocity in radians/second
    
    Example:
        >>> q1 = np.array([0, 0, 0, 1])  # Identity (no rotation)
        >>> q2 = np.array([0, 0, 0.707, 0.707])  # 90° around z-axis
        >>> dt = 1/30  # 30 FPS
        >>> velocity = compute_angular_velocity(q1, q2, dt)
        >>> # velocity ≈ 47 rad/s (90° in 1/30 second)
    """
    # Quaternion dot product (measures similarity)
    # Take absolute value because q and -q represent same rotation
    dot = np.abs(np.dot(q_prev, q_curr))
    
    # Clamp to valid range for arccos (numerical stability)
    dot = np.clip(dot, -1.0, 1.0)
    
    # Angular difference (radians)
    angle = 2.0 * np.arccos(dot)
    
    # Angular velocity
    if dt > 1e-8:
        return angle / dt
    return 0.0


def landmarks_to_bone_quaternions(
    landmarks: np.ndarray,      # Shape: (543, 3) - single frame
    confidence: np.ndarray,     # Shape: (543,) - confidence per landmark
    prev_quaternions: Optional[Dict[str, np.ndarray]] = None,  # Previous frame quaternions
    fps: float = 30.0
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, float]]:
    """
    Convert MediaPipe landmarks to bone quaternions for a single frame.
    
    Algorithm:
    1. For each bone, compute current direction vector from landmarks
    2. Compute rotation from rest pose direction to current direction
    3. Calculate angular velocity if previous frame available
    4. Compute confidence as minimum of start/end landmark confidences
    
    Args:
        landmarks (np.ndarray): 3D landmark positions with shape (543, 3)
            - 543 landmarks from MediaPipe Holistic
            - 3 coordinates (x, y, z)
        confidence (np.ndarray): Detection confidence per landmark, shape (543,)
            - Values in range [0, 1]
        prev_quaternions (Optional[Dict]): Quaternions from previous frame
            - Used for velocity calculation
            - None for first frame
        fps (float): Frame rate in Hz (default: 30.0)
            - Used for velocity calculation
    
    Returns:
        Tuple containing:
        - rotations (Dict[str, np.ndarray]): Bone name → quaternion [x, y, z, w]
        - velocities (Dict[str, float]): Bone name → angular velocity (rad/s)
        - confidences (Dict[str, float]): Bone name → confidence (0 to 1)
    
    Example:
        >>> # Single frame conversion
        >>> landmarks = np.load("frame_0_landmarks.npy")  # Shape: (543, 3)
        >>> confidence = np.load("frame_0_confidence.npy")  # Shape: (543,)
        >>> 
        >>> rotations, velocities, confidences = landmarks_to_bone_quaternions(
        ...     landmarks, confidence, prev_quaternions=None, fps=30.0
        ... )
        >>> 
        >>> print(rotations["leftUpperArm"])  # [x, y, z, w]
        >>> print(velocities["leftUpperArm"])  # rad/s
        >>> print(confidences["leftUpperArm"])  # 0 to 1
    """
    rotations = {}
    velocities = {}
    confidences = {}
    
    dt = 1.0 / fps  # Time between frames
    
    for bone in BONE_DEFINITIONS:
        # Extract landmark positions for bone endpoints
        start_pos = landmarks[bone.start_idx]  # Proximal joint
        end_pos = landmarks[bone.end_idx]      # Distal joint
        
        # Check if landmarks are valid (not NaN)
        if np.isnan(start_pos).any() or np.isnan(end_pos).any():
            # Missing data: use identity rotation (no movement)
            rotations[bone.name] = np.array([0, 0, 0, 1])
            velocities[bone.name] = 0.0
            confidences[bone.name] = 0.0
            continue
        
        # Compute current bone direction
        current_direction = end_pos - start_pos
        current_direction = normalize_vector(current_direction)
        
        # Compute rotation from rest pose to current pose
        quaternion = compute_bone_rotation(current_direction, bone.rest_direction)
        rotations[bone.name] = quaternion
        
        # Compute angular velocity (if previous frame available)
        if prev_quaternions is not None and bone.name in prev_quaternions:
            velocity = compute_angular_velocity(
                prev_quaternions[bone.name],
                quaternion,
                dt
            )
            velocities[bone.name] = velocity
        else:
            velocities[bone.name] = 0.0
        
        # Compute confidence as minimum of endpoint confidences
        # Rationale: bone is only as reliable as its least confident endpoint
        bone_confidence = min(
            confidence[bone.start_idx],
            confidence[bone.end_idx]
        )
        confidences[bone.name] = bone_confidence
    
    return rotations, velocities, confidences


def process_landmark_sequence(
    landmarks: np.ndarray,      # Shape: (T, 543, 3) - T frames
    confidence: np.ndarray,     # Shape: (T, 543)
    fps: float = 30.0
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, float]], List[Dict[str, float]]]:
    """
    Process entire landmark sequence to quaternion sequence.
    
    This is the main entry point for converting a full video's worth of landmarks
    to bone rotations. Processes frames sequentially to enable velocity calculation.
    
    Args:
        landmarks (np.ndarray): Landmark positions for all frames, shape (T, 543, 3)
            - T: Number of frames
            - 543: Number of landmarks
            - 3: Coordinates (x, y, z)
        confidence (np.ndarray): Confidence values for all frames, shape (T, 543)
        fps (float): Video frame rate in Hz
    
    Returns:
        Tuple of lists (one entry per frame):
        - rotations_sequence (List[Dict]): List of rotation dicts
        - velocities_sequence (List[Dict]): List of velocity dicts
        - confidences_sequence (List[Dict]): List of confidence dicts
    
    Example:
        >>> # Load full video pose data
        >>> data = np.load("sign.pose", allow_pickle=True)
        >>> landmarks = data["landmarks"]  # Shape: (100, 543, 3)
        >>> confidence = data["confidence"]  # Shape: (100, 543)
        >>> fps = float(data["fps"])
        >>> 
        >>> # Convert to quaternions
        >>> rotations, velocities, confidences = process_landmark_sequence(
        ...     landmarks, confidence, fps
        ... )
        >>> 
        >>> # Access results
        >>> print(f"Processed {len(rotations)} frames")
        >>> print(f"Frame 0, leftUpperArm: {rotations[0]['leftUpperArm']}")
        >>> print(f"Frame 0, velocity: {velocities[0]['leftUpperArm']:.4f} rad/s")
    """
    T = landmarks.shape[0]  # Number of frames
    
    rotations_sequence = []
    velocities_sequence = []
    confidences_sequence = []
    
    prev_quaternions = None
    
    # Process frames sequentially (needed for velocity calculation)
    for t in range(T):
        rotations, velocities, confidences_frame = landmarks_to_bone_quaternions(
            landmarks[t],
            confidence[t],
            prev_quaternions,
            fps
        )
        
        rotations_sequence.append(rotations)
        velocities_sequence.append(velocities)
        confidences_sequence.append(confidences_frame)
        
        # Save for next frame's velocity calculation
        prev_quaternions = rotations
    
    return rotations_sequence, velocities_sequence, confidences_sequence


# Example usage
if __name__ == "__main__":
    """
    Example: Convert synthetic landmarks to quaternions to demonstrate usage.
    """
    # Create synthetic data: arm raising from T-pose to overhead
    T = 60  # 2 seconds at 30 FPS
    landmarks = np.zeros((T, 543, 3))
    confidence = np.ones((T, 543))
    
    # Animate left arm raising
    for t in range(T):
        angle = (t / T) * np.pi / 2  # 0 to 90 degrees
        
        # Left shoulder (fixed)
        landmarks[t, PoseLandmark.LEFT_SHOULDER] = [0, 0, 0]
        
        # Left elbow (rotating upward)
        landmarks[t, PoseLandmark.LEFT_ELBOW] = [
            -np.cos(angle),  # x: starts at -1 (left), moves to 0
            np.sin(angle),   # y: starts at 0, moves to 1 (up)
            0                # z: no change
        ]
        
        # Left wrist (following elbow)
        landmarks[t, PoseLandmark.LEFT_WRIST] = [
            -2 * np.cos(angle),
            2 * np.sin(angle),
            0
        ]
        
        # Left index finger (for hand bone)
        landmarks[t, PoseLandmark.LEFT_INDEX] = [
            -2.2 * np.cos(angle),
            2.2 * np.sin(angle),
            0
        ]
    
    # Process sequence
    rotations, velocities, confidences = process_landmark_sequence(
        landmarks, confidence, fps=30.0
    )
    
    # Print results for first and last frame
    print("Frame 0 (T-pose):")
    print(f"  leftUpperArm rotation: {rotations[0]['leftUpperArm']}")
    print(f"  leftUpperArm velocity: {velocities[0]['leftUpperArm']:.4f} rad/s")
    print(f"  leftUpperArm confidence: {confidences[0]['leftUpperArm']:.2f}")
    
    print(f"\nFrame {T-1} (overhead):")
    print(f"  leftUpperArm rotation: {rotations[T-1]['leftUpperArm']}")
    print(f"  leftUpperArm velocity: {velocities[T-1]['leftUpperArm']:.4f} rad/s")
    print(f"  leftUpperArm confidence: {confidences[T-1]['leftUpperArm']:.2f}")
    
    # Verify quaternion is unit norm
    quat = rotations[T-1]['leftUpperArm']
    norm = np.linalg.norm(quat)
    print(f"\nQuaternion norm check: {norm:.6f} (should be ≈1.0)")
