#!/usr/bin/env python3
"""
Skeleton Normalization for Scale-Invariant Pose Representation
==============================================================

This module normalizes landmark positions to a standard scale based on
anatomical proportions. This makes animations consistent regardless of
video resolution or subject size.

Key Features:
    - Shoulder-width based normalization (robust anatomical reference)
    - Median-based scale computation (robust to outliers)
    - Metadata preservation for inverse transformation
    - Graceful handling of missing landmarks

Author: Nana Kwaku Amoako
Date: 2026-01-31
"""

import numpy as np
from typing import Dict, Tuple
from .quaternion_solver import PoseLandmark


def compute_bone_length(
    landmarks: np.ndarray,  # Shape: (543, 3)
    start_idx: int,
    end_idx: int
) -> float:
    """
    Compute Euclidean distance between two landmarks (bone length).
    
    Args:
        landmarks (np.ndarray): 3D landmark positions, shape (543, 3)
        start_idx (int): Index of proximal landmark
        end_idx (int): Index of distal landmark
    
    Returns:
        float: Bone length (Euclidean distance), or NaN if landmarks missing
    
    Example:
        >>> landmarks = np.random.rand(543, 3)
        >>> shoulder_width = compute_bone_length(
        ...     landmarks,
        ...     PoseLandmark.LEFT_SHOULDER,
        ...     PoseLandmark.RIGHT_SHOULDER
        ... )
    """
    start_pos = landmarks[start_idx]
    end_pos = landmarks[end_idx]
    
    # Handle missing landmarks
    if np.isnan(start_pos).any() or np.isnan(end_pos).any():
        return np.nan
    
    return np.linalg.norm(end_pos - start_pos)


def compute_reference_scale(landmarks: np.ndarray) -> float:
    """
    Compute reference scale from shoulder width.
    
    Shoulder width is chosen as the reference because it:
    - Is visible in most frontal sign language videos
    - Remains relatively constant across frames
    - Is easy to measure reliably
    - Provides intuitive scale (typical adult shoulder width ≈ 0.4-0.5m)
    
    Falls back to hip width if shoulders are not visible.
    
    Args:
        landmarks (np.ndarray): 3D landmark positions, shape (543, 3)
    
    Returns:
        float: Shoulder width (or hip width as fallback), or 1.0 if both unavailable
    
    Example:
        >>> landmarks = np.load("frame_landmarks.npy")
        >>> scale = compute_reference_scale(landmarks)
        >>> print(f"Reference scale: {scale:.3f}")
    """
    # Primary: shoulder width
    shoulder_width = compute_bone_length(
        landmarks,
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.RIGHT_SHOULDER
    )
    
    # Fallback: if shoulders not visible, use hip width
    if np.isnan(shoulder_width):
        shoulder_width = compute_bone_length(
            landmarks,
            PoseLandmark.LEFT_HIP,
            PoseLandmark.RIGHT_HIP
        )
    
    # Default scale if both fail (rare, indicates very poor detection)
    if np.isnan(shoulder_width) or shoulder_width < 0.01:
        shoulder_width = 1.0
    
    return shoulder_width


def normalize_landmarks(
    landmarks: np.ndarray,      # Shape: (T, 543, 3)
    reference_scale: float = 1.0
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize landmark sequence to standard scale.
    
    Algorithm:
    1. Compute shoulder width for each frame
    2. Take median shoulder width (robust to outliers)
    3. Scale all landmarks so median shoulder width = reference_scale
    
    This ensures consistent animation regardless of:
    - Video resolution (720p vs 1080p vs 4K)
    - Subject size (child vs adult)
    - Camera distance (close-up vs wide shot)
    
    Args:
        landmarks (np.ndarray): Raw landmark positions, shape (T, 543, 3)
            - T: Number of frames
            - 543: Number of landmarks
            - 3: Coordinates (x, y, z)
        reference_scale (float): Target shoulder width (default: 1.0)
            - 1.0 is a good default (normalized units)
            - Can use real-world units (e.g., 0.45 for 45cm shoulder width)
    
    Returns:
        Tuple containing:
        - normalized (np.ndarray): Normalized landmarks, same shape as input
        - metadata (Dict[str, float]): Normalization metadata
            - "original_shoulder_width": Median shoulder width before normalization
            - "scale_factor": Multiplicative factor applied
            - "reference_scale": Target scale used
    
    Example:
        >>> # Load pose data
        >>> data = np.load("sign.pose", allow_pickle=True)
        >>> landmarks = data["landmarks"]  # Shape: (100, 543, 3)
        >>> 
        >>> # Normalize to unit scale
        >>> normalized, metadata = normalize_landmarks(landmarks, reference_scale=1.0)
        >>> 
        >>> print(f"Original shoulder width: {metadata['original_shoulder_width']:.3f}")
        >>> print(f"Scale factor: {metadata['scale_factor']:.3f}")
        >>> 
        >>> # Verify normalization
        >>> # Compute shoulder width in first frame of normalized data
        >>> norm_shoulder = compute_bone_length(
        ...     normalized[0],
        ...     PoseLandmark.LEFT_SHOULDER,
        ...     PoseLandmark.RIGHT_SHOULDER
        ... )
        >>> print(f"Normalized shoulder width: {norm_shoulder:.3f}")  # Should be ≈1.0
    """
    T = landmarks.shape[0]
    
    # Compute shoulder width for each frame
    shoulder_widths = []
    for t in range(T):
        width = compute_reference_scale(landmarks[t])
        if not np.isnan(width):
            shoulder_widths.append(width)
    
    # Use median for robustness (ignores outlier frames)
    # Median is more robust than mean for this application because:
    # - Outliers from detection errors don't skew the result
    # - Frames with partial occlusion are automatically downweighted
    if len(shoulder_widths) > 0:
        median_shoulder_width = np.median(shoulder_widths)
    else:
        # Fallback if no valid measurements (very rare)
        median_shoulder_width = 1.0
    
    # Compute scale factor
    scale_factor = reference_scale / median_shoulder_width
    
    # Apply scaling to all landmarks
    # Broadcasting: (T, 543, 3) * scalar = (T, 543, 3)
    normalized = landmarks * scale_factor
    
    # Metadata for potential inverse transformation
    # This allows converting back to original scale if needed
    metadata = {
        "original_shoulder_width": float(median_shoulder_width),
        "scale_factor": float(scale_factor),
        "reference_scale": float(reference_scale)
    }
    
    return normalized, metadata


def denormalize_landmarks(
    normalized_landmarks: np.ndarray,
    metadata: Dict[str, float]
) -> np.ndarray:
    """
    Inverse transformation: convert normalized landmarks back to original scale.
    
    This is useful for:
    - Visualization in original video coordinates
    - Comparison with raw MediaPipe output
    - Debugging normalization issues
    
    Args:
        normalized_landmarks (np.ndarray): Normalized landmarks, shape (T, 543, 3)
        metadata (Dict[str, float]): Metadata from normalize_landmarks()
    
    Returns:
        np.ndarray: Landmarks in original scale, same shape as input
    
    Example:
        >>> # Normalize
        >>> normalized, metadata = normalize_landmarks(landmarks)
        >>> 
        >>> # Do some processing...
        >>> 
        >>> # Convert back to original scale
        >>> original_scale = denormalize_landmarks(normalized, metadata)
        >>> 
        >>> # Verify round-trip (should be very close to original)
        >>> np.allclose(original_scale, landmarks, atol=1e-6)
        True
    """
    scale_factor = metadata["scale_factor"]
    return normalized_landmarks / scale_factor


# Example usage
if __name__ == "__main__":
    """
    Example: Normalize a synthetic pose sequence and verify results.
    """
    # Create synthetic data with varying shoulder widths
    T = 100
    landmarks = np.random.rand(T, 543, 3)
    
    # Set shoulder positions with varying widths (simulating camera zoom)
    for t in range(T):
        # Shoulder width varies from 0.3 to 0.6 (simulating zoom)
        width = 0.3 + 0.3 * (t / T)
        landmarks[t, PoseLandmark.LEFT_SHOULDER] = [-width/2, 0, 0]
        landmarks[t, PoseLandmark.RIGHT_SHOULDER] = [width/2, 0, 0]
    
    # Normalize
    normalized, metadata = normalize_landmarks(landmarks, reference_scale=1.0)
    
    # Verify normalization
    print("Normalization Results:")
    print(f"  Original median shoulder width: {metadata['original_shoulder_width']:.3f}")
    print(f"  Scale factor: {metadata['scale_factor']:.3f}")
    print(f"  Reference scale: {metadata['reference_scale']:.3f}")
    
    # Check shoulder width in normalized data
    normalized_widths = []
    for t in range(T):
        width = compute_bone_length(
            normalized[t],
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.RIGHT_SHOULDER
        )
        normalized_widths.append(width)
    
    median_normalized = np.median(normalized_widths)
    print(f"\nVerification:")
    print(f"  Median normalized shoulder width: {median_normalized:.3f}")
    print(f"  Expected: {metadata['reference_scale']:.3f}")
    print(f"  Match: {np.isclose(median_normalized, metadata['reference_scale'])}")
    
    # Test round-trip
    denormalized = denormalize_landmarks(normalized, metadata)
    round_trip_error = np.max(np.abs(denormalized - landmarks))
    print(f"\nRound-trip test:")
    print(f"  Max error: {round_trip_error:.2e}")
    print(f"  Success: {round_trip_error < 1e-10}")
