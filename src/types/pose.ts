/**
 * Landmark-based Pose Data Types
 * ================================
 * 
 * TypeScript interfaces for raw landmark pose data from the API.
 * These differ from PoseDataV3 (quaternion-based) as they contain
 * raw x, y, z coordinates instead of bone rotations.
 * 
 * Author: Nana Kwaku Amoako
 * Date: 2026-02-06
 */

/**
 * Confidence scores for each body region in a frame.
 */
export interface FrameConfidences {
  pose: number;
  face: number;
  left_hand: number;
  right_hand: number;
}

/**
 * Single frame of landmark data.
 * 
 * Each landmark is [x, y, z] in normalized coordinates (0-1).
 */
export interface LandmarkFrame {
  /** Body pose landmarks (33 points) */
  pose_landmarks: [number, number, number][];
  /** Face mesh landmarks (468 points) */
  face_landmarks: [number, number, number][];
  /** Left hand landmarks (21 points) */
  left_hand_landmarks: [number, number, number][];
  /** Right hand landmarks (21 points) */
  right_hand_landmarks: [number, number, number][];
  /** Per-region confidence scores */
  confidences: FrameConfidences;
}

/**
 * Connection pair for skeleton drawing.
 */
export type Connection = [number, number];

/**
 * Skeleton connection definitions.
 */
export interface SkeletonConnections {
  pose: Connection[];
  hand: Connection[];
}

/**
 * Complete landmark pose data from API.
 */
export interface LandmarkPoseData {
  /** Gloss name */
  gloss: string;
  /** Frame rate in fps */
  fps: number;
  /** Total number of frames */
  frame_count: number;
  /** Frame data array */
  frames: LandmarkFrame[];
  /** Skeleton connection definitions */
  connections: SkeletonConnections;
}

/**
 * Debug metrics for a single frame.
 */
export interface FrameDebugMetrics {
  frame_index: number;
  pose_detection: number;
  face_detection: number;
  left_hand_detection: number;
  right_hand_detection: number;
  overall_confidence: number;
}

/**
 * Detailed debug information from API.
 */
export interface PoseDebugInfo {
  gloss: string;
  file_path: string;
  fps: number;
  frame_count: number;
  duration_sec: number;
  landmark_counts: {
    pose: number;
    face: number;
    left_hand: number;
    right_hand: number;
    total: number;
  };
  detection_rates: {
    pose: number;
    face: number;
    left_hand: number;
    right_hand: number;
  };
  confidence_stats: {
    mean: number;
    std: number;
    min: number;
    max: number;
  };
  frame_metrics: FrameDebugMetrics[];
}

/**
 * API response for listing available poses.
 */
export interface PoseListResponse {
  count: number;
  glosses: string[];
}
