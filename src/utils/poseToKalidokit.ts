/**
 * Pose to Kalidokit Conversion Utility
 *
 * Converts DuoSign pose data format to Kalidokit's expected MediaPipe format.
 * Handles null values and provides safe landmark extraction for VRM avatar animation.
 * 
 * IMPORTANT: Kalidokit.Pose.solve requires two sets of landmarks:
 * - poseLandmarks3D: 3D world coordinates (used for depth calculations)
 * - poseLandmarks: 2D normalized screen coordinates (used for angle calculations)
 * 
 * This version works with the new LandmarkFrame format (structured with separate arrays)
 */

import type { LandmarkFrame, LandmarkPoseData } from '@/types/pose';

/**
 * MediaPipe landmark structure expected by Kalidokit
 */
export interface MediaPipeLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

/**
 * Kalidokit input data structure
 * Extended to support both 2D and 3D pose landmarks as required by Kalidokit.Pose.solve
 */
export interface KalidokitData {
  /** 2D normalized pose landmarks (for Pose.solve second parameter) */
  poseLandmarks: MediaPipeLandmark[];
  /** 3D world pose landmarks (for Pose.solve first parameter) */
  poseLandmarks3D: MediaPipeLandmark[];
  faceLandmarks: MediaPipeLandmark[];
  leftHandLandmarks: MediaPipeLandmark[];
  rightHandLandmarks: MediaPipeLandmark[];
  /** Validation flags */
  hasValidPose: boolean;
  hasValidLeftHand: boolean;
  hasValidRightHand: boolean;
}

/**
 * Mapping from our 13-point pose format to standard MediaPipe 33-point indices
 * 
 * Our pose data uses reduced 13 upper body landmarks:
 *   0: nose
 *   1: left_shoulder, 2: right_shoulder
 *   3: left_elbow, 4: right_elbow
 *   5: left_wrist, 6: right_wrist
 *   7: left_pinky, 8: right_pinky
 *   9: left_index, 10: right_index
 *   11: left_thumb, 12: right_thumb
 */
export const POSE_LANDMARK_INDICES = {
  NOSE: 0,
  LEFT_SHOULDER: 1,
  RIGHT_SHOULDER: 2,
  LEFT_ELBOW: 3,
  RIGHT_ELBOW: 4,
  LEFT_WRIST: 5,
  RIGHT_WRIST: 6,
  LEFT_PINKY: 7,
  RIGHT_PINKY: 8,
  LEFT_INDEX: 9,
  RIGHT_INDEX: 10,
  LEFT_THUMB: 11,
  RIGHT_THUMB: 12,
} as const;

/** Minimum valid landmarks required for pose/hand rigging */
const MIN_POSE_LANDMARKS = 6;   // Need at least shoulders, elbows, wrists (6 key points)
const MIN_HAND_LANDMARKS = 15;

/**
 * Converts a LandmarkFrame (new structured format) to Kalidokit's expected format
 *
 * @param frame - Single frame data with separated landmark arrays
 * @returns Kalidokit-compatible landmark data structure with validation flags
 *
 * @example
 * ```typescript
 * const frame = poseData.frames[frameIndex];
 * const kalidokitData = convertFrameToKalidokit(frame);
 * if (kalidokitData.hasValidPose) {
 *   // Safe to call Kalidokit.Pose.solve
 * }
 * ```
 */
export function convertFrameToKalidokit(frame: LandmarkFrame): KalidokitData {
  // Extract pose landmarks as Kalidokit format
  const pose2D = frame.pose_landmarks.map((lm, i) => ({
    x: lm[0],
    y: lm[1],
    z: lm[2],
    visibility: frame.confidences?.pose ?? 0.9
  }));

  const pose3D = [...pose2D]; // Same data for 3D

  const faceLandmarks = frame.face_landmarks.map((lm) => ({
    x: lm[0],
    y: lm[1],
    z: lm[2],
    visibility: frame.confidences?.face ?? 0.9
  }));

  const leftHandLandmarks = frame.left_hand_landmarks.map((lm) => ({
    x: lm[0],
    y: lm[1],
    z: lm[2],
    visibility: frame.confidences?.left_hand ?? 0.9
  }));

  const rightHandLandmarks = frame.right_hand_landmarks.map((lm) => ({
    x: lm[0],
    y: lm[1],
    z: lm[2],
    visibility: frame.confidences?.right_hand ?? 0.9
  }));

  // Count valid landmarks
  const validPoseCount = pose2D.filter(lm => (lm.visibility ?? 0) > 0.3).length;
  const validLeftHandCount = leftHandLandmarks.filter(lm => lm.x !== 0 || lm.y !== 0).length;
  const validRightHandCount = rightHandLandmarks.filter(lm => lm.x !== 0 || lm.y !== 0).length;

  return {
    poseLandmarks: pose2D,
    poseLandmarks3D: pose3D,
    faceLandmarks,
    leftHandLandmarks,
    rightHandLandmarks,
    hasValidPose: validPoseCount >= MIN_POSE_LANDMARKS,
    hasValidLeftHand: validLeftHandCount >= MIN_HAND_LANDMARKS,
    hasValidRightHand: validRightHandCount >= MIN_HAND_LANDMARKS,
  };
}

/**
 * Legacy frame data format (flat 523-landmark array)
 * Kept for backwards compatibility
 */
export interface LegacyFrameData {
  landmarks: (number | null)[][];  // [523, 3]
  confidence: (number | null)[];   // [523]
}

/**
 * Landmark ranges for legacy 523-landmark format
 */
const LEGACY_LANDMARK_RANGES = {
  pose: { start: 0, end: 13 },
  face: { start: 13, end: 481 },
  leftHand: { start: 481, end: 502 },
  rightHand: { start: 502, end: 523 }
} as const;

/**
 * Converts legacy DuoSign pose data format to Kalidokit's expected format
 * For backwards compatibility with old [T, 523, 3] format
 *
 * @param frameData - Single frame data with landmarks and confidence values
 * @returns Kalidokit-compatible landmark data structure with validation flags
 */
export function convertToKalidokitFormat(frameData: LegacyFrameData): KalidokitData {
  const { landmarks, confidence } = frameData;

  const pose2D = extractLandmarks(landmarks, LEGACY_LANDMARK_RANGES.pose, confidence);
  const pose3D = extractLandmarks3D(landmarks, LEGACY_LANDMARK_RANGES.pose, confidence);
  const face = extractLandmarks(landmarks, LEGACY_LANDMARK_RANGES.face, confidence);
  const leftHand = extractLandmarks(landmarks, LEGACY_LANDMARK_RANGES.leftHand, confidence);
  const rightHand = extractLandmarks(landmarks, LEGACY_LANDMARK_RANGES.rightHand, confidence);

  const validPoseCount = pose2D.filter(lm => (lm.visibility ?? 0) > 0.3).length;
  const validLeftHandCount = leftHand.filter(lm => lm.x !== 0 || lm.y !== 0).length;
  const validRightHandCount = rightHand.filter(lm => lm.x !== 0 || lm.y !== 0).length;

  return {
    poseLandmarks: pose2D,
    poseLandmarks3D: pose3D,
    faceLandmarks: face,
    leftHandLandmarks: leftHand,
    rightHandLandmarks: rightHand,
    hasValidPose: validPoseCount >= MIN_POSE_LANDMARKS,
    hasValidLeftHand: validLeftHandCount >= MIN_HAND_LANDMARKS,
    hasValidRightHand: validRightHandCount >= MIN_HAND_LANDMARKS,
  };
}

function extractLandmarks(
  allLandmarks: (number | null)[][],
  range: { start: number; end: number },
  confidence: (number | null)[]
): MediaPipeLandmark[] {
  const landmarks: MediaPipeLandmark[] = [];

  for (let i = range.start; i < range.end; i++) {
    const landmark = allLandmarks[i];
    const conf = confidence[i];

    if (!landmark || landmark[0] === null || landmark[1] === null || landmark[2] === null) {
      landmarks.push({ x: 0, y: 0, z: 0, visibility: 0 });
      continue;
    }

    landmarks.push({
      x: landmark[0],
      y: landmark[1],
      z: landmark[2],
      visibility: (conf !== null && conf !== undefined) ? conf : 0.5
    });
  }

  return landmarks;
}

function extractLandmarks3D(
  allLandmarks: (number | null)[][],
  range: { start: number; end: number },
  confidence: (number | null)[]
): MediaPipeLandmark[] {
  return extractLandmarks(allLandmarks, range, confidence);
}

/**
 * Checks if a landmark has sufficient visibility for animation
 */
export function isLandmarkVisible(
  landmark: MediaPipeLandmark,
  threshold: number = 0.5
): boolean {
  return (landmark.visibility ?? 0) > threshold;
}

/**
 * Checks if a set of landmarks has sufficient data for rigging
 */
export function hasValidPoseData(
  landmarks: MediaPipeLandmark[],
  minVisibleCount: number = 5
): boolean {
  const visibleCount = landmarks.filter(lm => isLandmarkVisible(lm)).length;
  return visibleCount >= minVisibleCount;
}
