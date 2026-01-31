/**
 * Pose to Kalidokit Conversion Utility
 *
 * Converts DuoSign pose data format to Kalidokit's expected MediaPipe format.
 * Handles null values and provides safe landmark extraction for VRM avatar animation.
 * 
 * IMPORTANT: Kalidokit.Pose.solve requires two sets of landmarks:
 * - poseLandmarks3D: 3D world coordinates (used for depth calculations)
 * - poseLandmarks: 2D normalized screen coordinates (used for angle calculations)
 */

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
 * Frame data from DuoSign pose extraction
 */
export interface FrameData {
  landmarks: (number | null)[][];  // [523, 3]
  confidence: (number | null)[];   // [523]
}

/**
 * Landmark ranges for Kalidokit-compatible 543-landmark format
 *
 * This format uses the FULL 33-point MediaPipe BlazePose model,
 * which is required for proper Kalidokit.Pose.solve() operation.
 *
 * MediaPipe Pose Landmarks (33 points):
 *   0: nose
 *   1: left_eye_inner, 2: left_eye, 3: left_eye_outer
 *   4: right_eye_inner, 5: right_eye, 6: right_eye_outer
 *   7: left_ear, 8: right_ear
 *   9: mouth_left, 10: mouth_right
 *   11: left_shoulder, 12: right_shoulder
 *   13: left_elbow, 14: right_elbow
 *   15: left_wrist, 16: right_wrist
 *   17: left_pinky, 18: right_pinky
 *   19: left_index, 20: right_index
 *   21: left_thumb, 22: right_thumb
 *   23: left_hip, 24: right_hip
 *   25: left_knee, 26: right_knee
 *   27: left_ankle, 28: right_ankle
 *   29: left_heel, 30: right_heel
 *   31: left_foot_index, 32: right_foot_index
 *
 * Total: 33 + 468 + 21 + 21 = 543 landmarks
 */
const LANDMARK_RANGES = {
  // Pose landmarks (33 full body points: indices 0-32)
  pose: { start: 0, end: 33 },

  // Face landmarks (468 points: indices 33-500)
  face: { start: 33, end: 501 },

  // Left hand landmarks (21 points: indices 501-521)
  leftHand: { start: 501, end: 522 },

  // Right hand landmarks (21 points: indices 522-542)
  rightHand: { start: 522, end: 543 }
} as const;

/**
 * Standard MediaPipe 33-point pose landmark indices
 * These match Kalidokit's expected format exactly
 */
export const POSE_LANDMARK_INDICES = {
  NOSE: 0,
  LEFT_EYE_INNER: 1,
  LEFT_EYE: 2,
  LEFT_EYE_OUTER: 3,
  RIGHT_EYE_INNER: 4,
  RIGHT_EYE: 5,
  RIGHT_EYE_OUTER: 6,
  LEFT_EAR: 7,
  RIGHT_EAR: 8,
  MOUTH_LEFT: 9,
  MOUTH_RIGHT: 10,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_PINKY: 17,
  RIGHT_PINKY: 18,
  LEFT_INDEX: 19,
  RIGHT_INDEX: 20,
  LEFT_THUMB: 21,
  RIGHT_THUMB: 22,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
  LEFT_HEEL: 29,
  RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31,
  RIGHT_FOOT_INDEX: 32,
} as const;

/** Minimum valid landmarks required for pose/hand rigging */
const MIN_POSE_LANDMARKS = 15;  // Need key body points for Kalidokit rigging
const MIN_HAND_LANDMARKS = 15;

/**
 * Converts DuoSign pose data format to Kalidokit's expected format
 *
 * @param frameData - Single frame data with landmarks and confidence values
 * @returns Kalidokit-compatible landmark data structure with validation flags
 *
 * @example
 * ```typescript
 * const frameData = {
 *   landmarks: poseData.landmarks[frameIndex],
 *   confidence: poseData.confidence[frameIndex]
 * };
 * const kalidokitData = convertToKalidokitFormat(frameData);
 * if (kalidokitData.hasValidPose) {
 *   // Safe to call Kalidokit.Pose.solve
 * }
 * ```
 */
export function convertToKalidokitFormat(frameData: FrameData): KalidokitData {
  const { landmarks, confidence } = frameData;

  // Extract 2D landmarks for each body part (normalized screen coordinates)
  const pose2D = extractLandmarks(
    landmarks,
    LANDMARK_RANGES.pose,
    confidence
  );

  // Create 3D world landmarks from the same data
  // Kalidokit uses z-depth for 3D calculations
  const pose3D = extractLandmarks3D(
    landmarks,
    LANDMARK_RANGES.pose,
    confidence
  );

  const face = extractLandmarks(
    landmarks,
    LANDMARK_RANGES.face,
    confidence
  );

  const leftHand = extractLandmarks(
    landmarks,
    LANDMARK_RANGES.leftHand,
    confidence
  );

  const rightHand = extractLandmarks(
    landmarks,
    LANDMARK_RANGES.rightHand,
    confidence
  );

  // Count valid landmarks for validation
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

/**
 * Extracts and formats a range of landmarks from the full landmark array
 *
 * @param allLandmarks - Full array of 523 landmarks
 * @param range - Start and end indices for the desired body part
 * @param confidence - Confidence values for each landmark
 * @returns Array of MediaPipe-formatted landmarks
 *
 * @remarks
 * Handles null/undefined values gracefully by creating zero-visibility landmarks.
 * This ensures Kalidokit receives valid data even when some landmarks are missing.
 */
function extractLandmarks(
  allLandmarks: (number | null)[][],
  range: { start: number; end: number },
  confidence: (number | null)[]
): MediaPipeLandmark[] {
  const landmarks: MediaPipeLandmark[] = [];

  for (let i = range.start; i < range.end; i++) {
    const landmark = allLandmarks[i];
    const conf = confidence[i];

    // Handle missing or invalid landmarks
    if (!landmark || landmark[0] === null || landmark[1] === null || landmark[2] === null) {
      // Create a placeholder landmark with zero visibility
      landmarks.push({
        x: 0,
        y: 0,
        z: 0,
        visibility: 0
      });
      continue;
    }

    // Create valid landmark - keep native MediaPipe coordinates
    // Kalidokit with runtime: 'mediapipe' handles coordinate transformation internally
    landmarks.push({
      x: landmark[0],
      y: landmark[1],
      z: landmark[2],
      // Use confidence if available, otherwise default to low visibility
      visibility: (conf !== null && conf !== undefined) ? conf : 0.5
    });
  }

  return landmarks;
}

/**
 * Extracts landmarks as 3D world coordinates for Kalidokit.Pose.solve
 * 
 * Kalidokit expects landmarks in the same normalized format as MediaPipe.
 * The solve function uses both 2D and 3D landmarks to calculate rotations,
 * where 3D provides depth information for more accurate rigging.
 * 
 * Note: We use the same normalized coordinates for both, as our pose data
 * doesn't have separate world coordinates. Kalidokit handles this gracefully.
 *
 * @param allLandmarks - Full array of 523 landmarks
 * @param range - Start and end indices for the desired body part  
 * @param confidence - Confidence values for each landmark
 * @returns Array of landmarks with 3D coordinates
 */
function extractLandmarks3D(
  allLandmarks: (number | null)[][],
  range: { start: number; end: number },
  confidence: (number | null)[]
): MediaPipeLandmark[] {
  const landmarks: MediaPipeLandmark[] = [];

  for (let i = range.start; i < range.end; i++) {
    const landmark = allLandmarks[i];
    const conf = confidence[i];

    // Handle missing or invalid landmarks
    if (!landmark || landmark[0] === null || landmark[1] === null || landmark[2] === null) {
      landmarks.push({
        x: 0,
        y: 0,
        z: 0,
        visibility: 0
      });
      continue;
    }

    // Keep same normalized coordinates as 2D landmarks
    // The z-value provides depth information for 3D rigging
    // Kalidokit with runtime: 'mediapipe' handles coordinate transformation internally
    landmarks.push({
      x: landmark[0],
      y: landmark[1],
      z: landmark[2],
      visibility: (conf !== null && conf !== undefined) ? conf : 0.5
    });
  }

  return landmarks;
}


/**
 * Checks if a landmark has sufficient visibility for animation
 *
 * @param landmark - MediaPipe landmark to check
 * @param threshold - Minimum visibility threshold (default: 0.5)
 * @returns True if landmark is visible enough to use
 */
export function isLandmarkVisible(
  landmark: MediaPipeLandmark,
  threshold: number = 0.5
): boolean {
  return (landmark.visibility ?? 0) > threshold;
}

/**
 * Checks if a set of landmarks has sufficient data for rigging
 *
 * @param landmarks - Array of landmarks to validate
 * @param minVisibleCount - Minimum number of visible landmarks required
 * @returns True if sufficient landmarks are visible
 */
export function hasValidPoseData(
  landmarks: MediaPipeLandmark[],
  minVisibleCount: number = 5
): boolean {
  const visibleCount = landmarks.filter(lm => isLandmarkVisible(lm)).length;
  return visibleCount >= minVisibleCount;
}
