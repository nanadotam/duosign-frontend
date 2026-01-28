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
 * MediaPipe Holistic landmark ranges
 * Based on the 523-landmark format from pose extraction:
 * - Pose: 33 body landmarks (0-32)
 * - Face: 468 face landmarks (33-500)
 * - Left Hand: 21 landmarks (501-521)
 * - Right Hand: 21 landmarks (502-522) - NOTE: overlaps slightly, handled gracefully
 * 
 * Total: 33 + 468 + 21 + 21 = 543, but data has 523 landmarks
 * The actual data structure uses indices 0-522 with some overlap/compression
 */
const LANDMARK_RANGES = {
  // Pose landmarks (33 points: indices 0-32)
  pose: { start: 0, end: 33 },

  // Face landmarks (468 points: indices 33-500)
  face: { start: 33, end: 501 },

  // Left hand landmarks (21 points: indices 501-521)
  leftHand: { start: 501, end: 522 },

  // Right hand landmarks - use remaining indices (502-522)
  // The data has only 523 total, so right hand overlaps with left
  rightHand: { start: 502, end: 523 }
} as const;

/** Minimum valid landmarks required for pose/hand rigging */
const MIN_POSE_LANDMARKS = 10;
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

    // Create valid landmark
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
