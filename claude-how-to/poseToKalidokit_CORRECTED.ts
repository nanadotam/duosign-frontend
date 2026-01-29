/**
 * Pose to Kalidokit Conversion Utility (CORRECTED VERSION)
 *
 * Converts DuoSign pose data format to Kalidokit's expected MediaPipe format.
 * Based on official Kalidokit and MediaPipe documentation.
 * 
 * IMPORTANT: Kalidokit.Pose.solve requires two sets of landmarks:
 * - poseLandmarks3D: 3D world coordinates (ideally from MediaPipe worldLandmarks)
 * - poseLandmarks: 2D normalized screen coordinates (from MediaPipe landmarks)
 * 
 * COORDINATE SYSTEM FIX:
 * MediaPipe normalized landmarks use Y-down (screen space: 0=top, 1=bottom)
 * Kalidokit/VRM expect Y-up (3D space: 0=bottom, 1=top)
 * Solution: Invert Y-axis (1.0 - y) for both 2D and 3D landmarks
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
  landmarks: (number | null)[][];  // [523, 3] - normalized coordinates
  confidence: (number | null)[];   // [523]
  worldLandmarks?: (number | null)[][]; // [33, 3] - world coordinates (if available)
}

/**
 * MediaPipe Holistic landmark ranges
 * Based on the 523-landmark format from pose extraction:
 * - Pose: 33 body landmarks (0-32)
 * - Face: 468 face landmarks (33-500)
 * - Left Hand: 21 landmarks (501-521)
 * - Right Hand: 21 landmarks (502-522) - NOTE: overlaps slightly
 * 
 * Total: 33 + 468 + 21 + 21 = 543, but data has 523 landmarks
 */
const LANDMARK_RANGES = {
  // Pose landmarks (33 points: indices 0-32)
  pose: { start: 0, end: 33 },

  // Face landmarks (468 points: indices 33-500)
  face: { start: 33, end: 501 },

  // Left hand landmarks (21 points: indices 501-521)
  leftHand: { start: 501, end: 522 },

  // Right hand landmarks (21 points: indices 502-522)
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
 *   const riggedPose = Kalidokit.Pose.solve(
 *     kalidokitData.poseLandmarks3D,
 *     kalidokitData.poseLandmarks,
 *     { runtime: 'mediapipe', enableLegs: true }
 *   );
 * }
 * ```
 */
export function convertToKalidokitFormat(frameData: FrameData): KalidokitData {
  const { landmarks, confidence, worldLandmarks } = frameData;

  // Extract 2D normalized landmarks with Y-axis inversion
  const pose2D = extractNormalizedLandmarks(
    landmarks,
    LANDMARK_RANGES.pose,
    confidence
  );

  // Extract 3D landmarks (use world landmarks if available, otherwise approximate)
  let pose3D: MediaPipeLandmark[];
  if (worldLandmarks && worldLandmarks.length >= LANDMARK_RANGES.pose.end) {
    // Use actual world landmarks (already in Y-up coordinate system)
    pose3D = extractWorldLandmarks(
      worldLandmarks,
      LANDMARK_RANGES.pose,
      confidence
    );
  } else {
    // Approximate from normalized landmarks
    pose3D = extractNormalizedLandmarks(
      landmarks,
      LANDMARK_RANGES.pose,
      confidence
    );
  }

  const face = extractNormalizedLandmarks(
    landmarks,
    LANDMARK_RANGES.face,
    confidence
  );

  const leftHand = extractNormalizedLandmarks(
    landmarks,
    LANDMARK_RANGES.leftHand,
    confidence
  );

  const rightHand = extractNormalizedLandmarks(
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
 * Extracts normalized landmarks and converts from MediaPipe screen space to VRM 3D space
 * 
 * MediaPipe normalized landmarks use:
 * - x: 0 (left) → 1 (right)
 * - y: 0 (top) → 1 (bottom) [SCREEN SPACE]
 * - z: depth scale
 * 
 * Kalidokit/VRM expect:
 * - x: 0 (left) → 1 (right) [SAME]
 * - y: 0 (bottom) → 1 (top) [INVERTED]
 * - z: depth scale [SAME]
 *
 * @param allLandmarks - Full array of 523 landmarks
 * @param range - Start and end indices for the desired body part
 * @param confidence - Confidence values for each landmark
 * @returns Array of MediaPipe-formatted landmarks with Y-axis inverted
 */
function extractNormalizedLandmarks(
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

    // Create landmark with Y-axis inversion
    landmarks.push({
      x: landmark[0],                    // X unchanged (left-to-right)
      y: 1.0 - landmark[1],              // ✅ INVERT Y: screen space (Y-down) → 3D space (Y-up)
      z: landmark[2],                    // ✅ Z unchanged (depth scale is relative)
      visibility: (conf !== null && conf !== undefined) ? conf : 0.5
    });
  }

  return landmarks;
}

/**
 * Extracts world landmarks (if available from MediaPipe)
 * 
 * MediaPipe world landmarks are already in the correct Y-up coordinate system:
 * - x: left (-) → right (+) in meters
 * - y: down (-) → up (+) in meters [ALREADY Y-UP]
 * - z: back (-) → front (+) in meters
 * 
 * Origin: Center between hips
 * 
 * These can be used directly without transformation.
 *
 * @param worldLandmarks - World landmarks from MediaPipe (33 pose landmarks)
 * @param range - Start and end indices for the desired body part
 * @param confidence - Confidence values for each landmark
 * @returns Array of landmarks in world coordinates (no transformation needed)
 */
function extractWorldLandmarks(
  worldLandmarks: (number | null)[][],
  range: { start: number; end: number },
  confidence: (number | null)[]
): MediaPipeLandmark[] {
  const landmarks: MediaPipeLandmark[] = [];

  for (let i = range.start; i < range.end; i++) {
    const landmark = worldLandmarks[i];
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

    // World landmarks are already in correct Y-up coordinate system
    // Use them directly - NO transformation needed
    landmarks.push({
      x: landmark[0],   // meters, left-right
      y: landmark[1],   // meters, bottom-top (already Y-up! No inversion needed)
      z: landmark[2],   // meters, back-front
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
