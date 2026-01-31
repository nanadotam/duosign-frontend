# Code Walkthrough - Kalidokit-Compatible Pose System

**Date:** January 29, 2026
**Author:** Claude (AI Assistant)

---

## Overview

This document provides a detailed walkthrough of the code changes made to implement Kalidokit-compatible pose extraction and rendering for the DuoSign VRM avatar system.

---

## Part 1: Pose Extraction Pipeline

### File: `duosign_algo/pose_extraction/wlasl_pose_pipeline_kalidokit.py`

This is the main extraction pipeline that processes WLASL videos and extracts MediaPipe Holistic landmarks in a format compatible with Kalidokit.

#### Key Constants

```python
# Kalidokit requires exactly these landmark counts
LANDMARK_COUNTS = {
    "pose": 33,        # Full 33-point MediaPipe BlazePose
    "face": 468,       # MediaPipe Face Mesh
    "left_hand": 21,   # MediaPipe Hand landmarks
    "right_hand": 21,  # MediaPipe Hand landmarks
}

TOTAL_LANDMARKS = 543  # Sum of all landmark types

# Landmark ranges in the combined array
LANDMARK_RANGES = {
    "pose": (0, 33),
    "face": (33, 501),
    "left_hand": (501, 522),
    "right_hand": (522, 543),
}
```

#### Main Extraction Function

```python
def extract_pose_data(video_path: Path, save_path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract pose data from a video using MediaPipe Holistic.

    Returns Kalidokit-compatible format with full 33-point pose landmarks.
    """
```

The extraction process:

1. **Open video with OpenCV** - Read frames at native resolution and FPS
2. **Initialize MediaPipe Holistic** - Configure for pose, face, and hand detection
3. **Process each frame:**
   - Convert BGR to RGB for MediaPipe
   - Run holistic detection
   - Extract all 543 landmarks (or NaN for missing)
   - Store confidence values
4. **Save as NumPy archive** - Compressed `.pose` file with metadata

#### Landmark Extraction Logic

```python
# Extract full 33 pose landmarks (Kalidokit requirement)
if results.pose_landmarks:
    for i, lm in enumerate(results.pose_landmarks.landmark):
        idx = i  # Direct mapping, indices 0-32
        frame_landmarks[idx] = [lm.x, lm.y, lm.z]
        frame_confidence[idx] = lm.visibility

# Extract face landmarks (indices 33-500)
if results.face_landmarks:
    for i, lm in enumerate(results.face_landmarks.landmark):
        idx = LANDMARK_RANGES["face"][0] + i
        frame_landmarks[idx] = [lm.x, lm.y, lm.z]
        frame_confidence[idx] = 1.0  # Face landmarks don't have visibility

# Extract hand landmarks (indices 501-521, 522-542)
for hand_key, hand_landmarks in [
    ("left_hand", results.left_hand_landmarks),
    ("right_hand", results.right_hand_landmarks)
]:
    if hand_landmarks:
        start_idx = LANDMARK_RANGES[hand_key][0]
        for i, lm in enumerate(hand_landmarks.landmark):
            idx = start_idx + i
            frame_landmarks[idx] = [lm.x, lm.y, lm.z]
            frame_confidence[idx] = 1.0
```

**Key Difference from Original:**
The original pipeline selected only 13 upper body points and remapped indices. This version preserves all 33 pose landmarks with their original MediaPipe indices.

---

### File: `duosign_algo/pose_extraction/pose_to_json_kalidokit.py`

Converts NumPy `.pose` archives to JSON for browser consumption.

#### Conversion Logic

```python
def convert_pose_to_json(pose_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert pose data to JSON-serializable format.

    Optimizations:
    - Rounds coordinates to 6 decimal places
    - Converts NaN to null for JSON compatibility
    - Includes metadata for frontend validation
    """
```

Key features:
- **NaN Handling:** MediaPipe returns NaN for undetected landmarks; converted to `null` in JSON
- **Precision Control:** 6 decimal places for coordinates, 4 for confidence
- **Metadata Inclusion:** Format version, compatibility flags, landmark layout

---

## Part 2: Frontend Conversion Utility

### File: `src/utils/poseToKalidokit.ts`

Converts JSON pose data to Kalidokit's expected input format.

#### Updated Landmark Ranges

```typescript
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
```

#### Pose Landmark Index Reference

```typescript
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
```

#### Main Conversion Function

```typescript
export function convertToKalidokitFormat(frameData: FrameData): KalidokitData {
  // Extract 2D landmarks (normalized screen coordinates)
  const pose2D = extractLandmarks(landmarks, LANDMARK_RANGES.pose, confidence);

  // Extract 3D landmarks (same data, used by Kalidokit for depth)
  const pose3D = extractLandmarks3D(landmarks, LANDMARK_RANGES.pose, confidence);

  // Extract face and hands
  const face = extractLandmarks(landmarks, LANDMARK_RANGES.face, confidence);
  const leftHand = extractLandmarks(landmarks, LANDMARK_RANGES.leftHand, confidence);
  const rightHand = extractLandmarks(landmarks, LANDMARK_RANGES.rightHand, confidence);

  // Validate data quality
  const validPoseCount = pose2D.filter(lm => (lm.visibility ?? 0) > 0.3).length;

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
```

**Why Two Pose Arrays?**

Kalidokit.Pose.solve() requires both 2D and 3D landmarks:
- `poseLandmarks` (2D): Normalized screen coordinates for angle calculations
- `poseLandmarks3D` (3D): World coordinates for depth-aware rigging

Since our pose data doesn't have separate world coordinates, we use the same normalized data for both. Kalidokit handles this gracefully.

---

## Part 3: Avatar Renderer

### File: `src/components/app/AvatarRenderer.tsx`

Applies Kalidokit solutions to VRM avatar bones.

#### Pose Application Flow

```typescript
function applyPoseToAvatar(vrm: VRM, poseData: PoseData, frameIndex: number): void {
  // 1. Extract frame data
  const frameData = {
    landmarks: poseData.landmarks[frameIndex],
    confidence: poseData.confidence[frameIndex]
  };

  // 2. Convert to Kalidokit format
  const kalidokitData = convertToKalidokitFormat(frameData);

  // 3. Solve pose using Kalidokit
  const riggedPose = Kalidokit.Pose.solve(
    kalidokitData.poseLandmarks3D,  // 3D world landmarks
    kalidokitData.poseLandmarks,    // 2D normalized landmarks
    {
      runtime: 'mediapipe',
      enableLegs: false  // Upper body focus for sign language
    }
  );

  // 4. Apply to VRM bones
  if (riggedPose) {
    rigPose(vrm, riggedPose);
  }

  // 5. Solve and apply hands
  if (kalidokitData.hasValidLeftHand) {
    const riggedLeftHand = Kalidokit.Hand.solve(kalidokitData.leftHandLandmarks, 'Left');
    if (riggedLeftHand) {
      rigHand(vrm, riggedLeftHand, 'Left');
    }
  }
  // ... same for right hand
}
```

#### Pose Rigging Function

```typescript
function rigPose(vrm: VRM, riggedPose: any): void {
  const { humanoid } = vrm;

  const applyRotation = (boneName: string, euler: { x: number; y: number; z: number } | undefined) => {
    if (!euler) return;

    const bone = humanoid.getNormalizedBoneNode(boneName as any);
    if (bone) {
      const targetQuat = eulerToQuaternion(euler);
      bone.quaternion.slerp(targetQuat, ROTATION_SMOOTHING);
    }
  };

  // Disable body rotations (can cause avatar to lean/rotate unexpectedly)
  // applyRotation('hips', riggedPose.Hips?.rotation);
  // applyRotation('spine', riggedPose.Spine);

  // Apply arm rotations - critical for sign language
  applyRotation('leftUpperArm', riggedPose.LeftUpperArm);
  applyRotation('rightUpperArm', riggedPose.RightUpperArm);
  applyRotation('leftLowerArm', riggedPose.LeftLowerArm);
  applyRotation('rightLowerArm', riggedPose.RightLowerArm);

  // Apply shoulder and hand rotations
  applyRotation('leftShoulder', riggedPose.LeftShoulder);
  applyRotation('rightShoulder', riggedPose.RightShoulder);
  applyRotation('leftHand', riggedPose.LeftHand);
  applyRotation('rightHand', riggedPose.RightHand);
}
```

**Key Design Decisions:**

1. **Body rotations disabled:** Hips, spine, and chest rotations are commented out because they can cause the avatar to lean or rotate unexpectedly. For sign language, upper body should remain stable.

2. **Smoothing applied:** `ROTATION_SMOOTHING = 0.7` provides smooth interpolation between frames using quaternion slerp.

3. **Euler to Quaternion conversion:** Kalidokit outputs euler angles in radians, which must be converted to quaternions for Three.js bone rotation.

#### Hand Rigging Function

```typescript
function rigHand(vrm: VRM, riggedHand: any, side: 'Left' | 'Right'): void {
  const prefix = side.toLowerCase();

  const applyRotation = (boneName: string, euler: { x: number; y: number; z: number } | undefined) => {
    if (!euler || typeof euler.z !== 'number') return;

    const bone = humanoid.getNormalizedBoneNode(boneName as any);
    if (bone) {
      const targetQuat = eulerToQuaternion(euler);
      bone.quaternion.slerp(targetQuat, ROTATION_SMOOTHING);
    }
  };

  // Apply wrist rotation
  if (riggedHand[`${side}Wrist`]) {
    applyRotation(`${prefix}Hand`, riggedHand[`${side}Wrist`]);
  }

  // Apply finger rotations
  const fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'];
  const joints = ['Proximal', 'Intermediate', 'Distal'];

  fingers.forEach((finger) => {
    joints.forEach((joint) => {
      const rotationKey = `${side}${finger}${joint}`;  // e.g., RightThumbProximal
      const boneKey = `${prefix}${finger}${joint}`;    // e.g., rightThumbProximal
      applyRotation(boneKey, riggedHand[rotationKey]);
    });
  });
}
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VIDEO PROCESSING                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   WLASL Video (.mp4)                                                 │
│         │                                                             │
│         ▼                                                             │
│   ┌─────────────────────────────────────┐                           │
│   │   wlasl_pose_pipeline_kalidokit.py  │                           │
│   │   - MediaPipe Holistic              │                           │
│   │   - Extract 543 landmarks           │                           │
│   │   - Save as NumPy archive           │                           │
│   └─────────────────────────────────────┘                           │
│         │                                                             │
│         ▼                                                             │
│   ┌─────────────────────────────────────┐                           │
│   │   pose_to_json_kalidokit.py         │                           │
│   │   - Convert NaN → null              │                           │
│   │   - Round coordinates               │                           │
│   │   - Output JSON                     │                           │
│   └─────────────────────────────────────┘                           │
│         │                                                             │
│         ▼                                                             │
│   Pose JSON (public/poses_kalidokit/)                               │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND RENDERING                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   Pose JSON                                                          │
│         │                                                             │
│         ▼                                                             │
│   ┌─────────────────────────────────────┐                           │
│   │   poseToKalidokit.ts                │                           │
│   │   - convertToKalidokitFormat()      │                           │
│   │   - Extract pose/face/hands         │                           │
│   │   - Validate landmarks              │                           │
│   └─────────────────────────────────────┘                           │
│         │                                                             │
│         ▼                                                             │
│   ┌─────────────────────────────────────┐                           │
│   │   Kalidokit Library                 │                           │
│   │   - Pose.solve() → body rotations   │                           │
│   │   - Hand.solve() → finger rotations │                           │
│   └─────────────────────────────────────┘                           │
│         │                                                             │
│         ▼                                                             │
│   ┌─────────────────────────────────────┐                           │
│   │   AvatarRenderer.tsx                │                           │
│   │   - rigPose() → apply body          │                           │
│   │   - rigHand() → apply fingers       │                           │
│   │   - Euler → Quaternion conversion   │                           │
│   │   - Smoothing interpolation         │                           │
│   └─────────────────────────────────────┘                           │
│         │                                                             │
│         ▼                                                             │
│   VRM Avatar (Three.js Scene)                                        │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Testing the Implementation

### 1. Verify Extraction

```bash
# Check extraction output
ls -la duosign_algo/pose_extraction/poses_kalidokit/

# View extraction summary
cat duosign_algo/pose_extraction/poses_kalidokit/summary_kalidokit.json
```

### 2. Verify JSON Conversion

```bash
# Check JSON files
ls -la public/poses_kalidokit/

# View conversion summary
cat public/poses_kalidokit/_conversion_summary.json
```

### 3. Test in Browser

1. Start development server: `npm run dev`
2. Navigate to avatar rendering page
3. Load a pose file from `poses_kalidokit/`
4. Verify arms move correctly with the skeleton

### 4. Debug Console Output

The renderer logs debugging info on frame 0:

```javascript
console.log('Frame 0 rigged with Kalidokit', {
  poseValid: !!riggedPose,
  leftUpperArm: riggedPose?.LeftUpperArm,
  rightUpperArm: riggedPose?.RightUpperArm,
  leftShoulderLandmark: kalidokitData.poseLandmarks[11],
  rightShoulderLandmark: kalidokitData.poseLandmarks[12],
});
```

---

## Common Issues and Solutions

### Issue: Avatar arms still incorrect

**Possible causes:**
1. Using old pose data (523 landmarks instead of 543)
2. Loading from wrong directory (`poses/` instead of `poses_kalidokit/`)

**Solution:** Verify the loaded JSON has `total_landmarks: 543` and `kalidokit_compatible: true`

### Issue: Avatar body rotates unexpectedly

**Cause:** Hips/spine rotations being applied

**Solution:** Ensure body rotations are commented out in `rigPose()`:
```typescript
// applyRotation('hips', riggedPose.Hips?.rotation);
// applyRotation('spine', riggedPose.Spine);
```

### Issue: Hands not moving

**Cause:** Hand landmarks not detected in source video

**Solution:** Check `hasValidLeftHand` and `hasValidRightHand` flags. If false, hand was not visible in that frame.

---

## Future Enhancements

1. **Face Rigging:** Add `Kalidokit.Face.solve()` for facial expressions
2. **Leg Support:** Enable leg rigging for full body signs
3. **Smoothing Tuning:** Adjust `ROTATION_SMOOTHING` based on sign type
4. **Confidence Thresholding:** Skip low-confidence frames instead of rendering them
