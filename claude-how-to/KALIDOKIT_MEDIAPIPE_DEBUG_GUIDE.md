# 🔍 Complete KalidoKit + MediaPipe Debugging Guide

Based on official documentation from Kalidokit GitHub and MediaPipe AI Edge docs.

---

## 📚 Official Documentation Summary

### Kalidokit.Pose.solve() Signature

According to the [official Kalidokit README](https://github.com/yeemachine/kalidokit):

```javascript
Kalidokit.Pose.solve(poseWorld3DArray, poseLandmarkArray, {
    runtime: "tfjs", // `mediapipe` or `tfjs`
    video: HTMLVideoElement,
    imageSize: { height: 0, width: 0 },
    enableLegs: true,
});
```

**Parameters:**
1. **poseWorld3DArray** (First parameter): 33 landmarks in **world coordinates**
2. **poseLandmarkArray** (Second parameter): 33 landmarks in **normalized image coordinates**

### MediaPipe Coordinate Systems

According to [MediaPipe Pose documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker):

#### Normalized Landmarks (poseLandmarks)
- **x and y**: Normalized between 0.0 and 1.0 by image width (x) and height (y)
- **z**: Depth with midpoint of hips as origin. Smaller = closer to camera
- **Coordinate System**: 
  - X: 0 (left) → 1 (right)
  - Y: **0 (top) → 1 (bottom)** ⚠️ Screen coordinates
  - Z: Negative (close) → Positive (far)

#### World Landmarks (poseWorldLandmarks)
- **x, y, z**: Real-world 3D coordinates in meters
- **Origin**: Center between hips
- **Coordinate System**:
  - X: Left (-) → Right (+)
  - Y: **Down (-) → Up (+)** ⚠️ 3D world coordinates (Y-up convention)
  - Z: Back (-) → Front (+)

---

## 🚨 THE PROBLEM - Three Coordinate System Mismatches

### Issue 1: MediaPipe Normalized Y vs VRM Y

**MediaPipe normalized landmarks** (what you're passing):
- Y: 0 (top) → 1 (bottom) [screen space, Y-down]

**VRM/Three.js expects**:
- Y: -1 (bottom) → 1 (top) [3D space, Y-up]

**Your current code**: Passes MediaPipe Y directly → Avatar inverted ❌

### Issue 2: Missing World Coordinates

**Kalidokit expects**:
- First param: `poseWorldLandmarks` (3D world coords in meters, Y-up)
- Second param: `poseLandmarks` (2D normalized coords, Y-down)

**Your code provides**:
- First param: Converted from normalized landmarks (NOT true world coords)
- Second param: Converted from normalized landmarks

**Result**: Kalidokit doesn't get proper depth/scale info for 3D rigging

### Issue 3: Z-Axis Camera vs World Space

**MediaPipe Z**:
- Negative Z = closer to camera
- Depth relative to camera position

**VRM Z**:
- Positive Z = forward in world
- Depth in 3D world space

---

## ✅ SOLUTION: Proper Coordinate Transformation

### Option A: If You Have MediaPipe World Landmarks (BEST)

If your pose extraction includes `poseWorldLandmarks`, use them directly:

```typescript
// In your pose extraction or data structure
export interface PoseData {
  landmarks: (number | null)[][];      // Normalized landmarks [523, 3]
  worldLandmarks: (number | null)[][]; // World landmarks [33, 3] - ADD THIS
  confidence: (number | null)[];
}

// In convertToKalidokitFormat
export function convertToKalidokitFormat(frameData: FrameData): KalidokitData {
  // Extract world landmarks (already in correct 3D coordinate system)
  const poseWorld = extractWorldLandmarks(
    frameData.worldLandmarks,  // Use actual world landmarks
    LANDMARK_RANGES.pose,
    frameData.confidence
  );

  // Extract normalized landmarks and transform Y-axis
  const poseNormalized = extractNormalizedLandmarks(
    frameData.landmarks,
    LANDMARK_RANGES.pose,
    frameData.confidence
  );

  return {
    poseLandmarks: poseNormalized,     // 2D normalized (Y inverted)
    poseLandmarks3D: poseWorld,         // 3D world (use as-is)
    // ... rest
  };
}

function extractWorldLandmarks(
  worldLandmarks: (number | null)[][],
  range: { start: number; end: number },
  confidence: (number | null)[]
): MediaPipeLandmark[] {
  const landmarks: MediaPipeLandmark[] = [];

  for (let i = range.start; i < range.end; i++) {
    const landmark = worldLandmarks[i];
    const conf = confidence[i];

    if (!landmark || landmark[0] === null || landmark[1] === null || landmark[2] === null) {
      landmarks.push({ x: 0, y: 0, z: 0, visibility: 0 });
      continue;
    }

    // World landmarks are already in correct Y-up coordinate system
    // Just use them directly - NO transformation needed
    landmarks.push({
      x: landmark[0],   // meters, left-right
      y: landmark[1],   // meters, bottom-top (already Y-up!)
      z: landmark[2],   // meters, back-front
      visibility: (conf !== null && conf !== undefined) ? conf : 0.5
    });
  }

  return landmarks;
}

function extractNormalizedLandmarks(
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

    // Normalized landmarks need Y-inversion for Kalidokit
    // Kalidokit internally converts these to work with VRM
    landmarks.push({
      x: landmark[0],        // 0-1, left-right
      y: 1.0 - landmark[1],  // ✅ INVERT: 0-1 becomes 1-0 (top-bottom → bottom-top)
      z: landmark[2],        // depth scale (keep as-is)
      visibility: (conf !== null && conf !== undefined) ? conf : 0.5
    });
  }

  return landmarks;
}
```

### Option B: If You ONLY Have Normalized Landmarks (Your Current Situation)

If your pose data doesn't include world landmarks, you need to approximate:

```typescript
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
      y: 1.0 - landmark[1],  // ✅ INVERT Y for VRM coordinate system
      z: landmark[2],         // Keep depth as-is
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
  const landmarks: MediaPipeLandmark[] = [];

  for (let i = range.start; i < range.end; i++) {
    const landmark = allLandmarks[i];
    const conf = confidence[i];

    if (!landmark || landmark[0] === null || landmark[1] === null || landmark[2] === null) {
      landmarks.push({ x: 0, y: 0, z: 0, visibility: 0 });
      continue;
    }

    // Approximate world coordinates from normalized
    // This is NOT as accurate as true world landmarks, but works
    landmarks.push({
      x: landmark[0],
      y: 1.0 - landmark[1],  // ✅ INVERT Y
      z: landmark[2],         // Keep depth
      visibility: (conf !== null && conf !== undefined) ? conf : 0.5
    });
  }

  return landmarks;
}
```

---

## 🎯 Key Fixes Required

### 1. Runtime Configuration

Add runtime config to your Kalidokit call:

```typescript
// In AvatarRenderer.tsx, applyPoseToAvatar function
const riggedPose = Kalidokit.Pose.solve(
  kalidokitData.poseLandmarks3D,  // 3D landmarks
  kalidokitData.poseLandmarks,    // 2D normalized landmarks
  {
    runtime: 'mediapipe',  // ✅ ADD THIS - tells Kalidokit to expect MediaPipe format
    enableLegs: true       // Enable leg tracking
  }
);
```

### 2. Remove Z-Axis Negation (If Using Option B)

**IMPORTANT**: Do NOT negate Z when using normalized landmarks:

```typescript
// ❌ WRONG (from my earlier suggestion)
z: -landmark[2]

// ✅ CORRECT (for normalized landmarks)
z: landmark[2]
```

**Why?** MediaPipe's normalized Z is already in the correct relative scale. Negating it will flip front/back.

### 3. VRM Initial Rotation

Ensure your VRM is properly oriented:

```typescript
// In loadAvatar function
VRMUtils.rotateVRM0(vrm); // ✅ This fixes VRM 0.x coordinate system

// Optional: If avatar is facing wrong way
vrm.scene.rotation.y = Math.PI; // Rotate 180° to face camera
```

---

## 🧪 Testing & Verification

### Test 1: Check Coordinate Values

Add debug logging to see actual values:

```typescript
if (frameIndex === 0) {
  const leftShoulder = kalidokitData.poseLandmarks[11];
  const rightShoulder = kalidokitData.poseLandmarks[12];
  
  console.log('🔍 Shoulder Check:', {
    leftShoulder: leftShoulder,
    rightShoulder: rightShoulder,
    yInverted: leftShoulder.y > 0.5 // Should be true if arms are down
  });
}
```

**Expected Results:**
- Arms at rest (down): Y values should be > 0.5 (after inversion)
- Arms raised (up): Y values should be < 0.5 (after inversion)

### Test 2: Verify Kalidokit Output

Log the rigged pose angles:

```typescript
if (frameIndex === 0) {
  console.log('🎬 Rigged Pose:', {
    leftUpperArm: riggedPose.LeftUpperArm,
    rightUpperArm: riggedPose.RightUpperArm,
    // These should be reasonable euler angles (not NaN or Infinity)
  });
}
```

**Expected Results:**
- Angles in radians: typically -3.14 to 3.14
- Not NaN or Infinity
- Symmetric for similar arm positions

### Test 3: Visual Verification

Compare skeleton overlay with avatar:
- [ ] Avatar arms match skeleton arms
- [ ] Raising arms → both skeleton and avatar go up
- [ ] Moving forward → both move forward
- [ ] No 180° flips or inversions

---

## 📋 Checklist: Apply All Fixes

```bash
[ ] Step 1: Invert Y-axis in extractLandmarks() (line ~175)
      Change: y: landmark[1] → y: 1.0 - landmark[1]

[ ] Step 2: Invert Y-axis in extractLandmarks3D() (line ~225)
      Change: y: landmark[1] → y: 1.0 - landmark[1]

[ ] Step 3: Do NOT negate Z-axis (remove if present)
      Keep: z: landmark[2] (no negation)

[ ] Step 4: Add runtime config to Kalidokit.Pose.solve() (AvatarRenderer.tsx ~447)
      Add: { runtime: 'mediapipe', enableLegs: true }

[ ] Step 5: Verify VRMUtils.rotateVRM0(vrm) is called (line 384)
      Already present: ✓

[ ] Step 6: Test with debug logging (frames 0-2)
      Add console.logs to verify transformation

[ ] Step 7: Visual comparison test
      Compare skeleton overlay with avatar movement
```

---

## 🔗 References

- **Kalidokit GitHub**: https://github.com/yeemachine/kalidokit
- **MediaPipe Pose Docs**: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- **Kalidokit Glitch Demo**: https://glitch.com/edit/#!/kalidokit (working reference)
- **VRM Specification**: https://vrm.dev/en/docs/spec/

---

## 💡 Why Your Avatar Moves Incorrectly

**Root Cause**: MediaPipe uses screen coordinates (Y-down) but VRM/Kalidokit expects 3D coordinates (Y-up).

**Your Current Code**:
```typescript
// MediaPipe says: "Arms are at Y=0.3 (30% from top of screen)"
y: 0.3

// VRM interprets: "Arms are at Y=0.3 (30% from bottom in 3D space)"
// Result: Arms appear MUCH LOWER than they should be
```

**With Fix**:
```typescript
// MediaPipe says: "Arms are at Y=0.3 (30% from top of screen)"
y: 1.0 - 0.3 = 0.7

// VRM interprets: "Arms are at Y=0.7 (70% from bottom in 3D space)"
// Result: Arms appear at CORRECT HEIGHT (near top)
```

---

## 🎓 Understanding the Data Flow

```
MediaPipe Holistic Output
         ↓
    Your .pose Files
         ↓
convertToKalidokitFormat() ← ADD Y-INVERSION HERE
         ↓
   Kalidokit.Pose.solve() ← ADD runtime: 'mediapipe'
         ↓
     Euler Rotations
         ↓
    eulerToQuaternion()
         ↓
    VRM Bone Rotations
         ↓
     Avatar Movement
```

**Critical Points**:
1. Y-inversion must happen in `poseToKalidokit.ts`
2. Runtime must be specified in `AvatarRenderer.tsx`
3. Quaternion conversion already correct ✓
4. VRM coordinate fix already applied ✓

---

## 🆘 Still Not Working?

If after applying ALL fixes the avatar still doesn't move correctly:

### Check 1: Data Source
```typescript
// Verify your pose data is actually from MediaPipe Holistic
console.log('Source:', poseData.source_video);
console.log('Landmarks shape:', poseData.landmarks.length, 'x', poseData.landmarks[0].length);
// Should be: N frames x 523 landmarks
```

### Check 2: Landmark Format
```typescript
// Check a sample landmark
console.log('Sample landmark:', poseData.landmarks[0][11]);
// Should be: [x, y, z] where x,y,z are numbers between 0 and 1
```

### Check 3: VRM Model
```typescript
// Try with a different VRM model to rule out model-specific issues
// Test with official Kalidokit demo model from VRoid Hub
```

### Check 4: Three.js Version
```typescript
// Ensure Three.js and Three-VRM versions are compatible
console.log('THREE.REVISION:', THREE.REVISION);
// Should be: r128 or compatible with your VRM loader
```

---

**Last Updated**: Based on Kalidokit v1.1.5 and MediaPipe Solutions Preview (2024)
