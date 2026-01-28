# Data Flow Example: Your Pose → Avatar Animation

## Your Current Data Format

```json
{
  "landmarks": [
    // Frame 0
    [
      [0.5, 0.6, -0.1],  // Landmark 0 (nose)
      [0.52, 0.58, -0.12], // Landmark 1 (left eye inner)
      ...
      [0.45, 0.3, -0.05]  // Landmark 522 (right hand pinky tip)
    ],
    // Frame 1
    [...],
    // Frame N
    [...]
  ],
  "confidence": [[0.98, 0.97, ...], ...],
  "presenceMask": [[true, true, ...], ...],
  "fps": 30.0,
  "frameCount": 90
}
```

## Step 1: Extract Single Frame

```javascript
// Get frame 10
const frameIndex = 10;
const frameLandmarks = poseData.landmarks[frameIndex];  // [523, 3]
const frameConfidence = poseData.confidence[frameIndex]; // [523]
const framePresence = poseData.presenceMask[frameIndex]; // [523]
```

## Step 2: Split by Body Part

```javascript
// Your 523 landmarks are organized as:
// 0-32:   Pose (33 points)
// 33-500: Face (468 points)
// 501-521: Left Hand (21 points)
// 522-542: Right Hand (21 points)

const pose = frameLandmarks.slice(0, 33);
const face = frameLandmarks.slice(33, 501);
const leftHand = frameLandmarks.slice(501, 522);
const rightHand = frameLandmarks.slice(522, 543);
```

## Step 3: Format for Kalidokit

```javascript
// Kalidokit expects objects with x, y, z properties
const kalidokitPose = pose.map((point, i) => ({
  x: point[0],
  y: point[1],
  z: point[2],
  visibility: frameConfidence[i]
}));

// Same for face, leftHand, rightHand...
```

## Step 4: Solve for Bone Rotations

```javascript
import * as Kalidokit from 'kalidokit';

// Kalidokit converts landmarks → quaternions (bone rotations)
const riggedPose = Kalidokit.Pose.solve(kalidokitPose, {
  runtime: 'mediapipe',
  imageSize: { width: 640, height: 480 }
});

// Result looks like:
{
  Hips: { rotation: [x, y, z, w] },
  Spine: { rotation: [x, y, z, w] },
  LeftUpperArm: { rotation: [x, y, z, w] },
  RightUpperArm: { rotation: [x, y, z, w] },
  // ... more bones
}
```

## Step 5: Apply to Avatar

```javascript
// Get the VRM avatar's bone
const leftArm = vrm.humanoid.getNormalizedBoneNode('leftUpperArm');

// Apply the rotation using quaternion slerp (smooth interpolation)
leftArm.quaternion.slerp(
  new THREE.Quaternion(...riggedPose.LeftUpperArm.rotation),
  0.7  // Smoothing factor
);
```

## Complete Flow Diagram

```
┌─────────────────────────────────┐
│ Your Pose JSON                  │
│ Frame 10: [523 landmarks]       │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ Split by body part              │
│ - Pose: [0:33]                  │
│ - Face: [33:501]                │
│ - Hands: [501:543]              │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ Kalidokit.Pose.solve()          │
│ Converts 3D points → rotations  │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ VRM Avatar Bones                │
│ leftUpperArm.quaternion = ...   │
│ rightUpperArm.quaternion = ...  │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│ Three.js Renderer               │
│ Shows animated 3D avatar        │
└─────────────────────────────────┘
```

## Real Example: "HOW" Sign

```javascript
// Frame 30 of your "how.json"
const frame30 = {
  landmarks: [...],  // 523 points
  // Right hand is raised and rotating
  rightHand: [
    [0.6, 0.7, -0.1],  // wrist
    [0.62, 0.72, -0.11], // thumb base
    [0.64, 0.74, -0.12], // thumb tip
    // ... other fingers
  ]
};

// After Kalidokit
const rigged = Kalidokit.Hand.solve(frame30.rightHand, 'Right');
// Result:
{
  RightHand: { rotation: [0.1, 0.3, -0.2, 0.9] },
  RightThumbProximal: { rotation: [0.05, 0.1, 0, 0.99] },
  RightIndexProximal: { rotation: [0.2, 0.15, 0.1, 0.95] },
  // ... all finger joints
}

// Apply to avatar
vrm.humanoid.getNormalizedBoneNode('rightHand').quaternion.set(
  0.1, 0.3, -0.2, 0.9
);
// Now the avatar's right hand matches the sign!
```

## Key Concepts

### Quaternions
- Represent 3D rotations as [x, y, z, w]
- More stable than Euler angles (no gimbal lock)
- `slerp()` smoothly interpolates between rotations

### Slerp Factor (0.7)
- 0.0 = no movement
- 0.5 = halfway between current and target
- 1.0 = instant snap (can be jittery)
- 0.7 = smooth, responsive (recommended)

### Why Kalidokit?
It does the hard math:
1. Maps 33 pose landmarks → 15+ body bones
2. Handles inverse kinematics automatically
3. Converts hand landmarks → 15 finger joints per hand
4. Normalizes coordinates for any avatar size

## Testing Your Integration

```bash
# 1. Install
npm install kalidokit three @pixiv/three-vrm

# 2. Add a VRM avatar to public/avatars/

# 3. Load your existing "how.json" pose data

# 4. See your avatar perform the "HOW" sign!
```

## Common Issues

**Problem**: Avatar arms don't move
**Fix**: Check if `riggedPose.LeftUpperArm` exists
```javascript
if (riggedPose.LeftUpperArm) {
  // Apply rotation
}
```

**Problem**: Movements are jittery
**Fix**: Increase slerp factor to 0.9 or add frame skipping

**Problem**: Avatar rotates weirdly
**Fix**: Your VRM might need `VRMUtils.rotateVRM0(vrm)` called after loading
