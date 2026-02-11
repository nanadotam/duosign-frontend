# Kalidokit Analysis Report

**Repository:** `from-github/kalidokit`
**Version:** 1.1.5
**Purpose:** TypeScript/JavaScript library for converting MediaPipe landmarks to 3D avatar rotations

---

## 1. Executive Summary

Kalidokit is a **solver library** that transforms raw MediaPipe/TensorFlow.js landmark coordinates into Euler rotations suitable for rigging VRM avatars and Live2D characters. It is the computational bridge between pose detection output and 3D character animation.

**Status:** Officially deprecated (README states: "deprecated in favor of using MediaPipe's built-in solutions directly"), but remains widely used and functional.

**Key Insight:** Kalidokit does NOT perform pose detection. It only solves rotations from pre-detected landmarks.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     KALIDOKIT LIBRARY                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│   │  PoseSolver  │  │  FaceSolver  │  │  HandSolver  │      │
│   │              │  │              │  │              │      │
│   │ • calcArms   │  │ • calcHead   │  │ • calcWrist  │      │
│   │ • calcHips   │  │ • calcEyes   │  │ • calcFingers│      │
│   │ • calcLegs   │  │ • calcMouth  │  │              │      │
│   └──────────────┘  └──────────────┘  └──────────────┘      │
│           │                 │                 │              │
│           └─────────────────┴─────────────────┘              │
│                             │                                │
│                      ┌──────────────┐                        │
│                      │    Utils     │                        │
│                      │              │                        │
│                      │ • Vector     │                        │
│                      │ • Euler      │                        │
│                      │ • helpers    │                        │
│                      └──────────────┘                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. File Structure

```
kalidokit/
├── src/
│   ├── index.ts              # Main exports (Pose, Face, Hand, Vector, Utils)
│   ├── Types.ts              # TypeScript interfaces
│   ├── constants.ts          # RIGHT, LEFT, PI, TWO_PI
│   │
│   ├── PoseSolver/
│   │   ├── index.ts          # Pose.solve() - main entry
│   │   ├── calcArms.ts       # Upper/lower arm + hand rotation
│   │   ├── calcHips.ts       # Hip position/rotation + spine
│   │   └── calcLegs.ts       # Leg rotation (optional)
│   │
│   ├── FaceSolver/
│   │   ├── index.ts          # Face.solve() - main entry
│   │   ├── calcHead.ts       # Head roll/pitch/yaw
│   │   ├── calcEyes.ts       # Eye blink, pupil position, brow
│   │   └── calcMouth.ts      # Mouth shape (A, E, I, O, U phonemes)
│   │
│   ├── HandSolver/
│   │   └── index.ts          # Hand.solve() - 21 joint rotations
│   │
│   └── utils/
│       ├── vector.ts         # Vector class (30+ math methods)
│       ├── euler.ts          # Euler rotation class
│       └── helpers.ts        # clamp, remap, RestingDefault
│
├── test/                     # Jest unit tests
├── docs/                     # Live demo page
├── sample/                   # Integration examples
└── package.json
```

---

## 4. Core Solvers

### 4.1 Pose.solve()

**Purpose:** Convert 33 MediaPipe pose landmarks to body rotations

**Signature:**
```typescript
static solve(
  lm3d: Array<XYZ>,           // 33 3D world landmarks
  lm2d: Array<XYZ>,           // 33 2D normalized landmarks
  options?: {
    runtime?: 'mediapipe' | 'tfjs',
    video?: HTMLVideoElement,
    imageSize?: { width: number, height: number },
    enableLegs?: boolean
  }
): TPose
```

**Input Landmark Indices:**

| Index | Body Part | Usage |
|-------|-----------|-------|
| 11 | Left Shoulder | Arm reference |
| 12 | Right Shoulder | Arm reference |
| 13 | Left Elbow | Upper arm endpoint |
| 14 | Right Elbow | Upper arm endpoint |
| 15 | Left Wrist | Lower arm endpoint |
| 16 | Right Wrist | Lower arm endpoint |
| 17-20 | Hand keypoints | Wrist rotation |
| 23 | Left Hip | Hip/spine reference |
| 24 | Right Hip | Hip/spine reference |
| 25 | Left Knee | Upper leg endpoint |
| 26 | Right Knee | Upper leg endpoint |
| 27 | Left Ankle | Lower leg endpoint |
| 28 | Right Ankle | Lower leg endpoint |

**Output Structure:**
```typescript
interface TPose {
  RightUpperArm: Euler;      // {x, y, z} radians
  RightLowerArm: Euler;
  LeftUpperArm: Euler;
  LeftLowerArm: Euler;
  RightHand: Vector;         // Wrist rotation
  LeftHand: Vector;
  RightUpperLeg?: Euler;     // If enableLegs=true
  RightLowerLeg?: Euler;
  LeftUpperLeg?: Euler;
  LeftLowerLeg?: Euler;
  Spine: XYZ;                // Normalized spine rotation
  Hips: {
    position: XYZ,           // Hip center position
    worldPosition: XYZ,      // Computed world position
    rotation: Vector         // Hip rotation
  }
}
```

**Key Algorithm (calcArms):**
```typescript
// Calculate upper arm rotation (shoulder → elbow)
const upperArmRotation = Vector.findRotation(
  lm3d[11],  // Left shoulder
  lm3d[13]   // Left elbow
);

// Calculate lower arm rotation (elbow → wrist)
const lowerArmRotation = Vector.findRotation(
  lm3d[13],  // Left elbow
  lm3d[15]   // Left wrist
);

// Apply anatomical constraints
upperArmRotation.x = clamp(upperArmRotation.x, -PI/2, PI/2);
```

---

### 4.2 Face.solve()

**Purpose:** Convert 468-478 face mesh landmarks to head rotation and expressions

**Signature:**
```typescript
static solve(
  landmarks: Array<XYZ>,      // 468 or 478 face landmarks
  options?: {
    runtime?: 'mediapipe' | 'tfjs',
    video?: HTMLVideoElement,
    imageSize?: { width: number, height: number },
    smoothBlink?: boolean,
    blinkSettings?: [number, number]  // [low, high] thresholds
  }
): TFace
```

**Key Landmark Indices:**

| Index | Usage |
|-------|-------|
| 21, 251, 397, 172 | Head rotation plane (4-point Euler) |
| 130, 133, 160, 159, 158, 144, 145, 153 | Left eye contour |
| 263, 362, 387, 386, 385, 373, 374, 380 | Right eye contour |
| 468-472 | Left iris (5 points, if available) |
| 473-477 | Right iris (5 points, if available) |
| 13, 14 | Upper/lower inner lip |
| 61, 291 | Left/right mouth corners |

**Output Structure:**
```typescript
interface TFace {
  head: {
    x: number;           // Pitch (up/down) - radians
    y: number;           // Yaw (left/right) - radians
    z: number;           // Roll (tilt) - radians
    width: number;       // Face bounding box width
    height: number;      // Face bounding box height
    position: Vector;    // Face center point
    normalized: XYZ;     // Clamped to [-1, 1]
    degrees: XYZ;        // Rotation in degrees
  };
  eye: {
    l: number;           // Left eye openness [0, 1]
    r: number;           // Right eye openness [0, 1]
  };
  brow: number;          // Eyebrow raise [0, 1]
  pupil: {
    x: number;           // Pupil X position [-1, 1]
    y: number;           // Pupil Y position [-1, 1]
  };
  mouth: {
    x: number;           // Mouth width offset
    y: number;           // Mouth openness [0, 1]
    shape: {
      A: number;         // 'Ah' phoneme weight
      E: number;         // 'Eh' phoneme weight
      I: number;         // 'Ee' phoneme weight
      O: number;         // 'Oh' phoneme weight
      U: number;         // 'Oo' phoneme weight
    };
  };
}
```

**Eye Openness Algorithm:**
```typescript
// Compare eyelid distance to eye width
const eyeWidth = distance(lm[130], lm[133]);  // Eye corners
const eyeHeight = distance(lm[159], lm[145]); // Upper/lower lid

const ratio = eyeHeight / eyeWidth;

// Remap to 0-1 range with sensitivity thresholds
const openness = remap(ratio, lowThreshold, highThreshold);
```

**Mouth Shape Algorithm:**
```typescript
const mouthOpen = distance(lm[13], lm[14]);   // Lip distance
const mouthWidth = distance(lm[61], lm[291]); // Mouth corners
const eyeWidth = distance(lm[130], lm[133]);  // Reference scale

const ratio = mouthOpen / eyeWidth;

// Generate phoneme weights
shape.A = ratio * 0.4;
shape.I = complexFormula(mouthWidth, mouthOpen);
shape.E = ratio * (1 - shape.I) * 0.3;
shape.O = (1 - shape.I) * ratio * 0.4;
shape.U = ratio * 0.1;
```

---

### 4.3 Hand.solve()

**Purpose:** Convert 21 hand landmarks to finger joint rotations

**Signature:**
```typescript
static solve(
  landmarks: Array<XYZ>,      // 21 hand landmarks
  side: 'Right' | 'Left'
): THand
```

**Landmark Structure (21 points):**
```
0   = Wrist (palm base)
1-4 = Thumb (CMC, MCP, IP, TIP)
5-8 = Index (MCP, PIP, DIP, TIP)
9-12 = Middle (MCP, PIP, DIP, TIP)
13-16 = Ring (MCP, PIP, DIP, TIP)
17-20 = Pinky (MCP, PIP, DIP, TIP)
```

**Output Structure:**
```typescript
interface THand {
  [Side]Wrist: XYZ;                 // Full 3D rotation
  [Side]ThumbProximal: XYZ;         // 3D rotation (abduction)
  [Side]ThumbIntermediate: XYZ;     // 3D rotation
  [Side]ThumbDistal: XYZ;           // 3D rotation
  [Side]IndexProximal: XYZ;         // Primarily Z rotation
  [Side]IndexIntermediate: XYZ;     // Z rotation (curl)
  [Side]IndexDistal: XYZ;           // Z rotation
  [Side]MiddleProximal: XYZ;
  [Side]MiddleIntermediate: XYZ;
  [Side]MiddleDistal: XYZ;
  [Side]RingProximal: XYZ;
  [Side]RingIntermediate: XYZ;
  [Side]RingDistal: XYZ;
  [Side]LittleProximal: XYZ;
  [Side]LittleIntermediate: XYZ;
  [Side]LittleDistal: XYZ;
}
```

**Finger Curl Algorithm:**
```typescript
// For each finger (except thumb)
const proximal = Vector.findRotation(lm[0], lm[5]);   // Wrist → MCP
const intermediate = Vector.findRotation(lm[5], lm[6]); // MCP → PIP
const distal = Vector.findRotation(lm[6], lm[7]);     // PIP → DIP

// Apply multipliers and anatomical constraints
proximal.z *= 2.3;
intermediate.z = clamp(intermediate.z, -PI/2, 0);
distal.z = clamp(distal.z, -PI/4, 0);
```

---

## 5. Vector Utility Class

**Location:** `src/utils/vector.ts`

**Key Methods:**

```typescript
class Vector {
  // Distance calculations
  static distance(v1: XYZ, v2: XYZ, dimension?: 2 | 3): number

  // Angle calculations
  static angleTo(v1: XYZ, v2: XYZ): number
  static angleBetween3DCoords(a: XYZ, b: XYZ, c: XYZ): number

  // Rotation calculations
  static findRotation(a: XYZ, b: XYZ, normalize?: boolean): Vector
  static rollPitchYaw(a: XYZ, b: XYZ, c?: XYZ): Vector
  static getSphericalCoords(a: XYZ, b: XYZ, axisMap: string): {theta, phi}

  // Linear algebra
  lerp(v: Vector, fraction: number): Vector
  cross(v: Vector): Vector
  dot(v: Vector): number
  unit(): Vector

  // Conversion
  static fromArray(array: number[]): Vector
  toArray(n?: 2 | 3): number[]
  clone(): Vector
}
```

**findRotation Algorithm:**
```typescript
static findRotation(a: XYZ, b: XYZ): Vector {
  return new Vector(
    Math.atan2(b.y - a.y, b.z - a.z),  // X rotation (pitch)
    Math.atan2(b.x - a.x, b.z - a.z),  // Y rotation (yaw)
    Math.atan2(b.y - a.y, b.x - a.x)   // Z rotation (roll)
  );
}
```

**rollPitchYaw Algorithm:**
```typescript
static rollPitchYaw(a: XYZ, b: XYZ, c?: XYZ): Vector {
  // Define plane from 3 points
  const v1 = Vector.sub(b, a);  // Horizontal axis
  const v2 = Vector.sub(c || midpoint, a);  // Vertical axis
  const normal = v1.cross(v2);  // Plane normal

  // Extract Euler angles from normal
  return new Vector(
    Math.asin(normal.y),                    // Pitch
    Math.atan2(normal.x, normal.z),         // Yaw
    Math.atan2(v1.y, Math.sqrt(v1.x² + v1.z²))  // Roll
  );
}
```

---

## 6. Runtime Normalization

Kalidokit handles two different landmark coordinate systems:

### MediaPipe Runtime
- **2D landmarks:** Normalized [0, 1] relative to image
- **3D landmarks:** World coordinates (meters, hip-centered)
- **No additional scaling needed**

### TensorFlow.js Runtime
- **2D landmarks:** Pixel coordinates
- **3D landmarks:** Relative depth
- **Requires scaling:**
```typescript
if (runtime === 'tfjs') {
  lm3d = lm3d.map(l => ({
    x: l.x * 2.3,
    y: l.y * 2.3,
    z: l.z
  }));
}
```

---

## 7. Visibility and Confidence Handling

### Offscreen Detection
```typescript
// Skip arm calculation if landmarks are off-screen
if (lm2d[11].visibility < 0.23) {
  return RestingDefault.RightUpperArm;
}

// Skip leg calculation if hip is off-screen
if (lm2d[23].visibility < 0.63) {
  return RestingDefault.RightUpperLeg;
}
```

### Resting Defaults
```typescript
const RestingDefault = {
  RightUpperArm: { x: 0, y: 0, z: -1.25 },
  RightLowerArm: { x: 0, y: 0, z: 0 },
  LeftUpperArm: { x: 0, y: 0, z: 1.25 },
  LeftLowerArm: { x: 0, y: 0, z: 0 },
  Spine: { x: 0, y: 0, z: 0 },
  // ... etc
};
```

---

## 8. Usage Examples

### Basic Integration
```typescript
import { Pose, Face, Hand } from 'kalidokit';

// In MediaPipe callback
function onResults(results) {
  const poseRig = Pose.solve(
    results.poseWorldLandmarks,
    results.poseLandmarks,
    { runtime: 'mediapipe', enableLegs: true }
  );

  const faceRig = Face.solve(
    results.faceLandmarks,
    { runtime: 'mediapipe' }
  );

  const rightHandRig = Hand.solve(
    results.rightHandLandmarks,
    'Right'
  );

  // Apply to VRM avatar
  applyToVRM(vrm, poseRig, faceRig, rightHandRig);
}
```

### VRM Application
```typescript
function applyToVRM(vrm, pose, face, hand) {
  // Body
  vrm.humanoid.getNormalizedBoneNode('rightUpperArm')
    .rotation.set(pose.RightUpperArm.x, pose.RightUpperArm.y, pose.RightUpperArm.z);

  // Head
  vrm.humanoid.getNormalizedBoneNode('head')
    .rotation.set(face.head.x, face.head.y, face.head.z);

  // Eyes (blendshape)
  vrm.expressionManager.setValue('blink', 1 - face.eye.l);

  // Mouth (phoneme blendshapes)
  vrm.expressionManager.setValue('aa', face.mouth.shape.A);
}
```

---

## 9. Limitations

### Algorithmic Limitations
- **No temporal smoothing:** Each frame processed independently
- **No occlusion handling:** Returns defaults when landmarks lost
- **Euler output:** Susceptible to gimbal lock (quaternions preferred)
- **Fixed anatomical constraints:** Not adjustable per skeleton

### API Limitations
- **No batch processing:** One frame at a time
- **No confidence output:** Rotation confidence not exposed
- **No velocity output:** No angular velocity calculation
- **MediaPipe-coupled:** Assumes MediaPipe landmark ordering

### Deprecation Concerns
- No longer actively maintained
- MediaPipe has moved to Tasks API
- Some edge cases may remain unfixed

---

## 10. Comparison to DuoSign's Approach

| Aspect | Kalidokit | DuoSign |
|--------|-----------|---------|
| Output Format | Euler angles (radians) | Quaternions |
| Smoothing | None (per-frame) | Velocity-adaptive SLERP |
| Confidence | Used internally only | Exposed per-bone |
| Velocity | Not calculated | Stored per-bone |
| Temporal | Stateless | 1€ filter (pre-processing) |
| Gimbal Lock | Possible | Avoided (quaternions) |

---

## 11. Key Takeaways

1. **Kalidokit is a solver, not a detector** - It requires pre-computed landmarks
2. **Euler output is convenient but limited** - Consider quaternion conversion
3. **No built-in smoothing** - Must add temporal filtering separately
4. **Runtime flag is critical** - TFJS vs MediaPipe have different coordinate systems
5. **Visibility thresholds matter** - Adjust for your use case
6. **Both 2D and 3D needed for Pose** - Don't omit either input




