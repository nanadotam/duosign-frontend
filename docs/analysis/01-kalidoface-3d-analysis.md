# Kalidoface-3D Analysis Report

**Repository:** `from-github/kalidoface-3d`
**Version:** 1.0.0
**Purpose:** Real-time VRM avatar animation with webcam-based face, pose, and hand tracking

---

## 1. Executive Summary

Kalidoface-3D is a **Svelte web application** that animates VRM avatars in real-time using webcam input. It combines MediaPipe Holistic for landmark detection with Kalidokit for solving 3D rotations, then applies those rotations to a loaded VRM model via Three.js.

**Critical Finding:** This is a **real-time webcam application**, not an offline video processor. It cannot be used directly for processing pre-recorded videos or datasets like WLASL/NSLT.

---

## 2. Technology Stack

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| Framework | Svelte | ^3.44.0 | UI framework |
| Build Tool | Vite | ^2.7.2 | Development server & bundling |
| 3D Rendering | Three.js | ^0.138.3 | WebGL scene rendering |
| VRM Support | @pixiv/three-vrm | ^0.6.10 | VRM model loading & rigging |
| Pose Detection | @mediapipe/holistic | ^0.5.1635989137 | Full-body landmark detection |
| Pose Solving | Kalidokit | ^1.1.5 | Landmark → rotation conversion |
| P2P Communication | PeerJS | ^1.3.2 | Video chat with virtual avatars |
| Local Storage | localforage | ^1.9.0 | IndexedDB for VRM persistence |
| UI Components | InteractJS | ^1.10.11 | Drag/touch interactions |

---

## 3. Repository Structure

```
kalidoface-3d/
├── index.html          # Entry point (loads MediaPipe CDN scripts)
├── package.json        # Dependencies declaration
├── vite.config.js      # Vite configuration
├── jsconfig.json       # JavaScript configuration
├── docs/               # Pre-built static output (deployed version)
│   ├── index.html
│   ├── assets/
│   └── ...
└── README.md
```

**Source Code Availability:** None. This repository contains only the compiled/built output. The original Svelte components are not included.

---

## 4. MediaPipe Integration

### 4.1 Holistic Pipeline

Kalidoface-3D uses MediaPipe Holistic, which provides simultaneous detection of:

- **Pose landmarks:** 33 body keypoints (2D normalized + 3D world)
- **Face landmarks:** 468 face mesh points (478 with iris)
- **Hand landmarks:** 21 points per hand

### 4.2 Loading via CDN

From `index.html`:
```html
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js"
        crossorigin="anonymous"></script>
```

### 4.3 Expected Callback Structure

```javascript
const holistic = new Holistic({
  locateFile: (file) =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
});

holistic.setOptions({
  modelComplexity: 1,           // 0=lite, 1=full, 2=heavy
  smoothLandmarks: true,         // Temporal smoothing
  refineFaceLandmarks: true,     // Enable iris tracking (478 points)
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

holistic.onResults((results) => {
  // results.poseLandmarks        - 33 normalized 2D landmarks
  // results.poseWorldLandmarks   - 33 world-space 3D landmarks
  // results.faceLandmarks        - 468-478 face mesh points
  // results.leftHandLandmarks    - 21 left hand points
  // results.rightHandLandmarks   - 21 right hand points
});
```

### 4.4 Camera Integration

```javascript
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await holistic.send({ image: videoElement });
  },
  width: 640,
  height: 480
});
camera.start();
```

**Key Point:** The `Camera` class is tied to live webcam feeds. There is no file-based video input mechanism.

---

## 5. Kalidokit Integration Pattern

### 5.1 Data Flow

```
Webcam Frame
     ↓
MediaPipe Holistic.onResults()
     ↓
{poseLandmarks, poseWorldLandmarks, faceLandmarks, leftHandLandmarks, rightHandLandmarks}
     ↓
Kalidokit.Pose.solve()  →  Arm, Leg, Spine, Hip rotations
Kalidokit.Face.solve()  →  Head rotation, eye/mouth blendshapes
Kalidokit.Hand.solve()  →  21 finger joint rotations (×2 hands)
     ↓
VRM Humanoid Bone Mapping
     ↓
Three.js Mesh Deformation
     ↓
Render to Canvas
```

### 5.2 Solver Invocation

```javascript
// Inside onResults callback
const faceRig = Face.solve(results.faceLandmarks, {
  runtime: 'mediapipe',
  video: videoElement
});

const poseRig = Pose.solve(
  results.poseWorldLandmarks,    // 3D landmarks
  results.poseLandmarks,          // 2D landmarks
  {
    runtime: 'mediapipe',
    video: videoElement,
    enableLegs: true
  }
);

const rightHandRig = Hand.solve(results.rightHandLandmarks, 'Right');
const leftHandRig = Hand.solve(results.leftHandLandmarks, 'Left');
```

### 5.3 VRM Application

```javascript
// Apply face rotations
vrm.humanoid.getNormalizedBoneNode('head').rotation.set(
  faceRig.head.x,
  faceRig.head.y,
  faceRig.head.z
);

// Apply arm rotations
vrm.humanoid.getNormalizedBoneNode('rightUpperArm').rotation.set(
  poseRig.RightUpperArm.x,
  poseRig.RightUpperArm.y,
  poseRig.RightUpperArm.z
);

// Apply blendshapes (expressions)
vrm.expressionManager.setValue('blink', 1 - faceRig.eye.l);
vrm.expressionManager.setValue('aa', faceRig.mouth.shape.A);
```

---

## 6. Application Features

### 6.1 Core Features

| Feature | Description |
|---------|-------------|
| Face Tracking | Real-time facial expression animation |
| Full-Body Tracking | Pose detection with arm/leg/spine movement |
| Hand Tracking | 21-joint finger animations per hand |
| VRM Loading | Drag-drop .vrm file loading with IndexedDB persistence |
| Background Customization | Upload images/GIFs, chroma key (green screen) support |
| Peer Chat | P2P voice calls with virtual avatars (via PeerJS) |
| Camera Modes | Selfie, first-person, standard views |

### 6.2 UI Components

- **VRM Loader:** Drag-drop zone for .vrm files
- **Background Picker:** Image/GIF upload with color picker
- **Camera Controls:** Flip, zoom, position adjustments
- **Tracking Toggles:** Enable/disable face/pose/hands individually
- **Peer Connection:** Room ID sharing for video chat

---

## 7. VRM Model Requirements

### 7.1 Bone Naming Convention

Kalidoface expects VRM models with standard humanoid bone names:

```
Root
├── Hips
│   ├── Spine → Chest → UpperChest → Neck → Head
│   ├── LeftUpperLeg → LeftLowerLeg → LeftFoot
│   ├── RightUpperLeg → RightLowerLeg → RightFoot
│   ├── LeftShoulder → LeftUpperArm → LeftLowerArm → LeftHand
│   └── RightShoulder → RightUpperArm → RightLowerArm → RightHand
│       └── [Finger Bones: Thumb, Index, Middle, Ring, Little × 3 joints each]
```

### 7.2 Expression Blendshapes

For facial animation, VRM models should include:

- `blink` / `blinkLeft` / `blinkRight` - Eye closing
- `aa` / `ee` / `ih` / `oh` / `ou` - Mouth phonemes (AIUEO)
- Optional: `angry`, `happy`, `sad`, `surprised`

---

## 8. Limitations

### 8.1 Real-Time Only

- **No video file input:** Cannot process `.mp4`, `.avi`, `.mov` files
- **No frame-by-frame control:** Processes live stream only
- **No pose data export:** Does not save landmark data

### 8.2 Browser Constraints

- Requires WebGL support
- Camera permissions required
- Performance varies by device (mobile limited)

### 8.3 Missing Source Code

- Only compiled `docs/` output available
- Cannot inspect actual Svelte component logic
- Cannot modify MediaPipe configuration

### 8.4 Deprecated MediaPipe API

- Uses legacy `Holistic` class (pre-Tasks API)
- MediaPipe has since moved to new Task Vision API
- May have compatibility issues with future MediaPipe versions

---

## 9. Relevance to Offline Video Processing

### 9.1 What Can Be Learned

1. **Kalidokit solver integration pattern** - How to call `Pose.solve()`, `Face.solve()`, `Hand.solve()`
2. **VRM bone mapping** - Which bones map to which solver outputs
3. **Blendshape application** - How eye/mouth values drive expressions

### 9.2 What Cannot Be Reused

1. **Camera class** - Tied to live webcam, not video files
2. **MediaPipe loading** - CDN-based, not local model files
3. **Frame processing loop** - Uses `requestAnimationFrame`, not frame extraction

### 9.3 Adaptation Requirements

To adapt for offline video processing:

1. Replace `Camera` with `cv2.VideoCapture` or HTML5 video element
2. Use MediaPipe Python API with `static_image_mode=False`
3. Store landmarks to JSON/NPY between MediaPipe and Kalidokit
4. Batch process entire video before rendering

---

## 10. Key Takeaways

| Aspect | Finding |
|--------|---------|
| **Primary Use Case** | Real-time webcam avatar animation |
| **MediaPipe Mode** | Live stream processing |
| **Data Persistence** | None (no landmark export) |
| **VRM Integration** | Three.js + @pixiv/three-vrm |
| **Kalidokit Role** | Landmark → rotation solver |
| **Offline Capability** | None |
| **Source Availability** | Compiled only |

**Bottom Line:** Kalidoface-3D demonstrates the complete webcam-to-avatar pipeline but provides no direct utility for offline dataset processing. Its value lies in understanding the integration pattern, not in code reuse.
