# Kalidokit Integration Guide for DuoSign

## Overview

This guide shows how to integrate Kalidokit with your existing pose extraction pipeline to animate a 3D avatar using your MediaPipe landmark data.

---

## What is Kalidokit?

**Kalidokit** converts MediaPipe Holistic landmarks into bone rotations for 3D humanoid avatars. It handles:
- Pose (33 landmarks) → Upper body, arms, legs
- Face (468 landmarks) → Facial expressions, head rotation
- Hands (21 landmarks each) → Finger movements

**Perfect for your use case** because you already have all 523 landmarks extracted!

---

## Installation

```bash
npm install kalidokit three @pixiv/three-vrm
```

### Dependencies Explained

| Package | Purpose |
|---------|---------|
| `kalidokit` | Converts MediaPipe → avatar bone rotations |
| `three` | 3D rendering engine |
| `@pixiv/three-vrm` | Loads and animates VRM avatar models |

---

## Architecture

```
Your Pose Data (.json) 
    ↓
Kalidokit Solver
    ↓
Bone Rotations (Quaternions)
    ↓
VRM Avatar Model
    ↓
Three.js Renderer
```

---

## Step 1: Data Format Adapter

Your pose data needs to match Kalidokit's expected format. Create a utility to transform your 523 landmarks:

```typescript
// src/utils/poseToKalidokit.ts

interface MediaPipeLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

/**
 * Converts your pose data format to Kalidokit's expected format
 * Your data: { landmarks: [T, 523, 3], ... }
 * Kalidokit expects: { pose, face, leftHand, rightHand }
 */
export function convertToKalidokitFormat(frameData: {
  landmarks: number[][];  // [523, 3]
  confidence: number[];
  presenceMask: boolean[];
}) {
  const { landmarks, confidence, presenceMask } = frameData;
  
  // MediaPipe Holistic landmark ranges (from your extraction)
  const ranges = {
    pose: { start: 0, end: 33 },
    face: { start: 33, end: 501 },
    leftHand: { start: 501, end: 522 },
    rightHand: { start: 522, end: 543 }
  };
  
  // Extract and format landmarks
  const pose = extractLandmarks(landmarks, ranges.pose, confidence, presenceMask);
  const face = extractLandmarks(landmarks, ranges.face, confidence, presenceMask);
  const leftHand = extractLandmarks(landmarks, ranges.leftHand, confidence, presenceMask);
  const rightHand = extractLandmarks(landmarks, ranges.rightHand, confidence, presenceMask);
  
  return {
    poseLandmarks: pose,
    faceLandmarks: face,
    leftHandLandmarks: leftHand,
    rightHandLandmarks: rightHand
  };
}

function extractLandmarks(
  allLandmarks: number[][],
  range: { start: number; end: number },
  confidence: number[],
  presenceMask: boolean[]
): MediaPipeLandmark[] {
  const landmarks: MediaPipeLandmark[] = [];
  
  for (let i = range.start; i < range.end; i++) {
    if (!presenceMask[i]) {
      // If landmark not detected, use null placeholder
      landmarks.push({ x: 0, y: 0, z: 0, visibility: 0 });
      continue;
    }
    
    landmarks.push({
      x: allLandmarks[i][0],
      y: allLandmarks[i][1],
      z: allLandmarks[i][2],
      visibility: confidence[i]
    });
  }
  
  return landmarks;
}
```

---

## Step 2: Avatar Renderer Component

Create the main component that loads a VRM avatar and animates it:

```typescript
// src/components/app/AvatarRenderer.tsx

'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { VRM, VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import * as Kalidokit from 'kalidokit';
import { convertToKalidokitFormat } from '@/utils/poseToKalidokit';

interface AvatarRendererProps {
  poseData: {
    landmarks: number[][][];  // [T, 523, 3]
    confidence: number[][];
    presenceMask: boolean[][];
    fps: number;
    frameCount: number;
  } | null;
  isPlaying: boolean;
  playbackSpeed: number;
}

export default function AvatarRenderer({ 
  poseData, 
  isPlaying, 
  playbackSpeed 
}: AvatarRendererProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    renderer: THREE.WebGLRenderer;
    vrm: VRM | null;
    clock: THREE.Clock;
  } | null>(null);
  
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const animationFrameRef = useRef<number>();

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    
    const camera = new THREE.PerspectiveCamera(
      30,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      20
    );
    camera.position.set(0, 1.4, 2);
    
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(
      containerRef.current.clientWidth,
      containerRef.current.clientHeight
    );
    renderer.setPixelRatio(window.devicePixelRatio);
    containerRef.current.appendChild(renderer.domElement);
    
    // Lighting
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(1, 1, 1).normalize();
    scene.add(light);
    
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    const clock = new THREE.Clock();
    
    sceneRef.current = { scene, camera, renderer, vrm: null, clock };
    
    // Load VRM avatar
    loadAvatar(scene).then((vrm) => {
      if (sceneRef.current) {
        sceneRef.current.vrm = vrm;
        setIsLoading(false);
      }
    });
    
    // Cleanup
    return () => {
      renderer.dispose();
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  // Animation loop
  useEffect(() => {
    if (!sceneRef.current || !poseData || !sceneRef.current.vrm) return;

    const { scene, camera, renderer, vrm, clock } = sceneRef.current;
    const fps = poseData.fps;
    const frameDuration = 1000 / (fps * playbackSpeed);
    let lastFrameTime = Date.now();

    const animate = () => {
      if (isPlaying) {
        const now = Date.now();
        if (now - lastFrameTime >= frameDuration) {
          setCurrentFrame((prev) => {
            const next = (prev + 1) % poseData.frameCount;
            
            // Apply pose to avatar
            applyPoseToAvatar(vrm, poseData, next);
            
            return next;
          });
          lastFrameTime = now;
        }
      }
      
      // Update VRM
      const deltaTime = clock.getDelta();
      vrm.update(deltaTime);
      
      renderer.render(scene, camera);
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [poseData, isPlaying, playbackSpeed]);

  return (
    <div ref={containerRef} className="w-full h-full relative">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-white">Loading avatar...</div>
        </div>
      )}
      {poseData && (
        <div className="absolute top-4 right-4 text-white bg-black/50 px-3 py-1 rounded">
          Frame: {currentFrame + 1}/{poseData.frameCount}
        </div>
      )}
    </div>
  );
}

// Load VRM avatar
async function loadAvatar(scene: THREE.Scene): Promise<VRM> {
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  
  // You can use any VRM model - download from:
  // - https://hub.vroid.com/
  // - https://readyplayer.me/ (export as VRM)
  const gltf = await loader.loadAsync('/avatars/default-avatar.vrm');
  
  const vrm = gltf.userData.vrm as VRM;
  
  // Prepare VRM for rendering
  VRMUtils.removeUnnecessaryJoints(gltf.scene);
  VRMUtils.rotateVRM0(vrm);
  
  scene.add(vrm.scene);
  
  return vrm;
}

// Apply pose data to avatar using Kalidokit
function applyPoseToAvatar(
  vrm: VRM,
  poseData: AvatarRendererProps['poseData'],
  frameIndex: number
) {
  if (!poseData) return;
  
  // Extract current frame
  const frameData = {
    landmarks: poseData.landmarks[frameIndex],
    confidence: poseData.confidence[frameIndex],
    presenceMask: poseData.presenceMask[frameIndex]
  };
  
  // Convert to Kalidokit format
  const kalidokitData = convertToKalidokitFormat(frameData);
  
  // Solve for bone rotations
  const riggedPose = Kalidokit.Pose.solve(kalidokitData.poseLandmarks, {
    runtime: 'mediapipe',
    video: undefined as any,
    imageSize: { width: 640, height: 480 }
  });
  
  const riggedLeftHand = Kalidokit.Hand.solve(kalidokitData.leftHandLandmarks, 'Left');
  const riggedRightHand = Kalidokit.Hand.solve(kalidokitData.rightHandLandmarks, 'Right');
  const riggedFace = Kalidokit.Face.solve(kalidokitData.faceLandmarks, {
    runtime: 'mediapipe',
    video: undefined as any,
    imageSize: { width: 640, height: 480 }
  });
  
  // Apply to VRM
  if (riggedPose) {
    applyPoseToVRM(vrm, riggedPose);
  }
  
  if (riggedLeftHand) {
    applyHandToVRM(vrm, riggedLeftHand, 'Left');
  }
  
  if (riggedRightHand) {
    applyHandToVRM(vrm, riggedRightHand, 'Right');
  }
  
  if (riggedFace) {
    applyFaceToVRM(vrm, riggedFace);
  }
}

// Apply pose rotations to VRM bones
function applyPoseToVRM(vrm: VRM, riggedPose: any) {
  const { humanoid } = vrm;
  
  // Hips
  if (riggedPose.Hips) {
    humanoid.getNormalizedBoneNode('hips')?.quaternion.slerp(
      new THREE.Quaternion(...riggedPose.Hips.rotation),
      0.7
    );
  }
  
  // Spine
  if (riggedPose.Spine) {
    humanoid.getNormalizedBoneNode('spine')?.quaternion.slerp(
      new THREE.Quaternion(...riggedPose.Spine.rotation),
      0.7
    );
  }
  
  // Chest
  if (riggedPose.Chest) {
    humanoid.getNormalizedBoneNode('chest')?.quaternion.slerp(
      new THREE.Quaternion(...riggedPose.Chest.rotation),
      0.7
    );
  }
  
  // Arms
  applyArmRotations(humanoid, riggedPose, 'Left');
  applyArmRotations(humanoid, riggedPose, 'Right');
  
  // Legs
  applyLegRotations(humanoid, riggedPose, 'Left');
  applyLegRotations(humanoid, riggedPose, 'Right');
}

function applyArmRotations(humanoid: any, riggedPose: any, side: 'Left' | 'Right') {
  const shoulder = riggedPose[`${side}UpperArm`];
  const elbow = riggedPose[`${side}LowerArm`];
  
  if (shoulder) {
    humanoid.getNormalizedBoneNode(`${side.toLowerCase()}UpperArm`)?.quaternion.slerp(
      new THREE.Quaternion(...shoulder.rotation),
      0.7
    );
  }
  
  if (elbow) {
    humanoid.getNormalizedBoneNode(`${side.toLowerCase()}LowerArm`)?.quaternion.slerp(
      new THREE.Quaternion(...elbow.rotation),
      0.7
    );
  }
}

function applyLegRotations(humanoid: any, riggedPose: any, side: 'Left' | 'Right') {
  const leg = riggedPose[`${side}UpperLeg`];
  const knee = riggedPose[`${side}LowerLeg`];
  
  if (leg) {
    humanoid.getNormalizedBoneNode(`${side.toLowerCase()}UpperLeg`)?.quaternion.slerp(
      new THREE.Quaternion(...leg.rotation),
      0.7
    );
  }
  
  if (knee) {
    humanoid.getNormalizedBoneNode(`${side.toLowerCase()}LowerLeg`)?.quaternion.slerp(
      new THREE.Quaternion(...knee.rotation),
      0.7
    );
  }
}

function applyHandToVRM(vrm: VRM, riggedHand: any, side: 'Left' | 'Right') {
  const { humanoid } = vrm;
  const prefix = side.toLowerCase();
  
  // Apply finger rotations
  const fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'];
  const joints = ['Proximal', 'Intermediate', 'Distal'];
  
  fingers.forEach((finger) => {
    joints.forEach((joint) => {
      const rotation = riggedHand[`${finger}${joint}`];
      if (rotation) {
        const boneName = `${prefix}${finger}${joint}`;
        humanoid.getNormalizedBoneNode(boneName)?.quaternion.slerp(
          new THREE.Quaternion(...rotation),
          0.7
        );
      }
    });
  });
}

function applyFaceToVRM(vrm: VRM, riggedFace: any) {
  // VRM uses blend shapes (morph targets) for facial expressions
  const { expressionManager } = vrm;
  
  if (!expressionManager) return;
  
  // Map Kalidokit face values to VRM expressions
  if (riggedFace.eye) {
    expressionManager.setValue('blink', riggedFace.eye.l);
    expressionManager.setValue('blinkRight', riggedFace.eye.r);
  }
  
  if (riggedFace.mouth) {
    expressionManager.setValue('aa', riggedFace.mouth.shape.A);
    expressionManager.setValue('ee', riggedFace.mouth.shape.E);
    expressionManager.setValue('ih', riggedFace.mouth.shape.I);
    expressionManager.setValue('oh', riggedFace.mouth.shape.O);
    expressionManager.setValue('ou', riggedFace.mouth.shape.U);
  }
  
  if (riggedFace.brow) {
    expressionManager.setValue('angry', riggedFace.brow);
  }
}
```

---

## Step 3: Update Your Main App

Replace the `SkeletonRenderer` with `AvatarRenderer`:

```typescript
// src/views/app/OutputPlayer.tsx

import AvatarRenderer from '@/components/app/AvatarRenderer';

export default function OutputPlayer({ poseData }: { poseData: any }) {
  const [isPlaying, setIsPlaying] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);

  return (
    <div className="w-full h-full">
      <AvatarRenderer
        poseData={poseData}
        isPlaying={isPlaying}
        playbackSpeed={playbackSpeed}
      />
      
      {/* Playback controls */}
      <div className="controls">
        <button onClick={() => setIsPlaying(!isPlaying)}>
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <select onChange={(e) => setPlaybackSpeed(Number(e.target.value))}>
          <option value="0.5">0.5x</option>
          <option value="0.75">0.75x</option>
          <option value="1">1x</option>
        </select>
      </div>
    </div>
  );
}
```

---

## Step 4: Get a VRM Avatar

You need a 3D avatar file. Options:

### Option 1: VRoid Hub (Free)
1. Go to https://hub.vroid.com/
2. Download any avatar with a Creative Commons license
3. Place it in `public/avatars/default-avatar.vrm`

### Option 2: Ready Player Me
1. Go to https://readyplayer.me/
2. Create a custom avatar
3. Export as `.glb` and convert to VRM using:
   - https://vrm.dev/en/univrm/gltf/convert_from_glb/

### Option 3: Use a Sample
```bash
# Download a sample VRM
curl -L https://github.com/vrm-c/vrm-specification/raw/master/samples/AliciaSolid.vrm \
  -o public/avatars/default-avatar.vrm
```

---

## Step 5: Testing

1. Install dependencies:
```bash
npm install kalidokit three @pixiv/three-vrm
```

2. Add the VRM avatar to `public/avatars/`

3. Click on a gloss card (e.g., "HOW")

4. The avatar should animate with the sign!

---

## Troubleshooting

### Issue: "Cannot read property 'rotation' of undefined"
**Solution**: Some VRM models don't have all bones. Add null checks:
```typescript
const bone = humanoid.getNormalizedBoneNode('leftUpperArm');
if (bone && riggedPose.LeftUpperArm) {
  bone.quaternion.slerp(/* ... */);
}
```

### Issue: Avatar is too small/large
**Solution**: Scale the VRM scene:
```typescript
vrm.scene.scale.set(1.5, 1.5, 1.5);
```

### Issue: Jittery movements
**Solution**: Increase the `slerp` smoothing factor from 0.7 to 0.9

### Issue: Hands not moving
**Solution**: Check if your VRM has finger bones:
```typescript
console.log(vrm.humanoid.humanBones);
```

---

## Performance Optimization

```typescript
// Only update every N frames for smooth 30fps
let frameSkip = 0;
if (frameSkip++ % 2 === 0) {
  applyPoseToAvatar(vrm, poseData, currentFrame);
}
```

---

## Next Steps

1. ✅ Replace 2D skeleton with 3D avatar
2. Add avatar customization (skin tone, clothing)
3. Add multiple avatar options
4. Export avatar animations as video
5. Add real-time camera → avatar mapping

---

## Resources

- [Kalidokit Documentation](https://github.com/yeemachine/kalidokit)
- [VRM Specification](https://vrm.dev/en/)
- [Three.js VRM](https://github.com/pixiv/three-vrm)
- [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic)
