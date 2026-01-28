# Integrating Your Custom VRM: DuoSign-Proto.vrm

## Step 1: Place Your Avatar File

```bash
# Create the avatars directory if it doesn't exist
mkdir -p public/avatars

# Move your VRM file
mv DuoSign-Proto.vrm public/avatars/
```

Your project structure should now look like:
```
duosign-frontend/
├── public/
│   ├── avatars/
│   │   └── DuoSign-Proto.vrm  ← Your avatar here
│   └── lexicon/
└── src/
```

---

## Step 2: Install Required Dependencies

```bash
npm install kalidokit three @pixiv/three-vrm
```

---

## Step 3: Create the Pose-to-Kalidokit Adapter

Create `src/utils/poseToKalidokit.ts`:

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
 */
export function convertToKalidokitFormat(frameData: {
  landmarks: number[][];  // [523, 3]
  confidence: number[];
  presenceMask: boolean[];
}) {
  const { landmarks, confidence, presenceMask } = frameData;
  
  // MediaPipe Holistic landmark ranges
  const ranges = {
    pose: { start: 0, end: 33 },
    face: { start: 33, end: 501 },
    leftHand: { start: 501, end: 522 },
    rightHand: { start: 522, end: 543 }
  };
  
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

## Step 4: Create the Avatar Renderer Component

Create `src/components/app/AvatarRenderer.tsx`:

```typescript
'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { VRM, VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import * as Kalidokit from 'kalidokit';
import { convertToKalidokitFormat } from '@/utils/poseToKalidokit';

interface AvatarRendererProps {
  poseData: {
    landmarks: number[][][];
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
  const [error, setError] = useState<string | null>(null);
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
    camera.position.set(0, 1.4, 2.5);
    
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(
      containerRef.current.clientWidth,
      containerRef.current.clientHeight
    );
    renderer.setPixelRatio(window.devicePixelRatio);
    containerRef.current.appendChild(renderer.domElement);
    
    // Lighting - important for seeing the avatar clearly
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(1, 1, 1).normalize();
    scene.add(light);
    
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    // Add a second light from the back for better depth
    const backLight = new THREE.DirectionalLight(0xffffff, 0.4);
    backLight.position.set(-1, 1, -1).normalize();
    scene.add(backLight);
    
    const clock = new THREE.Clock();
    
    sceneRef.current = { scene, camera, renderer, vrm: null, clock };
    
    // Load your custom VRM avatar
    loadAvatar(scene).then((vrm) => {
      if (sceneRef.current) {
        sceneRef.current.vrm = vrm;
        setIsLoading(false);
        console.log('✅ Avatar loaded successfully!');
      }
    }).catch((err) => {
      console.error('❌ Failed to load avatar:', err);
      setError('Failed to load avatar. Check console for details.');
      setIsLoading(false);
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
      if (isPlaying && poseData) {
        const now = Date.now();
        if (now - lastFrameTime >= frameDuration) {
          setCurrentFrame((prev) => {
            const next = (prev + 1) % poseData.frameCount;
            
            // Apply pose to avatar
            try {
              applyPoseToAvatar(vrm, poseData, next);
            } catch (err) {
              console.error('Error applying pose:', err);
            }
            
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

  // Handle window resize
  useEffect(() => {
    if (!sceneRef.current || !containerRef.current) return;

    const handleResize = () => {
      if (!sceneRef.current || !containerRef.current) return;
      
      const { camera, renderer } = sceneRef.current;
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(
        containerRef.current.clientWidth,
        containerRef.current.clientHeight
      );
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div ref={containerRef} className="w-full h-full relative">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
          <div className="text-white text-center">
            <div className="mb-2">Loading DuoSign Avatar...</div>
            <div className="text-sm text-gray-400">This may take a few seconds</div>
          </div>
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-red-900/80">
          <div className="text-white text-center px-4">
            <div className="mb-2">⚠️ {error}</div>
            <div className="text-sm">Check that DuoSign-Proto.vrm is in public/avatars/</div>
          </div>
        </div>
      )}
      {poseData && !isLoading && !error && (
        <div className="absolute top-4 right-4 text-white bg-black/50 px-3 py-2 rounded">
          <div className="text-sm">Frame: {currentFrame + 1}/{poseData.frameCount}</div>
          <div className="text-xs text-gray-300 mt-1">{playbackSpeed}x speed</div>
        </div>
      )}
    </div>
  );
}

// Load your custom VRM avatar
async function loadAvatar(scene: THREE.Scene): Promise<VRM> {
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  
  console.log('🔄 Loading DuoSign-Proto.vrm...');
  
  const gltf = await loader.loadAsync('/avatars/DuoSign-Proto.vrm');
  
  const vrm = gltf.userData.vrm as VRM;
  
  if (!vrm) {
    throw new Error('Failed to load VRM from file');
  }
  
  // Prepare VRM for rendering
  VRMUtils.removeUnnecessaryJoints(gltf.scene);
  VRMUtils.rotateVRM0(vrm);
  
  // Scale avatar if needed (adjust based on how your avatar looks)
  vrm.scene.scale.set(1, 1, 1);
  
  // Position avatar
  vrm.scene.position.set(0, 0, 0);
  
  scene.add(vrm.scene);
  
  // Log available bones for debugging
  console.log('Avatar bones:', Object.keys(vrm.humanoid.humanBones));
  
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
  
  // Check if we have valid data
  const hasValidPose = kalidokitData.poseLandmarks.some(l => l.visibility > 0.5);
  if (!hasValidPose) return;
  
  try {
    // Solve for bone rotations
    const riggedPose = Kalidokit.Pose.solve(kalidokitData.poseLandmarks, {
      runtime: 'mediapipe',
      video: undefined as any,
      imageSize: { width: 640, height: 480 }
    });
    
    const riggedLeftHand = Kalidokit.Hand.solve(kalidokitData.leftHandLandmarks, 'Left');
    const riggedRightHand = Kalidokit.Hand.solve(kalidokitData.rightHandLandmarks, 'Right');
    
    // Apply to VRM
    if (riggedPose) {
      rigPose(vrm, riggedPose);
    }
    
    if (riggedLeftHand) {
      rigHand(vrm, riggedLeftHand, 'Left');
    }
    
    if (riggedRightHand) {
      rigHand(vrm, riggedRightHand, 'Right');
    }
  } catch (err) {
    // Silently handle errors for frames with insufficient data
    // console.warn('Could not rig frame:', err);
  }
}

// Apply pose rotations to VRM bones
function rigPose(vrm: VRM, riggedPose: any) {
  const { humanoid } = vrm;
  const smoothing = 0.7; // Adjust between 0-1 for smoothness
  
  // Helper function to safely apply rotation
  const applyRotation = (boneName: string, rotation: any) => {
    if (!rotation) return;
    const bone = humanoid.getNormalizedBoneNode(boneName);
    if (bone) {
      bone.quaternion.slerp(
        new THREE.Quaternion(...rotation),
        smoothing
      );
    }
  };
  
  // Core body
  applyRotation('hips', riggedPose.Hips?.rotation);
  applyRotation('spine', riggedPose.Spine?.rotation);
  applyRotation('chest', riggedPose.Chest?.rotation);
  applyRotation('neck', riggedPose.Neck?.rotation);
  applyRotation('head', riggedPose.Head?.rotation);
  
  // Arms
  applyRotation('leftUpperArm', riggedPose.LeftUpperArm?.rotation);
  applyRotation('leftLowerArm', riggedPose.LeftLowerArm?.rotation);
  applyRotation('leftHand', riggedPose.LeftHand?.rotation);
  
  applyRotation('rightUpperArm', riggedPose.RightUpperArm?.rotation);
  applyRotation('rightLowerArm', riggedPose.RightLowerArm?.rotation);
  applyRotation('rightHand', riggedPose.RightHand?.rotation);
  
  // Legs (less critical for ASL but included)
  applyRotation('leftUpperLeg', riggedPose.LeftUpperLeg?.rotation);
  applyRotation('leftLowerLeg', riggedPose.LeftLowerLeg?.rotation);
  applyRotation('rightUpperLeg', riggedPose.RightUpperLeg?.rotation);
  applyRotation('rightLowerLeg', riggedPose.RightLowerLeg?.rotation);
}

// Apply hand rotations to VRM finger bones
function rigHand(vrm: VRM, riggedHand: any, side: 'Left' | 'Right') {
  const { humanoid } = vrm;
  const smoothing = 0.7;
  const prefix = side.toLowerCase();
  
  const applyRotation = (boneName: string, rotation: any) => {
    if (!rotation) return;
    const bone = humanoid.getNormalizedBoneNode(boneName);
    if (bone) {
      bone.quaternion.slerp(
        new THREE.Quaternion(...rotation),
        smoothing
      );
    }
  };
  
  // Fingers
  const fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'];
  const joints = ['Proximal', 'Intermediate', 'Distal'];
  
  fingers.forEach((finger) => {
    joints.forEach((joint) => {
      const rotationKey = `${finger}${joint}`;
      const boneKey = `${prefix}${finger}${joint}`;
      applyRotation(boneKey, riggedHand[rotationKey]);
    });
  });
}
```

---

## Step 5: Update OutputPlayer to Use Avatar

Edit `src/views/app/OutputPlayer.tsx`:

```typescript
// Replace the import
import AvatarRenderer from '@/components/app/AvatarRenderer';

export default function OutputPlayer({ poseData }: { poseData: any }) {
  const [isPlaying, setIsPlaying] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);

  return (
    <div className="w-full h-full flex flex-col">
      {/* Avatar Renderer */}
      <div className="flex-1 relative">
        <AvatarRenderer
          poseData={poseData}
          isPlaying={isPlaying}
          playbackSpeed={playbackSpeed}
        />
      </div>
      
      {/* Controls */}
      <div className="p-4 bg-gray-800 flex items-center justify-center gap-4">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-white"
        >
          {isPlaying ? '⏸ Pause' : '▶ Play'}
        </button>
        
        <select
          value={playbackSpeed}
          onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
          className="px-3 py-2 bg-gray-700 text-white rounded-lg"
        >
          <option value="0.5">0.5x</option>
          <option value="0.75">0.75x</option>
          <option value="1">1x</option>
          <option value="1.25">1.25x</option>
          <option value="1.5">1.5x</option>
        </select>
      </div>
    </div>
  );
}
```

---

## Step 6: Test It!

1. Start your dev server:
```bash
npm run dev
```

2. Click on a gloss card (e.g., "HOW")

3. You should see your DuoSign-Proto avatar animate! 🎉

---

## Troubleshooting

### Avatar doesn't appear
**Check browser console for errors:**
```
Failed to load: /avatars/DuoSign-Proto.vrm
```
→ Make sure the file is in `public/avatars/`

### Avatar is too small/large
**Adjust the scale in loadAvatar():**
```typescript
vrm.scene.scale.set(1.5, 1.5, 1.5); // Make bigger
vrm.scene.scale.set(0.8, 0.8, 0.8); // Make smaller
```

### Avatar is in wrong position
**Adjust camera or avatar position:**
```typescript
camera.position.set(0, 1.4, 3);  // Move camera back
// or
vrm.scene.position.set(0, -0.5, 0); // Move avatar down
```

### Arms/hands don't move smoothly
**Try adjusting the smoothing factor:**
```typescript
const smoothing = 0.9; // Higher = smoother but slower
```

### "Bone not found" errors
**Check which bones your VRM has:**
```typescript
console.log(vrm.humanoid.humanBones);
```
Some VRM models might not have all finger bones defined.

---

## Next Steps

Once it's working:
- [ ] Adjust camera angle for best view
- [ ] Fine-tune avatar scale and position
- [ ] Test with all 10 gloss signs
- [ ] Add loading state improvements
- [ ] Consider adding camera rotation controls

---

## Testing Checklist

- [ ] Avatar loads without errors
- [ ] Avatar appears in viewport
- [ ] Clicking "HOW" shows hand movement
- [ ] Clicking "YES" shows head movement
- [ ] Playback controls work
- [ ] Frame counter updates
- [ ] Speed controls change animation speed
