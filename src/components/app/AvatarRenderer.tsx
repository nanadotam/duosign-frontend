'use client';

/**
 * Avatar Renderer Component
 *
 * Renders a 3D VRM avatar animated with pose data using Three.js and Kalidokit.
 * This component converts MediaPipe Holistic landmarks into bone rotations
 * and applies them to a VRM avatar model for realistic sign language visualization.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { VRM, VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import * as Kalidokit from 'kalidokit';
import { convertToKalidokitFormat, POSE_LANDMARK_INDICES } from '@/utils/poseToKalidokit';
import type { PoseData } from '@/components/app/SkeletonRenderer';
import { motion } from 'framer-motion';

/**
 * Props for AvatarRenderer component
 */
interface AvatarRendererProps {
  /** Pose data containing landmarks and confidence values */
  poseData: PoseData | null;
  /** Whether animation playback is active */
  isPlaying: boolean;
  /** Playback speed multiplier (0.5, 0.75, 1.0, etc.) */
  speed: number;
  /** Current frame index (controlled externally) */
  currentFrame?: number;
  /** Callback when frame changes */
  onFrameChange?: (frame: number) => void;
  /** Optional CSS class name */
  className?: string;
}

/**
 * Three.js scene reference structure
 */
interface SceneRef {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  vrm: VRM | null;
  clock: THREE.Clock;
}

/**
 * Smoothing factor for bone rotation interpolation
 * Higher values = smoother but slower response
 * Range: 0.0 (no movement) to 1.0 (instant snap)
 */
const ROTATION_SMOOTHING = 0.7;


/**
 * AvatarRenderer Component
 *
 * Loads and animates a VRM avatar using pose landmark data.
 * Handles the complete lifecycle: scene initialization, avatar loading,
 * animation loop, and cleanup.
 */
export function AvatarRenderer({
  poseData,
  isPlaying,
  speed,
  currentFrame: externalFrame,
  onFrameChange,
  className = ''
}: AvatarRendererProps) {
  // Refs
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<SceneRef | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const frameRef = useRef(0);
  const lastTimeRef = useRef(0);

  // State
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [loadingProgress, setLoadingProgress] = useState(0);

  /**
   * Initialize Three.js scene, camera, renderer, and lighting
   */
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;

    // Create scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a); // Match slate-900

    // Create camera
    const camera = new THREE.PerspectiveCamera(
      30, // Field of view
      container.clientWidth / container.clientHeight,
      0.1, // Near clipping plane
      20   // Far clipping plane
    );
    camera.position.set(0, 1.4, 2.5); // Position for upper body view
    camera.lookAt(0, 1.2, 0); // Look at upper chest area

    // Create renderer
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);

    // Add lighting for avatar visibility
    setupLighting(scene);

    // Create clock for delta time
    const clock = new THREE.Clock();

    // Store scene reference
    sceneRef.current = { scene, camera, renderer, vrm: null, clock };

    // Load VRM avatar
    loadAvatar(scene, setLoadingProgress)
      .then((vrm) => {
        if (sceneRef.current) {
          sceneRef.current.vrm = vrm;
          setIsLoading(false);
          console.log('✅ DuoSign avatar loaded successfully');
        }
      })
      .catch((err) => {
        console.error('❌ Failed to load avatar:', err);
        setError('Failed to load avatar. Please check console for details.');
        setIsLoading(false);
      });

    // Cleanup on unmount
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, []);

  /**
   * Handle external frame control
   */
  useEffect(() => {
    if (externalFrame !== undefined) {
      frameRef.current = externalFrame;
    }
  }, [externalFrame]);

  /**
   * Animation loop - handles frame timing and pose application
   */
  const animate = useCallback((timestamp: number) => {
    if (!poseData || !sceneRef.current) return;

    const { scene, camera, renderer, vrm, clock } = sceneRef.current;

    // Handle frame timing when playing
    if (isPlaying) {
      const frameDuration = 1000 / (poseData.fps * speed);
      const deltaTime = timestamp - lastTimeRef.current;

      if (deltaTime >= frameDuration) {
        frameRef.current = (frameRef.current + 1) % poseData.frame_count;
        lastTimeRef.current = timestamp;
        onFrameChange?.(frameRef.current);
      }
    }

    // Apply current frame pose to avatar
    if (vrm) {
      try {
        applyPoseToAvatar(vrm, poseData, frameRef.current);
      } catch (err) {
        console.warn('Error applying pose to avatar:', err);
      }

      // Update VRM (required for proper rendering)
      const deltaTime = clock.getDelta();
      vrm.update(deltaTime);
    }

    // Render scene
    renderer.render(scene, camera);

    // Continue animation loop
    animationFrameRef.current = requestAnimationFrame(animate);
  }, [poseData, isPlaying, speed, onFrameChange]);

  /**
   * Start/stop animation loop
   */
  useEffect(() => {
    if (sceneRef.current?.vrm && poseData) {
      lastTimeRef.current = performance.now();
      animationFrameRef.current = requestAnimationFrame(animate);
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [poseData, animate]);

  /**
   * Handle window resize
   */
  useEffect(() => {
    if (!sceneRef.current || !containerRef.current) return;

    const handleResize = () => {
      if (!sceneRef.current || !containerRef.current) return;

      const { camera, renderer } = sceneRef.current;
      const container = containerRef.current;

      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div
      ref={containerRef}
      className={`relative w-full h-full flex items-center justify-center bg-slate-900 ${className}`}
      style={{ minHeight: '350px' }}
    >
      {/* Loading State */}
      {isLoading && (
        <motion.div
          className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/95 backdrop-blur-sm z-10"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <div className="text-white text-center">
            <div className="w-16 h-16 mb-4 mx-auto">
              <svg className="animate-spin" viewBox="0 0 50 50">
                <circle
                  className="stroke-slate-600"
                  fill="none"
                  strokeWidth="4"
                  cx="25"
                  cy="25"
                  r="20"
                />
                <circle
                  className="stroke-blue-500"
                  fill="none"
                  strokeWidth="4"
                  strokeDasharray="80, 200"
                  strokeLinecap="round"
                  cx="25"
                  cy="25"
                  r="20"
                />
              </svg>
            </div>
            <div className="text-lg font-medium mb-2">Loading DuoSign Avatar...</div>
            <div className="text-sm text-slate-400">
              {loadingProgress > 0 ? `${loadingProgress}%` : 'Initializing...'}
            </div>
          </div>
        </motion.div>
      )}

      {/* Error State */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-red-900/80 backdrop-blur-sm z-10">
          <div className="text-white text-center px-4">
            <div className="text-4xl mb-3">⚠️</div>
            <div className="text-lg font-medium mb-2">{error}</div>
            <div className="text-sm text-red-200">
              Check that DuoSign-Proto.vrm is in public/avatars/
            </div>
          </div>
        </div>
      )}

      {/* Frame Counter */}
      {poseData && !isLoading && !error && (
        <div className="absolute top-4 right-4 z-10">
          <div className="bg-black/60 backdrop-blur-sm rounded-lg px-3 py-2">
            <div className="text-sm text-white font-mono">
              Frame: {frameRef.current + 1}/{poseData.frame_count}
            </div>
            <div className="text-xs text-slate-300 mt-1">
              {speed}x speed
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Setup scene lighting for avatar visibility
 *
 * @param scene - Three.js scene to add lights to
 */
function setupLighting(scene: THREE.Scene): void {
  // Main directional light (from front-right)
  const mainLight = new THREE.DirectionalLight(0xffffff, 1.2);
  mainLight.position.set(1, 1, 1).normalize();
  scene.add(mainLight);

  // Fill light (from front-left)
  const fillLight = new THREE.DirectionalLight(0xffffff, 0.6);
  fillLight.position.set(-1, 0.5, 1).normalize();
  scene.add(fillLight);

  // Back light for depth
  const backLight = new THREE.DirectionalLight(0xffffff, 0.4);
  backLight.position.set(0, 1, -1).normalize();
  scene.add(backLight);

  // Ambient light for overall brightness
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambientLight);
}

/**
 * Load VRM avatar from public directory
 *
 * @param scene - Three.js scene to add avatar to
 * @param onProgress - Progress callback (0-100)
 * @returns Promise resolving to loaded VRM
 */
async function loadAvatar(
  scene: THREE.Scene,
  onProgress?: (progress: number) => void
): Promise<VRM> {
  const loader = new GLTFLoader();

  // Register VRM plugin
  loader.register((parser: any) => new VRMLoaderPlugin(parser));

  console.log('🔄 Loading DuoSign-Proto.vrm...');

  // Load with progress tracking
  const gltf = await new Promise<any>((resolve, reject) => {
    loader.load(
      '/avatars/DuoSign-Proto.vrm',
      resolve,
      (progressEvent: any) => {
        if (progressEvent.lengthComputable && onProgress) {
          const percentComplete = Math.round((progressEvent.loaded / progressEvent.total) * 100);
          onProgress(percentComplete);
        }
      },
      reject
    );
  });

  const vrm = gltf.userData.vrm as VRM;

  if (!vrm) {
    throw new Error('Failed to load VRM from file');
  }

  // Prepare VRM for rendering
  VRMUtils.removeUnnecessaryJoints(gltf.scene);
  VRMUtils.rotateVRM0(vrm); // Fix coordinate system if VRM 0.x

  // Scale and position avatar (adjust as needed)
  vrm.scene.scale.set(1, 1, 1);
  vrm.scene.position.set(0, 0, 0);

  // Add to scene
  scene.add(vrm.scene);

  // Log available bones for debugging
  console.log('Avatar bones:', Object.keys(vrm.humanoid.humanBones));

  return vrm;
}

/**
 * Apply pose data to VRM avatar using custom geometric rigging
 *
 * This uses a custom solution designed for DuoSign's 13-point upper body format
 * instead of Kalidokit (which expects 33-point MediaPipe format).
 *
 * @param vrm - VRM avatar to animate
 * @param poseData - Full pose data structure
 * @param frameIndex - Current frame to apply
 */
function applyPoseToAvatar(
  vrm: VRM,
  poseData: PoseData,
  frameIndex: number
): void {
  if (!poseData || frameIndex >= poseData.frame_count) return;

  // Extract current frame data
  const frameData = {
    landmarks: poseData.landmarks[frameIndex],
    confidence: poseData.confidence[frameIndex]
  };

  // Convert to format with validation flags
  const kalidokitData = convertToKalidokitFormat(frameData);

  // Check if we have valid pose data
  if (!kalidokitData.hasValidPose) {
    if (frameIndex === 0) {
      console.warn('Frame 0 has insufficient pose landmarks');
    }
    return;
  }

  try {
    // Use custom geometric rigging for arms (designed for 13-point upper body data)
    rigArmsFromLandmarks(vrm, kalidokitData.poseLandmarks);

    // Use Kalidokit for hands (21-point hand data is standard)
    if (kalidokitData.hasValidLeftHand) {
      const riggedLeftHand = Kalidokit.Hand.solve(kalidokitData.leftHandLandmarks, 'Left');
      if (riggedLeftHand) {
        rigHand(vrm, riggedLeftHand, 'Left');
      }
    }

    if (kalidokitData.hasValidRightHand) {
      const riggedRightHand = Kalidokit.Hand.solve(kalidokitData.rightHandLandmarks, 'Right');
      if (riggedRightHand) {
        rigHand(vrm, riggedRightHand, 'Right');
      }
    }

    // Debug: Log first frame
    if (frameIndex === 0) {
      const { LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST } = POSE_LANDMARK_INDICES;
      console.log('✅ Frame 0 rigged with custom solution', {
        leftShoulder: kalidokitData.poseLandmarks[LEFT_SHOULDER],
        rightShoulder: kalidokitData.poseLandmarks[RIGHT_SHOULDER],
        leftElbow: kalidokitData.poseLandmarks[LEFT_ELBOW],
        rightElbow: kalidokitData.poseLandmarks[RIGHT_ELBOW],
        leftWrist: kalidokitData.poseLandmarks[LEFT_WRIST],
        rightWrist: kalidokitData.poseLandmarks[RIGHT_WRIST],
      });
    }
  } catch (err) {
    if (frameIndex === 0) {
      console.error('Could not rig frame 0:', err);
    }
  }
}

/**
 * Calculate arm rotations from 13-point upper body landmarks
 *
 * Uses geometric calculations to determine arm bone rotations:
 * - Upper arm: shoulder → elbow direction
 * - Lower arm: elbow → wrist direction
 *
 * @param vrm - VRM avatar
 * @param poseLandmarks - 13 upper body landmarks
 */
function rigArmsFromLandmarks(
  vrm: VRM,
  poseLandmarks: { x: number; y: number; z: number; visibility?: number }[]
): void {
  const { humanoid } = vrm;
  const { LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST } = POSE_LANDMARK_INDICES;

  // Get landmark positions
  const leftShoulder = poseLandmarks[LEFT_SHOULDER];
  const rightShoulder = poseLandmarks[RIGHT_SHOULDER];
  const leftElbow = poseLandmarks[LEFT_ELBOW];
  const rightElbow = poseLandmarks[RIGHT_ELBOW];
  const leftWrist = poseLandmarks[LEFT_WRIST];
  const rightWrist = poseLandmarks[RIGHT_WRIST];

  // Helper to create Vector3 from landmark
  // Convert from normalized screen coords to 3D space
  // MediaPipe: x=0-1 (left-right), y=0-1 (top-bottom), z=depth
  // VRM: x=left-right, y=up-down (inverted), z=forward-back
  const toVec3 = (lm: { x: number; y: number; z: number }) => {
    return new THREE.Vector3(
      (lm.x - 0.5) * 2,      // Center and scale x: -1 to 1
      -(lm.y - 0.5) * 2,     // Center, invert, and scale y: -1 to 1
      -lm.z                   // Invert z for VRM coordinate system
    );
  };

  // Calculate arm rotations
  // LEFT ARM
  if (leftShoulder && leftElbow && leftWrist) {
    const shoulderPos = toVec3(leftShoulder);
    const elbowPos = toVec3(leftElbow);
    const wristPos = toVec3(leftWrist);

    // Upper arm: shoulder to elbow direction
    const upperArmDir = new THREE.Vector3().subVectors(elbowPos, shoulderPos).normalize();
    const upperArmRotation = calculateArmRotation(upperArmDir, 'left');
    applyBoneRotation(humanoid, 'leftUpperArm', upperArmRotation);

    // Lower arm: elbow to wrist direction
    const lowerArmDir = new THREE.Vector3().subVectors(wristPos, elbowPos).normalize();
    const lowerArmRotation = calculateArmRotation(lowerArmDir, 'left');
    applyBoneRotation(humanoid, 'leftLowerArm', lowerArmRotation);
  }

  // RIGHT ARM
  if (rightShoulder && rightElbow && rightWrist) {
    const shoulderPos = toVec3(rightShoulder);
    const elbowPos = toVec3(rightElbow);
    const wristPos = toVec3(rightWrist);

    // Upper arm: shoulder to elbow direction
    const upperArmDir = new THREE.Vector3().subVectors(elbowPos, shoulderPos).normalize();
    const upperArmRotation = calculateArmRotation(upperArmDir, 'right');
    applyBoneRotation(humanoid, 'rightUpperArm', upperArmRotation);

    // Lower arm: elbow to wrist direction
    const lowerArmDir = new THREE.Vector3().subVectors(wristPos, elbowPos).normalize();
    const lowerArmRotation = calculateArmRotation(lowerArmDir, 'right');
    applyBoneRotation(humanoid, 'rightLowerArm', lowerArmRotation);
  }
}

/**
 * Calculate rotation quaternion for an arm bone from direction vector
 *
 * @param direction - Normalized direction vector of the arm segment
 * @param side - 'left' or 'right'
 * @returns Quaternion rotation for the bone
 */
function calculateArmRotation(
  direction: THREE.Vector3,
  side: 'left' | 'right'
): THREE.Quaternion {
  // VRM T-pose has arms pointing outward along X-axis
  // Left arm: negative X direction (-1, 0, 0)
  // Right arm: positive X direction (1, 0, 0)
  const restDirection = new THREE.Vector3(side === 'left' ? -1 : 1, 0, 0);

  // Calculate rotation from rest pose to target direction
  const quaternion = new THREE.Quaternion();
  quaternion.setFromUnitVectors(restDirection, direction);

  return quaternion;
}

/**
 * Apply a quaternion rotation to a VRM bone with smoothing
 *
 * @param humanoid - VRM humanoid reference
 * @param boneName - Name of the bone to rotate
 * @param targetRotation - Target quaternion rotation
 */
function applyBoneRotation(
  humanoid: any,
  boneName: string,
  targetRotation: THREE.Quaternion
): void {
  const bone = humanoid.getNormalizedBoneNode(boneName);
  if (bone) {
    bone.quaternion.slerp(targetRotation, ROTATION_SMOOTHING);
  }
}


/**
 * Convert Kalidokit euler angles to Three.js quaternion
 * Kalidokit outputs euler angles in radians as { x, y, z }
 * 
 * @param euler - Euler angles from Kalidokit { x, y, z }
 * @returns THREE.Quaternion
 */
function eulerToQuaternion(euler: { x: number; y: number; z: number }): THREE.Quaternion {
  const threeEuler = new THREE.Euler(euler.x, euler.y, euler.z, 'XYZ');
  return new THREE.Quaternion().setFromEuler(threeEuler);
}

/**
 * Apply hand rotations to VRM finger bones
 * 
 * IMPORTANT: Kalidokit.Hand.solve returns euler angles { x, y, z } in radians,
 * NOT quaternions. Note that fingers primarily move in the z-axis.
 *
 * @param vrm - VRM avatar
 * @param riggedHand - Kalidokit hand solution with euler angles
 * @param side - 'Left' or 'Right'
 */
function rigHand(vrm: VRM, riggedHand: any, side: 'Left' | 'Right'): void {
  const { humanoid } = vrm;
  const prefix = side.toLowerCase();

  /**
   * Helper to apply euler rotation to a finger bone
   *
   * @param boneName - VRM bone name
   * @param euler - Euler angles { x, y, z } in radians from Kalidokit
   */
  const applyRotation = (boneName: string, euler: { x: number; y: number; z: number } | undefined) => {
    if (!euler || typeof euler.z !== 'number') return;

    const bone = humanoid.getNormalizedBoneNode(boneName as any);
    if (bone) {
      // Convert euler angles to quaternion
      const targetQuat = eulerToQuaternion(euler);
      bone.quaternion.slerp(targetQuat, ROTATION_SMOOTHING);
    }
  };

  // Apply wrist rotation
  const wristKey = `${side}Wrist` as keyof typeof riggedHand;
  if (riggedHand[wristKey]) {
    applyRotation(`${prefix}Hand`, riggedHand[wristKey]);
  }

  // Finger names and joints
  const fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'];
  const joints = ['Proximal', 'Intermediate', 'Distal'];

  // Apply rotation to each finger joint
  fingers.forEach((finger) => {
    joints.forEach((joint) => {
      // Kalidokit uses format: RightThumbProximal, etc.
      const rotationKey = `${side}${finger}${joint}` as keyof typeof riggedHand;
      // VRM uses format: rightThumbProximal, etc.
      const boneKey = `${prefix}${finger}${joint}`;
      applyRotation(boneKey, riggedHand[rotationKey]);
    });
  });
}
