/**
 * Velocity-Adaptive SLERP for Quaternion Smoothing
 * =================================================
 * 
 * This module provides utilities for applying quaternion-based pose data
 * to VRM avatars with velocity-adaptive smoothing.
 * 
 * Key Features:
 *   - Direct quaternion application (no Kalidokit)
 *   - Velocity-adaptive SLERP (reduces jitter AND lag)
 *   - Type-safe interfaces for pose data
 * 
 * Author: Nana Kwaku Amoako
 * Date: 2026-01-31
 */

import * as THREE from 'three';
import { VRM } from '@pixiv/three-vrm';

/**
 * Single frame of pose data in V3 format.
 */
export interface PoseFrameV3 {
  /** Bone rotations as quaternions [x, y, z, w] */
  rotations: Record<string, [number, number, number, number]>;
  /** Angular velocities in rad/frame */
  velocities: Record<string, number>;
  /** Detection confidences (0-1) */
  confidences: Record<string, number>;
}

/**
 * Complete pose animation data in V3 format.
 */
export interface PoseDataV3 {
  format_version: string;
  fps: number;
  frame_count: number;
  source_video: string;
  frames: PoseFrameV3[];
  skeleton_info: {
    source: string;
    normalization: {
      original_shoulder_width: number;
      scale_factor: number;
      reference_scale: number;
    };
    bone_count: number;
  };
  filter_config: {
    min_cutoff: number;
    beta: number;
    d_cutoff: number;
  };
  metadata: {
    converted_at: string;
    converter_version: string;
  };
}

/**
 * Map angular velocity to smoothing factor.
 * 
 * Uses piecewise linear interpolation:
 * - velocity < LOW_THRESHOLD: MAX_SMOOTH (high smoothing to reduce jitter)
 * - velocity > HIGH_THRESHOLD: MIN_SMOOTH (low smoothing to reduce lag)
 * - Between thresholds: Linear interpolation
 * 
 * @param velocity Angular velocity in radians/frame
 * @returns Smoothing factor between 0 and 1
 * 
 * @example
 * ```typescript
 * const velocity = 0.05;  // Medium speed
 * const smoothFactor = velocityToSmoothFactor(velocity);
 * // smoothFactor ≈ 0.6 (interpolated between 0.3 and 0.9)
 * ```
 */
function velocityToSmoothFactor(velocity: number): number {
  // Thresholds (tunable based on sign language characteristics)
  const VELOCITY_LOW = 0.01;   // rad/frame - slow movement threshold
  const VELOCITY_HIGH = 0.1;   // rad/frame - fast movement threshold
  const SMOOTH_MIN = 0.3;      // Minimum smoothing (fast movements)
  const SMOOTH_MAX = 0.9;      // Maximum smoothing (slow movements)
  
  // Clamp velocity to threshold range
  if (velocity <= VELOCITY_LOW) {
    return SMOOTH_MAX;  // Slow: maximize smoothing to reduce jitter
  }
  if (velocity >= VELOCITY_HIGH) {
    return SMOOTH_MIN;  // Fast: minimize smoothing to reduce lag
  }
  
  // Linear interpolation between thresholds
  // t = 0 at VELOCITY_LOW, t = 1 at VELOCITY_HIGH
  const t = (velocity - VELOCITY_LOW) / (VELOCITY_HIGH - VELOCITY_LOW);
  
  // Interpolate smoothing factor (inverse relationship with velocity)
  return SMOOTH_MAX - t * (SMOOTH_MAX - SMOOTH_MIN);
}

/**
 * Apply velocity-adaptive SLERP to bone quaternion.
 * 
 * Modifies currentQuat in-place to smoothly interpolate toward targetQuat,
 * with smoothing strength adapted to movement velocity.
 * 
 * @param currentQuat Current bone quaternion (modified in-place)
 * @param targetQuat Target quaternion from pose data
 * @param velocity Angular velocity in rad/frame
 * 
 * @example
 * ```typescript
 * const currentQuat = bone.quaternion;  // Current VRM bone rotation
 * const targetQuat = new THREE.Quaternion(quat[0], quat[1], quat[2], quat[3]);
 * const velocity = 0.02;  // From pose data
 * 
 * applyAdaptiveSlerp(currentQuat, targetQuat, velocity);
 * // currentQuat is now smoothly interpolated toward targetQuat
 * ```
 */
export function applyAdaptiveSlerp(
  currentQuat: THREE.Quaternion,
  targetQuat: THREE.Quaternion,
  velocity: number
): void {
  // Compute adaptive smoothing factor based on velocity
  const smoothFactor = velocityToSmoothFactor(velocity);
  
  // SLERP: Spherical Linear Interpolation
  // currentQuat = slerp(currentQuat, targetQuat, smoothFactor)
  // Higher smoothFactor = more influence from target (less smoothing)
  currentQuat.slerp(targetQuat, smoothFactor);
}

/**
 * Apply a single pose frame to VRM avatar.
 * 
 * This is the main entry point for rendering pose data. It applies
 * quaternion rotations to all bones in the frame with velocity-adaptive
 * smoothing.
 * 
 * @param vrm VRM avatar instance
 * @param frame Pose frame data (V3 format)
 * 
 * @example
 * ```typescript
 * // In your animation loop
 * function animate() {
 *   const frame = poseData.frames[frameIndex];
 *   applyPoseFrame(vrm, frame);
 *   
 *   frameIndex = (frameIndex + 1) % poseData.frame_count;
 *   requestAnimationFrame(animate);
 * }
 * ```
 */
export function applyPoseFrame(
  vrm: VRM,
  frame: PoseFrameV3
): void {
  // Iterate over all bones in frame
  for (const [boneName, quat] of Object.entries(frame.rotations)) {
    // Get VRM bone
    // Note: VRM uses humanoid bone names (e.g., 'leftUpperArm')
    const bone = vrm.humanoid?.getNormalizedBoneNode(boneName as any);
    if (!bone) {
      // Bone not found in VRM (may not be rigged)
      continue;
    }
    
    // Create target quaternion from data
    // Format: [x, y, z, w] (scalar-last, THREE.js compatible)
    const targetQuat = new THREE.Quaternion(
      quat[0],  // x
      quat[1],  // y
      quat[2],  // z
      quat[3]   // w
    );
    
    // Get velocity for this bone
    const velocity = frame.velocities[boneName] || 0;
    
    // Apply velocity-adaptive SLERP
    applyAdaptiveSlerp(bone.quaternion, targetQuat, velocity);
  }
  
  // Update VRM (important for proper rendering)
  vrm.update(0);  // Delta time not needed for pose application
}

/**
 * Load pose data from API endpoint.
 * 
 * @param gloss Sign gloss (e.g., 'hello')
 * @param apiBaseUrl API base URL (default: http://localhost:8000)
 * @returns Promise resolving to pose data
 * 
 * @throws Error if fetch fails or data is invalid
 * 
 * @example
 * ```typescript
 * const poseData = await loadPoseData('hello');
 * console.log(`Loaded ${poseData.frame_count} frames`);
 * ```
 */
export async function loadPoseData(
  gloss: string,
  apiBaseUrl: string = 'http://localhost:8000'
): Promise<PoseDataV3> {
  const response = await fetch(`${apiBaseUrl}/api/sign/${gloss}`);
  
  if (!response.ok) {
    throw new Error(`Failed to load pose data: ${response.statusText}`);
  }
  
  const data = await response.json();
  
  // Validate format
  if (data.format_version !== '3.0-quaternion') {
    throw new Error(`Invalid format version: ${data.format_version}`);
  }
  
  return data as PoseDataV3;
}

/**
 * List available signs from API.
 * 
 * @param apiBaseUrl API base URL (default: http://localhost:8000)
 * @returns Promise resolving to list of sign metadata
 * 
 * @example
 * ```typescript
 * const signs = await listAvailableSigns();
 * console.log(`Available signs: ${signs.map(s => s.gloss).join(', ')}`);
 * ```
 */
export async function listAvailableSigns(
  apiBaseUrl: string = 'http://localhost:8000'
): Promise<Array<{
  gloss: string;
  frame_count: number;
  duration_sec: number;
  file_size_kb: number;
}>> {
  const response = await fetch(`${apiBaseUrl}/api/signs`);
  
  if (!response.ok) {
    throw new Error(`Failed to list signs: ${response.statusText}`);
  }
  
  return await response.json();
}
