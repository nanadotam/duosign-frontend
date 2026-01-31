'use client'

import { useRef, useEffect, useCallback, useState } from 'react'
import type { PoseDataV3 } from '@/utils/applyPoseFrame'

/**
 * Skeleton Renderer for PoseDataV3 (Quaternion-based)
 * 
 * Renders a 2D stick figure visualization of the bone rotations.
 * Uses forward kinematics to compute bone positions from quaternions.
 */

interface SkeletonRendererProps {
  poseData: PoseDataV3 | null
  isPlaying: boolean
  speed: number
  currentFrame?: number
  onFrameChange?: (frame: number) => void
  className?: string
}

// Colors
const COLORS = {
  background: '#0f172a',
  spine: '#22c55e',
  leftArm: '#60a5fa',
  rightArm: '#f472b6',
  head: '#fbbf24',
  joint: '#ffffff',
}

// Bone lengths (normalized units)
const BONE_LENGTHS = {
  spine: 0.25,
  neck: 0.08,
  head: 0.08,
  upperArm: 0.18,
  lowerArm: 0.15,
  hand: 0.08,
}

/**
 * Apply quaternion rotation to a direction vector (simplified 2D projection)
 */
function applyQuatToDirection(quat: number[], direction: [number, number, number]): [number, number] {
  // Quaternion rotation: v' = q * v * q^-1
  // For 2D visualization, we project to XY plane
  const [qx, qy, qz, qw] = quat
  const [dx, dy, dz] = direction
  
  // Simplified quaternion rotation
  const ix = qw * dx + qy * dz - qz * dy
  const iy = qw * dy + qz * dx - qx * dz
  const iz = qw * dz + qx * dy - qy * dx
  const iw = -qx * dx - qy * dy - qz * dz
  
  const rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
  const ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
  
  return [rx, ry]
}

/**
 * Canvas-based 2D skeleton renderer for PoseDataV3 (quaternion format)
 */
export function SkeletonRenderer({
  poseData,
  isPlaying,
  speed,
  currentFrame: externalFrame,
  onFrameChange,
  className = ''
}: SkeletonRendererProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const frameRef = useRef(0)
  const animationRef = useRef<number | null>(null)
  const lastTimeRef = useRef(0)
  const [canvasSize, setCanvasSize] = useState({ width: 400, height: 400 })

  /**
   * Draw a single frame of pose data
   */
  const drawFrame = useCallback((ctx: CanvasRenderingContext2D, frameIdx: number) => {
    if (!poseData || frameIdx >= poseData.frame_count) return

    const width = canvasSize.width
    const height = canvasSize.height
    const scale = Math.min(width, height) * 0.8

    // Clear canvas
    ctx.fillStyle = COLORS.background
    ctx.fillRect(0, 0, width, height)

    // Get frame data
    const frame = poseData.frames[frameIdx]
    if (!frame) return

    const rots = frame.rotations
    const confs = frame.confidences

    // Base position (center of canvas, slightly up)
    const baseX = width / 2
    const baseY = height * 0.65

    // Helper: draw bone
    const drawBone = (
      x1: number, y1: number,
      x2: number, y2: number,
      color: string,
      lineWidth: number = 6
    ) => {
      ctx.strokeStyle = color
      ctx.lineWidth = lineWidth
      ctx.lineCap = 'round'
      ctx.beginPath()
      ctx.moveTo(x1, y1)
      ctx.lineTo(x2, y2)
      ctx.stroke()
    }

    // Helper: draw joint
    const drawJoint = (x: number, y: number, color: string, radius: number = 8) => {
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.arc(x, y, radius, 0, Math.PI * 2)
      ctx.fill()
    }

    // ─────────────────────────────────────────────────────────────
    // Draw Spine (hip to shoulder)
    // ─────────────────────────────────────────────────────────────
    const hipY = baseY
    const spineQuat = rots['spine'] || [0, 0, 0, 1]
    const spineDir = applyQuatToDirection(spineQuat, [0, -1, 0])
    const shoulderX = baseX + spineDir[0] * BONE_LENGTHS.spine * scale
    const shoulderY = hipY + spineDir[1] * BONE_LENGTHS.spine * scale
    
    drawBone(baseX, hipY, shoulderX, shoulderY, COLORS.spine)
    drawJoint(baseX, hipY, COLORS.joint, 10)

    // ─────────────────────────────────────────────────────────────
    // Draw Neck and Head
    // ─────────────────────────────────────────────────────────────
    const neckQuat = rots['neck'] || [0, 0, 0, 1]
    const neckDir = applyQuatToDirection(neckQuat, [0, -1, 0])
    const neckX = shoulderX + neckDir[0] * BONE_LENGTHS.neck * scale
    const neckY = shoulderY + neckDir[1] * BONE_LENGTHS.neck * scale
    
    drawBone(shoulderX, shoulderY, neckX, neckY, COLORS.spine, 4)
    
    // Head (simplified as circle)
    const headRadius = BONE_LENGTHS.head * scale * 0.8
    ctx.fillStyle = COLORS.head
    ctx.beginPath()
    ctx.arc(neckX, neckY - headRadius, headRadius, 0, Math.PI * 2)
    ctx.fill()
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = 2
    ctx.stroke()

    // ─────────────────────────────────────────────────────────────
    // Draw Left Arm
    // ─────────────────────────────────────────────────────────────
    const leftShoulderX = shoulderX - 0.08 * scale
    const leftShoulderY = shoulderY
    
    // Upper arm
    const leftUpperQuat = rots['leftUpperArm'] || [0, 0, 0, 1]
    const leftUpperDir = applyQuatToDirection(leftUpperQuat, [0, 1, 0])
    const leftElbowX = leftShoulderX + leftUpperDir[0] * BONE_LENGTHS.upperArm * scale
    const leftElbowY = leftShoulderY + leftUpperDir[1] * BONE_LENGTHS.upperArm * scale
    
    const leftUpperConf = confs['leftUpperArm'] ?? 0.5
    ctx.globalAlpha = 0.4 + leftUpperConf * 0.6
    drawBone(leftShoulderX, leftShoulderY, leftElbowX, leftElbowY, COLORS.leftArm)
    drawJoint(leftShoulderX, leftShoulderY, COLORS.joint)
    
    // Lower arm
    const leftLowerQuat = rots['leftLowerArm'] || [0, 0, 0, 1]
    const leftLowerDir = applyQuatToDirection(leftLowerQuat, [0, 1, 0])
    const leftWristX = leftElbowX + leftLowerDir[0] * BONE_LENGTHS.lowerArm * scale
    const leftWristY = leftElbowY + leftLowerDir[1] * BONE_LENGTHS.lowerArm * scale
    
    const leftLowerConf = confs['leftLowerArm'] ?? 0.5
    ctx.globalAlpha = 0.4 + leftLowerConf * 0.6
    drawBone(leftElbowX, leftElbowY, leftWristX, leftWristY, COLORS.leftArm)
    drawJoint(leftElbowX, leftElbowY, COLORS.joint)
    
    // Hand
    const leftHandQuat = rots['leftHand'] || [0, 0, 0, 1]
    const leftHandDir = applyQuatToDirection(leftHandQuat, [0, 1, 0])
    const leftHandX = leftWristX + leftHandDir[0] * BONE_LENGTHS.hand * scale
    const leftHandY = leftWristY + leftHandDir[1] * BONE_LENGTHS.hand * scale
    
    drawBone(leftWristX, leftWristY, leftHandX, leftHandY, COLORS.leftArm, 4)
    drawJoint(leftWristX, leftWristY, COLORS.joint, 6)
    ctx.globalAlpha = 1

    // ─────────────────────────────────────────────────────────────
    // Draw Right Arm
    // ─────────────────────────────────────────────────────────────
    const rightShoulderX = shoulderX + 0.08 * scale
    const rightShoulderY = shoulderY
    
    // Upper arm
    const rightUpperQuat = rots['rightUpperArm'] || [0, 0, 0, 1]
    const rightUpperDir = applyQuatToDirection(rightUpperQuat, [0, 1, 0])
    const rightElbowX = rightShoulderX + rightUpperDir[0] * BONE_LENGTHS.upperArm * scale
    const rightElbowY = rightShoulderY + rightUpperDir[1] * BONE_LENGTHS.upperArm * scale
    
    const rightUpperConf = confs['rightUpperArm'] ?? 0.5
    ctx.globalAlpha = 0.4 + rightUpperConf * 0.6
    drawBone(rightShoulderX, rightShoulderY, rightElbowX, rightElbowY, COLORS.rightArm)
    drawJoint(rightShoulderX, rightShoulderY, COLORS.joint)
    
    // Lower arm
    const rightLowerQuat = rots['rightLowerArm'] || [0, 0, 0, 1]
    const rightLowerDir = applyQuatToDirection(rightLowerQuat, [0, 1, 0])
    const rightWristX = rightElbowX + rightLowerDir[0] * BONE_LENGTHS.lowerArm * scale
    const rightWristY = rightElbowY + rightLowerDir[1] * BONE_LENGTHS.lowerArm * scale
    
    const rightLowerConf = confs['rightLowerArm'] ?? 0.5
    ctx.globalAlpha = 0.4 + rightLowerConf * 0.6
    drawBone(rightElbowX, rightElbowY, rightWristX, rightWristY, COLORS.rightArm)
    drawJoint(rightElbowX, rightElbowY, COLORS.joint)
    
    // Hand
    const rightHandQuat = rots['rightHand'] || [0, 0, 0, 1]
    const rightHandDir = applyQuatToDirection(rightHandQuat, [0, 1, 0])
    const rightHandX = rightWristX + rightHandDir[0] * BONE_LENGTHS.hand * scale
    const rightHandY = rightWristY + rightHandDir[1] * BONE_LENGTHS.hand * scale
    
    drawBone(rightWristX, rightWristY, rightHandX, rightHandY, COLORS.rightArm, 4)
    drawJoint(rightWristX, rightWristY, COLORS.joint, 6)
    ctx.globalAlpha = 1

    // Shoulder joint
    drawJoint(shoulderX, shoulderY, COLORS.joint, 10)

    // ─────────────────────────────────────────────────────────────
    // Draw frame counter
    // ─────────────────────────────────────────────────────────────
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)'
    ctx.font = '14px monospace'
    ctx.fillText(`Frame: ${frameIdx + 1}/${poseData.frame_count}`, 12, 24)

  }, [poseData, canvasSize])

  /**
   * Animation loop
   */
  const animate = useCallback((timestamp: number) => {
    if (!poseData || !isPlaying) return

    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!ctx) return

    const frameDuration = 1000 / (poseData.fps * speed)
    const deltaTime = timestamp - lastTimeRef.current

    if (deltaTime >= frameDuration) {
      frameRef.current = (frameRef.current + 1) % poseData.frame_count
      lastTimeRef.current = timestamp
      onFrameChange?.(frameRef.current)
    }

    drawFrame(ctx, frameRef.current)
    animationRef.current = requestAnimationFrame(animate)
  }, [poseData, isPlaying, speed, drawFrame, onFrameChange])

  // Handle external frame control
  useEffect(() => {
    if (externalFrame !== undefined) {
      frameRef.current = externalFrame
    }
  }, [externalFrame])

  // Start/stop animation
  useEffect(() => {
    if (isPlaying && poseData) {
      lastTimeRef.current = performance.now()
      animationRef.current = requestAnimationFrame(animate)
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
        animationRef.current = null
      }
      const canvas = canvasRef.current
      const ctx = canvas?.getContext('2d')
      if (ctx && poseData) {
        drawFrame(ctx, frameRef.current)
      }
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isPlaying, poseData, animate, drawFrame])

  // Initial draw when pose data changes
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (ctx && poseData) {
      frameRef.current = 0
      drawFrame(ctx, 0)
    }
  }, [poseData, drawFrame])

  // Handle container resize
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const updateSize = () => {
      const rect = container.getBoundingClientRect()
      const size = Math.min(rect.width, rect.height)
      setCanvasSize({ width: size, height: size })
    }

    updateSize()
    
    const resizeObserver = new ResizeObserver(updateSize)
    resizeObserver.observe(container)
    
    return () => resizeObserver.disconnect()
  }, [])

  // Update canvas dimensions when size changes
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = canvasSize.width * dpr
    canvas.height = canvasSize.height * dpr
    
    const ctx = canvas.getContext('2d')
    if (ctx) {
      ctx.scale(dpr, dpr)
      if (poseData) {
        drawFrame(ctx, frameRef.current)
      }
    }
  }, [canvasSize, poseData, drawFrame])

  return (
    <div 
      ref={containerRef}
      className={`relative w-full h-full flex items-center justify-center ${className}`}
      style={{ minHeight: '300px' }}
    >
      <canvas
        ref={canvasRef}
        style={{ 
          width: canvasSize.width, 
          height: canvasSize.height,
          borderRadius: '16px',
        }}
      />
      
      {/* No data placeholder */}
      {!poseData && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center text-white/50">
            <svg 
              className="w-20 h-28 mx-auto mb-3 opacity-30"
              viewBox="0 0 100 150"
              fill="currentColor"
            >
              <ellipse cx="50" cy="30" rx="20" ry="25" />
              <path d="M20 70 Q20 55 50 55 Q80 55 80 70 L80 140 Q80 150 70 150 L30 150 Q20 150 20 140 Z" />
            </svg>
            <p className="text-sm font-medium">Select a sign to view</p>
          </div>
        </div>
      )}
    </div>
  )
}
