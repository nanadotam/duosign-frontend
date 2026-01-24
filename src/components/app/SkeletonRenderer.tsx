'use client'

import { useRef, useEffect, useCallback, useState } from 'react'

/**
 * Pose data structure matching .pose file format
 */
export interface PoseData {
  landmarks: (number | null)[][][]  // (T, 523, 3) - x, y, z (can be null)
  confidence: (number | null)[][]   // (T, 523) (can be null)
  fps: number
  frame_count: number
  source_video: string
}

/**
 * Landmark layout indices
 */
const LANDMARK_LAYOUT = {
  pose: { start: 0, end: 13 },
  face: { start: 13, end: 481 },
  leftHand: { start: 481, end: 502 },
  rightHand: { start: 502, end: 523 },
}

/**
 * Skeleton connections for pose landmarks
 */
const POSE_CONNECTIONS = [
  [1, 2],   // shoulders
  [1, 3], [3, 5],   // left arm
  [2, 4], [4, 6],   // right arm
  [1, 0], [2, 0],   // neck/head
]

/**
 * Hand connections (21 landmarks)
 */
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
]

interface SkeletonRendererProps {
  poseData: PoseData | null
  isPlaying: boolean
  speed: number
  currentFrame?: number
  onFrameChange?: (frame: number) => void
  className?: string
}

// Colors
const COLORS = {
  background: '#0f172a',
  pose: '#4ade80',
  poseJoint: '#22c55e',
  leftHand: '#60a5fa',
  leftHandJoint: '#3b82f6',
  rightHand: '#f472b6',
  rightHandJoint: '#ec4899',
  face: 'rgba(148, 163, 184, 0.4)',
}

/**
 * Check if a landmark value is valid (not null/undefined/NaN)
 */
function isValidLandmark(lm: (number | null)[] | undefined): lm is number[] {
  if (!lm || lm.length < 2) return false
  return lm[0] !== null && lm[1] !== null && !isNaN(lm[0] as number) && !isNaN(lm[1] as number)
}

/**
 * Canvas-based 2D skeleton renderer
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
   * Draw a single frame of pose data with confidence-based opacity and Z-depth sorting
   */
  const drawFrame = useCallback((ctx: CanvasRenderingContext2D, frameIdx: number) => {
    if (!poseData || frameIdx >= poseData.frame_count) return

    const width = canvasSize.width
    const height = canvasSize.height

    // Clear canvas
    ctx.fillStyle = COLORS.background
    ctx.fillRect(0, 0, width, height)

    // Get frame data
    const landmarks = poseData.landmarks[frameIdx]
    const confidence = poseData.confidence[frameIdx]

    if (!landmarks) return

    // Calculate scale - landmarks are normalized 0-1
    const scale = Math.min(width, height) * 0.85
    const offsetX = (width - scale) / 2
    const offsetY = (height - scale) / 2

    // Transform landmark to canvas coordinates
    const toCanvas = (lm: number[]) => ({
      x: offsetX + lm[0] * scale,
      y: offsetY + lm[1] * scale,
      z: lm[2] ?? 0,
    })

    // Get confidence value for a landmark (clamped to reasonable range)
    const getConfidence = (idx: number): number => {
      const conf = confidence[idx]
      if (conf === null || conf === undefined || isNaN(conf as number)) return 0.3
      return Math.max(0.15, Math.min(1, conf as number))
    }

    // ─────────────────────────────────────────────────────────────
    // Collect all draw operations for Z-depth sorting
    // ─────────────────────────────────────────────────────────────
    type DrawOp = {
      type: 'circle' | 'line'
      x: number
      y: number
      z: number
      radius?: number
      x2?: number
      y2?: number
      color: string
      opacity: number
      lineWidth?: number
    }

    const drawOps: DrawOp[] = []

    // ─────────────────────────────────────────────────────────────
    // Collect Face operations (subtle dots)
    // ─────────────────────────────────────────────────────────────
    for (let i = LANDMARK_LAYOUT.face.start; i < LANDMARK_LAYOUT.face.end; i++) {
      const lm = landmarks[i]
      if (!isValidLandmark(lm)) continue
      
      const pos = toCanvas(lm as number[])
      const opacity = getConfidence(i) * 0.4 // Face is more subtle
      
      drawOps.push({
        type: 'circle',
        x: pos.x,
        y: pos.y,
        z: pos.z,
        radius: 1.5,
        color: '#94a3b8',
        opacity,
      })
    }

    // ─────────────────────────────────────────────────────────────
    // Collect Pose Skeleton operations
    // ─────────────────────────────────────────────────────────────
    for (const [start, end] of POSE_CONNECTIONS) {
      const lm1 = landmarks[start]
      const lm2 = landmarks[end]
      
      if (!isValidLandmark(lm1) || !isValidLandmark(lm2)) continue
      
      const p1 = toCanvas(lm1 as number[])
      const p2 = toCanvas(lm2 as number[])
      const avgOpacity = (getConfidence(start) + getConfidence(end)) / 2
      const avgZ = (p1.z + p2.z) / 2
      
      drawOps.push({
        type: 'line',
        x: p1.x,
        y: p1.y,
        x2: p2.x,
        y2: p2.y,
        z: avgZ,
        color: COLORS.pose,
        opacity: avgOpacity,
        lineWidth: 4,
      })
    }

    // Pose joints
    for (let i = 0; i < LANDMARK_LAYOUT.pose.end; i++) {
      const lm = landmarks[i]
      if (!isValidLandmark(lm)) continue
      
      const pos = toCanvas(lm as number[])
      const opacity = getConfidence(i)
      
      drawOps.push({
        type: 'circle',
        x: pos.x,
        y: pos.y,
        z: pos.z,
        radius: 8,
        color: COLORS.poseJoint,
        opacity,
      })
    }

    // ─────────────────────────────────────────────────────────────
    // Collect Left Hand operations
    // ─────────────────────────────────────────────────────────────
    const leftHandStart = LANDMARK_LAYOUT.leftHand.start

    for (const [start, end] of HAND_CONNECTIONS) {
      const lm1 = landmarks[leftHandStart + start]
      const lm2 = landmarks[leftHandStart + end]
      
      if (!isValidLandmark(lm1) || !isValidLandmark(lm2)) continue
      
      const p1 = toCanvas(lm1 as number[])
      const p2 = toCanvas(lm2 as number[])
      const avgOpacity = (getConfidence(leftHandStart + start) + getConfidence(leftHandStart + end)) / 2
      
      drawOps.push({
        type: 'line',
        x: p1.x,
        y: p1.y,
        x2: p2.x,
        y2: p2.y,
        z: (p1.z + p2.z) / 2,
        color: COLORS.leftHand,
        opacity: avgOpacity,
        lineWidth: 2,
      })
    }

    for (let i = 0; i < 21; i++) {
      const lm = landmarks[leftHandStart + i]
      if (!isValidLandmark(lm)) continue
      
      const pos = toCanvas(lm as number[])
      const opacity = getConfidence(leftHandStart + i)
      
      drawOps.push({
        type: 'circle',
        x: pos.x,
        y: pos.y,
        z: pos.z,
        radius: 5,
        color: COLORS.leftHandJoint,
        opacity,
      })
    }

    // ─────────────────────────────────────────────────────────────
    // Collect Right Hand operations
    // ─────────────────────────────────────────────────────────────
    const rightHandStart = LANDMARK_LAYOUT.rightHand.start

    for (const [start, end] of HAND_CONNECTIONS) {
      const lm1 = landmarks[rightHandStart + start]
      const lm2 = landmarks[rightHandStart + end]
      
      if (!isValidLandmark(lm1) || !isValidLandmark(lm2)) continue
      
      const p1 = toCanvas(lm1 as number[])
      const p2 = toCanvas(lm2 as number[])
      const avgOpacity = (getConfidence(rightHandStart + start) + getConfidence(rightHandStart + end)) / 2
      
      drawOps.push({
        type: 'line',
        x: p1.x,
        y: p1.y,
        x2: p2.x,
        y2: p2.y,
        z: (p1.z + p2.z) / 2,
        color: COLORS.rightHand,
        opacity: avgOpacity,
        lineWidth: 2,
      })
    }

    for (let i = 0; i < 21; i++) {
      const lm = landmarks[rightHandStart + i]
      if (!isValidLandmark(lm)) continue
      
      const pos = toCanvas(lm as number[])
      const opacity = getConfidence(rightHandStart + i)
      
      drawOps.push({
        type: 'circle',
        x: pos.x,
        y: pos.y,
        z: pos.z,
        radius: 5,
        color: COLORS.rightHandJoint,
        opacity,
      })
    }

    // ─────────────────────────────────────────────────────────────
    // Sort by Z-depth (draw farther points first, so closer overlap)
    // Higher Z = farther from camera (in MediaPipe convention)
    // ─────────────────────────────────────────────────────────────
    drawOps.sort((a, b) => b.z - a.z)

    // ─────────────────────────────────────────────────────────────
    // Execute draw operations with confidence-based opacity
    // ─────────────────────────────────────────────────────────────
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    for (const op of drawOps) {
      ctx.globalAlpha = op.opacity

      if (op.type === 'circle') {
        ctx.fillStyle = op.color
        ctx.beginPath()
        ctx.arc(op.x, op.y, op.radius!, 0, Math.PI * 2)
        ctx.fill()
      } else if (op.type === 'line') {
        ctx.strokeStyle = op.color
        ctx.lineWidth = op.lineWidth!
        ctx.beginPath()
        ctx.moveTo(op.x, op.y)
        ctx.lineTo(op.x2!, op.y2!)
        ctx.stroke()
      }
    }

    // Reset alpha
    ctx.globalAlpha = 1

    // ─────────────────────────────────────────────────────────────
    // Draw frame counter (always on top)
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

