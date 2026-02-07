'use client'

import { useRef, useEffect, useCallback, useState } from 'react'
import type { LandmarkPoseData, LandmarkFrame, Connection } from '@/types/pose'

/**
 * Landmark Skeleton Renderer
 * ==========================
 * 
 * Canvas-based 2D skeleton renderer for raw landmark data.
 * Draws pose, face outline, and hand skeletons from x, y, z coordinates.
 * 
 * Author: Nana Kwaku Amoako
 * Date: 2026-02-06
 */

interface LandmarkSkeletonRendererProps {
  poseData: LandmarkPoseData | null
  isPlaying: boolean
  speed?: number
  currentFrame?: number
  onFrameChange?: (frame: number) => void
  showFace?: boolean
  showHands?: boolean
  className?: string
}

// Colors matching the design system
const COLORS = {
  background: 'var(--panel-content-bg, #0f172a)',
  pose: '#22c55e',
  face: '#a855f7',
  leftHand: '#60a5fa',
  rightHand: '#f472b6',
  joint: '#ffffff',
}

/**
 * Correct pose connections for 13-point upper body format
 * 
 * Our pose data uses reduced 13 upper body landmarks:
 *   0: nose
 *   1: left_shoulder, 2: right_shoulder
 *   3: left_elbow, 4: right_elbow
 *   5: left_wrist, 6: right_wrist
 *   7: left_pinky, 8: right_pinky
 *   9: left_index, 10: right_index
 *   11: left_thumb, 12: right_thumb
 */
const POSE_CONNECTIONS_13PT: [number, number][] = [
  [1, 2],   // shoulders
  [1, 3], [3, 5],   // left arm: shoulder → elbow → wrist
  [2, 4], [4, 6],   // right arm: shoulder → elbow → wrist
  [1, 0], [2, 0],   // neck/head connection (shoulders to nose)
  // Optional: connect wrists to finger roots for more detail
  [5, 7], [5, 9], [5, 11],   // left wrist to fingers
  [6, 8], [6, 10], [6, 12],  // right wrist to fingers
]

/**
 * Key joints to render as larger circles
 */
const KEY_JOINTS_13PT = [0, 1, 2, 3, 4, 5, 6] // nose, shoulders, elbows, wrists

/**
 * Draw a connection line between two landmarks.
 */
function drawConnection(
  ctx: CanvasRenderingContext2D,
  landmarks: [number, number, number][],
  conn: Connection,
  color: string,
  lineWidth: number,
  scale: number,
  offsetX: number,
  offsetY: number
) {
  const [i, j] = conn
  if (i >= landmarks.length || j >= landmarks.length) return
  
  const p1 = landmarks[i]
  const p2 = landmarks[j]
  
  // Skip if either point is missing (all zeros)
  if ((p1[0] === 0 && p1[1] === 0) || (p2[0] === 0 && p2[1] === 0)) return
  
  // Transform coordinates (flip Y, apply scale and offset)
  const x1 = p1[0] * scale + offsetX
  const y1 = (1 - p1[1]) * scale + offsetY
  const x2 = p2[0] * scale + offsetX
  const y2 = (1 - p2[1]) * scale + offsetY
  
  ctx.strokeStyle = color
  ctx.lineWidth = lineWidth
  ctx.lineCap = 'round'
  ctx.beginPath()
  ctx.moveTo(x1, y1)
  ctx.lineTo(x2, y2)
  ctx.stroke()
}

/**
 * Draw a joint point.
 */
function drawJoint(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  color: string,
  radius: number
) {
  ctx.fillStyle = color
  ctx.beginPath()
  ctx.arc(x, y, radius, 0, Math.PI * 2)
  ctx.fill()
}

export function LandmarkSkeletonRenderer({
  poseData,
  isPlaying,
  speed = 1,
  currentFrame: externalFrame,
  onFrameChange,
  showFace = false,
  showHands = true,
  className = ''
}: LandmarkSkeletonRendererProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const frameRef = useRef(0)
  const animationRef = useRef<number | null>(null)
  const lastTimeRef = useRef(0)
  const [canvasSize, setCanvasSize] = useState({ width: 400, height: 400 })

  /**
   * Draw a single frame
   */
  const drawFrame = useCallback((ctx: CanvasRenderingContext2D, frameIdx: number) => {
    if (!poseData || frameIdx >= poseData.frame_count) return

    const width = canvasSize.width
    const height = canvasSize.height
    
    // Calculate scale and offset to center the skeleton
    const scale = Math.min(width, height) * 0.85
    const offsetX = (width - scale) / 2
    const offsetY = (height - scale) / 2

    // Clear canvas with theme-aware background
    ctx.fillStyle = COLORS.background
    ctx.fillRect(0, 0, width, height)

    const frame = poseData.frames[frameIdx]
    if (!frame) return

    const { pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks, confidences } = frame
    const { connections } = poseData

    // ─────────────────────────────────────────────────────────────
    // Draw Pose Skeleton
    // ─────────────────────────────────────────────────────────────
    ctx.globalAlpha = 0.4 + confidences.pose * 0.6

    // Draw pose connections using correct 13-point format
    // (ignoring the incorrect JSON connections which use 33-point MediaPipe indices)
    for (const conn of POSE_CONNECTIONS_13PT) {
      drawConnection(ctx, pose_landmarks, conn, COLORS.pose, 4, scale, offsetX, offsetY)
    }

    // Draw pose joints (key ones for 13-point format)
    for (const idx of KEY_JOINTS_13PT) {
      if (idx >= pose_landmarks.length) continue
      const [x, y] = pose_landmarks[idx]
      if (x === 0 && y === 0) continue
      
      const px = x * scale + offsetX
      const py = (1 - y) * scale + offsetY
      drawJoint(ctx, px, py, COLORS.joint, idx === 0 ? 10 : 6)
    }

    // ─────────────────────────────────────────────────────────────
    // Draw Face (optional, simplified)
    // ─────────────────────────────────────────────────────────────
    if (showFace && face_landmarks.length > 0) {
      ctx.globalAlpha = 0.3 + confidences.face * 0.4
      
      // Draw face outline (simplified - just key points)
      const faceOutline = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
      
      ctx.strokeStyle = COLORS.face
      ctx.lineWidth = 1.5
      ctx.beginPath()
      
      for (let i = 0; i < faceOutline.length; i++) {
        const idx = faceOutline[i]
        if (idx >= face_landmarks.length) continue
        const [x, y] = face_landmarks[idx]
        const px = x * scale + offsetX
        const py = (1 - y) * scale + offsetY
        
        if (i === 0) {
          ctx.moveTo(px, py)
        } else {
          ctx.lineTo(px, py)
        }
      }
      ctx.closePath()
      ctx.stroke()
    }

    // ─────────────────────────────────────────────────────────────
    // Draw Hands
    // ─────────────────────────────────────────────────────────────
    if (showHands) {
      // Left hand
      if (left_hand_landmarks.length > 0 && confidences.left_hand > 0.3) {
        ctx.globalAlpha = 0.4 + confidences.left_hand * 0.6
        for (const conn of connections.hand) {
          drawConnection(ctx, left_hand_landmarks, conn, COLORS.leftHand, 2, scale, offsetX, offsetY)
        }
        // Wrist joint
        const [x, y] = left_hand_landmarks[0]
        if (x !== 0 || y !== 0) {
          const px = x * scale + offsetX
          const py = (1 - y) * scale + offsetY
          drawJoint(ctx, px, py, COLORS.joint, 4)
        }
      }

      // Right hand
      if (right_hand_landmarks.length > 0 && confidences.right_hand > 0.3) {
        ctx.globalAlpha = 0.4 + confidences.right_hand * 0.6
        for (const conn of connections.hand) {
          drawConnection(ctx, right_hand_landmarks, conn, COLORS.rightHand, 2, scale, offsetX, offsetY)
        }
        // Wrist joint
        const [x, y] = right_hand_landmarks[0]
        if (x !== 0 || y !== 0) {
          const px = x * scale + offsetX
          const py = (1 - y) * scale + offsetY
          drawJoint(ctx, px, py, COLORS.joint, 4)
        }
      }
    }

    ctx.globalAlpha = 1

    // ─────────────────────────────────────────────────────────────
    // Draw frame info
    // ─────────────────────────────────────────────────────────────
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)'
    ctx.font = '12px monospace'
    ctx.fillText(`Frame: ${frameIdx + 1}/${poseData.frame_count}`, 10, 20)
    ctx.fillText(`${poseData.gloss.toUpperCase()}`, 10, 36)

  }, [poseData, canvasSize, showFace, showHands])

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

  // Initial draw
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (ctx && poseData) {
      frameRef.current = 0
      drawFrame(ctx, 0)
    }
  }, [poseData, drawFrame])

  // Handle resize
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

  // Update canvas dimensions
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
          borderRadius: 'var(--radius-lg, 16px)',
        }}
      />
      
      {/* No data placeholder */}
      {!poseData && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center text-[var(--color-text-tertiary)]">
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
