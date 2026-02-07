'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Bug, ChevronDown, ChevronUp } from 'lucide-react'
import type { PoseDebugInfo, LandmarkPoseData } from '@/types/pose'

/**
 * Debug Panel Component
 * =====================
 * 
 * Collapsible debug overlay showing pose metrics:
 * - Frame index and timing
 * - Landmark detection rates
 * - Confidence scores
 * - Per-region statistics
 */

interface DebugPanelProps {
  /** Pose data for basic info */
  poseData?: LandmarkPoseData | null
  /** Detailed debug metrics */
  debugInfo?: PoseDebugInfo | null
  /** Current frame index */
  currentFrame?: number
  /** Whether to show the panel */
  visible?: boolean
  /** Callback to toggle visibility */
  onToggle?: () => void
  className?: string
}

export function DebugPanel({
  poseData,
  debugInfo,
  currentFrame = 0,
  visible = false,
  onToggle,
  className = ''
}: DebugPanelProps) {
  const [expanded, setExpanded] = useState(false)

  if (!visible) return null

  // Get current frame metrics
  const frameMetrics = debugInfo?.frame_metrics?.[currentFrame]

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 10 }}
      className={`absolute bottom-4 left-4 right-4 z-10 ${className}`}
    >
      <div 
        className="bg-black/80 backdrop-blur-md rounded-lg border border-white/10 
                   text-white/90 text-xs font-mono overflow-hidden"
      >
        {/* Header */}
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full flex items-center justify-between px-3 py-2 
                     hover:bg-white/5 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Bug className="w-3.5 h-3.5 text-green-400" />
            <span className="text-green-400 font-medium">DEBUG</span>
            {poseData && (
              <span className="text-white/50">
                {poseData.gloss} · Frame {currentFrame + 1}/{poseData.frame_count}
              </span>
            )}
          </div>
          {expanded ? (
            <ChevronDown className="w-4 h-4 text-white/50" />
          ) : (
            <ChevronUp className="w-4 h-4 text-white/50" />
          )}
        </button>

        {/* Collapsed summary */}
        {!expanded && poseData && (
          <div className="px-3 pb-2 flex gap-4 text-[10px]">
            <span>FPS: {poseData.fps}</span>
            {frameMetrics && (
              <>
                <span className="text-green-400">
                  Pose: {(frameMetrics.pose_detection * 100).toFixed(0)}%
                </span>
                <span className="text-blue-400">
                  L.Hand: {(frameMetrics.left_hand_detection * 100).toFixed(0)}%
                </span>
                <span className="text-pink-400">
                  R.Hand: {(frameMetrics.right_hand_detection * 100).toFixed(0)}%
                </span>
              </>
            )}
          </div>
        )}

        {/* Expanded details */}
        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ height: 0 }}
              animate={{ height: 'auto' }}
              exit={{ height: 0 }}
              className="overflow-hidden"
            >
              <div className="px-3 pb-3 space-y-3 border-t border-white/10 pt-3">
                {/* Basic Info */}
                {poseData && (
                  <div>
                    <div className="text-white/50 mb-1 text-[10px] uppercase tracking-wide">
                      Pose Info
                    </div>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                      <Row label="Gloss" value={poseData.gloss.toUpperCase()} />
                      <Row label="FPS" value={poseData.fps} />
                      <Row label="Frames" value={poseData.frame_count} />
                      <Row label="Duration" value={`${(poseData.frame_count / poseData.fps).toFixed(2)}s`} />
                    </div>
                  </div>
                )}

                {/* Current Frame Metrics */}
                {frameMetrics && (
                  <div>
                    <div className="text-white/50 mb-1 text-[10px] uppercase tracking-wide">
                      Frame Detection
                    </div>
                    <div className="space-y-1.5">
                      <DetectionBar 
                        label="Pose" 
                        value={frameMetrics.pose_detection} 
                        color="bg-green-500"
                      />
                      <DetectionBar 
                        label="Face" 
                        value={frameMetrics.face_detection} 
                        color="bg-purple-500"
                      />
                      <DetectionBar 
                        label="L. Hand" 
                        value={frameMetrics.left_hand_detection} 
                        color="bg-blue-500"
                      />
                      <DetectionBar 
                        label="R. Hand" 
                        value={frameMetrics.right_hand_detection} 
                        color="bg-pink-500"
                      />
                    </div>
                  </div>
                )}

                {/* Overall Statistics */}
                {debugInfo && (
                  <div>
                    <div className="text-white/50 mb-1 text-[10px] uppercase tracking-wide">
                      Overall Stats
                    </div>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                      <Row 
                        label="Pose Detection" 
                        value={`${(debugInfo.detection_rates.pose * 100).toFixed(1)}%`} 
                      />
                      <Row 
                        label="Face Detection" 
                        value={`${(debugInfo.detection_rates.face * 100).toFixed(1)}%`} 
                      />
                      <Row 
                        label="L. Hand Detection" 
                        value={`${(debugInfo.detection_rates.left_hand * 100).toFixed(1)}%`} 
                      />
                      <Row 
                        label="R. Hand Detection" 
                        value={`${(debugInfo.detection_rates.right_hand * 100).toFixed(1)}%`} 
                      />
                      <Row 
                        label="Mean Confidence" 
                        value={debugInfo.confidence_stats.mean.toFixed(3)} 
                      />
                      <Row 
                        label="Std Dev" 
                        value={debugInfo.confidence_stats.std.toFixed(3)} 
                      />
                    </div>
                  </div>
                )}

                {/* Landmark Counts */}
                {debugInfo && (
                  <div>
                    <div className="text-white/50 mb-1 text-[10px] uppercase tracking-wide">
                      Landmark Counts
                    </div>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                      <Row label="Pose" value={debugInfo.landmark_counts.pose} />
                      <Row label="Face" value={debugInfo.landmark_counts.face} />
                      <Row label="L. Hand" value={debugInfo.landmark_counts.left_hand} />
                      <Row label="R. Hand" value={debugInfo.landmark_counts.right_hand} />
                      <Row label="Total" value={debugInfo.landmark_counts.total} highlight />
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}

// Helper components
function Row({ 
  label, 
  value, 
  highlight = false 
}: { 
  label: string; 
  value: string | number; 
  highlight?: boolean 
}) {
  return (
    <div className="flex justify-between">
      <span className="text-white/50">{label}:</span>
      <span className={highlight ? 'text-green-400' : ''}>{value}</span>
    </div>
  )
}

function DetectionBar({ 
  label, 
  value, 
  color 
}: { 
  label: string; 
  value: number; 
  color: string 
}) {
  const percent = value * 100
  
  return (
    <div className="flex items-center gap-2">
      <span className="w-14 text-white/50">{label}</span>
      <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percent}%` }}
          className={`h-full ${color} rounded-full`}
          transition={{ duration: 0.3 }}
        />
      </div>
      <span className="w-10 text-right">{percent.toFixed(0)}%</span>
    </div>
  )
}
