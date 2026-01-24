'use client'

import { SkeletonRenderer, type PoseData } from '@/components/app/SkeletonRenderer'
import { motion } from 'framer-motion'

interface OutputPlayerProps {
  isReady: boolean
  isPlaying?: boolean
  speed?: number
  poseData?: PoseData | null
  currentFrame?: number
  onFrameChange?: (frame: number) => void
}

export function OutputPlayer({ 
  isReady, 
  isPlaying = false,
  speed = 1,
  poseData = null,
  currentFrame,
  onFrameChange
}: OutputPlayerProps) {
  return (
    <div className="relative w-full h-full min-h-[350px] rounded-2xl overflow-hidden bg-slate-900">
      {/* Skeleton Renderer */}
      {isReady && poseData ? (
        <motion.div 
          className="absolute inset-0"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
        >
          <SkeletonRenderer 
            poseData={poseData}
            isPlaying={isPlaying}
            speed={speed}
            currentFrame={currentFrame}
            onFrameChange={onFrameChange}
          />
        </motion.div>
      ) : (
        <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-slate-800 to-slate-900">
          {/* Placeholder avatar silhouette */}
          <motion.div
            className="w-48 h-64 rounded-full bg-gradient-to-b from-slate-700 to-slate-800 opacity-30"
            animate={{
              scale: [1, 1.02, 1],
              opacity: [0.2, 0.35, 0.2]
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
          <div className="absolute inset-0 flex items-center justify-center">
            <svg 
              className="w-32 h-48 text-slate-600" 
              viewBox="0 0 100 150"
              fill="currentColor"
            >
              <ellipse cx="50" cy="30" rx="20" ry="25" />
              <path d="M20 70 Q20 55 50 55 Q80 55 80 70 L80 140 Q80 150 70 150 L30 150 Q20 150 20 140 Z" />
            </svg>
          </div>
          <div className="absolute bottom-12 text-center">
            <p className="text-sm text-slate-400 font-medium">Select a sign to view</p>
          </div>
        </div>
      )}

      {/* Playing indicator */}
      {isPlaying && poseData && (
        <div className="absolute top-4 right-4 z-10">
          <div className="flex items-center gap-2 bg-white/90 backdrop-blur-sm rounded-full px-3 py-1.5 shadow-sm">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-xs font-medium text-neutral-600">Playing</span>
          </div>
        </div>
      )}

      {/* Gloss name indicator */}
      {poseData && (
        <div className="absolute top-4 left-4 z-10">
          <div className="bg-black/60 backdrop-blur-sm rounded-full px-3 py-1.5">
            <span className="text-xs font-mono text-white uppercase">
              {poseData.source_video?.split('/').pop()?.replace('.pose', '').replace('.json', '') || 'Unknown'}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}


