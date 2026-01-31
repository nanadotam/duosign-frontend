'use client'

/**
 * Output Player View
 *
 * Displays animated sign language output using either a 3D avatar or 2D skeleton.
 * Supports switching between rendering modes and handles playback state.
 */

import { useState } from 'react'
import { SkeletonRenderer } from '@/components/app/SkeletonRenderer'
import { AvatarRenderer } from '@/components/app/AvatarRenderer'
import type { PoseDataV3 } from '@/utils/applyPoseFrame'
import { motion } from 'framer-motion'
import { User, Activity } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface OutputPlayerProps {
  isReady: boolean
  isPlaying?: boolean
  speed?: number
  poseData?: PoseDataV3 | null
  currentFrame?: number
  onFrameChange?: (frame: number) => void
}

/**
 * Render mode: Avatar (3D VRM) or Skeleton (2D canvas)
 */
type RenderMode = 'avatar' | 'skeleton'

export function OutputPlayer({
  isReady,
  isPlaying = false,
  speed = 1,
  poseData = null,
  currentFrame,
  onFrameChange
}: OutputPlayerProps) {
  // State for toggling between avatar and skeleton
  const [renderMode, setRenderMode] = useState<RenderMode>('avatar')

  return (
    <div className="relative w-full h-full min-h-[350px] rounded-2xl overflow-hidden bg-slate-900">
      {/* Renderer - Avatar or Skeleton */}
      {isReady && poseData ? (
        <motion.div
          className="absolute inset-0"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
        >
          {renderMode === 'avatar' ? (
            <AvatarRenderer
              poseData={poseData}
              isPlaying={isPlaying}
              speed={speed}
              currentFrame={currentFrame}
              onFrameChange={onFrameChange}
            />
          ) : (
            <SkeletonRenderer
              poseData={poseData}
              isPlaying={isPlaying}
              speed={speed}
              currentFrame={currentFrame}
              onFrameChange={onFrameChange}
            />
          )}
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

      {/* Render Mode Toggle */}
      {poseData && (
        <div className="absolute top-4 left-4 z-20">
          <div className="flex items-center gap-2 bg-black/70 backdrop-blur-sm rounded-lg p-1">
            <Button
              variant={renderMode === 'avatar' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setRenderMode('avatar')}
              className={`h-8 px-3 transition-all ${
                renderMode === 'avatar'
                  ? 'bg-blue-600 hover:bg-blue-700 text-white'
                  : 'text-slate-300 hover:text-white hover:bg-slate-700'
              }`}
              title="3D Avatar View"
            >
              <User className="h-4 w-4 mr-1.5" />
              <span className="text-xs font-medium">Avatar</span>
            </Button>
            <Button
              variant={renderMode === 'skeleton' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setRenderMode('skeleton')}
              className={`h-8 px-3 transition-all ${
                renderMode === 'skeleton'
                  ? 'bg-blue-600 hover:bg-blue-700 text-white'
                  : 'text-slate-300 hover:text-white hover:bg-slate-700'
              }`}
              title="2D Skeleton View"
            >
              <Activity className="h-4 w-4 mr-1.5" />
              <span className="text-xs font-medium">Skeleton</span>
            </Button>
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
        <div className="absolute bottom-4 left-4 z-10">
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


