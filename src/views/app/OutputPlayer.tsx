'use client'

import { useState } from 'react'
import { SkeletonRenderer } from '@/components/app/SkeletonRenderer'
import { AvatarRenderer } from '@/components/app/AvatarRenderer'
import type { PoseDataV3 } from '@/utils/applyPoseFrame'
import { motion } from 'framer-motion'
import { User, Activity } from 'lucide-react'
import { Button } from '@/views/ui/button'

interface OutputPlayerProps {
  isReady: boolean
  isPlaying?: boolean
  speed?: number
  poseData?: PoseDataV3 | null
  currentFrame?: number
  onFrameChange?: (frame: number) => void
}

type RenderMode = 'avatar' | 'skeleton'

export function OutputPlayer({
  isReady,
  isPlaying = false,
  speed = 1,
  poseData = null,
  currentFrame,
  onFrameChange
}: OutputPlayerProps) {
  const [renderMode, setRenderMode] = useState<RenderMode>('avatar')

  return (
    <div className="relative w-full h-full min-h-[350px] rounded-[var(--radius-lg)] overflow-hidden bg-[var(--panel-content-bg)]">
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
        <div className="absolute inset-0 flex items-center justify-center">
          {/* Placeholder avatar silhouette */}
          <motion.div
            className="w-48 h-64 rounded-full bg-[var(--color-light-gray)] opacity-40"
            animate={{
              scale: [1, 1.02, 1],
              opacity: [0.3, 0.45, 0.3]
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
          <div className="absolute inset-0 flex items-center justify-center">
            <svg
              className="w-32 h-48 text-[var(--color-panel-gray)]"
              viewBox="0 0 100 150"
              fill="currentColor"
            >
              <ellipse cx="50" cy="30" rx="20" ry="25" />
              <path d="M20 70 Q20 55 50 55 Q80 55 80 70 L80 140 Q80 150 70 150 L30 150 Q20 150 20 140 Z" />
            </svg>
          </div>
          <div className="absolute bottom-12 text-center">
            <p className="text-sm text-[var(--color-mid-gray)] font-medium">Select a sign to view</p>
          </div>
        </div>
      )}

      {/* Render Mode Toggle */}
      {poseData && (
        <div className="absolute top-4 left-4 z-20">
          <div className="flex items-center gap-1 bg-[var(--panel-bg)] rounded-[var(--radius-md)] p-1">
            <Button
              variant={renderMode === 'avatar' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setRenderMode('avatar')}
              className="h-7 px-2.5"
              title="3D Avatar View"
            >
              <User className="h-3.5 w-3.5 mr-1" />
              <span className="text-xs font-medium">Avatar</span>
            </Button>
            <Button
              variant={renderMode === 'skeleton' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setRenderMode('skeleton')}
              className="h-7 px-2.5"
              title="2D Skeleton View"
            >
              <Activity className="h-3.5 w-3.5 mr-1" />
              <span className="text-xs font-medium">Skeleton</span>
            </Button>
          </div>
        </div>
      )}

      {/* Playing indicator */}
      {isPlaying && poseData && (
        <div className="absolute top-4 right-4 z-10">
          <div className="flex items-center gap-2 bg-[var(--panel-bg)] rounded-[var(--radius-full)] px-3 py-1.5">
            <div className="w-2 h-2 rounded-full bg-[var(--color-success)] animate-pulse" />
            <span className="text-xs font-medium text-[var(--color-text-secondary)]">Playing</span>
          </div>
        </div>
      )}

      {/* Gloss name indicator */}
      {poseData && (
        <div className="absolute bottom-4 left-4 z-10">
          <div className="bg-[var(--color-text-primary)]/80 rounded-[var(--radius-full)] px-3 py-1.5">
            <span className="text-xs font-mono text-[var(--panel-content-bg)] uppercase">
              {poseData.source_video?.split('/').pop()?.replace('.pose', '').replace('.json', '') || 'Unknown'}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
