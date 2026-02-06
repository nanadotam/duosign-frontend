'use client'

import { useState } from 'react'
import { SkeletonRenderer } from '@/components/app/SkeletonRenderer'
import { AvatarRenderer } from '@/components/app/AvatarRenderer'
import type { PoseDataV3 } from '@/utils/applyPoseFrame'
import { motion, AnimatePresence } from 'framer-motion'
import { User, Activity, Hand } from 'lucide-react'

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
    <div className="relative w-full h-full min-h-[350px] rounded-[var(--radius-xl)] overflow-hidden panel-content">
      {/* Renderer Content */}
      <AnimatePresence mode="wait">
        {isReady && poseData ? (
          <motion.div
            key="renderer"
            className="absolute inset-0"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
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
          <motion.div 
            key="placeholder"
            className="absolute inset-0 flex flex-col items-center justify-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {/* Animated placeholder silhouette */}
            <motion.div
              className="relative"
              animate={{ y: [0, -8, 0] }}
              transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
            >
              {/* Subtle glow behind */}
              <div className="absolute inset-0 blur-3xl opacity-20 bg-[var(--color-primary)]" />
              
              {/* Avatar silhouette */}
              <svg
                className="w-32 h-44 text-[var(--color-gray-300)] dark:text-[var(--color-gray-600)]"
                viewBox="0 0 100 140"
                fill="currentColor"
              >
                {/* Head */}
                <circle cx="50" cy="28" r="20" opacity="0.5" />
                {/* Body */}
                <path 
                  d="M25 58 Q25 48 50 48 Q75 48 75 58 L75 130 Q75 138 67 138 L33 138 Q25 138 25 130 Z" 
                  opacity="0.3"
                />
                {/* Hands hint */}
                <circle cx="15" cy="85" r="8" opacity="0.2" />
                <circle cx="85" cy="85" r="8" opacity="0.2" />
              </svg>
            </motion.div>
            
            {/* Instructions */}
            <motion.div 
              className="mt-8 text-center"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full 
                            bg-[var(--panel-bg)] border border-[var(--panel-border)]
                            shadow-[var(--shadow-sm)]">
                <Hand className="h-4 w-4 text-[var(--color-primary)]" />
                <span className="text-sm font-medium text-[var(--color-text-secondary)]">
                  Select a sign to view
                </span>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Render Mode Toggle */}
      <AnimatePresence>
        {poseData && (
          <motion.div 
            className="absolute top-4 left-4 z-20"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.3 }}
          >
            <div className="flex items-center gap-1 p-1 glass rounded-[var(--radius-lg)] shadow-[var(--shadow-md)]">
              <button
                onClick={() => setRenderMode('avatar')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-[var(--radius-md)] text-xs font-medium transition-all duration-200 ${
                  renderMode === 'avatar'
                    ? 'bg-[var(--color-primary)] text-white shadow-[var(--shadow-primary)]'
                    : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]'
                }`}
                title="3D Avatar View"
              >
                <User className="h-3.5 w-3.5" />
                <span>Avatar</span>
              </button>
              <button
                onClick={() => setRenderMode('skeleton')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-[var(--radius-md)] text-xs font-medium transition-all duration-200 ${
                  renderMode === 'skeleton'
                    ? 'bg-[var(--color-primary)] text-white shadow-[var(--shadow-primary)]'
                    : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]'
                }`}
                title="2D Skeleton View"
              >
                <Activity className="h-3.5 w-3.5" />
                <span>Skeleton</span>
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Playing Indicator */}
      <AnimatePresence>
        {isPlaying && poseData && (
          <motion.div 
            className="absolute top-4 right-4 z-10"
            initial={{ opacity: 0, x: 10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 10 }}
          >
            <div className="flex items-center gap-2 px-3 py-1.5 glass rounded-full shadow-[var(--shadow-sm)]">
              <div className="relative">
                <div className="w-2 h-2 rounded-full bg-[var(--color-success)]" />
                <div className="absolute inset-0 w-2 h-2 rounded-full bg-[var(--color-success)] animate-ping" />
              </div>
              <span className="text-xs font-medium text-[var(--color-text-secondary)]">Playing</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Gloss Name Badge */}
      <AnimatePresence>
        {poseData && (
          <motion.div 
            className="absolute bottom-4 left-4 z-10"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
          >
            <div className="badge badge-primary font-mono uppercase tracking-wide shadow-[var(--shadow-sm)]">
              {poseData.source_video?.split('/').pop()?.replace('.pose', '').replace('.json', '') || 'Unknown'}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
