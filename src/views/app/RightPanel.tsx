'use client'

import { OutputPlayer } from './OutputPlayer'
import { PlaybackControls } from './PlaybackControls'
import { StatusText } from './StatusText'
import type { AppState, PlaybackState } from '@/models'
import type { PoseDataV3 } from '@/utils/applyPoseFrame'
import { AlertCircle, WifiOff, Share2 } from 'lucide-react'
import { Button } from '@/views/ui/button'
import { motion, AnimatePresence } from 'framer-motion'

interface RightPanelProps {
  appState: AppState
  playback: PlaybackState
  onPlayPause: () => void
  onRestart: () => void
  onSpeedChange: () => void
  onRetry?: () => void
  onDownload?: () => void
  onForward?: () => void
  onShare?: () => void
  poseData?: PoseDataV3 | null
  currentFrame?: number
  onFrameChange?: (frame: number) => void
}

export function RightPanel({
  appState,
  playback,
  onPlayPause,
  onRestart,
  onSpeedChange,
  onRetry,
  onDownload,
  onForward,
  onShare,
  poseData,
  currentFrame,
  onFrameChange
}: RightPanelProps) {
  const isProcessing = appState === 'PROCESSING'
  const isReady = appState === 'READY'
  const isError = appState === 'ERROR'
  const isOffline = appState === 'OFFLINE'

  return (
    <div className={`flex flex-col h-full bg-[var(--panel-bg)] rounded-[var(--radius-xl)] overflow-hidden transition-all hover:shadow-[var(--shadow-panel)] ${
      isError ? 'ring-1 ring-[var(--color-error)]/20' : isOffline ? 'ring-1 ring-[var(--color-warning)]/20' : ''
    }`}>
      {/* Offline banner */}
      <AnimatePresence>
        {isOffline && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="bg-[var(--color-warning)]/10 border-b border-[var(--color-warning)]/20 px-4 py-3 flex items-center gap-2"
          >
            <WifiOff className="h-4 w-4 text-[var(--color-warning)]" />
            <span className="text-sm text-[var(--color-warning)]">You&apos;re offline. Please check your connection.</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Share button */}
      <div className="flex items-center justify-end p-4 pb-0">
        <button
          onClick={onShare}
          className="p-2 rounded-[var(--radius-md)] text-[var(--color-text-primary)] hover:bg-[var(--color-light-gray)] transition-colors"
          aria-label="Share"
        >
          <Share2 className="h-5 w-5" />
        </button>
      </div>

      {/* Main content area */}
      <div className="flex-1 px-[var(--panel-padding)] pb-2 flex flex-col items-center justify-center">
        <AnimatePresence mode="wait">
          {isError ? (
            <motion.div
              key="error"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="text-center"
            >
              <div className="w-16 h-16 rounded-full bg-[var(--color-error)]/10 flex items-center justify-center mx-auto mb-4">
                <AlertCircle className="h-8 w-8 text-[var(--color-error)]" />
              </div>
              <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-2">
                Translation failed
              </h3>
              <p className="text-sm text-[var(--color-text-secondary)] mb-4">
                Please try again.
              </p>
              {onRetry && (
                <Button variant="cta" onClick={onRetry}>
                  RETRY
                </Button>
              )}
            </motion.div>
          ) : (
            <motion.div
              key="player"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="w-full h-full"
            >
              <OutputPlayer
                isReady={isReady}
                isPlaying={playback.isPlaying}
                speed={playback.speed}
                poseData={poseData}
                currentFrame={currentFrame}
                onFrameChange={onFrameChange}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Status text / progress bars */}
      <StatusText isProcessing={isProcessing} />

      {/* Playback controls */}
      {isReady && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <PlaybackControls
            playback={playback}
            onPlayPause={onPlayPause}
            onRestart={onRestart}
            onSpeedChange={onSpeedChange}
            onDownload={onDownload}
            onForward={onForward}
          />
        </motion.div>
      )}
    </div>
  )
}
