'use client'

import { OutputPlayer } from './OutputPlayer'
import { PlaybackControls } from './PlaybackControls'
import { StatusText } from './StatusText'
import type { AppState, PlaybackState } from '@/models'
import type { PoseDataV3 } from '@/utils/applyPoseFrame'
import { AlertCircle, WifiOff, Share2, RefreshCw } from 'lucide-react'
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
    <motion.div 
      className={`flex flex-col h-full panel overflow-hidden ${
        isError ? 'ring-2 ring-[var(--color-error)]/20' : 
        isOffline ? 'ring-2 ring-[var(--color-warning)]/20' : ''
      }`}
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1], delay: 0.1 }}
    >
      {/* Offline Banner */}
      <AnimatePresence>
        {isOffline && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="bg-[var(--color-warning-subtle)] border-b border-[var(--color-warning)]/20 px-5 py-3 flex items-center gap-3"
          >
            <div className="w-8 h-8 rounded-full bg-[var(--color-warning)]/10 flex items-center justify-center">
              <WifiOff className="h-4 w-4 text-[var(--color-warning)]" />
            </div>
            <span className="text-sm font-medium text-[var(--color-warning)]">
              You&apos;re offline. Please check your connection.
            </span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header with Share */}
      <div className="flex items-center justify-end p-4 pb-0">
        <motion.button
          onClick={onShare}
          className="p-2.5 rounded-[var(--radius-md)] text-[var(--color-text-secondary)]
                     hover:text-[var(--color-text-primary)] hover:bg-[var(--panel-content-bg)]
                     border border-transparent hover:border-[var(--panel-border)]
                     transition-all duration-200"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          aria-label="Share"
        >
          <Share2 className="h-5 w-5" />
        </motion.button>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 px-[var(--panel-padding)] pb-4 flex flex-col items-center justify-center min-h-0">
        <AnimatePresence mode="wait">
          {isError ? (
            <motion.div
              key="error"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
              className="text-center max-w-sm"
            >
              {/* Error Icon */}
              <div className="w-20 h-20 rounded-2xl bg-[var(--color-error-subtle)] 
                            flex items-center justify-center mx-auto mb-6
                            shadow-[0_0_40px_-10px_var(--color-error)]">
                <AlertCircle className="h-10 w-10 text-[var(--color-error)]" />
              </div>
              
              <h3 className="text-xl font-semibold text-[var(--color-text-primary)] mb-2">
                Translation failed
              </h3>
              <p className="text-sm text-[var(--color-text-secondary)] mb-6 leading-relaxed">
                Something went wrong. Please try again or check your connection.
              </p>
              
              {onRetry && (
                <motion.button
                  onClick={onRetry}
                  className="btn btn-primary"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <RefreshCw className="h-4 w-4" />
                  <span>Try Again</span>
                </motion.button>
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

      {/* Status / Progress */}
      <StatusText isProcessing={isProcessing} />

      {/* Playback Controls */}
      <AnimatePresence>
        {isReady && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
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
      </AnimatePresence>
    </motion.div>
  )
}
