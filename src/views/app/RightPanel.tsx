'use client'

import { OutputPlayer } from './OutputPlayer'
import { PlaybackControls } from './PlaybackControls'
import { StatusText } from './StatusText'
import type { AppState, PlaybackState } from '@/models'
import type { PoseData } from '@/components/app/SkeletonRenderer'
import { AlertCircle, WifiOff } from 'lucide-react'
import { Button } from '@/views/ui/button'
import { motion, AnimatePresence } from 'framer-motion'

interface RightPanelProps {
  appState: AppState
  playback: PlaybackState
  onPlayPause: () => void
  onRestart: () => void
  onSpeedChange: () => void
  onRetry?: () => void
  poseData?: PoseData | null
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
  poseData,
  currentFrame,
  onFrameChange
}: RightPanelProps) {
  const isProcessing = appState === 'PROCESSING'
  const isReady = appState === 'READY'
  const isError = appState === 'ERROR'
  const isOffline = appState === 'OFFLINE'

  return (
    <div className={`flex flex-col h-full bg-white rounded-2xl shadow-sm border overflow-hidden transition-colors ${
      isError ? 'border-red-200' : isOffline ? 'border-amber-200' : 'border-neutral-100'
    }`}>
      {/* Offline banner */}
      <AnimatePresence>
        {isOffline && (
          <motion.div 
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="bg-amber-50 border-b border-amber-200 px-4 py-3 flex items-center gap-2"
          >
            <WifiOff className="h-4 w-4 text-amber-600" />
            <span className="text-sm text-amber-700">You&apos;re offline. Please check your connection.</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main content area */}
      <div className="flex-1 p-6 flex flex-col items-center justify-center">
        <AnimatePresence mode="wait">
          {isError ? (
            <motion.div
              key="error"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="text-center"
            >
              <div className="w-16 h-16 rounded-full bg-red-50 flex items-center justify-center mx-auto mb-4">
                <AlertCircle className="h-8 w-8 text-red-500" />
              </div>
              <h3 className="text-lg font-semibold text-neutral-900 mb-2">
                Something went wrong
              </h3>
              <p className="text-sm text-neutral-500 mb-4">
                We couldn&apos;t process your request. Please try again.
              </p>
              {onRetry && (
                <Button onClick={onRetry}>
                  Try Again
                </Button>
              )}
            </motion.div>
          ) : (
            <motion.div
              key="player"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="w-full"
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

      {/* Status text */}
      <StatusText isProcessing={isProcessing} />

      {/* Playback controls */}
      {isReady && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="border-t border-neutral-100"
        >
          <PlaybackControls
            playback={playback}
            onPlayPause={onPlayPause}
            onRestart={onRestart}
            onSpeedChange={onSpeedChange}
          />
        </motion.div>
      )}
    </div>
  )
}

