'use client'

import { Download, RotateCcw, Play, Pause, SkipForward, Gauge } from 'lucide-react'
import { motion } from 'framer-motion'
import type { PlaybackState } from '@/models'

interface PlaybackControlsProps {
  playback: PlaybackState
  onPlayPause: () => void
  onRestart: () => void
  onSpeedChange: () => void
  onDownload?: () => void
  onForward?: () => void
}

export function PlaybackControls({
  playback,
  onPlayPause,
  onRestart,
  onSpeedChange,
  onDownload,
  onForward
}: PlaybackControlsProps) {
  const speedLabel = `${playback.speed}x`

  const ControlButton = ({ 
    onClick, 
    ariaLabel, 
    children,
    primary = false 
  }: { 
    onClick?: () => void
    ariaLabel: string
    children: React.ReactNode
    primary?: boolean
  }) => (
    <motion.button
      onClick={onClick}
      className={`p-3 rounded-[var(--radius-lg)] transition-all duration-200 ${
        primary
          ? 'bg-[var(--color-text-primary)] text-[var(--panel-bg)] shadow-[var(--shadow-md)] hover:shadow-[var(--shadow-lg)]'
          : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:bg-[var(--panel-content-bg)]'
      }`}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      aria-label={ariaLabel}
    >
      {children}
    </motion.button>
  )

  return (
    <div className="flex items-center justify-center gap-2 py-5 px-6 border-t border-[var(--panel-border)] bg-[var(--panel-bg)]">
      {/* Download */}
      <ControlButton onClick={onDownload} ariaLabel="Download">
        <Download className="h-5 w-5" />
      </ControlButton>

      {/* Spacer */}
      <div className="w-4" />

      {/* Restart */}
      <ControlButton onClick={onRestart} ariaLabel="Restart">
        <RotateCcw className="h-5 w-5" />
      </ControlButton>

      {/* Play/Pause - Primary */}
      <ControlButton onClick={onPlayPause} ariaLabel={playback.isPlaying ? "Pause" : "Play"} primary>
        {playback.isPlaying ? (
          <Pause className="h-6 w-6" fill="currentColor" />
        ) : (
          <Play className="h-6 w-6 ml-0.5" fill="currentColor" />
        )}
      </ControlButton>

      {/* Forward */}
      <ControlButton onClick={onForward} ariaLabel="Skip forward">
        <SkipForward className="h-5 w-5" />
      </ControlButton>

      {/* Spacer */}
      <div className="w-4" />

      {/* Speed */}
      <motion.button
        onClick={onSpeedChange}
        className="flex items-center gap-2 px-3 py-2 rounded-[var(--radius-lg)] 
                   text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]
                   hover:bg-[var(--panel-content-bg)] transition-all duration-200"
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        aria-label={`Speed: ${speedLabel}`}
      >
        <Gauge className="h-4 w-4" />
        <span className="text-xs font-semibold min-w-[24px]">{speedLabel}</span>
      </motion.button>
    </div>
  )
}
