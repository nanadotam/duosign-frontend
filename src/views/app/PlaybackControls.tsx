'use client'

import { Download, ChevronsLeft, Play, Pause, ChevronsRight, Timer } from 'lucide-react'
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

  return (
    <div className="flex items-center justify-center gap-6 py-4 px-6 border-t border-[var(--panel-border)]">
      {/* Download */}
      <button
        onClick={onDownload}
        className="p-2 rounded-[var(--radius-md)] text-[var(--color-text-primary)] hover:bg-[var(--color-light-gray)] transition-colors"
        aria-label="Download"
      >
        <Download className="h-5 w-5" />
      </button>

      {/* Rewind */}
      <button
        onClick={onRestart}
        className="p-2 rounded-[var(--radius-md)] text-[var(--color-text-primary)] hover:bg-[var(--color-light-gray)] transition-colors"
        aria-label="Rewind"
      >
        <ChevronsLeft className="h-6 w-6" />
      </button>

      {/* Play/Pause */}
      <button
        onClick={onPlayPause}
        className="p-2 rounded-[var(--radius-md)] text-[var(--color-text-primary)] hover:bg-[var(--color-light-gray)] transition-colors"
        aria-label={playback.isPlaying ? "Pause" : "Play"}
      >
        {playback.isPlaying ? (
          <Pause className="h-6 w-6" />
        ) : (
          <Play className="h-6 w-6" />
        )}
      </button>

      {/* Forward */}
      <button
        onClick={onForward}
        className="p-2 rounded-[var(--radius-md)] text-[var(--color-text-primary)] hover:bg-[var(--color-light-gray)] transition-colors"
        aria-label="Forward"
      >
        <ChevronsRight className="h-6 w-6" />
      </button>

      {/* Speed */}
      <button
        onClick={onSpeedChange}
        className="p-2 rounded-[var(--radius-md)] text-[var(--color-text-primary)] hover:bg-[var(--color-light-gray)] transition-colors relative"
        aria-label={`Speed: ${speedLabel}`}
      >
        <Timer className="h-5 w-5" />
        <span className="absolute -top-1 -right-1 text-[9px] bg-[var(--color-primary)] text-white rounded-full w-4 h-4 flex items-center justify-center font-bold">
          {speedLabel}
        </span>
      </button>
    </div>
  )
}
