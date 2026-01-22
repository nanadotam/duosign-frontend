'use client'

import { Button } from '@/components/ui/button'
import { Play, Pause, RotateCcw, Gauge } from 'lucide-react'
import type { PlaybackState } from '@/lib/types'

interface PlaybackControlsProps {
  playback: PlaybackState
  onPlayPause: () => void
  onRestart: () => void
  onSpeedChange: () => void
}

export function PlaybackControls({ 
  playback, 
  onPlayPause, 
  onRestart, 
  onSpeedChange 
}: PlaybackControlsProps) {
  const speedLabel = `${playback.speed}x`

  return (
    <div className="flex items-center justify-center gap-4 p-4">
      {/* Restart */}
      <Button
        variant="playback"
        size="playbackIcon"
        onClick={onRestart}
        aria-label="Restart"
        className="shadow-lg hover:scale-105 transition-transform"
      >
        <RotateCcw className="h-6 w-6" />
      </Button>

      {/* Play/Pause - Primary */}
      <Button
        variant="default"
        size="playbackIcon"
        onClick={onPlayPause}
        aria-label={playback.isPlaying ? "Pause" : "Play"}
        className="h-16 w-16 shadow-xl hover:scale-105 transition-transform bg-blue-600 hover:bg-blue-700"
      >
        {playback.isPlaying ? (
          <Pause className="h-7 w-7" />
        ) : (
          <Play className="h-7 w-7 ml-1" />
        )}
      </Button>

      {/* Speed */}
      <Button
        variant="playback"
        size="playbackIcon"
        onClick={onSpeedChange}
        aria-label={`Speed: ${speedLabel}`}
        className="shadow-lg hover:scale-105 transition-transform relative"
      >
        <Gauge className="h-5 w-5" />
        <span className="absolute -bottom-1 -right-1 text-[10px] bg-blue-600 text-white rounded-full px-1.5 py-0.5 font-bold">
          {speedLabel}
        </span>
      </Button>
    </div>
  )
}
