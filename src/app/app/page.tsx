'use client'

import { useState, useEffect, useCallback } from 'react'
import { motion } from 'framer-motion'
import { useAppState } from '@/hooks/useAppState'
import { useTheme } from '@/hooks/useTheme'
import { LeftPanel } from '@/views/app/LeftPanel'
import { RightPanel } from '@/views/app/RightPanel'
import { Header } from '@/views/layout/Header'
import type { PlaybackState } from '@/models'
import type { GlossEntry } from '@/components/app/GlossPicker'
import type { PoseDataV3 } from '@/utils/applyPoseFrame'

export default function AppPage() {
  const {
    appState,
    setAppState,
    history,
    clearAllHistory,
    selectedHistoryItem,
    selectHistoryItem,
    playback,
    setPlayback,
    submitTranslation
  } = useAppState()

  const { theme, toggleTheme } = useTheme()

  const [initialText, setInitialText] = useState<string | null>(null)

  // Gloss and pose state
  const [selectedGloss, setSelectedGloss] = useState<string | null>(null)
  const [poseData, setPoseData] = useState<PoseDataV3 | null>(null)
  const [loadingPose, setLoadingPose] = useState(false)
  const [currentFrame, setCurrentFrame] = useState(0)

  // Check for initial text from landing page
  useEffect(() => {
    const storedText = sessionStorage.getItem('duosign_initial_text')
    if (storedText) {
      setInitialText(storedText)
      sessionStorage.removeItem('duosign_initial_text')
    }
  }, [])

  // Auto-submit initial text
  useEffect(() => {
    if (initialText && appState === 'HERO') {
      submitTranslation(initialText)
      setInitialText(null)
    }
  }, [initialText, appState, submitTranslation])

  // Handle gloss selection - load pose data from API
  const handleSelectGloss = useCallback(async (entry: GlossEntry) => {
    setSelectedGloss(entry.glosses ?? entry.gloss)
    setLoadingPose(true)
    setPlayback({ isPlaying: false })
    setCurrentFrame(0)

    try {
      const videoId = entry.video_id || entry.words?.toLowerCase() || entry.gloss
      const response = await fetch(`http://localhost:8000/api/sign/${videoId}`)

      if (!response.ok) {
        throw new Error(`Failed to load pose: ${response.status}`)
      }

      const data: PoseDataV3 = await response.json()
      setPoseData(data)
      setAppState('READY')
      setPlayback({ isPlaying: true })
    } catch (error) {
      console.error('Error loading pose:', error)
      setAppState('ERROR')
    } finally {
      setLoadingPose(false)
    }
  }, [setAppState, setPlayback])

  const handlePlayPause = useCallback(() => {
    setPlayback({ isPlaying: !playback.isPlaying })
  }, [playback.isPlaying, setPlayback])

  const handleRestart = useCallback(() => {
    setCurrentFrame(0)
    setPlayback({ currentTime: 0, isPlaying: true })
  }, [setPlayback])

  const handleSpeedChange = useCallback(() => {
    const speeds: PlaybackState['speed'][] = [0.5, 0.75, 1]
    const currentIndex = speeds.indexOf(playback.speed)
    const nextSpeed = speeds[(currentIndex + 1) % speeds.length]
    setPlayback({ speed: nextSpeed })
  }, [playback.speed, setPlayback])

  const handleRetry = useCallback(() => {
    if (selectedHistoryItem) {
      submitTranslation(selectedHistoryItem.text)
    }
  }, [selectedHistoryItem, submitTranslation])

  const handleFrameChange = useCallback((frame: number) => {
    setCurrentFrame(frame)
  }, [])

  const handleDownload = useCallback(() => {
    // Placeholder for download functionality
  }, [])

  const handleForward = useCallback(() => {
    // Skip forward in animation
    if (poseData) {
      const totalFrames = poseData.frames?.length || 0
      setCurrentFrame(prev => Math.min(prev + 10, totalFrames - 1))
    }
  }, [poseData])

  const handleShare = useCallback(() => {
    if (navigator.share) {
      navigator.share({
        title: 'DuoSign Translation',
        text: selectedHistoryItem?.text || 'Check out this sign language translation',
        url: window.location.href
      }).catch(() => {})
    }
  }, [selectedHistoryItem])

  return (
    <div className="min-h-screen bg-[var(--background)] transition-colors">
      <Header theme={theme} toggleTheme={toggleTheme} />

      <main className="px-[var(--panel-gap)] pb-[var(--panel-gap)]">
        <div className="h-[calc(100vh-var(--header-height)-var(--panel-gap))] grid grid-cols-1 md:grid-cols-[45fr_55fr] gap-[var(--panel-gap)] max-w-[var(--max-content-width)] mx-auto">
          {/* Left Panel */}
          <motion.div
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.4, delay: 0.1 }}
            className="h-full min-h-[400px] md:min-h-0 overflow-hidden"
          >
            <LeftPanel
              history={history}
              selectedItem={selectedHistoryItem}
              onSelectItem={selectHistoryItem}
              onClearHistory={clearAllHistory}
              onSelectGloss={handleSelectGloss}
              selectedGloss={selectedGloss}
            />
          </motion.div>

          {/* Right Panel */}
          <motion.div
            initial={{ x: 20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.4, delay: 0.2 }}
            className="h-full min-h-[500px] md:min-h-0"
          >
            <RightPanel
              appState={loadingPose ? 'PROCESSING' : appState}
              playback={playback}
              onPlayPause={handlePlayPause}
              onRestart={handleRestart}
              onSpeedChange={handleSpeedChange}
              onRetry={handleRetry}
              onDownload={handleDownload}
              onForward={handleForward}
              onShare={handleShare}
              poseData={poseData}
              currentFrame={currentFrame}
              onFrameChange={handleFrameChange}
            />
          </motion.div>
        </div>
      </main>
    </div>
  )
}
