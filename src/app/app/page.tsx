'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { motion } from 'framer-motion'
import { useAppState } from '@/hooks/useAppState'
import { useTheme } from '@/hooks/useTheme'
import { LeftPanel } from '@/views/app/LeftPanel'
import { RightPanel } from '@/views/app/RightPanel'
import { Header } from '@/views/layout/Header'
import type { PlaybackState } from '@/models'
import type { GlossEntry } from '@/components/app/GlossPicker'
import type { GlossItem } from '@/components/app/TextToGlossInput'
import type { PoseData } from '@/components/app/SkeletonRenderer'

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
  const [selectedGloss, setSelectedGloss] = useState<string | null>(null)
  const [poseData, setPoseData] = useState<PoseData | null>(null)
  const [loadingPose, setLoadingPose] = useState(false)
  const [currentFrame, setCurrentFrame] = useState(0)

  // Sequential playback state
  const [glossSequence, setGlossSequence] = useState<GlossItem[]>([])
  const [currentGlossIndex, setCurrentGlossIndex] = useState(-1)
  const [sequencePlaying, setSequencePlaying] = useState(false)
  const advancingRef = useRef(false)

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

  // ─── Load pose data for a gloss from /lexicon/ase/{gloss}.json ───
  const loadPoseForGloss = useCallback(async (gloss: string): Promise<PoseData | null> => {
    try {
      const response = await fetch(`/lexicon/ase/${gloss.toLowerCase()}.json`)
      if (!response.ok) return null
      return await response.json()
    } catch {
      return null
    }
  }, [])

  // ─── Handle single gloss selection (from GlossPicker or clicking a chip) ───
  const handleSelectGloss = useCallback(async (entry: GlossEntry) => {
    const gloss = entry.words?.toLowerCase() || entry.glosses?.toLowerCase() || entry.gloss?.toLowerCase() || ''

    // Stop any sequence playback
    setSequencePlaying(false)
    setCurrentGlossIndex(-1)
    setGlossSequence([])

    setSelectedGloss(gloss)
    setLoadingPose(true)
    setPlayback({ isPlaying: false })
    setCurrentFrame(0)

    const data = await loadPoseForGloss(gloss)
    if (data) {
      setPoseData(data)
      setAppState('READY')
      setPlayback({ isPlaying: true })
    } else {
      console.error('Failed to load pose for:', gloss)
      setAppState('ERROR')
    }

    setLoadingPose(false)
  }, [setAppState, setPlayback, loadPoseForGloss])

  // ─── Start sequence playback from a gloss list ───
  const handleSequenceReady = useCallback((glosses: GlossItem[]) => {
    // Filter to only available glosses
    const available = glosses.filter(g => g.available)
    if (available.length === 0) return

    setGlossSequence(glosses)
    setCurrentGlossIndex(0)
    setSequencePlaying(true)
    advancingRef.current = false
  }, [])

  // ─── Load and play the current gloss in the sequence ───
  useEffect(() => {
    if (!sequencePlaying || currentGlossIndex < 0) return

    // Find the next available gloss at or after currentGlossIndex
    let idx = currentGlossIndex
    while (idx < glossSequence.length && !glossSequence[idx].available) {
      idx++
    }

    if (idx >= glossSequence.length) {
      // Sequence finished
      setSequencePlaying(false)
      setCurrentGlossIndex(-1)
      return
    }

    // Update index if we skipped unavailable glosses
    if (idx !== currentGlossIndex) {
      setCurrentGlossIndex(idx)
      return
    }

    const gloss = glossSequence[idx].gloss.toLowerCase()
    setSelectedGloss(gloss)
    setLoadingPose(true)
    setCurrentFrame(0)
    advancingRef.current = false

    loadPoseForGloss(gloss).then(data => {
      if (data) {
        setPoseData(data)
        setAppState('READY')
        setPlayback({ isPlaying: true })
      } else {
        // Skip this gloss and advance
        if (idx + 1 < glossSequence.length) {
          setCurrentGlossIndex(idx + 1)
        } else {
          setSequencePlaying(false)
          setCurrentGlossIndex(-1)
        }
      }
      setLoadingPose(false)
    })
  }, [sequencePlaying, currentGlossIndex, glossSequence, loadPoseForGloss, setAppState, setPlayback])

  // ─── Called when a single gloss animation finishes (loop=false) ───
  const handleGlossComplete = useCallback(() => {
    if (!sequencePlaying || advancingRef.current) return
    advancingRef.current = true

    // Brief pause between glosses for visual clarity
    setTimeout(() => {
      const nextIdx = currentGlossIndex + 1
      if (nextIdx < glossSequence.length) {
        setCurrentGlossIndex(nextIdx)
      } else {
        // All done
        setSequencePlaying(false)
        setCurrentGlossIndex(-1)
        setPlayback({ isPlaying: false })
      }
    }, 200)
  }, [sequencePlaying, currentGlossIndex, glossSequence.length, setPlayback])

  // ─── Playback controls ───
  const handlePlayPause = useCallback(() => {
    setPlayback({ isPlaying: !playback.isPlaying })
  }, [playback.isPlaying, setPlayback])

  const handleRestart = useCallback(() => {
    if (sequencePlaying) {
      // Restart the entire sequence
      setCurrentGlossIndex(0)
      setCurrentFrame(0)
      advancingRef.current = false
    } else {
      setCurrentFrame(0)
      setPlayback({ currentTime: 0, isPlaying: true })
    }
  }, [setPlayback, sequencePlaying])

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

  return (
    <div className="min-h-screen bg-[var(--background)] transition-colors duration-300">
      <Header theme={theme} toggleTheme={toggleTheme} />

      <main className="px-4 md:px-6 pb-6">
        <motion.div
          className="h-[calc(100vh-var(--header-height)-24px)] grid grid-cols-1 lg:grid-cols-[1fr_1.2fr] gap-5 max-w-[var(--max-content-width)] mx-auto"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          {/* Left Panel */}
          <div className="h-full min-h-[400px] lg:min-h-0 overflow-hidden">
            <LeftPanel
              history={history}
              selectedItem={selectedHistoryItem}
              onSelectItem={selectHistoryItem}
              onClearHistory={clearAllHistory}
              onSelectGloss={handleSelectGloss}
              selectedGloss={selectedGloss}
              onSequenceReady={handleSequenceReady}
              activeGlossIndex={currentGlossIndex}
              sequencePlaying={sequencePlaying}
            />
          </div>

          {/* Right Panel */}
          <div className="h-full min-h-[500px] lg:min-h-0">
            <RightPanel
              appState={loadingPose ? 'PROCESSING' : appState}
              playback={playback}
              onPlayPause={handlePlayPause}
              onRestart={handleRestart}
              onSpeedChange={handleSpeedChange}
              onRetry={handleRetry}
              poseData={poseData}
              currentFrame={currentFrame}
              onFrameChange={handleFrameChange}
              onGlossComplete={handleGlossComplete}
              sequencePlaying={sequencePlaying}
            />
          </div>
        </motion.div>
      </main>
    </div>
  )
}
