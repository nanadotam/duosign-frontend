'use client'

import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useAppState } from '@/hooks/useAppState'
import { TextInput } from '@/components/app/TextInput'
import { LeftPanel } from '@/components/app/LeftPanel'
import { RightPanel } from '@/components/app/RightPanel'
import { Header } from '@/components/layout/Header'
import { Spotlight } from '@/components/ui/spotlight'
import type { PlaybackState } from '@/lib/types'

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

  const [initialText, setInitialText] = useState<string | null>(null)

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

  const handlePlayPause = useCallback(() => {
    setPlayback({ isPlaying: !playback.isPlaying })
  }, [playback.isPlaying, setPlayback])

  const handleRestart = useCallback(() => {
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

  const isHeroState = appState === 'HERO'
  const showTwoPanelLayout = !isHeroState

  return (
    <div className="min-h-screen bg-neutral-50">
      <Header />
      
      <main className="pt-[72px] min-h-screen">
        <AnimatePresence mode="wait">
          {isHeroState ? (
            /* Hero State - Centered input */
            <motion.div
              key="hero"
              className="relative min-h-[calc(100vh-72px)] flex items-center justify-center"
              initial={{ opacity: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.3 }}
            >
              <Spotlight className="-top-40 left-0 md:left-60 md:-top-20" fill="rgba(26, 115, 232, 0.1)" />
              
              {/* Background avatar */}
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <motion.div
                  className="w-[300px] h-[400px] opacity-[0.07]"
                  animate={{
                    y: [0, -8, 0],
                    scale: [1, 1.01, 1]
                  }}
                  transition={{
                    duration: 5,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                >
                  <svg viewBox="0 0 200 300" fill="currentColor" className="w-full h-full text-neutral-900">
                    <ellipse cx="100" cy="60" rx="45" ry="55" />
                    <path d="M35 140 Q35 100 100 100 Q165 100 165 140 L165 280 Q165 300 145 300 L55 300 Q35 300 35 280 Z" />
                  </svg>
                </motion.div>
              </div>

              <div className="relative z-10 w-full max-w-xl mx-auto px-6">
                <motion.h1
                  className="text-2xl md:text-3xl font-semibold text-neutral-900 text-center mb-8"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  What do you want to sign?
                </motion.h1>
                <TextInput onSubmit={submitTranslation} />
              </div>
            </motion.div>
          ) : (
            /* Two-Panel Workspace */
            <motion.div
              key="workspace"
              className="h-[calc(100vh-72px)] p-4 md:p-6"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.4, delay: 0.1 }}
            >
              <div className="h-full grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6 max-w-7xl mx-auto">
                {/* Left Panel */}
                <motion.div
                  initial={{ x: -20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ duration: 0.4, delay: 0.2 }}
                  className="h-full min-h-[400px] md:min-h-0"
                >
                  <LeftPanel
                    onSubmit={submitTranslation}
                    history={history}
                    selectedItem={selectedHistoryItem}
                    onSelectItem={selectHistoryItem}
                    onClearHistory={clearAllHistory}
                  />
                </motion.div>

                {/* Right Panel */}
                <motion.div
                  initial={{ x: 20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ duration: 0.4, delay: 0.3 }}
                  className="h-full min-h-[500px] md:min-h-0"
                >
                  <RightPanel
                    appState={appState}
                    playback={playback}
                    onPlayPause={handlePlayPause}
                    onRestart={handleRestart}
                    onSpeedChange={handleSpeedChange}
                    onRetry={handleRetry}
                  />
                </motion.div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  )
}
