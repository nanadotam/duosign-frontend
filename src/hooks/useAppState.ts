/**
 * useAppState Hook
 * 
 * React hook that integrates the MVC controllers with React's state management.
 * This hook bridges the imperative controller API with React's declarative model.
 */

'use client'

import { useState, useCallback, useEffect, useMemo } from 'react'
import type { AppState, HistoryItem, PlaybackState } from '@/models'
import { AppController } from '@/controllers'

interface UseAppStateReturn {
  // State
  appState: AppState
  history: HistoryItem[]
  selectedHistoryItem: HistoryItem | null
  playback: PlaybackState

  // State setters
  setAppState: (state: AppState) => void

  // History actions
  addToHistory: (text: string) => void
  clearAllHistory: () => void
  selectHistoryItem: (item: HistoryItem) => void

  // Playback actions
  setPlayback: (state: Partial<PlaybackState>) => void

  // Translation actions
  submitTranslation: (text: string) => Promise<void>
}

export function useAppState(): UseAppStateReturn {
  // Initialize controller once
  const [controller] = useState(() => new AppController())

  // Local state synced with controller
  const [appState, setAppStateLocal] = useState<AppState>('HERO')
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [selectedHistoryItem, setSelectedHistoryItem] = useState<HistoryItem | null>(null)
  const [playback, setPlaybackLocal] = useState<PlaybackState>({
    isPlaying: false,
    speed: 1,
    currentTime: 0
  })

  // Initialize controller and load data on mount
  useEffect(() => {
    const initialState = controller.initialize()
    setHistory(initialState.history)
    setAppStateLocal(initialState.appState)
    setPlaybackLocal(initialState.playback)
  }, [controller])

  // Set app state via controller
  const setAppState = useCallback((state: AppState) => {
    controller.setAppState(state)
    setAppStateLocal(state)
  }, [controller])

  // Add to history (internal use)
  const addToHistory = useCallback((text: string) => {
    // This is typically called through submitTranslation
  }, [])

  // Clear all history
  const clearAllHistory = useCallback(() => {
    controller.clearHistory()
    setHistory([])
    setSelectedHistoryItem(null)
  }, [controller])

  // Select a history item
  const selectHistoryItem = useCallback((item: HistoryItem) => {
    controller.selectHistoryItem(item)
    setSelectedHistoryItem(item)
    setAppStateLocal('READY')
  }, [controller])

  // Update playback state
  const setPlayback = useCallback((partial: Partial<PlaybackState>) => {
    controller.updatePlayback(partial)
    setPlaybackLocal(prev => ({ ...prev, ...partial }))
  }, [controller])

  // Submit translation
  const submitTranslation = useCallback(async (text: string) => {
    if (!text.trim()) return

    setAppStateLocal('PROCESSING')
    
    // Add to history locally for immediate feedback
    const newItem: HistoryItem = {
      id: crypto.randomUUID(),
      text,
      timestamp: new Date(),
    }
    setHistory(prev => [newItem, ...prev].slice(0, 50))
    setSelectedHistoryItem(newItem)

    // Submit via controller
    await controller.submitTranslation(text)

    // Sync state from controller
    const state = controller.getState()
    setAppStateLocal(state.appState)
    setPlaybackLocal(state.playback)
    setHistory(state.history)
  }, [controller])

  return {
    // State
    appState,
    history,
    selectedHistoryItem,
    playback,

    // State setters
    setAppState,

    // History actions
    addToHistory,
    clearAllHistory,
    selectHistoryItem,

    // Playback actions
    setPlayback,

    // Translation actions
    submitTranslation
  }
}
