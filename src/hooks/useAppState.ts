'use client'

import { useState, useCallback, useEffect } from 'react'
import type { AppState, HistoryItem, PlaybackState } from '@/lib/types'
import { getHistory, addHistoryItem, clearHistory } from '@/lib/storage'

interface UseAppStateReturn {
  appState: AppState
  setAppState: (state: AppState) => void
  history: HistoryItem[]
  addToHistory: (text: string) => void
  clearAllHistory: () => void
  selectedHistoryItem: HistoryItem | null
  selectHistoryItem: (item: HistoryItem) => void
  playback: PlaybackState
  setPlayback: (state: Partial<PlaybackState>) => void
  submitTranslation: (text: string) => Promise<void>
}

export function useAppState(): UseAppStateReturn {
  const [appState, setAppState] = useState<AppState>('HERO')
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [selectedHistoryItem, setSelectedHistoryItem] = useState<HistoryItem | null>(null)
  const [playback, setPlaybackState] = useState<PlaybackState>({
    isPlaying: false,
    speed: 1,
    currentTime: 0
  })

  // Load history on mount
  useEffect(() => {
    setHistory(getHistory())
  }, [])

  const addToHistory = useCallback((text: string) => {
    const newItem: HistoryItem = {
      id: crypto.randomUUID(),
      text,
      timestamp: new Date(),
    }
    const updated = addHistoryItem(newItem)
    setHistory(updated)
    setSelectedHistoryItem(newItem)
  }, [])

  const clearAllHistory = useCallback(() => {
    clearHistory()
    setHistory([])
    setSelectedHistoryItem(null)
  }, [])

  const selectHistoryItem = useCallback((item: HistoryItem) => {
    setSelectedHistoryItem(item)
    setAppState('READY')
  }, [])

  const setPlayback = useCallback((partial: Partial<PlaybackState>) => {
    setPlaybackState(prev => ({ ...prev, ...partial }))
  }, [])

  const submitTranslation = useCallback(async (text: string) => {
    if (!text.trim()) return

    setAppState('PROCESSING')
    addToHistory(text)

    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 2500))

    // Check if online (simplified check)
    if (!navigator.onLine) {
      setAppState('OFFLINE')
      return
    }

    // Simulate success (in real app, this would call API)
    setAppState('READY')
    setPlayback({ isPlaying: true })
  }, [addToHistory, setPlayback])

  return {
    appState,
    setAppState,
    history,
    addToHistory,
    clearAllHistory,
    selectedHistoryItem,
    selectHistoryItem,
    playback,
    setPlayback,
    submitTranslation
  }
}
