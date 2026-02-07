'use client'

import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, AlertTriangle, Sparkles, Check, X, ChevronDown, Terminal, Play } from 'lucide-react'

const API_BASE_URL = 'http://localhost:8000'

export interface GlossItem {
  index: number
  gloss: string
  original_word: string | null
  available: boolean
  video_id: string | null
}

interface DebugInfo {
  available_count: number
  missing_count: number
  total_count: number
  missing_glosses: string[]
  available_glosses: string[]
}

interface TextToGlossResponse {
  text: string
  gloss_string: string
  glosses: GlossItem[]
  method: string
  confidence: number
  debug: DebugInfo
}

interface TextToGlossInputProps {
  onGlossSelect?: (gloss: string, videoId: string | null) => void
  /** Called with full gloss sequence when text-to-gloss completes */
  onSequenceReady?: (glosses: GlossItem[]) => void
  /** Index of the currently-playing gloss in the sequence (-1 = none) */
  activeGlossIndex?: number
  /** Whether sequence playback is in progress */
  sequencePlaying?: boolean
}

export function TextToGlossInput({
  onGlossSelect,
  onSequenceReady,
  activeGlossIndex = -1,
  sequencePlaying = false
}: TextToGlossInputProps) {
  const [inputText, setInputText] = useState('')
  const [result, setResult] = useState<TextToGlossResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showDebug, setShowDebug] = useState(false)

  const handleConvert = useCallback(async () => {
    if (!inputText.trim()) return

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/text-to-gloss`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText.trim() })
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }

      const data: TextToGlossResponse = await response.json()
      setResult(data)

      // Notify parent with the full gloss sequence for auto-play
      if (onSequenceReady) {
        onSequenceReady(data.glosses)
      }
    } catch (err) {
      console.error('Text-to-gloss conversion failed:', err)
      setError('Could not connect to pose API. Make sure the server is running.')
    } finally {
      setLoading(false)
    }
  }, [inputText, onSequenceReady])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleConvert()
    }
  }

  const handlePlayAll = useCallback(() => {
    if (result && onSequenceReady) {
      onSequenceReady(result.glosses)
    }
  }, [result, onSequenceReady])

  /**
   * Get chip style based on playback state:
   * - Green: currently playing
   * - Yellow/amber: queued (not yet played) or already played
   * - Default: no sequence playing
   */
  const getChipClass = (itemIndex: number, available: boolean): string => {
    if (!available) return 'gloss-chip gloss-chip-unavailable'

    if (sequencePlaying && activeGlossIndex >= 0) {
      if (itemIndex === activeGlossIndex) {
        return 'gloss-chip border-2 border-green-500 bg-green-500/20 text-green-300 shadow-[0_0_8px_rgba(34,197,94,0.3)]'
      }
      return 'gloss-chip border border-amber-500/50 bg-amber-500/10 text-amber-300'
    }

    return 'gloss-chip gloss-chip-available'
  }

  return (
    <div className="space-y-5">
      {/* Input Section */}
      <div className="space-y-2">
        <label className="text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider">
          Enter English Text
        </label>

        <div className="relative">
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a sentence... (e.g., 'Can you help me?')"
            className="input resize-none pr-14"
            rows={2}
          />

          {/* Send Button - Inside input */}
          <motion.button
            onClick={handleConvert}
            disabled={loading || !inputText.trim()}
            className="absolute right-2 top-1/2 -translate-y-1/2
                       w-10 h-10 rounded-[var(--radius-md)] flex items-center justify-center
                       bg-[var(--color-primary)] text-white
                       hover:bg-[var(--color-primary-dark)]
                       disabled:bg-[var(--color-gray-300)] dark:disabled:bg-[var(--color-gray-600)]
                       disabled:cursor-not-allowed
                       shadow-[var(--shadow-primary)] disabled:shadow-none
                       transition-all duration-200"
            whileHover={{ scale: loading ? 1 : 1.05 }}
            whileTap={{ scale: loading ? 1 : 0.95 }}
          >
            {loading ? (
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </motion.button>
        </div>
      </div>

      {/* Error Message */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10, height: 0 }}
            animate={{ opacity: 1, y: 0, height: 'auto' }}
            exit={{ opacity: 0, y: -10, height: 0 }}
            className="overflow-hidden"
          >
            <div className="p-4 rounded-[var(--radius-lg)] bg-[var(--color-error-subtle)] border border-[var(--color-error-border)]">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-lg bg-[var(--color-error)]/10 flex items-center justify-center shrink-0">
                  <AlertTriangle className="h-4 w-4 text-[var(--color-error)]" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-[var(--color-error)]">{error}</p>
                  <div className="mt-2 flex items-center gap-2 text-xs text-[var(--color-error)]/70">
                    <Terminal className="h-3 w-3" />
                    <code className="font-mono bg-[var(--color-error)]/10 px-1.5 py-0.5 rounded">
                      cd duosign_algo && uvicorn api.main:app --port 8000
                    </code>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Results Section */}
      <AnimatePresence mode="wait">
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-4"
          >
            {/* Gloss String Result */}
            <div className="p-4 rounded-[var(--radius-lg)] bg-gradient-to-br from-[var(--color-primary-subtle)] to-[var(--color-accent-subtle)] border border-[var(--color-primary)]/10">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-[var(--color-primary)]" />
                  <span className="text-xs font-medium text-[var(--color-text-tertiary)] uppercase tracking-wide">
                    ASL Gloss
                  </span>
                </div>
                {/* Play All button */}
                {result.debug.available_count > 0 && !sequencePlaying && (
                  <motion.button
                    onClick={handlePlayAll}
                    className="flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium
                             bg-green-600 text-white hover:bg-green-700 transition-colors"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Play className="h-3 w-3" />
                    <span>Play All</span>
                  </motion.button>
                )}
                {sequencePlaying && (
                  <span className="flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium bg-green-600/20 text-green-400">
                    <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                    Playing...
                  </span>
                )}
              </div>
              <p className="font-mono text-lg font-semibold text-[var(--color-text-primary)] tracking-wide">
                {result.gloss_string}
              </p>
            </div>

            {/* Gloss Breakdown */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider">
                  Breakdown
                </span>
                <span className="badge badge-success">
                  {result.debug.available_count}/{result.debug.total_count} available
                </span>
              </div>

              <div className="flex flex-wrap gap-2">
                {result.glosses.map((item, index) => (
                  <motion.button
                    key={item.index}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.03, ease: [0.16, 1, 0.3, 1] }}
                    onClick={() => item.available && onGlossSelect?.(item.gloss, item.video_id)}
                    disabled={!item.available}
                    className={getChipClass(item.index, item.available)}
                    title={item.available
                      ? `Click to play: ${item.gloss}`
                      : `No pose data for: ${item.gloss}`
                    }
                  >
                    {item.available ? (
                      <Check className="h-3 w-3" />
                    ) : (
                      <X className="h-3 w-3" />
                    )}
                    <span>{item.gloss}</span>
                    {item.original_word && (
                      <span className="text-[10px] opacity-60">({item.original_word})</span>
                    )}
                  </motion.button>
                ))}
              </div>
            </div>

            {/* Debug Info - Collapsible */}
            <div className="border-t border-[var(--panel-border)] pt-3">
              <button
                onClick={() => setShowDebug(!showDebug)}
                className="flex items-center gap-2 text-xs text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)] transition-colors"
              >
                <motion.div
                  animate={{ rotate: showDebug ? 180 : 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <ChevronDown className="h-3 w-3" />
                </motion.div>
                <span>Debug Info</span>
              </button>

              <AnimatePresence>
                {showDebug && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="mt-3 p-3 rounded-[var(--radius-md)] bg-[var(--panel-content-bg)] font-mono text-xs space-y-1 text-[var(--color-text-tertiary)]">
                      <div><span className="text-[var(--color-text-secondary)]">Method:</span> {result.method}</div>
                      <div><span className="text-[var(--color-text-secondary)]">Confidence:</span> {(result.confidence * 100).toFixed(0)}%</div>
                      <div><span className="text-[var(--color-success)]">Available:</span> {result.debug.available_glosses.join(', ') || 'none'}</div>
                      <div><span className="text-[var(--color-error)]">Missing:</span> {result.debug.missing_glosses.join(', ') || 'none'}</div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
