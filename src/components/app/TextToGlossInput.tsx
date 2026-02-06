'use client'

import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const API_BASE_URL = 'http://localhost:8000'

/**
 * Gloss item from API response
 */
interface GlossItem {
  index: number
  gloss: string
  original_word: string | null
  available: boolean
  video_id: string | null
}

/**
 * Debug info from API response
 */
interface DebugInfo {
  available_count: number
  missing_count: number
  total_count: number
  missing_glosses: string[]
  available_glosses: string[]
}

/**
 * Full API response
 */
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
}

/**
 * Text input and gloss breakdown component
 * Allows users to enter English text and see ASL gloss conversion
 */
export function TextToGlossInput({ onGlossSelect }: TextToGlossInputProps) {
  const [inputText, setInputText] = useState('')
  const [result, setResult] = useState<TextToGlossResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

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
    } catch (err) {
      console.error('Text-to-gloss conversion failed:', err)
      setError('Could not connect to API. Make sure the server is running.')
    } finally {
      setLoading(false)
    }
  }, [inputText])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleConvert()
    }
  }

  return (
    <div className="space-y-4">
      {/* Input section */}
      <div className="space-y-2">
        <label className="text-xs font-semibold text-neutral-500 uppercase tracking-wide">
          Enter English Text
        </label>
        <div className="flex gap-2">
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a sentence... (e.g., 'Can you help me?')"
            className="flex-1 px-3 py-2 text-sm border border-neutral-200 rounded-lg 
                       focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                       resize-none bg-white"
            rows={2}
          />
          <button
            onClick={handleConvert}
            disabled={loading || !inputText.trim()}
            className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg
                       hover:bg-blue-700 disabled:bg-neutral-300 disabled:cursor-not-allowed
                       transition-colors flex items-center gap-2"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Converting...</span>
              </>
            ) : (
              'Translate'
            )}
          </button>
        </div>
      </div>

      {/* Error message */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-3 bg-red-50 border border-red-200 rounded-lg"
        >
          <p className="text-red-600 text-sm">{error}</p>
          <p className="text-red-400 text-xs mt-1">
            Start the API: <code className="bg-red-100 px-1 rounded">cd duosign_algo && uvicorn api.main:app --port 8000</code>
          </p>
        </motion.div>
      )}

      {/* Results section */}
      <AnimatePresence mode="wait">
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-3"
          >
            {/* Gloss string */}
            <div className="p-3 bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-100 rounded-lg">
              <div className="text-xs text-neutral-500 mb-1">ASL Gloss</div>
              <div className="font-mono text-lg font-semibold text-blue-800">
                {result.gloss_string}
              </div>
            </div>

            {/* Gloss breakdown */}
            <div className="space-y-2">
              <div className="text-xs font-semibold text-neutral-500 uppercase tracking-wide">
                Breakdown ({result.debug.available_count}/{result.debug.total_count} available)
              </div>
              <div className="flex flex-wrap gap-2">
                {result.glosses.map((item) => (
                  <motion.button
                    key={item.index}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: item.index * 0.05 }}
                    onClick={() => item.available && onGlossSelect?.(item.gloss, item.video_id)}
                    disabled={!item.available}
                    className={`px-3 py-2 rounded-lg text-sm font-medium border-2 transition-all
                      ${item.available
                        ? 'bg-green-50 border-green-300 text-green-800 hover:bg-green-100 hover:border-green-400 cursor-pointer'
                        : 'bg-neutral-50 border-neutral-200 text-neutral-400 cursor-not-allowed'
                      }`}
                    title={item.available 
                      ? `Click to play: ${item.gloss} (video: ${item.video_id})`
                      : `No pose data for: ${item.gloss}`
                    }
                  >
                    <span className="mr-1">{item.available ? '✓' : '✗'}</span>
                    {item.gloss}
                    {item.original_word && (
                      <span className="ml-1 text-xs opacity-60">({item.original_word})</span>
                    )}
                  </motion.button>
                ))}
              </div>
            </div>

            {/* Debug info */}
            <details className="text-xs text-neutral-500">
              <summary className="cursor-pointer hover:text-neutral-700">Debug Info</summary>
              <div className="mt-2 p-2 bg-neutral-50 rounded-lg font-mono space-y-1">
                <div>Method: {result.method}</div>
                <div>Confidence: {(result.confidence * 100).toFixed(0)}%</div>
                <div>Available: {result.debug.available_glosses.join(', ') || 'none'}</div>
                <div>Missing: {result.debug.missing_glosses.join(', ') || 'none'}</div>
              </div>
            </details>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
