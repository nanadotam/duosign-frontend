'use client'

import { useState, useEffect } from 'react'
import { GlossCard } from './GlossCard'
import { motion } from 'framer-motion'

/**
 * Sign entry from API
 */
export interface GlossEntry {
  gloss: string       // Human-readable gloss name (e.g., "able")
  video_id: string    // WLASL video ID (e.g., "00384")
  frame_count: number
  duration_sec: number
  file_size_kb: number
  // Legacy fields for backwards compatibility
  words?: string
  glosses?: string
}

interface GlossPickerProps {
  onSelectGloss: (entry: GlossEntry) => void
  selectedGloss?: string | null
}

const API_BASE_URL = 'http://localhost:8000'

/**
 * Grid of clickable gloss cards
 * Loads entries from the pose data API
 */
export function GlossPicker({ onSelectGloss, selectedGloss }: GlossPickerProps) {
  const [entries, setEntries] = useState<GlossEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Load available signs from API on mount
  useEffect(() => {
    async function loadSigns() {
      try {
        const response = await fetch(`${API_BASE_URL}/api/signs`)
        if (!response.ok) throw new Error(`API error: ${response.status}`)
        
        const data: GlossEntry[] = await response.json()
        
        // Add backwards compatibility fields
        const enriched = data.map(entry => ({
          ...entry,
          words: entry.gloss,
          glosses: entry.gloss.toUpperCase(),
        }))
        
        // Group by gloss name to avoid duplicates in UI
        const uniqueGlosses = new Map<string, GlossEntry>()
        for (const entry of enriched) {
          if (!uniqueGlosses.has(entry.gloss)) {
            uniqueGlosses.set(entry.gloss, entry)
          }
        }
        
        setEntries(Array.from(uniqueGlosses.values()))
        setLoading(false)
      } catch (err) {
        console.error('Failed to load signs:', err)
        setError('Could not connect to pose API. Make sure the server is running.')
        setLoading(false)
      }
    }
    
    loadSigns()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-4">
        <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
        <span className="ml-2 text-sm text-neutral-500">Loading signs...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center py-4">
        <p className="text-red-500 text-sm mb-2">{error}</p>
        <p className="text-xs text-neutral-400">
          Start the API: <code className="bg-neutral-100 px-1 rounded">cd duosign_algo && uvicorn api.main:app --port 8000</code>
        </p>
      </div>
    )
  }

  if (entries.length === 0) {
    return (
      <div className="text-center py-4 text-neutral-500 text-sm">
        No signs available. Extract some poses first.
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold text-neutral-500 uppercase tracking-wide">
          Available Signs ({entries.length})
        </h3>
      </div>
      
      <motion.div 
        className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-2"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ staggerChildren: 0.05 }}
      >
        {entries.map((entry) => (
          <motion.div
            key={entry.video_id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <GlossCard
              gloss={entry.gloss.toUpperCase()}
              word={entry.gloss}
              isSelected={selectedGloss === entry.glosses || selectedGloss === entry.gloss.toUpperCase()}
              onClick={() => onSelectGloss(entry)}
            />
          </motion.div>
        ))}
      </motion.div>
    </div>
  )
}
