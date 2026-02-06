'use client'

import { useState, useEffect } from 'react'
import { GlossCard } from './GlossCard'
import { motion, AnimatePresence } from 'framer-motion'
import { AlertTriangle, Grid3X3, Terminal } from 'lucide-react'

/**
 * Sign entry from API
 */
export interface GlossEntry {
  gloss: string
  video_id: string
  frame_count: number
  duration_sec: number
  file_size_kb: number
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
 */
export function GlossPicker({ onSelectGloss, selectedGloss }: GlossPickerProps) {
  const [entries, setEntries] = useState<GlossEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function loadSigns() {
      try {
        const response = await fetch(`${API_BASE_URL}/api/signs`)
        if (!response.ok) throw new Error(`API error: ${response.status}`)
        
        const data: GlossEntry[] = await response.json()
        
        const enriched = data.map(entry => ({
          ...entry,
          words: entry.gloss,
          glosses: entry.gloss.toUpperCase(),
        }))
        
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
        setError('Could not connect to pose API')
        setLoading(false)
      }
    }
    
    loadSigns()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center gap-3 py-8">
        <div className="w-5 h-5 border-2 border-[var(--color-primary)] border-t-transparent rounded-full animate-spin" />
        <span className="text-sm text-[var(--color-text-secondary)]">Loading signs...</span>
      </div>
    )
  }

  if (error) {
    return (
      <motion.div 
        className="py-6"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex flex-col items-center text-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-[var(--color-error-subtle)] flex items-center justify-center">
            <AlertTriangle className="h-5 w-5 text-[var(--color-error)]" />
          </div>
          <p className="text-sm font-medium text-[var(--color-error)]">{error}</p>
          <div className="flex items-center gap-2 text-xs text-[var(--color-text-tertiary)]">
            <Terminal className="h-3 w-3" />
            <code className="font-mono bg-[var(--panel-content-bg)] px-2 py-1 rounded-[var(--radius-sm)]">
              uvicorn api.main:app --port 8000
            </code>
          </div>
        </div>
      </motion.div>
    )
  }

  if (entries.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-center gap-3">
        <div className="w-12 h-12 rounded-xl bg-[var(--panel-content-bg)] border border-[var(--panel-border)] flex items-center justify-center">
          <Grid3X3 className="h-5 w-5 text-[var(--color-text-tertiary)]" />
        </div>
        <p className="text-sm text-[var(--color-text-secondary)]">
          No signs available
        </p>
        <p className="text-xs text-[var(--color-text-tertiary)]">
          Extract some poses first
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-3 pt-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider">
          Available Signs
        </h3>
        <span className="badge badge-primary">
          {entries.length}
        </span>
      </div>
      
      {/* Grid */}
      <motion.div 
        className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2 max-h-[180px] overflow-y-auto pr-1"
        initial="hidden"
        animate="visible"
        variants={{
          hidden: { opacity: 0 },
          visible: {
            opacity: 1,
            transition: { staggerChildren: 0.03 }
          }
        }}
      >
        <AnimatePresence>
          {entries.map((entry) => (
            <motion.div
              key={entry.video_id}
              variants={{
                hidden: { opacity: 0, y: 10 },
                visible: { opacity: 1, y: 0 }
              }}
            >
              <GlossCard
                gloss={entry.gloss.toUpperCase()}
                word={entry.gloss}
                isSelected={selectedGloss === entry.glosses || selectedGloss === entry.gloss.toUpperCase()}
                onClick={() => onSelectGloss(entry)}
              />
            </motion.div>
          ))}
        </AnimatePresence>
      </motion.div>
    </div>
  )
}
