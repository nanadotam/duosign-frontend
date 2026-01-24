'use client'

import { useState, useEffect } from 'react'
import { GlossCard } from './GlossCard'
import { motion } from 'framer-motion'

/**
 * Gloss entry from lexicon
 */
export interface GlossEntry {
  path: string
  spoken_language: string
  signed_language: string
  start: number
  end: number
  words: string
  glosses: string
  priority: number
}

interface GlossPickerProps {
  onSelectGloss: (entry: GlossEntry) => void
  selectedGloss?: string | null
}

/**
 * Grid of clickable gloss cards
 * Loads entries from lexicon index.csv
 */
export function GlossPicker({ onSelectGloss, selectedGloss }: GlossPickerProps) {
  const [entries, setEntries] = useState<GlossEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Load lexicon on mount
  useEffect(() => {
    async function loadLexicon() {
      try {
        const response = await fetch('/lexicon/ase/index.csv')
        if (!response.ok) throw new Error('Failed to load lexicon')
        
        const text = await response.text()
        const lines = text.trim().split('\n')
        const headers = lines[0].split(',')
        
        const parsed: GlossEntry[] = lines.slice(1).map(line => {
          const values = line.split(',')
          return {
            path: values[0],
            spoken_language: values[1],
            signed_language: values[2],
            start: parseInt(values[3]) || 0,
            end: parseInt(values[4]) || 0,
            words: values[5],
            glosses: values[6],
            priority: parseInt(values[7]) || 0,
          }
        })
        
        setEntries(parsed)
        setLoading(false)
      } catch (err) {
        setError('Could not load sign lexicon')
        setLoading(false)
      }
    }
    
    loadLexicon()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-4">
        <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center py-4 text-red-500 text-sm">
        {error}
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
            key={entry.glosses}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <GlossCard
              gloss={entry.glosses}
              word={entry.words}
              isSelected={selectedGloss === entry.glosses}
              onClick={() => onSelectGloss(entry)}
            />
          </motion.div>
        ))}
      </motion.div>
    </div>
  )
}
