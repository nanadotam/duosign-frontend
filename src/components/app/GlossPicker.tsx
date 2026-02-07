'use client'

import { useState, useEffect } from 'react'
import { GlossCard } from './GlossCard'
import { SignSearchModal } from './SignSearchModal'
import { motion, AnimatePresence } from 'framer-motion'
import { AlertTriangle, Search, Terminal, Hand } from 'lucide-react'

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

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

/**
 * Compact sign picker with button to open full search modal
 */
export function GlossPicker({ onSelectGloss, selectedGloss }: GlossPickerProps) {
  const [recentSigns, setRecentSigns] = useState<string[]>([])
  const [totalCount, setTotalCount] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showModal, setShowModal] = useState(false)

  // Fetch available signs count and some recent ones
  useEffect(() => {
    async function loadSigns() {
      try {
        const response = await fetch('/api/glosses')
        if (!response.ok) throw new Error(`API error: ${response.status}`)
        
        const data = await response.json()
        const glosses: string[] = data.glosses || []
        
        setTotalCount(glosses.length)
        // Show first 8 as "recent" for quick access
        setRecentSigns(glosses.slice(0, 8))
        setLoading(false)
      } catch (err) {
        console.error('Failed to load signs:', err)
        setError('Could not load available signs')
        setLoading(false)
      }
    }
    
    loadSigns()
  }, [])

  // Handle sign selection from modal
  const handleSelectSign = (gloss: string) => {
    onSelectGloss({
      gloss: gloss.toLowerCase(),
      video_id: gloss.toLowerCase(),
      glosses: gloss.toUpperCase(),
      words: gloss.toLowerCase(),
      frame_count: 0,
      duration_sec: 0,
      file_size_kb: 0
    })
  }

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

  return (
    <>
      <div className="space-y-3 pt-4">
        {/* Header with search button */}
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider">
            Available Signs
          </h3>
          <motion.button
            onClick={() => setShowModal(true)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium
                     bg-[var(--color-primary)] text-white shadow-[var(--shadow-primary)]
                     hover:opacity-90 transition-opacity"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <Search className="h-3.5 w-3.5" />
            <span>Browse {totalCount.toLocaleString()}</span>
          </motion.button>
        </div>
        
        {/* Quick access grid (first few signs) */}
        {recentSigns.length > 0 && (
          <motion.div 
            className="grid grid-cols-4 gap-2"
            initial="hidden"
            animate="visible"
            variants={{
              hidden: { opacity: 0 },
              visible: {
                opacity: 1,
                transition: { staggerChildren: 0.02 }
              }
            }}
          >
            <AnimatePresence>
              {recentSigns.map((gloss) => (
                <motion.div
                  key={gloss}
                  variants={{
                    hidden: { opacity: 0, y: 10 },
                    visible: { opacity: 1, y: 0 }
                  }}
                >
                  <GlossCard
                    gloss={gloss.toUpperCase()}
                    word={gloss}
                    isSelected={selectedGloss === gloss.toUpperCase() || selectedGloss === gloss}
                    onClick={() => handleSelectSign(gloss)}
                  />
                </motion.div>
              ))}
            </AnimatePresence>
          </motion.div>
        )}

        {/* Hint text */}
        <p className="text-xs text-[var(--color-text-tertiary)] text-center pt-2">
          <span className="inline-flex items-center gap-1">
            <Hand className="h-3 w-3" />
            Click "Browse" to search all {totalCount.toLocaleString()} signs
          </span>
        </p>
      </div>

      {/* Search Modal */}
      <SignSearchModal
        isOpen={showModal}
        onClose={() => setShowModal(false)}
        onSelectSign={handleSelectSign}
      />
    </>
  )
}
