'use client'

import { useState, useEffect, useMemo, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, X, Hand, Loader2, Grid3X3 } from 'lucide-react'

interface SignEntry {
  gloss: string
  video_id?: string
}

interface SignSearchModalProps {
  isOpen: boolean
  onClose: () => void
  onSelectSign: (gloss: string) => void
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export function SignSearchModal({ isOpen, onClose, onSelectSign }: SignSearchModalProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [allSigns, setAllSigns] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Fetch all available signs on mount
  useEffect(() => {
    async function loadSigns() {
      try {
        setLoading(true)
        const response = await fetch('/api/glosses')
        if (!response.ok) throw new Error(`API error: ${response.status}`)
        
        const data = await response.json()
        setAllSigns(data.glosses || [])
        setError(null)
      } catch (err) {
        console.error('Failed to load signs:', err)
        setError('Could not load available signs')
      } finally {
        setLoading(false)
      }
    }
    
    if (isOpen) {
      loadSigns()
    }
  }, [isOpen])

  // Focus search input when modal opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [isOpen])

  // Filter signs by search query
  const filteredSigns = useMemo(() => {
    if (!searchQuery.trim()) {
      return allSigns.slice(0, 100) // Show first 100 if no search
    }
    const query = searchQuery.toUpperCase()
    return allSigns
      .filter(s => s.toUpperCase().includes(query))
      .slice(0, 100)
  }, [allSigns, searchQuery])

  // Handle sign selection
  const handleSelect = useCallback((gloss: string) => {
    onSelectSign(gloss)
    onClose()
    setSearchQuery('')
  }, [onSelectSign, onClose])

  // Close on escape key
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    if (isOpen) {
      document.addEventListener('keydown', handleEsc)
      return () => document.removeEventListener('keydown', handleEsc)
    }
  }, [isOpen, onClose])

  if (!isOpen) return null

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          
          {/* Modal */}
          <motion.div
            className="fixed inset-4 md:inset-auto md:left-1/2 md:top-1/2 md:-translate-x-1/2 md:-translate-y-1/2
                       md:w-[600px] md:max-h-[80vh] bg-[var(--panel-bg)] rounded-2xl shadow-2xl
                       flex flex-col overflow-hidden z-50 border border-[var(--panel-border)]"
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
          >
            {/* Header */}
            <div className="flex items-center gap-3 p-4 border-b border-[var(--panel-border)]">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[var(--color-text-tertiary)]" />
                <input
                  ref={inputRef}
                  type="text"
                  placeholder="Search 2000+ signs..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2.5 bg-[var(--panel-content-bg)] border border-[var(--panel-border)]
                           rounded-xl text-sm text-[var(--color-text-primary)] placeholder-[var(--color-text-tertiary)]
                           focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]
                           transition-all duration-200"
                />
              </div>
              <button
                onClick={onClose}
                className="p-2 rounded-lg text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)]
                         hover:bg-[var(--panel-content-bg)] transition-colors"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4">
              {loading ? (
                <div className="flex flex-col items-center justify-center py-12 gap-3">
                  <Loader2 className="h-8 w-8 text-[var(--color-primary)] animate-spin" />
                  <p className="text-sm text-[var(--color-text-secondary)]">Loading signs...</p>
                </div>
              ) : error ? (
                <div className="flex flex-col items-center justify-center py-12 gap-3 text-center">
                  <div className="w-12 h-12 rounded-xl bg-[var(--color-error-subtle)] flex items-center justify-center">
                    <X className="h-5 w-5 text-[var(--color-error)]" />
                  </div>
                  <p className="text-sm text-[var(--color-error)]">{error}</p>
                </div>
              ) : filteredSigns.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 gap-3 text-center">
                  <div className="w-12 h-12 rounded-xl bg-[var(--panel-content-bg)] flex items-center justify-center">
                    <Grid3X3 className="h-5 w-5 text-[var(--color-text-tertiary)]" />
                  </div>
                  <p className="text-sm text-[var(--color-text-secondary)]">
                    No signs match "{searchQuery}"
                  </p>
                </div>
              ) : (
                <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
                  {filteredSigns.map((gloss) => (
                    <motion.button
                      key={gloss}
                      onClick={() => handleSelect(gloss)}
                      className="group px-3 py-2.5 rounded-xl bg-[var(--panel-content-bg)] border border-[var(--panel-border)]
                               hover:border-[var(--color-primary)] hover:bg-[var(--color-primary)]/5
                               transition-all duration-200 text-left"
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <div className="flex items-center gap-2">
                        <Hand className="h-3.5 w-3.5 text-[var(--color-primary)] opacity-0 group-hover:opacity-100 transition-opacity" />
                        <span className="text-xs font-mono font-semibold text-[var(--color-text-primary)] uppercase truncate">
                          {gloss}
                        </span>
                      </div>
                    </motion.button>
                  ))}
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-[var(--panel-border)] bg-[var(--panel-content-bg)]">
              <div className="flex items-center justify-between text-xs text-[var(--color-text-tertiary)]">
                <span>
                  {loading ? 'Loading...' : `${allSigns.length} signs available`}
                </span>
                <span className="flex items-center gap-1">
                  <kbd className="px-1.5 py-0.5 rounded bg-[var(--panel-bg)] border border-[var(--panel-border)] font-mono text-[10px]">
                    ESC
                  </kbd>
                  <span>to close</span>
                </span>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
