'use client'

import { motion } from 'framer-motion'
import { Check } from 'lucide-react'

interface GlossCardProps {
  gloss: string
  word: string
  isSelected?: boolean
  onClick?: () => void
}

/**
 * Clickable card displaying a sign gloss
 */
export function GlossCard({ gloss, word, isSelected, onClick }: GlossCardProps) {
  return (
    <motion.button
      onClick={onClick}
      className={`
        relative w-full px-3 py-2.5 rounded-[var(--radius-lg)] text-left
        border-2 transition-all duration-200
        ${isSelected 
          ? 'bg-[var(--color-primary-subtle)] border-[var(--color-primary)] shadow-[var(--shadow-primary)]' 
          : 'bg-[var(--panel-content-bg)] border-[var(--panel-border)] hover:border-[var(--color-gray-300)] dark:hover:border-[var(--color-gray-600)] hover:shadow-[var(--shadow-sm)]'
        }
      `}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      {/* Gloss label */}
      <div className="text-[10px] font-mono text-[var(--color-text-tertiary)] uppercase tracking-wider mb-0.5">
        {gloss}
      </div>
      
      {/* Word */}
      <div className={`text-sm font-medium truncate ${
        isSelected ? 'text-[var(--color-primary)]' : 'text-[var(--color-text-primary)]'
      }`}>
        {word}
      </div>
      
      {/* Selected indicator */}
      {isSelected && (
        <motion.div
          className="absolute -top-1.5 -right-1.5 w-5 h-5 bg-[var(--color-primary)] rounded-full 
                     flex items-center justify-center shadow-[var(--shadow-primary)]"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", stiffness: 500, damping: 30 }}
        >
          <Check className="w-3 h-3 text-white" strokeWidth={3} />
        </motion.div>
      )}
    </motion.button>
  )
}
