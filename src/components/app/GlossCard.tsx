'use client'

import { motion } from 'framer-motion'

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
        relative px-3 py-2 rounded-xl text-left transition-all
        border-2 
        ${isSelected 
          ? 'bg-blue-50 border-blue-400 shadow-md' 
          : 'bg-white border-neutral-200 hover:border-neutral-300 hover:shadow-sm'
        }
      `}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <div className="text-xs font-mono text-neutral-400 uppercase tracking-wide">
        {gloss}
      </div>
      <div className={`text-sm font-medium ${isSelected ? 'text-blue-700' : 'text-neutral-700'}`}>
        {word}
      </div>
      
      {/* Selected indicator */}
      {isSelected && (
        <motion.div
          className="absolute -top-1 -right-1 w-3 h-3 bg-blue-500 rounded-full"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
        />
      )}
    </motion.button>
  )
}
