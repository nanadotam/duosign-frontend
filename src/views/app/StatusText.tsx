'use client'

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const processingMessages = [
  "Getting your sign...",
  "Matching lexicon...",
  "Preparing animation..."
]

interface StatusTextProps {
  isProcessing: boolean
}

export function StatusText({ isProcessing }: StatusTextProps) {
  const [messageIndex, setMessageIndex] = useState(0)
  const [progress, setProgress] = useState([0, 0, 0])
  const messageIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const progressIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (!isProcessing) {
      // Clean up intervals without calling setState synchronously
      if (messageIntervalRef.current) clearInterval(messageIntervalRef.current)
      if (progressIntervalRef.current) clearInterval(progressIntervalRef.current)
      messageIntervalRef.current = null
      progressIntervalRef.current = null
      return
    }

    messageIntervalRef.current = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % processingMessages.length)
    }, 800)

    progressIntervalRef.current = setInterval(() => {
      setProgress(prev => prev.map((val, i) =>
        Math.min(val + (3 - i) * 2 + Math.random() * 5, 100)
      ))
    }, 150)

    return () => {
      if (messageIntervalRef.current) clearInterval(messageIntervalRef.current)
      if (progressIntervalRef.current) clearInterval(progressIntervalRef.current)
    }
  }, [isProcessing])

  // Reset state when not processing (derived from props, not in effect)
  const displayIndex = isProcessing ? messageIndex : 0
  const displayProgress = isProcessing ? progress : [0, 0, 0]

  if (!isProcessing) return null

  return (
    <div className="px-4 py-3">
      {/* Text status */}
      <AnimatePresence mode="wait">
        <motion.div
          key={displayIndex}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.2 }}
          className="flex items-center justify-center gap-3 mb-3"
        >
          <div className="flex gap-1">
            {[0, 1, 2].map((i) => (
              <motion.div
                key={i}
                className="w-1.5 h-1.5 rounded-full bg-[var(--color-info)]"
                animate={{
                  scale: [1, 1.3, 1],
                  opacity: [0.4, 1, 0.4]
                }}
                transition={{
                  duration: 0.6,
                  repeat: Infinity,
                  delay: i * 0.15
                }}
              />
            ))}
          </div>
          <span className="text-sm font-medium text-[var(--color-text-secondary)]">
            {processingMessages[displayIndex]}
          </span>
        </motion.div>
      </AnimatePresence>

      {/* Progress bars */}
      <div className="flex items-center justify-center gap-3">
        <motion.div
          className="h-5 rounded-[var(--radius-sm)] bg-[#FCA5A5]"
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(displayProgress[0], 100)}px` }}
          transition={{ ease: 'easeOut' }}
        />
        <motion.div
          className="h-5 rounded-[var(--radius-sm)] bg-[#86EFAC]"
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(displayProgress[1], 100)}px` }}
          transition={{ ease: 'easeOut' }}
        />
        <motion.div
          className="h-5 rounded-[var(--radius-sm)] bg-[#93C5FD]"
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(displayProgress[2], 100)}px` }}
          transition={{ ease: 'easeOut' }}
        />
      </div>
    </div>
  )
}
