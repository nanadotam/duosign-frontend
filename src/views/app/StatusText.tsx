'use client'

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const processingMessages = [
  "Fetching sign data...",
  "Matching vocabulary...",
  "Preparing animation..."
]

interface StatusTextProps {
  isProcessing: boolean
}

export function StatusText({ isProcessing }: StatusTextProps) {
  const [messageIndex, setMessageIndex] = useState(0)
  const [progress, setProgress] = useState(0)
  const messageIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const progressIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (!isProcessing) {
      if (messageIntervalRef.current) clearInterval(messageIntervalRef.current)
      if (progressIntervalRef.current) clearInterval(progressIntervalRef.current)
      messageIntervalRef.current = null
      progressIntervalRef.current = null
      return
    }

    // Reset on start
    setProgress(0)
    setMessageIndex(0)

    messageIntervalRef.current = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % processingMessages.length)
    }, 1200)

    progressIntervalRef.current = setInterval(() => {
      setProgress(prev => Math.min(prev + Math.random() * 8, 95))
    }, 100)

    return () => {
      if (messageIntervalRef.current) clearInterval(messageIntervalRef.current)
      if (progressIntervalRef.current) clearInterval(progressIntervalRef.current)
    }
  }, [isProcessing])

  if (!isProcessing) return null

  return (
    <motion.div 
      className="px-6 py-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      {/* Message */}
      <div className="flex items-center justify-center gap-3 mb-4">
        {/* Animated dots */}
        <div className="flex gap-1">
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              className="w-1.5 h-1.5 rounded-full bg-[var(--color-primary)]"
              animate={{
                scale: [1, 1.4, 1],
                opacity: [0.3, 1, 0.3]
              }}
              transition={{
                duration: 0.8,
                repeat: Infinity,
                delay: i * 0.15,
                ease: "easeInOut"
              }}
            />
          ))}
        </div>
        
        <AnimatePresence mode="wait">
          <motion.span
            key={messageIndex}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="text-sm font-medium text-[var(--color-text-secondary)]"
          >
            {processingMessages[messageIndex]}
          </motion.span>
        </AnimatePresence>
      </div>

      {/* Single elegant progress bar */}
      <div className="w-full max-w-xs mx-auto">
        <div className="h-1.5 bg-[var(--panel-border)] rounded-full overflow-hidden">
          <motion.div
            className="h-full rounded-full bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-accent)]"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ ease: "easeOut", duration: 0.1 }}
          />
        </div>
      </div>
    </motion.div>
  )
}
