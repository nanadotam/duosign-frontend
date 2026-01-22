'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const processingMessages = [
  "Getting your sign…",
  "Matching lexicon…",
  "Preparing animation…"
]

interface StatusTextProps {
  isProcessing: boolean
}

export function StatusText({ isProcessing }: StatusTextProps) {
  const [messageIndex, setMessageIndex] = useState(0)

  useEffect(() => {
    if (!isProcessing) {
      setMessageIndex(0)
      return
    }

    const interval = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % processingMessages.length)
    }, 800)

    return () => clearInterval(interval)
  }, [isProcessing])

  if (!isProcessing) return null

  return (
    <div className="flex items-center justify-center py-4">
      <AnimatePresence mode="wait">
        <motion.div
          key={messageIndex}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2 }}
          className="flex items-center gap-3"
        >
          {/* Loader dots */}
          <div className="flex gap-1">
            {[0, 1, 2].map((i) => (
              <motion.div
                key={i}
                className="w-2 h-2 rounded-full bg-blue-500"
                animate={{
                  scale: [1, 1.2, 1],
                  opacity: [0.5, 1, 0.5]
                }}
                transition={{
                  duration: 0.6,
                  repeat: Infinity,
                  delay: i * 0.15
                }}
              />
            ))}
          </div>
          <span className="text-neutral-600 text-sm font-medium">
            {processingMessages[messageIndex]}
          </span>
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
