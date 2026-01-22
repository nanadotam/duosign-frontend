'use client'

import { SplineScene } from '@/components/ui/splite'
import { motion } from 'framer-motion'

interface OutputPlayerProps {
  isReady: boolean
  isPlaying?: boolean
}

export function OutputPlayer({ isReady, isPlaying }: OutputPlayerProps) {
  return (
    <div className="relative w-full aspect-square max-h-[400px] rounded-2xl overflow-hidden bg-gradient-to-br from-neutral-100 to-neutral-200">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50/50 via-white to-purple-50/30" />
      
      {/* 3D Avatar Scene */}
      {isReady ? (
        <motion.div 
          className="absolute inset-0"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <SplineScene 
            scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode"
            className="w-full h-full"
          />
        </motion.div>
      ) : (
        <div className="absolute inset-0 flex items-center justify-center">
          {/* Placeholder avatar silhouette */}
          <motion.div
            className="w-48 h-64 rounded-full bg-gradient-to-b from-neutral-200 to-neutral-300 opacity-20"
            animate={{
              scale: [1, 1.02, 1],
              opacity: [0.15, 0.25, 0.15]
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
          <div className="absolute inset-0 flex items-center justify-center">
            <svg 
              className="w-32 h-48 text-neutral-300" 
              viewBox="0 0 100 150"
              fill="currentColor"
            >
              {/* Simple avatar silhouette */}
              <ellipse cx="50" cy="30" rx="20" ry="25" />
              <path d="M20 70 Q20 55 50 55 Q80 55 80 70 L80 140 Q80 150 70 150 L30 150 Q20 150 20 140 Z" />
            </svg>
          </div>
        </div>
      )}

      {/* Playing indicator */}
      {isPlaying && (
        <div className="absolute top-4 right-4">
          <div className="flex items-center gap-2 bg-white/90 backdrop-blur-sm rounded-full px-3 py-1.5 shadow-sm">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-xs font-medium text-neutral-600">Playing</span>
          </div>
        </div>
      )}
    </div>
  )
}
