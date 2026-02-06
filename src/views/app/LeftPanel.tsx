'use client'

import { HistoryList } from './HistoryList'
import { GlossPicker, type GlossEntry } from '@/components/app/GlossPicker'
import { TextToGlossInput } from '@/components/app/TextToGlossInput'
import type { HistoryItem } from '@/models'
import { Star, Clock } from 'lucide-react'
import { motion } from 'framer-motion'

interface LeftPanelProps {
  history: HistoryItem[]
  selectedItem: HistoryItem | null
  onSelectItem: (item: HistoryItem) => void
  onClearHistory: () => void
  onSelectGloss?: (entry: GlossEntry) => void
  selectedGloss?: string | null
}

export function LeftPanel({
  history,
  selectedItem,
  onSelectItem,
  onClearHistory,
  onSelectGloss,
  selectedGloss
}: LeftPanelProps) {
  return (
    <motion.div 
      className="flex flex-col h-full panel p-[var(--panel-padding)] overflow-hidden"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
    >
      {/* Text Input Section */}
      <div className="mb-6">
        <TextToGlossInput
          onGlossSelect={(gloss, videoId) => {
            if (videoId && onSelectGloss) {
              onSelectGloss({ 
                gloss: gloss.toLowerCase(), 
                video_id: videoId, 
                frame_count: 0, 
                duration_sec: 0, 
                file_size_kb: 0 
              })
            }
          }}
        />
      </div>

      {/* Gloss Picker Section */}
      <div className="mb-6">
        <div className="divider" />
        <GlossPicker
          onSelectGloss={(entry) => onSelectGloss?.(entry)}
          selectedGloss={selectedGloss}
        />
      </div>

      {/* History Section */}
      <div className="flex-1 overflow-y-auto min-h-0">
        <div className="divider" />
        <HistoryList
          items={history}
          selectedItem={selectedItem}
          onSelectItem={onSelectItem}
          onClearHistory={onClearHistory}
        />
      </div>

      {/* Footer Actions */}
      <div className="pt-4 mt-4 border-t border-[var(--panel-border)]">
        <div className="flex items-center justify-center gap-4">
          <motion.button 
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium
                       text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]
                       hover:bg-[var(--panel-content-bg)] rounded-[var(--radius-md)]
                       transition-all duration-200"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <Star className="h-4 w-4" />
            <span>Favorites</span>
          </motion.button>
          <motion.button 
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium
                       text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]
                       hover:bg-[var(--panel-content-bg)] rounded-[var(--radius-md)]
                       transition-all duration-200"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <Clock className="h-4 w-4" />
            <span>History</span>
          </motion.button>
        </div>
      </div>
    </motion.div>
  )
}
