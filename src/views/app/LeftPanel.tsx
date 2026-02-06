'use client'

import { HistoryList } from './HistoryList'
import { GlossPicker, type GlossEntry } from '@/components/app/GlossPicker'
import { TextToGlossInput } from '@/components/app/TextToGlossInput'
import type { HistoryItem } from '@/models'
import { ArrowUpRight } from 'lucide-react'

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
    <div className="flex flex-col h-full bg-[var(--panel-bg)] rounded-[var(--radius-xl)] p-[var(--panel-padding)] overflow-hidden transition-shadow hover:shadow-[var(--shadow-panel)]">
      {/* Text Input with send button */}
      <div className="mb-4">
        <TextToGlossInput
          onGlossSelect={(gloss, videoId) => {
            if (videoId && onSelectGloss) {
              onSelectGloss({ gloss: gloss.toLowerCase(), video_id: videoId, frame_count: 0, duration_sec: 0, file_size_kb: 0 })
            }
          }}
        />
      </div>

      {/* Gloss Picker */}
      <div className="mb-4 pb-4 border-b border-[var(--panel-border)]">
        <GlossPicker
          onSelectGloss={(entry) => onSelectGloss?.(entry)}
          selectedGloss={selectedGloss}
        />
      </div>

      {/* History section */}
      <div className="flex-1 overflow-y-auto">
        <HistoryList
          items={history}
          selectedItem={selectedItem}
          onSelectItem={onSelectItem}
          onClearHistory={onClearHistory}
        />
      </div>

      {/* Footer links */}
      <div className="flex items-center justify-center gap-6 pt-4 mt-4 border-t border-[var(--panel-border)]">
        <button className="flex items-center gap-1 text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors">
          <ArrowUpRight className="h-3.5 w-3.5" />
          View Favorites
        </button>
        <button className="flex items-center gap-1 text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors">
          <ArrowUpRight className="h-3.5 w-3.5" />
          View History
        </button>
      </div>
    </div>
  )
}
