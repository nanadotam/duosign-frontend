'use client'

import { TextInput } from './TextInput'
import { HistoryList } from './HistoryList'
import { GlossPicker, type GlossEntry } from '@/components/app/GlossPicker'
import type { HistoryItem } from '@/models'

interface LeftPanelProps {
  onSubmit: (text: string) => void
  history: HistoryItem[]
  selectedItem: HistoryItem | null
  onSelectItem: (item: HistoryItem) => void
  onClearHistory: () => void
  onSelectGloss?: (entry: GlossEntry) => void
  selectedGloss?: string | null
}

export function LeftPanel({ 
  onSubmit, 
  history, 
  selectedItem, 
  onSelectItem, 
  onClearHistory,
  onSelectGloss,
  selectedGloss
}: LeftPanelProps) {
  return (
    <div className="flex flex-col h-full bg-white rounded-2xl shadow-sm border border-neutral-100 overflow-hidden">
      {/* Input section */}
      <div className="p-4 border-b border-neutral-100">
        <TextInput onSubmit={onSubmit} compact />
      </div>

      {/* Gloss Picker section */}
      <div className="p-4 border-b border-neutral-100 bg-neutral-50/50">
        <GlossPicker 
          onSelectGloss={(entry) => onSelectGloss?.(entry)}
          selectedGloss={selectedGloss}
        />
      </div>

      {/* History section */}
      <div className="flex-1 overflow-y-auto p-4">
        <HistoryList
          items={history}
          selectedItem={selectedItem}
          onSelectItem={onSelectItem}
          onClearHistory={onClearHistory}
        />
      </div>
    </div>
  )
}
