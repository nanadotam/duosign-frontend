'use client'

import type { HistoryItem } from '@/lib/types'
import { Play } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface HistoryItemProps {
  item: HistoryItem
  isSelected?: boolean
  onClick: () => void
}

function HistoryItemComponent({ item, isSelected, onClick }: HistoryItemProps) {
  const truncatedText = item.text.length > 50 
    ? item.text.slice(0, 50) + '...' 
    : item.text

  const formattedTime = new Date(item.timestamp).toLocaleTimeString([], { 
    hour: '2-digit', 
    minute: '2-digit' 
  })

  return (
    <button
      onClick={onClick}
      className={`w-full p-4 rounded-xl text-left transition-all hover:bg-neutral-100 ${
        isSelected ? 'bg-blue-50 border border-blue-200' : 'bg-white border border-neutral-100'
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-neutral-900 truncate">
            {truncatedText}
          </p>
          <p className="text-xs text-neutral-400 mt-1">
            {formattedTime}
          </p>
        </div>
        <Button 
          variant="ghost" 
          size="icon" 
          className="h-8 w-8 rounded-full shrink-0"
          aria-label="Play this translation"
        >
          <Play className="h-4 w-4" />
        </Button>
      </div>
    </button>
  )
}

interface HistoryListProps {
  items: HistoryItem[]
  selectedItem?: HistoryItem | null
  onSelectItem: (item: HistoryItem) => void
  onClearHistory?: () => void
}

export function HistoryList({ items, selectedItem, onSelectItem, onClearHistory }: HistoryListProps) {
  if (items.length === 0) {
    return (
      <div className="text-center py-8 text-neutral-400 text-sm">
        No history yet. Type something to get started.
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
          Recent
        </h3>
        {onClearHistory && (
          <button 
            onClick={onClearHistory}
            className="text-xs text-neutral-400 hover:text-neutral-600 transition-colors"
          >
            Clear all
          </button>
        )}
      </div>
      <div className="space-y-2 max-h-[400px] overflow-y-auto">
        {items.map((item) => (
          <HistoryItemComponent
            key={item.id}
            item={item}
            isSelected={selectedItem?.id === item.id}
            onClick={() => onSelectItem(item)}
          />
        ))}
      </div>
    </div>
  )
}
