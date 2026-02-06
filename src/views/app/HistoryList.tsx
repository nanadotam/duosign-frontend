'use client'

import type { HistoryItem } from '@/models'
import { Play } from 'lucide-react'

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
      className={`w-full px-4 py-3 rounded-[var(--radius-lg)] text-left transition-all ${
        isSelected
          ? 'bg-[var(--color-primary)]/10 ring-1 ring-[var(--color-primary)]/20'
          : 'bg-[var(--panel-content-bg)] hover:bg-[var(--panel-content-bg)]/80'
      }`}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-[var(--color-text-primary)] truncate">
            {truncatedText}
          </p>
          <p className="text-xs text-[var(--color-mid-gray)] mt-0.5">
            {formattedTime}
          </p>
        </div>
        <div className="w-8 h-8 rounded-full bg-[var(--color-text-primary)] flex items-center justify-center shrink-0">
          <Play className="h-3.5 w-3.5 text-[var(--panel-content-bg)] ml-0.5" fill="currentColor" />
        </div>
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
      <div className="text-center py-8 text-[var(--color-mid-gray)] text-sm">
        No history yet. Type something to get started.
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">
          History
        </h3>
        {onClearHistory && (
          <button
            onClick={onClearHistory}
            className="text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors"
          >
            Clear All
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
