'use client'

import type { HistoryItem } from '@/models'
import { Play, Inbox } from 'lucide-react'
import { motion } from 'framer-motion'

interface HistoryItemProps {
  item: HistoryItem
  isSelected?: boolean
  onClick: () => void
  index: number
}

function HistoryItemComponent({ item, isSelected, onClick, index }: HistoryItemProps) {
  const truncatedText = item.text.length > 40
    ? item.text.slice(0, 40) + '...'
    : item.text

  const formattedTime = new Date(item.timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit'
  })

  return (
    <motion.button
      onClick={onClick}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.05, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
      className={`w-full history-item ${isSelected ? 'history-item-selected' : ''}`}
    >
      <div className="flex-1 min-w-0 text-left">
        <p className="text-sm font-medium text-[var(--color-text-primary)] truncate leading-snug">
          {truncatedText}
        </p>
        <p className="text-xs text-[var(--color-text-tertiary)] mt-1">
          {formattedTime}
        </p>
      </div>
      
      {/* Play button */}
      <div className={`w-9 h-9 rounded-full flex items-center justify-center shrink-0 transition-all duration-200 ${
        isSelected 
          ? 'bg-[var(--color-primary)] shadow-[var(--shadow-primary)]' 
          : 'bg-[var(--color-text-primary)]'
      }`}>
        <Play 
          className={`h-3.5 w-3.5 ml-0.5 ${
            isSelected ? 'text-white' : 'text-[var(--panel-bg)]'
          }`} 
          fill="currentColor" 
        />
      </div>
    </motion.button>
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
      <motion.div 
        className="flex flex-col items-center justify-center py-12 text-center"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.4 }}
      >
        <div className="w-14 h-14 rounded-2xl bg-[var(--panel-content-bg)] 
                      flex items-center justify-center mb-4
                      border border-[var(--panel-border)]">
          <Inbox className="h-6 w-6 text-[var(--color-text-tertiary)]" />
        </div>
        <p className="text-sm font-medium text-[var(--color-text-secondary)] mb-1">
          No history yet
        </p>
        <p className="text-xs text-[var(--color-text-tertiary)]">
          Type something to get started
        </p>
      </motion.div>
    )
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-4 pt-4">
        <h3 className="text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider">
          Recent
        </h3>
        {onClearHistory && (
          <motion.button
            onClick={onClearHistory}
            className="text-xs font-medium text-[var(--color-text-tertiary)] 
                     hover:text-[var(--color-error)] transition-colors duration-200"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            Clear All
          </motion.button>
        )}
      </div>

      {/* List */}
      <div className="space-y-2 max-h-[320px] overflow-y-auto pr-1">
        {items.slice(0, 10).map((item, index) => (
          <HistoryItemComponent
            key={item.id}
            item={item}
            isSelected={selectedItem?.id === item.id}
            onClick={() => onSelectItem(item)}
            index={index}
          />
        ))}
      </div>

      {/* Show more indicator */}
      {items.length > 10 && (
        <motion.p 
          className="text-xs text-center text-[var(--color-text-tertiary)] mt-4 py-2"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          +{items.length - 10} more items
        </motion.p>
      )}
    </div>
  )
}
