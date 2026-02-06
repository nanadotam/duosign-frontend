'use client'

import { useState, KeyboardEvent } from 'react'
import { ArrowRight } from 'lucide-react'

interface TextInputProps {
  onSubmit: (text: string) => void
  placeholder?: string
  compact?: boolean
  disabled?: boolean
}

export function TextInput({
  onSubmit,
  placeholder = "Type English text to convert into sign language...",
  compact = false,
  disabled = false
}: TextInputProps) {
  const [value, setValue] = useState('')

  const handleSubmit = () => {
    if (value.trim() && !disabled) {
      onSubmit(value.trim())
      setValue('')
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSubmit()
    }
  }

  return (
    <div className="relative w-full">
      <input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        maxLength={500}
        className={`w-full bg-[var(--panel-content-bg)] text-[var(--color-text-primary)] placeholder:text-[var(--color-mid-gray)] border-2 border-transparent focus:border-[var(--color-primary-light)] focus:outline-none rounded-[var(--radius-md)] pr-14 transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
          compact ? 'h-12 px-4 text-base' : 'h-14 px-4 text-base'
        }`}
        aria-label="Text to translate"
      />
      <button
        onClick={handleSubmit}
        disabled={!value.trim() || disabled}
        className="absolute right-2 top-1/2 -translate-y-1/2 w-10 h-10 rounded-full bg-[var(--color-primary)] text-white flex items-center justify-center hover:bg-[var(--color-primary-dark)] disabled:opacity-30 disabled:cursor-not-allowed transition-all hover:scale-105 active:scale-95"
        aria-label="Send translation"
      >
        <ArrowRight className="h-5 w-5" />
      </button>
    </div>
  )
}
