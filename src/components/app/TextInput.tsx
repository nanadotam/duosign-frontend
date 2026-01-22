'use client'

import { useState, KeyboardEvent } from 'react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { X } from 'lucide-react'

interface TextInputProps {
  onSubmit: (text: string) => void
  placeholder?: string
  compact?: boolean
}

export function TextInput({ 
  onSubmit, 
  placeholder = "What do you want to sign?",
  compact = false 
}: TextInputProps) {
  const [value, setValue] = useState('')

  const handleSubmit = () => {
    if (value.trim()) {
      onSubmit(value.trim())
      setValue('')
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSubmit()
    }
  }

  const handleClear = () => {
    setValue('')
  }

  return (
    <div className="relative w-full">
      <Input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        className={compact ? "h-12 text-base" : "h-16 text-lg"}
        showCharCount={!compact}
      />
      {value && (
        <Button
          variant="ghost"
          size="icon"
          onClick={handleClear}
          className="absolute right-14 top-1/2 -translate-y-1/2 h-8 w-8 rounded-full"
          aria-label="Clear input"
        >
          <X className="h-4 w-4" />
        </Button>
      )}
      {!compact && (
        <p className="mt-2 text-center text-sm text-neutral-400">
          Press Enter to translate into sign language
        </p>
      )}
    </div>
  )
}
