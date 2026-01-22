'use client'

import * as React from "react"
import { cn } from "@/lib/utils"

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  showCharCount?: boolean
  maxChars?: number
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, showCharCount, maxChars = 500, value, ...props }, ref) => {
    const charCount = typeof value === 'string' ? value.length : 0
    
    return (
      <div className="relative w-full">
        <input
          type={type}
          className={cn(
            "flex h-16 w-full rounded-2xl border border-neutral-200 bg-white px-6 py-4 text-lg ring-offset-white file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-neutral-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 shadow-sm transition-shadow hover:shadow-md focus:shadow-lg",
            className
          )}
          ref={ref}
          value={value}
          maxLength={maxChars}
          {...props}
        />
        {showCharCount && (
          <span className="absolute right-4 top-1/2 -translate-y-1/2 text-sm text-neutral-400">
            {charCount}/{maxChars}
          </span>
        )}
      </div>
    )
  }
)
Input.displayName = "Input"

export { Input }
