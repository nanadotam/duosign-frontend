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
    const isWarning = charCount > 450
    const isError = charCount >= maxChars

    return (
      <div className="relative w-full">
        <input
          type={type}
          className={cn(
            "flex h-14 w-full rounded-[var(--radius-md)] border-2 border-transparent bg-[var(--panel-content-bg)] px-4 py-3 text-base text-[var(--color-text-primary)] placeholder:text-[var(--color-mid-gray)] focus:border-[var(--color-primary-light)] focus:outline-none disabled:cursor-not-allowed disabled:opacity-50 transition-colors",
            className
          )}
          ref={ref}
          value={value}
          maxLength={maxChars}
          {...props}
        />
        {showCharCount && (
          <span className={cn(
            "absolute right-4 top-1/2 -translate-y-1/2 text-[13px] font-medium",
            isError ? "text-[var(--color-error)]" : isWarning ? "text-[var(--color-warning)]" : "text-[var(--color-mid-gray)]"
          )}>
            {charCount} / {maxChars}
          </span>
        )}
      </div>
    )
  }
)
Input.displayName = "Input"

export { Input }
