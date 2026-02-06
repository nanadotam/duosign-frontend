'use client'

import Link from 'next/link'
import Image from 'next/image'
import { Moon, Sun } from 'lucide-react'

interface HeaderProps {
  theme?: 'light' | 'dark'
  toggleTheme?: () => void
}

export function Header({ theme = 'light', toggleTheme }: HeaderProps) {
  return (
    <header className="h-[var(--header-height)] flex items-center justify-between px-8 max-w-[var(--max-content-width)] mx-auto w-full">
      {/* Logo */}
      <Link href="/" className="flex items-center gap-3 group">
        <Image
          src="/duosign-hands-logo.svg"
          alt=""
          width={40}
          height={29}
          className={theme === 'dark' ? 'invert-0 brightness-200' : 'invert brightness-0'}
        />
        <span className="text-2xl font-[family-name:var(--font-brand)] text-[var(--color-text-primary)]">
          DuoSign
        </span>
      </Link>

      {/* Dark Mode Toggle */}
      {toggleTheme && (
        <button
          onClick={toggleTheme}
          className="flex items-center gap-2 px-4 py-2 rounded-[var(--radius-full)] hover:bg-[var(--color-light-gray)] transition-colors"
          aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
        >
          {theme === 'light' ? (
            <Moon className="h-4 w-4 text-[var(--color-text-secondary)]" />
          ) : (
            <Sun className="h-4 w-4 text-[var(--color-text-secondary)]" />
          )}
          <span className="text-sm font-[family-name:var(--font-brand)] italic text-[var(--color-text-primary)]">
            {theme === 'light' ? 'Dark Mode' : 'Light Mode'}
          </span>
        </button>
      )}
    </header>
  )
}
