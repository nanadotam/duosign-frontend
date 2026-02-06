'use client'

import Link from 'next/link'
import Image from 'next/image'
import { Moon, Sun } from 'lucide-react'
import { motion } from 'framer-motion'

interface HeaderProps {
  theme?: 'light' | 'dark'
  toggleTheme?: () => void
}

export function Header({ theme = 'light', toggleTheme }: HeaderProps) {
  return (
    <motion.header 
      className="h-[var(--header-height)] flex items-center justify-between px-6 md:px-8 max-w-[var(--max-content-width)] mx-auto w-full"
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
    >
      {/* Logo */}
      <Link 
        href="/" 
        className="flex items-center gap-3 group"
      >
        <div className="relative">
          <Image
            src="/duosign-hands-logo.svg"
            alt=""
            width={36}
            height={26}
            className={`transition-transform duration-300 group-hover:scale-110 ${
              theme === 'dark' ? 'invert-0 brightness-200' : 'invert brightness-0'
            }`}
          />
        </div>
        <span className="text-xl font-[family-name:var(--font-brand)] text-[var(--color-text-primary)] tracking-tight">
          DuoSign
        </span>
      </Link>

      {/* Right side controls */}
      <div className="flex items-center gap-3">
        {/* Theme Toggle */}
        {toggleTheme && (
          <motion.button
            onClick={toggleTheme}
            className="flex items-center gap-2.5 px-4 py-2 rounded-full 
                       bg-[var(--panel-content-bg)] border border-[var(--panel-border)]
                       hover:border-[var(--color-gray-300)] dark:hover:border-[var(--color-gray-600)]
                       shadow-[var(--shadow-sm)] hover:shadow-[var(--shadow-md)]
                       transition-all duration-200"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
          >
            <motion.div
              initial={false}
              animate={{ rotate: theme === 'light' ? 0 : 180 }}
              transition={{ duration: 0.3 }}
            >
              {theme === 'light' ? (
                <Moon className="h-4 w-4 text-[var(--color-text-secondary)]" />
              ) : (
                <Sun className="h-4 w-4 text-[var(--color-accent)]" />
              )}
            </motion.div>
            <span className="text-sm font-medium text-[var(--color-text-primary)]">
              {theme === 'light' ? 'Dark' : 'Light'}
            </span>
          </motion.button>
        )}
      </div>
    </motion.header>
  )
}
