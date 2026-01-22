'use client'

import Link from 'next/link'
import { Button } from '@/components/ui/button'

export function Header() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-neutral-100">
      <div className="max-w-7xl mx-auto px-6 h-[72px] flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2">
          <span className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent">
            DuoSign
          </span>
        </Link>

        {/* Navigation */}
        <nav className="hidden md:flex items-center gap-8">
          <Link 
            href="/about" 
            className="text-neutral-600 hover:text-neutral-900 transition-colors text-sm font-medium"
          >
            About
          </Link>
          <Link 
            href="/accessibility" 
            className="text-neutral-600 hover:text-neutral-900 transition-colors text-sm font-medium"
          >
            Accessibility
          </Link>
          <Link 
            href="/contact" 
            className="text-neutral-600 hover:text-neutral-900 transition-colors text-sm font-medium"
          >
            Contact
          </Link>
        </nav>

        {/* CTA */}
        <Link href="/app">
          <Button size="default">
            Try DuoSign
          </Button>
        </Link>
      </div>
    </header>
  )
}
