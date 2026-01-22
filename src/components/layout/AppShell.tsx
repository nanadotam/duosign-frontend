'use client'

import { Header } from './Header'
import { Footer } from './Footer'

interface AppShellProps {
  children: React.ReactNode
  showHeader?: boolean
  showFooter?: boolean
}

export function AppShell({ children, showHeader = true, showFooter = true }: AppShellProps) {
  return (
    <div className="min-h-screen flex flex-col bg-neutral-50">
      {showHeader && <Header />}
      <main className="flex-1 pt-[72px]">
        {children}
      </main>
      {showFooter && <Footer />}
    </div>
  )
}
