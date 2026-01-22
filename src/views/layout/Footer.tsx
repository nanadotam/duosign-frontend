import Link from 'next/link'

export function Footer() {
  return (
    <footer className="border-t border-neutral-100 bg-neutral-50/50 py-8">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-sm text-neutral-500">
            <span className="font-semibold text-neutral-700">DuoSign</span>
            <span>v1.0</span>
          </div>
          
          <nav className="flex items-center gap-6 text-sm text-neutral-500">
            <Link 
              href="/accessibility" 
              className="hover:text-neutral-900 transition-colors"
            >
              Accessibility
            </Link>
            <Link 
              href="/contact" 
              className="hover:text-neutral-900 transition-colors"
            >
              Contact
            </Link>
            <Link 
              href="/about" 
              className="hover:text-neutral-900 transition-colors"
            >
              About
            </Link>
          </nav>

          <p className="text-sm text-neutral-400">
            © 2026 DuoSign. Deaf-first design.
          </p>
        </div>
      </div>
    </footer>
  )
}
