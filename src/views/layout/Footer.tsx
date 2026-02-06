import Link from 'next/link'

export function Footer() {
  return (
    <footer className="border-t border-[var(--panel-border)] bg-[var(--background)] py-8">
      <div className="max-w-[var(--max-content-width)] mx-auto px-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
            <span className="font-semibold text-[var(--color-text-primary)] font-[family-name:var(--font-brand)]">DuoSign</span>
            <span>v1.0</span>
          </div>

          <nav className="flex items-center gap-6 text-sm text-[var(--color-text-secondary)]">
            <Link
              href="/accessibility"
              className="hover:text-[var(--color-text-primary)] transition-colors"
            >
              Accessibility
            </Link>
            <Link
              href="/contact"
              className="hover:text-[var(--color-text-primary)] transition-colors"
            >
              Contact
            </Link>
            <Link
              href="/about"
              className="hover:text-[var(--color-text-primary)] transition-colors"
            >
              About
            </Link>
          </nav>

          <p className="text-sm text-[var(--color-mid-gray)]">
            &copy; 2026 DuoSign. Deaf-first design.
          </p>
        </div>
      </div>
    </footer>
  )
}
