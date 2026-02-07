'use client'

import Link from 'next/link'
import Image from 'next/image'
import { motion } from 'framer-motion'
import { useRouter } from 'next/navigation'

export default function LandingPage() {
  const router = useRouter()

  return (
    <div className="min-h-screen bg-[#1E5EDF] relative overflow-hidden flex flex-col">
      {/* Navigation */}
      <nav className="flex items-center justify-end gap-8 px-10 pt-7 relative z-10">
        {['API', 'DOCUMENTATION', 'FAQ', 'CONTACT'].map((link) => (
          <Link
            key={link}
            href={`/${link.toLowerCase()}`}
            className="text-[13px] font-medium text-white/70 hover:text-white transition-colors tracking-wider"
          >
            {link}
          </Link>
        ))}
      </nav>

      {/* Centered Brand Section */}
      <div className="flex-1 flex flex-col items-center justify-center relative z-10 px-6">
        {/* Logo + Brand */}
        <motion.div
          className="flex flex-col items-center"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: 'easeOut' }}
        >
          <Image
            src="/duosign-hands-logo.svg"
            alt="DuoSign signing hands"
            width={139}
            height={101}
            className="mb-2"
            priority
          />
          <h1 className="text-[64px] leading-[1.1] font-[family-name:var(--font-brand)] text-white">
            DuoSign
          </h1>
        </motion.div>

        {/* Hero Text */}
        <motion.div
          className="max-w-[800px] text-center mt-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <p className="text-[48px] leading-[1.2] font-[family-name:var(--font-brand)] text-white">
            Translate text into accurate sign language animations.
          </p>
        </motion.div>

        <motion.p
          className="text-[18px] text-white/80 mt-6 max-w-[600px] text-center leading-relaxed"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.35 }}
        >
          Built for accessibility, education, and everyday communication.
        </motion.p>

        {/* CTA Button */}
        <motion.div
          className="mt-12 flex flex-col items-center gap-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
        >
          <button
            onClick={() => router.push('/app')}
            className="bg-[#FF6B35] hover:bg-[#E55A2B] active:bg-[#CC4F24] text-white font-semibold text-sm uppercase tracking-wider px-10 py-4 rounded-[var(--radius-md)] hover:-translate-y-[2px] active:translate-y-0 transition-all shadow-lg hover:shadow-xl"
          >
            GO TO APP
          </button>
          <Link
            href="/api"
            className="text-sm font-medium text-white/70 hover:text-white transition-colors"
          >
            View API Documentation &rarr;
          </Link>
        </motion.div>
      </div>

      {/* Trust Strip */}
      <motion.div
        className="flex items-center justify-center gap-12 pb-16 relative z-10"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.7 }}
      >
        {[
          'Accessibility-first design',
          'AI-powered sign synthesis',
          'Production-ready platform'
        ].map((text) => (
          <span
            key={text}
            className="text-[13px] font-medium text-white/60"
          >
            {text}
          </span>
        ))}
      </motion.div>
    </div>
  )
}
