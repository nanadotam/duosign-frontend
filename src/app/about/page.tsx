import { AppShell } from '@/views/layout/AppShell'
import { Button } from '@/views/ui/button'
import Link from 'next/link'
import { ArrowRight } from 'lucide-react'

export default function AboutPage() {
  return (
    <AppShell>
      <div className="max-w-4xl mx-auto px-6 py-16">
        <h1 className="text-4xl md:text-5xl font-bold text-neutral-900 mb-8 font-serif">
          About DuoSign
        </h1>

        <div className="prose prose-lg max-w-none">
          <section className="mb-12">
            <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Our Vision</h2>
            <p className="text-neutral-600 leading-relaxed mb-4">
              DuoSign is a visual-first translation workspace that converts typed text into sign language 
              using a 2D skeletal signing animation. It is designed to feel like a professional interpreter 
              console, not a chat app or social platform.
            </p>
          </section>

          <section className="mb-12">
            <h2 className="text-2xl font-semibold text-neutral-900 mb-4">The Problem We Solve</h2>
            <p className="text-neutral-600 leading-relaxed mb-4">
              Most digital translation tools prioritize audio and text, leaving Deaf users with interfaces 
              that feel secondary or adapted.
            </p>
            <p className="text-neutral-600 leading-relaxed mb-4">
              Deaf users require visual clarity, predictable motion, control over playback, and clear system 
              feedback without sound. Hearing users require a simple input mechanism, confidence that the 
              system is working, and immediate visible output.
            </p>
            <p className="text-neutral-600 leading-relaxed">
              DuoSign solves this by making sign output the primary object of the interface.
            </p>
          </section>

          <section className="mb-12">
            <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Who We Serve</h2>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-neutral-50 rounded-xl p-6">
                <h3 className="font-semibold text-neutral-900 mb-2">Primary Users</h3>
                <ul className="text-neutral-600 space-y-2 text-sm">
                  <li>• Deaf users receiving information in sign language</li>
                  <li>• Hearing users communicating with Deaf users via sign output</li>
                </ul>
              </div>
              <div className="bg-neutral-50 rounded-xl p-6">
                <h3 className="font-semibold text-neutral-900 mb-2">Secondary Users</h3>
                <ul className="text-neutral-600 space-y-2 text-sm">
                  <li>• Interpreters and interpreter trainees</li>
                  <li>• Educators and accessibility-focused institutions</li>
                </ul>
              </div>
            </div>
          </section>

          <section className="mb-12">
            <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Key Features</h2>
            <ul className="text-neutral-600 space-y-3">
              <li>✓ Text to sign language translation</li>
              <li>✓ 2D skeletal animation player</li>
              <li>✓ Interpreter-grade playback controls (play, pause, speed)</li>
              <li>✓ Session-based conversation history</li>
              <li>✓ High-contrast, reduced-motion compliant design</li>
              <li>✓ Fully keyboard accessible</li>
            </ul>
          </section>

          <div className="text-center pt-8">
            <Link href="/app">
              <Button size="lg">
                Try DuoSign Now
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </AppShell>
  )
}
