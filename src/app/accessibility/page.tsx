import { AppShell } from '@/components/layout/AppShell'

export default function AccessibilityPage() {
  return (
    <AppShell>
      <div className="max-w-4xl mx-auto px-6 py-16">
        <h1 className="text-4xl md:text-5xl font-bold text-neutral-900 mb-8 font-serif">
          Accessibility Statement
        </h1>

        <div className="prose prose-lg max-w-none text-neutral-600">
          <section className="mb-12">
            <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Our Commitment</h2>
            <p className="leading-relaxed mb-4">
              DuoSign is committed to ensuring digital accessibility for people with disabilities. 
              We are continually improving the user experience for everyone and applying the relevant 
              accessibility standards.
            </p>
          </section>

          <section className="mb-12">
            <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Deaf-First Design Principles</h2>
            <ul className="space-y-3">
              <li>• <strong>No sound dependency:</strong> All functionality works without audio</li>
              <li>• <strong>Visual feedback:</strong> Every interaction provides clear visual feedback</li>
              <li>• <strong>Predictable motion:</strong> Animations are consistent and controllable</li>
              <li>• <strong>Playback control:</strong> Users can adjust speed, pause, and replay</li>
            </ul>
          </section>

          <section className="mb-12">
            <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Accessibility Features</h2>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-neutral-50 rounded-xl p-6">
                <h3 className="font-semibold text-neutral-900 mb-3">Vision</h3>
                <ul className="text-sm space-y-2">
                  <li>• High contrast color schemes</li>
                  <li>• Resizable text support</li>
                  <li>• Clear visual hierarchy</li>
                  <li>• Focus indicators on all interactive elements</li>
                </ul>
              </div>
              <div className="bg-neutral-50 rounded-xl p-6">
                <h3 className="font-semibold text-neutral-900 mb-3">Motor</h3>
                <ul className="text-sm space-y-2">
                  <li>• Full keyboard navigation</li>
                  <li>• Large touch targets (minimum 44px)</li>
                  <li>• Skip navigation links</li>
                  <li>• No time-sensitive interactions</li>
                </ul>
              </div>
              <div className="bg-neutral-50 rounded-xl p-6">
                <h3 className="font-semibold text-neutral-900 mb-3">Cognitive</h3>
                <ul className="text-sm space-y-2">
                  <li>• Clear, consistent navigation</li>
                  <li>• Simple, predictable interactions</li>
                  <li>• Progress indicators</li>
                  <li>• Error prevention and recovery</li>
                </ul>
              </div>
              <div className="bg-neutral-50 rounded-xl p-6">
                <h3 className="font-semibold text-neutral-900 mb-3">Vestibular</h3>
                <ul className="text-sm space-y-2">
                  <li>• Reduced motion support</li>
                  <li>• No auto-playing animations</li>
                  <li>• Smooth, subtle transitions</li>
                  <li>• User-controlled playback</li>
                </ul>
              </div>
            </div>
          </section>

          <section className="mb-12">
            <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Standards Compliance</h2>
            <p className="leading-relaxed mb-4">
              We aim to conform to the Web Content Accessibility Guidelines (WCAG) 2.1 Level AA. 
              These guidelines explain how to make web content more accessible for people with disabilities.
            </p>
          </section>

          <section className="mb-12">
            <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Feedback</h2>
            <p className="leading-relaxed mb-4">
              We welcome your feedback on the accessibility of DuoSign. If you encounter any 
              accessibility barriers or have suggestions for improvement, please contact us.
            </p>
          </section>
        </div>
      </div>
    </AppShell>
  )
}
