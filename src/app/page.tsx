'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { AppShell } from '@/components/layout/AppShell'
import { TextInput } from '@/components/app/TextInput'
import { Button } from '@/components/ui/button'
import { Spotlight } from '@/components/ui/spotlight'
import { useRouter } from 'next/navigation'
import { ArrowRight, Users, Eye, Zap } from 'lucide-react'

export default function LandingPage() {
  const router = useRouter()

  const handleSubmit = (text: string) => {
    // Store the text and navigate to app
    sessionStorage.setItem('duosign_initial_text', text)
    router.push('/app')
  }

  return (
    <AppShell>
      {/* Hero Section */}
      <section className="relative min-h-[calc(100vh-72px)] flex items-center justify-center overflow-hidden">
        <Spotlight className="-top-40 left-0 md:left-60 md:-top-20" fill="rgba(26, 115, 232, 0.15)" />
        
        {/* Background avatar silhouette */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <motion.div
            className="w-[400px] h-[500px] opacity-[0.05]"
            animate={{
              y: [0, -10, 0],
              scale: [1, 1.01, 1]
            }}
            transition={{
              duration: 6,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          >
            <svg viewBox="0 0 200 300" fill="currentColor" className="w-full h-full text-neutral-900">
              <ellipse cx="100" cy="60" rx="45" ry="55" />
              <path d="M35 140 Q35 100 100 100 Q165 100 165 140 L165 280 Q165 300 145 300 L55 300 Q35 300 35 280 Z" />
              <ellipse cx="55" cy="180" rx="25" ry="15" />
              <ellipse cx="145" cy="180" rx="25" ry="15" />
            </svg>
          </motion.div>
        </div>

        {/* Hero content */}
        <div className="relative z-10 max-w-2xl mx-auto px-6 text-center">
          <motion.h1 
            className="text-4xl md:text-5xl lg:text-6xl font-bold text-neutral-900 mb-6 font-serif"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            Visual-first sign language translation
          </motion.h1>
          
          <motion.p 
            className="text-lg md:text-xl text-neutral-600 mb-10"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            Type text, see it signed. DuoSign makes communication accessible for everyone.
          </motion.p>

          <motion.div
            className="max-w-xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <TextInput onSubmit={handleSubmit} placeholder="What do you want to sign?" />
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 bg-white">
        <div className="max-w-6xl mx-auto px-6">
          <motion.div 
            className="text-center mb-16"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-4 font-serif">
              Designed for accessibility
            </h2>
            <p className="text-lg text-neutral-600 max-w-2xl mx-auto">
              Built from the ground up with Deaf users in mind. Visual clarity, predictable motion, and complete control.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: Eye,
                title: "Visual-first design",
                description: "No sound dependency. Every interaction is designed for visual clarity and feedback."
              },
              {
                icon: Users,
                title: "For everyone",
                description: "Whether you're Deaf, hearing, or learning sign language, DuoSign bridges the communication gap."
              },
              {
                icon: Zap,
                title: "Instant translation",
                description: "Type your message and see it translated to sign language in real-time with our 2D avatar."
              }
            ].map((feature, index) => (
              <motion.div
                key={feature.title}
                className="bg-neutral-50 rounded-2xl p-6 text-center"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <div className="w-14 h-14 rounded-full bg-blue-100 flex items-center justify-center mx-auto mb-4">
                  <feature.icon className="w-7 h-7 text-blue-600" />
                </div>
                <h3 className="text-lg font-semibold text-neutral-900 mb-2">
                  {feature.title}
                </h3>
                <p className="text-neutral-600 text-sm">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* About Section */}
      <section className="py-24 bg-neutral-50" id="about">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
            >
              <h2 className="text-3xl md:text-4xl font-bold text-neutral-900 mb-6 font-serif">
                About DuoSign
              </h2>
              <p className="text-neutral-600 mb-4">
                DuoSign is a visual-first translation workspace that converts typed text into sign language using a 2D skeletal signing animation. It&apos;s designed to feel like a professional interpreter console, not a chat app or social platform.
              </p>
              <p className="text-neutral-600 mb-6">
                Most digital translation tools prioritize audio and text, leaving Deaf users with interfaces that feel secondary or adapted. DuoSign makes sign output the primary object of the interface.
              </p>
              <Link href="/app">
                <Button size="lg">
                  Try DuoSign
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </motion.div>
            
            <motion.div
              className="bg-white rounded-2xl p-8 shadow-lg"
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
            >
              <div className="aspect-square bg-gradient-to-br from-blue-50 to-purple-50 rounded-xl flex items-center justify-center">
                <svg viewBox="0 0 200 200" fill="none" className="w-32 h-32 text-neutral-300">
                  <ellipse cx="100" cy="50" rx="35" ry="40" fill="currentColor" />
                  <path d="M40 100 Q40 80 100 80 Q160 80 160 100 L160 180 Q160 195 145 195 L55 195 Q40 195 40 180 Z" fill="currentColor" />
                </svg>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Mission Section */}
      <section className="py-24 bg-gradient-to-b from-blue-600 to-blue-800 text-white">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <motion.h2 
            className="text-3xl md:text-4xl font-bold mb-6 font-serif"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            Our Mission
          </motion.h2>
          <motion.p 
            className="text-xl text-blue-100 leading-relaxed"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
          >
            To make communication accessible for everyone by putting sign language first. We believe that technology should adapt to users, not the other way around.
          </motion.p>
        </div>
      </section>
    </AppShell>
  )
}
