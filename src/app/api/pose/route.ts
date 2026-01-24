import { NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import path from 'path'

/**
 * GET /api/pose?gloss=YES
 * Returns pose data for a given gloss
 */
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const gloss = searchParams.get('gloss')

  if (!gloss) {
    return NextResponse.json(
      { error: 'Missing gloss parameter' },
      { status: 400 }
    )
  }

  try {
    // Build path to pose file
    const posePath = path.join(
      process.cwd(),
      'public',
      'lexicon',
      'ase',
      `${gloss.toLowerCase()}.pose`
    )

    // Check if file exists
    try {
      await fs.access(posePath)
    } catch {
      return NextResponse.json(
        { error: `Pose file not found for gloss: ${gloss}` },
        { status: 404 }
      )
    }

    // Read the .pose file (it's a compressed npz file)
    // For browser compatibility, we'll read it as binary and parse on client
    // OR we can pre-convert to JSON format
    
    // For now, return file info and let client fetch directly
    const stats = await fs.stat(posePath)
    
    return NextResponse.json({
      gloss: gloss.toUpperCase(),
      path: `/lexicon/ase/${gloss.toLowerCase()}.pose`,
      size: stats.size,
      available: true
    })

  } catch (error) {
    console.error('Error loading pose:', error)
    return NextResponse.json(
      { error: 'Failed to load pose data' },
      { status: 500 }
    )
  }
}
