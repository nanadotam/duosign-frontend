import { NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import path from 'path'

/**
 * GET /api/pose?gloss=hello
 * Returns pose landmark data in JSON format for direct frontend rendering
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
    // Build path to JSON file
    const jsonPath = path.join(
      process.cwd(),
      'public',
      'lexicon',
      'ase',
      `${gloss.toLowerCase()}.json`
    )

    // Check if file exists
    try {
      await fs.access(jsonPath)
    } catch {
      return NextResponse.json(
        { error: `Pose data not found for gloss: ${gloss}` },
        { status: 404 }
      )
    }

    // Read and parse JSON file
    const fileContent = await fs.readFile(jsonPath, 'utf-8')
    const poseData = JSON.parse(fileContent)
    
    // Add gloss name to response
    return NextResponse.json({
      gloss: gloss.toUpperCase(),
      ...poseData
    })

  } catch (error) {
    console.error('Error loading pose:', error)
    return NextResponse.json(
      { error: 'Failed to load pose data' },
      { status: 500 }
    )
  }
}
