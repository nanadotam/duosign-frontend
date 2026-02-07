import { NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import path from 'path'

/**
 * GET /api/glosses
 * Returns list of all available glosses by scanning JSON files in lexicon folder
 */
export async function GET() {
  try {
    const lexiconPath = path.join(
      process.cwd(),
      'public',
      'lexicon',
      'ase'
    )

    // Read directory and filter for JSON files
    const files = await fs.readdir(lexiconPath)
    const jsonFiles = files.filter(f => f.endsWith('.json'))
    
    // Extract gloss names (remove .json extension)
    const glosses = jsonFiles.map(f => f.replace('.json', ''))
    
    // Sort alphabetically
    glosses.sort((a, b) => a.localeCompare(b))

    return NextResponse.json({
      count: glosses.length,
      glosses: glosses
    })

  } catch (error) {
    console.error('Error loading glosses:', error)
    return NextResponse.json(
      { error: 'Failed to load glosses list' },
      { status: 500 }
    )
  }
}
