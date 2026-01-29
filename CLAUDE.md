# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DuoSign Frontend is a Deaf-first visual translation web application that converts typed text into sign language using 2D skeletal animation and 3D VRM avatars. Sign language is the primary output—all feedback is visual with zero sound dependency.

## Commands

```bash
npm run dev      # Start development server (localhost:3000)
npm run build    # Build for production
npm run start    # Start production server
npm run lint     # Run ESLint
```

## Architecture

This codebase follows a strict **MVC (Model-View-Controller) pattern**:

```
src/
├── app/            # Next.js App Router pages
├── models/         # Data layer & business logic
├── controllers/    # Orchestration layer (state management)
├── views/          # React components (presentation)
├── hooks/          # React bridge (useAppState)
├── components/     # DEPRECATED - use views/
└── lib/            # DEPRECATED - use models/
```

### State Management Pattern

- **Controllers** manage state directly (not Redux/Context)
- **Models** handle data persistence and business logic
- `useAppState()` hook bridges controllers with React components
- State changes propagate through controller callbacks

### Controller Hierarchy

- `AppController` - Main orchestrator, coordinates all sub-controllers
- `HistoryController` - History CRUD operations (localStorage, max 50 items)
- `PlaybackController` - Animation playback state (speeds: 0.5x, 0.75x, 1x)
- `TranslationController` - Translation API handling

### Application States

```typescript
type AppState = "HERO" | "PROCESSING" | "READY" | "ERROR" | "OFFLINE"
```

## Key Patterns

- **Server Components** are default; use `'use client'` directive for interactive views
- **Path aliases**: Use `@/` prefix for imports (e.g., `@/models`, `@/views/app`)
- **Styling**: Tailwind CSS inline classes + Framer Motion for animations
- **Rendering modes**: Avatar (3D VRM with Kalidokit) or Skeleton (2D Canvas)
- **API endpoint**: `/api/pose` (POST) for pose data

## Technology Stack

- Next.js 16.1.4 with React 19.2.3
- TypeScript (strict mode)
- Three.js + @pixiv/three-vrm + Kalidokit for 3D avatar rendering
- Framer Motion for animations
- Tailwind CSS v4

## Important Considerations

- This is an **accessibility-first** application for Deaf users—visual feedback is critical
- New feature components go in `src/views/app/`, reusable UI in `src/views/ui/`
- Controllers handle orchestration—avoid prop drilling
- The codebase was recently restructured from component-based to MVC architecture
