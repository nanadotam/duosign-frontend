# DuoSign Frontend

[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-blue?logo=typescript)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-v4-38B2AC?logo=tailwind-css)](https://tailwindcss.com/)

A Deaf-first visual translation web application built with a focus on accessibility and real-time visual feedback.

![DuoSign Preview](./Screenshot%202026-01-22%20at%2014.02.08.png)

## 🌟 Overview

DuoSign is a visual translation platform designed specifically for the Deaf community. It utilizes a **3D Spline avatar** to provide sign language translations, following a strict **MVC (Model-View-Controller)** architecture to ensure scalability and maintainability.

## ✨ Key Features

-   **Visual-first Design**: Zero sound dependency; all feedback is visual and intuitive.
-   **State-Driven UI**: Smooth transitions between `HERO`, `PROCESSING`, and `READY` states.
-   **3D Avatar Integration**: Interactive Spline avatar with lazy loading for optimized performance.
-   **Playback Controls**: Granular control over translation speed (0.5x, 0.75x, 1x) and playback state.
-   **History Persistence**: LocalStorage integration with a 50-item limit for session continuity.
-   **Accessibility**: Full keyboard navigation, high-contrast focus states, and reduced motion support.
-   **Responsive Layout**: Mobile-first design with a sophisticated two-panel desktop workspace.

## 🛠️ Tech Stack

-   **Framework**: Next.js 15 (App Router)
-   **Language**: TypeScript
-   **Styling**: Tailwind CSS v4
-   **UI Components**: shadcn/ui (Radix UI)
-   **Animation/3D**: Spline, Framer Motion
-   **Architecture**: MVC (Model-View-Controller)

## 📂 Project Structure

```
src/
├── app/           # View: Pages and Routing
├── components/    # View: UI Components
│   ├── ui/        # shadcn primitives (Button, Input, etc.)
│   ├── layout/    # AppShell, Header, Footer
│   └── app/       # Feature-specific components (Panels, Controls)
├── hooks/         # Controller: State logic (useAppState)
├── lib/           # Model: Types, Utilities, and Storage logic
└── public/        # Assets and Static files
```

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ 
- npm / pnpm / yarn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/duosign-frontend.git
   cd duosign-frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## 🗺️ Sitemap

| Page | Path | Description |
|------|------|-------------|
| **Landing** | `/` | Hero section, features, and mission statement. |
| **App** | `/app` | The core translation workspace with history and 3D output. |
| **About** | `/about` | Deep dive into the product vision. |
| **Accessibility** | `/accessibility` | Documentation on Deaf-first design principles used. |
| **Contact** | `/contact` | Support and feedback form. |

---

Built with ❤️ for the Deaf community.
