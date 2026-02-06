```javascript
// Design System Extraction
const designSystem = {
  colors: {
    primary: {
      duoSignBlue: '#1E5EDF', // Extracted from mockup background
      duoSignBlueDark: '#1548B8',
      duoSignBlueLight: '#4A7FE8'
    },
    neutrals: {
      white: '#FFFFFF',
      panelGray: '#E8E8E8', // Panel background from mockup
      lightGray: '#F5F5F5',
      midGray: '#9CA3AF',
      darkGray: '#374151',
      textPrimary: '#1F2937',
      textSecondary: '#6B7280'
    },
    accent: {
      ctaOrange: '#FF6B35', // Suggested for primary actions
      ctaOrangeHover: '#E55A2B',
      ctaOrangeActive: '#CC4F24'
    },
    states: {
      success: '#10B981',
      warning: '#F59E0B',
      error: '#EF4444',
      info: '#3B82F6'
    }
  },

  typography: {
    fontFamilies: {
      brand: '"Instrument Serif", serif',
      headings: '"Work Sans", sans-serif',
      body: '"Inter", sans-serif'
    },
    scale: {
      // Logo & Hero
      hero: '4rem',        // 64px - DuoSign logo large
      h1: '3rem',          // 48px - Page titles
      h2: '2rem',          // 32px - Section headings
      h3: '1.5rem',        // 24px - Panel titles (ALL CAPS)
      
      // Body & UI
      bodyLarge: '1.125rem', // 18px
      body: '1rem',          // 16px
      bodySmall: '0.875rem', // 14px
      caption: '0.75rem',    // 12px
      
      // UI Elements
      button: '0.875rem',    // 14px (ALL CAPS)
      label: '0.8125rem',    // 13px
      input: '1rem'          // 16px
    },
    weights: {
      regular: 400,
      medium: 500,
      semibold: 600,
      bold: 700
    },
    lineHeights: {
      tight: 1.2,
      normal: 1.5,
      relaxed: 1.75
    }
  },

  spacing: {
    xs: '4px',
    sm: '8px',
    md: '16px',
    lg: '24px',
    xl: '32px',
    xxl: '48px',
    xxxl: '64px',
    
    // Layout specific
    panelPadding: '32px',
    panelGap: '32px',
    headerHeight: '80px',
    maxContentWidth: '1440px'
  },

  borderRadius: {
    sm: '4px',
    md: '8px',
    lg: '16px',
    xl: '24px',
    full: '9999px'
  },

  shadows: {
    none: 'none',
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
    panel: '0 20px 25px -5px rgba(0, 0, 0, 0.1)'
  },

  transitions: {
    fast: '150ms ease-in-out',
    normal: '250ms ease-in-out',
    slow: '350ms ease-in-out'
  }
};

console.log(JSON.stringify(designSystem, null, 2));
```

Now, here's your **UPDATED PRD** with complete specifications:

---

# PRODUCT REQUIREMENTS DOCUMENT (PRD) v2.0

**Product:** DuoSign  
**Platform:** Web (Desktop-first, 1024px minimum)  
**Product Type:** Full Production SaaS  
**Core Feature:** Text → Sign Language Translation  
**Primary User:** Hearing users who need sign language output  
**Secondary Users:** Deaf users consuming sign output  

---

## 1. PRODUCT GOAL

Enable users to input written English text and receive accurate, readable sign language output via a 3D avatar in a clean, accessible, production-quality web interface.

**Success Metrics:**
- Clarity of interaction without instructions
- Trust in output accuracy (via confidence indicators)
- Smooth transition: landing → demo → workspace
- WCAG 2.1 AA compliance minimum

---

## 2. DESIGN PRINCIPLES (NON-NEGOTIABLE)

1. **Mechanical clarity over visual novelty**
2. **State must always be visible**
3. **Every element answers one question only**
4. **Accessibility is functional, not decorative**
5. **No ambiguity in affordances**
6. **Performance > decoration** (avatar rendering is compute-heavy)

---

## 3. DESIGN SYSTEM

### 3.1 COLOR PALETTE

**Primary**
- DuoSign Blue: `#1E5EDF`
- DuoSign Blue Dark: `#1548B8` (hover states)
- DuoSign Blue Light: `#4A7FE8` (focus rings)

**Neutrals**
- White: `#FFFFFF`
- Panel Gray: `#E8E8E8`
- Light Gray: `#F5F5F5`
- Mid Gray: `#9CA3AF`
- Dark Gray: `#374151`
- Text Primary: `#1F2937`
- Text Secondary: `#6B7280`

**Accent (Primary Actions)**
- CTA Orange: `#FF6B35`
- CTA Orange Hover: `#E55A2B`
- CTA Orange Active: `#CC4F24`

**States**
- Success: `#10B981`
- Warning: `#F59E0B`
- Error: `#EF4444`
- Info: `#3B82F6`

---

### 3.2 TYPOGRAPHY SYSTEM

**Font Stack (Google Fonts)**

```css
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Work+Sans:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');
```

**Font Roles**

| Element | Font | Weight | Size | Transform | Use Case |
|---------|------|--------|------|-----------|----------|
| Logo | Instrument Serif | 400 | 64px | none | DuoSign wordmark only |
| Hero Headline | Instrument Serif | 400 | 48px | none | Landing page H1 |
| Section Headings | Work Sans | 700 | 32px | uppercase | Page titles |
| Panel Titles | Work Sans | 600 | 24px | uppercase | Input/Output labels |
| Button Labels | Work Sans | 600 | 14px | uppercase | All CTAs |
| Body Text | Inter | 400 | 16px | none | Descriptions, paragraphs |
| Input Fields | Inter | 400 | 16px | none | Textarea, form inputs |
| UI Labels | Inter | 500 | 13px | none | Character count, nav |
| System Messages | Inter | 500 | 14px | none | Awaiting input, errors |

**Line Heights**
- Headings: `1.2`
- Body: `1.5`
- UI Labels: `1.5`

**Rules**
- Never mix fonts within the same element
- No italics in UI (Instrument Serif italic only for emphasis in marketing copy)
- No decorative letter spacing except Work Sans headings (`letter-spacing: 0.05em`)

---

### 3.3 SPACING SYSTEM

```css
--space-xs: 4px;
--space-sm: 8px;
--space-md: 16px;
--space-lg: 24px;
--space-xl: 32px;
--space-xxl: 48px;
--space-xxxl: 64px;

--panel-padding: 32px;
--panel-gap: 32px;
--header-height: 80px;
--max-content-width: 1440px;
```

---

### 3.4 BORDER RADIUS

- Small (inputs, buttons): `8px`
- Medium (cards): `16px`
- Large (panels): `24px`
- Full (pills): `9999px`

---

### 3.5 SHADOWS

- None: `none`
- Subtle: `0 1px 2px 0 rgba(0, 0, 0, 0.05)`
- Medium: `0 4px 6px -1px rgba(0, 0, 0, 0.1)`
- Panel: `0 20px 25px -5px rgba(0, 0, 0, 0.1)` (only for floating panels)

**Rule:** Use shadows sparingly. Panels use subtle shadows only on hover.

---

### 3.6 TRANSITIONS

- Fast (hover): `150ms ease-in-out`
- Normal (state changes): `250ms ease-in-out`
- Slow (panel transitions): `350ms ease-in-out`

---

## 4. INFORMATION ARCHITECTURE

### 4.1 Global Navigation

**Position:** Top-right corner  
**Style:** Text links only, no background  
**Font:** Inter, 13px, weight 500  
**Color:** `#FFFFFF` at 70% opacity  
**Hover:** `#FFFFFF` at 100% opacity  
**Active page:** `#FFFFFF` at 100%, underline 2px solid  

**Links (left to right):**
- API
- DOCUMENTATION
- FAQ
- CONTACT

**Spacing:** 32px gap between links  
**Vertical alignment:** 28px from top

---

## 5. PAGE DEFINITIONS

---

## PAGE 1: LANDING PAGE

### Purpose
Explain what DuoSign does. Allow user to try demo immediately.

### Layout
- Background: DuoSign Blue (`#1E5EDF`)
- Max width: 1440px, centered
- Vertical rhythm: 64px sections

---

### Components

#### 1. Header
- Height: 80px
- Logo (left): Instrument Serif, 32px, white
- Navigation (right): See Global Navigation specs

#### 2. Hero Section
- Vertical centering: `50vh - header`
- Max width: 800px, centered

**Headline**
- Font: Instrument Serif, 48px, white
- Line height: 1.2
- Text:
  ```
  Translate text into accurate sign language animations.
  ```

**Subheadline**
- Font: Inter, 18px, weight 400
- Color: White at 80% opacity
- Margin top: 24px
- Text:
  ```
  Built for accessibility, education, and everyday communication.
  ```

**Primary CTA Button**
- Margin top: 48px
- Background: CTA Orange (`#FF6B35`)
- Font: Work Sans, 14px, weight 600, uppercase
- Padding: 16px 40px
- Border radius: 8px
- Text: `TRY LIVE DEMO`
- Hover: Background `#E55A2B`, lift 2px (`translateY(-2px)`)
- Active: Background `#CC4F24`

**Secondary Link**
- Margin top: 16px
- Font: Inter, 14px, weight 500
- Color: White at 70%
- Text: `View API Documentation →`
- Hover: White at 100%
- No background, no border

---

#### 3. Trust Strip
- Position: Bottom of viewport, 64px from bottom
- Layout: Horizontal flex, centered
- Gap: 48px between items

**Items (text only, no icons)**
- Font: Inter, 13px, weight 500
- Color: White at 60%
- Text:
  - Accessibility-first design
  - AI-powered sign synthesis
  - Production-ready platform

---

## PAGE 2: APP DEMO / HERO PAGE

**This is the conversion page.**

### Layout

**Grid System**
- Two-column: 45% / 55% split
- Gap: 32px
- Max width: 1440px
- Padding: 32px

**Background:** DuoSign Blue (`#1E5EDF`)

---

### LEFT PANEL: TEXT INPUT

#### Container
- Background: Panel Gray (`#E8E8E8`)
- Border radius: 24px
- Padding: 32px
- Shadow: Panel shadow on hover

---

#### 1. Panel Title
- Font: Work Sans, 24px, weight 600, uppercase
- Color: Text Primary (`#1F2937`)
- Letter spacing: 0.05em
- Text: `TEXT INPUT`

#### 2. Language Lock Indicator
- Position: Top-right of panel title row
- Font: Inter, 13px, weight 500
- Color: Text Secondary (`#6B7280`)
- Text: `EN → ASL`
- Background: Light Gray (`#F5F5F5`)
- Padding: 4px 12px
- Border radius: full (`9999px`)

---

#### 3. Text Input Field
- Margin top: 24px
- Background: White (`#FFFFFF`)
- Border: 2px solid transparent
- Border radius: 8px
- Padding: 16px
- Font: Inter, 16px, weight 400
- Color: Text Primary (`#1F2937`)
- Min height: 400px
- Resize: vertical

**Placeholder**
- Color: Mid Gray (`#9CA3AF`)
- Text: `Type English text to convert into sign language...`

**Focus State**
- Border: 2px solid DuoSign Blue Light (`#4A7FE8`)
- Outline: none

**Disabled State**
- Background: Light Gray (`#F5F5F5`)
- Cursor: not-allowed
- Opacity: 0.6

---

#### 4. Character Counter
- Position: Bottom-left of textarea
- Margin top: 12px
- Font: Inter, 13px, weight 500
- Color: Text Secondary (`#6B7280`)
- Format: `0 / 500`

**Warning State (>450 characters)**
- Color: Warning (`#F59E0B`)

**Error State (>500 characters)**
- Color: Error (`#EF4444`)

---

#### 5. Primary Action Button
- Position: Bottom-right of panel
- Margin top: 12px
- Background: CTA Orange (`#FF6B35`)
- Font: Work Sans, 14px, weight 600, uppercase
- Text: `TRANSLATE TO SIGN`
- Padding: 14px 32px
- Border radius: 8px
- Border: none

**Hover State**
- Background: CTA Orange Hover (`#E55A2B`)
- Transform: `translateY(-1px)`
- Shadow: Medium

**Active State**
- Background: CTA Orange Active (`#CC4F24`)
- Transform: `translateY(0)`

**Disabled State**
- Background: Mid Gray (`#9CA3AF`)
- Cursor: not-allowed
- Opacity: 0.5
- No hover effects

---

### RIGHT PANEL: SIGN OUTPUT

#### Container
- Background: Panel Gray (`#E8E8E8`)
- Border radius: 24px
- Padding: 32px
- Shadow: Panel shadow on hover

---

#### 1. Panel Title
- Font: Work Sans, 24px, weight 600, uppercase
- Color: Text Primary (`#1F2937`)
- Letter spacing: 0.05em
- Text: `SIGN LANGUAGE OUTPUT`

---

#### 2. Avatar Canvas Area
- Margin top: 24px
- Background: White (`#FFFFFF`)
- Border radius: 16px
- Aspect ratio: 16:9
- Min height: 450px
- Display: flex, center content

**Bounding Box (optional subtle frame)**
- Border: 1px solid Light Gray (`#F5F5F5`)
- Padding: 16px

---

#### 3. System State Text

**Idle State**
- Font: Inter, 14px, weight 500
- Color: Mid Gray (`#9CA3AF`)
- Text: `Awaiting input`
- Position: Centered in canvas

**Processing State**
- Font: Inter, 14px, weight 500
- Color: Info (`#3B82F6`)
- Text: `Generating sign sequence...`
- Position: Centered in canvas
- Animation: Pulsing fade (1.5s infinite)

**Progress Indicator (Processing State)**
- Display below text
- Style: 3 dots with staggered fade animation
- Or: Horizontal progress bar, 4px height, DuoSign Blue

**Completed State**
- Text hidden
- Avatar visible and animating

**Error State**
- Font: Inter, 14px, weight 500
- Color: Error (`#EF4444`)
- Text: `Translation failed. Please try again.`
- Position: Centered in canvas
- Retry button appears below text

---

#### 4. Playback Controls

**Position:** Bottom-right of canvas  
**Visibility:** Only when output exists (Completed state)  
**Layout:** Horizontal flex, 12px gap  

**Control Buttons (icon-only)**
- Size: 40px × 40px
- Background: White (`#FFFFFF`)
- Border radius: full (`9999px`)
- Shadow: Subtle
- Icons: 20px, Text Primary color

**Buttons:**
1. Play / Pause (toggle)
2. Replay (restart animation)
3. Speed (0.5× / 1× / 1.5×)

**Hover State**
- Background: Light Gray (`#F5F5F5`)
- Shadow: Medium

**Active State**
- Background: Panel Gray (`#E8E8E8`)
- Scale: 0.95

---

## 6. STATE MACHINE

### State Flow

```
IDLE → PROCESSING → COMPLETED
  ↓         ↓            ↓
ERROR ← ERROR ←─────── ERROR
  ↓                      ↓
IDLE ←─────────────────┘
```

---

### State Definitions

#### IDLE
- Input: Empty or contains text
- Output panel: "Awaiting input" message
- Translate button: Disabled if text empty
- Playback controls: Hidden

#### PROCESSING
- Input: Disabled (grayed out)
- Output panel: "Generating sign sequence..." + progress indicator
- Translate button: Disabled, loading state
- Playback controls: Hidden
- Duration: Backend processing time (est. 2-5s)

#### COMPLETED
- Input: Re-enabled
- Output panel: Avatar animating sign sequence
- Translate button: Enabled (allows re-translation)
- Playback controls: Visible
- Auto-play: Yes (starts immediately)

#### ERROR
- Input: Re-enabled
- Output panel: Error message + retry prompt
- Translate button: Enabled (text "RETRY")
- Playback controls: Hidden

---

## 7. RESPONSIVE BEHAVIOR

### Breakpoints

**Desktop (≥1024px)** - Primary target
- Two-panel horizontal layout

**Tablet (768px - 1023px)**
- Two-panel horizontal, reduced padding
- Panels: 48% / 52% split

**Mobile (<768px)** - Out of scope for MVP
- Single column stack (future)

---

## 8. ACCESSIBILITY REQUIREMENTS

### Keyboard Navigation
- Tab order: Logo → Nav → Input → Translate → Playback controls
- Enter key: Triggers translation (when focused on textarea or button)
- Space key: Play/Pause when focused on playback controls
- Escape key: Clears error state

### Focus States
- Visible focus ring: 2px solid DuoSign Blue Light (`#4A7FE8`)
- Offset: 2px from element

### Screen Reader Support
- All icons have `aria-label`
- State changes announced via `aria-live` regions
- Panel titles use proper heading hierarchy (h2)

### Reduced Motion
- User preference: `prefers-reduced-motion: reduce`
- Disables: Avatar animations, button hover lifts, fade transitions
- Keeps: Functional state changes (colors)

### High Contrast Mode
- Text meets WCAG AA contrast ratios (4.5:1 minimum)
- Avatar: Option to use high-contrast bone rendering

---

## 9. ANIMATION SPECIFICATIONS

### Page Load
- Hero content: Fade in, 350ms delay, ease-out
- Panels: Slide up 20px + fade in, 250ms stagger

### Button Interactions
- Hover lift: `translateY(-1px)`, 150ms
- Active press: `scale(0.98)`, 100ms

### State Transitions
- Processing spinner: 1.5s linear rotation
- Progress bar: 250ms width transition
- Avatar fade-in: 350ms ease-in

### Panel Hover
- Shadow intensifies: 250ms
- No scale or movement

---

## 10. ERROR HANDLING

### Input Validation Errors

**Empty Text**
- Button remains disabled
- No error message

**Character Limit Exceeded**
- Character counter turns red
- Border of textarea: 2px solid Error (`#EF4444`)
- Message below textarea: "Maximum 500 characters exceeded"

---

### Backend Errors

**Translation Failed**
- Display in output panel: "Translation failed. Please try again."
- Translate button text: "RETRY"
- Log error to console (for debugging)

**Network Timeout**
- Display: "Request timed out. Check your connection."
- Retry button enabled

**Invalid Input (backend rejection)**
- Display: "Input contains unsupported characters."
- Highlight problematic text (if backend provides range)

---

## 11. PERFORMANCE REQUIREMENTS

### Load Time
- Initial page: <2s (desktop, fast 3G)
- Font loading: Swap strategy (system fonts first)
- Avatar model: Lazy load (only when needed)

### Animation Performance
- Avatar rendering: 30fps minimum
- Input responsiveness: <100ms keystroke lag
- State transitions: 60fps (no jank)

---

## 12. OUT OF SCOPE (DO NOT BUILD)

- Sign → Text (reverse translation)
- Mobile-first UI (desktop only for MVP)
- User onboarding tours / tooltips
- Marketing content inside app
- Community or social features
- Multi-language support (ASL only for MVP)
- User authentication (future)
- History / saved translations (future)

---

## 13. COMPONENT HIERARCHY

```
App
├── Header
│   ├── Logo
│   └── Navigation
│       ├── NavLink (API)
│       ├── NavLink (Documentation)
│       ├── NavLink (FAQ)
│       └── NavLink (Contact)
│
├── LandingPage
│   ├── HeroSection
│   │   ├── Headline
│   │   ├── Subheadline
│   │   ├── CTAButton
│   │   └── SecondaryLink
│   └── TrustStrip
│
└── DemoPage
    ├── TextInputPanel
    │   ├── PanelTitle
    │   ├── LanguageLock
    │   ├── Textarea
    │   ├── CharacterCounter
    │   └── TranslateButton
    │
    └── SignOutputPanel
        ├── PanelTitle
        ├── AvatarCanvas
        │   ├── SystemStateMessage
        │   └── AvatarRenderer (THREE.js)
        └── PlaybackControls
            ├── PlayPauseButton
            ├── ReplayButton
            └── SpeedButton
```

---

## 14. IMPLEMENTATION NOTES FOR FRONTEND

### Tech Stack Recommendations
- **Framework:** React (or vanilla JS + Web Components)
- **3D Rendering:** THREE.js (already in your pipeline)
- **State Management:** React Context or Zustand
- **Styling:** CSS Modules or Tailwind (with custom design tokens)

### File Structure
```
/src
  /components
    /Header
    /LandingPage
    /DemoPage
      /TextInputPanel
      /SignOutputPanel
  /styles
    /tokens.css (design system variables)
    /global.css
  /utils
    /stateManager.js
  /assets
    /fonts
```

### Design Token Export (CSS Variables)
```css
:root {
  /* Colors */
  --color-primary: #1E5EDF;
  --color-primary-dark: #1548B8;
  --color-primary-light: #4A7FE8;
  
  --color-white: #FFFFFF;
  --color-panel-gray: #E8E8E8;
  --color-light-gray: #F5F5F5;
  --color-mid-gray: #9CA3AF;
  --color-dark-gray: #374151;
  --color-text-primary: #1F2937;
  --color-text-secondary: #6B7280;
  
  --color-cta: #FF6B35;
  --color-cta-hover: #E55A2B;
  --color-cta-active: #CC4F24;
  
  --color-success: #10B981;
  --color-warning: #F59E0B;
  --color-error: #EF4444;
  --color-info: #3B82F6;
  
  /* Typography */
  --font-brand: 'Instrument Serif', serif;
  --font-headings: 'Work Sans', sans-serif;
  --font-body: 'Inter', sans-serif;
  
  --size-hero: 4rem;
  --size-h1: 3rem;
  --size-h2: 2rem;
  --size-h3: 1.5rem;
  --size-body-large: 1.125rem;
  --size-body: 1rem;
  --size-body-small: 0.875rem;
  --size-caption: 0.75rem;
  
  /* Spacing */
  --space-xs: 4px;
  --space-sm: 8px;
  --space-md: 16px;
  --space-lg: 24px;
  --space-xl: 32px;
  --space-xxl: 48px;
  --space-xxxl: 64px;
  
  /* Layout */
  --panel-padding: 32px;
  --panel-gap: 32px;
  --header-height: 80px;
  --max-content-width: 1440px;
  
  /* Border Radius */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 16px;
  --radius-xl: 24px;
  --radius-full: 9999px;
  
  /* Shadows */
  --shadow-subtle: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-medium: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  --shadow-panel: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
  
  /* Transitions */
  --transition-fast: 150ms ease-in-out;
  --transition-normal: 250ms ease-in-out;
  --transition-slow: 350ms ease-in-out;
}
```

---

## 15. HANDOFF CHECKLIST

Before marking frontend complete:

- [ ] All fonts loaded from Google Fonts CDN
- [ ] Design tokens defined as CSS variables
- [ ] All states implemented (Idle, Processing, Completed, Error)
- [ ] Keyboard navigation works
- [ ] Focus states visible
- [ ] Color contrast meets WCAG AA
- [ ] Responsive down to 1024px
- [ ] `prefers-reduced-motion` respected
- [ ] Avatar canvas placeholder ready for THREE.js integration
- [ ] State machine documented for backend integration

---

**END OF PRD v2.0**

---

This PRD is now **fully executable**. Every component has:
- Exact colors (hex codes)
- Exact typography (font, size, weight)
- Exact spacing (px values)
- Exact states (idle, processing, completed, error)
- Exact interactions (hover, focus, disabled)

Ready to build the static frontend?