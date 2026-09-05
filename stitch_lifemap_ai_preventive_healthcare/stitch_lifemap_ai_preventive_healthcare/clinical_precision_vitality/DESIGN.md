---
name: Clinical Precision & Vitality
colors:
  surface: '#faf8ff'
  surface-dim: '#d2d9f4'
  surface-bright: '#faf8ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f2f3ff'
  surface-container: '#eaedff'
  surface-container-high: '#e2e7ff'
  surface-container-highest: '#dae2fd'
  on-surface: '#131b2e'
  on-surface-variant: '#3d4947'
  inverse-surface: '#283044'
  inverse-on-surface: '#eef0ff'
  outline: '#6d7a77'
  outline-variant: '#bcc9c6'
  surface-tint: '#006a61'
  primary: '#00685f'
  on-primary: '#ffffff'
  primary-container: '#008378'
  on-primary-container: '#f4fffc'
  inverse-primary: '#6bd8cb'
  secondary: '#006398'
  on-secondary: '#ffffff'
  secondary-container: '#5bb8fe'
  on-secondary-container: '#00476e'
  tertiary: '#006860'
  on-tertiary: '#ffffff'
  tertiary-container: '#248279'
  on-tertiary-container: '#f3fffc'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#89f5e7'
  primary-fixed-dim: '#6bd8cb'
  on-primary-fixed: '#00201d'
  on-primary-fixed-variant: '#005049'
  secondary-fixed: '#cce5ff'
  secondary-fixed-dim: '#93ccff'
  on-secondary-fixed: '#001d31'
  on-secondary-fixed-variant: '#004b73'
  tertiary-fixed: '#9cf2e8'
  tertiary-fixed-dim: '#80d5cb'
  on-tertiary-fixed: '#00201d'
  on-tertiary-fixed-variant: '#00504a'
  background: '#faf8ff'
  on-background: '#131b2e'
  surface-variant: '#dae2fd'
typography:
  headline-xl:
    fontFamily: plusJakartaSans
    fontSize: 40px
    fontWeight: '700'
    lineHeight: 48px
    letterSpacing: -0.025em
  headline-xl-mobile:
    fontFamily: plusJakartaSans
    fontSize: 30px
    fontWeight: '700'
    lineHeight: 38px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: plusJakartaSans
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: plusJakartaSans
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: plusJakartaSans
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.015em
  headline-sm:
    fontFamily: plusJakartaSans
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
    letterSpacing: -0.005em
  body-md:
    fontFamily: inter
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: 0em
  body-sm:
    fontFamily: inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0.005em
  label-lg:
    fontFamily: inter
    fontSize: 14px
    fontWeight: '600'
    lineHeight: 20px
    letterSpacing: 0.01em
  label-md:
    fontFamily: inter
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-sm:
    fontFamily: inter
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.04em
  metric-display:
    fontFamily: plusJakartaSans
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 52px
    letterSpacing: -0.03em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  space-2xs: 0.25rem
  space-xs: 0.5rem
  space-sm: 0.75rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2rem
  space-2xl: 3rem
  space-3xl: 4rem
  gutter-mobile: 1rem
  gutter-desktop: 1.5rem
  container-max: 88rem
---

## Brand & Style

This design system establishes a clinical-grade yet deeply human visual language for preventative healthcare and predictive AI diagnostics. Combining the disciplined clarity of clinical documentation systems with the warm, reassuring poise of modern wellness interfaces (evoking Apple Health and modern health-tech platforms), the visual style centers on **Clean Modern Health-Tech**.

Key tenets:
- **Clinical Serenity:** High visual breathing room, deliberate whitespace, and a soothing maritime-teal palette dispel the anxiety commonly associated with medical screenings and risk predictions.
- **Data Lucidity:** Complex biometric models, longitudinal trends, and algorithmic risk vectors are rendered with crisp contrast, unambiguous visual hierarchies, and instant semantic recognition.
- **Tactile Trust:** Subtle micro-shadows, soft structural borders, and rounded card envelopes frame dense physiological telemetry without appearing sterile or institutional.

## Colors

The color architecture is calibrated for high diagnostic legibility, strict WCAG AAA accessibility, and intuitive triage comprehension.

### Core Roles
- **Primary Deep Teal (`#0D9488` / `#0F766E`):** Represents vitality, clinical precision, and authoritative confidence. Applied to primary CTAs, active telemetry indicators, and focused navigational highlights.
- **Secondary Slate Blue (`#0284C7`):** Acts as the clinical diagnostic accent for auxiliary actions, trendlines, longitudinal telemetry charts, and interactive informative states.
- **Tertiary Forest Teal (`#0F766E`):** Used for pressed states, high-contrast headings, and grounding graphical anchors.
- **Neutral Deep Slate (`#0F172A` / `#334155`):** High-clarity typography system avoiding harsh optical blacks to prevent cognitive strain during prolonged analysis.
- **Base Canvas & Card Layers (`#F8FAFC` canvas, `#FFFFFF` containers, `#E2E8F0` hairline borders):** Ensure pristine separation between background wash and interactive diagnostic surfaces.

### Semantic Risk Stratification
- **Low Risk (Optimal / Safe):** Emerald `#10B981` (Surface: `#ECFDF5`, Border: `#A7F3D0`, Ink: `#065F46`).
- **Moderate Risk (Precautionary / Monitor):** Amber `#F59E0B` (Surface: `#FFFBEB`, Border: `#FDE68A`, Ink: `#92400E`).
- **High Risk (Critical / Intervention):** Crimson `#EF4444` (Surface: `#FEF2F2`, Border: `#FECACA`, Ink: `#991B1B`).

## Typography

The typography pairings strike a deliberate equilibrium between clinical rigor and approachable patient communication.

- **Display & Headings (Plus Jakarta Sans):** Its geometric foundation with rounded humanist terminations introduces reassuring warmth to health assessments, preventing diagnostic outputs from feeling intimidating.
- **Body & Telemetry Labels (Inter):** Leveraged for its tall x-height, structural neutrality, and exceptional numerical clarity. In numerical medical tables and longitudinal reports, tabular lining figures (`tnum`) must always be enabled.
- **Hierarchy Rules:** Large numerical metrics (`metric-display`) anchor clinical dashboards with instant scannability, paired with uppercase micro-labels (`label-sm`) for units (e.g., `mg/dL`, `bpm`, `% AI CONFIDENCE`).

## Layout & Spacing

This design system uses an **8pt geometric spacing scale** embedded inside an adaptable fluid column grid system.

### Grid Specifications
- **Desktop (≥1024px):** 12-column responsive layout, 24px gutters, maximum container width `88rem` (1408px), horizontal outer margins of 32px to 64px.
- **Tablet (768px - 1023px):** 8-column layout, 20px gutters, 24px page margins.
- **Mobile (<768px):** 4-column layout, 16px gutters, 16px page margins. Metrics collapse from multi-column grids to vertical stacked cards.

### Spatial Rhythm
Diagnostic dashboards require rhythm without claustrophobia. Spacing between major analytical clusters defaults to `space-2xl` (48px), while internal card paddings use `space-lg` (24px) on desktop and `space-md` (16px) on compact viewports. Critical status indicators always retain at least `space-xs` (8px) internal breathing room from adjacent copy.

## Elevation & Depth

Visual hierarchy uses **tonal layering combined with ultra-diffused micro-shadows**, ensuring that critical clinical cards rise softly off the `#F8FAFC` foundation canvas.

### Depth Stratification
1. **Base Layer (L0 - Background):** `#F8FAFC` canvas. Flat, neutral, zero elevation.
2. **Container Layer (L1 - Medical Cards & Modules):** Pure white (`#FFFFFF`) surface framed by a hairline border (`1px solid #E2E8F0`). Enhanced with an ambient clinical micro-shadow:
   - `0 1px 3px 0 rgba(15, 23, 42, 0.04), 0 1px 2px -1px rgba(15, 23, 42, 0.02)`
3. **Interactive & Hover Layer (L2 - Popovers, Action Menus, Hovered Cards):** 
   - `0 10px 15px -3px rgba(15, 23, 42, 0.06), 0 4px 6px -4px rgba(15, 23, 42, 0.03)`
   - Border slightly brightens to `#CBD5E1`.
4. **Overlay Layer (L3 - Risk Modals & Urgent Clinical Interventions):**
   - Backdrop: `rgba(15, 23, 42, 0.45)` with `backdrop-blur(8px)`.
   - Card: `0 20px 25px -5px rgba(15, 23, 42, 0.1), 0 8px 10px -6px rgba(15, 23, 42, 0.04)`.

## Shapes

The design system incorporates a **Rounded (Level 2)** shape vocabulary, utilizing generous `rounded-2xl` corner radii across macro modules to reduce clinical sharpness and foster emotional reassurance.

- **Primary Cards & Modals:** `1rem` (16px) standard; `1.5rem` (24px) for hero assessment overviews (`rounded-2xl`).
- **Interactive Controls (Buttons, Inputs):** `0.625rem` (10px) to `0.75rem` (12px), balancing modern ergonomic tap targets with clinical authority.
- **Telemetry Chips, Risk Badges & Status Pills:** Fully rounded (`rounded-full` / 9999px) to communicate modular encapsulation and categorical distinction.

## Components

### Buttons
- **Primary Action:** Solid Deep Teal (`#0D9488`), text pure white, height 44px (touch-compliant 48px on mobile), radius `10px`. Hover state: `#0F766E`. Active: scale `0.99`.
- **Secondary Action:** White background, `1px solid #E2E8F0`, slate text `#334155`. Hover: `#F8FAFC` background with `#CBD5E1` border.
- **Destructive/Emergency:** Crimson `#EF4444` solid fill with white text, or light crimson surface (`#FEF2F2`) with `#DC2626` text for lower priority alerts.

### Risk Badges & Severity Chips
- Displayed as compact pills (`rounded-full`, padding `4px 12px`, typography `label-sm`).
- Low Risk: Tinted emerald background `#ECFDF5`, border `#A7F3D0`, text `#065F46`, paired with an optional 6px pulsing indicator dot.
- Moderate Risk: Amber surface `#FFFBEB`, border `#FDE68A`, text `#92400E`.
- High Risk: Coral/Crimson surface `#FEF2F2`, border `#FECACA`, text `#991B1B`.

### Diagnostic Cards
- White `#FFFFFF` core surface with `rounded-2xl` radius, `1px solid #E2E8F0`, and ambient micro-elevation.
- Divided into three internal zones: 
  1. *Header Zone:* Contains metric taxonomy, AI confidence index chip, and contextual icon.
  2. *Payload Zone:* High-contrast visual metric (`metric-display`) or sparkline trend.
  3. *Triage Footnote:* Actionable clinician interpretation or status delta (e.g., "+4.2% vs 90-day baseline").

### Input Fields & Controls
- Height 44px, background `#FFFFFF`, border `1px solid #CBD5E1`, typography `body-md` in `#0F172A`. Placeholder in `#94A3B8`.
- Focus state: Border transitions to `#0D9488` with a soft 3px halo `rgba(13, 148, 136, 0.15)`.
- Checkboxes & Radios: 20px precision controls. Checked state utilizes `#0D9488` with a crisp white glyph; radios feature a centered 8px solid dot.

### Specialized Health Components
- **Vital Sign Sparklines:** High-density mini-charts rendered with `#0284C7` or appropriate risk token colors, backed by a gentle 12% opacity vertical gradient falloff.
- **AI Recommendation Alert Banner:** Framed within a soft gradient band (`linear-gradient(135deg, #F0FDFA 0%, #E0F2FE 100%)`) bordered by `#99F6E4`, presenting predictive lifestyle interventions and medical advisory notifications.