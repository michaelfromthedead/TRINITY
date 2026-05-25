/**
 * Shared Theme Configuration for Trinity Node Renderers
 *
 * Centralizes all color definitions, layout constants, and font specifications
 * used across the different node types (Component, System, Resource, Event).
 *
 * This ensures visual consistency and makes theme changes easier to manage.
 */

// =============================================================================
// TRINITY COLOR PALETTE
// =============================================================================

/**
 * Trinity ECS color palette.
 * These colors match the Python backend constants.
 */
export const TRINITY_COLORS = {
  /** Component nodes - Blue theme */
  component: {
    primary: '#3B82F6',
    primaryDark: '#2563EB',
    background: '#EFF6FF',
    backgroundHover: '#DBEAFE',
    border: '#93C5FD',
    accent: '#60A5FA',
  },

  /** System nodes - Green theme */
  system: {
    primary: '#22C55E',
    primaryDark: '#16A34A',
    background: '#F0FDF4',
    backgroundHover: '#DCFCE7',
    border: '#86EFAC',
    accent: '#4ADE80',
  },

  /** Resource nodes - Purple theme */
  resource: {
    primary: '#A855F7',
    primaryDark: '#9333EA',
    background: '#FAF5FF',
    backgroundHover: '#F3E8FF',
    border: '#C084FC',
    accent: '#7C3AED',
  },

  /** Event nodes - Orange theme */
  event: {
    primary: '#F97316',
    primaryDark: '#EA580C',
    background: '#FFF7ED',
    backgroundHover: '#FFEDD5',
    border: '#FB923C',
    accent: '#F59E0B',
  },

  /** Shared neutral colors */
  neutral: {
    white: '#FFFFFF',
    black: '#000000',
    textPrimary: '#1F2937',
    textSecondary: '#374151',
    textMuted: '#6B7280',
    textLight: '#9CA3AF',
  },
} as const

// =============================================================================
// LAYOUT CONSTANTS
// =============================================================================

/**
 * Shared layout measurements for all node types.
 */
export const NODE_LAYOUT = {
  /** Header/title bar height */
  headerHeight: 32,

  /** Internal padding */
  padding: 12,

  /** Border radius for rounded corners */
  borderRadius: 8,

  /** Height per field row */
  fieldRowHeight: 20,

  /** Gap between fields */
  fieldGap: 4,

  /** Slot height */
  slotHeight: 22,

  /** Slot radius */
  slotRadius: 6,

  /** Icon size in header */
  iconSize: 14,

  /** Footer height for source info */
  footerHeight: 16,

  /** Minimum node width */
  minWidth: 200,

  /** Maximum node width */
  maxWidth: 350,

  /** Section header height */
  sectionHeaderHeight: 20,

  /** Collapse indicator height */
  collapseIndicatorHeight: 20,

  /** Maximum visible fields before collapse */
  maxVisibleFields: 8,
} as const

// =============================================================================
// FONT DEFINITIONS
// =============================================================================

/**
 * Shared font definitions for all node types.
 */
export const NODE_FONTS = {
  /** Header title font */
  header: 'bold 12px Inter, system-ui, sans-serif',

  /** Section header font */
  sectionHeader: 'bold 10px Inter, system-ui, sans-serif',

  /** Field name font */
  fieldName: '11px Inter, system-ui, sans-serif',

  /** Field type font (monospace) */
  fieldType: '10px "JetBrains Mono", "Fira Code", Consolas, monospace',

  /** Method signature font (monospace) */
  method: '11px "JetBrains Mono", "Fira Code", Consolas, monospace',

  /** Default value font */
  defaultValue: '10px "JetBrains Mono", "Fira Code", Consolas, monospace',

  /** Icon font */
  icon: '12px Inter, system-ui, sans-serif',

  /** Footer/source file font */
  footer: '9px Inter, system-ui, sans-serif',

  /** Badge font */
  badge: 'bold 9px Inter, system-ui, sans-serif',

  /** Collapse indicator font */
  collapse: '10px Inter, system-ui, sans-serif',
} as const

// =============================================================================
// EDGE COLORS
// =============================================================================

/**
 * Edge/link colors for different relationship types.
 */
export const EDGE_COLORS = {
  /** System -> Component query relationship */
  query: '#22C55E',

  /** Field type reference */
  reference: '#6B7280',

  /** Class inheritance */
  inheritance: '#3B82F6',

  /** System handles Event */
  eventHandler: '#F97316',

  /** System emits Event */
  eventEmit: '#F97316',

  /** Default fallback */
  default: '#9CA3AF',
} as const

// =============================================================================
// TYPE DETECTION
// =============================================================================

/**
 * Types that are considered "complex" and should create connections
 * rather than being displayed as simple literals.
 */
export const COMPLEX_TYPES = new Set([
  'Entity',
  'Vec2',
  'Vec3',
  'Vec4',
  'Mat3',
  'Mat4',
  'Transform',
  'Texture',
  'Mesh',
  'Material',
  'Sprite',
  'AudioSource',
  'Collider',
])

/**
 * Check if a type name represents a complex type that should
 * create a connection slot rather than a simple field.
 */
export function isComplexType(typeName: string): boolean {
  // Check for direct match
  if (COMPLEX_TYPES.has(typeName)) {
    return true
  }

  // Check for generic types containing complex types (e.g., List[Entity])
  for (const complexType of COMPLEX_TYPES) {
    if (typeName.includes(complexType)) {
      return true
    }
  }

  return false
}

// =============================================================================
// NODE TYPE REGISTRATION NAMES
// =============================================================================

/**
 * Consistent registration type names for LiteGraph.
 */
export const NODE_TYPE_NAMES = {
  component: 'trinity/component',
  system: 'trinity/system',
  resource: 'trinity/resource',
  event: 'trinity/event',
} as const

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Truncate text to fit within a maximum width, adding ellipsis if needed.
 *
 * @param ctx - Canvas 2D context for text measurement
 * @param text - The text to truncate
 * @param maxWidth - Maximum width in pixels
 * @returns The truncated text with ellipsis if needed
 */
export function truncateText(
  ctx: CanvasRenderingContext2D,
  text: string,
  maxWidth: number
): string {
  const width = ctx.measureText(text).width
  if (width <= maxWidth) return text

  const ellipsis = '...'
  const ellipsisWidth = ctx.measureText(ellipsis).width

  if (ellipsisWidth >= maxWidth) return ellipsis

  let truncated = text
  while (ctx.measureText(truncated).width + ellipsisWidth > maxWidth && truncated.length > 0) {
    truncated = truncated.slice(0, -1)
  }

  return truncated + ellipsis
}

/**
 * Draw a rounded rectangle path on a canvas.
 *
 * @param ctx - Canvas 2D context
 * @param x - X position
 * @param y - Y position
 * @param width - Rectangle width
 * @param height - Rectangle height
 * @param radius - Corner radius (single value or array [tl, tr, br, bl])
 */
export function drawRoundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number | number[] | { tl: number; tr: number; br: number; bl: number }
): void {
  let tl: number, tr: number, br: number, bl: number

  if (typeof radius === 'number') {
    tl = tr = br = bl = radius
  } else if (Array.isArray(radius)) {
    tl = radius[0] ?? 0
    tr = radius[1] ?? tl
    br = radius[2] ?? tl
    bl = radius[3] ?? tr
  } else {
    tl = radius.tl
    tr = radius.tr
    br = radius.br
    bl = radius.bl
  }

  ctx.beginPath()
  ctx.moveTo(x + tl, y)
  ctx.lineTo(x + width - tr, y)
  ctx.quadraticCurveTo(x + width, y, x + width, y + tr)
  ctx.lineTo(x + width, y + height - br)
  ctx.quadraticCurveTo(x + width, y + height, x + width - br, y + height)
  ctx.lineTo(x + bl, y + height)
  ctx.quadraticCurveTo(x, y + height, x, y + height - bl)
  ctx.lineTo(x, y + tl)
  ctx.quadraticCurveTo(x, y, x + tl, y)
  ctx.closePath()
}

/**
 * Create a linear gradient for node headers.
 *
 * @param ctx - Canvas 2D context
 * @param width - Width of the gradient
 * @param primaryColor - Start color
 * @param darkColor - End color
 * @returns The created gradient
 */
export function createHeaderGradient(
  ctx: CanvasRenderingContext2D,
  width: number,
  primaryColor: string,
  darkColor: string
): CanvasGradient {
  const gradient = ctx.createLinearGradient(0, 0, width, 0)
  gradient.addColorStop(0, primaryColor)
  gradient.addColorStop(1, darkColor)
  return gradient
}

// =============================================================================
// TYPE EXPORTS
// =============================================================================

export type TrinityNodeType = keyof typeof TRINITY_COLORS
export type NodeTypeName = keyof typeof NODE_TYPE_NAMES
