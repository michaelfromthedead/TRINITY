/**
 * Edge Styling Configuration for Trinity ECS
 *
 * Defines visual styles for different relationship types between Trinity nodes:
 * - Query edges: System -> Component (green, arrow)
 * - Reference edges: Field type references (gray, solid)
 * - Inheritance edges: Class extends (blue, dashed)
 * - Event Handler edges: System handles Event (orange, arrow toward system)
 * - Event Emit edges: System emits Event (orange, dotted arrow toward event)
 */

import type { LGraph } from '../src/LGraph'
import type { LLink } from '../src/LLink'
import type { CanvasColour, Point } from '../src/interfaces'
import { EDGE_COLORS as SHARED_EDGE_COLORS } from '../../nodes/nodeTheme'

/**
 * Edge style pattern types.
 */
export type EdgePattern = 'solid' | 'dashed' | 'dotted'

/**
 * Edge style configuration interface.
 */
export interface EdgeStyle {
  /** The color of the edge line */
  color: string
  /** The thickness of the edge line in pixels */
  thickness: number
  /** The line pattern: solid, dashed, or dotted */
  pattern: EdgePattern
  /** Whether to draw an arrow at the end of the edge */
  arrow: boolean
  /** Size of the arrow head in pixels (if arrow is true) */
  arrowSize?: number
  /** Optional label to display on the edge */
  label?: string
  /** The dash array pattern for dashed/dotted lines */
  dashArray?: number[]
  /** Glow effect color (optional, for emphasis) */
  glowColor?: string
  /** Opacity of the edge (0-1) */
  opacity?: number
}

/**
 * Trinity edge relationship types.
 * These correspond to the relationships parsed from Python source code.
 */
export type TrinityEdgeType =
  | 'query'          // System -> Component
  | 'reference'      // Field type reference
  | 'inheritance'    // Class extends another class
  | 'event_handler'  // System handles Event
  | 'event_emit'     // System emits Event
  | 'default'        // Default fallback style

/**
 * Edge style definitions for each Trinity relationship type.
 * Colors are imported from the shared nodeTheme.
 */
export const EDGE_STYLES: Record<TrinityEdgeType, EdgeStyle> = {
  /**
   * Query Edge: System -> Component
   * Green color with arrow pointing to the component.
   * Represents a system querying for entities with a specific component.
   */
  query: {
    color: SHARED_EDGE_COLORS.query,
    thickness: 2,
    pattern: 'solid',
    arrow: true,
    arrowSize: 8,
    label: 'queries',
    glowColor: 'rgba(34, 197, 94, 0.3)',
    opacity: 1,
  },

  /**
   * Reference Edge: Field type reference
   * Gray color, solid line without arrow.
   * Represents a field in one type that references another type.
   */
  reference: {
    color: SHARED_EDGE_COLORS.reference,
    thickness: 1,
    pattern: 'solid',
    arrow: false,
    opacity: 0.8,
  },

  /**
   * Inheritance Edge: Class extends
   * Blue color with dashed line.
   * Represents class inheritance relationships.
   */
  inheritance: {
    color: SHARED_EDGE_COLORS.inheritance,
    thickness: 2,
    pattern: 'dashed',
    arrow: true,
    arrowSize: 10,
    label: 'extends',
    dashArray: [8, 4],
    glowColor: 'rgba(59, 130, 246, 0.3)',
    opacity: 1,
  },

  /**
   * Event Handler Edge: System handles Event
   * Orange color with arrow pointing toward the system.
   * Represents a system that handles/responds to an event.
   */
  event_handler: {
    color: SHARED_EDGE_COLORS.eventHandler,
    thickness: 2,
    pattern: 'solid',
    arrow: true,
    arrowSize: 8,
    label: 'handles',
    glowColor: 'rgba(249, 115, 22, 0.3)',
    opacity: 1,
  },

  /**
   * Event Emit Edge: System emits Event
   * Orange color with dotted line and arrow toward the event.
   * Represents a system that emits/triggers an event.
   */
  event_emit: {
    color: SHARED_EDGE_COLORS.eventEmit,
    thickness: 2,
    pattern: 'dotted',
    arrow: true,
    arrowSize: 8,
    label: 'emits',
    dashArray: [3, 3],
    glowColor: 'rgba(249, 115, 22, 0.3)',
    opacity: 1,
  },

  /**
   * Default Edge: Fallback style
   * Neutral gray color for unspecified edge types.
   */
  default: {
    color: SHARED_EDGE_COLORS.default,
    thickness: 1.5,
    pattern: 'solid',
    arrow: true,
    arrowSize: 6,
    opacity: 0.9,
  },
}

/**
 * Extended edge style with computed properties.
 */
export interface ComputedEdgeStyle extends EdgeStyle {
  /** Computed dash array for canvas setLineDash */
  computedDashArray: number[]
}

/**
 * Get the edge style for a given edge type.
 * Returns a computed style with dash array resolved.
 *
 * @param edgeType - The Trinity edge type
 * @returns The computed edge style configuration
 */
export function getEdgeStyle(edgeType: string): ComputedEdgeStyle {
  const style = EDGE_STYLES[edgeType as TrinityEdgeType] || EDGE_STYLES.default

  // Compute dash array based on pattern if not explicitly set
  let computedDashArray: number[] = []
  if (style.dashArray) {
    computedDashArray = style.dashArray
  } else {
    switch (style.pattern) {
      case 'dashed':
        computedDashArray = [8, 4]
        break
      case 'dotted':
        computedDashArray = [2, 4]
        break
      case 'solid':
      default:
        computedDashArray = []
    }
  }

  return {
    ...style,
    computedDashArray,
  }
}

/**
 * Check if an edge type is a Trinity edge type.
 *
 * @param edgeType - The edge type string to check
 * @returns True if it's a recognized Trinity edge type
 */
export function isTrinityEdgeType(edgeType: string): edgeType is TrinityEdgeType {
  return edgeType in EDGE_STYLES
}

/**
 * Extended link information for Trinity edges.
 * This can be stored in LLink.data or a custom property.
 */
export interface TrinityLinkData {
  /** The Trinity edge type */
  edgeType: TrinityEdgeType
  /** Optional label override */
  label?: string
  /** Whether to show the label */
  showLabel?: boolean
  /** Source node class name */
  sourceClass?: string
  /** Target node class name */
  targetClass?: string
}

/**
 * Apply Trinity edge styles to a graph's links.
 * This function iterates through all links and sets custom colors
 * based on their Trinity edge type.
 *
 * @param graph - The LiteGraph instance to style
 */
export function applyEdgeStyles(graph: LGraph): void {
  if (!graph.links) return

  graph.links.forEach((link) => {
    applyEdgeStyleToLink(link)
  })
}

/**
 * Apply Trinity edge style to a single link.
 *
 * @param link - The link to style
 */
export function applyEdgeStyleToLink(link: LLink): void {
  const linkData = getTrinityLinkData(link)

  if (linkData?.edgeType) {
    const style = getEdgeStyle(linkData.edgeType)
    link.color = style.color
  }
}

/**
 * Get the Trinity link data from a link.
 * Uses the _data field which supports arbitrary data.
 *
 * @param link - The link to get data from
 * @returns The Trinity link data or undefined
 */
export function getTrinityLinkData(link: LLink): TrinityLinkData | undefined {
  return link._data as TrinityLinkData | undefined
}

/**
 * Set the Trinity link data on a link.
 * Uses the _data field which supports arbitrary data.
 *
 * @param link - The link to set data on
 * @param data - The Trinity link data
 */
export function setTrinityLinkData(link: LLink, data: TrinityLinkData): void {
  link._data = data
}

/**
 * Context for custom link rendering.
 */
export interface TrinityLinkRenderContext {
  ctx: CanvasRenderingContext2D
  startPos: Point
  endPos: Point
  link: LLink
  style: ComputedEdgeStyle
  scale: number
}

/**
 * Draw a styled Trinity edge on the canvas.
 * This function can be used to override default link rendering.
 *
 * @param context - The rendering context
 */
export function drawTrinityEdge(context: TrinityLinkRenderContext): void {
  const { ctx, startPos, endPos, style, scale } = context
  const [x1, y1] = startPos
  const [x2, y2] = endPos

  ctx.save()

  // Set line style
  ctx.strokeStyle = style.color
  ctx.lineWidth = style.thickness * Math.max(0.5, Math.min(1, scale))
  ctx.globalAlpha = style.opacity ?? 1

  // Set dash pattern
  if (style.computedDashArray.length > 0) {
    ctx.setLineDash(style.computedDashArray.map(v => v * scale))
  } else {
    ctx.setLineDash([])
  }

  // Optional glow effect
  if (style.glowColor) {
    ctx.shadowColor = style.glowColor
    ctx.shadowBlur = 4 * scale
  }

  // Draw the bezier curve
  ctx.beginPath()
  ctx.moveTo(x1, y1)

  // Calculate control points for smooth bezier curve
  const midX = (x1 + x2) / 2
  const dx = Math.abs(x2 - x1)
  const controlOffset = Math.min(dx * 0.5, 100)

  ctx.bezierCurveTo(
    x1 + controlOffset,
    y1,
    x2 - controlOffset,
    y2,
    x2,
    y2
  )

  ctx.stroke()

  // Draw arrow if enabled
  if (style.arrow && style.arrowSize) {
    drawArrowHead(ctx, x1, y1, x2, y2, style.arrowSize * scale, style.color)
  }

  // Draw label if present
  if (style.label) {
    drawEdgeLabel(ctx, midX, (y1 + y2) / 2, style.label, style.color, scale)
  }

  // Reset context
  ctx.setLineDash([])
  ctx.shadowBlur = 0
  ctx.restore()
}

/**
 * Draw an arrow head at the end of an edge.
 *
 * @param ctx - Canvas rendering context
 * @param x1 - Start X coordinate
 * @param y1 - Start Y coordinate
 * @param x2 - End X coordinate
 * @param y2 - End Y coordinate
 * @param size - Size of the arrow head
 * @param color - Color of the arrow
 */
function drawArrowHead(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  size: number,
  color: CanvasColour
): void {
  // Calculate the angle of the line
  const angle = Math.atan2(y2 - y1, x2 - x1)

  // Calculate arrow points
  const arrowAngle = Math.PI / 6 // 30 degrees

  ctx.save()
  ctx.fillStyle = color
  ctx.beginPath()
  ctx.moveTo(x2, y2)
  ctx.lineTo(
    x2 - size * Math.cos(angle - arrowAngle),
    y2 - size * Math.sin(angle - arrowAngle)
  )
  ctx.lineTo(
    x2 - size * Math.cos(angle + arrowAngle),
    y2 - size * Math.sin(angle + arrowAngle)
  )
  ctx.closePath()
  ctx.fill()
  ctx.restore()
}

/**
 * Draw a label on an edge.
 *
 * @param ctx - Canvas rendering context
 * @param x - X position for the label
 * @param y - Y position for the label
 * @param label - The text to display
 * @param color - Color of the label
 * @param scale - Current canvas scale
 */
function drawEdgeLabel(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  label: string,
  color: CanvasColour,
  scale: number
): void {
  const fontSize = Math.max(10, 12 * scale)

  ctx.save()
  ctx.font = `${fontSize}px Inter, system-ui, sans-serif`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'

  // Draw background for readability
  const metrics = ctx.measureText(label)
  const padding = 4 * scale
  const bgWidth = metrics.width + padding * 2
  const bgHeight = fontSize + padding

  ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
  ctx.fillRect(
    x - bgWidth / 2,
    y - bgHeight / 2,
    bgWidth,
    bgHeight
  )

  // Draw text
  ctx.fillStyle = color
  ctx.fillText(label, x, y)
  ctx.restore()
}

/**
 * Create a custom link renderer that applies Trinity edge styles.
 * This can be used to intercept link rendering in LiteGraph.
 *
 * @returns A function that renders styled links
 */
export function createTrinityLinkRenderer() {
  return function renderTrinityLink(
    ctx: CanvasRenderingContext2D,
    startPos: Point,
    endPos: Point,
    link: LLink | null,
    scale: number = 1
  ): boolean {
    if (!link) return false

    const linkData = getTrinityLinkData(link)

    // Only handle Trinity-styled links
    if (!linkData?.edgeType || !isTrinityEdgeType(linkData.edgeType)) {
      return false // Let default renderer handle it
    }

    const style = getEdgeStyle(linkData.edgeType)

    drawTrinityEdge({
      ctx,
      startPos,
      endPos,
      link,
      style,
      scale,
    })

    return true // We handled the rendering
  }
}

/**
 * Utility to set Trinity edge data on a link.
 *
 * @param link - The link to configure
 * @param edgeType - The Trinity edge type
 * @param options - Additional options
 */
export function setTrinityEdgeType(
  link: LLink,
  edgeType: TrinityEdgeType,
  options: Partial<Omit<TrinityLinkData, 'edgeType'>> = {}
): void {
  const style = getEdgeStyle(edgeType)

  // Set the link color
  link.color = style.color

  // Store the edge data using _data field (supports arbitrary data)
  const linkData: TrinityLinkData = {
    edgeType,
    showLabel: true,
    ...options,
  }
  setTrinityLinkData(link, linkData)
}

/**
 * Color palette for Trinity edges (re-exported from shared theme).
 */
export const EDGE_COLORS = {
  query: SHARED_EDGE_COLORS.query,
  reference: SHARED_EDGE_COLORS.reference,
  inheritance: SHARED_EDGE_COLORS.inheritance,
  event_handler: SHARED_EDGE_COLORS.eventHandler,
  event_emit: SHARED_EDGE_COLORS.eventEmit,
  default: SHARED_EDGE_COLORS.default,
} as const

/**
 * Get all available edge types.
 *
 * @returns Array of Trinity edge type strings
 */
export function getAvailableEdgeTypes(): TrinityEdgeType[] {
  return Object.keys(EDGE_STYLES) as TrinityEdgeType[]
}

/**
 * Configuration for the Trinity edge styling system.
 */
export interface TrinityEdgeConfig {
  /** Whether to show edge labels by default */
  showLabels: boolean
  /** Whether to enable glow effects */
  enableGlow: boolean
  /** Global opacity multiplier (0-1) */
  opacityMultiplier: number
  /** Minimum scale at which to render edge details */
  detailThreshold: number
}

/**
 * Default edge configuration.
 */
export const DEFAULT_EDGE_CONFIG: TrinityEdgeConfig = {
  showLabels: true,
  enableGlow: true,
  opacityMultiplier: 1,
  detailThreshold: 0.3,
}
