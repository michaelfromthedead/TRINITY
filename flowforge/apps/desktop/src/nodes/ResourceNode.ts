/**
 * ResourceNode - Custom LiteGraph Node Renderer for Trinity ECS Resources
 *
 * A specialized node renderer for Trinity resources with:
 * - Purple gradient header (#A855F7 to #9333EA)
 * - Singleton badge indicator
 * - Field list display
 * - Output slots only (resources are read by systems)
 * - Distinctive rounded shape to differentiate from Components
 */

import { LGraphNode, LiteGraph } from '../litegraph'
import type { Size } from '../litegraph'
import {
  TRINITY_COLORS,
  NODE_LAYOUT,
  NODE_FONTS,
  NODE_TYPE_NAMES,
  truncateText as sharedTruncateText,
  drawRoundRect,
} from './nodeTheme'

// =============================================================================
// TYPES
// =============================================================================

/**
 * Data structure from Python parser for resource definitions.
 */
export interface ResourceNodeData {
  name: string
  fields: Array<{
    name: string
    type: string
    default?: string
  }>
  is_singleton: boolean
  docstring?: string
  source?: { file: string; line: number }
}

/**
 * Field definition for resource nodes.
 */
export interface ResourceField {
  name: string
  type: string
  default?: string | undefined
}

// =============================================================================
// CONSTANTS (re-exported for external use, built from shared theme)
// =============================================================================

/** Resource node color palette */
export const RESOURCE_COLORS = {
  /** Primary purple */
  primary: TRINITY_COLORS.resource.primary,
  /** Dark purple for gradients */
  primaryDark: TRINITY_COLORS.resource.primaryDark,
  /** Light purple for accents */
  primaryLight: TRINITY_COLORS.resource.border,
  /** Very light purple for backgrounds */
  bgLight: TRINITY_COLORS.resource.background,
  /** Node background with alpha */
  bgAlpha: 'rgba(168, 85, 247, 0.12)',
  /** Singleton badge background */
  singletonBg: TRINITY_COLORS.resource.accent,
  /** White text */
  textWhite: TRINITY_COLORS.neutral.white,
  /** Muted text */
  textMuted: TRINITY_COLORS.neutral.textMuted,
  /** Field name text */
  textField: TRINITY_COLORS.neutral.textSecondary,
  /** Type text */
  textType: TRINITY_COLORS.resource.primaryDark,
} as const

/** Node layout configuration (using shared layout) */
const LAYOUT = {
  /** Header/title bar height */
  headerHeight: NODE_LAYOUT.headerHeight,
  /** Height per field row */
  fieldHeight: NODE_LAYOUT.fieldRowHeight,
  /** Internal padding */
  padding: NODE_LAYOUT.padding,
  /** Slot height */
  slotHeight: NODE_LAYOUT.slotHeight,
  /** Corner radius for rounded shape */
  cornerRadius: NODE_LAYOUT.borderRadius,
  /** Singleton badge dimensions */
  singletonBadge: {
    width: 22,
    height: 16,
    radius: 4,
  },
  /** Icon size */
  iconSize: NODE_LAYOUT.iconSize,
  /** Bottom padding for source info */
  sourceHeight: NODE_LAYOUT.footerHeight,
} as const

/** Font definitions (using shared fonts) */
const FONTS = {
  header: NODE_FONTS.header,
  field: NODE_FONTS.fieldName,
  fieldType: NODE_FONTS.fieldType,
  badge: NODE_FONTS.badge,
  icon: NODE_FONTS.icon,
  source: NODE_FONTS.footer,
} as const

// =============================================================================
// RESOURCE NODE CLASS
// =============================================================================

/**
 * Custom Resource Node renderer for LiteGraph.
 *
 * Features:
 * - Purple gradient header with "@resource" icon
 * - Singleton indicator badge
 * - Field list with types
 * - Rounded corners for visual distinction
 * - Output slots only (resources provide data to systems)
 *
 * @example
 * ```typescript
 * import { ResourceNode } from '@/nodes/ResourceNode'
 * import { LiteGraph } from '@/litegraph'
 *
 * // Register the node type
 * LiteGraph.registerNodeType('flowforge/resource', ResourceNode)
 *
 * // Create and configure a resource node
 * const node = LiteGraph.createNode('flowforge/resource') as ResourceNode
 * node.configureFromData({
 *   name: 'GameConfig',
 *   fields: [{ name: 'difficulty', type: 'int', default: '1' }],
 *   is_singleton: true,
 *   docstring: 'Global game configuration',
 *   source: { file: 'config.py', line: 42 }
 * })
 * ```
 */
export class ResourceNode extends LGraphNode {
  static title = 'Resource'
  static desc = 'A Trinity ECS resource (shared data singleton)'
  static category = 'flowforge/ecs'

  // Resource-specific properties
  resourceName: string = 'Resource'
  fields: ResourceField[] = []
  isSingleton: boolean = true
  docstring?: string
  sourceFile?: string
  sourceLine?: number

  constructor(title?: string) {
    super(title || 'Resource', 'flowforge/resource')

    // Set Trinity purple theme colors
    this.color = RESOURCE_COLORS.primary
    this.bgcolor = RESOURCE_COLORS.bgAlpha

    // Initialize with default size
    this.size = [220, 100] as Size

    // Resources only have outputs (they provide data)
    this.addOutput('data', 'resource')
  }

  /**
   * Configure the node from parsed Python data.
   */
  configureFromData(data: ResourceNodeData): void {
    this.resourceName = data.name
    this.title = data.name
    this.fields = data.fields.map((f): ResourceField => ({
      name: f.name,
      type: f.type,
      default: f.default,
    }))
    this.isSingleton = data.is_singleton
    if (data.docstring !== undefined) {
      this.docstring = data.docstring
    }

    if (data.source) {
      this.sourceFile = data.source.file
      this.sourceLine = data.source.line
    }

    // Recalculate node size based on content
    this.updateSize()
  }

  /**
   * Calculate and update the node size based on content.
   */
  updateSize(): void {
    const fieldsHeight = this.fields.length * LAYOUT.fieldHeight
    const outputsHeight = (this.outputs?.length || 1) * LAYOUT.slotHeight
    const sourceHeight = this.sourceFile ? LAYOUT.sourceHeight : 0

    const height =
      LAYOUT.headerHeight +
      LAYOUT.padding +
      Math.max(fieldsHeight, outputsHeight) +
      LAYOUT.padding +
      sourceHeight

    const maxFieldWidth = this.fields.reduce((max, field) => {
      const fieldText = `${field.name}: ${field.type}`
      return Math.max(max, fieldText.length * 7)
    }, 0)

    const width = Math.max(220, maxFieldWidth + LAYOUT.padding * 4)

    this.size = [width, height] as Size
  }

  /**
   * Custom background rendering with rounded corners and gradient.
   */
  override onDrawBackground(ctx: CanvasRenderingContext2D): void {
    if (this.flags?.collapsed) return

    const [width, height] = this.size
    const radius = LAYOUT.cornerRadius

    // Draw rounded rectangle background
    ctx.save()
    drawRoundRect(ctx, 0, 0, width, height, radius)
    ctx.fillStyle = RESOURCE_COLORS.bgLight
    ctx.fill()

    // Draw left accent border (thicker for resources)
    ctx.fillStyle = RESOURCE_COLORS.primary
    drawRoundRect(ctx, 0, 0, 4, height, { tl: radius, bl: radius, tr: 0, br: 0 })
    ctx.fill()

    // Draw subtle bottom border
    ctx.strokeStyle = RESOURCE_COLORS.primaryLight
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(radius, height)
    ctx.lineTo(width - radius, height)
    ctx.stroke()

    ctx.restore()
  }

  /**
   * Custom foreground rendering with header, fields, and badges.
   */
  override onDrawForeground(ctx: CanvasRenderingContext2D): void {
    if (this.flags?.collapsed) return

    const [width] = this.size

    // Draw header with gradient
    this.drawHeader(ctx, width)

    // Draw singleton badge if applicable
    if (this.isSingleton) {
      this.drawSingletonBadge(ctx, width)
    }

    // Draw fields
    this.drawFields(ctx)

    // Draw source file indicator
    if (this.sourceFile) {
      this.drawSourceIndicator(ctx)
    }
  }

  /**
   * Draw the header with gradient background and icon.
   */
  private drawHeader(ctx: CanvasRenderingContext2D, width: number): void {
    const headerHeight = LAYOUT.headerHeight
    const radius = LAYOUT.cornerRadius

    ctx.save()

    // Create gradient for header
    const gradient = ctx.createLinearGradient(0, 0, width, 0)
    gradient.addColorStop(0, RESOURCE_COLORS.primary)
    gradient.addColorStop(1, RESOURCE_COLORS.primaryDark)

    // Draw rounded header background
    drawRoundRect(ctx, 0, 0, width, headerHeight, { tl: radius, tr: radius, bl: 0, br: 0 })
    ctx.fillStyle = gradient
    ctx.fill()

    // Draw "@resource" icon
    ctx.fillStyle = RESOURCE_COLORS.textWhite
    ctx.font = FONTS.icon
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'
    ctx.fillText('@', LAYOUT.padding, headerHeight / 2)

    // Draw resource name
    ctx.font = FONTS.header
    const iconOffset = LAYOUT.padding + LAYOUT.iconSize + 4
    const maxTextWidth = this.isSingleton ? width - iconOffset - 36 : width - iconOffset - LAYOUT.padding
    const displayName = sharedTruncateText(ctx, this.resourceName, maxTextWidth)
    ctx.fillText(displayName, iconOffset, headerHeight / 2)

    ctx.restore()
  }

  /**
   * Draw the singleton badge indicator.
   */
  private drawSingletonBadge(ctx: CanvasRenderingContext2D, width: number): void {
    const badge = LAYOUT.singletonBadge
    const x = width - badge.width - 6
    const y = (LAYOUT.headerHeight - badge.height) / 2

    ctx.save()

    // Badge background
    ctx.fillStyle = RESOURCE_COLORS.singletonBg
    drawRoundRect(ctx, x, y, badge.width, badge.height, badge.radius)
    ctx.fill()

    // Badge text "S"
    ctx.fillStyle = RESOURCE_COLORS.textWhite
    ctx.font = FONTS.badge
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText('S', x + badge.width / 2, y + badge.height / 2)

    ctx.restore()
  }

  /**
   * Draw the field list.
   */
  private drawFields(ctx: CanvasRenderingContext2D): void {
    if (!this.fields || this.fields.length === 0) return

    ctx.save()

    const startY = LAYOUT.headerHeight + LAYOUT.padding + 4
    const [width] = this.size

    this.fields.forEach((field, index) => {
      const y = startY + index * LAYOUT.fieldHeight

      // Field name
      ctx.font = FONTS.field
      ctx.fillStyle = RESOURCE_COLORS.textField
      ctx.textAlign = 'left'
      ctx.textBaseline = 'top'
      ctx.fillText(field.name, LAYOUT.padding, y)

      // Field type
      const nameWidth = ctx.measureText(field.name).width
      ctx.font = FONTS.fieldType
      ctx.fillStyle = RESOURCE_COLORS.textType
      ctx.fillText(`: ${field.type}`, LAYOUT.padding + nameWidth, y)

      // Default value (if present)
      if (field.default !== undefined) {
        const typeText = `: ${field.type}`
        const typeWidth = ctx.measureText(typeText).width
        ctx.fillStyle = RESOURCE_COLORS.textMuted
        const defaultText = ` = ${field.default}`
        const maxWidth = width - LAYOUT.padding * 2 - nameWidth - typeWidth - 10
        const truncatedDefault = sharedTruncateText(ctx, defaultText, maxWidth)
        ctx.fillText(truncatedDefault, LAYOUT.padding + nameWidth + typeWidth, y)
      }
    })

    ctx.restore()
  }

  /**
   * Draw the source file indicator at the bottom.
   */
  private drawSourceIndicator(ctx: CanvasRenderingContext2D): void {
    if (!this.sourceFile) return

    ctx.save()

    const [, height] = this.size
    const sourceText = this.sourceLine
      ? `${this.sourceFile}:${this.sourceLine}`
      : this.sourceFile

    // Truncate if too long
    ctx.font = FONTS.source
    const truncated =
      sourceText.length > 35 ? '...' + sourceText.slice(-32) : sourceText

    ctx.fillStyle = RESOURCE_COLORS.textMuted
    ctx.textAlign = 'left'
    ctx.textBaseline = 'bottom'
    ctx.fillText(truncated, LAYOUT.padding, height - 4)

    ctx.restore()
  }


  /**
   * Serialize node data for persistence.
   */
  override serialize(): ReturnType<LGraphNode['serialize']> {
    const data = super.serialize()
    return {
      ...data,
      properties: {
        ...data.properties,
        resourceName: this.resourceName,
        fields: this.fields,
        isSingleton: this.isSingleton,
        docstring: this.docstring,
        sourceFile: this.sourceFile,
        sourceLine: this.sourceLine,
      },
    }
  }

  /**
   * Configure node from serialized data.
   */
  override configure(info: Parameters<LGraphNode['configure']>[0]): void {
    super.configure(info)

    const props = info.properties as Record<string, unknown> | undefined
    if (props) {
      if (typeof props['resourceName'] === 'string') this.resourceName = props['resourceName']
      if (Array.isArray(props['fields'])) this.fields = props['fields'] as ResourceField[]
      if (typeof props['isSingleton'] === 'boolean') this.isSingleton = props['isSingleton']
      if (typeof props['docstring'] === 'string') this.docstring = props['docstring']
      if (typeof props['sourceFile'] === 'string') this.sourceFile = props['sourceFile']
      if (typeof props['sourceLine'] === 'number') this.sourceLine = props['sourceLine']
    }

    this.updateSize()
  }

  /**
   * Handle node creation to ensure proper initialization.
   */
  override onNodeCreated(): void {
    // Resources only provide output (they are read by systems)
    if (!this.outputs || this.outputs.length === 0) {
      this.addOutput('data', 'resource')
    }
  }

  /**
   * Handle mouse down for source navigation.
   * Single click on the source info area navigates to source.
   */
  onMouseDown(_e: MouseEvent, localPos: [number, number], _canvas: unknown): boolean {
    // Check if clicking on source info area (bottom of node)
    if (this.sourceFile) {
      const [, y] = localPos
      const [, height] = this.size
      const sourceAreaTop = height - LAYOUT.sourceHeight - LAYOUT.padding

      if (y >= sourceAreaTop) {
        // Dispatch navigation event
        const event = new CustomEvent('flowforge:navigate-to-source', {
          detail: {
            file: this.sourceFile,
            line: this.sourceLine || 1,
          },
        })
        window.dispatchEvent(event)
        return true
      }
    }

    return false
  }

  /**
   * Handle double-click to open source file in editor.
   */
  onDblClick(_e: MouseEvent, _localPos: [number, number], _canvas: unknown): boolean {
    // If source exists, open in editor
    if (this.sourceFile) {
      const event = new CustomEvent('flowforge:navigate-to-source', {
        detail: {
          file: this.sourceFile,
          line: this.sourceLine || 1,
        },
      })
      window.dispatchEvent(event)
      return true
    }
    return false
  }

  /**
   * Get the tooltip for this node.
   */
  override getExtraMenuOptions(): { content: string; callback: () => void }[] {
    const options: { content: string; callback: () => void }[] = []

    if (this.docstring) {
      options.push({
        content: `Docs: ${this.docstring.slice(0, 50)}${this.docstring.length > 50 ? '...' : ''}`,
        callback: () => {
          console.log('Resource docstring:', this.docstring)
        },
      })
    }

    if (this.sourceFile) {
      options.push({
        content: `Go to source: ${this.sourceFile}:${this.sourceLine || 1}`,
        callback: () => {
          // Emit custom event for IDE integration
          const event = new CustomEvent('flowforge:navigate-to-source', {
            detail: { file: this.sourceFile, line: this.sourceLine || 1 },
          })
          window.dispatchEvent(event)
        },
      })
    }

    return options
  }
}

// =============================================================================
// REGISTRATION
// =============================================================================

/**
 * Register the ResourceNode with LiteGraph.
 * Call this during application initialization.
 */
export function registerResourceNode(): void {
  LiteGraph.registerNodeType(NODE_TYPE_NAMES.resource, ResourceNode)
  console.log('[FlowForge] ResourceNode registered as', NODE_TYPE_NAMES.resource)
}

/**
 * Unregister the ResourceNode from LiteGraph.
 */
export function unregisterResourceNode(): void {
  LiteGraph.unregisterNodeType(NODE_TYPE_NAMES.resource)
  console.log('[FlowForge] ResourceNode unregistered')
}

/**
 * Create a ResourceNode from parsed Python data.
 * Convenience function for the node factory.
 */
export function createResourceNode(data: ResourceNodeData): ResourceNode {
  const node = LiteGraph.createNode(NODE_TYPE_NAMES.resource) as ResourceNode
  if (node) {
    node.configureFromData(data)
  }
  return node
}

export default ResourceNode
