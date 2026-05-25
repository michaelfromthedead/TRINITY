/**
 * Event Node Renderer for FlowForge
 *
 * Custom LiteGraph node for visualizing Trinity ECS Event patterns.
 * Features:
 * - Orange gradient header (#F97316 to #EA580C)
 * - Lightning bolt icon indicating event type
 * - Payload fields section showing event data
 * - Distinctive hexagonal/angled corner shape
 */

import { LGraphNode, LiteGraph } from '../litegraph'
import type { Size } from '../litegraph'
import {
  TRINITY_COLORS,
  NODE_LAYOUT,
  NODE_FONTS,
  NODE_TYPE_NAMES,
} from './nodeTheme'

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================

/**
 * Data structure received from Python parser for Event nodes.
 */
export interface EventNodeData {
  name: string
  payload_fields: Array<{
    name: string
    type: string
    default?: string
  }>
  docstring?: string
  source?: {
    file: string
    line: number
  }
}

/**
 * Internal field representation for payload display.
 */
export interface EventPayloadField {
  name: string
  type: string
  defaultValue?: string
  description?: string
}

// =============================================================================
// COLOR CONSTANTS (re-exported for external use, built from shared theme)
// =============================================================================

/** Event node theme colors */
export const EVENT_COLORS = {
  /** Primary orange color */
  primary: TRINITY_COLORS.event.primary,
  /** Darker orange for gradient end */
  primaryDark: TRINITY_COLORS.event.primaryDark,
  /** Light orange for accents */
  primaryLight: TRINITY_COLORS.event.border,
  /** Very light orange for background */
  background: TRINITY_COLORS.event.background,
  /** Background with alpha for node body */
  backgroundAlpha: 'rgba(255, 247, 237, 0.95)',
  /** Border color */
  border: TRINITY_COLORS.event.primary,
  /** Text on header (white) */
  headerText: TRINITY_COLORS.neutral.white,
  /** Muted text for labels */
  mutedText: TRINITY_COLORS.neutral.textLight,
  /** Field name text color */
  fieldText: TRINITY_COLORS.neutral.textSecondary,
  /** Type annotation color */
  typeText: TRINITY_COLORS.event.primary,
} as const

/** Layout constants specific to Event nodes (using shared layout) */
const EVENT_LAYOUT = {
  /** Header height */
  headerHeight: NODE_LAYOUT.headerHeight,
  /** Icon size */
  iconSize: NODE_LAYOUT.iconSize,
  /** Corner angle size for hexagonal shape */
  cornerAngle: 8,
  /** Padding inside the node */
  padding: NODE_LAYOUT.padding,
  /** Height per payload field row */
  fieldRowHeight: NODE_LAYOUT.fieldRowHeight,
  /** Section label height */
  sectionLabelHeight: NODE_LAYOUT.sectionHeaderHeight,
  /** Slot row height */
  slotHeight: NODE_LAYOUT.slotHeight,
  /** Source indicator height */
  sourceHeight: NODE_LAYOUT.footerHeight,
  /** Minimum node width */
  minWidth: NODE_LAYOUT.minWidth,
  /** Icon margin from left */
  iconMarginLeft: 10,
  /** Text margin from icon */
  textMarginLeft: 32,
} as const

/** Font constants for Event nodes (using shared fonts) */
const EVENT_FONTS = {
  /** Label font for section headers */
  label: NODE_FONTS.sectionHeader,
  /** Small label for source file indicator */
  labelSmall: NODE_FONTS.footer,
  /** Monospace font for code/fields */
  code: NODE_FONTS.method,
} as const

/** Text color constants for Event nodes */
const EVENT_TEXT_COLORS = {
  /** Source file indicator color */
  sourceFile: TRINITY_COLORS.neutral.textMuted,
} as const

// =============================================================================
// EVENT NODE CLASS
// =============================================================================

/**
 * Enhanced Event Node with Trinity orange theme and distinctive styling.
 *
 * Renders events with:
 * - Gradient orange header
 * - Lightning bolt icon
 * - Payload fields section
 * - Angled/hexagonal corners
 *
 * @example
 * ```typescript
 * const eventNode = new EventNode('PlayerDied')
 * eventNode.setFromData({
 *   name: 'PlayerDied',
 *   payload_fields: [
 *     { name: 'player_id', type: 'int' },
 *     { name: 'cause', type: 'str' }
 *   ],
 *   docstring: 'Fired when a player dies',
 *   source: { file: 'events.py', line: 42 }
 * })
 * graph.add(eventNode)
 * ```
 */
export class EventNode extends LGraphNode {
  static title = 'Event'
  static desc = 'An event that can be triggered and listened to'
  static category = 'trinity'

  // Node type identifier
  readonly trinityType = 'event' as const

  // Event-specific data
  eventName: string = ''
  payloadFields: EventPayloadField[] = []
  docstring?: string
  sourceFile?: string
  sourceLine?: number

  constructor(title?: string) {
    super(title || 'Event', 'trinity/event')

    // Set default colors from Trinity theme
    this.color = EVENT_COLORS.primary
    this.bgcolor = EVENT_COLORS.backgroundAlpha

    // Set initial size
    this.size = [EVENT_LAYOUT.minWidth, 100] as Size

    // Store the event name
    this.eventName = title || 'Event'
  }

  /**
   * Called when the node is created.
   * Sets up default input/output slots.
   */
  override onNodeCreated(): void {
    // Input: event trigger source
    this.addInput('trigger', 'exec')

    // Output: event signal for handlers
    this.addOutput('signal', 'event')
  }

  /**
   * Configure the node from EventNodeData (Python parser output).
   */
  setFromData(data: EventNodeData): void {
    this.eventName = data.name
    this.title = data.name

    // Convert payload fields (handle optional defaultValue)
    this.payloadFields = data.payload_fields.map((field): EventPayloadField => ({
      name: field.name,
      type: field.type,
      ...(field.default !== undefined && { defaultValue: field.default }),
    }))

    // Set optional metadata (only set if defined)
    if (data.docstring !== undefined) {
      this.docstring = data.docstring
    }
    if (data.source) {
      this.sourceFile = data.source.file
      this.sourceLine = data.source.line
    }

    // Recalculate size based on content
    this.updateSize()
  }

  /**
   * Set payload fields directly.
   */
  setPayloadFields(fields: EventPayloadField[]): void {
    this.payloadFields = fields
    this.updateSize()
  }

  /**
   * Calculate and update node size based on content.
   */
  private updateSize(): void {
    const {
      headerHeight,
      padding,
      fieldRowHeight,
      sectionLabelHeight,
      slotHeight,
      sourceHeight,
    } = EVENT_LAYOUT

    // Calculate height components
    const inputSlots = this.inputs?.length || 0
    const outputSlots = this.outputs?.length || 0
    const maxSlots = Math.max(inputSlots, outputSlots)
    const slotsHeight = maxSlots * slotHeight

    // Payload section height
    const hasPayload = this.payloadFields.length > 0
    const payloadLabelHeight = hasPayload ? sectionLabelHeight : 0
    const payloadFieldsHeight = this.payloadFields.length * fieldRowHeight

    // Source indicator height
    const sourceIndicatorHeight = this.sourceFile ? sourceHeight : 0

    // Total height calculation
    const totalHeight =
      headerHeight +
      padding +
      slotsHeight +
      payloadLabelHeight +
      payloadFieldsHeight +
      padding +
      sourceIndicatorHeight

    // Calculate width based on content
    let maxWidth = EVENT_LAYOUT.minWidth

    // Check event name width
    const titleWidth = this.eventName.length * 8 + EVENT_LAYOUT.textMarginLeft + 20
    maxWidth = Math.max(maxWidth, titleWidth)

    // Check payload field widths
    this.payloadFields.forEach(field => {
      const fieldWidth = (field.name.length + field.type.length + 4) * 7 + padding * 2
      maxWidth = Math.max(maxWidth, fieldWidth)
    })

    this.size = [maxWidth, totalHeight] as Size
  }

  /**
   * Get the rendering color (primary orange).
   */
  override get renderingColor(): string {
    return this.color || EVENT_COLORS.primary
  }

  /**
   * Get the background color (light orange).
   */
  override get renderingBgColor(): string {
    return this.bgcolor || EVENT_COLORS.backgroundAlpha
  }

  /**
   * Get the box outline color.
   */
  override get renderingBoxColor(): string {
    return this.boxcolor || EVENT_COLORS.primary
  }

  /**
   * Draw the node background with distinctive shape.
   */
  override onDrawBackground(ctx: CanvasRenderingContext2D): void {
    if (this.flags?.collapsed) return

    const [width, height] = this.size
    const { headerHeight, cornerAngle } = EVENT_LAYOUT

    ctx.save()

    // Draw angled/hexagonal body shape
    this.drawHexagonalShape(ctx, width, height, cornerAngle)

    // Fill with light orange background
    ctx.fillStyle = EVENT_COLORS.backgroundAlpha
    ctx.fill()

    // Draw left accent border
    ctx.fillStyle = EVENT_COLORS.primary
    ctx.fillRect(0, headerHeight, 3, height - headerHeight)

    ctx.restore()
  }

  /**
   * Draw the node foreground (header, icon, fields).
   */
  override onDrawForeground(ctx: CanvasRenderingContext2D): void {
    if (this.flags?.collapsed) return

    const [width] = this.size
    const {
      headerHeight,
      cornerAngle,
      padding,
      iconMarginLeft,
      textMarginLeft,
      slotHeight,
      sectionLabelHeight,
      fieldRowHeight,
    } = EVENT_LAYOUT

    ctx.save()

    // Draw gradient header
    this.drawHeader(ctx, width, headerHeight, cornerAngle)

    // Draw lightning bolt icon
    this.drawLightningIcon(ctx, iconMarginLeft + 4, headerHeight / 2)

    // Draw "@event" prefix and event name
    ctx.fillStyle = EVENT_COLORS.headerText
    ctx.font = 'bold 11px Arial'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'

    // Draw @event prefix in slightly smaller font
    ctx.font = '10px Arial'
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
    ctx.fillText('@event', textMarginLeft, headerHeight / 2 - 6)

    // Draw event name
    ctx.font = 'bold 12px Arial'
    ctx.fillStyle = EVENT_COLORS.headerText
    ctx.fillText(this.eventName, textMarginLeft, headerHeight / 2 + 6)

    // Calculate content start position (after slots)
    const inputSlots = this.inputs?.length || 0
    const outputSlots = this.outputs?.length || 0
    const maxSlots = Math.max(inputSlots, outputSlots)
    let yOffset = headerHeight + padding + (maxSlots * slotHeight)

    // Draw payload section
    if (this.payloadFields.length > 0) {
      // Section label
      ctx.fillStyle = EVENT_COLORS.mutedText
      ctx.font = EVENT_FONTS.label
      ctx.textAlign = 'left'
      ctx.fillText('Payload:', padding, yOffset)
      yOffset += sectionLabelHeight

      // Draw each payload field
      ctx.font = EVENT_FONTS.code
      this.payloadFields.forEach((field, index) => {
        const fieldY = yOffset + (index * fieldRowHeight)

        // Field name
        ctx.fillStyle = EVENT_COLORS.fieldText
        ctx.fillText(field.name, padding + 4, fieldY)

        // Type annotation
        const nameWidth = ctx.measureText(field.name).width
        ctx.fillStyle = EVENT_COLORS.typeText
        ctx.fillText(`: ${field.type}`, padding + 4 + nameWidth, fieldY)

        // Default value (if present)
        if (field.defaultValue) {
          const typeWidth = ctx.measureText(`: ${field.type}`).width
          ctx.fillStyle = EVENT_COLORS.mutedText
          ctx.fillText(
            ` = ${field.defaultValue}`,
            padding + 4 + nameWidth + typeWidth,
            fieldY
          )
        }
      })
    }

    // Draw source file indicator
    if (this.sourceFile) {
      ctx.fillStyle = EVENT_TEXT_COLORS.sourceFile
      ctx.font = EVENT_FONTS.labelSmall
      ctx.textAlign = 'left'
      const sourceText = this.sourceLine
        ? `${this.sourceFile}:${this.sourceLine}`
        : this.sourceFile
      const truncated =
        sourceText.length > 35 ? '...' + sourceText.slice(-32) : sourceText
      ctx.fillText(truncated, 8, this.size[1] - 6)
    }

    ctx.restore()
  }

  /**
   * Draw the hexagonal/angled shape path.
   */
  private drawHexagonalShape(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    angle: number
  ): void {
    ctx.beginPath()

    // Start from top-left corner (with angle cut)
    ctx.moveTo(angle, 0)

    // Top edge
    ctx.lineTo(width - angle, 0)

    // Top-right corner (angled)
    ctx.lineTo(width, angle)

    // Right edge
    ctx.lineTo(width, height - angle)

    // Bottom-right corner (angled)
    ctx.lineTo(width - angle, height)

    // Bottom edge
    ctx.lineTo(angle, height)

    // Bottom-left corner (angled)
    ctx.lineTo(0, height - angle)

    // Left edge
    ctx.lineTo(0, angle)

    // Close to top-left
    ctx.closePath()
  }

  /**
   * Draw the gradient header with angled corners.
   */
  private drawHeader(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    angle: number
  ): void {
    // Create gradient
    const gradient = ctx.createLinearGradient(0, 0, width, 0)
    gradient.addColorStop(0, EVENT_COLORS.primary)
    gradient.addColorStop(1, EVENT_COLORS.primaryDark)

    ctx.beginPath()

    // Draw header shape (top portion with angled corners)
    ctx.moveTo(angle, 0)
    ctx.lineTo(width - angle, 0)
    ctx.lineTo(width, angle)
    ctx.lineTo(width, height)
    ctx.lineTo(0, height)
    ctx.lineTo(0, angle)
    ctx.closePath()

    ctx.fillStyle = gradient
    ctx.fill()

    // Add subtle bottom shadow line
    ctx.strokeStyle = EVENT_COLORS.primaryDark
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, height)
    ctx.lineTo(width, height)
    ctx.stroke()
  }

  /**
   * Draw the lightning bolt icon.
   */
  private drawLightningIcon(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number
  ): void {
    const size = EVENT_LAYOUT.iconSize

    ctx.save()
    ctx.translate(x, y - size / 2)

    // Lightning bolt path
    ctx.beginPath()
    ctx.fillStyle = EVENT_COLORS.headerText

    // Draw lightning bolt shape
    const scale = size / 16 // Base size is 16px

    ctx.moveTo(9 * scale, 0)
    ctx.lineTo(4 * scale, 7 * scale)
    ctx.lineTo(7 * scale, 7 * scale)
    ctx.lineTo(5 * scale, 16 * scale)
    ctx.lineTo(12 * scale, 6 * scale)
    ctx.lineTo(9 * scale, 6 * scale)
    ctx.lineTo(11 * scale, 0)
    ctx.closePath()

    ctx.fill()

    // Add subtle glow effect
    ctx.shadowColor = EVENT_COLORS.headerText
    ctx.shadowBlur = 4
    ctx.fill()

    ctx.restore()
  }

  /**
   * Serialize the node for storage/export.
   */
  override serialize(): ReturnType<LGraphNode['serialize']> {
    const data = super.serialize()
    return {
      ...data,
      properties: {
        ...data.properties,
        trinityType: this.trinityType,
        eventName: this.eventName,
        payloadFields: this.payloadFields,
        docstring: this.docstring,
        sourceFile: this.sourceFile,
        sourceLine: this.sourceLine,
      },
    }
  }

  /**
   * Configure the node from serialized data.
   */
  override configure(info: Parameters<LGraphNode['configure']>[0]): void {
    super.configure(info)

    const props = info.properties as Record<string, unknown> | undefined
    if (props) {
      if (typeof props['eventName'] === 'string') {
        this.eventName = props['eventName']
      }
      if (Array.isArray(props['payloadFields'])) {
        this.payloadFields = props['payloadFields'] as EventPayloadField[]
      }
      if (typeof props['docstring'] === 'string') {
        this.docstring = props['docstring']
      }
      if (typeof props['sourceFile'] === 'string') {
        this.sourceFile = props['sourceFile']
      }
      if (typeof props['sourceLine'] === 'number') {
        this.sourceLine = props['sourceLine']
      }
    }

    // Update size after loading
    this.updateSize()
  }

  /**
   * Get tooltip text for the node.
   */
  getTooltipText(): string {
    let tooltip = `Event: ${this.eventName}`
    if (this.docstring) {
      tooltip += `\n\n${this.docstring}`
    }
    if (this.payloadFields.length > 0) {
      tooltip += '\n\nPayload:'
      this.payloadFields.forEach(field => {
        tooltip += `\n  ${field.name}: ${field.type}`
        if (field.defaultValue) {
          tooltip += ` = ${field.defaultValue}`
        }
      })
    }
    if (this.sourceFile) {
      tooltip += `\n\nSource: ${this.sourceFile}`
      if (this.sourceLine) {
        tooltip += `:${this.sourceLine}`
      }
    }
    return tooltip
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
      const sourceAreaTop = height - EVENT_LAYOUT.sourceHeight - EVENT_LAYOUT.padding

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
   * Get extra context menu options including source navigation.
   */
  override getExtraMenuOptions(): { content: string; callback: () => void }[] {
    const options: { content: string; callback: () => void }[] = []

    if (this.docstring) {
      options.push({
        content: `Docs: ${this.docstring.slice(0, 50)}${this.docstring.length > 50 ? '...' : ''}`,
        callback: () => {
          console.log('Event docstring:', this.docstring)
        },
      })
    }

    if (this.sourceFile) {
      options.push({
        content: `Go to source: ${this.sourceFile}:${this.sourceLine || 1}`,
        callback: () => {
          // Dispatch navigation event for IDE integration
          const event = new CustomEvent('flowforge:navigate-to-source', {
            detail: {
              file: this.sourceFile,
              line: this.sourceLine || 1,
            },
          })
          window.dispatchEvent(event)
        },
      })
    }

    return options
  }
}

// =============================================================================
// REGISTRATION HELPERS
// =============================================================================

/**
 * Register the EventNode type with LiteGraph.
 * Should be called once during application initialization.
 */
export function registerEventNode(): void {
  LiteGraph.registerNodeType(NODE_TYPE_NAMES.event, EventNode)
  console.log('[FlowForge] EventNode registered as', NODE_TYPE_NAMES.event)
}

/**
 * Unregister the EventNode type from LiteGraph.
 */
export function unregisterEventNode(): void {
  LiteGraph.unregisterNodeType(NODE_TYPE_NAMES.event)
  console.log('[FlowForge] EventNode unregistered')
}

/**
 * Create an EventNode from Python parser data.
 *
 * @param data - The EventNodeData from the Python parser
 * @param position - Optional position [x, y] for the node
 * @returns Configured EventNode instance
 *
 * @example
 * ```typescript
 * const data = {
 *   name: 'CollisionDetected',
 *   payload_fields: [
 *     { name: 'entity_a', type: 'Entity' },
 *     { name: 'entity_b', type: 'Entity' },
 *     { name: 'point', type: 'Vec3' }
 *   ],
 *   docstring: 'Fired when two entities collide',
 *   source: { file: 'physics.py', line: 156 }
 * }
 *
 * const node = createEventNodeFromData(data, [100, 200])
 * graph.add(node)
 * ```
 */
export function createEventNodeFromData(
  data: EventNodeData,
  position?: [number, number]
): EventNode {
  const node = new EventNode(data.name)
  node.setFromData(data)

  if (position) {
    node.pos = [position[0], position[1]]
  }

  // Trigger node created callback
  if (node.onNodeCreated) {
    node.onNodeCreated()
  }

  return node
}

// Export the node class as default
export default EventNode
