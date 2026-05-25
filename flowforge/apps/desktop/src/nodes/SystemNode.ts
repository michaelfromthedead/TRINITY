/**
 * System Node Renderer for FlowForge
 *
 * Custom LiteGraph node renderer for Trinity ECS System nodes.
 * Systems are game logic processors that query entities with specific components.
 *
 * Visual characteristics:
 * - Green gradient header (#22C55E to #16A34A)
 * - White text on header
 * - Light green background (#F0FDF4)
 * - Query types displayed prominently as input connections
 * - Collapsible methods section
 */

import { LGraphNode, LiteGraph, LGraphCanvas } from '../litegraph'
import type { Size, INodeInputSlot, INodeOutputSlot } from '../litegraph'
import {
  TRINITY_COLORS,
  NODE_LAYOUT,
  NODE_FONTS,
  NODE_TYPE_NAMES,
  truncateText as sharedTruncateText,
} from './nodeTheme'

// =============================================================================
// COLOR CONSTANTS (re-exported for external use, built from shared theme)
// =============================================================================

/** System node color palette - Trinity green theme */
export const SYSTEM_COLORS = {
  /** Primary green color */
  primary: TRINITY_COLORS.system.primary,
  /** Darker green for gradients */
  primaryDark: TRINITY_COLORS.system.primaryDark,
  /** Light green for backgrounds */
  bgLight: TRINITY_COLORS.system.background,
  /** Background with transparency */
  bgTransparent: 'rgba(240, 253, 244, 0.95)',
  /** Header text color */
  headerText: TRINITY_COLORS.neutral.white,
  /** Body text color */
  bodyText: '#166534',
  /** Muted text color */
  mutedText: TRINITY_COLORS.system.border,
  /** Query badge background */
  queryBadge: 'rgba(34, 197, 94, 0.2)',
  /** Query badge border */
  queryBorder: TRINITY_COLORS.system.primary,
  /** Method text color */
  methodText: '#15803D',
  /** Input slot color */
  inputSlot: TRINITY_COLORS.system.primaryDark,
  /** Output slot color */
  outputSlot: TRINITY_COLORS.system.primary,
  /** Separator line color */
  separator: 'rgba(34, 197, 94, 0.3)',
} as const

// =============================================================================
// LAYOUT CONSTANTS (re-exported for external use, built from shared layout)
// =============================================================================

/** Layout measurements for system nodes */
export const SYSTEM_LAYOUT = {
  /** Header height with gradient */
  headerHeight: NODE_LAYOUT.headerHeight,
  /** Icon size in header */
  iconSize: NODE_LAYOUT.iconSize,
  /** Padding inside the node */
  padding: NODE_LAYOUT.padding,
  /** Vertical spacing between sections */
  sectionSpacing: 8,
  /** Height of query badge */
  queryBadgeHeight: 24,
  /** Spacing between query badges */
  querySpacing: 4,
  /** Height of each method row */
  methodRowHeight: 18,
  /** Height of collapsible section header */
  sectionHeaderHeight: NODE_LAYOUT.sectionHeaderHeight,
  /** Slot height */
  slotHeight: NODE_LAYOUT.slotHeight,
  /** Slot radius */
  slotRadius: NODE_LAYOUT.slotRadius,
  /** Source file indicator height */
  sourceHeight: 14,
  /** Minimum node width */
  minWidth: 240,
  /** Border radius */
  borderRadius: NODE_LAYOUT.borderRadius,
} as const

// =============================================================================
// FONTS (re-exported for external use, built from shared fonts)
// =============================================================================

/** Font definitions for system nodes */
export const SYSTEM_FONTS = {
  /** Header title font */
  header: NODE_FONTS.header,
  /** Section header font */
  sectionHeader: NODE_FONTS.sectionHeader,
  /** Query type font */
  query: NODE_FONTS.method,
  /** Method signature font */
  method: NODE_FONTS.method,
  /** Parameter type font */
  paramType: NODE_FONTS.fieldType,
  /** Source file font */
  source: NODE_FONTS.footer,
  /** Icon font (for @system icon) */
  icon: NODE_FONTS.icon,
} as const

// =============================================================================
// DATA INTERFACE
// =============================================================================

/**
 * System node data structure from Python parser.
 * This interface defines what data the node accepts.
 */
export interface SystemNodeData {
  /** System class name */
  name: string
  /** Methods defined in the system */
  methods: Array<{
    name: string
    parameters: Array<{ name: string; type: string }>
    return_type?: string
    /** Components queried via Query[...] */
    query_types?: string[]
  }>
  /** Quick reference to all query types */
  queries: string[]
  /** System fields/properties */
  fields?: Array<{ name: string; type: string; default?: string }>
  /** Documentation string */
  docstring?: string
  /** Source location */
  source?: { file: string; line: number }
}

/**
 * Method definition with full signature
 */
export interface SystemMethod {
  name: string
  parameters: Array<{ name: string; type: string }>
  returnType?: string
  queryTypes?: string[]
}

// =============================================================================
// SYSTEM NODE CLASS
// =============================================================================

/**
 * Custom System Node for Trinity ECS visualization.
 *
 * Features:
 * - Green gradient header with @system icon
 * - Query visualization showing which components are queried
 * - Method list with parameters and return types
 * - Input slots for queried component types
 * - Output slots for side effects/events
 *
 * @example
 * ```typescript
 * const node = new SystemNodeRenderer('MovementSystem')
 * node.configure({
 *   queries: ['Position', 'Velocity'],
 *   methods: [{
 *     name: 'update',
 *     parameters: [{name: 'dt', type: 'float'}],
 *     return_type: 'None',
 *     query_types: ['Position', 'Velocity']
 *   }]
 * })
 * graph.add(node)
 * ```
 */
export class SystemNodeRenderer extends LGraphNode {
  static title = 'System'
  static desc = 'A Trinity ECS system that processes entities with specific components'
  static category = 'trinity/systems'

  // System-specific data
  systemName: string = 'System'
  methods: SystemMethod[] = []
  queries: string[] = []
  fields: Array<{ name: string; type: string; default?: string }> = []
  docstring?: string
  sourceFile?: string
  sourceLine?: number

  // UI state
  methodsCollapsed: boolean = false

  constructor(title?: string) {
    super(title || 'System')
    this.systemName = title || 'System'

    // Set Trinity green theme colors
    this.color = SYSTEM_COLORS.primary
    this.bgcolor = SYSTEM_COLORS.bgTransparent

    // Initial size
    this.size = [SYSTEM_LAYOUT.minWidth, 100] as Size

    // Enable custom rendering
    this.flags = this.flags || {}
  }

  /**
   * Called when the node is first created.
   * Sets up default slots.
   */
  override onNodeCreated(): void {
    // Add default execution flow slots
    this.addInput('trigger', 'exec')
    this.addOutput('next', 'exec')
    this.addOutput('events', 'event')
  }

  /**
   * Configure the node from SystemNodeData.
   *
   * @param data - System data from Python parser
   */
  configureFromData(data: SystemNodeData): void {
    this.systemName = data.name
    this.title = data.name

    // Store queries
    this.queries = data.queries || []

    // Convert methods
    this.methods = (data.methods || []).map((m): SystemMethod => {
      const method: SystemMethod = {
        name: m.name,
        parameters: m.parameters || [],
      }
      if (m.return_type !== undefined) {
        method.returnType = m.return_type
      }
      if (m.query_types !== undefined) {
        method.queryTypes = m.query_types
      }
      return method
    })

    // Store fields
    this.fields = data.fields || []

    // Store metadata
    if (data.docstring !== undefined) {
      this.docstring = data.docstring
    }
    if (data.source) {
      this.sourceFile = data.source.file
      this.sourceLine = data.source.line
    }

    // Create input slots for each query type
    this.setupQueryInputs()

    // Recalculate size
    this.updateSize()
  }

  /**
   * Set up input slots for query types.
   */
  private setupQueryInputs(): void {
    // Remove existing query inputs (keep trigger)
    const triggerInput = this.inputs?.find((i) => i.name === 'trigger')

    // Clear inputs and re-add trigger
    this.inputs = []
    if (triggerInput) {
      this.inputs.push(triggerInput)
    } else {
      this.addInput('trigger', 'exec')
    }

    // Add input for each query type
    this.queries.forEach((queryType) => {
      this.addInput(queryType, 'query')
    })
  }

  /**
   * Calculate and update the node size based on content.
   */
  private updateSize(): void {
    const height = this.calculateHeight()
    this.size = [Math.max(this.size[0], SYSTEM_LAYOUT.minWidth), height] as Size
  }

  /**
   * Calculate required node height based on content.
   */
  private calculateHeight(): number {
    const { headerHeight, padding, sectionSpacing, queryBadgeHeight, querySpacing, methodRowHeight, sectionHeaderHeight, sourceHeight, slotHeight } =
      SYSTEM_LAYOUT

    let height = headerHeight + padding

    // Query badges section
    if (this.queries.length > 0) {
      height += sectionHeaderHeight
      height += this.queries.length * (queryBadgeHeight + querySpacing)
      height += sectionSpacing
    }

    // Slots section (max of inputs/outputs)
    const maxSlots = Math.max(this.inputs?.length || 0, this.outputs?.length || 0)
    if (maxSlots > 0) {
      height += maxSlots * slotHeight
      height += sectionSpacing
    }

    // Methods section (if not collapsed)
    if (this.methods.length > 0) {
      height += sectionHeaderHeight
      if (!this.methodsCollapsed) {
        height += this.methods.length * methodRowHeight
      }
      height += sectionSpacing
    }

    // Source file indicator
    if (this.sourceFile) {
      height += sourceHeight + padding
    }

    return Math.max(height, 80)
  }

  /**
   * Custom background rendering.
   * Draws the node body with rounded corners and subtle effects.
   */
  override onDrawBackground(ctx: CanvasRenderingContext2D): void {
    if (this.flags?.collapsed) return

    const [width, height] = this.size
    const { borderRadius } = SYSTEM_LAYOUT

    // Draw main background with rounded corners
    ctx.save()
    ctx.beginPath()
    ctx.roundRect(0, 0, width, height, borderRadius)
    ctx.fillStyle = SYSTEM_COLORS.bgLight
    ctx.fill()

    // Draw subtle border
    ctx.strokeStyle = SYSTEM_COLORS.separator
    ctx.lineWidth = 1
    ctx.stroke()

    // Draw left accent bar
    ctx.beginPath()
    ctx.roundRect(0, 0, 4, height, [borderRadius, 0, 0, borderRadius])
    ctx.fillStyle = SYSTEM_COLORS.primary
    ctx.fill()

    ctx.restore()
  }

  /**
   * Custom foreground rendering.
   * Draws header, queries, methods, and other UI elements.
   */
  override onDrawForeground(ctx: CanvasRenderingContext2D): void {
    if (this.flags?.collapsed) {
      this.drawCollapsedView(ctx)
      return
    }

    const [width] = this.size

    // Draw header with gradient
    this.drawHeader(ctx, width)

    // Current Y position for content
    let y = SYSTEM_LAYOUT.headerHeight + SYSTEM_LAYOUT.padding

    // Draw query badges
    if (this.queries.length > 0) {
      y = this.drawQuerySection(ctx, y, width)
    }

    // Draw methods section
    if (this.methods.length > 0) {
      y = this.drawMethodsSection(ctx, y, width)
    }

    // Draw source file indicator
    if (this.sourceFile) {
      this.drawSourceIndicator(ctx)
    }
  }

  /**
   * Draw the header with gradient and system icon.
   */
  private drawHeader(ctx: CanvasRenderingContext2D, width: number): void {
    const { headerHeight, borderRadius, padding } = SYSTEM_LAYOUT

    ctx.save()

    // Create gradient
    const gradient = ctx.createLinearGradient(0, 0, width, 0)
    gradient.addColorStop(0, SYSTEM_COLORS.primary)
    gradient.addColorStop(1, SYSTEM_COLORS.primaryDark)

    // Draw header background with top rounded corners
    ctx.beginPath()
    ctx.roundRect(0, 0, width, headerHeight, [borderRadius, borderRadius, 0, 0])
    ctx.fillStyle = gradient
    ctx.fill()

    // Draw @system icon
    ctx.fillStyle = SYSTEM_COLORS.headerText
    ctx.font = SYSTEM_FONTS.icon
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'
    ctx.fillText('@', padding, headerHeight / 2)

    // Draw system class name
    ctx.font = SYSTEM_FONTS.header
    const iconWidth = ctx.measureText('@').width + 4
    const titleX = padding + iconWidth
    const maxTitleWidth = width - titleX - padding
    const title = this.truncateText(ctx, this.systemName, maxTitleWidth)
    ctx.fillText(title, titleX, headerHeight / 2)

    ctx.restore()
  }

  /**
   * Draw the query visualization section.
   */
  private drawQuerySection(ctx: CanvasRenderingContext2D, startY: number, width: number): number {
    const { padding, sectionHeaderHeight, queryBadgeHeight, querySpacing } = SYSTEM_LAYOUT

    ctx.save()

    // Section header
    ctx.fillStyle = SYSTEM_COLORS.bodyText
    ctx.font = SYSTEM_FONTS.sectionHeader
    ctx.textAlign = 'left'
    ctx.textBaseline = 'top'
    ctx.fillText('QUERIES', padding, startY)

    let y = startY + sectionHeaderHeight

    // Draw query badges
    this.queries.forEach((query) => {
      this.drawQueryBadge(ctx, query, padding, y, width - padding * 2)
      y += queryBadgeHeight + querySpacing
    })

    ctx.restore()

    return y + SYSTEM_LAYOUT.sectionSpacing
  }

  /**
   * Draw a single query badge.
   */
  private drawQueryBadge(ctx: CanvasRenderingContext2D, query: string, x: number, y: number, maxWidth: number): void {
    const { queryBadgeHeight } = SYSTEM_LAYOUT
    const badgePadding = 8

    ctx.save()

    // Measure text to size badge
    ctx.font = SYSTEM_FONTS.query
    const textWidth = ctx.measureText(`Query[${query}]`).width
    const badgeWidth = Math.min(textWidth + badgePadding * 2, maxWidth)

    // Draw badge background
    ctx.beginPath()
    ctx.roundRect(x, y, badgeWidth, queryBadgeHeight, 4)
    ctx.fillStyle = SYSTEM_COLORS.queryBadge
    ctx.fill()
    ctx.strokeStyle = SYSTEM_COLORS.queryBorder
    ctx.lineWidth = 1
    ctx.stroke()

    // Draw query text
    ctx.fillStyle = SYSTEM_COLORS.bodyText
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'
    const displayText = this.truncateText(ctx, `Query[${query}]`, badgeWidth - badgePadding * 2)
    ctx.fillText(displayText, x + badgePadding, y + queryBadgeHeight / 2)

    ctx.restore()
  }

  /**
   * Draw the methods section with collapsible header.
   */
  private drawMethodsSection(ctx: CanvasRenderingContext2D, startY: number, width: number): number {
    const { padding, sectionHeaderHeight, methodRowHeight } = SYSTEM_LAYOUT

    ctx.save()

    // Section header with collapse indicator
    ctx.fillStyle = SYSTEM_COLORS.bodyText
    ctx.font = SYSTEM_FONTS.sectionHeader
    ctx.textAlign = 'left'
    ctx.textBaseline = 'top'

    const collapseIcon = this.methodsCollapsed ? '\u25B6' : '\u25BC' // Right/Down triangle
    ctx.fillText(`${collapseIcon} METHODS (${this.methods.length})`, padding, startY)

    let y = startY + sectionHeaderHeight

    // Draw methods if not collapsed
    if (!this.methodsCollapsed) {
      ctx.font = SYSTEM_FONTS.method
      ctx.fillStyle = SYSTEM_COLORS.methodText

      this.methods.forEach((method) => {
        const signature = this.formatMethodSignature(method)
        const maxWidth = width - padding * 2
        const displayText = this.truncateText(ctx, signature, maxWidth)
        ctx.fillText(displayText, padding, y)
        y += methodRowHeight
      })
    }

    ctx.restore()

    return y + SYSTEM_LAYOUT.sectionSpacing
  }

  /**
   * Format a method signature for display.
   */
  private formatMethodSignature(method: SystemMethod): string {
    const params = method.parameters.map((p) => `${p.name}: ${p.type}`).join(', ')
    const returnType = method.returnType ? ` -> ${method.returnType}` : ''
    return `${method.name}(${params})${returnType}`
  }

  /**
   * Draw the source file indicator at the bottom.
   */
  private drawSourceIndicator(ctx: CanvasRenderingContext2D): void {
    const { padding } = SYSTEM_LAYOUT
    const [width, height] = this.size

    ctx.save()

    ctx.fillStyle = SYSTEM_COLORS.mutedText
    ctx.font = SYSTEM_FONTS.source
    ctx.textAlign = 'left'
    ctx.textBaseline = 'bottom'

    const sourceText = this.sourceLine ? `${this.sourceFile}:${this.sourceLine}` : this.sourceFile || ''

    const maxWidth = width - padding * 2
    const displayText = this.truncateText(ctx, sourceText, maxWidth)
    ctx.fillText(displayText, padding, height - padding / 2)

    ctx.restore()
  }

  /**
   * Draw collapsed view showing minimal info.
   */
  private drawCollapsedView(ctx: CanvasRenderingContext2D): void {
    const collapsedWidth = this._collapsed_width || 100

    ctx.save()

    // Draw collapsed header
    const gradient = ctx.createLinearGradient(0, 0, collapsedWidth, 0)
    gradient.addColorStop(0, SYSTEM_COLORS.primary)
    gradient.addColorStop(1, SYSTEM_COLORS.primaryDark)

    ctx.beginPath()
    ctx.roundRect(0, 0, collapsedWidth, LiteGraph.NODE_TITLE_HEIGHT, SYSTEM_LAYOUT.borderRadius)
    ctx.fillStyle = gradient
    ctx.fill()

    // Draw title
    ctx.fillStyle = SYSTEM_COLORS.headerText
    ctx.font = SYSTEM_FONTS.header
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'
    ctx.fillText('@' + this.systemName, 8, LiteGraph.NODE_TITLE_HEIGHT / 2)

    ctx.restore()
  }

  /**
   * Handle mouse click for collapsible sections and source navigation.
   */
  override onMouseDown(
    _e: MouseEvent,
    localPos: [number, number],
    _graphCanvas: LGraphCanvas
  ): boolean {
    const [x, y] = localPos
    const { padding, headerHeight, sectionHeaderHeight, queryBadgeHeight, querySpacing, sourceHeight } = SYSTEM_LAYOUT

    // Check if clicking on source info area (bottom of node)
    if (this.sourceFile) {
      const [, height] = this.size
      const sourceAreaTop = height - sourceHeight - padding

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

    // Calculate methods section Y position
    let methodsY = headerHeight + padding
    if (this.queries.length > 0) {
      methodsY += sectionHeaderHeight
      methodsY += this.queries.length * (queryBadgeHeight + querySpacing)
      methodsY += SYSTEM_LAYOUT.sectionSpacing
    }

    // Check if click is in methods header area
    if (this.methods.length > 0 && y >= methodsY && y <= methodsY + sectionHeaderHeight && x >= padding && x <= this.size[0] - padding) {
      this.methodsCollapsed = !this.methodsCollapsed
      this.updateSize()
      return true
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
          console.log('System docstring:', this.docstring)
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

  /**
   * Truncate text to fit within a given width.
   * Uses shared utility from nodeTheme.
   */
  private truncateText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string {
    return sharedTruncateText(ctx, text, maxWidth)
  }

  /**
   * Serialize node state.
   */
  override serialize(): ReturnType<LGraphNode['serialize']> {
    const data = super.serialize()
    return {
      ...data,
      properties: {
        ...data.properties,
        systemName: this.systemName,
        queries: this.queries,
        methods: this.methods,
        fields: this.fields,
        docstring: this.docstring,
        sourceFile: this.sourceFile,
        sourceLine: this.sourceLine,
        methodsCollapsed: this.methodsCollapsed,
      },
    }
  }

  /**
   * Configure from serialized data.
   */
  override configure(info: Parameters<LGraphNode['configure']>[0]): void {
    super.configure(info)

    const props = info.properties as Record<string, unknown> | undefined
    if (props) {
      if (props['systemName']) this.systemName = props['systemName'] as string
      if (props['queries']) this.queries = props['queries'] as string[]
      if (props['methods']) this.methods = props['methods'] as SystemMethod[]
      if (props['fields']) this.fields = props['fields'] as Array<{ name: string; type: string; default?: string }>
      if (props['docstring']) this.docstring = props['docstring'] as string
      if (props['sourceFile'] !== undefined) this.sourceFile = props['sourceFile'] as string
      if (props['sourceLine'] !== undefined) this.sourceLine = props['sourceLine'] as number
      if (props['methodsCollapsed'] !== undefined) this.methodsCollapsed = props['methodsCollapsed'] as boolean

      this.setupQueryInputs()
      this.updateSize()
    }
  }

  /**
   * Get rendering colors for slots.
   */
  getSlotColor(slot: INodeInputSlot | INodeOutputSlot, isInput: boolean): string {
    if (slot.type === 'query') {
      return SYSTEM_COLORS.inputSlot
    }
    if (slot.type === 'event') {
      return SYSTEM_COLORS.outputSlot
    }
    return isInput ? SYSTEM_COLORS.inputSlot : SYSTEM_COLORS.outputSlot
  }
}

// =============================================================================
// REGISTRATION
// =============================================================================

/**
 * Register the SystemNodeRenderer with LiteGraph.
 * Call this during application initialization.
 */
export function registerSystemNode(): void {
  LiteGraph.registerNodeType(NODE_TYPE_NAMES.system, SystemNodeRenderer)

  // Set up node colors in canvas
  LGraphCanvas.node_colors['trinity_system'] = {
    color: SYSTEM_COLORS.primary,
    bgcolor: SYSTEM_COLORS.bgTransparent,
    groupcolor: SYSTEM_COLORS.primaryDark,
  }

  console.log('[FlowForge] SystemNodeRenderer registered as', NODE_TYPE_NAMES.system)
}

/**
 * Unregister the SystemNodeRenderer.
 */
export function unregisterSystemNode(): void {
  LiteGraph.unregisterNodeType(NODE_TYPE_NAMES.system)
  console.log('[FlowForge] SystemNodeRenderer unregistered')
}

/**
 * Create a SystemNodeRenderer from parsed Python data.
 *
 * @param data - System data from Python parser
 * @returns Configured SystemNodeRenderer instance
 */
export function createSystemNode(data: SystemNodeData): SystemNodeRenderer {
  const node = new SystemNodeRenderer(data.name)
  node.configureFromData(data)
  return node
}

// Default export
export default SystemNodeRenderer
