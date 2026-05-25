/**
 * Custom Component Node Renderer for FlowForge
 *
 * A specialized LiteGraph node for visualizing Trinity ECS component classes
 * from Python source code. Features a blue gradient theme, expandable field
 * list, and hover tooltips.
 *
 * @module nodes/ComponentNode
 */

import { LGraphNode, LiteGraph } from '../litegraph'
import type { Size } from '../litegraph'
import {
  TRINITY_COLORS,
  NODE_LAYOUT,
  NODE_FONTS,
  NODE_TYPE_NAMES,
  isComplexType,
  truncateText,
  drawRoundRect,
} from './nodeTheme'

// =============================================================================
// COMPONENT-SPECIFIC COLORS (extending shared theme)
// =============================================================================

/** Component-specific colors (blue theme) */
const COMPONENT_COLORS = {
  ...TRINITY_COLORS.component,
  /** Field type text color (gray) */
  typeText: TRINITY_COLORS.neutral.textMuted,
  /** Field name text color (black) */
  nameText: TRINITY_COLORS.neutral.textPrimary,
  /** Default value text color */
  defaultText: TRINITY_COLORS.neutral.textLight,
  /** White text for header */
  headerText: TRINITY_COLORS.neutral.white,
  /** Header icon color */
  iconColor: '#BFDBFE',
  /** Connection slot color */
  slotColor: TRINITY_COLORS.component.accent,
} as const

// =============================================================================
// TYPES
// =============================================================================

/**
 * Data structure for component nodes received from Python parser.
 */
export interface ComponentNodeData {
  /** The class name of the component */
  name: string
  /** Array of field definitions */
  fields: ComponentField[]
  /** Optional docstring for the component */
  docstring?: string
  /** Source file information */
  source?: {
    /** The source file path */
    file: string
    /** Line number in the source file */
    line: number
  }
}

/**
 * Individual field definition within a component.
 */
export interface ComponentField {
  /** Field name */
  name: string
  /** Field type (e.g., 'float', 'Vec3', 'Entity') */
  type: string
  /** Default value as a string representation */
  default?: string
}

/**
 * Hover state tracking for tooltips.
 */
interface HoverState {
  /** Index of the field being hovered (-1 if none) */
  fieldIndex: number
  /** Position of the hover for tooltip placement */
  position: { x: number; y: number }
}

// =============================================================================
// COMPONENT NODE CLASS
// =============================================================================

/**
 * Custom Component Node for LiteGraph.
 *
 * Renders Trinity ECS components with:
 * - Blue gradient header with @component icon
 * - Field list showing name: type = default
 * - Input/output connection slots
 * - Collapse/expand for many fields
 * - Hover tooltips for field types
 *
 * @example
 * ```typescript
 * const node = new ComponentNodeRenderer('Position')
 * node.setComponentData({
 *   name: 'Position',
 *   fields: [
 *     { name: 'x', type: 'float', default: '0.0' },
 *     { name: 'y', type: 'float', default: '0.0' },
 *     { name: 'z', type: 'float', default: '0.0' }
 *   ],
 *   source: { file: 'components.py', line: 10 }
 * })
 * graph.add(node)
 * ```
 */
export class ComponentNodeRenderer extends LGraphNode {
  // Static properties for LiteGraph registration
  static title = 'Component'
  static desc = 'A Trinity ECS component class'
  static category = 'trinity/components'

  // Component data
  private componentData: ComponentNodeData = {
    name: 'Component',
    fields: [],
  }

  // UI state
  private isExpanded: boolean = true
  private hoverState: HoverState = { fieldIndex: -1, position: { x: 0, y: 0 } }

  constructor(title?: string) {
    super(title || 'Component')
    this.title = title || 'Component'

    // Set initial size
    this.size = [220, 100] as Size

    // Node appearance
    this.color = COMPONENT_COLORS.primary
    this.bgcolor = COMPONENT_COLORS.background
    this.boxcolor = COMPONENT_COLORS.primary

    // Enable mouse tracking for tooltips
    this.flags = this.flags || {}
  }

  /**
   * Called when the node is created.
   * Sets up default input/output slots.
   */
  onNodeCreated(): void {
    // Add default output slot for component data
    this.addOutput('data', 'component')
  }

  /**
   * Set the component data and update the node display.
   *
   * @param data - The component data from Python parser
   */
  setComponentData(data: ComponentNodeData): void {
    this.componentData = data
    this.title = data.name

    // Recalculate size based on content
    this.updateSize()

    // Add input slots for field dependencies
    this.updateSlots()

    // Trigger redraw
    this.setDirtyCanvas(true, true)
  }

  /**
   * Get the current component data.
   */
  getComponentData(): ComponentNodeData {
    return this.componentData
  }

  /**
   * Toggle expand/collapse state.
   */
  toggleExpand(): void {
    this.isExpanded = !this.isExpanded
    this.updateSize()
    this.setDirtyCanvas(true, true)
  }

  /**
   * Update node size based on content.
   */
  private updateSize(): void {
    const fields = this.componentData.fields
    const visibleFields = this.isExpanded
      ? fields.length
      : Math.min(fields.length, NODE_LAYOUT.maxVisibleFields)

    const contentHeight =
      NODE_LAYOUT.headerHeight +
      NODE_LAYOUT.padding +
      visibleFields * (NODE_LAYOUT.fieldRowHeight + NODE_LAYOUT.fieldGap) +
      (!this.isExpanded && fields.length > NODE_LAYOUT.maxVisibleFields
        ? NODE_LAYOUT.collapseIndicatorHeight
        : 0) +
      (this.componentData.source ? NODE_LAYOUT.footerHeight : 0) +
      NODE_LAYOUT.padding

    // Calculate width based on longest field
    let maxWidth = NODE_LAYOUT.minWidth
    for (const field of fields) {
      const fieldText = `${field.name}: ${field.type}${field.default ? ` = ${field.default}` : ''}`
      // Rough estimate: 7px per character
      const estimatedWidth = fieldText.length * 7 + NODE_LAYOUT.padding * 2 + 20
      maxWidth = Math.max(maxWidth, estimatedWidth)
    }

    this.size = [Math.min(maxWidth, NODE_LAYOUT.maxWidth), contentHeight] as Size
  }

  /**
   * Update input/output slots based on component fields.
   */
  private updateSlots(): void {
    // Clear existing inputs (keep outputs)
    while (this.inputs && this.inputs.length > 0) {
      this.removeInput(0)
    }

    // Add an input slot for entity attachment
    this.addInput('entity', 'entity')

    // For complex types, add dependency inputs
    for (const field of this.componentData.fields) {
      if (this.isComplexType(field.type)) {
        this.addInput(field.name, field.type.toLowerCase())
      }
    }
  }

  /**
   * Check if a type is complex (requires connection rather than literal).
   * Uses shared type detection from nodeTheme.
   */
  private isComplexType(type: string): boolean {
    return isComplexType(type)
  }

  /**
   * Custom background drawing for the node.
   * Renders the blue gradient header and rounded background.
   */
  onDrawBackground(ctx: CanvasRenderingContext2D): void {
    if (this.flags?.collapsed) return

    const [width, height] = this.size

    // Save context state
    ctx.save()

    // Draw rounded rectangle background
    drawRoundRect(ctx, 0, 0, width, height, NODE_LAYOUT.borderRadius)
    ctx.fillStyle = COMPONENT_COLORS.background
    ctx.fill()

    // Draw border
    ctx.strokeStyle = COMPONENT_COLORS.border
    ctx.lineWidth = 1
    ctx.stroke()

    // Draw header with gradient
    ctx.beginPath()
    drawRoundRect(ctx, 0, 0, width, NODE_LAYOUT.headerHeight, [
      NODE_LAYOUT.borderRadius,
      NODE_LAYOUT.borderRadius,
      0,
      0,
    ])
    ctx.clip()

    const gradient = ctx.createLinearGradient(0, 0, width, 0)
    gradient.addColorStop(0, COMPONENT_COLORS.primary)
    gradient.addColorStop(1, COMPONENT_COLORS.primaryDark)
    ctx.fillStyle = gradient
    ctx.fillRect(0, 0, width, NODE_LAYOUT.headerHeight)

    ctx.restore()

    // Draw left accent bar
    ctx.save()
    ctx.fillStyle = COMPONENT_COLORS.primary
    ctx.fillRect(0, NODE_LAYOUT.headerHeight, 3, height - NODE_LAYOUT.headerHeight)
    ctx.restore()
  }

  /**
   * Custom foreground drawing for the node.
   * Renders header text, fields, and source info.
   */
  onDrawForeground(ctx: CanvasRenderingContext2D): void {
    if (this.flags?.collapsed) return

    const [width, _height] = this.size
    const { headerHeight, padding, fieldRowHeight, fieldGap } = NODE_LAYOUT

    // Draw header content
    this.drawHeader(ctx, width)

    // Draw fields
    const fields = this.componentData.fields
    const visibleFields = this.isExpanded
      ? fields
      : fields.slice(0, NODE_LAYOUT.maxVisibleFields)

    let y = headerHeight + padding

    for (let i = 0; i < visibleFields.length; i++) {
      const field = visibleFields[i]
      if (!field) continue

      const isHovered = i === this.hoverState.fieldIndex

      // Draw hover background
      if (isHovered) {
        ctx.save()
        ctx.fillStyle = COMPONENT_COLORS.backgroundHover
        ctx.fillRect(3, y - 2, width - 6, fieldRowHeight)
        ctx.restore()
      }

      // Draw field
      this.drawField(ctx, field, padding, y, width, isHovered)

      y += fieldRowHeight + fieldGap
    }

    // Draw collapse indicator if needed
    if (!this.isExpanded && fields.length > NODE_LAYOUT.maxVisibleFields) {
      this.drawCollapseIndicator(ctx, y, width, fields.length - NODE_LAYOUT.maxVisibleFields)
      y += NODE_LAYOUT.collapseIndicatorHeight
    }

    // Draw expand/collapse toggle if many fields
    if (fields.length > NODE_LAYOUT.maxVisibleFields) {
      this.drawExpandToggle(ctx, width)
    }

    // Draw source file info
    if (this.componentData.source) {
      this.drawSourceInfo(ctx, width)
    }

    // Draw tooltip if hovering
    if (this.hoverState.fieldIndex >= 0) {
      this.drawTooltip(ctx)
    }
  }

  /**
   * Draw the header with icon and class name.
   */
  private drawHeader(ctx: CanvasRenderingContext2D, width: number): void {
    const { headerHeight, padding } = NODE_LAYOUT

    ctx.save()

    // Draw @component icon
    ctx.fillStyle = COMPONENT_COLORS.iconColor
    ctx.font = NODE_FONTS.icon
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'
    ctx.fillText('@', padding, headerHeight / 2)

    // Draw class name
    ctx.fillStyle = COMPONENT_COLORS.headerText
    ctx.font = NODE_FONTS.header
    ctx.textAlign = 'left'

    const iconWidth = ctx.measureText('@').width + 4
    const maxTitleWidth = width - padding * 2 - iconWidth - 10
    const title = truncateText(ctx, this.componentData.name, maxTitleWidth)
    ctx.fillText(title, padding + iconWidth, headerHeight / 2)

    ctx.restore()
  }

  /**
   * Draw a single field row.
   */
  private drawField(
    ctx: CanvasRenderingContext2D,
    field: ComponentField,
    x: number,
    y: number,
    _width: number,
    _isHovered: boolean
  ): void {
    ctx.save()

    const textY = y + NODE_LAYOUT.fieldRowHeight / 2 + 3

    // Field name (black)
    ctx.fillStyle = COMPONENT_COLORS.nameText
    ctx.font = NODE_FONTS.fieldName
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'
    ctx.fillText(field.name, x, textY)

    const nameWidth = ctx.measureText(field.name).width

    // Colon and type (gray)
    ctx.fillStyle = COMPONENT_COLORS.typeText
    ctx.font = NODE_FONTS.fieldType
    ctx.fillText(': ', x + nameWidth, textY)

    const colonWidth = ctx.measureText(': ').width
    ctx.fillText(field.type, x + nameWidth + colonWidth, textY)

    // Default value if present
    if (field.default) {
      const typeWidth = ctx.measureText(field.type).width
      ctx.fillStyle = COMPONENT_COLORS.defaultText
      ctx.font = NODE_FONTS.defaultValue
      ctx.fillText(` = ${field.default}`, x + nameWidth + colonWidth + typeWidth, textY)
    }

    ctx.restore()
  }

  /**
   * Draw the collapse indicator showing how many more fields exist.
   */
  private drawCollapseIndicator(
    ctx: CanvasRenderingContext2D,
    y: number,
    width: number,
    hiddenCount: number
  ): void {
    ctx.save()

    ctx.fillStyle = COMPONENT_COLORS.typeText
    ctx.font = NODE_FONTS.collapse
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(`... ${hiddenCount} more fields`, width / 2, y + NODE_LAYOUT.collapseIndicatorHeight / 2)

    ctx.restore()
  }

  /**
   * Draw the expand/collapse toggle button.
   */
  private drawExpandToggle(ctx: CanvasRenderingContext2D, width: number): void {
    ctx.save()

    const toggleSize = 16
    const toggleX = width - NODE_LAYOUT.padding - toggleSize
    const toggleY = NODE_LAYOUT.headerHeight + 4

    // Draw toggle background
    ctx.fillStyle = COMPONENT_COLORS.backgroundHover
    ctx.beginPath()
    ctx.arc(toggleX + toggleSize / 2, toggleY + toggleSize / 2, toggleSize / 2, 0, Math.PI * 2)
    ctx.fill()

    // Draw toggle icon
    ctx.fillStyle = COMPONENT_COLORS.primary
    ctx.font = '12px sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(this.isExpanded ? '-' : '+', toggleX + toggleSize / 2, toggleY + toggleSize / 2)

    ctx.restore()
  }

  /**
   * Draw source file information at the bottom.
   */
  private drawSourceInfo(ctx: CanvasRenderingContext2D, width: number): void {
    if (!this.componentData.source) return

    const [_w, height] = this.size

    ctx.save()

    ctx.fillStyle = COMPONENT_COLORS.typeText
    ctx.font = NODE_FONTS.footer
    ctx.textAlign = 'left'
    ctx.textBaseline = 'bottom'

    const sourceText = `${this.componentData.source.file}:${this.componentData.source.line}`
    const truncatedText = truncateText(ctx, sourceText, width - NODE_LAYOUT.padding * 2 - 10)
    ctx.fillText(truncatedText, NODE_LAYOUT.padding, height - 4)

    ctx.restore()
  }

  /**
   * Draw a tooltip for the hovered field.
   */
  private drawTooltip(ctx: CanvasRenderingContext2D): void {
    const field = this.componentData.fields[this.hoverState.fieldIndex]
    if (!field) return

    const tooltipText = field.type

    ctx.save()

    ctx.font = NODE_FONTS.fieldType
    const textWidth = ctx.measureText(tooltipText).width
    const tooltipWidth = textWidth + 16
    const tooltipHeight = 24

    const tooltipX = this.hoverState.position.x + 10
    const tooltipY = this.hoverState.position.y - tooltipHeight - 5

    // Draw tooltip background
    ctx.fillStyle = TRINITY_COLORS.neutral.textPrimary
    drawRoundRect(ctx, tooltipX, tooltipY, tooltipWidth, tooltipHeight, 4)
    ctx.fill()

    // Draw tooltip text
    ctx.fillStyle = TRINITY_COLORS.neutral.white
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'
    ctx.fillText(tooltipText, tooltipX + 8, tooltipY + tooltipHeight / 2)

    ctx.restore()
  }

  /**
   * Handle mouse move for hover detection.
   */
  onMouseMove(_e: MouseEvent, localPos: [number, number], _canvas: unknown): boolean {
    const [x, y] = localPos

    // Check if over a field
    const { headerHeight, padding, fieldRowHeight, fieldGap } = NODE_LAYOUT
    const fields = this.isExpanded
      ? this.componentData.fields
      : this.componentData.fields.slice(0, NODE_LAYOUT.maxVisibleFields)

    let fieldY = headerHeight + padding

    let foundIndex = -1
    for (let i = 0; i < fields.length; i++) {
      if (y >= fieldY && y < fieldY + fieldRowHeight) {
        foundIndex = i
        break
      }
      fieldY += fieldRowHeight + fieldGap
    }

    if (foundIndex !== this.hoverState.fieldIndex) {
      this.hoverState.fieldIndex = foundIndex
      this.hoverState.position = { x, y }
      this.setDirtyCanvas(true, true)
    }

    return false
  }

  /**
   * Handle mouse leave to clear hover state.
   */
  onMouseLeave(): void {
    if (this.hoverState.fieldIndex !== -1) {
      this.hoverState.fieldIndex = -1
      this.setDirtyCanvas(true, true)
    }
  }

  /**
   * Handle double-click to toggle expand/collapse or navigate to source.
   */
  onDblClick(_e: MouseEvent, localPos: [number, number], _canvas: unknown): boolean {
    const [, y] = localPos

    // If clicking in header area, toggle expand
    if (y < NODE_LAYOUT.headerHeight && this.componentData.fields.length > NODE_LAYOUT.maxVisibleFields) {
      this.toggleExpand()
      return true
    }

    // If source exists, double-click anywhere else navigates to source
    if (this.componentData.source) {
      const event = new CustomEvent('flowforge:navigate-to-source', {
        detail: {
          file: this.componentData.source.file,
          line: this.componentData.source.line,
        },
      })
      window.dispatchEvent(event)
      return true
    }

    return false
  }

  /**
   * Handle mouse down for source navigation.
   * Single click on the source info area navigates to source.
   */
  onMouseDown(_e: MouseEvent, localPos: [number, number], _canvas: unknown): boolean {
    // Check if clicking on source info area (bottom of node)
    if (this.componentData.source) {
      const [, y] = localPos
      const [, height] = this.size
      const sourceAreaTop = height - NODE_LAYOUT.footerHeight - NODE_LAYOUT.padding

      if (y >= sourceAreaTop) {
        // Dispatch navigation event
        const event = new CustomEvent('flowforge:navigate-to-source', {
          detail: {
            file: this.componentData.source.file,
            line: this.componentData.source.line,
          },
        })
        window.dispatchEvent(event)
        return true
      }
    }

    return false
  }

  /**
   * Get extra context menu options including source navigation.
   */
  override getExtraMenuOptions(): { content: string; callback: () => void }[] {
    const options: { content: string; callback: () => void }[] = []

    if (this.componentData.docstring) {
      options.push({
        content: `Docs: ${this.componentData.docstring.slice(0, 50)}${this.componentData.docstring.length > 50 ? '...' : ''}`,
        callback: () => {
          console.log('Component docstring:', this.componentData.docstring)
        },
      })
    }

    if (this.componentData.source) {
      options.push({
        content: `Go to source: ${this.componentData.source.file}:${this.componentData.source.line}`,
        callback: () => {
          // Dispatch navigation event for IDE integration
          const event = new CustomEvent('flowforge:navigate-to-source', {
            detail: {
              file: this.componentData.source!.file,
              line: this.componentData.source!.line,
            },
          })
          window.dispatchEvent(event)
        },
      })
    }

    return options
  }

  /**
   * Serialize node data for saving.
   */
  override serialize(): ReturnType<LGraphNode['serialize']> {
    const data = super.serialize()
    return {
      ...data,
      properties: {
        ...data.properties,
        componentData: this.componentData,
        isExpanded: this.isExpanded,
      },
    }
  }

  /**
   * Configure node from serialized data.
   */
  override configure(info: Parameters<LGraphNode['configure']>[0]): void {
    super.configure(info)
    const props = info.properties as {
      componentData?: ComponentNodeData
      isExpanded?: boolean
    } | undefined

    if (props?.componentData) {
      this.setComponentData(props.componentData)
    }
    if (props?.isExpanded !== undefined) {
      this.isExpanded = props.isExpanded
    }
  }

}

// =============================================================================
// FACTORY FUNCTIONS
// =============================================================================

/**
 * Create a ComponentNodeRenderer from parsed Python data.
 *
 * @param data - The component data from Python parser
 * @returns A configured ComponentNodeRenderer instance
 */
export function createComponentNode(data: ComponentNodeData): ComponentNodeRenderer {
  const node = new ComponentNodeRenderer(data.name)
  node.setComponentData(data)
  return node
}

/**
 * Register the ComponentNodeRenderer with LiteGraph.
 * Should be called once during application initialization.
 */
export function registerComponentNodeRenderer(): void {
  LiteGraph.registerNodeType(NODE_TYPE_NAMES.component, ComponentNodeRenderer)
  console.log('[FlowForge] ComponentNodeRenderer registered as', NODE_TYPE_NAMES.component)
}

/**
 * Unregister the ComponentNodeRenderer from LiteGraph.
 */
export function unregisterComponentNodeRenderer(): void {
  LiteGraph.unregisterNodeType(NODE_TYPE_NAMES.component)
}

// Export default for convenience
export default ComponentNodeRenderer
