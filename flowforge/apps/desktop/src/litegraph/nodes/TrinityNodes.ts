/**
 * Trinity ECS Node Types for LiteGraph
 *
 * Custom LiteGraph node classes for visualizing Trinity ECS patterns:
 * - Component nodes (blue) - Reusable game components
 * - System nodes (green) - Core engine systems
 * - Resource nodes (purple) - Shared data/assets
 * - Event nodes (orange) - Triggers and signals
 */

import { LGraphNode } from '../index'
import type { Size } from '../index'
import {
  TRINITY_COLORS,
  NODE_CONFIG,
  FONTS,
  TEXT_COLORS,
} from '@/config/flowforge.config'

// Trinity node type literal
export type TrinityNodeType = 'component' | 'system' | 'resource' | 'event'

// Re-export TRINITY_COLORS for backwards compatibility
export { TRINITY_COLORS }

// Field definition for Trinity nodes
export interface TrinityField {
  name: string
  type: string
  defaultValue?: unknown
  description?: string
}

// Method definition for System nodes
export interface TrinityMethod {
  name: string
  parameters?: Array<{ name: string; type: string }>
  returnType?: string
  description?: string
}

// Properties stored on Trinity nodes
export interface TrinityNodeProperties {
  trinityType: TrinityNodeType
  className: string
  fields: TrinityField[]
  methods?: TrinityMethod[]
  sourceFile?: string
  sourceLine?: number
  description?: string
  isSingleton?: boolean
  queries?: string[]
  payloadFields?: TrinityField[]
}

/**
 * Base class for all Trinity ECS nodes.
 * Provides common rendering and property handling.
 */
export class TrinityNode extends LGraphNode {
  static category = 'trinity'

  // Trinity-specific properties
  trinityType: TrinityNodeType = 'component'
  sourceFile?: string
  sourceLine?: number
  className: string = ''
  fields: TrinityField[] = []

  constructor(title?: string, type?: string) {
    super(title || 'Trinity Node', type)
    this.size = [...NODE_CONFIG.sizes.component] as Size
  }

  /**
   * Get the color configuration for this node's Trinity type.
   */
  protected getTrinityColors() {
    return TRINITY_COLORS[this.trinityType] || TRINITY_COLORS.component
  }

  /**
   * Override to use Trinity-themed colors.
   */
  override get renderingColor(): string {
    return this.color || this.getTrinityColors().color
  }

  override get renderingBgColor(): string {
    return this.bgcolor || this.getTrinityColors().bgcolor
  }

  override get renderingBoxColor(): string {
    return this.boxcolor || this.getTrinityColors().color
  }

  /**
   * Calculate required node height based on content.
   */
  protected calculateHeight(): number {
    const { titleHeight, slotHeight, fieldHeight, padding } = NODE_CONFIG.layout

    const inputSlots = this.inputs?.length || 0
    const outputSlots = this.outputs?.length || 0
    const maxSlots = Math.max(inputSlots, outputSlots)
    const fieldsCount = this.fields?.length || 0

    return titleHeight + padding + (maxSlots * slotHeight) + (fieldsCount * fieldHeight) + padding
  }

  /**
   * Hit-test whether a local (node-relative) position falls on a field name.
   * Returns the field index and field object if hit, or null otherwise.
   */
  getFieldAtPosition(localX: number, localY: number): { index: number; field: TrinityField } | null {
    if (!this.fields || this.fields.length === 0) return null

    const { titleHeight, slotHeight, fieldHeight, padding } = NODE_CONFIG.layout
    const startY = titleHeight + 8 + (Math.max(this.inputs?.length || 0, this.outputs?.length || 0) * slotHeight)

    for (let i = 0; i < this.fields.length; i++) {
      const fieldTop = startY + (i * fieldHeight) - fieldHeight + 2
      const fieldBottom = fieldTop + fieldHeight

      if (localY >= fieldTop && localY <= fieldBottom && localX >= padding && localX <= this.size[0] - padding) {
        return { index: i, field: this.fields[i] }
      }
    }

    return null
  }

  /**
   * Custom foreground rendering for Trinity nodes.
   * Draws fields and type indicators.
   */
  override onDrawForeground(ctx: CanvasRenderingContext2D): void {
    if (this.flags?.collapsed) return

    const colors = this.getTrinityColors()
    const { titleHeight, slotHeight, fieldHeight, padding } = NODE_CONFIG.layout
    const nodeWidth = this.size[0]

    // Draw type indicator badge
    ctx.save()
    ctx.fillStyle = colors.color
    ctx.font = FONTS.label
    ctx.textAlign = 'right'
    ctx.fillText(this.trinityType.toUpperCase(), nodeWidth - 8, titleHeight - 8)
    ctx.restore()

    // Draw fields
    if (this.fields && this.fields.length > 0) {
      ctx.save()
      ctx.font = FONTS.code
      ctx.textAlign = 'left'

      const startY = titleHeight + 8 + (Math.max(this.inputs?.length || 0, this.outputs?.length || 0) * slotHeight)

      this.fields.forEach((field, index) => {
        const y = startY + (index * fieldHeight)

        // Field name
        ctx.fillStyle = TEXT_COLORS.fieldName
        ctx.fillText(field.name, padding, y)

        // Field type
        ctx.fillStyle = colors.colorLight
        const nameWidth = ctx.measureText(field.name).width
        ctx.fillText(`: ${field.type}`, padding + nameWidth, y)
      })

      ctx.restore()
    }

    // Draw source file indicator if available
    if (this.sourceFile) {
      ctx.save()
      ctx.fillStyle = TEXT_COLORS.sourceFile
      ctx.font = FONTS.labelSmall
      ctx.textAlign = 'left'
      const sourceText = this.sourceLine
        ? `${this.sourceFile}:${this.sourceLine}`
        : this.sourceFile
      const truncated = sourceText.length > 30
        ? '...' + sourceText.slice(-27)
        : sourceText
      ctx.fillText(truncated, 8, this.size[1] - 6)
      ctx.restore()
    }
  }

  /**
   * Custom background rendering for Trinity nodes.
   * Adds subtle gradient and border effects.
   */
  override onDrawBackground(ctx: CanvasRenderingContext2D): void {
    if (this.flags?.collapsed) return

    const colors = this.getTrinityColors()
    const nodeHeight = this.size[1]

    // Draw subtle left border accent
    ctx.save()
    ctx.fillStyle = colors.color
    ctx.fillRect(0, 0, 3, nodeHeight)
    ctx.restore()
  }

  /**
   * Serialize Trinity-specific properties.
   */
  override serialize(): ReturnType<LGraphNode['serialize']> {
    const data = super.serialize()
    return {
      ...data,
      properties: {
        ...data.properties,
        trinityType: this.trinityType,
        className: this.className,
        fields: this.fields,
        sourceFile: this.sourceFile,
        sourceLine: this.sourceLine
      }
    }
  }

  /**
   * Configure from serialized data.
   */
  override configure(info: Parameters<LGraphNode['configure']>[0]): void {
    super.configure(info)
    const props = info.properties as Partial<TrinityNodeProperties> | undefined
    if (props) {
      if (props.trinityType) this.trinityType = props.trinityType
      if (props.className) this.className = props.className
      if (props.fields) this.fields = props.fields
      if (props.sourceFile !== undefined) this.sourceFile = props.sourceFile
      if (props.sourceLine !== undefined) this.sourceLine = props.sourceLine
    }
  }
}

/**
 * Component Node - Blue theme
 * Represents reusable game components (data containers).
 */
export class ComponentNode extends TrinityNode {
  static title = 'Component'
  static desc = 'A data component that can be attached to entities'

  override trinityType: TrinityNodeType = 'component'

  constructor(title?: string) {
    super(title || 'Component', 'trinity/component')
    this.color = TRINITY_COLORS.component.color
    this.bgcolor = TRINITY_COLORS.component.bgcolor
  }

  override onNodeCreated(): void {
    // Add default output slot for component data
    this.addOutput('data', 'component')
  }

  /**
   * Set the component fields and update the node display.
   */
  setFields(fields: TrinityField[]): void {
    this.fields = fields
    // Recalculate size based on fields
    const height = this.calculateHeight()
    this.size = [Math.max(this.size[0], 180), height] as Size
  }
}

/**
 * System Node - Green theme
 * Represents systems that process entities with specific components.
 */
export class SystemNode extends TrinityNode {
  static title = 'System'
  static desc = 'A system that processes entities with specific components'

  override trinityType: TrinityNodeType = 'system'
  methods: TrinityMethod[] = []
  queries: string[] = []

  constructor(title?: string) {
    super(title || 'System', 'trinity/system')
    this.color = TRINITY_COLORS.system.color
    this.bgcolor = TRINITY_COLORS.system.bgcolor
    this.size = [...NODE_CONFIG.sizes.system] as Size
  }

  override onNodeCreated(): void {
    // Systems typically have execution flow
    this.addInput('trigger', 'exec')
    this.addOutput('next', 'exec')
  }

  /**
   * Set the query types this system operates on.
   */
  setQueries(queries: string[]): void {
    this.queries = queries
    // Add input slots for each query type
    queries.forEach((query) => {
      if (!this.inputs?.find(i => i.name === query)) {
        this.addInput(query, 'query')
      }
    })
  }

  /**
   * Set the methods this system exposes.
   */
  setMethods(methods: TrinityMethod[]): void {
    this.methods = methods
  }

  override onDrawForeground(ctx: CanvasRenderingContext2D): void {
    super.onDrawForeground(ctx)

    if (this.flags?.collapsed) return

    const colors = this.getTrinityColors()
    const { titleHeight, slotHeight, fieldHeight, methodHeight, padding } = NODE_CONFIG.layout

    // Draw methods section if present
    if (this.methods && this.methods.length > 0) {
      ctx.save()
      ctx.font = FONTS.codeSmall
      ctx.textAlign = 'left'

      const startY = titleHeight + 8 + (Math.max(this.inputs?.length || 0, this.outputs?.length || 0) * slotHeight)
      const fieldsOffset = (this.fields?.length || 0) * fieldHeight

      ctx.fillStyle = TEXT_COLORS.muted
      ctx.fillText('Methods:', padding, startY + fieldsOffset)

      this.methods.forEach((method, index) => {
        const y = startY + fieldsOffset + methodHeight + (index * methodHeight)
        ctx.fillStyle = colors.colorLight
        ctx.fillText(`${method.name}()`, padding + 4, y)
      })

      ctx.restore()
    }
  }

  override calculateHeight(): number {
    const base = super.calculateHeight()
    const { methodHeight } = NODE_CONFIG.layout
    const methodsHeight = this.methods?.length ? (methodHeight + this.methods.length * methodHeight) : 0
    return base + methodsHeight
  }
}

/**
 * Resource Node - Purple theme
 * Represents singleton resources (global shared data).
 */
export class ResourceNode extends TrinityNode {
  static title = 'Resource'
  static desc = 'A singleton resource shared across the application'

  override trinityType: TrinityNodeType = 'resource'
  isSingleton: boolean = true

  constructor(title?: string) {
    super(title || 'Resource', 'trinity/resource')
    this.color = TRINITY_COLORS.resource.color
    this.bgcolor = TRINITY_COLORS.resource.bgcolor
  }

  override onNodeCreated(): void {
    // Resources provide data output
    this.addOutput('data', 'resource')
  }

  override onDrawForeground(ctx: CanvasRenderingContext2D): void {
    super.onDrawForeground(ctx)

    if (this.flags?.collapsed) return

    // Draw singleton indicator
    const colors = this.getTrinityColors()
    const { titleHeight } = NODE_CONFIG.layout

    ctx.save()
    ctx.fillStyle = colors.colorLight
    ctx.font = FONTS.labelBold
    ctx.textAlign = 'left'
    ctx.fillText('SINGLETON', 8, titleHeight - 8)
    ctx.restore()
  }
}

/**
 * Event Node - Orange theme
 * Represents events (signals/triggers with payload).
 */
export class EventNode extends TrinityNode {
  static title = 'Event'
  static desc = 'An event that can be triggered and listened to'

  override trinityType: TrinityNodeType = 'event'
  payloadFields: TrinityField[] = []

  constructor(title?: string) {
    super(title || 'Event', 'trinity/event')
    this.color = TRINITY_COLORS.event.color
    this.bgcolor = TRINITY_COLORS.event.bgcolor
  }

  override onNodeCreated(): void {
    // Events have trigger input and signal output
    this.addInput('trigger', 'exec')
    this.addOutput('signal', 'event')
  }

  /**
   * Set the event payload fields.
   */
  setPayloadFields(fields: TrinityField[]): void {
    this.payloadFields = fields
    this.fields = fields // Use base class fields for rendering
    const height = this.calculateHeight()
    this.size = [Math.max(this.size[0], 180), height] as Size
  }

  override onDrawForeground(ctx: CanvasRenderingContext2D): void {
    // Override to show "Payload:" label before fields
    if (this.flags?.collapsed) return

    const colors = this.getTrinityColors()
    const { titleHeight, slotHeight, fieldHeight, payloadLabelHeight, padding } = NODE_CONFIG.layout
    const [width] = this.size

    // Draw type indicator badge
    ctx.save()
    ctx.fillStyle = colors.color
    ctx.font = FONTS.label
    ctx.textAlign = 'right'
    ctx.fillText('EVENT', width - 8, titleHeight - 8)
    ctx.restore()

    // Draw payload label and fields
    if (this.payloadFields && this.payloadFields.length > 0) {
      ctx.save()

      const startY = titleHeight + 8 + (Math.max(this.inputs?.length || 0, this.outputs?.length || 0) * slotHeight)

      // Payload label
      ctx.fillStyle = TEXT_COLORS.muted
      ctx.font = FONTS.label
      ctx.textAlign = 'left'
      ctx.fillText('Payload:', padding, startY)

      // Payload fields
      ctx.font = FONTS.code
      this.payloadFields.forEach((field, index) => {
        const y = startY + payloadLabelHeight + (index * fieldHeight)

        ctx.fillStyle = TEXT_COLORS.fieldName
        ctx.fillText(field.name, padding + 4, y)

        ctx.fillStyle = colors.colorLight
        const nameWidth = ctx.measureText(field.name).width
        ctx.fillText(`: ${field.type}`, padding + 4 + nameWidth, y)
      })

      ctx.restore()
    }

    // Draw source file indicator
    if (this.sourceFile) {
      ctx.save()
      ctx.fillStyle = TEXT_COLORS.sourceFile
      ctx.font = FONTS.labelSmall
      ctx.textAlign = 'left'
      const sourceText = this.sourceLine
        ? `${this.sourceFile}:${this.sourceLine}`
        : this.sourceFile
      const truncated = sourceText.length > 30
        ? '...' + sourceText.slice(-27)
        : sourceText
      ctx.fillText(truncated, 8, this.size[1] - 6)
      ctx.restore()
    }
  }

  override calculateHeight(): number {
    const { titleHeight, slotHeight, fieldHeight, payloadLabelHeight, padding, sourceIndicatorHeight } = NODE_CONFIG.layout

    const inputSlots = this.inputs?.length || 0
    const outputSlots = this.outputs?.length || 0
    const maxSlots = Math.max(inputSlots, outputSlots)
    const payloadCount = this.payloadFields?.length || 0

    // Extra space for "Payload:" label
    const labelHeight = payloadCount > 0 ? payloadLabelHeight : 0

    return titleHeight + padding + (maxSlots * slotHeight) + labelHeight + (payloadCount * fieldHeight) + padding + sourceIndicatorHeight
  }

  override serialize(): ReturnType<LGraphNode['serialize']> {
    const data = super.serialize()
    return {
      ...data,
      properties: {
        ...data.properties,
        payloadFields: this.payloadFields
      }
    }
  }

  override configure(info: Parameters<LGraphNode['configure']>[0]): void {
    super.configure(info)
    const props = info.properties as TrinityNodeProperties | undefined
    if (props?.payloadFields) {
      this.payloadFields = props.payloadFields
      this.fields = props.payloadFields
    }
  }
}
