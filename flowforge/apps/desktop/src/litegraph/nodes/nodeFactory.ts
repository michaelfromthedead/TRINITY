/**
 * Node Factory for FlowForge
 *
 * Creates LiteGraph nodes from various node definitions,
 * including Python parser output and Trinity ECS patterns.
 */

import { LGraphNode, LiteGraph } from '../index'
import type { Point, Size } from '../index'
import {
  ComponentNode,
  SystemNode,
  ResourceNode,
  EventNode,
  TrinityNode,
  type TrinityNodeType,
  type TrinityField,
  type TrinityMethod
} from './TrinityNodes'
import { getTrinityNodeClass } from './index'
import {
  ComponentNodeRenderer,
  createComponentNode,
  type ComponentNodeData
} from '../../nodes/ComponentNode'

/**
 * Node definition from the Python parser (API GraphNode format).
 */
export interface ParsedNodeDefinition {
  id: string
  type: 'component' | 'system' | 'resource' | 'event'
  name: string
  position: [number, number]
  data: {
    fields?: Array<{ name: string; type: string; default?: unknown }>
    methods?: Array<{
      name: string
      parameters?: Array<{ name: string; type: string }>
      returnType?: string
    }>
    queries?: string[]
    payloadFields?: Array<{ name: string; type: string }>
    description?: string
    isSingleton?: boolean
  }
  source: {
    file: string
    line: number
  }
}

/**
 * Graph node definition used in the store.
 */
export interface GraphNodeDefinition {
  id: string
  type: string
  title: string
  pos: [number, number]
  size: [number, number]
  inputs?: Array<{ name: string; type: string; link?: string | null }>
  outputs?: Array<{ name: string; type: string; links?: string[] }>
  properties?: Record<string, unknown>
  widgets?: Array<{ name: string; type: string; value: unknown }>
  flags?: { collapsed?: boolean; pinned?: boolean }
}

/**
 * Options for node creation.
 */
export interface CreateNodeOptions {
  /** Override the node position */
  position?: Point
  /** Override the node size */
  size?: Size
  /** Additional properties to set */
  properties?: Record<string, unknown>
  /** Whether to trigger onNodeCreated callback */
  triggerCreated?: boolean
}

/**
 * Create a Trinity node from a parsed definition (from Python parser).
 *
 * @param def - The parsed node definition from the API
 * @param options - Optional creation options
 * @returns A configured LGraphNode instance
 *
 * @example
 * ```typescript
 * const apiNode = {
 *   id: 'node_1',
 *   type: 'component',
 *   name: 'Position',
 *   position: [100, 200],
 *   data: {
 *     fields: [
 *       { name: 'x', type: 'float' },
 *       { name: 'y', type: 'float' }
 *     ]
 *   },
 *   source: { file: 'components.py', line: 10 }
 * }
 *
 * const node = createNodeFromParsedDef(apiNode)
 * graph.add(node)
 * ```
 */
export function createNodeFromParsedDef(
  def: ParsedNodeDefinition,
  options: CreateNodeOptions = {}
): TrinityNode {
  const NodeClass = getTrinityNodeClass(def.type as TrinityNodeType)

  // Create the node instance
  const node = new NodeClass(def.name) as TrinityNode

  // Set position
  const pos = options.position || def.position
  node.pos = [pos[0], pos[1]]

  // Set Trinity-specific properties
  node.className = def.name
  node.sourceFile = def.source.file
  node.sourceLine = def.source.line

  // Convert and set fields
  if (def.data.fields) {
    const fields: TrinityField[] = def.data.fields.map(f => ({
      name: f.name,
      type: f.type,
      defaultValue: f.default
    }))

    if (node instanceof ComponentNode) {
      node.setFields(fields)
    } else {
      node.fields = fields
    }
  }

  // Handle type-specific data
  if (node instanceof SystemNode) {
    if (def.data.methods) {
      const methods: TrinityMethod[] = def.data.methods.map(m => {
        const method: TrinityMethod = { name: m.name }
        if (m.parameters) method.parameters = m.parameters
        if (m.returnType) method.returnType = m.returnType
        return method
      })
      node.setMethods(methods)
    }
    if (def.data.queries) {
      node.setQueries(def.data.queries)
    }
  }

  if (node instanceof ResourceNode) {
    node.isSingleton = def.data.isSingleton ?? true
  }

  if (node instanceof EventNode && def.data.payloadFields) {
    const payloadFields: TrinityField[] = def.data.payloadFields.map(f => ({
      name: f.name,
      type: f.type
    }))
    node.setPayloadFields(payloadFields)
  }

  // Apply size override or calculate based on content
  if (options.size) {
    node.size = [options.size[0], options.size[1]]
  }

  // Apply additional properties
  if (options.properties) {
    Object.assign(node.properties, options.properties)
  }

  // Trigger created callback if requested
  if (options.triggerCreated !== false && node.onNodeCreated) {
    node.onNodeCreated()
  }

  return node
}

/**
 * Create a Trinity node from a graph store node definition.
 *
 * @param def - The graph node definition from the store
 * @param options - Optional creation options
 * @returns A configured LGraphNode instance
 */
export function createNodeFromGraphDef(
  def: GraphNodeDefinition,
  options: CreateNodeOptions = {}
): LGraphNode {
  // Check if this is a Trinity node type
  if (def.type.startsWith('trinity/')) {
    const trinityType = def.type.replace('trinity/', '') as TrinityNodeType
    const NodeClass = getTrinityNodeClass(trinityType)
    const node = new NodeClass(def.title) as TrinityNode

    // Set position and size
    node.pos = options.position || [...def.pos]
    node.size = options.size || [...def.size]

    // Set fields from properties
    if (def.properties) {
      node.className = (def.properties['className'] as string) || def.title
      if (def.properties['sourceFile'] !== undefined) {
        node.sourceFile = def.properties['sourceFile'] as string
      }
      if (def.properties['sourceLine'] !== undefined) {
        node.sourceLine = def.properties['sourceLine'] as number
      }

      if (def.properties['fields']) {
        node.fields = def.properties['fields'] as TrinityField[]
      }

      // Handle type-specific properties
      if (node instanceof SystemNode && def.properties['methods']) {
        node.methods = def.properties['methods'] as TrinityMethod[]
      }
      if (node instanceof SystemNode && def.properties['queries']) {
        node.queries = def.properties['queries'] as string[]
      }
      if (node instanceof EventNode && def.properties['payloadFields']) {
        node.payloadFields = def.properties['payloadFields'] as TrinityField[]
      }
    }

    // Configure inputs
    if (def.inputs) {
      def.inputs.forEach(input => {
        node.addInput(input.name, input.type)
      })
    }

    // Configure outputs
    if (def.outputs) {
      def.outputs.forEach(output => {
        node.addOutput(output.name, output.type)
      })
    }

    // Set flags
    if (def.flags) {
      node.flags = { ...def.flags }
    }

    return node
  }

  // For non-Trinity nodes, use LiteGraph's createNode
  const node = LiteGraph.createNode(def.type)
  if (!node) {
    throw new Error(`Unknown node type: ${def.type}`)
  }

  node.title = def.title
  node.pos = options.position || [...def.pos]
  node.size = options.size || [...def.size]

  if (def.properties) {
    Object.assign(node.properties, def.properties)
  }

  if (def.inputs) {
    def.inputs.forEach(input => {
      node.addInput(input.name, input.type)
    })
  }

  if (def.outputs) {
    def.outputs.forEach(output => {
      node.addOutput(output.name, output.type)
    })
  }

  if (def.flags) {
    node.flags = { ...def.flags }
  }

  return node
}

/**
 * Create a new empty Trinity node of the specified type.
 *
 * @param type - The Trinity node type
 * @param title - Optional title for the node
 * @param options - Optional creation options
 * @returns A new Trinity node instance
 */
export function createEmptyTrinityNode(
  type: TrinityNodeType,
  title?: string,
  options: CreateNodeOptions = {}
): TrinityNode {
  const NodeClass = getTrinityNodeClass(type)
  const node = new NodeClass(title) as TrinityNode

  if (options.position) {
    node.pos = [...options.position]
  }

  if (options.size) {
    node.size = [...options.size]
  }

  if (options.properties) {
    Object.assign(node.properties, options.properties)
  }

  if (options.triggerCreated !== false && node.onNodeCreated) {
    node.onNodeCreated()
  }

  return node
}

/**
 * Convert a LiteGraph node back to a GraphNodeDefinition.
 * Useful for serialization to the store.
 *
 * @param node - The LiteGraph node to convert
 * @returns A GraphNodeDefinition object
 */
export function nodeToGraphDef(node: LGraphNode): GraphNodeDefinition {
  const def: GraphNodeDefinition = {
    id: String(node.id),
    type: node.type,
    title: node.title,
    pos: [node.pos[0], node.pos[1]],
    size: [node.size[0], node.size[1]],
    properties: { ...node.properties }
  }

  // Set flags if present (only include defined properties)
  if (node.flags) {
    const flags: { collapsed?: boolean; pinned?: boolean } = {}
    if (node.flags.collapsed !== undefined) flags.collapsed = node.flags.collapsed
    if (node.flags.pinned !== undefined) flags.pinned = node.flags.pinned
    if (Object.keys(flags).length > 0) {
      def.flags = flags
    }
  }

  // Convert inputs
  if (node.inputs && node.inputs.length > 0) {
    def.inputs = node.inputs.map(input => ({
      name: input.name,
      type: input.type as string,
      link: input.link ? String(input.link) : null
    }))
  }

  // Convert outputs
  if (node.outputs && node.outputs.length > 0) {
    def.outputs = node.outputs.map(output => ({
      name: output.name,
      type: output.type as string,
      links: output.links?.map(l => String(l)) || []
    }))
  }

  // Add Trinity-specific properties
  if (node instanceof TrinityNode) {
    if (!def.properties) def.properties = {}
    def.properties['trinityType'] = node.trinityType
    def.properties['className'] = node.className
    def.properties['fields'] = node.fields
    def.properties['sourceFile'] = node.sourceFile
    def.properties['sourceLine'] = node.sourceLine

    if (node instanceof SystemNode) {
      def.properties['methods'] = node.methods
      def.properties['queries'] = node.queries
    }

    if (node instanceof EventNode) {
      def.properties['payloadFields'] = node.payloadFields
    }

    if (node instanceof ResourceNode) {
      def.properties['isSingleton'] = node.isSingleton
    }
  }

  return def
}

/**
 * Batch create nodes from an array of parsed definitions.
 *
 * @param defs - Array of parsed node definitions
 * @param options - Optional creation options applied to all nodes
 * @returns Array of created nodes
 */
export function createNodesFromParsedDefs(
  defs: ParsedNodeDefinition[],
  options: Omit<CreateNodeOptions, 'position'> = {}
): TrinityNode[] {
  return defs.map(def => createNodeFromParsedDef(def, options))
}

/**
 * Create a ComponentNodeRenderer from a parsed definition.
 *
 * This factory function creates the new custom-rendered Component node
 * with enhanced visual styling (blue gradient header, field list, etc.).
 *
 * @param def - The parsed node definition from the API
 * @param options - Optional creation options
 * @returns A configured ComponentNodeRenderer instance
 *
 * @example
 * ```typescript
 * const apiNode = {
 *   id: 'node_1',
 *   type: 'component',
 *   name: 'Position',
 *   position: [100, 200],
 *   data: {
 *     fields: [
 *       { name: 'x', type: 'float', default: '0.0' },
 *       { name: 'y', type: 'float', default: '0.0' }
 *     ],
 *     description: 'Position in 2D space'
 *   },
 *   source: { file: 'components.py', line: 10 }
 * }
 *
 * const node = createComponentNodeFromParsedDef(apiNode)
 * graph.add(node)
 * ```
 */
export function createComponentNodeFromParsedDef(
  def: ParsedNodeDefinition,
  options: CreateNodeOptions = {}
): ComponentNodeRenderer {
  // Convert ParsedNodeDefinition to ComponentNodeData
  // Map fields, only including default if it has a value
  const mappedFields = (def.data.fields || []).map(f => {
    const field: { name: string; type: string; default?: string } = {
      name: f.name,
      type: f.type
    }
    if (f.default !== undefined) {
      field.default = String(f.default)
    }
    return field
  })

  // Build component data, only including optional fields if they have values
  const componentData: ComponentNodeData = {
    name: def.name,
    fields: mappedFields,
    source: {
      file: def.source.file,
      line: def.source.line
    }
  }

  // Only add docstring if it exists
  if (def.data.description !== undefined) {
    componentData.docstring = def.data.description
  }

  // Create the node using the factory function
  const node = createComponentNode(componentData)

  // Set position
  const pos = options.position || def.position
  node.pos = [pos[0], pos[1]]

  // Apply size override if provided
  if (options.size) {
    node.size = [options.size[0], options.size[1]]
  }

  // Apply additional properties
  if (options.properties) {
    Object.assign(node.properties, options.properties)
  }

  // Trigger created callback if requested
  if (options.triggerCreated !== false && node.onNodeCreated) {
    node.onNodeCreated()
  }

  return node
}

/**
 * Create a node from a parsed definition, using ComponentNodeRenderer
 * for component types if useEnhancedRenderer is true.
 *
 * @param def - The parsed node definition from the API
 * @param options - Optional creation options
 * @param useEnhancedRenderer - Whether to use ComponentNodeRenderer for components (default: false)
 * @returns A configured LGraphNode instance
 */
export function createNodeFromParsedDefEnhanced(
  def: ParsedNodeDefinition,
  options: CreateNodeOptions = {},
  useEnhancedRenderer: boolean = false
): LGraphNode {
  // For components, optionally use the enhanced renderer
  if (def.type === 'component' && useEnhancedRenderer) {
    return createComponentNodeFromParsedDef(def, options)
  }

  // Fall back to standard node creation
  return createNodeFromParsedDef(def, options)
}
