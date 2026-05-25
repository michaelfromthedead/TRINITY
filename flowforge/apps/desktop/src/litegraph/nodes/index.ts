/**
 * LiteGraph Node Types for FlowForge
 *
 * This module exports all custom node classes and provides
 * registration functions for use at application startup.
 */

import { LiteGraph, LGraphCanvas } from '../index'
import {
  TrinityNode,
  ComponentNode,
  SystemNode,
  ResourceNode,
  EventNode,
  TRINITY_COLORS,
  type TrinityNodeType,
  type TrinityField,
  type TrinityMethod,
  type TrinityNodeProperties
} from './TrinityNodes'

// Import the custom Component Node Renderer
import {
  ComponentNodeRenderer,
  createComponentNode,
  registerComponentNodeRenderer,
  unregisterComponentNodeRenderer,
  type ComponentNodeData,
  type ComponentField
} from '../../nodes/ComponentNode'

// Re-export all node classes
export {
  TrinityNode,
  ComponentNode,
  SystemNode,
  ResourceNode,
  EventNode,
  TRINITY_COLORS,
  // Custom Component Node Renderer
  ComponentNodeRenderer,
  createComponentNode,
  registerComponentNodeRenderer,
  unregisterComponentNodeRenderer
}

// Re-export types
export type {
  TrinityNodeType,
  TrinityField,
  TrinityMethod,
  TrinityNodeProperties,
  // Component Node Renderer types
  ComponentNodeData,
  ComponentField
}

// Edge styling exports
export {
  EDGE_STYLES,
  EDGE_COLORS,
  DEFAULT_EDGE_CONFIG,
  getEdgeStyle,
  isTrinityEdgeType,
  applyEdgeStyles,
  applyEdgeStyleToLink,
  drawTrinityEdge,
  createTrinityLinkRenderer,
  setTrinityEdgeType,
  getAvailableEdgeTypes,
  getTrinityLinkData,
  setTrinityLinkData,
} from './edgeStyles'

export type {
  EdgeStyle,
  EdgePattern,
  TrinityEdgeType,
  TrinityLinkData,
  TrinityLinkRenderContext,
  TrinityEdgeConfig,
  ComputedEdgeStyle,
} from './edgeStyles'

/**
 * Map of Trinity node types to their corresponding classes.
 */
export const TRINITY_NODE_CLASSES = {
  component: ComponentNode,
  system: SystemNode,
  resource: ResourceNode,
  event: EventNode
} as const

/**
 * Flag to track if Trinity nodes have been registered.
 * Prevents duplicate registration warnings and unnecessary work.
 */
let trinityNodesRegistered = false

/**
 * Register all Trinity ECS node types with LiteGraph.
 * Should be called once at application startup before creating any graphs.
 * Safe to call multiple times - will only register once.
 *
 * @example
 * ```typescript
 * import { registerTrinityNodes } from '@/litegraph/nodes'
 *
 * // Call during app initialization
 * registerTrinityNodes()
 *
 * // Now Trinity nodes can be created
 * const node = LiteGraph.createNode('trinity/component')
 * ```
 */
export function registerTrinityNodes(): void {
  // Guard against duplicate registration
  if (trinityNodesRegistered) {
    return
  }

  // Register base Trinity node
  LiteGraph.registerNodeType('trinity/base', TrinityNode)

  // Register Component node
  LiteGraph.registerNodeType('trinity/component', ComponentNode)

  // Register System node
  LiteGraph.registerNodeType('trinity/system', SystemNode)

  // Register Resource node
  LiteGraph.registerNodeType('trinity/resource', ResourceNode)

  // Register Event node
  LiteGraph.registerNodeType('trinity/event', EventNode)

  // NOTE: Do NOT call registerComponentNodeRenderer() here.
  // It would override 'trinity/component' with ComponentNodeRenderer (extends LGraphNode)
  // which is incompatible with the TrinityNode-based sync flow in GraphCanvas.

  // Set up node colors in LGraphCanvas configuration
  LGraphCanvas.node_colors['trinity_component'] = {
    color: TRINITY_COLORS.component.color,
    bgcolor: TRINITY_COLORS.component.bgcolor,
    groupcolor: TRINITY_COLORS.component.colorDark
  }
  LGraphCanvas.node_colors['trinity_system'] = {
    color: TRINITY_COLORS.system.color,
    bgcolor: TRINITY_COLORS.system.bgcolor,
    groupcolor: TRINITY_COLORS.system.colorDark
  }
  LGraphCanvas.node_colors['trinity_resource'] = {
    color: TRINITY_COLORS.resource.color,
    bgcolor: TRINITY_COLORS.resource.bgcolor,
    groupcolor: TRINITY_COLORS.resource.colorDark
  }
  LGraphCanvas.node_colors['trinity_event'] = {
    color: TRINITY_COLORS.event.color,
    bgcolor: TRINITY_COLORS.event.bgcolor,
    groupcolor: TRINITY_COLORS.event.colorDark
  }

  // Mark as registered
  trinityNodesRegistered = true

  console.log('[FlowForge] Trinity ECS node types registered')
}

/**
 * Check if Trinity nodes have been registered.
 * @returns True if registerTrinityNodes() has been called
 */
export function areTrinityNodesRegistered(): boolean {
  return trinityNodesRegistered
}

/**
 * Unregister all Trinity ECS node types from LiteGraph.
 * Useful for testing or dynamic node type management.
 */
export function unregisterTrinityNodes(): void {
  if (!trinityNodesRegistered) {
    return
  }

  LiteGraph.unregisterNodeType('trinity/base')
  LiteGraph.unregisterNodeType('trinity/component')
  LiteGraph.unregisterNodeType('trinity/system')
  LiteGraph.unregisterNodeType('trinity/resource')
  LiteGraph.unregisterNodeType('trinity/event')

  // Unregister the custom Component Node Renderer
  unregisterComponentNodeRenderer()

  // Reset the registration flag
  trinityNodesRegistered = false

  console.log('[FlowForge] Trinity ECS node types unregistered')
}

/**
 * Get the node class for a given Trinity type.
 *
 * @param type - The Trinity node type
 * @returns The corresponding node class constructor
 */
export function getTrinityNodeClass(type: TrinityNodeType): typeof TrinityNode {
  return TRINITY_NODE_CLASSES[type] || TrinityNode
}

/**
 * Check if a node type string is a Trinity node type.
 *
 * @param type - The node type string to check
 * @returns True if the type is a registered Trinity node type
 */
export function isTrinityNodeType(type: string): type is `trinity/${TrinityNodeType}` {
  return type.startsWith('trinity/') && ['component', 'system', 'resource', 'event'].includes(
    type.replace('trinity/', '')
  )
}
