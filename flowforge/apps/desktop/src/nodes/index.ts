/**
 * Custom Node Renderers for FlowForge
 *
 * This module exports specialized LiteGraph node renderers
 * for Trinity ECS patterns with enhanced visual styling.
 */

// Component Node - Blue themed component visualization
export {
  ComponentNodeRenderer,
  registerComponentNodeRenderer,
  unregisterComponentNodeRenderer,
  createComponentNode,
} from './ComponentNode'

export type { ComponentNodeData, ComponentField } from './ComponentNode'

// Event Node - Orange themed event/signal visualization
export {
  EventNode,
  registerEventNode,
  unregisterEventNode,
  createEventNodeFromData,
  EVENT_COLORS,
} from './EventNode'

export type { EventNodeData, EventPayloadField } from './EventNode'

// System Node - Green themed system processors
export {
  SystemNodeRenderer,
  registerSystemNode,
  unregisterSystemNode,
  createSystemNode,
  SYSTEM_COLORS,
  SYSTEM_LAYOUT,
  SYSTEM_FONTS,
} from './SystemNode'

export type { SystemNodeData, SystemMethod } from './SystemNode'

// Resource Node - Purple themed singleton resources
export {
  ResourceNode,
  registerResourceNode,
  unregisterResourceNode,
  createResourceNode,
  RESOURCE_COLORS,
} from './ResourceNode'

export type { ResourceNodeData, ResourceField } from './ResourceNode'

// Shared theme exports
export {
  TRINITY_COLORS,
  NODE_LAYOUT,
  NODE_FONTS,
  EDGE_COLORS,
  COMPLEX_TYPES,
  NODE_TYPE_NAMES,
  isComplexType,
  truncateText,
  drawRoundRect,
  createHeaderGradient,
} from './nodeTheme'

export type { TrinityNodeType, NodeTypeName } from './nodeTheme'

/**
 * Register all custom node types with LiteGraph.
 * Call this once during application initialization.
 *
 * @returns Promise that resolves when all nodes are registered
 */
export async function registerAllCustomNodes(): Promise<void> {
  const registrations = await Promise.all([
    import('./ComponentNode').then(({ registerComponentNodeRenderer }) => {
      registerComponentNodeRenderer()
      return 'ComponentNode'
    }),
    import('./EventNode').then(({ registerEventNode }) => {
      registerEventNode()
      return 'EventNode'
    }),
    import('./SystemNode').then(({ registerSystemNode }) => {
      registerSystemNode()
      return 'SystemNode'
    }),
    import('./ResourceNode').then(({ registerResourceNode }) => {
      registerResourceNode()
      return 'ResourceNode'
    }),
  ])

  console.log('[FlowForge] All custom node types registered:', registrations.join(', '))
}

/**
 * Unregister all custom node types from LiteGraph.
 *
 * @returns Promise that resolves when all nodes are unregistered
 */
export async function unregisterAllCustomNodes(): Promise<void> {
  await Promise.all([
    import('./ComponentNode').then(({ unregisterComponentNodeRenderer }) => {
      unregisterComponentNodeRenderer()
    }),
    import('./EventNode').then(({ unregisterEventNode }) => {
      unregisterEventNode()
    }),
    import('./SystemNode').then(({ unregisterSystemNode }) => {
      unregisterSystemNode()
    }),
    import('./ResourceNode').then(({ unregisterResourceNode }) => {
      unregisterResourceNode()
    }),
  ])

  console.log('[FlowForge] All custom node types unregistered')
}
