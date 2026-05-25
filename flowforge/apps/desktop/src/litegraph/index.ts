/**
 * LiteGraph Library for FlowForge
 *
 * A self-contained node graph library extracted from ComfyUI_frontend.
 * This module provides the core classes and utilities for building
 * visual node-based editors.
 *
 * @example
 * ```typescript
 * import { LGraph, LGraphCanvas, LGraphNode, LiteGraph } from '@/litegraph'
 *
 * // Create a new graph
 * const graph = new LGraph()
 *
 * // Create a canvas to render the graph
 * const canvas = new LGraphCanvas('#mycanvas', graph)
 *
 * // Register a custom node type
 * class MyNode extends LGraphNode {
 *   static title = 'My Node'
 *   constructor() {
 *     super()
 *     this.addInput('in', 'number')
 *     this.addOutput('out', 'number')
 *   }
 * }
 * LiteGraph.registerNodeType('custom/mynode', MyNode)
 * ```
 */

// Core classes
export {
  LiteGraph,
  LGraph,
  LGraphCanvas,
  LGraphNode,
  LGraphGroup,
  LLink,
  Reroute,
  Subgraph,
  SubgraphNode
} from './src/litegraph'

// Badge and button components
export { BadgePosition, LGraphBadge } from './src/litegraph'
export { LGraphButton } from './src/litegraph'

// Canvas utilities
export { CanvasPointer } from './src/litegraph'
export { DragAndScale } from './src/litegraph'
export { LinkConnector } from './src/litegraph'
export { Rectangle } from './src/litegraph'

// Constants
export * as Constants from './src/constants'
export { SUBGRAPH_INPUT_ID } from './src/litegraph'

// Context menu
export { ContextMenu } from './src/litegraph'

// Enums
export {
  EaseFunction,
  LGraphEventMode,
  LinkDirection,
  LinkMarkerShape,
  RenderShape
} from './src/litegraph'

// Error types
export { RecursionError } from './src/litegraph'

// Node slots
export { NodeInputSlot } from './src/litegraph'
export { NodeOutputSlot } from './src/litegraph'
export { inputAsSerialisable, outputAsSerialisable } from './src/litegraph'

// Subgraph utilities
export {
  ExecutableNodeDTO,
  findUsedSubgraphIds,
  getDirectSubgraphIds,
  isSubgraphInput,
  isSubgraphOutput
} from './src/litegraph'

// Widgets
export { BaseWidget } from './src/litegraph'
export { LegacyWidget } from './src/litegraph'
export { isComboWidget, isAssetWidget } from './src/litegraph'

// Utility functions
export { createBounds } from './src/litegraph'
export { createUuidv4 } from './src/litegraph'
export { truncateText } from './src/litegraph'
export { getWidgetStep } from './src/litegraph'
export { distributeSpace } from './src/litegraph'
export { isColorable } from './src/litegraph'
export { isOverNodeInput, isOverNodeOutput } from './src/litegraph'

// Types
export type {
  CanvasColour,
  ColorOption,
  IContextMenuOptions,
  IContextMenuValue,
  INodeInputSlot,
  INodeOutputSlot,
  INodeSlot,
  ISlotType,
  LinkNetwork,
  Point,
  Positionable,
  Size,
  ConnectingLink
} from './src/litegraph'

export type {
  NodeId
} from './src/litegraph'

export type {
  RerouteId
} from './src/litegraph'

export type {
  LGraphTriggerEvent,
  GroupNodeConfigEntry,
  GroupNodeWorkflowData,
  LGraphTriggerAction,
  LGraphTriggerParam
} from './src/litegraph'

export type {
  CanvasPointerEvent
} from './src/litegraph'

export type {
  ExportedSubgraph,
  ExportedSubgraphInstance,
  ISerialisedGraph,
  ISerialisedNode,
  SerialisableGraph
} from './src/litegraph'

export type {
  IWidget,
  TWidgetType,
  TWidgetValue,
  IWidgetOptions
} from './src/litegraph'

export type {
  ExecutableLGraphNode,
  ExecutionId
} from './src/litegraph'

export type {
  UUID
} from './src/litegraph'

export type {
  SpaceRequest
} from './src/litegraph'

export type {
  LGraphNodeConstructor,
  LinkReleaseContextExtended,
  LiteGraphCanvasEvent,
  Vector2
} from './src/litegraph'

// LiteGraph utility functions
export {
  isImageNode,
  isVideoNode,
  isAudioNode,
  isLGraphNode,
  isLGraphGroup,
  isReroute,
  addToComboValues,
  getItemsColorOption,
  executeWidgetsCallback,
  migrateWidgetsValues,
  fixLinkInputSlots,
  compressWidgetInputSlots,
  isLoad3dNode
} from './litegraphUtil'

// Type augmentations
export type {
  NodeExecutionOutput,
  NodeDef,
  DOMWidget,
  DOMWidgetOptions,
  Rect
} from './litegraph-augmentation'

// Trinity ECS Node Types
export {
  TrinityNode,
  ComponentNode,
  SystemNode,
  ResourceNode,
  EventNode,
  TRINITY_COLORS,
  TRINITY_NODE_CLASSES,
  registerTrinityNodes,
  unregisterTrinityNodes,
  areTrinityNodesRegistered,
  getTrinityNodeClass,
  isTrinityNodeType,
  // Custom Component Node Renderer
  ComponentNodeRenderer,
  createComponentNode,
  registerComponentNodeRenderer,
  unregisterComponentNodeRenderer
} from './nodes'

export type {
  TrinityNodeType,
  TrinityField,
  TrinityMethod,
  TrinityNodeProperties,
  // Component Node Renderer types
  ComponentNodeData,
  ComponentField
} from './nodes'

// Trinity Edge Styling
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
} from './nodes'

export type {
  EdgeStyle,
  EdgePattern,
  TrinityEdgeType,
  TrinityLinkData,
  TrinityLinkRenderContext,
  TrinityEdgeConfig,
  ComputedEdgeStyle,
} from './nodes'

// Node Factory
export {
  createNodeFromParsedDef,
  createNodeFromGraphDef,
  createEmptyTrinityNode,
  nodeToGraphDef,
  createNodesFromParsedDefs
} from './nodes/nodeFactory'

export type {
  ParsedNodeDefinition,
  GraphNodeDefinition,
  CreateNodeOptions
} from './nodes/nodeFactory'
