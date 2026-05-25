/**
 * FlowForge LiteGraph Type Augmentations
 *
 * This file provides type augmentations for the LiteGraph library.
 * These extend the base types with FlowForge-specific functionality.
 */

import './src/litegraph'
import type {
  ExecutableLGraphNode,
  ExecutionId,
  LLink,
  LGraphNode,
  Size,
  Point
} from './src/litegraph'
import type { IBaseWidget } from './src/types/widgets'

// Re-export NodeId from LGraphNode
export type { NodeId } from './src/LGraphNode'

/** Output from node execution */
export interface NodeExecutionOutput {
  images?: Array<{ filename: string; subfolder: string; type: string }>
  audio?: Array<{ filename: string; subfolder: string; type: string }>
  video?: Array<{ filename: string; subfolder: string; type: string }>
  text?: string[]
  [key: string]: unknown
}

/** Node definition interface */
export interface NodeDef {
  name: string
  display_name?: string
  description?: string
  category?: string
  input?: {
    required?: Record<string, unknown>
    optional?: Record<string, unknown>
  }
  output?: string[]
  output_name?: string[]
}

/** DOM Widget interface */
export interface DOMWidget<T extends HTMLElement = HTMLElement, V = string> {
  element: T
  name: string
  type: string
  value: V
  options?: DOMWidgetOptions<V>
}

/** DOM Widget options */
export interface DOMWidgetOptions<V = string> {
  getValue?: () => V
  setValue?: (value: V) => void
  onHide?: (widget: DOMWidget) => void
}

/** Rect type for image bounds */
export type Rect = [x: number, y: number, width: number, height: number]

/** FlowForge extensions of litegraph widget types */
declare module './src/types/widgets' {
  interface IWidgetOptions {
    /** Callback when widget is hidden */
    onHide?: (widget: DOMWidget) => void
    /**
     * Controls whether the widget's value is included in the workflow.
     * @default true
     */
    serialize?: boolean
    /** Rounding value for numeric float widgets */
    round?: number
    /** The minimum size of the node if the widget is present */
    minNodeSize?: Size
    /** If the widget is advanced, this will be set to true */
    advanced?: boolean
    /** If the widget is hidden, this will be set to true */
    hidden?: boolean
  }

  interface IBaseWidget {
    onRemove?(): void
    beforeQueued?(): unknown
    afterQueued?(): unknown
    serializeValue?(node: LGraphNode, index: number): Promise<unknown> | unknown
    /** Refreshes the widget's value or options from its remote source */
    refresh?(): unknown
  }
}

/** FlowForge extensions of litegraph interfaces */
declare module './src/interfaces' {
  interface IWidgetLocator {
    [key: symbol]: unknown
  }
}

/** FlowForge extensions of litegraph */
declare module './src/litegraph' {
  interface LGraphNodeConstructor<T extends LGraphNode = LGraphNode> {
    type?: string
    nodeClass?: string
    title: string
    nodeData?: NodeDef & { [key: symbol]: unknown }
    category?: string
    new (): T
  }

  interface BaseWidget extends IBaseWidget {}

  interface LGraphNode {
    constructor: LGraphNodeConstructor

    /** Callback fired on each node after the graph is configured */
    onAfterGraphConfigured?(): void
    onGraphConfigured?(): void
    /** Callback fired when node execution completes */
    onExecuted?(output: NodeExecutionOutput): void
    onNodeCreated?(this: LGraphNode): void

    /** Get inner nodes for subgraph execution */
    getInnerNodes?(
      nodesByExecutionId: Map<ExecutionId, ExecutableLGraphNode>,
      subgraphNodePath?: readonly (number | string)[],
      nodes?: ExecutableLGraphNode[],
      subgraphs?: Set<LGraphNode>
    ): ExecutableLGraphNode[]

    recreate?(): Promise<LGraphNode>
    onExecutionStart?(): unknown

    /** Callback invoked when the node is dragged over */
    onDragOver?(e: DragEvent): boolean
    /** Callback invoked when something is dropped on the node */
    onDragDrop?(e: DragEvent): Promise<boolean> | boolean

    index?: number
    nodeClass?: string

    /** If the node is a frontend only node */
    isVirtualNode?: boolean

    /** Add a DOM widget to the node */
    addDOMWidget<
      T extends HTMLElement = HTMLElement,
      V extends object | string = string
    >(
      name: string,
      type: string,
      element: T,
      options?: DOMWidgetOptions<V>
    ): DOMWidget<T, V>

    animatedImages?: boolean
    imgs?: HTMLImageElement[]
    images?: NodeExecutionOutput['images']
    videoContainer?: HTMLElement
    isLoading?: boolean
    previewMediaType?: 'image' | 'video' | 'audio' | 'model'
    preview: string[]
    imageIndex?: number | null
    imageRects: Rect[]
    overIndex?: number | null
    pointerDown?: { index: number | null; pos: Point } | null
    setSizeForImage?(force?: boolean): void
    imageOffset?: number
    pasteFile?(file: File): void
    pasteFiles?(files: File[]): void
    widgets_values?: unknown[]
  }

  interface INodeOutputSlot {
    widget?: { name: string; [key: symbol]: unknown }
  }
}
