/**
 * LiteGraph Utility Functions
 *
 * Helper functions for working with LiteGraph nodes and graphs.
 */

import _ from 'es-toolkit/compat'

import type { ColorOption, LGraph } from './src/litegraph'
import {
  LGraphGroup,
  LGraphNode,
  Reroute,
  isColorable
} from './src/litegraph'
import type {
  ExportedSubgraph,
  ISerialisableNodeInput,
  ISerialisedGraph
} from './src/types/serialisation'
import type { IBaseWidget, IComboWidget } from './src/types/widgets'

type ImageNode = LGraphNode & { imgs: HTMLImageElement[] | undefined }
type VideoNode = LGraphNode & {
  videoContainer: HTMLElement | undefined
  imgs: HTMLVideoElement[] | undefined
}

export function isImageNode(node: LGraphNode | undefined): node is ImageNode {
  if (!node) return false
  return (
    (node as any).previewMediaType === 'image' ||
    ((node as any).previewMediaType !== 'video' && !!(node as any).imgs?.length)
  )
}

export function isVideoNode(node: LGraphNode | undefined): node is VideoNode {
  if (!node) return false
  return (node as any).previewMediaType === 'video' || !!(node as any).videoContainer
}

export function isAudioNode(node: LGraphNode | undefined): boolean {
  return !!node && (node as any).previewMediaType === 'audio'
}

export function addToComboValues(widget: IComboWidget, value: string) {
  if (!widget.options) widget.options = { values: [] }
  if (!widget.options.values) widget.options.values = []
  // @ts-expect-error Combo widget values may be a dictionary or legacy function type
  if (!widget.options.values.includes(value)) {
    // @ts-expect-error Combo widget values may be a dictionary or legacy function type
    widget.options.values.push(value)
  }
}

export const isLGraphNode = (item: unknown): item is LGraphNode => {
  return item instanceof LGraphNode
}

export const isLGraphGroup = (item: unknown): item is LGraphGroup => {
  return item instanceof LGraphGroup
}

export const isReroute = (item: unknown): item is Reroute => {
  return item instanceof Reroute
}

/**
 * Get the color option of all canvas items if they are all the same.
 * @param items - The items to get the color option of.
 * @returns The color option of the item.
 */
export const getItemsColorOption = (items: unknown[]): ColorOption | null => {
  const validItems = _.filter(items, isColorable)
  if (_.isEmpty(validItems)) return null

  const colorOptions = _.map(validItems, (item) => item.getColorOption())

  return _.every(colorOptions, (option) =>
    _.isEqual(option, _.head(colorOptions))
  )
    ? _.head(colorOptions)!
    : null
}

export function executeWidgetsCallback(
  nodes: LGraphNode[],
  callbackName: 'onRemove' | 'beforeQueued' | 'afterQueued'
) {
  for (const node of nodes) {
    for (const widget of node.widgets ?? []) {
      (widget as any)[callbackName]?.()
    }
  }
}

/** Input specification interface (simplified from ComfyUI) */
export interface InputSpec {
  name: string
  forceInput?: boolean
  control_after_generate?: boolean
}

/**
 * Migrate widget values to handle forceInput changes.
 *
 * @param inputDefs the input definitions
 * @param widgets the widgets on the node instance
 * @param widgetsValues the widgets values to populate
 * @returns the migrated widgets values
 */
export function migrateWidgetsValues<TWidgetValue>(
  inputDefs: Record<string, InputSpec>,
  widgets: IBaseWidget[],
  widgetsValues: TWidgetValue[]
): TWidgetValue[] {
  const widgetNames = new Set(widgets.map((w) => w.name))
  const originalWidgetsInputs = Object.values(inputDefs).filter(
    (input) => widgetNames.has(input.name) || input.forceInput
  )
  // Count the number of original widgets inputs.
  const numOriginalWidgets = _.sum(
    originalWidgetsInputs.map((input) =>
      // If the input has control, it will have 2 widgets.
      input.control_after_generate ||
      ['seed', 'noise_seed'].includes(input.name)
        ? 2
        : 1
    )
  )

  if (numOriginalWidgets === widgetsValues?.length) {
    return _.zip(originalWidgetsInputs, widgetsValues)
      .filter(([input]) => !input?.forceInput)
      .map(([_, value]) => value as TWidgetValue)
  }
  return widgetsValues
}

/**
 * Fix link input slots after loading a graph. Updates link target_slot indices
 * to match the current node inputs array order.
 *
 * @param graph - The graph to fix links for.
 */
export function fixLinkInputSlots(graph: LGraph) {
  for (const node of graph.nodes) {
    for (const [inputIndex, input] of node.inputs.entries()) {
      const linkId = input.link
      if (!linkId) continue

      const link = graph.links.get(linkId)
      if (!link) continue

      link.target_slot = inputIndex
    }

    // Recursively fix links in subgraphs
    if ((node as any).isSubgraphNode?.() && (node as any).subgraph) {
      fixLinkInputSlots((node as any).subgraph)
    }
  }
}

/**
 * Compress widget input slots by removing all unconnected widget input slots.
 *
 * @param graph - The graph to compress widget input slots for.
 * @throws If an infinite loop is detected.
 */
export function compressWidgetInputSlots(graph: ISerialisedGraph) {
  for (const node of graph.nodes) {
    node.inputs = node.inputs?.filter(matchesLegacyApi)

    for (const [inputIndex, input] of node.inputs?.entries() ?? []) {
      if (input.link) {
        const link = graph.links.find((link) => link[0] === input.link)
        if (link) {
          link[4] = inputIndex
        }
      }
    }
  }

  compressSubgraphWidgetInputSlots(graph.definitions?.subgraphs)
}

function matchesLegacyApi(input: ISerialisableNodeInput) {
  return !(input.widget && input.link === null && !input.label)
}

function compressSubgraphWidgetInputSlots(
  subgraphs: ExportedSubgraph[] | undefined,
  visited = new WeakSet<ExportedSubgraph>()
) {
  if (!subgraphs) return

  for (const subgraph of subgraphs) {
    if (visited.has(subgraph)) throw new Error('Infinite loop detected')
    visited.add(subgraph)

    if (subgraph.nodes) {
      for (const node of subgraph.nodes) {
        node.inputs = node.inputs?.filter(matchesLegacyApi)

        if (!subgraph.links) continue

        for (const [inputIndex, input] of node.inputs?.entries() ?? []) {
          if (input.link) {
            const link = subgraph.links.find((link) => link.id === input.link)
            if (link) link.target_slot = inputIndex
          }
        }
      }
    }

    compressSubgraphWidgetInputSlots(subgraph.definitions?.subgraphs, visited)
  }
}

export function isLoad3dNode(node: LGraphNode) {
  return (
    node &&
    node.type &&
    (node.type === 'Load3D' || node.type === 'Load3DAnimation')
  )
}
