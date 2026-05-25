<template>
  <div ref="containerEl" class="graph-canvas-container">
    <canvas ref="canvasEl" class="litegraph-canvas"></canvas>
    <div v-if="!isReady" class="canvas-loading">
      <span>Initializing canvas...</span>
    </div>

    <!-- Node Context Menu (right-click on node) -->
    <NodeContextMenu
      :is-visible="nodeContextMenu.isVisible"
      :position="nodeContextMenu.position"
      :node="nodeContextMenu.node"
      :node-type="nodeContextMenu.nodeType"
      @close="closeNodeContextMenu"
      @add-field="handleContextMenuAddField"
      @add-method="handleContextMenuAddMethod"
      @rename="handleContextMenuRename"
      @delete="handleContextMenuDelete"
      @view-source="handleContextMenuViewSource"
      @open-in-editor="handleContextMenuOpenInEditor"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, shallowRef, reactive, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { LGraph, LGraphCanvas, LiteGraph, LGraphNode, LLink } from '@/litegraph'
import { registerTrinityNodes } from '@/litegraph/nodes'
import { TrinityNode } from '@/litegraph/nodes/TrinityNodes'
import { useGraphStore, type GraphNode, type GraphLink } from '@/stores/graphStore'
// import { useNodeDefStore } from '@/stores/nodeDefStore' // Currently unused
import { useWorkspaceStore } from '@/stores/workspaceStore'
import { useTypeFilter, TRINITY_TYPES, type FilterableTrinityType } from '@/composables/useTypeFilter'
import { useEventHighlight } from '@/composables/useEventHighlight'
import { useNodeEditing, type TrinityNodeType } from '@/composables/useNodeEditing'
import { CANVAS_CONFIG } from '@/config/flowforge.config'
import NodeContextMenu from '@/components/graph/NodeContextMenu.vue'

// Flag to prevent sync loops
let isSyncing = false
// Flag to debounce syncFromStore calls (undo/redo triggers both stateVersion and nodes.length watchers)
let syncFromStoreScheduled = false

// =============================================================================
// PROPS & EMITS
// =============================================================================

const emit = defineEmits<{
  (e: 'nodeSelected', node: LGraphNode | null): void
  (e: 'nodesSelected', nodes: LGraphNode[]): void
  (e: 'linkCreated', originId: number, originSlot: number, targetId: number, targetSlot: number): void
  (e: 'linkRemoved', linkId: number): void
  (e: 'graphChanged'): void
  (e: 'ready'): void
  (e: 'viewSource', file: string, line?: number): void
  (e: 'openInEditor', file: string, line?: number): void
  (e: 'addFieldRequested', nodeId: string): void
  (e: 'renameRequested', nodeId: string): void
  (e: 'fieldRenameRequested', nodeId: string, fieldIndex: number, fieldName: string): void
}>()

// =============================================================================
// STATE
// =============================================================================

const containerEl = ref<HTMLDivElement | null>(null)
const canvasEl = ref<HTMLCanvasElement | null>(null)

// Use shallowRef for complex objects to prevent deep reactivity
const graph = shallowRef<LGraph | null>(null)
const canvas = shallowRef<LGraphCanvas | null>(null)

const isReady = ref(false)
const resizeObserver = ref<ResizeObserver | null>(null)

// Stores
const graphStore = useGraphStore()
const workspaceStore = useWorkspaceStore()

// Type filter composable
const { visibleTypes } = useTypeFilter()

// Event highlight composable - activates when graph/canvas are ready
const {
  highlightedNodeIds,
  highlightEventNode,
  highlightNodeById,
  clearHighlights,
  isListening: isEventHighlightListening,
} = useEventHighlight({ graph, canvas })

// Node editing composable for context menu actions
const nodeEditing = useNodeEditing({
  onModified: (_event) => {
    emit('graphChanged')
  }
})

// =============================================================================
// NODE CONTEXT MENU STATE
// =============================================================================

interface NodeContextMenuState {
  isVisible: boolean
  position: { x: number; y: number }
  node: GraphNode | null
  nodeType: TrinityNodeType
  lgNode: LGraphNode | null
}

const nodeContextMenu = reactive<NodeContextMenuState>({
  isVisible: false,
  position: { x: 0, y: 0 },
  node: null,
  nodeType: 'component',
  lgNode: null
})

// =============================================================================
// INITIALIZATION
// =============================================================================

onMounted(async () => {
  await initializeCanvas()
})

onUnmounted(() => {
  cleanup()
})

async function initializeCanvas() {
  if (!canvasEl.value || !containerEl.value) return

  try {
    // Register Trinity node types before creating the graph
    registerTrinityNodes()

    // Create the graph
    graph.value = new LGraph()

    // Configure graph defaults
    graph.value.config.align_to_grid = workspaceStore.setting.get('snapToGrid') as boolean ?? true

    // Create the canvas
    canvas.value = new LGraphCanvas(canvasEl.value, graph.value)

    // Configure canvas defaults
    configureCanvas()

    // Set up event listeners
    setupEventListeners()

    // Set up resize observer
    setupResizeObserver()

    // Initial resize to fit container
    await nextTick()
    handleResize()

    // Sync initial state from store
    syncFromStore()

    // Apply type filter to initial nodes
    applyTypeFilter()

    isReady.value = true
    emit('ready')
  } catch (error) {
    console.error('[GraphCanvas] Failed to initialize:', error)
  }
}

function configureCanvas() {
  if (!canvas.value) return

  // Dark theme background
  canvas.value.background_color = CANVAS_CONFIG.backgroundColor
  canvas.value.clear_background_color = CANVAS_CONFIG.backgroundColor

  // Grid settings
  canvas.value.render_canvas_border = false
  canvas.value.render_shadows = true
  canvas.value.render_curved_connections = true
  canvas.value.render_connection_arrows = 'middle_right'

  // Interaction settings
  canvas.value.allow_searchbox = true
  canvas.value.allow_dragnodes = true
  canvas.value.allow_interaction = true

  // Override context menu processing for Trinity nodes
  const originalProcessContextMenu = canvas.value.processContextMenu.bind(canvas.value)
  canvas.value.processContextMenu = (node: LGraphNode | undefined, event: PointerEvent & { canvasX?: number; canvasY?: number }) => {
    // Check if this is a Trinity node that should use our custom context menu
    if (node && isTrinityNode(node)) {
      showNodeContextMenu(node, event)
      return
    }
    // Fall back to default LiteGraph context menu for non-Trinity nodes
    originalProcessContextMenu(node, event as Parameters<typeof originalProcessContextMenu>[1])
  }

  // Start rendering
  canvas.value.startRendering()
}

function setupEventListeners() {
  if (!graph.value || !canvas.value) return

  // Node selection events - LiteGraph uses callback properties, not addEventListener
  canvas.value.onNodeSelected = (node: LGraphNode | null) => {
    emit('nodeSelected', node ?? null)

    if (node) {
      graphStore.selectNode(node.id.toString())
    } else {
      graphStore.clearSelection()
    }
  }

  // Node deselection
  canvas.value.onNodeDeselected = (node: LGraphNode) => {
    // When a node is deselected, check if anything is still selected
    const selectedNodes = canvas.value?.selectedItems ?? []
    if (selectedNodes.length === 0) {
      graphStore.clearSelection()
    }
    emit('nodesSelected', selectedNodes as LGraphNode[])
  }

  // Double-click on Trinity node: rename field if click hit a field name, else open editor
  canvas.value.onNodeDblClicked = (node: LGraphNode, pos?: [number, number]) => {
    if (!isTrinityNode(node)) return

    // Check if the click hit a field name area using the node-local position
    if (node instanceof TrinityNode && pos) {
      const fieldHit = node.getFieldAtPosition(pos[0], pos[1])
      if (fieldHit) {
        emit('fieldRenameRequested', String(node.id), fieldHit.index, fieldHit.field.name)
        return
      }
    }

    // No field hit - open in external editor
    const sourceFile = node.properties?.['sourceFile'] as string | undefined
      ?? (node instanceof TrinityNode ? node.sourceFile : undefined)
    const sourceLine = node.properties?.['sourceLine'] as number | undefined
      ?? (node instanceof TrinityNode ? node.sourceLine : undefined)

    if (sourceFile) {
      emit('openInEditor', sourceFile, sourceLine)
    }
  }

  // Click on empty canvas space should deselect all nodes.
  // LiteGraph does not reliably fire onNodeSelected(null) when clicking empty space.
  if (canvasEl.value) {
    canvasEl.value.addEventListener('pointerdown', handleCanvasPointerDown)
  }

  // Graph change events - LGraph uses onNodeAdded, onNodeRemoved callbacks
  // Guard with isSyncing to prevent history corruption during syncFromStore
  graph.value.onNodeAdded = (node: LGraphNode) => {
    if (isSyncing) return
    syncToStore()
    graphStore.markModified()
    graphStore.pushHistory(`Add node: ${node.title || 'node'}`)
    emit('graphChanged')
  }

  graph.value.onNodeRemoved = (node: LGraphNode) => {
    if (isSyncing) return
    syncToStore()
    graphStore.markModified()
    graphStore.pushHistory(`Remove node: ${node.title || 'node'}`)
    emit('graphChanged')
  }

  // Connection changes
  graph.value.onConnectionChange = () => {
    if (isSyncing) return
    syncToStore()
    graphStore.markModified()
    graphStore.pushHistory('Connection changed')
    emit('graphChanged')
  }
}

/**
 * Handle pointer down on canvas to detect clicks on empty space.
 * If the click did not hit any node, deselect all.
 */
function handleCanvasPointerDown(event: PointerEvent) {
  if (!canvas.value || !graph.value) return

  // Only handle left-click (button 0)
  if (event.button !== 0) return

  // Convert screen coordinates to graph space
  const graphPos = canvas.value.convertEventToCanvasOffset(event) as [number, number]

  // Check if any node is under the pointer
  const nodeAtPos = graph.value.getNodeOnPos(graphPos[0], graphPos[1], graph.value._nodes)

  if (!nodeAtPos) {
    // Clicked empty space - deselect all
    graphStore.clearSelection()
    // Clear LiteGraph's full internal selection state: selectedItems set,
    // selected_nodes hash, current_node, highlighted_links, and each
    // node's .selected flag. This is what removes the visual highlight.
    canvas.value.deselectAll()
    canvas.value.selected_group = null
    // Force full canvas repaint to clear visual highlights
    canvas.value.setDirty(true, true)
    emit('nodeSelected', null)
    emit('nodesSelected', [])
  }
}

function setupResizeObserver() {
  if (!containerEl.value) return

  resizeObserver.value = new ResizeObserver(() => {
    handleResize()
  })

  resizeObserver.value.observe(containerEl.value)
}

function handleResize() {
  if (!canvas.value || !containerEl.value) return

  const rect = containerEl.value.getBoundingClientRect()
  canvas.value.resize(rect.width, rect.height)
}

// =============================================================================
// CONTEXT MENU HANDLING
// =============================================================================

/**
 * Check if a LGraphNode is a Trinity node type
 */
function isTrinityNode(node: LGraphNode): boolean {
  if (node instanceof TrinityNode) return true
  if (node.type?.startsWith('trinity/')) return true
  return false
}

/**
 * Get the Trinity type from a LGraphNode
 */
function getTrinityType(node: LGraphNode): TrinityNodeType {
  if (node instanceof TrinityNode) {
    return node.trinityType as TrinityNodeType
  }
  if (node.type) {
    const match = node.type.match(/^trinity\/(.+)$/)
    if (match && ['component', 'system', 'resource', 'event'].includes(match[1])) {
      return match[1] as TrinityNodeType
    }
  }
  // Default to component
  return 'component'
}

/**
 * Convert LGraphNode to store GraphNode format for context menu
 */
function lgNodeToGraphNodeForMenu(lgNode: LGraphNode): GraphNode {
  return {
    id: String(lgNode.id),
    type: lgNode.type || 'unknown',
    title: lgNode.title || 'Untitled',
    pos: [lgNode.pos[0], lgNode.pos[1]],
    size: [lgNode.size[0], lgNode.size[1]],
    properties: { ...lgNode.properties },
    flags: lgNode.flags ? { ...lgNode.flags } : {}
  }
}

/**
 * Show the node context menu at the mouse position
 */
function showNodeContextMenu(node: LGraphNode, event: MouseEvent) {
  event.preventDefault()
  event.stopPropagation()

  nodeContextMenu.isVisible = true
  nodeContextMenu.position = { x: event.clientX, y: event.clientY }
  nodeContextMenu.lgNode = node
  nodeContextMenu.node = lgNodeToGraphNodeForMenu(node)
  nodeContextMenu.nodeType = getTrinityType(node)
}

/**
 * Close the node context menu
 */
function closeNodeContextMenu() {
  nodeContextMenu.isVisible = false
  nodeContextMenu.lgNode = null
  nodeContextMenu.node = null
}

/**
 * Handle Add Field action from context menu
 */
function handleContextMenuAddField() {
  if (!nodeContextMenu.node) return
  emit('addFieldRequested', nodeContextMenu.node.id)
  closeNodeContextMenu()
}

/**
 * Handle Add Method action from context menu (for system nodes)
 */
function handleContextMenuAddMethod() {
  if (!nodeContextMenu.node) return
  // For now, treat add method same as add field
  // The dialog can differentiate based on node type
  emit('addFieldRequested', nodeContextMenu.node.id)
  closeNodeContextMenu()
}

/**
 * Handle Rename action from context menu
 */
function handleContextMenuRename() {
  if (!nodeContextMenu.node) return
  emit('renameRequested', nodeContextMenu.node.id)
  closeNodeContextMenu()
}

/**
 * Handle Delete action from context menu
 */
async function handleContextMenuDelete() {
  if (!nodeContextMenu.node) return

  const nodeId = nodeContextMenu.node.id
  closeNodeContextMenu()

  // Delete using node editing composable
  const success = await nodeEditing.deleteNode(nodeId)
  if (success) {
    // Also remove from LiteGraph if still present
    if (graph.value) {
      const lgNode = graph.value.getNodeById(parseInt(nodeId))
      if (lgNode) {
        graph.value.remove(lgNode)
      }
    }
  }
}

/**
 * Handle View Source action from context menu
 */
function handleContextMenuViewSource() {
  if (!nodeContextMenu.node?.properties) return

  const sourceFile = nodeContextMenu.node.properties['sourceFile'] as string | undefined
  const sourceLine = nodeContextMenu.node.properties['sourceLine'] as number | undefined

  if (sourceFile) {
    emit('viewSource', sourceFile, sourceLine)
  }
  closeNodeContextMenu()
}

/**
 * Handle Open in Editor action from context menu
 */
function handleContextMenuOpenInEditor() {
  if (!nodeContextMenu.node?.properties) return

  const sourceFile = nodeContextMenu.node.properties['sourceFile'] as string | undefined
  const sourceLine = nodeContextMenu.node.properties['sourceLine'] as number | undefined

  if (sourceFile) {
    emit('openInEditor', sourceFile, sourceLine)
  }
  closeNodeContextMenu()
}

// =============================================================================
// STORE SYNC
// =============================================================================

/**
 * Convert a LiteGraph node to store format.
 */
function lgNodeToStoreNode(lgNode: LGraphNode): GraphNode {
  const storeNode: GraphNode = {
    id: String(lgNode.id),
    type: lgNode.type || 'unknown',
    title: lgNode.title || 'Untitled',
    pos: [lgNode.pos[0], lgNode.pos[1]],
    size: [lgNode.size[0], lgNode.size[1]],
    properties: lgNode.properties ? JSON.parse(JSON.stringify(lgNode.properties)) : {},
    flags: {}
  }

  // Copy flags
  if (lgNode.flags) {
    if (lgNode.flags.collapsed !== undefined) storeNode.flags!.collapsed = lgNode.flags.collapsed
    if (lgNode.flags.pinned !== undefined) storeNode.flags!.pinned = lgNode.flags.pinned
  }

  // Copy inputs
  if (lgNode.inputs && lgNode.inputs.length > 0) {
    storeNode.inputs = lgNode.inputs.map(input => ({
      name: input.name,
      type: String(input.type || '*'),
      link: input.link != null ? String(input.link) : null
    }))
  }

  // Copy outputs
  if (lgNode.outputs && lgNode.outputs.length > 0) {
    storeNode.outputs = lgNode.outputs.map(output => ({
      name: output.name,
      type: String(output.type || '*'),
      links: output.links?.map(l => String(l)) || []
    }))
  }

  // Copy Trinity-specific properties (deep clone arrays/objects to prevent reference sharing)
  if (lgNode instanceof TrinityNode) {
    storeNode.properties = storeNode.properties || {}
    storeNode.properties['trinityType'] = lgNode.trinityType
    storeNode.properties['className'] = lgNode.className
    storeNode.properties['fields'] = lgNode.fields ? JSON.parse(JSON.stringify(lgNode.fields)) : []
    storeNode.properties['sourceFile'] = lgNode.sourceFile
    storeNode.properties['sourceLine'] = lgNode.sourceLine

    // Copy SystemNode methods and queries
    if ('methods' in lgNode && (lgNode as any).methods) {
      storeNode.properties['methods'] = JSON.parse(JSON.stringify((lgNode as any).methods))
    }
    if ('queries' in lgNode && (lgNode as any).queries) {
      storeNode.properties['queries'] = JSON.parse(JSON.stringify((lgNode as any).queries))
    }
    // Copy EventNode payload fields
    if ('payloadFields' in lgNode && (lgNode as any).payloadFields) {
      storeNode.properties['payloadFields'] = JSON.parse(JSON.stringify((lgNode as any).payloadFields))
    }
  }

  return storeNode
}

/**
 * Convert a LiteGraph link to store format.
 */
function lgLinkToStoreLink(lgLink: LLink): GraphLink {
  return {
    id: String(lgLink.id),
    originId: String(lgLink.origin_id),
    originSlot: lgLink.origin_slot,
    targetId: String(lgLink.target_id),
    targetSlot: lgLink.target_slot,
    type: String(lgLink.type || '*')
  }
}

/**
 * Sync LiteGraph state to Pinia store.
 * Serializes the current graph state and updates the store.
 */
function syncToStore() {
  if (!graph.value || isSyncing) return

  try {
    isSyncing = true

    // Convert all nodes
    const nodes: GraphNode[] = []
    for (const lgNode of graph.value._nodes || []) {
      nodes.push(lgNodeToStoreNode(lgNode))
    }

    // Convert all links
    const links: GraphLink[] = []
    for (const linkId in graph.value.links) {
      const lgLink = graph.value.links[linkId]
      if (lgLink) {
        links.push(lgLinkToStoreLink(lgLink))
      }
    }

    // Calculate next IDs
    let maxNodeId = 0
    let maxLinkId = 0
    for (const node of nodes) {
      const numId = parseInt(node.id.replace(/\D/g, '')) || 0
      maxNodeId = Math.max(maxNodeId, numId)
    }
    for (const link of links) {
      const numId = parseInt(link.id.replace(/\D/g, '')) || 0
      maxLinkId = Math.max(maxLinkId, numId)
    }

    // Update store state directly (avoid triggering loadGraphState which clears history)
    graphStore.nodes.splice(0, graphStore.nodes.length, ...nodes)
    graphStore.links.splice(0, graphStore.links.length, ...links)

  } catch (error) {
    console.error('[GraphCanvas] Failed to sync to store:', error)
  } finally {
    isSyncing = false
  }
}

/**
 * Sync Pinia store state to LiteGraph.
 * Restores nodes and links from the store.
 */
function syncFromStore() {
  if (!graph.value || isSyncing) return

  const state = graphStore.getGraphState()
  console.log(`[GraphCanvas] syncFromStore: ${state.nodes.length} nodes, ${state.links.length} links, canvas nodes: ${graph.value._nodes?.length ?? 0}`)

  if (state.nodes.length === 0 && state.links.length === 0 && (!graph.value._nodes || graph.value._nodes.length === 0)) {
    return // Both store and canvas are empty, nothing to sync
  }

  try {
    isSyncing = true

    // Clear existing graph
    graph.value.clear()

    // Map to track store ID -> LiteGraph node ID
    const nodeIdMap = new Map<string, number>()

    // Restore nodes
    for (const storeNode of state.nodes) {
      let lgNode: LGraphNode | null = null

      // Check if type is registered
      const nodeType = LiteGraph.registered_node_types[storeNode.type]
      if (nodeType) {
        lgNode = LiteGraph.createNode(storeNode.type)
      } else {
        // Try creating a generic node if type not found
        console.warn(`[GraphCanvas] Unknown node type: ${storeNode.type}, creating generic node`)
        lgNode = new LGraphNode(storeNode.title)
        lgNode.type = storeNode.type
      }

      if (!lgNode) continue

      // Set basic properties
      lgNode.pos = [...storeNode.pos] as [number, number]
      lgNode.size = [...storeNode.size] as [number, number]
      lgNode.title = storeNode.title

      // Set flags
      if (storeNode.flags) {
        lgNode.flags = { ...storeNode.flags }
      }

      // Set properties (deep clone to prevent mutation of history state)
      if (storeNode.properties) {
        lgNode.properties = JSON.parse(JSON.stringify(storeNode.properties))
      }

      // Restore Trinity-specific properties
      if (lgNode instanceof TrinityNode && storeNode.properties) {
        const props = storeNode.properties
        if (props['trinityType']) lgNode.trinityType = props['trinityType'] as TrinityNode['trinityType']
        if (props['className']) lgNode.className = props['className'] as string
        if (props['fields']) lgNode.fields = props['fields'] as TrinityNode['fields']
        if (props['sourceFile'] !== undefined) lgNode.sourceFile = props['sourceFile'] as string
        if (props['sourceLine'] !== undefined) lgNode.sourceLine = props['sourceLine'] as number

        // Restore SystemNode-specific properties
        if ('methods' in lgNode && props['methods']) {
          (lgNode as any).methods = props['methods']
        }
        if ('queries' in lgNode && props['queries']) {
          (lgNode as any).queries = props['queries']
        }
        // Restore EventNode-specific properties
        // EventNode renders via payloadFields, but Python sends fields as 'fields'
        if ('payloadFields' in lgNode) {
          if (props['payloadFields']) {
            (lgNode as any).payloadFields = props['payloadFields']
          } else if (props['fields']) {
            // Event payload fields come through as regular fields from Python
            (lgNode as any).payloadFields = props['fields']
          }
        }
      }

      // Clear default slots and add from store
      lgNode.inputs = []
      lgNode.outputs = []

      if (storeNode.inputs) {
        for (const input of storeNode.inputs) {
          lgNode.addInput(input.name, input.type)
        }
      }

      if (storeNode.outputs) {
        for (const output of storeNode.outputs) {
          lgNode.addOutput(output.name, output.type)
        }
      }

      // Add to graph and track ID mapping
      graph.value.add(lgNode)
      nodeIdMap.set(storeNode.id, lgNode.id)
    }

    // Restore links
    for (const storeLink of state.links) {
      const originLgId = nodeIdMap.get(storeLink.originId)
      const targetLgId = nodeIdMap.get(storeLink.targetId)

      if (originLgId === undefined || targetLgId === undefined) {
        console.warn(`[GraphCanvas] Could not restore link: missing node. Origin: ${storeLink.originId}, Target: ${storeLink.targetId}`)
        continue
      }

      const originNode = graph.value.getNodeById(originLgId)
      const targetNode = graph.value.getNodeById(targetLgId)

      if (!originNode || !targetNode) {
        console.warn(`[GraphCanvas] Could not find nodes for link restoration`)
        continue
      }

      // Connect the nodes
      originNode.connect(storeLink.originSlot, targetNode, storeLink.targetSlot)
    }

    // Update store node IDs to match LiteGraph's assigned numeric IDs.
    // This is critical: LiteGraph assigns its own numeric IDs when graph.add() is called,
    // and onNodeSelected reports those IDs. The store must use the same IDs so selection works.
    for (const [storeId, lgId] of nodeIdMap) {
      const storeNode = graphStore.nodes.find(n => n.id === storeId)
      if (storeNode) {
        storeNode.id = String(lgId)
      }
    }

    // Also update link origin/target IDs to match the new node IDs
    for (const storeLink of graphStore.links) {
      const newOriginId = nodeIdMap.get(storeLink.originId)
      const newTargetId = nodeIdMap.get(storeLink.targetId)
      if (newOriginId !== undefined) storeLink.originId = String(newOriginId)
      if (newTargetId !== undefined) storeLink.targetId = String(newTargetId)
    }

    // Refresh canvas
    if (canvas.value) {
      canvas.value.setDirty(true, true)
    }

    console.log(`[GraphCanvas] Synced from store: ${state.nodes.length} nodes, ${state.links.length} links`)

    // Apply type filter after syncing
    applyTypeFilter()

  } catch (error) {
    console.error('[GraphCanvas] Failed to sync from store:', error)
  } finally {
    isSyncing = false
    // Clear the restoring flag now that the full undo/redo + rebuild cycle is complete.
    // This allows pushHistory to work again for normal user operations.
    if (graphStore.isRestoringState) {
      graphStore.isRestoringState = false
    }
  }
}

// Watch for store changes
watch(
  () => graphStore.canvasOffset,
  (offset) => {
    if (canvas.value) {
      canvas.value.ds.offset = offset
      canvas.value.setDirty(true, true)
    }
  }
)

watch(
  () => graphStore.canvasScale,
  (scale) => {
    if (canvas.value) {
      canvas.value.ds.scale = scale
      canvas.value.setDirty(true, true)
    }
  }
)

/**
 * Schedule a debounced syncFromStore call via nextTick.
 * Prevents multiple watchers (nodes.length + stateVersion) from triggering
 * duplicate syncs during undo/redo, which could cause the second clear+rebuild
 * to race and produce an empty canvas.
 */
function scheduleSyncFromStore() {
  if (isSyncing || syncFromStoreScheduled) return
  syncFromStoreScheduled = true
  nextTick(() => {
    syncFromStoreScheduled = false
    syncFromStore()
  })
}

// Watch for external node changes (e.g., file loaded via API)
// Use deep watch on nodes array length to detect load operations
watch(
  () => graphStore.nodes.length,
  (newLength, oldLength) => {
    // Only sync if not currently syncing (prevents loops),
    // not restoring state (undo/redo handles sync via stateVersion),
    // and if the change came from outside (e.g., loadFromPythonFile)
    if (!isSyncing && !graphStore.isRestoringState && newLength !== oldLength) {
      scheduleSyncFromStore()
    }
  }
)

// Watch for undo/redo state restoration (stateVersion changes)
watch(
  () => graphStore.stateVersion,
  () => {
    if (!isSyncing) {
      scheduleSyncFromStore()
    }
  }
)

// =============================================================================
// TYPE FILTER
// =============================================================================

/**
 * Apply type filter to all nodes in the graph.
 * Sets visibility based on the current filter state.
 */
function applyTypeFilter() {
  if (!graph.value || !canvas.value) return

  const nodes = graph.value._nodes
  if (!nodes) return

  for (const node of nodes) {
    // Determine the Trinity type of this node
    let trinityType: FilterableTrinityType | null = null

    if (node instanceof TrinityNode) {
      trinityType = node.trinityType as FilterableTrinityType
    } else if (node.type) {
      // Check if it's a trinity node type (e.g., 'trinity/component')
      const typeMatch = node.type.match(/^trinity\/(.+)$/)
      if (typeMatch && TRINITY_TYPES.includes(typeMatch[1] as FilterableTrinityType)) {
        trinityType = typeMatch[1] as FilterableTrinityType
      }
    }

    if (trinityType) {
      // Apply visibility based on filter state
      const shouldBeVisible = visibleTypes[trinityType]

      // Use flags to control visibility
      if (!node.flags) {
        node.flags = {}
      }

      // Set custom hidden flag - nodes with this flag are not rendered
      node.flags.hidden = !shouldBeVisible

      // Also set a custom property for any custom rendering logic
      node.properties = node.properties || {}
      node.properties['_typeFilterHidden'] = !shouldBeVisible
    }
  }

  // Refresh canvas to reflect changes
  canvas.value.setDirty(true, true)
}

// Watch for type filter changes
watch(
  () => ({ ...visibleTypes }),
  () => {
    applyTypeFilter()
  },
  { deep: true }
)

// Also watch for currentFilePath changes (indicates file load)
watch(
  () => graphStore.currentFilePath,
  (newPath, oldPath) => {
    if (!isSyncing && !graphStore.isRestoringState && newPath !== oldPath && newPath !== null) {
      scheduleSyncFromStore()
    }
  }
)

// =============================================================================
// PUBLIC API
// =============================================================================

/**
 * Add a node to the graph
 */
function addNode(type: string, pos?: [number, number]): LGraphNode | null {
  if (!graph.value) return null

  const node = LiteGraph.createNode(type)
  if (!node) {
    console.warn(`[GraphCanvas] Unknown node type: ${type}`)
    return null
  }

  if (pos) {
    node.pos = pos
  } else if (canvas.value) {
    // Place at center of canvas
    const center = canvas.value.convertOffsetToCanvas([
      canvas.value.canvas.width / 2,
      canvas.value.canvas.height / 2
    ])
    node.pos = center as [number, number]
  }

  graph.value.add(node)
  return node
}

/**
 * Remove a node from the graph
 */
function removeNode(nodeId: number | string) {
  if (!graph.value) return

  const node = graph.value.getNodeById(typeof nodeId === 'string' ? parseInt(nodeId) : nodeId)
  if (node) {
    graph.value.remove(node)
  }
}

/**
 * Clear the entire graph
 */
function clearGraph() {
  if (!graph.value) return
  graph.value.clear()
  graphStore.clearGraph()
}

/**
 * Serialize the graph to JSON
 */
function serializeGraph(): object | null {
  if (!graph.value) return null
  return graph.value.serialize()
}

/**
 * Load a graph from JSON
 */
function loadGraph(data: object) {
  if (!graph.value) return
  graph.value.configure(data)
  syncToStore()
}

/**
 * Zoom to fit all nodes
 */
function zoomToFit() {
  if (!canvas.value || !graph.value) return

  const nodes = graph.value._nodes
  if (!nodes || nodes.length === 0) return

  // Calculate bounding box
  let minX = Infinity, minY = Infinity
  let maxX = -Infinity, maxY = -Infinity

  for (const node of nodes) {
    minX = Math.min(minX, node.pos[0])
    minY = Math.min(minY, node.pos[1])
    maxX = Math.max(maxX, node.pos[0] + node.size[0])
    maxY = Math.max(maxY, node.pos[1] + node.size[1])
  }

  const padding = CANVAS_CONFIG.zoomFitPadding
  canvas.value.ds.offset = [
    -(minX + maxX) / 2 + canvas.value.canvas.width / 2,
    -(minY + maxY) / 2 + canvas.value.canvas.height / 2
  ]

  // Calculate scale to fit
  const graphWidth = maxX - minX + padding * 2
  const graphHeight = maxY - minY + padding * 2
  const scaleX = canvas.value.canvas.width / graphWidth
  const scaleY = canvas.value.canvas.height / graphHeight
  canvas.value.ds.scale = Math.min(scaleX, scaleY, 1)

  canvas.value.setDirty(true, true)
}

/**
 * Reset view to center
 */
function resetView() {
  if (!canvas.value) return
  canvas.value.ds.reset()
  canvas.value.setDirty(true, true)
  graphStore.resetView()
}

// =============================================================================
// CLEANUP
// =============================================================================

function cleanup() {
  // Remove canvas pointer listener
  if (canvasEl.value) {
    canvasEl.value.removeEventListener('pointerdown', handleCanvasPointerDown)
  }

  if (resizeObserver.value) {
    resizeObserver.value.disconnect()
    resizeObserver.value = null
  }

  if (canvas.value) {
    canvas.value.stopRendering()
    canvas.value = null
  }

  if (graph.value) {
    graph.value.clear()
    graph.value = null
  }
}

// =============================================================================
// EXPOSE
// =============================================================================

defineExpose({
  graph,
  canvas,
  isReady,
  addNode,
  removeNode,
  clearGraph,
  serializeGraph,
  loadGraph,
  zoomToFit,
  resetView,
  syncToStore,
  syncFromStore,
  applyTypeFilter,
  // Event highlighting
  highlightedNodeIds,
  highlightEventNode,
  highlightNodeById,
  clearHighlights,
  isEventHighlightListening,
  // Context menu
  showNodeContextMenu,
  closeNodeContextMenu,
  nodeContextMenu,
  // Node editing
  nodeEditing,
})
</script>

<style scoped>
.graph-canvas-container {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  /* Uses --canvas-bg CSS variable from :root, falls back to dark theme */
  background-color: var(--canvas-bg, #1a1a2e);
}

.litegraph-canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  outline: none;
}

.canvas-loading {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: var(--text-muted, #666);
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}
</style>
