/**
 * Graph Store - Node graph state management for FlowForge
 * New store for managing Python visual scripting graph state
 */
import { defineStore } from 'pinia'
import { computed, ref, shallowRef, watch } from 'vue'
import { getApi } from '@/services'
import type { NodeGraph, GraphNode as ApiGraphNode, GraphEdge } from '@/services'
import { CANVAS_CONFIG, NODE_CONFIG, UI_CONFIG } from '@/config/flowforge.config'
import { getFileInfo } from '@/bridge/files'

export interface GraphNode {
  id: string
  type: string
  title: string
  pos: [number, number]
  size: [number, number]
  inputs?: GraphNodeSlot[]
  outputs?: GraphNodeSlot[]
  properties?: Record<string, unknown>
  widgets?: GraphWidget[]
  flags?: {
    collapsed?: boolean
    pinned?: boolean
  }
}

export interface GraphNodeSlot {
  name: string
  type: string
  link?: string | null
  links?: string[]
}

export interface GraphWidget {
  name: string
  type: string
  value: unknown
  options?: Record<string, unknown>
}

export interface GraphLink {
  id: string
  originId: string
  originSlot: number
  targetId: string
  targetSlot: number
  type: string
}

export interface GraphGroup {
  id: string
  title: string
  color: string
  bounds: [number, number, number, number] // x, y, width, height
  nodes: string[] // node IDs in this group
}

export interface GraphState {
  nodes: GraphNode[]
  links: GraphLink[]
  groups: GraphGroup[]
  lastNodeId: number
  lastLinkId: number
  version: number
}

export interface GraphHistoryEntry {
  timestamp: number
  description: string
  state: GraphState
}

export const useGraphStore = defineStore('graph', () => {
  // Current graph state
  const nodes = ref<GraphNode[]>([])
  const links = ref<GraphLink[]>([])
  const groups = ref<GraphGroup[]>([])
  const lastNodeId = ref(0)
  const lastLinkId = ref(0)

  // Selection state
  const selectedNodeIds = ref<Set<string>>(new Set())
  const selectedLinkIds = ref<Set<string>>(new Set())

  // File state
  const currentFilePath = ref<string | null>(null)
  const isModified = ref(false)
  const fileName = computed(() => {
    if (!currentFilePath.value) return 'Untitled'
    const parts = currentFilePath.value.split('/')
    return parts[parts.length - 1]
  })

  // File watcher state for external change detection
  const lastKnownMtime = ref<number | null>(null)
  const hasExternalChanges = ref(false)
  const fileWatcherIntervalId = ref<ReturnType<typeof setInterval> | null>(null)
  const FILE_WATCH_POLL_INTERVAL = 2000

  // Canvas state
  const canvasOffset = ref<[number, number]>([0, 0])
  const canvasScale = ref(1)

  // History for undo/redo
  const history = shallowRef<GraphHistoryEntry[]>([])
  const historyIndex = ref(-1)
  const maxHistorySize = UI_CONFIG.history.maxSize

  // Version counter - incremented on undo/redo to notify canvas to re-sync
  const stateVersion = ref(0)
  // Flag to indicate a state restore is in progress (undo/redo).
  // GraphCanvas watchers should ignore changes while this is true
  // and wait for the explicit stateVersion bump to trigger one syncFromStore.
  const isRestoringState = ref(false)
  // When true, pushHistory() is suppressed. Used for batch operations (e.g., multi-node delete)
  // so that only one history entry is created for the whole operation.
  const suppressHistory = ref(false)

  // Computed properties
  const selectedNodes = computed(() =>
    nodes.value.filter((n) => selectedNodeIds.value.has(n.id))
  )

  const selectedLinks = computed(() =>
    links.value.filter((l) => selectedLinkIds.value.has(l.id))
  )

  const hasSelection = computed(() =>
    selectedNodeIds.value.size > 0 || selectedLinkIds.value.size > 0
  )

  const canUndo = computed(() => historyIndex.value > 0)
  const canRedo = computed(() => historyIndex.value < history.value.length - 1)

  // Node operations
  function addNode(node: Omit<GraphNode, 'id'>): GraphNode {
    lastNodeId.value++
    const newNode: GraphNode = {
      ...node,
      id: `node_${lastNodeId.value}`
    }
    nodes.value = [...nodes.value, newNode]
    markModified()
    pushHistory(`Add node: ${newNode.title}`)
    return newNode
  }

  function removeNode(nodeId: string) {
    const node = nodes.value.find((n) => n.id === nodeId)
    if (!node) return

    // Remove all links connected to this node
    links.value = links.value.filter(
      (l) => l.originId !== nodeId && l.targetId !== nodeId
    )

    // Remove the node
    nodes.value = nodes.value.filter((n) => n.id !== nodeId)

    // Remove from groups
    groups.value.forEach((g) => {
      g.nodes = g.nodes.filter((id) => id !== nodeId)
    })

    // Remove from selection
    selectedNodeIds.value.delete(nodeId)

    markModified()
    pushHistory(`Remove node: ${node.title}`)
  }

  function updateNode(nodeId: string, updates: Partial<GraphNode>) {
    const index = nodes.value.findIndex((n) => n.id === nodeId)
    if (index === -1) return

    nodes.value[index] = { ...nodes.value[index], ...updates }
    markModified()
  }

  function moveNode(nodeId: string, pos: [number, number]) {
    updateNode(nodeId, { pos })
  }

  // Link operations
  function addLink(link: Omit<GraphLink, 'id'>): GraphLink {
    lastLinkId.value++
    const newLink: GraphLink = {
      ...link,
      id: `link_${lastLinkId.value}`
    }
    links.value = [...links.value, newLink]
    markModified()
    pushHistory('Add connection')
    return newLink
  }

  function removeLink(linkId: string) {
    links.value = links.value.filter((l) => l.id !== linkId)
    selectedLinkIds.value.delete(linkId)
    markModified()
    pushHistory('Remove connection')
  }

  /**
   * Get all links connected to a specific node.
   * Returns links where the node is either the origin or target.
   */
  function getConnectedEdges(nodeId: string): GraphLink[] {
    return links.value.filter(
      (link) => link.originId === nodeId || link.targetId === nodeId
    )
  }

  // Selection operations
  function selectNode(nodeId: string, additive = false) {
    if (!additive) {
      selectedNodeIds.value.clear()
      selectedLinkIds.value.clear()
    }
    selectedNodeIds.value.add(nodeId)
  }

  function deselectNode(nodeId: string) {
    selectedNodeIds.value.delete(nodeId)
  }

  function selectLink(linkId: string, additive = false) {
    if (!additive) {
      selectedNodeIds.value.clear()
      selectedLinkIds.value.clear()
    }
    selectedLinkIds.value.add(linkId)
  }

  function selectAll() {
    nodes.value.forEach((n) => selectedNodeIds.value.add(n.id))
    links.value.forEach((l) => selectedLinkIds.value.add(l.id))
  }

  function clearSelection() {
    selectedNodeIds.value.clear()
    selectedLinkIds.value.clear()
  }

  function deleteSelected() {
    // Delete selected links first
    selectedLinkIds.value.forEach((id) => {
      links.value = links.value.filter((l) => l.id !== id)
    })

    // Delete selected nodes (this also removes their links)
    selectedNodeIds.value.forEach((id) => {
      const node = nodes.value.find((n) => n.id === id)
      if (node) {
        links.value = links.value.filter(
          (l) => l.originId !== id && l.targetId !== id
        )
        nodes.value = nodes.value.filter((n) => n.id !== id)
      }
    })

    clearSelection()
    markModified()
    pushHistory('Delete selected')
  }

  // Group operations
  function createGroup(title: string, nodeIds: string[]): GraphGroup {
    const groupNodes = nodes.value.filter((n) => nodeIds.includes(n.id))
    if (groupNodes.length === 0) {
      throw new Error('Cannot create empty group')
    }

    // Calculate bounds
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
    groupNodes.forEach((n) => {
      minX = Math.min(minX, n.pos[0])
      minY = Math.min(minY, n.pos[1])
      maxX = Math.max(maxX, n.pos[0] + n.size[0])
      maxY = Math.max(maxY, n.pos[1] + n.size[1])
    })

    const padding = CANVAS_CONFIG.groupPadding
    const group: GraphGroup = {
      id: `group_${Date.now()}`,
      title,
      color: CANVAS_CONFIG.defaultGroupColor,
      bounds: [minX - padding, minY - padding, maxX - minX + padding * 2, maxY - minY + padding * 2],
      nodes: nodeIds
    }

    groups.value = [...groups.value, group]
    markModified()
    pushHistory(`Create group: ${title}`)
    return group
  }

  function removeGroup(groupId: string) {
    groups.value = groups.value.filter((g) => g.id !== groupId)
    markModified()
    pushHistory('Remove group')
  }

  // File operations
  function setCurrentFile(filePath: string | null) {
    currentFilePath.value = filePath
    isModified.value = false
  }

  function markModified() {
    isModified.value = true
  }

  function markSaved() {
    isModified.value = false
  }

  // Serialization
  function getGraphState(): GraphState {
    return {
      nodes: JSON.parse(JSON.stringify(nodes.value)),
      links: JSON.parse(JSON.stringify(links.value)),
      groups: JSON.parse(JSON.stringify(groups.value)),
      lastNodeId: lastNodeId.value,
      lastLinkId: lastLinkId.value,
      version: 1
    }
  }

  function loadGraphState(state: GraphState) {
    nodes.value = state.nodes || []
    links.value = state.links || []
    groups.value = state.groups || []
    lastNodeId.value = state.lastNodeId || 0
    lastLinkId.value = state.lastLinkId || 0
    clearSelection()
    history.value = []
    historyIndex.value = -1
    pushHistory('Load graph')
  }

  function clearGraph() {
    nodes.value = []
    links.value = []
    groups.value = []
    lastNodeId.value = 0
    lastLinkId.value = 0
    clearSelection()
    currentFilePath.value = null
    isModified.value = false
    history.value = []
    historyIndex.value = -1
    pushHistory('New graph')
  }

  // History operations
  function pushHistory(description: string) {
    // Suppress history during undo/redo restore cycle or batch operations
    if (isRestoringState.value || suppressHistory.value) return

    // Remove any redo history
    if (historyIndex.value < history.value.length - 1) {
      history.value = history.value.slice(0, historyIndex.value + 1)
    }

    // Add new entry
    const entry: GraphHistoryEntry = {
      timestamp: Date.now(),
      description,
      state: JSON.parse(JSON.stringify(getGraphState()))
    }

    history.value = [...history.value, entry]

    // Limit history size
    if (history.value.length > maxHistorySize) {
      history.value = history.value.slice(-maxHistorySize)
    }

    historyIndex.value = history.value.length - 1
  }

  function undo(): GraphHistoryEntry | null {
    if (!canUndo.value) return null

    historyIndex.value--
    const entry = history.value[historyIndex.value]
    if (entry) {
      restoreState(entry.state)
      return entry
    }
    return null
  }

  function redo(): GraphHistoryEntry | null {
    if (!canRedo.value) return null

    historyIndex.value++
    const entry = history.value[historyIndex.value]
    console.log(`[graphStore] redo: index=${historyIndex.value}, entry="${entry?.description}", nodes=${entry?.state?.nodes?.length ?? 'N/A'}, historyLen=${history.value.length}`)
    if (entry) {
      restoreState(entry.state)
      return entry
    }
    return null
  }

  function restoreState(state: GraphState) {
    const restoredNodes = JSON.parse(JSON.stringify(state.nodes))
    const restoredLinks = JSON.parse(JSON.stringify(state.links))
    const restoredGroups = JSON.parse(JSON.stringify(state.groups))

    console.log(`[graphStore] restoreState: ${restoredNodes.length} nodes, ${restoredLinks.length} links, historyIndex=${historyIndex.value}`)

    // Set flag so GraphCanvas watchers on nodes.length ignore intermediate changes
    isRestoringState.value = true

    nodes.value = restoredNodes
    links.value = restoredLinks
    groups.value = restoredGroups
    lastNodeId.value = state.lastNodeId
    lastLinkId.value = state.lastLinkId
    isModified.value = true

    // Bump stateVersion to trigger syncFromStore. Keep isRestoringState=true so that
    // pushHistory is suppressed during the entire rebuild cycle. GraphCanvas.syncFromStore
    // will clear isRestoringState when it finishes.
    stateVersion.value++
  }

  /**
   * Clear all undo/redo history.
   */
  function clearHistory() {
    history.value = []
    historyIndex.value = -1
  }

  /**
   * Get the current history for debugging/display.
   */
  function getHistoryList(): GraphHistoryEntry[] {
    return [...history.value]
  }

  /**
   * Get the current position in history.
   */
  function getHistoryIndex(): number {
    return historyIndex.value
  }

  /**
   * Get the description of what would be undone.
   */
  function getUndoDescription(): string | null {
    if (!canUndo.value) return null
    return history.value[historyIndex.value]?.description ?? null
  }

  /**
   * Get the description of what would be redone.
   */
  function getRedoDescription(): string | null {
    if (!canRedo.value) return null
    return history.value[historyIndex.value + 1]?.description ?? null
  }

  /**
   * Create a snapshot of the current state (for use with batch operations).
   */
  function createSnapshot(): GraphState {
    return JSON.parse(JSON.stringify(getGraphState()))
  }

  /**
   * Restore from a snapshot.
   */
  function restoreFromSnapshot(snapshot: GraphState, description?: string) {
    restoreState(snapshot)
    if (description) {
      pushHistory(description)
    }
  }

  /**
   * Execute multiple operations as a single undo step.
   */
  function batch(operations: () => void, description: string) {
    // Capture state before operations
    const beforeState = createSnapshot()

    // Temporarily disable history
    const originalPushHistory = pushHistory
    let historyPushed = false
    const tempPushHistory = () => {
      historyPushed = true
    }

    // Replace pushHistory temporarily (operations will call it)
    // We want to batch all changes into one history entry
    try {
      operations()
    } finally {
      // If any history was pushed, add a single entry for all changes
      if (historyPushed || JSON.stringify(beforeState) !== JSON.stringify(getGraphState())) {
        pushHistory(description)
      }
    }
  }

  // Canvas operations
  function setCanvasOffset(offset: [number, number]) {
    canvasOffset.value = offset
  }

  function setCanvasScale(scale: number) {
    canvasScale.value = Math.max(CANVAS_CONFIG.minScale, Math.min(CANVAS_CONFIG.maxScale, scale))
  }

  function resetView() {
    canvasOffset.value = [0, 0]
    canvasScale.value = 1
  }

  // ===========================================================================
  // API-Connected File Operations
  // ===========================================================================

  /**
   * Convert API GraphNode to store GraphNode format.
   */
  function apiNodeToStoreNode(apiNode: ApiGraphNode, index: number): GraphNode {
    // Map API type to registered LiteGraph type name (e.g., 'component' -> 'trinity/component')
    const trinityType = `trinity/${apiNode.type}`

    // Build inputs/outputs based on type so edges have slots to connect to
    const inputs: Array<{ name: string; type: string; link: string | null }> = []
    const outputs: Array<{ name: string; type: string; links: string[] }> = []

    // Every node gets at least one input and one output so edges can connect.
    // Use wildcard type '*' so that cross-type Trinity relationship edges
    // (e.g. component -> system query) pass LiteGraph's isValidConnection check.
    switch (apiNode.type) {
      case 'component':
        inputs.push({ name: 'in', type: '*', link: null })
        outputs.push({ name: 'data', type: '*', links: [] })
        break
      case 'system':
        inputs.push({ name: 'trigger', type: '*', link: null })
        outputs.push({ name: 'next', type: '*', links: [] })
        break
      case 'resource':
        inputs.push({ name: 'in', type: '*', link: null })
        outputs.push({ name: 'data', type: '*', links: [] })
        break
      case 'event':
        inputs.push({ name: 'trigger', type: '*', link: null })
        outputs.push({ name: 'signal', type: '*', links: [] })
        break
    }

    // Calculate node height based on field count for proper sizing
    const fields = (apiNode.data.fields as Array<{ name: string; type: string }>) || []
    const methods = (apiNode.data.methods as Array<{ name: string }>) || []
    const slotCount = Math.max(inputs.length, outputs.length)
    const fieldHeight = 18
    const methodHeight = 16
    const baseHeight = 40 + 8 + (slotCount * 20) + (fields.length * fieldHeight) + 8
    const methodsExtra = methods.length > 0 ? (methodHeight + methods.length * methodHeight) : 0
    const height = Math.max(baseHeight + methodsExtra, NODE_CONFIG.defaultSize[1])

    return {
      id: apiNode.id,
      type: trinityType,
      title: apiNode.name,
      pos: apiNode.position,
      size: [Math.max(NODE_CONFIG.defaultSize[0], 200), height] as [number, number],
      properties: {
        ...apiNode.data,
        trinityType: apiNode.type,
        className: apiNode.name,
        sourceFile: apiNode.source.file,
        sourceLine: apiNode.source.line,
      },
      inputs,
      outputs,
      flags: {}
    }
  }

  /**
   * Convert API GraphEdge to store GraphLink format.
   */
  function apiEdgeToStoreLink(edge: GraphEdge): GraphLink {
    return {
      id: edge.id,
      originId: edge.source,
      originSlot: 0, // Default slot
      targetId: edge.target,
      targetSlot: 0, // Default slot
      type: edge.type
    }
  }

  /**
   * Convert store state to API NodeGraph format.
   */
  function storeToApiGraph(): NodeGraph {
    return {
      nodes: nodes.value.map((node): ApiGraphNode => ({
        id: node.id,
        type: node.type.replace('trinity/', '') as 'component' | 'system' | 'resource' | 'event',
        name: node.title,
        position: node.pos,
        data: node.properties || {},
        source: { file: currentFilePath.value || '', line: 0 }
      })),
      edges: links.value.map((link): GraphEdge => ({
        id: link.id,
        source: link.originId,
        target: link.targetId,
        type: link.type as 'reference' | 'inheritance' | 'query'
      }))
    }
  }

  /**
   * Load a graph from a Python file path.
   * Calls the API to parse the file and updates the store state.
   */
  async function loadFromPythonFile(path: string): Promise<void> {
    // Validate file extension before attempting to parse
    const extension = path.split('.').pop()?.toLowerCase()
    if (extension !== 'py') {
      throw new Error(
        `Cannot open "${path.split('/').pop()}": only Python (.py) files are supported. ` +
        `Got .${extension || '(no extension)'} file.`
      )
    }

    const api = getApi()
    let graph
    try {
      graph = await api.parsePythonFile(path)
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)

      // Check for sidecar crash / connectivity errors and add context
      if (message.includes('Sidecar') || message.includes('sidecar') || message.includes('NotRunning')) {
        throw new Error(
          `Python sidecar crashed while parsing "${path.split('/').pop()}". ` +
          `A restart was attempted automatically. Please try again. ` +
          `Details: ${message}`
        )
      }

      // Check for syntax errors - pass through with context
      if (message.includes('SyntaxError') || message.includes('syntax') || message.includes('parse error')) {
        throw new Error(`Syntax error in "${path.split('/').pop()}": ${message}`)
      }

      throw new Error(`Failed to parse "${path.split('/').pop()}": ${message}`)
    }

    // Convert API nodes to store format
    nodes.value = graph.nodes.map(apiNodeToStoreNode)
    links.value = graph.edges.map(apiEdgeToStoreLink)
    groups.value = []

    // Update IDs tracking
    lastNodeId.value = nodes.value.length
    lastLinkId.value = links.value.length

    // Update file state
    currentFilePath.value = path
    isModified.value = false

    // Clear selection and history
    clearSelection()
    history.value = []
    historyIndex.value = -1
    pushHistory('Load from Python file')
  }

  /**
   * Save the current graph to a file.
   * If no path provided, uses currentFilePath.
   */
  async function saveToFile(path?: string): Promise<void> {
    const savePath = path || currentFilePath.value
    if (!savePath) {
      throw new Error('No file path specified and no current file path')
    }

    const api = getApi()
    const graph = storeToApiGraph()
    await api.savePythonFile(savePath, graph)

    // Update file state
    currentFilePath.value = savePath
    isModified.value = false
  }

  /**
   * Open a Python file using native file dialog.
   * If a file is selected, loads it into the store.
   */
  async function openFile(): Promise<void> {
    const api = getApi()
    const result = await api.openPythonFile()

    if (result) {
      await loadFromPythonFile(result.path)
    }
  }

  /**
   * Save the current graph with a native save dialog.
   * Returns the saved path or null if cancelled.
   */
  async function saveFileAs(): Promise<string | null> {
    const api = getApi()
    const graph = storeToApiGraph()

    if (api.savePythonFileAs) {
      const savedPath = await api.savePythonFileAs(graph)
      if (savedPath) {
        currentFilePath.value = savedPath
        isModified.value = false
      }
      return savedPath
    }

    return null
  }

  // ===========================================================================
  // File Watcher Operations (External Change Detection)
  // ===========================================================================

  /**
   * Start watching the current file for external changes.
   * Called automatically when a file is opened.
   */
  async function startFileWatcher(): Promise<void> {
    stopFileWatcher()

    if (!currentFilePath.value) {
      return
    }

    try {
      const info = await getFileInfo(currentFilePath.value)
      if (info.exists && info.modified !== undefined) {
        lastKnownMtime.value = info.modified
        hasExternalChanges.value = false
      }

      fileWatcherIntervalId.value = setInterval(checkForExternalChanges, FILE_WATCH_POLL_INTERVAL)
      console.log('[graphStore] Started file watcher for:', currentFilePath.value)
    } catch (err) {
      console.error('[graphStore] Error starting file watcher:', err)
    }
  }

  /**
   * Stop the file watcher.
   */
  function stopFileWatcher(): void {
    if (fileWatcherIntervalId.value !== null) {
      clearInterval(fileWatcherIntervalId.value)
      fileWatcherIntervalId.value = null
      console.log('[graphStore] Stopped file watcher')
    }
  }

  /**
   * Check if the file has been modified externally.
   */
  async function checkForExternalChanges(): Promise<void> {
    if (!currentFilePath.value || lastKnownMtime.value === null) {
      return
    }

    try {
      const info = await getFileInfo(currentFilePath.value)

      if (!info.exists) {
        console.warn('[graphStore] File no longer exists:', currentFilePath.value)
        return
      }

      if (info.modified !== undefined && info.modified > lastKnownMtime.value) {
        if (!hasExternalChanges.value) {
          console.log('[graphStore] External change detected:', {
            path: currentFilePath.value,
            oldMtime: lastKnownMtime.value,
            newMtime: info.modified,
          })
          hasExternalChanges.value = true
        }
      }
    } catch (err) {
      console.error('[graphStore] Error checking for external changes:', err)
    }
  }

  /**
   * Update the last known mtime after saving.
   * Call this after successfully saving the file.
   */
  async function updateLastMtime(): Promise<void> {
    if (!currentFilePath.value) {
      return
    }

    try {
      const info = await getFileInfo(currentFilePath.value)
      if (info.exists && info.modified !== undefined) {
        lastKnownMtime.value = info.modified
        hasExternalChanges.value = false
        console.log('[graphStore] Updated mtime to:', info.modified)
      }
    } catch (err) {
      console.error('[graphStore] Error updating mtime:', err)
    }
  }

  /**
   * Acknowledge external changes.
   * Call this when user chooses to reload, overwrite, or compare.
   */
  function acknowledgeExternalChanges(): void {
    hasExternalChanges.value = false
  }

  /**
   * Reload the file from disk, discarding local changes.
   */
  async function reloadFromDisk(): Promise<void> {
    if (!currentFilePath.value) {
      throw new Error('No file path to reload from')
    }

    await loadFromPythonFile(currentFilePath.value)
    hasExternalChanges.value = false
  }

  // Watch currentFilePath to start/stop file watcher
  watch(currentFilePath, async (newPath, oldPath) => {
    if (newPath !== oldPath) {
      if (newPath) {
        await startFileWatcher()
      } else {
        stopFileWatcher()
        lastKnownMtime.value = null
        hasExternalChanges.value = false
      }
    }
  })

  return {
    // State
    nodes,
    links,
    groups,
    selectedNodeIds,
    selectedLinkIds,
    currentFilePath,
    isModified,
    fileName,
    stateVersion,
    isRestoringState,
    suppressHistory,
    canvasOffset,
    canvasScale,

    // Computed
    selectedNodes,
    selectedLinks,
    hasSelection,
    canUndo,
    canRedo,

    // Node operations
    addNode,
    removeNode,
    updateNode,
    moveNode,

    // Link operations
    addLink,
    removeLink,
    getConnectedEdges,

    // Selection operations
    selectNode,
    deselectNode,
    selectLink,
    selectAll,
    clearSelection,
    deleteSelected,

    // Group operations
    createGroup,
    removeGroup,

    // File operations
    setCurrentFile,
    markModified,
    markSaved,

    // Serialization
    getGraphState,
    loadGraphState,
    clearGraph,

    // History
    pushHistory,
    undo,
    redo,
    clearHistory,
    getHistoryList,
    getHistoryIndex,
    getUndoDescription,
    getRedoDescription,
    createSnapshot,
    restoreFromSnapshot,
    batch,

    // Canvas operations
    setCanvasOffset,
    setCanvasScale,
    resetView,

    // API-connected file operations
    loadFromPythonFile,
    saveToFile,
    openFile,
    saveFileAs,

    // Graph conversion for codegen
    storeToApiGraph,

    // File watcher state and operations
    hasExternalChanges,
    lastKnownMtime,
    startFileWatcher,
    stopFileWatcher,
    checkForExternalChanges,
    updateLastMtime,
    acknowledgeExternalChanges,
    reloadFromDisk,
  }
})
