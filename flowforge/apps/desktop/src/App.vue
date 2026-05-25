<script setup lang="ts">
import { ref, computed, shallowRef, watch, onMounted, onUnmounted, provide, markRaw } from 'vue'
import AppLayout from '@/components/layout/AppLayout.vue'
import GraphCanvas from '@/components/canvas/GraphCanvas.vue'
import NodePalette from '@/components/sidebar/NodePalette.vue'
import FileExplorer from '@/components/sidebar/FileExplorer.vue'
import TypeFilter from '@/components/common/TypeFilter.vue'
import NodeSearch from '@/components/graph/NodeSearch.vue'
import { useWindowTitle } from '@/composables/useWindowTitle'
import { useTypeFilter } from '@/composables/useTypeFilter'
import { useNodeSearch, type NodeTypeFilter, type SearchResult } from '@/composables/useNodeSearch'
import { useSourceNavigation } from '@/composables/useSourceNavigation'
import SourceIndicator from '@/components/canvas/SourceIndicator.vue'
import CanvasMinimap from '@/components/canvas/CanvasMinimap.vue'
import { useWorkspaceStore } from '@/stores/workspaceStore'
import { useGraphStore } from '@/stores/graphStore'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import { useSidebarTabStore } from '@/stores/sidebarTabStore'
import { useBottomPanelStore } from '@/stores/bottomPanelStore'
import { useCommandStore } from '@/stores/commandStore'
import { useKeybindingStore, KeybindingImpl, KeyComboImpl } from '@/stores/keybindingStore'
import { useTrinityStore } from '@/stores/trinityStore'
import { useEditHistoryStore } from '@/stores/editHistoryStore'
import { useFileExplorerStore } from '@/stores/fileExplorerStore'
import { registerTrinityNodes } from '@/litegraph/nodes'
import { TrinityNode } from '@/litegraph/nodes/TrinityNodes'
import { useFileOperations } from '@/composables/useFileOperations'
import { useRecentFiles } from '@/composables/useRecentFiles'
import { useFileConflict } from '@/composables/useFileConflict'
import { useDiffPreview } from '@/composables/useDiffPreview'
import RecentFilesMenu from '@/components/common/RecentFilesMenu.vue'
import { GlobalDialog } from '@/components'
import { useDialogStore } from '@/stores/dialogStore'
import FlowForgeLogo from '@/components/common/FlowForgeLogo.vue'
import AboutDialog from '@/components/dialogs/AboutDialog.vue'
import ConfirmDialog from '@/components/dialogs/ConfirmDialog.vue'
import AddFieldDialog from '@/components/dialogs/AddFieldDialog.vue'
import type { AddFieldResult } from '@/components/dialogs/AddFieldDialog.vue'
import { openInEditor } from '@/bridge/editor'
import { useNodeEditing } from '@/composables/useNodeEditing'

// Trinity introspection panels
import {
  EventLogPanel,
  InstancesPanel,
  RegistryPanel,
  InspectorPanel
} from '@/components/panels'

// Import global styles
import '@/styles/main.css'

// =============================================================================
// STATE
// =============================================================================

const isReady = ref(false)
const graphCanvasRef = ref<InstanceType<typeof GraphCanvas> | null>(null)
const nodeSearchRef = ref<InstanceType<typeof NodeSearch> | null>(null)

// Set up window title management
const { currentTitle } = useWindowTitle()

// Stores
const workspaceStore = useWorkspaceStore()
const graphStore = useGraphStore()
const nodeDefStore = useNodeDefStore()
const sidebarTabStore = useSidebarTabStore()
const bottomPanelStore = useBottomPanelStore()
const commandStore = useCommandStore()
const keybindingStore = useKeybindingStore()
const trinityStore = useTrinityStore()
const editHistoryStore = useEditHistoryStore()
const fileExplorerStore = useFileExplorerStore()

// File operations composable
const fileOperations = useFileOperations()

// Node editing composable for add field / rename operations
const nodeEditing = useNodeEditing()

// Recent files
const { addRecentFile } = useRecentFiles()

// File conflict detection and resolution
const dialogStore = useDialogStore()
const fileConflict = useFileConflict({
  onResolved: (action) => {
    console.log('[App] File conflict resolved:', action)
  },
  onError: (error) => {
    console.error('[App] File conflict error:', error)
  },
})

// Diff preview composable for save with preview
const diffPreview = useDiffPreview({
  onApply: () => {
    console.log('[App] Diff changes applied successfully')
    graphStore.markSaved()
  },
  onCancel: () => {
    console.log('[App] Diff preview cancelled')
  },
})

// Trinity panel visibility state
const showInspectorPanel = ref(false)
const showMinimap = ref(true)
const trinityPanelMode = ref<'registry' | 'instances'>('registry')

// Type filter composable for filtering nodes by type
const typeFilter = useTypeFilter()

// Node search composable - initialized after canvas is ready
const graphRef = computed(() => graphCanvasRef.value?.graph ?? null)
const canvasRef = computed(() => graphCanvasRef.value?.canvas ?? null)

// Create a shallow ref wrapper for the composable
const graphShallowRef = shallowRef(graphRef.value)
const canvasShallowRef = shallowRef(canvasRef.value)

// Watch for changes and update shallow refs
watch(graphRef, (val) => { graphShallowRef.value = val })
watch(canvasRef, (val) => { canvasShallowRef.value = val })

const nodeSearch = useNodeSearch({
  graph: graphShallowRef,
  canvas: canvasShallowRef
})

// Source navigation composable - handles node click -> source file navigation
const sourceNavigation = useSourceNavigation({
  onNavigate: (location) => {
    console.log('[App] Navigate to source:', location.file, location.line)
  }
})

// Provide stores for child components
provide('workspaceStore', workspaceStore)
provide('graphStore', graphStore)
provide('editHistoryStore', editHistoryStore)

// =============================================================================
// INITIALIZATION
// =============================================================================

onMounted(async () => {
  try {
    // Register Trinity node types
    registerTrinityNodes()

    // Load node definitions
    await nodeDefStore.loadNodeTypes()

    // Register sidebar tabs
    registerSidebarTabs()

    // Register bottom panel tabs (includes Trinity EventLogPanel)
    registerBottomPanelTabs()

    // Register commands and keybindings
    registerCommands()
    registerKeybindings()

    // Set up keyboard event listeners
    setupKeyboardListeners()

    // Set up window focus listener for file change detection
    window.addEventListener('focus', handleWindowFocus)

    // Initialize Trinity store with polling
    await trinityStore.initialize(true)

    // Initialize File Explorer store
    await fileExplorerStore.initialize()

    isReady.value = true
  } catch (error) {
    console.error('[App] Initialization failed:', error)
  }
})

onUnmounted(() => {
  // Clean up keyboard listeners
  window.removeEventListener('keydown', handleKeyDown)
  window.removeEventListener('keyup', handleKeyUp)

  // Clean up window focus listener
  window.removeEventListener('focus', handleWindowFocus)

  // Clean up Trinity store polling
  trinityStore.cleanup()
})

// =============================================================================
// WINDOW FOCUS HANDLING
// =============================================================================

/**
 * Handle window focus event - check for external file changes.
 * This provides immediate feedback when the user switches back to the app.
 */
function handleWindowFocus() {
  if (graphStore.currentFilePath) {
    // Trigger an immediate check for external changes
    graphStore.checkForExternalChanges()
  }
}

/**
 * Handle opening a file from the Recent Files menu.
 */
async function handleRecentFileOpen(path: string) {
  try {
    await graphStore.loadFromPythonFile(path)
    addRecentFile(path)
  } catch (error) {
    console.error('[App] Failed to open recent file:', error)
  }
}

// =============================================================================
// SIDEBAR TABS
// =============================================================================

function registerSidebarTabs() {
  // Node Palette tab
  sidebarTabStore.registerSidebarTab({
    id: 'node-palette',
    title: 'Nodes',
    icon: 'pi pi-box',
    tooltip: 'Node Palette',
    type: 'vue',
    order: 1
  })

  // File Explorer tab
  sidebarTabStore.registerSidebarTab({
    id: 'file-explorer',
    title: 'Files',
    icon: 'pi pi-folder',
    tooltip: 'File Explorer',
    type: 'vue',
    order: 2
  })

  // Trinity Introspection tab
  sidebarTabStore.registerSidebarTab({
    id: 'trinity',
    title: 'Trinity',
    icon: 'pi pi-sitemap',
    tooltip: 'Trinity Runtime Introspection',
    type: 'vue',
    order: 3
  })

  // Open the node palette by default
  sidebarTabStore.setActiveTab('node-palette')
}

/**
 * Register bottom panel tabs including Trinity EventLogPanel
 */
function registerBottomPanelTabs() {
  // Register core tabs first
  bottomPanelStore.registerCoreBottomPanelTabs()

  // Register Trinity Event Log tab
  bottomPanelStore.registerBottomPanelTab({
    id: 'trinity-events',
    title: 'Trinity Events',
    icon: 'pi pi-bolt',
    targetPanel: 'console',
    order: 10
  })
}

// =============================================================================
// COMMANDS
// =============================================================================

function registerCommands() {
  // File commands
  commandStore.registerCommand({
    id: 'File.New',
    label: 'New Graph',
    icon: 'pi pi-file',
    category: 'file',
    function: () => {
      graphStore.clearGraph()
      graphCanvasRef.value?.clearGraph()
    }
  })

  commandStore.registerCommand({
    id: 'File.Save',
    label: 'Save',
    icon: 'pi pi-save',
    category: 'file',
    function: async () => {
      console.log('[App] Save graph')

      // If no current file, fall back to Save As (no diff to show)
      if (!graphStore.currentFilePath) {
        const success = await fileOperations.saveFileAs()
        if (success) {
          console.log('[App] File saved successfully via Save As')
        }
        return
      }

      // If not modified, nothing to save
      if (!graphStore.isModified) {
        console.log('[App] No changes to save')
        return
      }

      // Generate diff and show preview dialog
      try {
        // Read original file content
        const { readPythonFile } = await import('@/bridge/files')
        const { content: originalContent } = await readPythonFile(graphStore.currentFilePath)

        // Generate modified content from current graph state
        const modifiedContent = fileOperations.generateCurrentCode()

        // Extract filename from path
        const filename = graphStore.currentFilePath.split('/').pop() || graphStore.currentFilePath

        // Show diff preview dialog
        await diffPreview.showDiffPreview(
          originalContent,
          modifiedContent,
          filename,
          graphStore.currentFilePath,
        )
      } catch (error) {
        console.error('[App] Failed to generate diff preview, falling back to direct save:', error)
        // Fall back to direct save if diff preview fails
        const success = await fileOperations.saveFile()
        if (success) {
          console.log('[App] File saved successfully (direct)')
        }
      }
    }
  })

  // Save As command
  commandStore.registerCommand({
    id: 'File.SaveAs',
    label: 'Save As...',
    icon: 'pi pi-save',
    category: 'file',
    function: async () => {
      console.log('[App] Save As')
      const success = await fileOperations.saveFileAs()
      if (success) {
        console.log('[App] File saved successfully')
      }
    }
  })

  // Open command
  commandStore.registerCommand({
    id: 'File.Open',
    label: 'Open',
    icon: 'pi pi-folder-open',
    category: 'file',
    function: async () => {
      console.log('[App] Open file')
      const success = await fileOperations.openFile()
      if (success) {
        console.log('[App] File opened successfully')
      }
    }
  })

  // Edit commands
  commandStore.registerCommand({
    id: 'Edit.Undo',
    label: 'Undo',
    icon: 'pi pi-undo',
    category: 'edit',
    function: () => { graphStore.undo() }
  })

  commandStore.registerCommand({
    id: 'Edit.Redo',
    label: 'Redo',
    icon: 'pi pi-replay',
    category: 'edit',
    function: () => { graphStore.redo() }
  })

  commandStore.registerCommand({
    id: 'Edit.Delete',
    label: 'Delete Selected',
    icon: 'pi pi-trash',
    category: 'edit',
    function: () => confirmAndDeleteSelected()
  })

  commandStore.registerCommand({
    id: 'Edit.SelectAll',
    label: 'Select All',
    icon: 'pi pi-check-square',
    category: 'edit',
    function: () => graphStore.selectAll()
  })

  // View commands
  commandStore.registerCommand({
    id: 'View.ZoomToFit',
    label: 'Zoom to Fit',
    icon: 'pi pi-expand',
    category: 'view-controls',
    function: () => graphCanvasRef.value?.zoomToFit()
  })

  commandStore.registerCommand({
    id: 'View.ResetView',
    label: 'Reset View',
    icon: 'pi pi-home',
    category: 'view-controls',
    function: () => graphCanvasRef.value?.resetView()
  })

  commandStore.registerCommand({
    id: 'View.ToggleFocusMode',
    label: 'Toggle Focus Mode',
    icon: 'pi pi-eye',
    category: 'view-controls',
    function: () => workspaceStore.toggleFocusMode()
  })

  // Search command
  commandStore.registerCommand({
    id: 'Edit.SearchNodes',
    label: 'Search Nodes',
    icon: 'pi pi-search',
    category: 'edit',
    function: () => openNodeSearch()
  })

  // Trinity Inspector command
  commandStore.registerCommand({
    id: 'View.ToggleInspector',
    label: 'Toggle Inspector Panel',
    icon: 'pi pi-info-circle',
    category: 'view-controls',
    function: () => toggleInspectorPanel()
  })
}

// =============================================================================
// KEYBINDINGS
// =============================================================================

function registerKeybindings() {
  // Helper to safely add keybinding (skips if combo already exists)
  const safeAddKeybinding = (binding: KeybindingImpl) => {
    try {
      keybindingStore.addDefaultKeybinding(binding)
    } catch {
      // Key combo already registered by another command, skip
    }
  }

  // File operations
  safeAddKeybinding(new KeybindingImpl({
    commandId: 'File.New',
    combo: { key: 'n', ctrl: true }
  }))

  safeAddKeybinding(new KeybindingImpl({
    commandId: 'File.Save',
    combo: { key: 's', ctrl: true }
  }))

  safeAddKeybinding(new KeybindingImpl({
    commandId: 'File.SaveAs',
    combo: { key: 's', ctrl: true, shift: true }
  }))

  safeAddKeybinding(new KeybindingImpl({
    commandId: 'File.Open',
    combo: { key: 'o', ctrl: true }
  }))

  // Edit operations
  safeAddKeybinding(new KeybindingImpl({
    commandId: 'Edit.Undo',
    combo: { key: 'z', ctrl: true }
  }))

  safeAddKeybinding(new KeybindingImpl({
    commandId: 'Edit.Redo',
    combo: { key: 'z', ctrl: true, shift: true }
  }))

  // Also support Ctrl+Y for redo (common on Windows)
  safeAddKeybinding(new KeybindingImpl({
    commandId: 'Edit.Redo',
    combo: { key: 'y', ctrl: true }
  }))

  safeAddKeybinding(new KeybindingImpl({
    commandId: 'Edit.Delete',
    combo: { key: 'Delete' }
  }))

  safeAddKeybinding(new KeybindingImpl({
    commandId: 'Edit.SelectAll',
    combo: { key: 'a', ctrl: true }
  }))

  // View operations
  safeAddKeybinding(new KeybindingImpl({
    commandId: 'View.ZoomToFit',
    combo: { key: '1', ctrl: true }
  }))

  safeAddKeybinding(new KeybindingImpl({
    commandId: 'View.ResetView',
    combo: { key: '0', ctrl: true }
  }))

  safeAddKeybinding(new KeybindingImpl({
    commandId: 'View.ToggleFocusMode',
    combo: { key: 'f', ctrl: true, shift: true }
  }))

  // Search nodes (Ctrl+F / Cmd+F)
  safeAddKeybinding(new KeybindingImpl({
    commandId: 'Edit.SearchNodes',
    combo: { key: 'f', ctrl: true }
  }))

  // Toggle Inspector (Ctrl+Shift+I)
  safeAddKeybinding(new KeybindingImpl({
    commandId: 'View.ToggleInspector',
    combo: { key: 'i', ctrl: true, shift: true }
  }))
}

// =============================================================================
// KEYBOARD HANDLING
// =============================================================================

/**
 * Show confirmation dialog then delete selected nodes from both LiteGraph and the store.
 * This ensures visual and data state stay in sync.
 */
function confirmAndDeleteSelected() {
  if (!graphStore.hasSelection) return

  const nodeCount = graphStore.selectedNodeIds.size
  const linkCount = graphStore.selectedLinkIds.size

  const parts: string[] = []
  if (nodeCount > 0) parts.push(`${nodeCount} node${nodeCount > 1 ? 's' : ''}`)
  if (linkCount > 0) parts.push(`${linkCount} connection${linkCount > 1 ? 's' : ''}`)
  const itemDescription = parts.join(' and ')

  dialogStore.showDialog({
    key: 'confirm-delete',
    title: 'Delete Selected',
    component: markRaw(ConfirmDialog),
    props: {
      title: 'Delete Selected',
      message: `Are you sure you want to delete ${itemDescription}? This action can be undone with Ctrl+Z.`,
      confirmText: 'Delete',
      cancelText: 'Cancel',
      type: 'danger',
      onConfirm: () => {
        performDeleteSelected()
        dialogStore.closeDialog({ key: 'confirm-delete' })
      },
      onCancel: () => {
        dialogStore.closeDialog({ key: 'confirm-delete' })
      }
    },
    dialogComponentProps: {
      modal: true,
      closable: true,
      closeOnEscape: true,
      dismissableMask: true
    }
  })
}

/**
 * Actually delete selected items from both LiteGraph canvas and Pinia store.
 * The key fix: remove from LiteGraph FIRST (which triggers syncToStore),
 * so the visual and data layers stay in sync.
 */
function performDeleteSelected() {
  const canvasComponent = graphCanvasRef.value
  const lgGraph = canvasComponent?.graph

  // Suppress per-node pushHistory calls during batch delete.
  // We'll push one history entry at the end.
  graphStore.suppressHistory = true

  try {
    if (lgGraph) {
      const selectedStoreIds = new Set(graphStore.selectedNodeIds)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const lgNodesToRemove: any[] = []

      for (const lgNode of lgGraph._nodes || []) {
        if (selectedStoreIds.has(String(lgNode.id))) {
          lgNodesToRemove.push(lgNode)
        }
      }

      for (const lgNode of lgNodesToRemove) {
        lgGraph.remove(lgNode)
      }
    } else {
      graphStore.deleteSelected()
      return
    }

    if (graphStore.selectedLinkIds.size > 0) {
      const linkIdsToRemove = new Set(graphStore.selectedLinkIds)
      graphStore.links.splice(0, graphStore.links.length,
        ...graphStore.links.filter((l) => !linkIdsToRemove.has(l.id))
      )
    }

    graphStore.clearSelection()
    graphStore.markModified()
  } finally {
    graphStore.suppressHistory = false
  }

  // Push one history entry for the entire delete operation
  graphStore.pushHistory('Delete selected')
}

function setupKeyboardListeners() {
  window.addEventListener('keydown', handleKeyDown)
  window.addEventListener('keyup', handleKeyUp)
}

function handleKeyDown(event: KeyboardEvent) {
  // Track shift key state
  if (event.key === 'Shift') {
    workspaceStore.setShiftDown(true)
  }

  // Don't handle shortcuts when typing in inputs
  const target = event.target as HTMLElement
  if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
    return
  }

  // Handle Delete/Backspace directly for node deletion
  if (event.key === 'Delete' || event.key === 'Backspace') {
    if (graphStore.hasSelection) {
      event.preventDefault()
      event.stopPropagation()
      confirmAndDeleteSelected()
      return
    }
  }

  // Ignore key repeat for undo/redo to prevent rapid-fire state changes
  if (event.repeat) {
    return
  }

  // Check for registered keybindings
  const combo = KeyComboImpl.fromEvent(event)
  const keybinding = keybindingStore.getKeybinding(combo)

  if (keybinding) {
    event.preventDefault()
    commandStore.execute(keybinding.commandId)
  }
}

function handleKeyUp(event: KeyboardEvent) {
  if (event.key === 'Shift') {
    workspaceStore.setShiftDown(false)
  }
}

// =============================================================================
// EVENT HANDLERS
// =============================================================================

function handleNodeSelected(nodeType: string) {
  console.log('[App] Node selected from palette:', nodeType)
  // Add node at center of canvas
  graphCanvasRef.value?.addNode(nodeType)
}

function handleCanvasReady() {
  console.log('[App] Canvas ready')
}

function handleGraphChanged() {
  console.log('[App] Graph changed')
}

// =============================================================================
// NODE SEARCH HANDLERS
// =============================================================================

function openNodeSearch() {
  nodeSearch.setActive(true)
}

function closeNodeSearch() {
  nodeSearch.clearSearch()
}

function handleNodeSearchSearch(query: string, type: NodeTypeFilter) {
  nodeSearch.search(query, type)
}

function handleNodeSearchSelect(result: SearchResult) {
  nodeSearch.selectNode(result.id)
}

function handleNodeSearchSelectedIndexUpdate(index: number) {
  nodeSearch.selectedIndex.value = index
}

// =============================================================================
// SOURCE NAVIGATION HANDLERS
// =============================================================================

function handleOpenInEditor(file: string, line?: number) {
  console.log('[App] Open in editor:', file, line)
  sourceNavigation.emitExternalNavigation({ file, line: line ?? 1 })
  openInEditor(file, line).then((response) => {
    if (!response.success) {
      console.warn('[App] Failed to open in editor:', response.message)
    }
  })
}

function handleViewSource(file: string, line?: number) {
  console.log('[App] View source:', file, line)
  sourceNavigation.emitExternalNavigation({ file, line: line ?? 1 })
  openInEditor(file, line).then((response) => {
    if (!response.success) {
      console.warn('[App] Failed to open source:', response.message)
    }
  })
}

function handleSourceCopied(path: string) {
  console.log('[App] Source path copied:', path)
}

function handleSourceIndicatorClosed() {
  console.log('[App] Source indicator closed')
}

// =============================================================================
// CONTEXT MENU HANDLERS
// =============================================================================

/**
 * Handle add field request from node context menu.
 * Opens the AddFieldDialog and applies the result to the node.
 */
function handleAddFieldRequested(nodeId: string) {
  console.log('[App] Add field requested for node:', nodeId)

  const existingFields = nodeEditing.getNodeFields(nodeId).map(f => f.name)

  dialogStore.showDialog({
    key: `add-field-${nodeId}`,
    title: 'Add Field',
    component: AddFieldDialog,
    props: {
      existingFields,
      onConfirm: (result: AddFieldResult) => {
        const field: { name: string; type: string; default?: string } = {
          name: result.name,
          type: result.type,
        }
        if (result.default != null) {
          field.default = result.default
        }
        const success = nodeEditing.addFieldToNode(nodeId, field)
        if (success) {
          console.log('[App] Field added:', result.name)
          // Update the LiteGraph node's fields directly so the canvas renders them
          syncLiteGraphNodeFields(nodeId)
        }
        dialogStore.closeDialog({ key: `add-field-${nodeId}` })
      },
      onCancel: () => {
        dialogStore.closeDialog({ key: `add-field-${nodeId}` })
      }
    },
    dialogComponentProps: {
      modal: true,
      closable: true,
      closeOnEscape: true,
      dismissableMask: true
    }
  })
}

/**
 * Directly update a LiteGraph node's fields from the store,
 * without doing a full graph rebuild via syncFromStore().
 * This ensures the canvas visual reflects field changes immediately.
 */
function syncLiteGraphNodeFields(nodeId: string) {
  const graphCanvas = graphCanvasRef.value
  if (!graphCanvas?.graph) return

  const lgGraph = graphCanvas.graph
  const lgNode = lgGraph.getNodeById(parseInt(nodeId))
  if (!lgNode) return

  // Get updated fields from the store
  const storeNode = graphStore.nodes.find(n => n.id === nodeId)
  if (!storeNode?.properties) return

  const updatedFields = storeNode.properties['fields'] ??
    (storeNode.properties['componentData'] as { fields?: unknown[] } | undefined)?.fields ?? []

  // Update the LiteGraph node's fields directly
  if (lgNode instanceof TrinityNode) {
    lgNode.fields = JSON.parse(JSON.stringify(updatedFields))
  }
  // Also update properties so they stay in sync
  if (lgNode.properties) {
    lgNode.properties['fields'] = JSON.parse(JSON.stringify(updatedFields))
  }

  // Recalculate node height to fit new fields and redraw
  if (lgNode instanceof TrinityNode) {
    const newHeight = (lgNode as any).calculateHeight()
    lgNode.size[1] = Math.max(lgNode.size[1], newHeight)
  }

  // Mark canvas dirty to trigger redraw
  graphCanvas.canvas?.setDirty(true, true)
}

/**
 * Handle rename request from node context menu.
 * Prompts for a new name and updates the node title.
 */
function handleRenameRequested(nodeId: string) {
  console.log('[App] Rename requested for node:', nodeId)

  const node = graphStore.nodes.find(n => n.id === nodeId)
  if (!node) return

  const currentName = (node.properties?.['className'] as string) || node.title

  // Use setTimeout to avoid blocking the JS thread (window.prompt is synchronous
  // and can break the LiteGraph canvas rendering context in Tauri webview)
  setTimeout(() => {
    const newName = window.prompt('Rename node:', currentName)
    if (newName === null || newName.trim() === '' || newName.trim() === currentName) return

    const trimmed = newName.trim()
    const validationError = nodeEditing.validateClassName(trimmed)
    if (validationError) {
      window.alert(`Invalid name: ${validationError}`)
      return
    }

    if (nodeEditing.classNameExists(trimmed, nodeId)) {
      window.alert(`A node named "${trimmed}" already exists.`)
      return
    }

    // Update store
    const updatedProps = { ...(node.properties ?? {}), className: trimmed }
    graphStore.updateNode(nodeId, { title: trimmed, properties: updatedProps })

    // Update LiteGraph node directly (no full syncFromStore rebuild)
    const graphCanvas = graphCanvasRef.value
    if (graphCanvas?.graph) {
      const lgNode = graphCanvas.graph.getNodeById(parseInt(nodeId))
      if (lgNode) {
        lgNode.title = trimmed
        if (lgNode.properties) {
          lgNode.properties['className'] = trimmed
        }
        if (lgNode instanceof TrinityNode) {
          lgNode.className = trimmed
        }
        graphCanvas.canvas?.setDirty(true, true)
      }
    }
  }, 50)
}

/**
 * Handle field rename request from double-clicking a field name on a node.
 * Prompts for a new field name and updates the node's fields array.
 */
function handleFieldRenameRequested(nodeId: string, fieldIndex: number, currentFieldName: string) {
  console.log('[App] Field rename requested:', nodeId, fieldIndex, currentFieldName)

  const node = graphStore.nodes.find(n => n.id === nodeId)
  if (!node) return

  const fields = (node.properties?.['fields'] as Array<{ name: string; type: string }>) || []
  if (fieldIndex < 0 || fieldIndex >= fields.length) return

  const newName = window.prompt('Rename field:', currentFieldName)
  if (newName === null || newName.trim() === '' || newName.trim() === currentFieldName) return

  const trimmed = newName.trim()

  // Check for duplicate field names within this node
  if (fields.some((f, i) => i !== fieldIndex && f.name === trimmed)) {
    window.alert(`A field named "${trimmed}" already exists on this node.`)
    return
  }

  // Update the field name in the store
  const updatedFields = fields.map((f, i) => i === fieldIndex ? { ...f, name: trimmed } : f)
  const updatedProps = { ...(node.properties ?? {}), fields: updatedFields }
  graphStore.updateNode(nodeId, { properties: updatedProps })
  graphStore.markModified()
  graphStore.pushHistory(`Rename field "${currentFieldName}" to "${trimmed}"`)

  // Sync the LiteGraph node's fields directly
  syncLiteGraphNodeFields(nodeId)
}

// =============================================================================
// TRINITY PANEL HANDLERS
// =============================================================================

/**
 * Toggle the Inspector panel visibility
 */
function toggleInspectorPanel() {
  showInspectorPanel.value = !showInspectorPanel.value
}

/**
 * Handle entry click from RegistryPanel - shows inspector for the entry
 */
function handleRegistryEntryClick(entry: unknown) {
  console.log('[App] Registry entry clicked:', entry)
  showInspectorPanel.value = true
}

/**
 * Handle node highlight from Trinity panels
 */
function handleTrinityNodeHighlight(nodeId: number | string) {
  console.log('[App] Trinity highlight node:', nodeId)
  // Canvas will handle the node highlight via the graph reference
}

/**
 * Handle component click from InstancesPanel
 */
function handleInstancesComponentClick(componentType: string, trinityType: string) {
  console.log('[App] Instances component clicked:', componentType, trinityType)
}

/**
 * Handle open in editor from InspectorPanel
 */
function handleInspectorOpenInEditor(file: string, line: number) {
  console.log('[App] Open in editor from inspector:', file, line)
  sourceNavigation.emitExternalNavigation({ file, line })
  openInEditor(file, line).then((response) => {
    if (!response.success) {
      console.warn('[App] Failed to open in editor from inspector:', response.message)
    }
  })
}

/**
 * Switch Trinity panel mode (registry/instances)
 */
function setTrinityPanelMode(mode: 'registry' | 'instances') {
  trinityPanelMode.value = mode
}

// =============================================================================
// ABOUT DIALOG
// =============================================================================

function showAboutDialog() {
  dialogStore.showDialog({
    key: 'about-dialog',
    title: 'About',
    component: AboutDialog,
    dialogComponentProps: {
      modal: true,
      closable: true,
      closeOnEscape: true,
      dismissableMask: true
    }
  })
}
</script>

<template>
  <div class="app-container">
    <!-- Header -->
    <header class="app-header">
      <div class="header-left">
        <FlowForgeLogo :size="18" color="#a5b4fc" />
        <h1 class="app-title">{{ currentTitle }}</h1>
        <span class="file-name">{{ graphStore.fileName }}</span>
        <span v-if="graphStore.isModified" class="modified-indicator">*</span>
      </div>
      <div class="header-right">
        <!-- Recent Files menu -->
        <RecentFilesMenu @open-file="handleRecentFileOpen" />
        <!-- About button -->
        <button
          class="header-btn"
          title="About FlowForge"
          @click="showAboutDialog"
        >
          <i class="pi pi-question-circle" />
        </button>
        <!-- Trinity Inspector toggle -->
        <button
          v-if="trinityStore.isAvailable"
          class="header-btn"
          :class="{ active: showInspectorPanel }"
          title="Toggle Inspector Panel (Ctrl+Shift+I)"
          @click="toggleInspectorPanel"
        >
          <i class="pi pi-info-circle" />
        </button>
        <!-- Trinity status indicator -->
        <div
          class="trinity-status"
          :class="{ connected: trinityStore.isAvailable }"
          :title="trinityStore.isAvailable ? `Trinity v${trinityStore.version}` : 'Trinity not connected'"
        >
          <i class="pi pi-circle-fill" />
        </div>
      </div>
    </header>

    <!-- Main content -->
    <main class="app-main">
      <div v-if="!isReady" class="loading">
        <span>Loading FlowForge...</span>
      </div>

      <AppLayout v-else :show-right-panel="showInspectorPanel && trinityStore.isAvailable">
        <!-- Sidebar content based on active tab -->
        <template #sidebar>
          <NodePalette
            v-if="sidebarTabStore.activeSidebarTabId === 'node-palette'"
            @node-selected="handleNodeSelected"
          />
          <div v-else-if="sidebarTabStore.activeSidebarTabId === 'file-explorer'" class="file-explorer">
            <FileExplorer />
          </div>
          <!-- Trinity Introspection Panel -->
          <div v-else-if="sidebarTabStore.activeSidebarTabId === 'trinity'" class="trinity-sidebar">
            <!-- Trinity connection status -->
            <div v-if="!trinityStore.isAvailable" class="trinity-disconnected">
              <i class="pi pi-exclamation-triangle" />
              <span>Trinity not connected</span>
            </div>
            <!-- Panel mode toggle -->
            <div v-else class="trinity-mode-toggle">
              <button
                class="mode-btn"
                :class="{ active: trinityPanelMode === 'registry' }"
                @click="setTrinityPanelMode('registry')"
              >
                Registry
              </button>
              <button
                class="mode-btn"
                :class="{ active: trinityPanelMode === 'instances' }"
                @click="setTrinityPanelMode('instances')"
              >
                Instances
              </button>
            </div>
            <!-- Registry Panel -->
            <RegistryPanel
              v-if="trinityStore.isAvailable && trinityPanelMode === 'registry'"
              :graph="graphRef"
              :canvas="canvasRef"
              @entry-click="handleRegistryEntryClick"
            />
            <!-- Instances Panel -->
            <InstancesPanel
              v-else-if="trinityStore.isAvailable && trinityPanelMode === 'instances'"
              @highlight-node="handleTrinityNodeHighlight"
              @component-click="handleInstancesComponentClick"
            />
          </div>
        </template>

        <!-- Graph canvas -->
        <template #canvas>
          <GraphCanvas
            ref="graphCanvasRef"
            @ready="handleCanvasReady"
            @graph-changed="handleGraphChanged"
            @view-source="handleViewSource"
            @open-in-editor="handleOpenInEditor"
            @add-field-requested="handleAddFieldRequested"
            @rename-requested="handleRenameRequested"
            @field-rename-requested="handleFieldRenameRequested"
          />
        </template>

        <!-- Canvas overlays -->
        <template #canvas-overlays>
          <!-- Type filter toolbar -->
          <div class="canvas-toolbar">
            <TypeFilter />
          </div>

          <!-- Node Search -->
          <NodeSearch
            ref="nodeSearchRef"
            :is-visible="nodeSearch.isActive.value"
            :results="nodeSearch.results.value"
            :selected-index="nodeSearch.selectedIndex.value"
            @search="handleNodeSearchSearch"
            @select="handleNodeSearchSelect"
            @close="closeNodeSearch"
            @update:selected-index="handleNodeSearchSelectedIndexUpdate"
            @select-next="nodeSearch.selectNextResult"
            @select-previous="nodeSearch.selectPreviousResult"
            @confirm="nodeSearch.confirmSelection"
          />

          <!-- Source Navigation Indicator -->
          <SourceIndicator
            @open-in-editor="handleOpenInEditor"
            @copied="handleSourceCopied"
            @closed="handleSourceIndicatorClosed"
          />

          <!-- Canvas Minimap -->
          <CanvasMinimap
            :graph="graphShallowRef"
            :canvas="canvasShallowRef"
            :visible="showMinimap"
            @toggle="showMinimap = !showMinimap"
          />

          <!-- Minimap toggle button (shown when minimap is hidden) -->
          <button
            v-if="!showMinimap"
            class="minimap-toggle-btn"
            title="Show minimap"
            @click="showMinimap = true"
          >
            <i class="pi pi-map" />
          </button>
        </template>

        <!-- Right panel (Inspector) -->
        <template #right-panel>
          <InspectorPanel
            v-if="showInspectorPanel && trinityStore.isAvailable"
            @open-in-editor="handleInspectorOpenInEditor"
          />
        </template>

        <!-- Bottom panel -->
        <template #bottom-panel>
          <div class="bottom-panel-content-wrapper">
            <!-- Tab bar -->
            <div class="bottom-panel-tabs">
              <button
                class="tab-btn"
                :class="{ active: bottomPanelStore.activeBottomPanelTabId === 'output' }"
                @click="bottomPanelStore.setActiveTab('output')"
              >
                Output
              </button>
              <button
                v-if="trinityStore.isAvailable"
                class="tab-btn"
                :class="{ active: bottomPanelStore.activeBottomPanelTabId === 'trinity-events' }"
                @click="bottomPanelStore.setActiveTab('trinity-events')"
              >
                Trinity Events
                <span v-if="trinityStore.eventCount > 0" class="event-badge">
                  {{ trinityStore.eventCount }}
                </span>
              </button>
            </div>
            <!-- Tab content -->
            <div class="bottom-panel-tab-content">
              <div v-if="bottomPanelStore.activeBottomPanelTabId === 'output'" class="output-content">
                <span class="placeholder-text">Output will appear here</span>
              </div>
              <EventLogPanel
                v-else-if="bottomPanelStore.activeBottomPanelTabId === 'trinity-events'"
                :graph="graphRef"
                :canvas="canvasRef"
                @node-highlight="handleTrinityNodeHighlight"
              />
            </div>
          </div>
        </template>
      </AppLayout>
    </main>

    <!-- Global dialog container for file conflicts and other dialogs -->
    <GlobalDialog :dialog-stack="dialogStore.dialogStack" />
  </div>
</template>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100%;
  background-color: var(--app-bg, #1e1e1e);
  color: var(--text-primary, #ffffff);
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 16px;
  height: 40px;
  background-color: var(--header-bg, #2d2d2d);
  border-bottom: 1px solid var(--border-color, #3d3d3d);
  flex-shrink: 0;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.app-title {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary, #ffffff);
}

.file-name {
  font-size: 13px;
  color: var(--text-secondary, #cccccc);
}

.modified-indicator {
  font-size: 14px;
  color: var(--accent-color, #4fc3f7);
}

.header-right {
  display: flex;
  align-items: center;
  gap: 8px;
}

.header-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  padding: 0;
  background: transparent;
  border: none;
  border-radius: 4px;
  color: var(--text-secondary, #cccccc);
  cursor: pointer;
  transition: all 0.15s ease;
}

.header-btn:hover {
  background-color: var(--hover-bg, #3a3a3a);
  color: var(--text-primary, #ffffff);
}

.header-btn.active {
  background-color: var(--accent-color, #6366f1);
  color: #ffffff;
}

.header-btn i {
  font-size: 14px;
}

.trinity-status {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 4px;
}

.trinity-status i {
  font-size: 8px;
  color: var(--text-muted, #666);
  transition: color 0.3s ease;
}

.trinity-status.connected i {
  color: var(--success-color, #22c55e);
}

.app-main {
  flex: 1;
  display: flex;
  overflow: hidden;
  position: relative;
}

.loading {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  color: var(--text-muted, #888);
  background-color: var(--app-bg, #1e1e1e);
}

.file-explorer {
  padding: 16px;
}

.placeholder-text {
  color: var(--text-muted, #888);
  font-size: 13px;
}

/* Bottom panel styles */
.bottom-panel-content-wrapper {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.bottom-panel-tabs {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 12px;
  background-color: var(--panel-header-bg, #2d2d2d);
  border-bottom: 1px solid var(--border-color, #3d3d3d);
  flex-shrink: 0;
}

.tab-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  font-size: 12px;
  color: var(--text-secondary, #cccccc);
  background: transparent;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.tab-btn:hover {
  background-color: var(--hover-bg, #3a3a3a);
  color: var(--text-primary, #ffffff);
}

.tab-btn.active {
  background-color: var(--selected-bg, #404040);
  color: var(--text-primary, #ffffff);
}

.event-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 18px;
  height: 16px;
  padding: 0 4px;
  font-size: 10px;
  font-weight: 600;
  color: #fff;
  background-color: var(--accent-color, #6366f1);
  border-radius: 8px;
}

.bottom-panel-tab-content {
  flex: 1;
  overflow: hidden;
}

.output-content {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--text-muted, #888);
}

/* Trinity sidebar styles */
.trinity-sidebar {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.trinity-disconnected {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 32px 16px;
  color: var(--text-muted, #888);
  text-align: center;
}

.trinity-disconnected i {
  font-size: 24px;
  color: var(--warning-color, #f59e0b);
}

.trinity-mode-toggle {
  display: flex;
  gap: 4px;
  padding: 8px 12px;
  background-color: var(--panel-header-bg, #2d2d2d);
  border-bottom: 1px solid var(--border-color, #3d3d3d);
  flex-shrink: 0;
}

.mode-btn {
  flex: 1;
  padding: 6px 12px;
  font-size: 12px;
  color: var(--text-secondary, #cccccc);
  background-color: transparent;
  border: 1px solid var(--border-color, #3d3d3d);
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.mode-btn:hover {
  background-color: var(--hover-bg, #3a3a3a);
  color: var(--text-primary, #ffffff);
}

.mode-btn.active {
  background-color: var(--accent-color, #6366f1);
  border-color: var(--accent-color, #6366f1);
  color: #ffffff;
}

.canvas-toolbar {
  position: absolute;
  top: 12px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 100;
  pointer-events: none;
}

.canvas-toolbar > * {
  pointer-events: auto;
}

.minimap-toggle-btn {
  position: absolute;
  bottom: 16px;
  right: 16px;
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  padding: 0;
  background-color: var(--panel-header-bg, #2d2d2d);
  border: 1px solid var(--border-color, #3d3d3d);
  border-radius: 6px;
  color: var(--text-secondary, #cccccc);
  cursor: pointer;
  pointer-events: auto;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
  transition: all 0.15s ease;
}

.minimap-toggle-btn:hover {
  background-color: var(--hover-bg, #3a3a3a);
  color: var(--text-primary, #ffffff);
}

.minimap-toggle-btn i {
  font-size: 14px;
}
</style>
