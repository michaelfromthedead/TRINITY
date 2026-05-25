/**
 * Store Initialization
 *
 * Initializes all stores with their default commands, keybindings, and menu items.
 * This should be called once when the application starts.
 */
import { useCommandStore } from './commandStore'
import { useKeybindingStore } from './keybindingStore'
import { useMenuItemStore } from './menuItemStore'
import { useGraphStore } from './graphStore'
import { useSettingsStore } from './settingsStore'
import { initEditorBridge, cleanupEditorBridge } from '@/bridge/editor'

let initialized = false

/**
 * Initialize all stores with default configurations.
 * Safe to call multiple times - will only initialize once.
 */
export async function initializeStores(): Promise<void> {
  if (initialized) {
    return
  }

  const commandStore = useCommandStore()
  const keybindingStore = useKeybindingStore()
  const menuItemStore = useMenuItemStore()
  const settingsStore = useSettingsStore()

  // 1. Initialize settings (loads from storage, sets up editor bridge)
  await settingsStore.initialize()

  // 2. Initialize editor bridge (sets up navigate-to-source event listener)
  initEditorBridge()

  // 3. Register core file commands
  commandStore.registerCoreFileCommands()

  // 4. Register additional core commands (edit, view, graph)
  registerEditCommands(commandStore)
  registerViewCommands(commandStore)
  registerGraphCommands(commandStore)

  // 5. Register core keybindings
  keybindingStore.registerCoreKeybindings()

  // 6. Register menu items
  menuItemStore.registerCoreMenuCommands()

  initialized = true
}

/**
 * Cleanup stores and bridges.
 * Call this when the application is being destroyed.
 */
export function cleanupStores(): void {
  cleanupEditorBridge()
}

/**
 * Register edit commands (undo, redo, cut, copy, paste, select all)
 */
function registerEditCommands(commandStore: ReturnType<typeof useCommandStore>): void {
  const graphStore = useGraphStore()

  commandStore.registerCommands([
    {
      id: 'FlowForge.Undo',
      label: 'Undo',
      menubarLabel: 'Undo',
      icon: 'undo',
      tooltip: 'Undo the last action',
      category: 'edit',
      function: () => {
        graphStore.undo()
      },
    },
    {
      id: 'FlowForge.Redo',
      label: 'Redo',
      menubarLabel: 'Redo',
      icon: 'redo',
      tooltip: 'Redo the last undone action',
      category: 'edit',
      function: () => {
        graphStore.redo()
      },
    },
    {
      id: 'FlowForge.Cut',
      label: 'Cut',
      menubarLabel: 'Cut',
      icon: 'scissors',
      tooltip: 'Cut selected nodes',
      category: 'edit',
      function: () => {
        // Placeholder - implement clipboard operations
        console.log('Cut not yet implemented')
      },
    },
    {
      id: 'FlowForge.Copy',
      label: 'Copy',
      menubarLabel: 'Copy',
      icon: 'copy',
      tooltip: 'Copy selected nodes',
      category: 'edit',
      function: () => {
        // Placeholder - implement clipboard operations
        console.log('Copy not yet implemented')
      },
    },
    {
      id: 'FlowForge.Paste',
      label: 'Paste',
      menubarLabel: 'Paste',
      icon: 'clipboard',
      tooltip: 'Paste nodes from clipboard',
      category: 'edit',
      function: () => {
        // Placeholder - implement clipboard operations
        console.log('Paste not yet implemented')
      },
    },
    {
      id: 'FlowForge.SelectAll',
      label: 'Select All',
      menubarLabel: 'Select All',
      icon: 'check-square',
      tooltip: 'Select all nodes',
      category: 'edit',
      function: () => {
        graphStore.selectAll()
      },
    },
  ])
}

/**
 * Register view commands (zoom, fit, toggle panels)
 */
function registerViewCommands(commandStore: ReturnType<typeof useCommandStore>): void {
  const graphStore = useGraphStore()

  commandStore.registerCommands([
    {
      id: 'FlowForge.ZoomIn',
      label: 'Zoom In',
      menubarLabel: 'Zoom In',
      icon: 'zoom-in',
      tooltip: 'Zoom in on the canvas',
      category: 'view-controls',
      function: () => {
        graphStore.setCanvasScale(graphStore.canvasScale * 1.2)
      },
    },
    {
      id: 'FlowForge.ZoomOut',
      label: 'Zoom Out',
      menubarLabel: 'Zoom Out',
      icon: 'zoom-out',
      tooltip: 'Zoom out of the canvas',
      category: 'view-controls',
      function: () => {
        graphStore.setCanvasScale(graphStore.canvasScale / 1.2)
      },
    },
    {
      id: 'FlowForge.FitView',
      label: 'Fit to View',
      menubarLabel: 'Fit to View',
      icon: 'maximize',
      tooltip: 'Fit all nodes in view',
      category: 'view-controls',
      function: () => {
        graphStore.resetView()
      },
    },
    {
      id: 'FlowForge.ToggleSidebar',
      label: 'Toggle Sidebar',
      menubarLabel: 'Toggle Sidebar',
      icon: 'sidebar',
      tooltip: 'Show or hide the sidebar',
      category: 'view-controls',
      function: () => {
        // Placeholder - implement sidebar toggle
        console.log('Toggle sidebar not yet implemented')
      },
    },
    {
      id: 'FlowForge.ToggleBottomPanel',
      label: 'Toggle Bottom Panel',
      menubarLabel: 'Toggle Bottom Panel',
      icon: 'panel-bottom',
      tooltip: 'Show or hide the bottom panel',
      category: 'view-controls',
      function: () => {
        // Placeholder - implement bottom panel toggle
        console.log('Toggle bottom panel not yet implemented')
      },
    },
  ])
}

/**
 * Register graph commands (add node, delete, group)
 */
function registerGraphCommands(commandStore: ReturnType<typeof useCommandStore>): void {
  const graphStore = useGraphStore()

  commandStore.registerCommands([
    {
      id: 'FlowForge.AddNode',
      label: 'Add Node',
      menubarLabel: 'Add Node',
      icon: 'plus-circle',
      tooltip: 'Add a new node to the graph',
      category: 'graph',
      function: () => {
        // Placeholder - implement node palette/search
        console.log('Add node not yet implemented')
      },
    },
    {
      id: 'FlowForge.DeleteSelected',
      label: 'Delete Selected',
      menubarLabel: 'Delete Selected',
      icon: 'trash',
      tooltip: 'Delete selected nodes and links',
      category: 'graph',
      function: () => {
        graphStore.deleteSelected()
      },
    },
    {
      id: 'FlowForge.GroupSelected',
      label: 'Group Selected',
      menubarLabel: 'Group Selected',
      icon: 'folder',
      tooltip: 'Group selected nodes',
      category: 'graph',
      function: () => {
        const selectedIds = Array.from(graphStore.selectedNodeIds)
        if (selectedIds.length > 0) {
          try {
            graphStore.createGroup('New Group', selectedIds)
          } catch (error) {
            console.error('Failed to create group:', error)
          }
        }
      },
    },
    {
      id: 'FlowForge.UngroupSelected',
      label: 'Ungroup Selected',
      menubarLabel: 'Ungroup Selected',
      icon: 'folder-minus',
      tooltip: 'Remove grouping from selected nodes',
      category: 'graph',
      function: () => {
        // Placeholder - implement ungroup
        console.log('Ungroup not yet implemented')
      },
    },
  ])
}

/**
 * Reset initialization state (useful for testing)
 */
export function resetStoreInitialization(): void {
  initialized = false
}
