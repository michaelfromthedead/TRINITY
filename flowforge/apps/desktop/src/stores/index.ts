/**
 * FlowForge Stores - Pinia store exports
 *
 * These stores are extracted from ComfyUI frontend with SD-specific code removed.
 * They provide the foundation for FlowForge's state management.
 */

// Dialog management
export { useDialogStore } from './dialogStore'
export type { DialogComponentProps, ShowDialogOptions } from './dialogStore'

// Command palette and actions
export { useCommandStore, FlowForgeCommandImpl } from './commandStore'
export type { FlowForgeCommand } from './commandStore'

// Keyboard shortcuts
export { useKeybindingStore, KeybindingImpl, KeyComboImpl } from './keybindingStore'
export type { Keybinding, KeyCombo } from './keybindingStore'

// Menu bar
export { useMenuItemStore } from './menuItemStore'
export type { MenuItem } from './menuItemStore'

// Sidebar panels
export { useSidebarTabStore } from './sidebarTabStore'
export type { SidebarTabExtension } from './sidebarTabStore'

// Bottom panels (terminal, output, etc.)
export { useBottomPanelStore } from './bottomPanelStore'
export type { BottomPanelExtension, PanelType } from './bottomPanelStore'

// Node graph state (NEW - FlowForge specific)
export { useGraphStore } from './graphStore'
export type {
  GraphNode,
  GraphNodeSlot,
  GraphWidget,
  GraphLink,
  GraphGroup,
  GraphState,
  GraphHistoryEntry
} from './graphStore'

// Trinity node type definitions
export { useNodeDefStore } from './nodeDefStore'

// Trinity runtime introspection
export { useTrinityStore } from './trinityStore'
export type {
  TrinityStatus,
  RegistryEntry,
  RegistryEntryType,
  TrinityInstance,
  TrinityEvent,
  RegistryContents,
  InstancesQueryResult,
  RecentEventsResult,
} from './trinityStore'

// Central workspace state
export { useWorkspaceStore } from './workspaceStore'
export type { ToastMessage } from './workspaceStore'

// Edit history tracking
export { useEditHistoryStore } from './editHistoryStore'
export type {
  PendingChange,
  SaveResult,
  GraphDiff
} from './editHistoryStore'

// File explorer for workspace navigation
export { useFileExplorerStore } from './fileExplorerStore'

// Store initialization
export { initializeStores, resetStoreInitialization } from './initializeStores'
