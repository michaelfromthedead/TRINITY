/**
 * Keyboard Shortcuts Composable
 *
 * Provides centralized keyboard shortcut handling with enhanced undo/redo support.
 * Integrates with graphStore and editHistoryStore for file operations.
 */
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useGraphStore } from '@/stores/graphStore'
import { useEditHistoryStore } from '@/stores/editHistoryStore'
import { useCommandStore } from '@/stores/commandStore'
import { useKeybindingStore, KeyComboImpl } from '@/stores/keybindingStore'

/**
 * Options for keyboard shortcuts composable.
 */
export interface UseKeyboardShortcutsOptions {
  /** Whether to enable global keyboard handling (default: true) */
  enableGlobal?: boolean
  /** Element to attach listeners to (default: window) */
  target?: EventTarget
  /** Callback when undo is triggered */
  onUndo?: (description: string | null) => void
  /** Callback when redo is triggered */
  onRedo?: (description: string | null) => void
  /** Callback when save is triggered */
  onSave?: () => void
  /** Callback when delete is triggered (returns true if confirmed) */
  onDelete?: () => Promise<boolean> | boolean
  /** Callback when add node is triggered */
  onAddNode?: () => void
  /** Whether to show toast notifications (default: true) */
  showToasts?: boolean
  /** Whether to require confirmation for delete (default: true) */
  requireDeleteConfirmation?: boolean
}

/**
 * Creates keyboard shortcuts composable with enhanced undo/redo support.
 */
export function useKeyboardShortcuts(options: UseKeyboardShortcutsOptions = {}) {
  const {
    enableGlobal = true,
    target = typeof window !== 'undefined' ? window : null,
    onUndo,
    onRedo,
    onSave,
    onDelete,
    onAddNode,
    // showToasts can be used for future toast notification integration
  } = options

  const graphStore = useGraphStore()
  const editHistoryStore = useEditHistoryStore()
  const commandStore = useCommandStore()
  const keybindingStore = useKeybindingStore()

  // State
  const isShiftDown = ref(false)
  const isCtrlDown = ref(false)
  const isAltDown = ref(false)
  const lastAction = ref<string | null>(null)

  // Computed
  const canUndo = computed(() => graphStore.canUndo)
  const canRedo = computed(() => graphStore.canRedo)
  const undoDescription = computed(() => graphStore.getUndoDescription())
  const redoDescription = computed(() => graphStore.getRedoDescription())
  const hasPendingChanges = computed(() => editHistoryStore.hasPendingChanges)

  /**
   * Handle undo action.
   */
  function handleUndo(): boolean {
    if (!canUndo.value) {
      return false
    }

    const description = undoDescription.value
    const result = graphStore.undo()

    if (result) {
      lastAction.value = `Undo: ${description ?? 'action'}`

      if (onUndo) {
        onUndo(description)
      }

      return true
    }

    return false
  }

  /**
   * Handle redo action.
   */
  function handleRedo(): boolean {
    if (!canRedo.value) {
      return false
    }

    const description = redoDescription.value
    const result = graphStore.redo()

    if (result) {
      lastAction.value = `Redo: ${description ?? 'action'}`

      if (onRedo) {
        onRedo(description)
      }

      return true
    }

    return false
  }

  /**
   * Handle save action.
   */
  async function handleSave(): Promise<boolean> {
    if (onSave) {
      onSave()
      return true
    }

    // Execute the save command
    try {
      await commandStore.execute('File.Save')
      lastAction.value = 'File saved'
      return true
    } catch (error) {
      console.error('Save failed:', error)
      return false
    }
  }

  /**
   * Handle delete action for selected nodes.
   */
  async function handleDelete(): Promise<boolean> {
    if (!graphStore.hasSelection) {
      return false
    }

    // Use custom handler if provided
    if (onDelete) {
      const result = await onDelete()
      if (result) {
        lastAction.value = 'Deleted selection'
      }
      return result
    }

    // Default: delete without custom confirmation (confirmation should be handled externally)
    graphStore.deleteSelected()
    lastAction.value = 'Deleted selection'
    return true
  }

  /**
   * Handle add node action.
   */
  function handleAddNode(): void {
    if (onAddNode) {
      onAddNode()
      lastAction.value = 'Opening add node dialog'
    }
  }

  /**
   * Check if target is an input element where shortcuts should be disabled.
   */
  function isInputElement(target: EventTarget | null): boolean {
    if (!target || !(target instanceof HTMLElement)) {
      return false
    }

    const tagName = target.tagName.toUpperCase()
    return tagName === 'INPUT' || tagName === 'TEXTAREA' || target.isContentEditable
  }

  /**
   * Handle keydown event.
   */
  function handleKeyDown(event: KeyboardEvent): void {
    // Update modifier state
    if (event.key === 'Shift') {
      isShiftDown.value = true
    }
    if (event.key === 'Control' || event.key === 'Meta') {
      isCtrlDown.value = true
    }
    if (event.key === 'Alt') {
      isAltDown.value = true
    }

    // Skip if typing in an input
    if (isInputElement(event.target)) {
      return
    }

    // Handle Delete key for deleting selected nodes
    if (event.key === 'Delete' || event.key === 'Backspace') {
      if (graphStore.hasSelection) {
        event.preventDefault()
        event.stopPropagation()
        handleDelete()
        return
      }
    }

    // Handle Ctrl+N or Cmd+Shift+N for adding new node
    if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'N') {
      event.preventDefault()
      event.stopPropagation()
      handleAddNode()
      return
    }

    // Check for registered keybindings
    const combo = KeyComboImpl.fromEvent(event)
    const keybinding = keybindingStore.getKeybinding(combo)

    if (keybinding) {
      event.preventDefault()
      event.stopPropagation()

      // Handle specific commands with enhanced behavior
      switch (keybinding.commandId) {
        case 'Edit.Undo':
        case 'FlowForge.Undo':
          handleUndo()
          break
        case 'Edit.Redo':
        case 'FlowForge.Redo':
          handleRedo()
          break
        case 'File.Save':
        case 'FlowForge.SaveFile':
          handleSave()
          break
        case 'Edit.Delete':
        case 'FlowForge.DeleteSelected':
          handleDelete()
          break
        case 'Edit.AddNode':
        case 'FlowForge.AddNode':
          handleAddNode()
          break
        default:
          // Execute other commands through command store
          commandStore.execute(keybinding.commandId)
      }
    }
  }

  /**
   * Handle keyup event.
   */
  function handleKeyUp(event: KeyboardEvent): void {
    if (event.key === 'Shift') {
      isShiftDown.value = false
    }
    if (event.key === 'Control' || event.key === 'Meta') {
      isCtrlDown.value = false
    }
    if (event.key === 'Alt') {
      isAltDown.value = false
    }
  }

  /**
   * Set up event listeners.
   */
  function setup(): void {
    if (!enableGlobal || !target) {
      return
    }

    target.addEventListener('keydown', handleKeyDown as EventListener)
    target.addEventListener('keyup', handleKeyUp as EventListener)
  }

  /**
   * Clean up event listeners.
   */
  function cleanup(): void {
    if (!target) {
      return
    }

    target.removeEventListener('keydown', handleKeyDown as EventListener)
    target.removeEventListener('keyup', handleKeyUp as EventListener)
  }

  // Auto setup/cleanup with lifecycle
  onMounted(() => {
    setup()
  })

  onUnmounted(() => {
    cleanup()
  })

  return {
    // State
    isShiftDown,
    isCtrlDown,
    isAltDown,
    lastAction,

    // Computed
    canUndo,
    canRedo,
    undoDescription,
    redoDescription,
    hasPendingChanges,

    // Actions
    handleUndo,
    handleRedo,
    handleSave,
    handleDelete,
    handleAddNode,

    // Lifecycle
    setup,
    cleanup,
  }
}

export type UseKeyboardShortcutsReturn = ReturnType<typeof useKeyboardShortcuts>
