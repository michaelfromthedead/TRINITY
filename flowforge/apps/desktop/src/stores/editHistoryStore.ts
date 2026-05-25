/**
 * Edit History Store - Tracks pending changes and file save state
 *
 * Manages the relationship between in-memory graph state and persisted file state.
 * Provides functionality for tracking, saving, and discarding changes.
 */
import { defineStore } from 'pinia'
import { computed, ref, shallowRef, watch } from 'vue'
import { useGraphStore, type GraphState } from './graphStore'
import { writePythonFile, writePythonFileWithBackup } from '@/bridge/files'
import type { WriteResult } from '@/bridge/files'

/**
 * Represents a pending change that hasn't been saved to disk.
 */
export interface PendingChange {
  /** Unique identifier for the change */
  id: string
  /** Type of change */
  type: 'add_node' | 'remove_node' | 'update_node' | 'add_link' | 'remove_link' | 'add_group' | 'remove_group' | 'move_node' | 'other'
  /** Human-readable description */
  description: string
  /** Timestamp of the change */
  timestamp: number
  /** Affected node/link IDs */
  affectedIds?: string[]
}

/**
 * Result of a save operation.
 */
export interface SaveResult {
  success: boolean
  path?: string
  error?: string
  backupPath?: string
}

/**
 * Diff between two graph states.
 */
export interface GraphDiff {
  addedNodes: string[]
  removedNodes: string[]
  modifiedNodes: string[]
  addedLinks: string[]
  removedLinks: string[]
  addedGroups: string[]
  removedGroups: string[]
  hasChanges: boolean
}

export const useEditHistoryStore = defineStore('editHistory', () => {
  const graphStore = useGraphStore()

  // Last saved state (snapshot of state when file was last saved)
  const lastSavedState = shallowRef<GraphState | null>(null)
  const lastSavedPath = ref<string | null>(null)
  const lastSavedTimestamp = ref<number | null>(null)

  // Pending changes tracking
  const pendingChanges = ref<PendingChange[]>([])
  const changeIdCounter = ref(0)

  // Save operation state
  const isSaving = ref(false)
  const lastSaveError = ref<string | null>(null)
  const autoSaveEnabled = ref(true)
  const autoSaveInterval = ref(60000) // 1 minute default

  // Computed properties
  const hasPendingChanges = computed(() => pendingChanges.value.length > 0)
  const pendingChangeCount = computed(() => pendingChanges.value.length)

  const currentState = computed<GraphState>(() => graphStore.getGraphState())

  /**
   * Calculate diff between current state and last saved state.
   */
  const stateDiff = computed<GraphDiff>(() => {
    if (!lastSavedState.value) {
      return {
        addedNodes: currentState.value.nodes.map(n => n.id),
        removedNodes: [],
        modifiedNodes: [],
        addedLinks: currentState.value.links.map(l => l.id),
        removedLinks: [],
        addedGroups: currentState.value.groups.map(g => g.id),
        removedGroups: [],
        hasChanges: currentState.value.nodes.length > 0 || currentState.value.links.length > 0,
      }
    }

    const saved = lastSavedState.value
    const current = currentState.value

    // Calculate node differences
    const savedNodeIds = new Set(saved.nodes.map(n => n.id))
    const currentNodeIds = new Set(current.nodes.map(n => n.id))

    const addedNodes = current.nodes.filter(n => !savedNodeIds.has(n.id)).map(n => n.id)
    const removedNodes = saved.nodes.filter(n => !currentNodeIds.has(n.id)).map(n => n.id)

    // Find modified nodes (same ID but different content)
    const savedNodeMap = new Map(saved.nodes.map(n => [n.id, n]))
    const modifiedNodes = current.nodes
      .filter(n => savedNodeIds.has(n.id))
      .filter(n => JSON.stringify(n) !== JSON.stringify(savedNodeMap.get(n.id)))
      .map(n => n.id)

    // Calculate link differences
    const savedLinkIds = new Set(saved.links.map(l => l.id))
    const currentLinkIds = new Set(current.links.map(l => l.id))

    const addedLinks = current.links.filter(l => !savedLinkIds.has(l.id)).map(l => l.id)
    const removedLinks = saved.links.filter(l => !currentLinkIds.has(l.id)).map(l => l.id)

    // Calculate group differences
    const savedGroupIds = new Set(saved.groups.map(g => g.id))
    const currentGroupIds = new Set(current.groups.map(g => g.id))

    const addedGroups = current.groups.filter(g => !savedGroupIds.has(g.id)).map(g => g.id)
    const removedGroups = saved.groups.filter(g => !currentGroupIds.has(g.id)).map(g => g.id)

    const hasChanges = addedNodes.length > 0 || removedNodes.length > 0 || modifiedNodes.length > 0 ||
      addedLinks.length > 0 || removedLinks.length > 0 ||
      addedGroups.length > 0 || removedGroups.length > 0

    return {
      addedNodes,
      removedNodes,
      modifiedNodes,
      addedLinks,
      removedLinks,
      addedGroups,
      removedGroups,
      hasChanges,
    }
  })

  /**
   * Record a change as pending.
   */
  function recordChange(
    type: PendingChange['type'],
    description: string,
    affectedIds?: string[]
  ): string {
    const id = `change_${++changeIdCounter.value}`
    const change: PendingChange = {
      id,
      type,
      description,
      timestamp: Date.now(),
      affectedIds,
    }
    pendingChanges.value.push(change)
    return id
  }

  /**
   * Clear all pending changes.
   */
  function clearPendingChanges(): void {
    pendingChanges.value = []
  }

  /**
   * Mark the current state as saved.
   */
  function markAsSaved(path?: string): void {
    lastSavedState.value = JSON.parse(JSON.stringify(currentState.value))
    lastSavedPath.value = path ?? graphStore.currentFilePath
    lastSavedTimestamp.value = Date.now()
    clearPendingChanges()
    graphStore.markSaved()
    lastSaveError.value = null
  }

  /**
   * Save changes to file.
   *
   * @param path - File path to save to (uses current file if not provided)
   * @param content - Content to write (required)
   * @param createBackup - Whether to create a backup file
   */
  async function saveChanges(
    content: string,
    path?: string,
    createBackup = true
  ): Promise<SaveResult> {
    const savePath = path ?? graphStore.currentFilePath

    if (!savePath) {
      return {
        success: false,
        error: 'No file path specified',
      }
    }

    isSaving.value = true
    lastSaveError.value = null

    try {
      let result: WriteResult | boolean

      if (createBackup) {
        result = await writePythonFileWithBackup(savePath, content)
      } else {
        result = await writePythonFile(savePath, content)
      }

      // Handle both old boolean result and new WriteResult
      const success = typeof result === 'boolean' ? result : result.success
      const backupPath = typeof result === 'object' ? result.backupPath : undefined

      if (success) {
        markAsSaved(savePath)
        return {
          success: true,
          path: savePath,
          backupPath,
        }
      } else {
        const error = typeof result === 'object' && result.error ? result.error : 'Write failed'
        lastSaveError.value = error
        return {
          success: false,
          path: savePath,
          error,
        }
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      lastSaveError.value = errorMessage
      return {
        success: false,
        path: savePath,
        error: errorMessage,
      }
    } finally {
      isSaving.value = false
    }
  }

  /**
   * Discard all changes and revert to last saved state.
   */
  function discardChanges(): boolean {
    if (!lastSavedState.value) {
      // No saved state, clear the graph
      graphStore.clearGraph()
      clearPendingChanges()
      return true
    }

    // Restore the last saved state
    graphStore.loadGraphState(lastSavedState.value)
    graphStore.markSaved()
    clearPendingChanges()
    return true
  }

  /**
   * Initialize with a saved state (call after loading a file).
   */
  function initializeFromFile(state: GraphState, path: string): void {
    lastSavedState.value = JSON.parse(JSON.stringify(state))
    lastSavedPath.value = path
    lastSavedTimestamp.value = Date.now()
    clearPendingChanges()
  }

  /**
   * Get a human-readable summary of pending changes.
   */
  function getChangesSummary(): string {
    const diff = stateDiff.value

    const parts: string[] = []

    if (diff.addedNodes.length > 0) {
      parts.push(`${diff.addedNodes.length} node(s) added`)
    }
    if (diff.removedNodes.length > 0) {
      parts.push(`${diff.removedNodes.length} node(s) removed`)
    }
    if (diff.modifiedNodes.length > 0) {
      parts.push(`${diff.modifiedNodes.length} node(s) modified`)
    }
    if (diff.addedLinks.length > 0) {
      parts.push(`${diff.addedLinks.length} connection(s) added`)
    }
    if (diff.removedLinks.length > 0) {
      parts.push(`${diff.removedLinks.length} connection(s) removed`)
    }
    if (diff.addedGroups.length > 0) {
      parts.push(`${diff.addedGroups.length} group(s) added`)
    }
    if (diff.removedGroups.length > 0) {
      parts.push(`${diff.removedGroups.length} group(s) removed`)
    }

    if (parts.length === 0) {
      return 'No changes'
    }

    return parts.join(', ')
  }

  /**
   * Check if the file has been modified externally.
   * This would require file watching which is not implemented yet.
   */
  function checkExternalModification(): boolean {
    // TODO: Implement file watching to detect external changes
    return false
  }

  /**
   * Reset the store state.
   */
  function reset(): void {
    lastSavedState.value = null
    lastSavedPath.value = null
    lastSavedTimestamp.value = null
    pendingChanges.value = []
    changeIdCounter.value = 0
    isSaving.value = false
    lastSaveError.value = null
  }

  // Watch for graph modifications and sync with graphStore.isModified
  watch(
    () => stateDiff.value.hasChanges,
    (hasChanges) => {
      if (hasChanges && !graphStore.isModified) {
        graphStore.markModified()
      }
    }
  )

  return {
    // State
    lastSavedState,
    lastSavedPath,
    lastSavedTimestamp,
    pendingChanges,
    isSaving,
    lastSaveError,
    autoSaveEnabled,
    autoSaveInterval,

    // Computed
    hasPendingChanges,
    pendingChangeCount,
    currentState,
    stateDiff,

    // Actions
    recordChange,
    clearPendingChanges,
    markAsSaved,
    saveChanges,
    discardChanges,
    initializeFromFile,
    getChangesSummary,
    checkExternalModification,
    reset,
  }
})
