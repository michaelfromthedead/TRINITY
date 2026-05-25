/**
 * File Conflict Composable
 *
 * Manages file conflict detection and resolution for FlowForge.
 * Integrates the graphStore file watcher with the FileConflictDialog.
 *
 * @module composables/useFileConflict
 */

import { ref, watch, onUnmounted, markRaw } from 'vue'
import { useGraphStore } from '@/stores/graphStore'
import { useDialogStore } from '@/stores/dialogStore'
import FileConflictDialog from '@/components/dialogs/FileConflictDialog.vue'
import DiffPreviewDialog from '@/components/dialogs/DiffPreviewDialog.vue'
import { readPythonFile } from '@/bridge/files'
import { generateCode } from '@/bridge/codegen'
import { useDiffPreview } from '@/composables/useDiffPreview'

// =============================================================================
// TYPES
// =============================================================================

export interface FileConflictOptions {
  /** Callback when conflict is resolved */
  onResolved?: (action: FileConflictAction) => void
  /** Callback when error occurs */
  onError?: (error: Error) => void
}

export type FileConflictAction = 'reload' | 'overwrite' | 'save-as' | 'compare' | 'cancel'

export interface UseFileConflictReturn {
  /** Whether a conflict is currently being shown */
  isConflictDialogOpen: ReturnType<typeof ref<boolean>>
  /** Show the conflict dialog manually */
  showConflictDialog: () => void
  /** Hide the conflict dialog */
  hideConflictDialog: () => void
  /** Handle reload action - discard local, load external */
  handleReload: () => Promise<void>
  /** Handle overwrite action - keep local, save to disk */
  handleOverwrite: () => Promise<void>
  /** Handle save-as action - save local to new file */
  handleSaveAs: () => Promise<void>
  /** Handle compare action - show diff preview */
  handleCompare: () => Promise<void>
  /** Handle cancel action - dismiss dialog without action */
  handleCancel: () => void
}

// =============================================================================
// CONSTANTS
// =============================================================================

const DIALOG_KEY = 'file-conflict-dialog'
const DIFF_DIALOG_KEY = 'file-conflict-diff-preview'

// =============================================================================
// COMPOSABLE
// =============================================================================

/**
 * Composable for managing file conflict detection and resolution.
 *
 * @example
 * ```typescript
 * const { isConflictDialogOpen } = useFileConflict({
 *   onResolved: (action) => console.log('Conflict resolved:', action),
 * })
 * ```
 */
export function useFileConflict(options: FileConflictOptions = {}): UseFileConflictReturn {
  const { onResolved, onError } = options

  // ---------------------------------------------------------------------------
  // STORES
  // ---------------------------------------------------------------------------

  const graphStore = useGraphStore()
  const dialogStore = useDialogStore()

  // ---------------------------------------------------------------------------
  // COMPOSABLES
  // ---------------------------------------------------------------------------

  const diffPreview = useDiffPreview({
    onCancel: () => {
      // Re-show conflict dialog when diff preview is cancelled
      showConflictDialog()
    },
  })

  // ---------------------------------------------------------------------------
  // STATE
  // ---------------------------------------------------------------------------

  const isConflictDialogOpen = ref(false)

  // ---------------------------------------------------------------------------
  // WATCH FOR EXTERNAL CHANGES
  // ---------------------------------------------------------------------------

  // Watch the graphStore's hasExternalChanges flag
  const stopWatch = watch(
    () => graphStore.hasExternalChanges,
    (hasChanges) => {
      if (hasChanges && !isConflictDialogOpen.value) {
        showConflictDialog()
      }
    },
    { immediate: true }
  )

  // Cleanup on unmount
  onUnmounted(() => {
    stopWatch()
  })

  // ---------------------------------------------------------------------------
  // DIALOG MANAGEMENT
  // ---------------------------------------------------------------------------

  /**
   * Show the file conflict dialog.
   */
  function showConflictDialog(): void {
    if (isConflictDialogOpen.value) {
      return
    }

    const filePath = graphStore.currentFilePath
    if (!filePath) {
      return
    }

    isConflictDialogOpen.value = true

    // Get current graph state as serialized content for the dialog
    const graphState = graphStore.getGraphState()
    const localContent = JSON.stringify(graphState, null, 2)

    dialogStore.showDialog({
      key: DIALOG_KEY,
      component: markRaw(FileConflictDialog),
      props: {
        filePath,
        localContent,
        onReload: handleReload,
        onOverwrite: handleOverwrite,
        onSaveAs: handleSaveAs,
        onCompare: handleCompare,
        onCancel: handleCancel,
      },
      dialogComponentProps: {
        modal: true,
        closable: true,
        closeOnEscape: false, // Don't allow ESC to dismiss - user must choose action
        dismissableMask: false, // Don't allow clicking outside to dismiss
      },
    })
  }

  /**
   * Hide the file conflict dialog.
   */
  function hideConflictDialog(): void {
    if (!isConflictDialogOpen.value) {
      return
    }

    dialogStore.closeDialog({ key: DIALOG_KEY })
    isConflictDialogOpen.value = false
  }

  // ---------------------------------------------------------------------------
  // ACTION HANDLERS
  // ---------------------------------------------------------------------------

  /**
   * Handle reload action - discard local changes, load from disk.
   */
  async function handleReload(): Promise<void> {
    try {
      await graphStore.reloadFromDisk()
      hideConflictDialog()
      onResolved?.('reload')
    } catch (error) {
      console.error('[useFileConflict] Error reloading file:', error)
      onError?.(error instanceof Error ? error : new Error(String(error)))
    }
  }

  /**
   * Handle overwrite action - keep local changes, write to disk.
   */
  async function handleOverwrite(): Promise<void> {
    try {
      const filePath = graphStore.currentFilePath
      if (!filePath) {
        throw new Error('No file path to overwrite')
      }

      // Save the current graph to the file
      await graphStore.saveToFile(filePath)

      // Acknowledge the external changes and update mtime
      graphStore.acknowledgeExternalChanges()
      await graphStore.updateLastMtime()

      hideConflictDialog()
      onResolved?.('overwrite')
    } catch (error) {
      console.error('[useFileConflict] Error overwriting file:', error)
      onError?.(error instanceof Error ? error : new Error(String(error)))
    }
  }

  /**
   * Handle save-as action - save local changes to a new file.
   */
  async function handleSaveAs(): Promise<void> {
    try {
      const savedPath = await graphStore.saveFileAs()

      if (savedPath) {
        // Acknowledge the external changes since we've saved to a new file
        graphStore.acknowledgeExternalChanges()
        hideConflictDialog()
        onResolved?.('save-as')
      }
      // If savedPath is null, user cancelled the save dialog - keep conflict dialog open
    } catch (error) {
      console.error('[useFileConflict] Error saving file as:', error)
      onError?.(error instanceof Error ? error : new Error(String(error)))
    }
  }

  /**
   * Handle compare action - show diff between local and external versions.
   *
   * This closes the conflict dialog and opens a diff preview showing:
   * - Left/Original: External file content (what's on disk)
   * - Right/Modified: Local graph's generated code (unsaved changes)
   *
   * The user can then decide to apply (overwrite external) or cancel (go back).
   */
  async function handleCompare(): Promise<void> {
    try {
      const filePath = graphStore.currentFilePath
      if (!filePath) {
        throw new Error('No file path to compare')
      }

      // Read the external file content (what's on disk)
      const externalResult = await readPythonFile(filePath)
      const externalContent = externalResult.content

      // Generate code from the current graph (local unsaved changes)
      const graph = graphStore.storeToApiGraph()
      const generatedResult = await generateCode(graph)

      if (!generatedResult.validation.success) {
        throw new Error(`Failed to generate code from graph: ${generatedResult.validation.errors?.join(', ') || 'Unknown error'}`)
      }

      const localContent = generatedResult.source

      // Close the conflict dialog before showing diff
      hideConflictDialog()

      // Generate and show the diff preview
      // Original = external (disk), Modified = local (graph)
      await diffPreview.showDiffPreview(
        externalContent,
        localContent,
        filePath.split('/').pop() || 'file.py',
        filePath
      )

      // Show the diff preview dialog
      dialogStore.showDialog({
        key: DIFF_DIALOG_KEY,
        component: markRaw(DiffPreviewDialog),
        props: {
          diffResult: diffPreview.diffResult.value,
          sideBySideDiff: diffPreview.sideBySideDiff.value,
          viewMode: diffPreview.viewMode.value,
          filePath: filePath,
          isLoading: diffPreview.isLoading.value,
          error: diffPreview.error.value,
          isApplying: diffPreview.isApplying.value,
          onApply: async () => {
            // Apply means overwrite external with local changes
            const result = await diffPreview.applyChanges()
            if (result.success) {
              dialogStore.closeDialog({ key: DIFF_DIALOG_KEY })
              graphStore.acknowledgeExternalChanges()
              await graphStore.updateLastMtime()
              onResolved?.('compare')
            }
          },
          onCancel: () => {
            dialogStore.closeDialog({ key: DIFF_DIALOG_KEY })
            // Re-show the conflict dialog so user can choose another action
            showConflictDialog()
          },
          'onSet-view-mode': (mode: 'unified' | 'split') => {
            diffPreview.setViewMode(mode)
          },
        },
        dialogComponentProps: {
          modal: true,
          closable: true,
          closeOnEscape: true,
          dismissableMask: false,
        },
      })

      console.log('[useFileConflict] Showing diff preview - external vs local')
    } catch (error) {
      console.error('[useFileConflict] Error comparing files:', error)
      onError?.(error instanceof Error ? error : new Error(String(error)))
    }
  }

  /**
   * Handle cancel action - dismiss dialog without taking action.
   * The conflict flag remains set, so the dialog may show again on next poll.
   */
  function handleCancel(): void {
    // Acknowledge the changes to prevent dialog from re-showing immediately
    // User explicitly chose to ignore the conflict for now
    graphStore.acknowledgeExternalChanges()
    hideConflictDialog()
    onResolved?.('cancel')
  }

  // ---------------------------------------------------------------------------
  // RETURN
  // ---------------------------------------------------------------------------

  return {
    isConflictDialogOpen,
    showConflictDialog,
    hideConflictDialog,
    handleReload,
    handleOverwrite,
    handleSaveAs,
    handleCompare,
    handleCancel,
  }
}

export default useFileConflict
