/**
 * Diff Preview Composable
 *
 * Provides diff generation and preview management for code changes.
 * Integrates with the Python backend for diff calculation and manages
 * the diff preview dialog state.
 *
 * @module composables/useDiffPreview
 */

import { ref, computed, readonly, shallowRef, markRaw, type Ref, type ComputedRef } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import { useDialogStore } from '@/stores/dialogStore'
import { useGraphStore } from '@/stores/graphStore'
import { useWorkspaceStore } from '@/stores/workspaceStore'
import {
  applyContent as applyChangesToFile,
  type DiffLineType,
  type DiffLine,
  type DiffHunk,
  type DiffStats,
  type DiffResult,
  type SideBySideLine,
  type SideBySideDiff,
  type ApplyResult,
} from '@/bridge/codegen'
import { UI_CONFIG } from '@/config/flowforge.config'

// Re-export diff types for consumers of this composable
export type { DiffLineType, DiffLine, DiffHunk, DiffStats, DiffResult, SideBySideLine, SideBySideDiff, ApplyResult }

/**
 * Diff view mode.
 */
export type DiffViewMode = 'unified' | 'split'

/**
 * Options for the useDiffPreview composable.
 */
export interface UseDiffPreviewOptions {
  /** Default view mode */
  defaultViewMode?: DiffViewMode
  /** Callback when changes are applied successfully */
  onApply?: (filePath: string, backupPath: string | null) => void
  /** Callback when preview is cancelled */
  onCancel?: () => void
  /** Callback on error */
  onError?: (error: Error) => void
  /** Whether to show toast notifications (default: true) */
  showToasts?: boolean
  /** Whether to refresh graph after apply (default: true) */
  refreshGraphOnApply?: boolean
}

/**
 * Return type for useDiffPreview composable.
 */
export interface UseDiffPreviewReturn {
  /** Current diff result */
  readonly diffResult: ComputedRef<DiffResult | null>
  /** Side-by-side diff data */
  readonly sideBySideDiff: ComputedRef<SideBySideDiff | null>
  /** Whether diff is being generated */
  readonly isLoading: Ref<boolean>
  /** Error message if any */
  readonly error: Ref<string | null>
  /** Whether there are changes to apply */
  readonly hasChanges: ComputedRef<boolean>
  /** Current view mode */
  readonly viewMode: Ref<DiffViewMode>
  /** Original source code */
  readonly originalSource: Ref<string>
  /** Modified source code */
  readonly modifiedSource: Ref<string>
  /** File path being modified */
  readonly filePath: Ref<string | null>
  /** Whether applying changes */
  readonly isApplying: Ref<boolean>
  /** Last apply result */
  readonly lastApplyResult: Ref<ApplyResult | null>

  /** Generate diff between original and modified source */
  generateDiff: (original: string, modified: string, filename?: string, filePath?: string) => Promise<DiffResult | null>
  /** Generate side-by-side diff */
  generateSideBySideDiff: (original: string, modified: string, filename?: string) => Promise<SideBySideDiff | null>
  /** Show the diff preview dialog */
  showDiffPreview: (original: string, modified: string, filename?: string, filePath?: string) => Promise<void>
  /** Apply the changes (write to file with backup) */
  applyChanges: () => Promise<ApplyResult>
  /** Cancel and close the preview */
  cancel: () => void
  /** Toggle view mode */
  toggleViewMode: () => void
  /** Set view mode */
  setViewMode: (mode: DiffViewMode) => void
  /** Clear the diff state */
  clear: () => void
}

// =============================================================================
// CONSTANTS
// =============================================================================

const DIALOG_KEY = 'diff-preview-dialog'

// =============================================================================
// COMPOSABLE
// =============================================================================

/**
 * Composable for managing diff generation and preview.
 *
 * @example
 * ```typescript
 * const {
 *   diffResult,
 *   isLoading,
 *   hasChanges,
 *   generateDiff,
 *   showDiffPreview,
 *   applyChanges,
 * } = useDiffPreview({
 *   onApply: (path) => console.log('Applied to:', path),
 * })
 *
 * // Generate and show diff
 * await showDiffPreview(originalCode, modifiedCode, 'example.py', '/path/to/example.py')
 * ```
 */
export function useDiffPreview(options: UseDiffPreviewOptions = {}): UseDiffPreviewReturn {
  const {
    defaultViewMode = 'unified',
    onApply,
    onCancel,
    onError,
    showToasts = true,
    refreshGraphOnApply = true,
  } = options

  const dialogStore = useDialogStore()
  const graphStore = useGraphStore()
  const workspaceStore = useWorkspaceStore()

  // ---------------------------------------------------------------------------
  // STATE
  // ---------------------------------------------------------------------------

  const diffResultRef = shallowRef<DiffResult | null>(null)
  const sideBySideDiffRef = shallowRef<SideBySideDiff | null>(null)
  const isLoading = ref(false)
  const error = ref<string | null>(null)
  const viewMode = ref<DiffViewMode>(defaultViewMode)
  const originalSource = ref('')
  const modifiedSource = ref('')
  const filePath = ref<string | null>(null)
  const isApplying = ref(false)
  const lastApplyResult = ref<ApplyResult | null>(null)

  // ---------------------------------------------------------------------------
  // COMPUTED
  // ---------------------------------------------------------------------------

  const diffResult = computed(() => diffResultRef.value)
  const sideBySideDiff = computed(() => sideBySideDiffRef.value)

  const hasChanges = computed(() => {
    return diffResultRef.value?.hasChanges ?? false
  })

  // ---------------------------------------------------------------------------
  // METHODS
  // ---------------------------------------------------------------------------

  /**
   * Generate a unified diff between original and modified source.
   */
  async function generateDiff(
    original: string,
    modified: string,
    filename: string = '',
    path?: string,
  ): Promise<DiffResult | null> {
    isLoading.value = true
    error.value = null

    try {
      const result = await invoke<DiffResult>('ipc_call', {
        request: {
          id: `diff-${Date.now()}`,
          method: 'generate_diff',
          params: {
            original,
            modified,
            filename,
            original_path: path,
            context_lines: UI_CONFIG.diff.contextLines,
            side_by_side: false,
          },
        },
      })

      // Handle IPC response format
      const diffData = (result as unknown as { result?: DiffResult })?.result ?? result

      diffResultRef.value = diffData
      originalSource.value = original
      modifiedSource.value = modified
      filePath.value = path ?? null

      console.log('[useDiffPreview] Generated diff:', {
        hasChanges: diffData.hasChanges,
        additions: diffData.stats.additions,
        deletions: diffData.stats.deletions,
      })

      return diffData
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to generate diff'
      error.value = errorMessage
      console.error('[useDiffPreview] Error generating diff:', err)

      if (onError && err instanceof Error) {
        onError(err)
      }

      return null
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Generate a side-by-side diff.
   */
  async function generateSideBySideDiff(
    original: string,
    modified: string,
    filename: string = '',
  ): Promise<SideBySideDiff | null> {
    isLoading.value = true
    error.value = null

    try {
      const result = await invoke<SideBySideDiff>('ipc_call', {
        request: {
          id: `diff-sbs-${Date.now()}`,
          method: 'generate_diff',
          params: {
            original,
            modified,
            filename,
            side_by_side: true,
          },
        },
      })

      // Handle IPC response format
      const diffData = (result as unknown as { result?: SideBySideDiff })?.result ?? result

      sideBySideDiffRef.value = diffData

      console.log('[useDiffPreview] Generated side-by-side diff')

      return diffData
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to generate side-by-side diff'
      error.value = errorMessage
      console.error('[useDiffPreview] Error generating side-by-side diff:', err)

      if (onError && err instanceof Error) {
        onError(err)
      }

      return null
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Show the diff preview dialog.
   */
  async function showDiffPreview(
    original: string,
    modified: string,
    filename: string = '',
    path?: string,
  ): Promise<void> {
    // Store the file path and sources
    filePath.value = path ?? null
    originalSource.value = original
    modifiedSource.value = modified

    // Generate both diff formats
    const [unifiedResult] = await Promise.all([
      generateDiff(original, modified, filename, path),
      generateSideBySideDiff(original, modified, filename),
    ])

    if (!unifiedResult) {
      console.error('[useDiffPreview] Failed to generate diff for preview')
      return
    }

    // Open the diff preview dialog via dialogStore
    const DiffPreviewDialog = (await import('@/components/dialogs/DiffPreviewDialog.vue')).default
    dialogStore.showDialog({
      key: DIALOG_KEY,
      title: 'Code Changes Preview',
      component: markRaw(DiffPreviewDialog),
      props: {
        diffResult: diffResultRef.value,
        sideBySideDiff: sideBySideDiffRef.value,
        viewMode: viewMode.value,
        filePath: filePath.value,
        isLoading: isLoading.value,
        error: error.value,
        isApplying: isApplying.value,
        // Event handlers (Vue maps onXxx to emitted events)
        onApply: () => applyChanges(),
        onCancel: () => cancel(),
        'onSet-view-mode': (mode: DiffViewMode) => setViewMode(mode),
      },
      dialogComponentProps: {
        modal: true,
        closable: true,
        closeOnEscape: true,
        dismissableMask: false,
        maximizable: true,
      },
    })

    console.log('[useDiffPreview] Diff preview dialog opened')
  }

  /**
   * Apply the changes by writing to file with backup.
   *
   * This function:
   * 1. Creates a timestamped backup of the original file
   * 2. Writes the modified content to the file
   * 3. Updates the graph store (clears modified flag, updates mtime)
   * 4. Shows success/error toast notifications
   * 5. Closes the dialog on success
   */
  async function applyChanges(): Promise<ApplyResult> {
    const failureResult = (errorMsg: string): ApplyResult => ({
      success: false,
      backupPath: null,
      error: errorMsg,
    })

    if (!filePath.value) {
      const result = failureResult('No file path specified')
      error.value = result.error
      lastApplyResult.value = result
      return result
    }

    if (!hasChanges.value) {
      const result = failureResult('No changes to apply')
      error.value = result.error
      lastApplyResult.value = result
      return result
    }

    isApplying.value = true
    error.value = null

    try {
      // Call the codegen bridge to apply changes with backup
      const result = await applyChangesToFile(
        filePath.value,
        modifiedSource.value,
        true // Always create backup
      )

      lastApplyResult.value = result

      if (!result.success) {
        error.value = result.error
        console.error('[useDiffPreview] Failed to apply changes:', result.error)

        // Show error toast
        if (showToasts) {
          workspaceStore.addToast({
            severity: 'error',
            summary: 'Failed to Apply Changes',
            detail: result.error ?? 'Unknown error occurred',
            life: 5000,
          })
        }

        if (onError) {
          onError(new Error(result.error ?? 'Failed to apply changes'))
        }

        return result
      }

      console.log('[useDiffPreview] Applied changes to:', filePath.value, 'Backup:', result.backupPath)

      // Update graph store state
      if (refreshGraphOnApply) {
        // Clear modified flag since we just saved
        graphStore.markSaved()

        // Update the file path if it was a new file
        if (graphStore.currentFilePath !== filePath.value) {
          graphStore.setCurrentFile(filePath.value)
        }

        // Update last known mtime to prevent false external change detection
        await graphStore.updateLastMtime()
      }

      // Show success toast
      if (showToasts) {
        const backupInfo = result.backupPath
          ? ` Backup saved.`
          : ''
        workspaceStore.addToast({
          severity: 'success',
          summary: 'Changes Applied',
          detail: `Successfully saved to ${filePath.value}.${backupInfo}`,
          life: 3000,
        })
      }

      // Call user callback
      if (onApply) {
        onApply(filePath.value, result.backupPath)
      }

      // Close dialog on success
      dialogStore.closeDialog({ key: DIALOG_KEY })

      return result
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to apply changes'
      error.value = errorMessage
      console.error('[useDiffPreview] Error applying changes:', err)

      const result: ApplyResult = {
        success: false,
        backupPath: null,
        error: errorMessage,
      }
      lastApplyResult.value = result

      // Show error toast
      if (showToasts) {
        workspaceStore.addToast({
          severity: 'error',
          summary: 'Failed to Apply Changes',
          detail: errorMessage,
          life: 5000,
        })
      }

      if (onError && err instanceof Error) {
        onError(err)
      }

      return result
    } finally {
      isApplying.value = false
    }
  }

  /**
   * Cancel and close the preview.
   */
  function cancel(): void {
    dialogStore.closeDialog({ key: DIALOG_KEY })

    if (onCancel) {
      onCancel()
    }

    console.log('[useDiffPreview] Preview cancelled')
  }

  /**
   * Toggle between unified and split view modes.
   */
  function toggleViewMode(): void {
    viewMode.value = viewMode.value === 'unified' ? 'split' : 'unified'
    console.log('[useDiffPreview] View mode:', viewMode.value)
  }

  /**
   * Set the view mode.
   */
  function setViewMode(mode: DiffViewMode): void {
    viewMode.value = mode
  }

  /**
   * Clear all diff state.
   */
  function clear(): void {
    diffResultRef.value = null
    sideBySideDiffRef.value = null
    error.value = null
    originalSource.value = ''
    modifiedSource.value = ''
    filePath.value = null
    lastApplyResult.value = null
    console.log('[useDiffPreview] State cleared')
  }

  // ---------------------------------------------------------------------------
  // RETURN
  // ---------------------------------------------------------------------------

  return {
    diffResult: readonly(diffResult) as ComputedRef<DiffResult | null>,
    sideBySideDiff: readonly(sideBySideDiff) as ComputedRef<SideBySideDiff | null>,
    isLoading: readonly(isLoading) as Ref<boolean>,
    error: readonly(error) as Ref<string | null>,
    hasChanges,
    viewMode,
    originalSource: readonly(originalSource) as Ref<string>,
    modifiedSource: readonly(modifiedSource) as Ref<string>,
    filePath: readonly(filePath) as Ref<string | null>,
    isApplying: readonly(isApplying) as Ref<boolean>,
    lastApplyResult: readonly(lastApplyResult) as Ref<ApplyResult | null>,
    generateDiff,
    generateSideBySideDiff,
    showDiffPreview,
    applyChanges,
    cancel,
    toggleViewMode,
    setViewMode,
    clear,
  }
}

export default useDiffPreview
