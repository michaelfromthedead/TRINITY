/**
 * File Watcher Composable
 *
 * Monitors open Python files for external changes and notifies when
 * the file has been modified outside of FlowForge.
 *
 * Uses polling via the Tauri `get_file_info` command to detect mtime changes.
 *
 * @module composables/useFileWatcher
 */

import { ref, onUnmounted, type Ref } from 'vue'
import { getFileInfo } from '@/bridge/files'
import { UI_CONFIG } from '@/config/flowforge.config'

// =============================================================================
// TYPES
// =============================================================================

/**
 * Options for the file watcher composable.
 */
export interface FileWatcherOptions {
  /** Polling interval in ms (default: 2000) */
  pollInterval?: number
  /** Callback when file changes externally */
  onExternalChange?: (filePath: string, newMtime: number) => void
  /** Callback on error */
  onError?: (filePath: string, error: Error) => void
}

/**
 * Internal state for a watched file.
 */
interface WatchedFile {
  /** File path */
  path: string
  /** Last known modification time (Unix seconds) */
  lastMtime: number
  /** Whether this file has unacknowledged external changes */
  hasExternalChanges: boolean
}

/**
 * Return type for useFileWatcher composable.
 */
export interface UseFileWatcherReturn {
  /** Start watching a file */
  watch: (filePath: string) => Promise<void>
  /** Stop watching a file */
  unwatch: (filePath: string) => void
  /** Stop watching all files */
  unwatchAll: () => void
  /** Check if file has been modified externally */
  hasExternalChanges: (filePath: string) => boolean
  /** Get last known mtime for a file */
  getLastMtime: (filePath: string) => number | null
  /** Update stored mtime (call after saving) */
  updateMtime: (filePath: string, mtime: number) => void
  /** Acknowledge external changes (clears the flag) */
  acknowledgeChanges: (filePath: string) => void
  /** Currently watched files */
  watchedFiles: Ref<Set<string>>
  /** Files with external changes */
  filesWithChanges: Ref<Set<string>>
}

// =============================================================================
// CONSTANTS
// =============================================================================

const DEFAULT_POLL_INTERVAL = UI_CONFIG.fileWatcher.pollInterval

// =============================================================================
// COMPOSABLE
// =============================================================================

/**
 * Composable for monitoring files for external changes.
 *
 * @example
 * ```typescript
 * const {
 *   watch,
 *   unwatch,
 *   hasExternalChanges,
 *   updateMtime,
 * } = useFileWatcher({
 *   pollInterval: 2000,
 *   onExternalChange: (path, mtime) => {
 *     console.log(`File ${path} changed externally at ${mtime}`)
 *   },
 * })
 *
 * // Start watching when file is opened
 * await watch('/path/to/file.py')
 *
 * // After saving, update the mtime
 * updateMtime('/path/to/file.py', newMtime)
 *
 * // Stop watching when file is closed
 * unwatch('/path/to/file.py')
 * ```
 */
export function useFileWatcher(options: FileWatcherOptions = {}): UseFileWatcherReturn {
  const {
    pollInterval = DEFAULT_POLL_INTERVAL,
    onExternalChange,
    onError,
  } = options

  // ---------------------------------------------------------------------------
  // STATE
  // ---------------------------------------------------------------------------

  /** Map of watched file paths to their state */
  const watchedFilesMap = new Map<string, WatchedFile>()

  /** Reactive set of watched file paths */
  const watchedFiles = ref<Set<string>>(new Set())

  /** Reactive set of files with external changes */
  const filesWithChanges = ref<Set<string>>(new Set())

  /** Polling interval ID */
  let pollIntervalId: ReturnType<typeof setInterval> | null = null

  /** Whether polling is active */
  let isPolling = false

  // ---------------------------------------------------------------------------
  // INTERNAL METHODS
  // ---------------------------------------------------------------------------

  /**
   * Start the polling loop if not already running.
   */
  function startPolling(): void {
    if (isPolling || pollIntervalId !== null) {
      return
    }

    isPolling = true
    pollIntervalId = setInterval(pollFiles, pollInterval)
    console.log('[useFileWatcher] Started polling with interval:', pollInterval)
  }

  /**
   * Stop the polling loop.
   */
  function stopPolling(): void {
    if (pollIntervalId !== null) {
      clearInterval(pollIntervalId)
      pollIntervalId = null
    }
    isPolling = false
    console.log('[useFileWatcher] Stopped polling')
  }

  /**
   * Poll all watched files for changes.
   */
  async function pollFiles(): Promise<void> {
    if (watchedFilesMap.size === 0) {
      stopPolling()
      return
    }

    const filesToPoll = Array.from(watchedFilesMap.values())

    await Promise.all(filesToPoll.map(checkFileForChanges))
  }

  /**
   * Check a single file for external changes.
   */
  async function checkFileForChanges(watchedFile: WatchedFile): Promise<void> {
    try {
      const info = await getFileInfo(watchedFile.path)

      if (!info.exists) {
        // File was deleted externally
        console.warn('[useFileWatcher] File no longer exists:', watchedFile.path)
        return
      }

      const currentMtime = info.modified

      if (currentMtime === undefined) {
        // Could not get mtime
        return
      }

      // Check if mtime changed since we last knew about it
      if (currentMtime > watchedFile.lastMtime && !watchedFile.hasExternalChanges) {
        console.log('[useFileWatcher] External change detected:', {
          path: watchedFile.path,
          oldMtime: watchedFile.lastMtime,
          newMtime: currentMtime,
        })

        watchedFile.hasExternalChanges = true
        filesWithChanges.value = new Set(filesWithChanges.value).add(watchedFile.path)

        if (onExternalChange) {
          onExternalChange(watchedFile.path, currentMtime)
        }
      }
    } catch (err) {
      console.error('[useFileWatcher] Error checking file:', watchedFile.path, err)

      if (onError && err instanceof Error) {
        onError(watchedFile.path, err)
      }
    }
  }

  // ---------------------------------------------------------------------------
  // PUBLIC METHODS
  // ---------------------------------------------------------------------------

  /**
   * Start watching a file for external changes.
   * Gets the initial mtime and adds the file to the watch list.
   */
  async function watch(filePath: string): Promise<void> {
    if (watchedFilesMap.has(filePath)) {
      console.log('[useFileWatcher] Already watching:', filePath)
      return
    }

    try {
      const info = await getFileInfo(filePath)

      if (!info.exists) {
        console.warn('[useFileWatcher] Cannot watch non-existent file:', filePath)
        return
      }

      const mtime = info.modified ?? Math.floor(Date.now() / 1000)

      const watchedFile: WatchedFile = {
        path: filePath,
        lastMtime: mtime,
        hasExternalChanges: false,
      }

      watchedFilesMap.set(filePath, watchedFile)
      watchedFiles.value = new Set(watchedFiles.value).add(filePath)

      console.log('[useFileWatcher] Now watching:', filePath, 'mtime:', mtime)

      // Start polling if this is the first file
      if (watchedFilesMap.size === 1) {
        startPolling()
      }
    } catch (err) {
      console.error('[useFileWatcher] Error starting watch:', filePath, err)

      if (onError && err instanceof Error) {
        onError(filePath, err)
      }
    }
  }

  /**
   * Stop watching a file.
   */
  function unwatch(filePath: string): void {
    if (!watchedFilesMap.has(filePath)) {
      return
    }

    watchedFilesMap.delete(filePath)

    const newWatchedFiles = new Set(watchedFiles.value)
    newWatchedFiles.delete(filePath)
    watchedFiles.value = newWatchedFiles

    const newFilesWithChanges = new Set(filesWithChanges.value)
    newFilesWithChanges.delete(filePath)
    filesWithChanges.value = newFilesWithChanges

    console.log('[useFileWatcher] Stopped watching:', filePath)

    // Stop polling if no more files
    if (watchedFilesMap.size === 0) {
      stopPolling()
    }
  }

  /**
   * Stop watching all files.
   */
  function unwatchAll(): void {
    watchedFilesMap.clear()
    watchedFiles.value = new Set()
    filesWithChanges.value = new Set()
    stopPolling()
    console.log('[useFileWatcher] Stopped watching all files')
  }

  /**
   * Check if a file has unacknowledged external changes.
   */
  function hasExternalChanges(filePath: string): boolean {
    const watchedFile = watchedFilesMap.get(filePath)
    return watchedFile?.hasExternalChanges ?? false
  }

  /**
   * Get the last known mtime for a watched file.
   */
  function getLastMtime(filePath: string): number | null {
    const watchedFile = watchedFilesMap.get(filePath)
    return watchedFile?.lastMtime ?? null
  }

  /**
   * Update the stored mtime for a file.
   * Call this after saving the file to reset the baseline.
   */
  function updateMtime(filePath: string, mtime: number): void {
    const watchedFile = watchedFilesMap.get(filePath)
    if (watchedFile) {
      watchedFile.lastMtime = mtime
      watchedFile.hasExternalChanges = false

      const newFilesWithChanges = new Set(filesWithChanges.value)
      newFilesWithChanges.delete(filePath)
      filesWithChanges.value = newFilesWithChanges

      console.log('[useFileWatcher] Updated mtime for:', filePath, 'to:', mtime)
    }
  }

  /**
   * Acknowledge external changes for a file.
   * Clears the hasExternalChanges flag without updating mtime.
   */
  function acknowledgeChanges(filePath: string): void {
    const watchedFile = watchedFilesMap.get(filePath)
    if (watchedFile) {
      watchedFile.hasExternalChanges = false

      const newFilesWithChanges = new Set(filesWithChanges.value)
      newFilesWithChanges.delete(filePath)
      filesWithChanges.value = newFilesWithChanges

      console.log('[useFileWatcher] Acknowledged changes for:', filePath)
    }
  }

  // ---------------------------------------------------------------------------
  // CLEANUP
  // ---------------------------------------------------------------------------

  onUnmounted(() => {
    unwatchAll()
  })

  // ---------------------------------------------------------------------------
  // RETURN
  // ---------------------------------------------------------------------------

  return {
    watch,
    unwatch,
    unwatchAll,
    hasExternalChanges,
    getLastMtime,
    updateMtime,
    acknowledgeChanges,
    watchedFiles,
    filesWithChanges,
  }
}

export default useFileWatcher
