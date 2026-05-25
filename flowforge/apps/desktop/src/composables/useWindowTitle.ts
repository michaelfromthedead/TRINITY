/**
 * Window Title Composable
 *
 * Manages the application window title based on current file and modified state.
 * Title format: "FlowForge - filename.py" or "FlowForge - filename.py *" if modified.
 */
import { watch, onUnmounted, ref } from 'vue'
import { useGraphStore } from '../stores/graphStore'

const APP_NAME = 'FlowForge'
const MODIFIED_INDICATOR = '*'

export interface WindowTitleOptions {
  /** Custom app name (default: "FlowForge") */
  appName?: string
  /** Separator between app name and file name (default: " - ") */
  separator?: string
  /** Indicator for unsaved changes (default: "*") */
  modifiedIndicator?: string
  /** Whether to auto-start watching (default: true) */
  autoStart?: boolean
}

/**
 * Creates window title composable that watches graphStore and updates document.title.
 */
export function useWindowTitle(options?: WindowTitleOptions) {
  const graphStore = useGraphStore()

  const appName = options?.appName ?? APP_NAME
  const separator = options?.separator ?? ' - '
  const modifiedIndicator = options?.modifiedIndicator ?? MODIFIED_INDICATOR
  const autoStart = options?.autoStart ?? true

  const currentTitle = ref(appName)
  let stopWatch: (() => void) | null = null

  /**
   * Generates the window title based on current state.
   */
  function generateTitle(): string {
    const fileName = graphStore.fileName
    const isModified = graphStore.isModified

    let title = appName

    if (fileName) {
      title += separator + fileName
    }

    if (isModified) {
      title += ' ' + modifiedIndicator
    }

    return title
  }

  /**
   * Updates the document title.
   */
  function updateTitle(): void {
    const title = generateTitle()
    currentTitle.value = title
    document.title = title
  }

  /**
   * Starts watching for changes and updating the title.
   */
  function start(): void {
    if (stopWatch) {
      return // Already watching
    }

    // Watch both currentFilePath and isModified
    stopWatch = watch(
      () => [graphStore.currentFilePath, graphStore.isModified],
      () => {
        updateTitle()
      },
      { immediate: true }
    )
  }

  /**
   * Stops watching for changes.
   */
  function stop(): void {
    if (stopWatch) {
      stopWatch()
      stopWatch = null
    }
  }

  /**
   * Sets a custom title (overrides automatic title until next state change).
   */
  function setTitle(title: string): void {
    currentTitle.value = title
    document.title = title
  }

  /**
   * Resets the title to the automatic value based on current state.
   */
  function resetTitle(): void {
    updateTitle()
  }

  // Auto-start if enabled
  if (autoStart) {
    start()
  }

  // Cleanup on unmount
  onUnmounted(() => {
    stop()
  })

  return {
    // State
    currentTitle,

    // Methods
    generateTitle,
    updateTitle,
    setTitle,
    resetTitle,
    start,
    stop,
  }
}

export type UseWindowTitleReturn = ReturnType<typeof useWindowTitle>
