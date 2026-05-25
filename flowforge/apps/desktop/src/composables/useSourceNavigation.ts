/**
 * Source Navigation Composable
 *
 * Provides source file navigation capabilities for FlowForge.
 * When a Trinity node is clicked, this composable handles navigation
 * events to highlight and navigate to source code locations.
 *
 * @module composables/useSourceNavigation
 */

import { ref, readonly, onMounted, onUnmounted } from 'vue'

// =============================================================================
// TYPES
// =============================================================================

/**
 * Represents a source code location.
 */
export interface SourceLocation {
  /** The source file path (relative or absolute) */
  file: string
  /** The line number in the source file (1-indexed) */
  line: number
}

/**
 * Event detail structure for navigation events.
 */
export interface NavigateToSourceEventDetail {
  file: string
  line: number
}

/**
 * Options for the useSourceNavigation composable.
 */
export interface SourceNavigationOptions {
  /** Whether to automatically start listening for events (default: true) */
  autoStart?: boolean
  /** Callback when navigation is requested */
  onNavigate?: (location: SourceLocation) => void
  /** Callback when highlight is cleared */
  onClear?: () => void
}

/**
 * Return type for the useSourceNavigation composable.
 */
export interface UseSourceNavigationReturn {
  /** Currently highlighted source location (null if none) */
  readonly currentSource: Readonly<typeof currentSource>
  /** Whether a source is currently highlighted */
  readonly hasHighlight: Readonly<typeof hasHighlight>
  /** Navigate to a source location */
  navigateToSource: (file: string, line: number) => void
  /** Clear the current highlight */
  clearHighlight: () => void
  /** Copy the current source path to clipboard */
  copyToClipboard: () => Promise<boolean>
  /** Format the source location as a string */
  formatLocation: () => string
  /** Start listening for navigation events */
  start: () => void
  /** Stop listening for navigation events */
  stop: () => void
  /** Emit an external navigation event (for IDE integration) */
  emitExternalNavigation: (location: SourceLocation) => void
}

// =============================================================================
// EVENT NAMES
// =============================================================================

/** Event name for internal navigation requests (from nodes) */
export const NAVIGATE_TO_SOURCE_EVENT = 'flowforge:navigate-to-source'

/** Event name for external navigation (for IDE/editor integration) */
export const EXTERNAL_NAVIGATION_EVENT = 'flowforge:external-navigate'

/** Event name for highlight clear requests */
export const CLEAR_HIGHLIGHT_EVENT = 'flowforge:clear-source-highlight'

// =============================================================================
// COMPOSABLE STATE
// =============================================================================

// Shared reactive state (singleton pattern for global access)
const currentSource = ref<SourceLocation | null>(null)
const hasHighlight = ref(false)

// =============================================================================
// COMPOSABLE
// =============================================================================

/**
 * Composable for handling source code navigation in FlowForge.
 *
 * Listens for `flowforge:navigate-to-source` CustomEvents dispatched by
 * Trinity nodes when they are clicked, and maintains the current
 * highlighted source location.
 *
 * @example
 * ```typescript
 * // In a Vue component
 * const {
 *   currentSource,
 *   hasHighlight,
 *   navigateToSource,
 *   clearHighlight,
 *   copyToClipboard
 * } = useSourceNavigation({
 *   onNavigate: (location) => {
 *     console.log('Navigate to:', location.file, location.line)
 *   }
 * })
 *
 * // Manual navigation
 * navigateToSource('components.py', 42)
 *
 * // Clear highlight
 * clearHighlight()
 * ```
 */
export function useSourceNavigation(
  options: SourceNavigationOptions = {}
): UseSourceNavigationReturn {
  const { autoStart = true, onNavigate, onClear } = options

  let isListening = false

  // ---------------------------------------------------------------------------
  // EVENT HANDLERS
  // ---------------------------------------------------------------------------

  /**
   * Handle navigation events from nodes.
   */
  function handleNavigateEvent(event: Event): void {
    const customEvent = event as CustomEvent<NavigateToSourceEventDetail>
    const { file, line } = customEvent.detail || {}

    if (file && typeof line === 'number') {
      navigateToSource(file, line)
    }
  }

  /**
   * Handle clear highlight events.
   */
  function handleClearEvent(): void {
    clearHighlight()
  }

  // ---------------------------------------------------------------------------
  // PUBLIC METHODS
  // ---------------------------------------------------------------------------

  /**
   * Navigate to a source location and update the highlight.
   *
   * @param file - The source file path
   * @param line - The line number (1-indexed)
   */
  function navigateToSource(file: string, line: number): void {
    const location: SourceLocation = { file, line }

    currentSource.value = location
    hasHighlight.value = true

    // Call the optional callback
    if (onNavigate) {
      onNavigate(location)
    }

    // Log for debugging
    console.log('[SourceNavigation] Navigate to:', formatLocation())
  }

  /**
   * Clear the current source highlight.
   */
  function clearHighlight(): void {
    currentSource.value = null
    hasHighlight.value = false

    // Call the optional callback
    if (onClear) {
      onClear()
    }

    console.log('[SourceNavigation] Highlight cleared')
  }

  /**
   * Copy the current source location to clipboard.
   *
   * @returns Promise resolving to true if successful, false otherwise
   */
  async function copyToClipboard(): Promise<boolean> {
    if (!currentSource.value) {
      return false
    }

    const text = formatLocation()

    try {
      await navigator.clipboard.writeText(text)
      console.log('[SourceNavigation] Copied to clipboard:', text)
      return true
    } catch (error) {
      console.error('[SourceNavigation] Failed to copy to clipboard:', error)
      return false
    }
  }

  /**
   * Format the current source location as a string.
   *
   * @returns Formatted string like "file.py:42" or empty string if no highlight
   */
  function formatLocation(): string {
    if (!currentSource.value) {
      return ''
    }

    return `${currentSource.value.file}:${currentSource.value.line}`
  }

  /**
   * Emit an external navigation event for IDE/editor integration.
   *
   * External tools can listen for `flowforge:external-navigate` events
   * to open files in their respective editors.
   *
   * @param location - The source location to navigate to
   */
  function emitExternalNavigation(location: SourceLocation): void {
    const event = new CustomEvent(EXTERNAL_NAVIGATION_EVENT, {
      detail: location,
      bubbles: true,
    })
    window.dispatchEvent(event)

    console.log('[SourceNavigation] External navigation event emitted:', location)
  }

  /**
   * Start listening for navigation events.
   */
  function start(): void {
    if (isListening) {
      return
    }

    window.addEventListener(NAVIGATE_TO_SOURCE_EVENT, handleNavigateEvent)
    window.addEventListener(CLEAR_HIGHLIGHT_EVENT, handleClearEvent)
    isListening = true

    console.log('[SourceNavigation] Started listening for events')
  }

  /**
   * Stop listening for navigation events.
   */
  function stop(): void {
    if (!isListening) {
      return
    }

    window.removeEventListener(NAVIGATE_TO_SOURCE_EVENT, handleNavigateEvent)
    window.removeEventListener(CLEAR_HIGHLIGHT_EVENT, handleClearEvent)
    isListening = false

    console.log('[SourceNavigation] Stopped listening for events')
  }

  // ---------------------------------------------------------------------------
  // LIFECYCLE
  // ---------------------------------------------------------------------------

  onMounted(() => {
    if (autoStart) {
      start()
    }
  })

  onUnmounted(() => {
    stop()
  })

  // ---------------------------------------------------------------------------
  // RETURN
  // ---------------------------------------------------------------------------

  return {
    currentSource: readonly(currentSource),
    hasHighlight: readonly(hasHighlight),
    navigateToSource,
    clearHighlight,
    copyToClipboard,
    formatLocation,
    start,
    stop,
    emitExternalNavigation,
  }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Dispatch a navigation event from anywhere in the application.
 *
 * This is useful for triggering navigation from non-Vue code
 * (like LiteGraph node click handlers).
 *
 * @param file - The source file path
 * @param line - The line number
 */
export function dispatchNavigateToSource(file: string, line: number): void {
  const event = new CustomEvent<NavigateToSourceEventDetail>(
    NAVIGATE_TO_SOURCE_EVENT,
    {
      detail: { file, line },
      bubbles: true,
    }
  )
  window.dispatchEvent(event)
}

/**
 * Dispatch a clear highlight event.
 */
export function dispatchClearHighlight(): void {
  const event = new CustomEvent(CLEAR_HIGHLIGHT_EVENT, { bubbles: true })
  window.dispatchEvent(event)
}

// Default export
export default useSourceNavigation
