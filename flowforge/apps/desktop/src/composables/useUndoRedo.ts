/**
 * Undo/Redo Composable
 *
 * Provides undo/redo functionality for graph operations.
 * Manages state history with configurable stack size limits.
 */
import { ref, computed, shallowRef } from 'vue'
import { UI_CONFIG } from '@/config/flowforge.config'

/**
 * History entry representing a saved state.
 */
export interface HistoryEntry<T> {
  /** Timestamp when the state was saved */
  timestamp: number
  /** Human-readable description of the change */
  description: string
  /** The saved state */
  state: T
}

/**
 * Options for the undo/redo composable.
 */
export interface UseUndoRedoOptions<T> {
  /** Maximum number of states to keep in history (default: from UI_CONFIG) */
  maxSize?: number
  /** Function to clone the state (default: JSON parse/stringify) */
  cloneState?: (state: T) => T
  /** Callback when state changes */
  onStateChange?: (state: T, description: string) => void
}

/**
 * Creates an undo/redo composable for managing state history.
 *
 * @example
 * ```typescript
 * const { pushState, undo, redo, canUndo, canRedo } = useUndoRedo<GraphState>()
 *
 * // Save current state before making changes
 * pushState(currentState, 'Add node')
 *
 * // Undo the last change
 * if (canUndo.value) {
 *   const previousState = undo()
 *   applyState(previousState)
 * }
 * ```
 */
export function useUndoRedo<T>(options?: UseUndoRedoOptions<T>) {
  const maxSize = options?.maxSize ?? UI_CONFIG.history.maxSize
  const cloneState = options?.cloneState ?? ((state: T) => JSON.parse(JSON.stringify(state)))

  // History stack - using shallowRef for performance with large state objects
  const history = shallowRef<HistoryEntry<T>[]>([])
  const historyIndex = ref(-1)

  // Computed properties
  const canUndo = computed(() => historyIndex.value > 0)
  const canRedo = computed(() => historyIndex.value < history.value.length - 1)

  const currentState = computed<T | null>(() => {
    if (historyIndex.value >= 0 && historyIndex.value < history.value.length) {
      const entry = history.value[historyIndex.value]
      return entry ? entry.state : null
    }
    return null
  })

  const undoDescription = computed<string | null>(() => {
    if (historyIndex.value > 0) {
      const entry = history.value[historyIndex.value]
      return entry ? entry.description : null
    }
    return null
  })

  const redoDescription = computed<string | null>(() => {
    if (historyIndex.value < history.value.length - 1) {
      const entry = history.value[historyIndex.value + 1]
      return entry ? entry.description : null
    }
    return null
  })

  const historyLength = computed(() => history.value.length)

  /**
   * Push a new state onto the undo stack.
   *
   * @param state - The state to save
   * @param description - Human-readable description of the change
   */
  function pushState(state: T, description: string): void {
    // Clone the state to prevent reference issues
    const clonedState = cloneState(state)

    // Remove any redo history (states after current index)
    if (historyIndex.value < history.value.length - 1) {
      history.value = history.value.slice(0, historyIndex.value + 1)
    }

    // Create new entry
    const entry: HistoryEntry<T> = {
      timestamp: Date.now(),
      description,
      state: clonedState,
    }

    // Add to history
    const newHistory = [...history.value, entry]

    // Trim history if it exceeds max size
    if (newHistory.length > maxSize) {
      history.value = newHistory.slice(-maxSize)
    } else {
      history.value = newHistory
    }

    // Update index to point to the new entry
    historyIndex.value = history.value.length - 1

    // Callback
    if (options?.onStateChange) {
      options.onStateChange(clonedState, description)
    }
  }

  /**
   * Undo the last change and return the previous state.
   *
   * @returns The previous state, or null if cannot undo
   */
  function undo(): T | null {
    if (!canUndo.value) {
      return null
    }

    historyIndex.value--
    const entry = history.value[historyIndex.value]

    if (!entry) {
      return null
    }

    if (options?.onStateChange) {
      options.onStateChange(entry.state, `Undo: ${entry.description}`)
    }

    return entry.state
  }

  /**
   * Redo the last undone change and return the next state.
   *
   * @returns The next state, or null if cannot redo
   */
  function redo(): T | null {
    if (!canRedo.value) {
      return null
    }

    historyIndex.value++
    const entry = history.value[historyIndex.value]

    if (!entry) {
      return null
    }

    if (options?.onStateChange) {
      options.onStateChange(entry.state, `Redo: ${entry.description}`)
    }

    return entry.state
  }

  /**
   * Clear all history.
   */
  function clearHistory(): void {
    history.value = []
    historyIndex.value = -1
  }

  /**
   * Get the full history stack (for debugging/display).
   */
  function getHistory(): HistoryEntry<T>[] {
    return [...history.value]
  }

  /**
   * Get the current history index.
   */
  function getHistoryIndex(): number {
    return historyIndex.value
  }

  /**
   * Jump to a specific history index.
   *
   * @param index - The index to jump to
   * @returns The state at that index, or null if invalid
   */
  function jumpTo(index: number): T | null {
    if (index < 0 || index >= history.value.length) {
      return null
    }

    historyIndex.value = index
    const entry = history.value[index]

    if (!entry) {
      return null
    }

    if (options?.onStateChange) {
      options.onStateChange(entry.state, `Jump to: ${entry.description}`)
    }

    return entry.state
  }

  /**
   * Get the last saved state without changing the index.
   */
  function peek(): T | null {
    if (history.value.length === 0) {
      return null
    }
    const lastEntry = history.value[history.value.length - 1]
    return lastEntry ? lastEntry.state : null
  }

  /**
   * Replace the current state without adding a new history entry.
   * Useful for updating the current state without affecting history.
   */
  function replaceCurrentState(state: T, description?: string): void {
    if (historyIndex.value < 0 || historyIndex.value >= history.value.length) {
      // No current state, push instead
      pushState(state, description ?? 'Initial state')
      return
    }

    const clonedState = cloneState(state)
    const newHistory = [...history.value]
    const currentEntry = newHistory[historyIndex.value]
    const entryDescription = description ?? (currentEntry ? currentEntry.description : 'State update')
    newHistory[historyIndex.value] = {
      timestamp: Date.now(),
      description: entryDescription,
      state: clonedState,
    }
    history.value = newHistory
  }

  return {
    // State
    canUndo,
    canRedo,
    currentState,
    undoDescription,
    redoDescription,
    historyLength,

    // Actions
    pushState,
    undo,
    redo,
    clearHistory,
    jumpTo,
    replaceCurrentState,

    // Utilities
    getHistory,
    getHistoryIndex,
    peek,
  }
}

export type UseUndoRedoReturn<T> = ReturnType<typeof useUndoRedo<T>>
