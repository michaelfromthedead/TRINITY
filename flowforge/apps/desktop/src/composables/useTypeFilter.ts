/**
 * Type Filter Composable
 *
 * Provides reactive state and methods for filtering Trinity node types.
 * Supports toggling visibility of component, system, resource, and event nodes.
 * Persists filter state to localStorage.
 */

import { reactive, watch, computed } from 'vue'

/**
 * Trinity node types that can be filtered.
 * This is a subset of the full TrinityNodeType which includes 'neutral'.
 */
export type FilterableTrinityType = 'component' | 'system' | 'resource' | 'event'

/** Storage key for persisting filter state */
const STORAGE_KEY = 'flowforge-type-filter'

/** Default visibility state (all types visible) */
const DEFAULT_STATE: Record<FilterableTrinityType, boolean> = {
  component: true,
  system: true,
  resource: true,
  event: true
}

/** Type-safe list of all filterable Trinity node types */
export const TRINITY_TYPES: FilterableTrinityType[] = ['component', 'system', 'resource', 'event']

/**
 * Load filter state from localStorage
 */
function loadFromStorage(): Record<FilterableTrinityType, boolean> {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      const parsed = JSON.parse(stored) as Record<string, boolean>
      // Validate and merge with defaults to handle new types
      return {
        component: parsed['component'] ?? DEFAULT_STATE.component,
        system: parsed['system'] ?? DEFAULT_STATE.system,
        resource: parsed['resource'] ?? DEFAULT_STATE.resource,
        event: parsed['event'] ?? DEFAULT_STATE.event
      }
    }
  } catch (error) {
    console.warn('[useTypeFilter] Failed to load from storage:', error)
  }
  return { ...DEFAULT_STATE }
}

/**
 * Save filter state to localStorage
 */
function saveToStorage(state: Record<FilterableTrinityType, boolean>): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
  } catch (error) {
    console.warn('[useTypeFilter] Failed to save to storage:', error)
  }
}

// Singleton reactive state
const visibleTypes = reactive<Record<FilterableTrinityType, boolean>>(loadFromStorage())

// Watch for changes and persist
watch(
  () => ({ ...visibleTypes }),
  (newState) => {
    saveToStorage(newState)
  },
  { deep: true }
)

/**
 * Type filter composable for managing Trinity node type visibility.
 * Uses a singleton pattern so all components share the same state.
 */
export function useTypeFilter() {
  /**
   * Toggle visibility of a specific type
   */
  function toggleType(type: FilterableTrinityType): void {
    visibleTypes[type] = !visibleTypes[type]
  }

  /**
   * Set visibility of a specific type
   */
  function setTypeVisibility(type: FilterableTrinityType, visible: boolean): void {
    visibleTypes[type] = visible
  }

  /**
   * Show all node types
   */
  function showAll(): void {
    for (const type of TRINITY_TYPES) {
      visibleTypes[type] = true
    }
  }

  /**
   * Hide all node types
   */
  function hideAll(): void {
    for (const type of TRINITY_TYPES) {
      visibleTypes[type] = false
    }
  }

  /**
   * Check if a type is currently visible
   */
  function isVisible(type: FilterableTrinityType): boolean {
    return visibleTypes[type] ?? true
  }

  /**
   * Get an array of currently visible types
   */
  const activeTypes = computed<FilterableTrinityType[]>(() =>
    TRINITY_TYPES.filter((type) => visibleTypes[type])
  )

  /**
   * Check if all types are visible
   */
  const allVisible = computed(() =>
    TRINITY_TYPES.every((type) => visibleTypes[type])
  )

  /**
   * Check if no types are visible
   */
  const noneVisible = computed(() =>
    TRINITY_TYPES.every((type) => !visibleTypes[type])
  )

  /**
   * Get the count of visible types
   */
  const visibleCount = computed(() =>
    TRINITY_TYPES.filter((type) => visibleTypes[type]).length
  )

  /**
   * Reset to default state (all visible)
   */
  function reset(): void {
    showAll()
  }

  return {
    // State
    visibleTypes,

    // Methods
    toggleType,
    setTypeVisibility,
    showAll,
    hideAll,
    isVisible,
    reset,

    // Computed
    activeTypes,
    allVisible,
    noneVisible,
    visibleCount
  }
}

export type UseTypeFilterReturn = ReturnType<typeof useTypeFilter>
