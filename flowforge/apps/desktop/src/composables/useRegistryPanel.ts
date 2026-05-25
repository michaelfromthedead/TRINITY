/**
 * Registry Panel Composable
 *
 * Provides reactive state and methods for the Trinity Registry Panel.
 * Manages registry entries for components, systems, resources, and events,
 * with filtering capabilities and graph node highlighting integration.
 */

import { ref, computed, watch, type Ref, type ShallowRef } from 'vue'
import type { LGraph, LGraphCanvas } from '@/litegraph'
import { useGraphStore } from '@/stores/graphStore'
import type { FilterableTrinityType } from './useTypeFilter'

// =============================================================================
// TYPES
// =============================================================================

/**
 * Registration status of an entry
 */
export type RegistrationStatus = 'registered' | 'pending' | 'error' | 'unknown'

/**
 * A single registry entry representing a Trinity type
 */
export interface RegistryEntry {
  /** Unique identifier */
  id: string
  /** Display name */
  name: string
  /** Type category */
  type: FilterableTrinityType
  /** Module path (e.g., 'game.components.player') */
  modulePath: string
  /** Registration status */
  status: RegistrationStatus
  /** Whether this type exists in parsed AST */
  existsInAST: boolean
  /** Source file path */
  sourceFile?: string
  /** Line number in source */
  sourceLine?: number
  /** Associated graph node ID (if any) */
  nodeId?: string
}

/**
 * Options for useRegistryPanel composable
 */
export interface UseRegistryPanelOptions {
  /** Reference to the LiteGraph graph instance */
  graph?: ShallowRef<LGraph | null>
  /** Reference to the LiteGraph canvas instance */
  canvas?: ShallowRef<LGraphCanvas | null>
}

/**
 * Return type for useRegistryPanel composable
 */
export interface UseRegistryPanelReturn {
  // State
  entries: Ref<RegistryEntry[]>
  isLoading: Ref<boolean>
  error: Ref<string | null>
  searchQuery: Ref<string>
  expandedSections: Ref<Record<FilterableTrinityType, boolean>>
  isConnected: Ref<boolean>

  // Computed
  filteredEntries: Ref<RegistryEntry[]>
  componentEntries: Ref<RegistryEntry[]>
  systemEntries: Ref<RegistryEntry[]>
  resourceEntries: Ref<RegistryEntry[]>
  eventEntries: Ref<RegistryEntry[]>
  entryCounts: Ref<Record<FilterableTrinityType, number>>
  filteredCounts: Ref<Record<FilterableTrinityType, number>>
  isEmpty: Ref<boolean>

  // Methods
  refresh: () => Promise<void>
  setSearchQuery: (query: string) => void
  toggleSection: (type: FilterableTrinityType) => void
  expandAll: () => void
  collapseAll: () => void
  highlightNode: (entry: RegistryEntry) => void
  clearHighlight: () => void
  getEntriesByType: (type: FilterableTrinityType) => RegistryEntry[]
}

// =============================================================================
// STORAGE
// =============================================================================

const STORAGE_KEY = 'flowforge-registry-panel-expanded'

function loadExpandedState(): Record<FilterableTrinityType, boolean> {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      return JSON.parse(stored)
    }
  } catch (error) {
    console.warn('[useRegistryPanel] Failed to load expanded state:', error)
  }
  return {
    component: true,
    system: true,
    resource: true,
    event: true
  }
}

function saveExpandedState(state: Record<FilterableTrinityType, boolean>): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
  } catch (error) {
    console.warn('[useRegistryPanel] Failed to save expanded state:', error)
  }
}

// =============================================================================
// COMPOSABLE
// =============================================================================

/**
 * Registry Panel composable for managing Trinity registry display.
 * Provides filtering, section expansion, and node highlighting functionality.
 */
export function useRegistryPanel(options: UseRegistryPanelOptions = {}): UseRegistryPanelReturn {
  const { graph, canvas } = options

  // Stores
  const graphStore = useGraphStore()

  // State
  const entries = ref<RegistryEntry[]>([])
  const isLoading = ref(false)
  const error = ref<string | null>(null)
  const searchQuery = ref('')
  const expandedSections = ref<Record<FilterableTrinityType, boolean>>(loadExpandedState())
  const isConnected = ref(false)
  const highlightedNodeId = ref<string | null>(null)

  // Persist expanded state
  watch(
    expandedSections,
    (newState) => {
      saveExpandedState(newState)
    },
    { deep: true }
  )

  // =============================================================================
  // COMPUTED PROPERTIES
  // =============================================================================

  /**
   * Filter entries by search query
   */
  const filteredEntries = computed<RegistryEntry[]>(() => {
    if (!searchQuery.value.trim()) {
      return entries.value
    }

    const query = searchQuery.value.toLowerCase()
    return entries.value.filter((entry) => {
      return (
        entry.name.toLowerCase().includes(query) ||
        entry.modulePath.toLowerCase().includes(query) ||
        entry.type.toLowerCase().includes(query)
      )
    })
  })

  /**
   * Get entries filtered by type
   */
  const componentEntries = computed<RegistryEntry[]>(() =>
    filteredEntries.value.filter((e) => e.type === 'component')
  )

  const systemEntries = computed<RegistryEntry[]>(() =>
    filteredEntries.value.filter((e) => e.type === 'system')
  )

  const resourceEntries = computed<RegistryEntry[]>(() =>
    filteredEntries.value.filter((e) => e.type === 'resource')
  )

  const eventEntries = computed<RegistryEntry[]>(() =>
    filteredEntries.value.filter((e) => e.type === 'event')
  )

  /**
   * Total entry counts by type
   */
  const entryCounts = computed<Record<FilterableTrinityType, number>>(() => ({
    component: entries.value.filter((e) => e.type === 'component').length,
    system: entries.value.filter((e) => e.type === 'system').length,
    resource: entries.value.filter((e) => e.type === 'resource').length,
    event: entries.value.filter((e) => e.type === 'event').length
  }))

  /**
   * Filtered entry counts by type
   */
  const filteredCounts = computed<Record<FilterableTrinityType, number>>(() => ({
    component: componentEntries.value.length,
    system: systemEntries.value.length,
    resource: resourceEntries.value.length,
    event: eventEntries.value.length
  }))

  /**
   * Check if registry is empty
   */
  const isEmpty = computed(() => entries.value.length === 0)

  // =============================================================================
  // METHODS
  // =============================================================================

  /**
   * Refresh registry entries from the graph store
   */
  async function refresh(): Promise<void> {
    isLoading.value = true
    error.value = null

    try {
      // Build entries from graph store nodes
      const newEntries: RegistryEntry[] = []

      for (const node of graphStore.nodes) {
        // Map node type to FilterableTrinityType
        const nodeType = node.type.toLowerCase()
        let trinityType: FilterableTrinityType | null = null

        if (nodeType.includes('component')) {
          trinityType = 'component'
        } else if (nodeType.includes('system')) {
          trinityType = 'system'
        } else if (nodeType.includes('resource')) {
          trinityType = 'resource'
        } else if (nodeType.includes('event')) {
          trinityType = 'event'
        }

        if (trinityType) {
          const props = node.properties as Record<string, unknown> | undefined
          const entry: RegistryEntry = {
            id: String(node.id),
            name: node.title || node.type.split('/').pop() || 'Unknown',
            type: trinityType,
            modulePath: (props?.['module'] as string) || node.type,
            status: 'registered',
            existsInAST: true,
            nodeId: String(node.id)
          }
          // Only add optional properties if they exist
          const sourceFile = props?.['sourceFile']
          const sourceLine = props?.['sourceLine']
          if (typeof sourceFile === 'string') entry.sourceFile = sourceFile
          if (typeof sourceLine === 'number') entry.sourceLine = sourceLine
          newEntries.push(entry)
        }
      }

      entries.value = newEntries
      isConnected.value = newEntries.length > 0 || graphStore.nodes.length > 0

    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to refresh registry'
      console.error('[useRegistryPanel] Refresh error:', err)
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Set the search query
   */
  function setSearchQuery(query: string): void {
    searchQuery.value = query
  }

  /**
   * Toggle a section's expanded state
   */
  function toggleSection(type: FilterableTrinityType): void {
    expandedSections.value[type] = !expandedSections.value[type]
  }

  /**
   * Expand all sections
   */
  function expandAll(): void {
    expandedSections.value = {
      component: true,
      system: true,
      resource: true,
      event: true
    }
  }

  /**
   * Collapse all sections
   */
  function collapseAll(): void {
    expandedSections.value = {
      component: false,
      system: false,
      resource: false,
      event: false
    }
  }

  /**
   * Highlight a node in the graph canvas
   */
  function highlightNode(entry: RegistryEntry): void {
    if (!entry.nodeId) return

    // Clear previous highlight
    clearHighlight()

    // Select the node in graph store
    graphStore.selectNode(entry.nodeId, false)

    // If we have canvas reference, center on the node
    if (canvas?.value && graph?.value) {
      const node = graph.value.getNodeById(parseInt(entry.nodeId.replace('node_', ''), 10))
      if (node) {
        canvas.value.centerOnNode(node)
        canvas.value.setDirty(true, true)
      }
    }

    highlightedNodeId.value = entry.nodeId
  }

  /**
   * Clear node highlighting
   */
  function clearHighlight(): void {
    if (highlightedNodeId.value) {
      graphStore.deselectNode(highlightedNodeId.value)
      highlightedNodeId.value = null
    }
  }

  /**
   * Get entries by type
   */
  function getEntriesByType(type: FilterableTrinityType): RegistryEntry[] {
    return filteredEntries.value.filter((e) => e.type === type)
  }

  // Watch graph store for changes
  watch(
    () => graphStore.nodes,
    () => {
      refresh()
    },
    { deep: true }
  )

  // Initial refresh
  refresh()

  return {
    // State
    entries,
    isLoading,
    error,
    searchQuery,
    expandedSections,
    isConnected,

    // Computed
    filteredEntries,
    componentEntries,
    systemEntries,
    resourceEntries,
    eventEntries,
    entryCounts,
    filteredCounts,
    isEmpty,

    // Methods
    refresh,
    setSearchQuery,
    toggleSection,
    expandAll,
    collapseAll,
    highlightNode,
    clearHighlight,
    getEntriesByType
  }
}

export type { FilterableTrinityType }
