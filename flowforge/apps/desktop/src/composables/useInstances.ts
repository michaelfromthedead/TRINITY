/**
 * Instances Composable
 *
 * Provides reactive state and methods for managing Trinity component instances.
 * Integrates with the graph store and supports tree expansion state,
 * instance data formatting, and filtering.
 */

import { ref, computed, reactive } from 'vue'
import { useGraphStore } from '@/stores/graphStore'
import type { FilterableTrinityType } from './useTypeFilter'

// =============================================================================
// TYPES
// =============================================================================

/**
 * A Trinity component instance at runtime.
 */
export interface TrinityInstance {
  /** Unique instance identifier */
  id: string
  /** The component type (name) */
  componentType: string
  /** Trinity node type (component, system, resource, event) */
  trinityType: FilterableTrinityType
  /** Instance data/properties */
  data: Record<string, unknown>
  /** Timestamp when instance was created */
  createdAt: number
  /** Associated node ID in the graph (if any) */
  nodeId?: string
}

/**
 * Grouped instances by Trinity type and component name.
 */
export interface InstanceGroup {
  /** Trinity type (component, system, resource, event) */
  trinityType: FilterableTrinityType
  /** Component type name */
  componentType: string
  /** Instances of this component type */
  instances: TrinityInstance[]
  /** Whether this group is expanded in the tree view */
  isExpanded: boolean
}

/**
 * Tree node for hierarchical display.
 */
export interface InstanceTreeNode {
  /** Trinity type category */
  trinityType: FilterableTrinityType
  /** Groups under this Trinity type */
  groups: InstanceGroup[]
  /** Total instance count for this type */
  totalCount: number
  /** Whether this type category is expanded */
  isExpanded: boolean
}

/**
 * Connection status with Trinity runtime.
 */
export type ConnectionStatus = 'connected' | 'disconnected' | 'connecting'

// =============================================================================
// STATE
// =============================================================================

/** All active instances */
const instances = ref<TrinityInstance[]>([])

/** Connection status with Trinity runtime */
const connectionStatus = ref<ConnectionStatus>('disconnected')

/** Loading state for refresh operations */
const isLoading = ref(false)

/** Error state */
const error = ref<string | null>(null)

/** Search/filter query */
const searchQuery = ref('')

/** Tree expansion state by Trinity type */
const typeExpansion = reactive<Record<FilterableTrinityType, boolean>>({
  component: true,
  system: true,
  resource: true,
  event: true
})

/** Group expansion state by component type */
const groupExpansion = reactive<Record<string, boolean>>({})

/** Instance expansion state by instance ID */
const instanceExpansion = reactive<Record<string, boolean>>({})

// =============================================================================
// COMPOSABLE
// =============================================================================

export function useInstances() {
  const graphStore = useGraphStore()

  // ===========================================================================
  // Computed: Filtered Instances
  // ===========================================================================

  /**
   * Filter instances by search query.
   */
  const filteredInstances = computed<TrinityInstance[]>(() => {
    if (!searchQuery.value.trim()) {
      return instances.value
    }

    const query = searchQuery.value.toLowerCase()
    return instances.value.filter((instance) => {
      return (
        instance.componentType.toLowerCase().includes(query) ||
        instance.id.toLowerCase().includes(query)
      )
    })
  })

  // ===========================================================================
  // Computed: Instance Tree
  // ===========================================================================

  /**
   * Build hierarchical tree structure from instances.
   */
  const instanceTree = computed<InstanceTreeNode[]>(() => {
    const typeOrder: FilterableTrinityType[] = ['component', 'system', 'resource', 'event']
    const tree: InstanceTreeNode[] = []

    for (const trinityType of typeOrder) {
      const typeInstances = filteredInstances.value.filter(
        (inst) => inst.trinityType === trinityType
      )

      if (typeInstances.length === 0) {
        // Still show the type node even if empty (for consistency)
        tree.push({
          trinityType,
          groups: [],
          totalCount: 0,
          isExpanded: typeExpansion[trinityType]
        })
        continue
      }

      // Group by component type
      const groupMap = new Map<string, TrinityInstance[]>()
      for (const inst of typeInstances) {
        const existing = groupMap.get(inst.componentType) || []
        existing.push(inst)
        groupMap.set(inst.componentType, existing)
      }

      // Convert to groups
      const groups: InstanceGroup[] = []
      for (const [componentType, groupInstances] of groupMap) {
        const groupKey = `${trinityType}:${componentType}`
        groups.push({
          trinityType,
          componentType,
          instances: groupInstances,
          isExpanded: groupExpansion[groupKey] ?? false
        })
      }

      // Sort groups alphabetically
      groups.sort((a, b) => a.componentType.localeCompare(b.componentType))

      tree.push({
        trinityType,
        groups,
        totalCount: typeInstances.length,
        isExpanded: typeExpansion[trinityType]
      })
    }

    return tree
  })

  // ===========================================================================
  // Computed: Statistics
  // ===========================================================================

  /**
   * Total instance count.
   */
  const totalCount = computed(() => instances.value.length)

  /**
   * Instance counts by Trinity type.
   */
  const countsByType = computed<Record<FilterableTrinityType, number>>(() => {
    const counts: Record<FilterableTrinityType, number> = {
      component: 0,
      system: 0,
      resource: 0,
      event: 0
    }

    for (const inst of instances.value) {
      counts[inst.trinityType]++
    }

    return counts
  })

  /**
   * Check if there are any instances.
   */
  const hasInstances = computed(() => instances.value.length > 0)

  /**
   * Check if Trinity is connected.
   */
  const isConnected = computed(() => connectionStatus.value === 'connected')

  // ===========================================================================
  // Actions: Tree Expansion
  // ===========================================================================

  /**
   * Toggle expansion of a Trinity type category.
   */
  function toggleTypeExpansion(type: FilterableTrinityType): void {
    typeExpansion[type] = !typeExpansion[type]
  }

  /**
   * Toggle expansion of a component group.
   */
  function toggleGroupExpansion(trinityType: FilterableTrinityType, componentType: string): void {
    const key = `${trinityType}:${componentType}`
    groupExpansion[key] = !groupExpansion[key]
  }

  /**
   * Toggle expansion of an instance's data view.
   */
  function toggleInstanceExpansion(instanceId: string): void {
    instanceExpansion[instanceId] = !instanceExpansion[instanceId]
  }

  /**
   * Check if a group is expanded.
   */
  function isGroupExpanded(trinityType: FilterableTrinityType, componentType: string): boolean {
    const key = `${trinityType}:${componentType}`
    return groupExpansion[key] ?? false
  }

  /**
   * Check if an instance's data is expanded.
   */
  function isInstanceExpanded(instanceId: string): boolean {
    return instanceExpansion[instanceId] ?? false
  }

  /**
   * Expand all groups.
   */
  function expandAll(): void {
    for (const type of Object.keys(typeExpansion) as FilterableTrinityType[]) {
      typeExpansion[type] = true
    }
    for (const inst of instances.value) {
      const key = `${inst.trinityType}:${inst.componentType}`
      groupExpansion[key] = true
    }
  }

  /**
   * Collapse all groups.
   */
  function collapseAll(): void {
    for (const type of Object.keys(typeExpansion) as FilterableTrinityType[]) {
      typeExpansion[type] = false
    }
    for (const key of Object.keys(groupExpansion)) {
      groupExpansion[key] = false
    }
    for (const key of Object.keys(instanceExpansion)) {
      instanceExpansion[key] = false
    }
  }

  // ===========================================================================
  // Actions: Instance Management
  // ===========================================================================

  /**
   * Add a new instance (typically from Trinity runtime events).
   */
  function addInstance(instance: TrinityInstance): void {
    instances.value.push(instance)
  }

  /**
   * Remove an instance by ID.
   */
  function removeInstance(instanceId: string): void {
    const index = instances.value.findIndex((inst) => inst.id === instanceId)
    if (index !== -1) {
      instances.value.splice(index, 1)
      delete instanceExpansion[instanceId]
    }
  }

  /**
   * Update an instance's data.
   */
  function updateInstance(instanceId: string, data: Record<string, unknown>): void {
    const instance = instances.value.find((inst) => inst.id === instanceId)
    if (instance) {
      instance.data = { ...instance.data, ...data }
    }
  }

  /**
   * Clear all instances.
   */
  function clearInstances(): void {
    instances.value = []
    // Clear expansion state
    for (const key of Object.keys(groupExpansion)) {
      delete groupExpansion[key]
    }
    for (const key of Object.keys(instanceExpansion)) {
      delete instanceExpansion[key]
    }
  }

  /**
   * Set all instances (bulk update).
   */
  function setInstances(newInstances: TrinityInstance[]): void {
    instances.value = newInstances
  }

  // ===========================================================================
  // Actions: Refresh & Connection
  // ===========================================================================

  /**
   * Refresh instances from Trinity runtime.
   * In a real implementation, this would call the Tauri backend.
   */
  async function refreshInstances(): Promise<void> {
    if (isLoading.value) return

    isLoading.value = true
    error.value = null

    try {
      // TODO: Implement actual Trinity runtime query
      // For now, we simulate by checking graph nodes
      const mockInstances: TrinityInstance[] = []

      // Get nodes from graph store and create mock instances
      const storeNodes = graphStore.nodes
      for (const node of storeNodes) {
        // Determine Trinity type from node type
        const nodeType = (node.type ?? '').toLowerCase()
        let trinityType: FilterableTrinityType = 'component'

        if (nodeType.includes('system')) {
          trinityType = 'system'
        } else if (nodeType.includes('resource')) {
          trinityType = 'resource'
        } else if (nodeType.includes('event')) {
          trinityType = 'event'
        }

        mockInstances.push({
          id: `inst-${node.id}`,
          componentType: node.title || node.type || 'Unknown',
          trinityType,
          data: node.properties || {},
          createdAt: Date.now(),
          nodeId: node.id
        })
      }

      instances.value = mockInstances
      connectionStatus.value = 'connected'
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to refresh instances'
      error.value = message
      console.error('[useInstances] Refresh failed:', err)
    } finally {
      isLoading.value = false
    }
  }

  /**
   * Set connection status.
   */
  function setConnectionStatus(status: ConnectionStatus): void {
    connectionStatus.value = status
    if (status === 'disconnected') {
      clearInstances()
    }
  }

  // ===========================================================================
  // Actions: Search & Filter
  // ===========================================================================

  /**
   * Set search query for filtering.
   */
  function setSearchQuery(query: string): void {
    searchQuery.value = query
  }

  /**
   * Clear search query.
   */
  function clearSearch(): void {
    searchQuery.value = ''
  }

  // ===========================================================================
  // Actions: Graph Integration
  // ===========================================================================

  /**
   * Find the graph node ID for an instance.
   */
  function getNodeIdForInstance(instanceId: string): string | undefined {
    const instance = instances.value.find((inst) => inst.id === instanceId)
    return instance?.nodeId
  }

  /**
   * Highlight the associated node in the graph.
   */
  function highlightInstanceNode(instanceId: string): void {
    const nodeId = getNodeIdForInstance(instanceId)
    if (nodeId !== undefined) {
      // Select the node in the graph
      graphStore.selectNode(nodeId)
    }
  }

  // ===========================================================================
  // Utilities
  // ===========================================================================

  /**
   * Format instance data for display.
   */
  function formatInstanceData(data: Record<string, unknown>): string {
    try {
      return JSON.stringify(data, null, 2)
    } catch {
      return '{}'
    }
  }

  /**
   * Get a short preview of instance data.
   */
  function getDataPreview(data: Record<string, unknown>, maxLength: number = 50): string {
    const keys = Object.keys(data)
    if (keys.length === 0) return 'Empty'

    const preview = keys.slice(0, 3).join(', ')
    const suffix = keys.length > 3 ? ` +${keys.length - 3} more` : ''
    const text = preview + suffix

    return text.length > maxLength ? text.slice(0, maxLength - 3) + '...' : text
  }

  return {
    // State (readonly)
    instances: computed(() => instances.value),
    connectionStatus: computed(() => connectionStatus.value),
    isLoading: computed(() => isLoading.value),
    error: computed(() => error.value),
    searchQuery: computed(() => searchQuery.value),

    // Computed
    filteredInstances,
    instanceTree,
    totalCount,
    countsByType,
    hasInstances,
    isConnected,

    // Tree expansion
    toggleTypeExpansion,
    toggleGroupExpansion,
    toggleInstanceExpansion,
    isGroupExpanded,
    isInstanceExpanded,
    expandAll,
    collapseAll,

    // Instance management
    addInstance,
    removeInstance,
    updateInstance,
    clearInstances,
    setInstances,

    // Refresh & connection
    refreshInstances,
    setConnectionStatus,

    // Search & filter
    setSearchQuery,
    clearSearch,

    // Graph integration
    getNodeIdForInstance,
    highlightInstanceNode,

    // Utilities
    formatInstanceData,
    getDataPreview
  }
}

export type UseInstancesReturn = ReturnType<typeof useInstances>
