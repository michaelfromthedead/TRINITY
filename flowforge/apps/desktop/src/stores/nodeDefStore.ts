/**
 * Node Definition Store - Trinity node type definitions management
 * Manages the loading and access of Trinity ECS node type definitions.
 */
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { getApi } from '@/services'
import type { TrinityNodeTypes, NodeTypeDefinition } from '@/services'

export const useNodeDefStore = defineStore('nodeDef', () => {
  // ===========================================================================
  // State
  // ===========================================================================

  /**
   * Trinity node type definitions (component, system, resource, event).
   */
  const nodeTypes = ref<TrinityNodeTypes | null>(null)

  /**
   * Loading state indicator.
   */
  const loading = ref(false)

  /**
   * Error message if loading failed.
   */
  const error = ref<string | null>(null)

  // ===========================================================================
  // Getters
  // ===========================================================================

  /**
   * Get the component node type definition.
   */
  const componentDef = computed<NodeTypeDefinition | null>(() => {
    return nodeTypes.value?.component ?? null
  })

  /**
   * Get the system node type definition.
   */
  const systemDef = computed<NodeTypeDefinition | null>(() => {
    return nodeTypes.value?.system ?? null
  })

  /**
   * Get the resource node type definition.
   */
  const resourceDef = computed<NodeTypeDefinition | null>(() => {
    return nodeTypes.value?.resource ?? null
  })

  /**
   * Get the event node type definition.
   */
  const eventDef = computed<NodeTypeDefinition | null>(() => {
    return nodeTypes.value?.event ?? null
  })

  /**
   * Check if node types have been loaded.
   */
  const isLoaded = computed(() => nodeTypes.value !== null)

  /**
   * Get all available node type names.
   */
  const availableTypes = computed<string[]>(() => {
    if (!nodeTypes.value) return []
    return Object.keys(nodeTypes.value)
  })

  // ===========================================================================
  // Actions
  // ===========================================================================

  /**
   * Load Trinity node type definitions from the API.
   * Sets loading state and handles errors.
   */
  async function loadNodeTypes(): Promise<void> {
    if (loading.value) return // Prevent concurrent loads

    loading.value = true
    error.value = null

    try {
      const api = getApi()

      // Check if the API supports Trinity node types
      if (!api.getTrinityNodeTypes) {
        throw new Error('API does not support getTrinityNodeTypes')
      }

      nodeTypes.value = await api.getTrinityNodeTypes()
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load node types'
      error.value = message
      console.error('[NodeDefStore] Failed to load node types:', err)
    } finally {
      loading.value = false
    }
  }

  /**
   * Get a specific node type definition by name.
   */
  function getNodeDef(typeName: string): NodeTypeDefinition | null {
    if (!nodeTypes.value) return null

    switch (typeName) {
      case 'component':
        return nodeTypes.value.component
      case 'system':
        return nodeTypes.value.system
      case 'resource':
        return nodeTypes.value.resource
      case 'event':
        return nodeTypes.value.event
      default:
        return null
    }
  }

  /**
   * Clear loaded node types and reset state.
   */
  function clearNodeTypes(): void {
    nodeTypes.value = null
    error.value = null
  }

  /**
   * Reload node types (clear and load fresh).
   */
  async function reloadNodeTypes(): Promise<void> {
    clearNodeTypes()
    await loadNodeTypes()
  }

  return {
    // State
    nodeTypes,
    loading,
    error,

    // Getters
    componentDef,
    systemDef,
    resourceDef,
    eventDef,
    isLoaded,
    availableTypes,

    // Actions
    loadNodeTypes,
    getNodeDef,
    clearNodeTypes,
    reloadNodeTypes
  }
})
