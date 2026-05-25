/**
 * Node Search Composable
 *
 * Provides search functionality for nodes in the graph canvas.
 * Supports filtering by name and type with highlight capabilities.
 */
import { ref, shallowRef, type Ref, type ShallowRef, watch } from 'vue'
import type { LGraph, LGraphCanvas, LGraphNode, NodeId } from '@/litegraph'
import type { TrinityNodeType } from '@/litegraph/nodes/TrinityNodes'
import { UI_CONFIG } from '@/config/flowforge.config'

export type NodeTypeFilter = TrinityNodeType | 'all'

export interface SearchResult {
  id: NodeId
  title: string
  type: string
  trinityType: NodeTypeFilter
  pos: [number, number]
  node: LGraphNode
}

export interface UseNodeSearchOptions {
  /** Reference to the LiteGraph graph instance */
  graph: ShallowRef<LGraph | null>
  /** Reference to the LiteGraph canvas instance */
  canvas: ShallowRef<LGraphCanvas | null>
}

export interface UseNodeSearchReturn {
  /** Current search query */
  query: Ref<string>
  /** Current type filter */
  typeFilter: Ref<NodeTypeFilter>
  /** Search results */
  results: ShallowRef<SearchResult[]>
  /** Whether the search is active */
  isActive: Ref<boolean>
  /** Currently selected result index */
  selectedIndex: Ref<number>
  /** Currently highlighted node IDs */
  highlightedNodeIds: Ref<Set<NodeId>>
  /** Perform search with query and optional type filter */
  search: (searchQuery: string, type?: NodeTypeFilter) => void
  /** Select a node by ID and center view on it */
  selectNode: (nodeId: NodeId) => void
  /** Select next result in the list */
  selectNextResult: () => void
  /** Select previous result in the list */
  selectPreviousResult: () => void
  /** Confirm current selection (center view on selected result) */
  confirmSelection: () => void
  /** Highlight all matching nodes */
  highlightMatches: () => void
  /** Clear search, highlights, and reset state */
  clearSearch: () => void
  /** Set active state */
  setActive: (active: boolean) => void
}

/**
 * Extract Trinity type from a node.
 */
function getTrinityType(node: LGraphNode): NodeTypeFilter {
  // Check for Trinity node type
  const nodeType = node.type?.toLowerCase() || ''
  if (nodeType.includes('component')) return 'component'
  if (nodeType.includes('system')) return 'system'
  if (nodeType.includes('resource')) return 'resource'
  if (nodeType.includes('event')) return 'event'

  // Check properties using index signature access
  const props = node.properties as Record<string, unknown> | undefined
  const trinityType = props?.['trinityType']
  if (typeof trinityType === 'string') {
    if (['component', 'system', 'resource', 'event'].includes(trinityType)) {
      return trinityType as TrinityNodeType
    }
  }

  return 'all'
}

/**
 * Creates node search functionality for the graph canvas.
 */
export function useNodeSearch(options: UseNodeSearchOptions): UseNodeSearchReturn {
  const { graph, canvas } = options

  // State
  const query = ref('')
  const typeFilter = ref<NodeTypeFilter>('all')
  const results = shallowRef<SearchResult[]>([])
  const isActive = ref(false)
  const selectedIndex = ref(0)
  const highlightedNodeIds = ref<Set<NodeId>>(new Set())

  // Store original node colors for restoration
  const originalNodeColors = new Map<NodeId, { color: string | undefined; boxcolor: string | undefined }>()

  /**
   * Get all nodes from the graph.
   */
  function getAllNodes(): LGraphNode[] {
    if (!graph.value) return []
    return graph.value.nodes || []
  }

  /**
   * Perform search with query and type filter.
   */
  function search(searchQuery: string, type?: NodeTypeFilter): void {
    query.value = searchQuery
    if (type !== undefined) {
      typeFilter.value = type
    }

    const nodes = getAllNodes()
    const normalizedQuery = searchQuery.toLowerCase().trim()

    if (!normalizedQuery && typeFilter.value === 'all') {
      results.value = []
      selectedIndex.value = 0
      clearHighlights()
      return
    }

    const matchingResults: SearchResult[] = []

    for (const node of nodes) {
      const title = (node.title || '').toLowerCase()
      const nodeType = (node.type || '').toLowerCase()
      const trinityType = getTrinityType(node)

      // Type filter
      if (typeFilter.value !== 'all' && trinityType !== typeFilter.value) {
        continue
      }

      // Query filter (search in title and type)
      if (normalizedQuery) {
        const matchesTitle = title.includes(normalizedQuery)
        const matchesType = nodeType.includes(normalizedQuery)
        if (!matchesTitle && !matchesType) {
          continue
        }
      }

      matchingResults.push({
        id: node.id,
        title: node.title || 'Untitled',
        type: node.type || 'unknown',
        trinityType,
        pos: [node.pos[0], node.pos[1]],
        node
      })
    }

    // Sort results by title
    matchingResults.sort((a, b) => a.title.localeCompare(b.title))

    results.value = matchingResults
    selectedIndex.value = 0

    // Update highlights
    highlightMatches()
  }

  /**
   * Select a node by ID and center the canvas view on it.
   */
  function selectNode(nodeId: NodeId): void {
    if (!graph.value || !canvas.value) return

    const node = graph.value.getNodeById(nodeId)
    if (!node) return

    // Select the node in LiteGraph
    canvas.value.selectNode(node, false)

    // Center view on the node
    centerOnNode(node)

    // Clear search state
    isActive.value = false
  }

  /**
   * Center the canvas view on a node.
   */
  function centerOnNode(node: LGraphNode): void {
    if (!canvas.value) return

    const canvasWidth = canvas.value.canvas.width
    const canvasHeight = canvas.value.canvas.height
    const nodeWidth = node.size[0]
    const nodeHeight = node.size[1]

    // Calculate center position of the node
    const nodeCenterX = node.pos[0] + nodeWidth / 2
    const nodeCenterY = node.pos[1] + nodeHeight / 2

    // Set canvas offset to center the node
    canvas.value.ds.offset = [
      canvasWidth / 2 - nodeCenterX * canvas.value.ds.scale,
      canvasHeight / 2 - nodeCenterY * canvas.value.ds.scale
    ]

    // Redraw
    canvas.value.setDirty(true, true)
  }

  /**
   * Select next result in the list.
   */
  function selectNextResult(): void {
    if (results.value.length === 0) return
    selectedIndex.value = (selectedIndex.value + 1) % results.value.length
  }

  /**
   * Select previous result in the list.
   */
  function selectPreviousResult(): void {
    if (results.value.length === 0) return
    selectedIndex.value = selectedIndex.value <= 0
      ? results.value.length - 1
      : selectedIndex.value - 1
  }

  /**
   * Confirm current selection (select and center on the currently selected result).
   */
  function confirmSelection(): void {
    if (results.value.length === 0 || selectedIndex.value < 0) return
    const result = results.value[selectedIndex.value]
    if (result) {
      selectNode(result.id)
    }
  }

  /**
   * Highlight all matching nodes with a visual effect.
   */
  function highlightMatches(): void {
    // Clear previous highlights
    clearHighlights()

    if (results.value.length === 0) return

    const newHighlightedIds = new Set<NodeId>()

    for (const result of results.value) {
      const node = result.node
      const nodeId = node.id

      // Store original colors
      if (!originalNodeColors.has(nodeId)) {
        originalNodeColors.set(nodeId, {
          color: node.color,
          boxcolor: node.boxcolor
        })
      }

      // Apply highlight effect - brighten the box color
      node.boxcolor = UI_CONFIG.search.highlightColor

      newHighlightedIds.add(nodeId)
    }

    highlightedNodeIds.value = newHighlightedIds

    // Trigger canvas redraw
    if (canvas.value) {
      canvas.value.setDirty(true, true)
    }
  }

  /**
   * Clear all highlights and restore original node colors.
   */
  function clearHighlights(): void {
    for (const [nodeId, colors] of originalNodeColors.entries()) {
      if (!graph.value) continue
      const node = graph.value.getNodeById(nodeId)
      if (node) {
        if (colors.color !== undefined) {
          node.color = colors.color
        }
        if (colors.boxcolor !== undefined) {
          node.boxcolor = colors.boxcolor
        }
      }
    }

    originalNodeColors.clear()
    highlightedNodeIds.value = new Set()

    // Trigger canvas redraw
    if (canvas.value) {
      canvas.value.setDirty(true, true)
    }
  }

  /**
   * Clear search state, results, and highlights.
   */
  function clearSearch(): void {
    query.value = ''
    typeFilter.value = 'all'
    results.value = []
    selectedIndex.value = 0
    isActive.value = false
    clearHighlights()
  }

  /**
   * Set active state of the search.
   */
  function setActive(active: boolean): void {
    isActive.value = active
    if (!active) {
      clearHighlights()
    }
  }

  // Watch for typeFilter changes and re-search
  watch(typeFilter, () => {
    if (query.value || typeFilter.value !== 'all') {
      search(query.value)
    }
  })

  return {
    query,
    typeFilter,
    results,
    isActive,
    selectedIndex,
    highlightedNodeIds,
    search,
    selectNode,
    selectNextResult,
    selectPreviousResult,
    confirmSelection,
    highlightMatches,
    clearSearch,
    setActive
  }
}
