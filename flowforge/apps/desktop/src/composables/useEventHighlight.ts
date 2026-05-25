/**
 * Event Highlight Composable
 *
 * Provides functionality to highlight event nodes on the graph canvas
 * when Trinity fires events. Uses a pulse/flash animation that briefly
 * highlights the corresponding event node.
 */
import { ref, onMounted, onUnmounted, type ShallowRef, type Ref } from 'vue'
import type { LGraph, LGraphCanvas, LGraphNode, NodeId } from '@/litegraph'
import { TRINITY_COLORS } from '@/config/flowforge.config'
import { TRINITY_NEW_EVENTS, type TrinityNewEventsDetail } from '@/stores/trinityStore'

// =============================================================================
// TYPES
// =============================================================================

/**
 * Options for the useEventHighlight composable
 */
export interface UseEventHighlightOptions {
  /** Reference to the LiteGraph graph instance */
  graph: ShallowRef<LGraph | null>
  /** Reference to the LiteGraph canvas instance */
  canvas: ShallowRef<LGraphCanvas | null>
  /** Duration of the highlight animation in milliseconds (default: 1500) */
  highlightDuration?: number
  /** Number of pulses in the animation (default: 3) */
  pulseCount?: number
}

/**
 * Return type for the useEventHighlight composable
 */
export interface UseEventHighlightReturn {
  /** Currently highlighted node IDs */
  highlightedNodeIds: Ref<Set<NodeId>>
  /** Highlight a node by event type name */
  highlightEventNode: (eventTypeName: string) => void
  /** Highlight a node by node ID */
  highlightNodeById: (nodeId: NodeId) => void
  /** Clear all highlights */
  clearHighlights: () => void
  /** Whether the composable is listening for events */
  isListening: Ref<boolean>
  /** Start listening for Trinity events */
  startListening: () => void
  /** Stop listening for Trinity events */
  stopListening: () => void
}

// =============================================================================
// CONSTANTS
// =============================================================================

/** Default duration of flash highlight animation in ms */
const DEFAULT_HIGHLIGHT_DURATION = 1500

/** Default number of pulses */
const DEFAULT_PULSE_COUNT = 3

/** Event node type prefix */
const EVENT_NODE_TYPE = 'trinity/event'

// =============================================================================
// COMPOSABLE
// =============================================================================

/**
 * Creates event highlighting functionality for Trinity events.
 * When Trinity fires an event, the corresponding event node on the canvas
 * will briefly flash/pulse with the event color.
 */
export function useEventHighlight(options: UseEventHighlightOptions): UseEventHighlightReturn {
  const {
    graph,
    canvas,
    highlightDuration = DEFAULT_HIGHLIGHT_DURATION,
    pulseCount = DEFAULT_PULSE_COUNT,
  } = options

  // State
  const highlightedNodeIds = ref<Set<NodeId>>(new Set())
  const isListening = ref(false)

  // Track active animations
  interface AnimationState {
    timerId: ReturnType<typeof setTimeout>
    originalBoxColor: string | undefined
    pulseIntervalId: ReturnType<typeof setInterval> | undefined
  }
  const activeAnimations = new Map<NodeId, AnimationState>()

  // Event listener reference
  let eventHandler: ((event: Event) => void) | null = null

  /**
   * Find event nodes matching an event type name.
   * Searches by node title or className property.
   */
  function findEventNodesByTypeName(eventTypeName: string): LGraphNode[] {
    if (!graph.value) return []

    const nodes = graph.value._nodes || []
    const matchingNodes: LGraphNode[] = []

    for (const node of nodes) {
      // Check if this is an event node
      if (node.type !== EVENT_NODE_TYPE) continue

      // Match by title (exact or partial match)
      if (node.title === eventTypeName || node.title?.includes(eventTypeName)) {
        matchingNodes.push(node)
        continue
      }

      // Match by className property
      const className = node.properties?.['className'] as string | undefined
      if (className === eventTypeName || className?.includes(eventTypeName)) {
        matchingNodes.push(node)
      }
    }

    return matchingNodes
  }

  /**
   * Apply pulse highlight animation to a node.
   */
  function applyHighlightAnimation(node: LGraphNode): void {
    if (!canvas.value) return

    const nodeId = node.id as NodeId

    // Clear any existing animation for this node
    clearNodeAnimation(nodeId)

    // Store original box color
    const originalBoxColor = node.boxcolor

    // Add to highlighted set
    highlightedNodeIds.value = new Set([...highlightedNodeIds.value, nodeId])

    // Calculate pulse timing
    const pulseInterval = highlightDuration / (pulseCount * 2)
    let pulseState = true
    let pulseCounter = 0

    // Create pulse animation
    const pulseIntervalId = setInterval(() => {
      if (!graph.value || !canvas.value) return

      const currentNode = graph.value.getNodeById(nodeId)
      if (!currentNode) {
        clearNodeAnimation(nodeId)
        return
      }

      // Toggle highlight - ensure we restore to original or leave undefined
      if (pulseState) {
        currentNode.boxcolor = TRINITY_COLORS.event.color
      } else if (originalBoxColor !== undefined) {
        currentNode.boxcolor = originalBoxColor
      } else {
        delete currentNode.boxcolor
      }

      pulseState = !pulseState
      pulseCounter++

      // Trigger canvas redraw
      canvas.value?.setDirty(true, true)

      // End animation after all pulses
      if (pulseCounter >= pulseCount * 2) {
        clearNodeAnimation(nodeId)
      }
    }, pulseInterval)

    // Set up cleanup timer
    const timerId = setTimeout(() => {
      clearNodeAnimation(nodeId)
    }, highlightDuration)

    // Store animation state
    activeAnimations.set(nodeId, {
      timerId,
      originalBoxColor,
      pulseIntervalId,
    })

    // Trigger initial canvas redraw
    canvas.value.setDirty(true, true)
  }

  /**
   * Clear animation for a specific node.
   */
  function clearNodeAnimation(nodeId: NodeId): void {
    const animation = activeAnimations.get(nodeId)
    if (!animation) return

    // Clear timers
    clearTimeout(animation.timerId)
    if (animation.pulseIntervalId) {
      clearInterval(animation.pulseIntervalId)
    }

    // Restore original color
    if (graph.value) {
      const node = graph.value.getNodeById(nodeId)
      if (node) {
        if (animation.originalBoxColor !== undefined) {
          node.boxcolor = animation.originalBoxColor
        } else {
          delete node.boxcolor
        }
      }
    }

    // Remove from tracking
    activeAnimations.delete(nodeId)

    // Update highlighted set
    const newHighlighted = new Set(highlightedNodeIds.value)
    newHighlighted.delete(nodeId)
    highlightedNodeIds.value = newHighlighted

    // Trigger canvas redraw
    canvas.value?.setDirty(true, true)
  }

  /**
   * Highlight event nodes matching an event type name.
   */
  function highlightEventNode(eventTypeName: string): void {
    const matchingNodes = findEventNodesByTypeName(eventTypeName)

    for (const node of matchingNodes) {
      applyHighlightAnimation(node)
    }
  }

  /**
   * Highlight a node by its ID.
   */
  function highlightNodeById(nodeId: NodeId): void {
    if (!graph.value) return

    const node = graph.value.getNodeById(nodeId)
    if (node) {
      applyHighlightAnimation(node)
    }
  }

  /**
   * Clear all active highlights.
   */
  function clearHighlights(): void {
    for (const nodeId of activeAnimations.keys()) {
      clearNodeAnimation(nodeId)
    }
  }

  /**
   * Handle new Trinity events.
   */
  function handleNewEvents(event: Event): void {
    const customEvent = event as CustomEvent<TrinityNewEventsDetail>
    const { events } = customEvent.detail

    for (const trinityEvent of events) {
      highlightEventNode(trinityEvent.eventType)
    }
  }

  /**
   * Start listening for Trinity events.
   */
  function startListening(): void {
    if (isListening.value) return

    eventHandler = handleNewEvents
    window.addEventListener(TRINITY_NEW_EVENTS, eventHandler)
    isListening.value = true
  }

  /**
   * Stop listening for Trinity events.
   */
  function stopListening(): void {
    if (!isListening.value || !eventHandler) return

    window.removeEventListener(TRINITY_NEW_EVENTS, eventHandler)
    eventHandler = null
    isListening.value = false
  }

  // Lifecycle hooks
  onMounted(() => {
    startListening()
  })

  onUnmounted(() => {
    stopListening()
    clearHighlights()
  })

  return {
    highlightedNodeIds,
    highlightEventNode,
    highlightNodeById,
    clearHighlights,
    isListening,
    startListening,
    stopListening,
  }
}
