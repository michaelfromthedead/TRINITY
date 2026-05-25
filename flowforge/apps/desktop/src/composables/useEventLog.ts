/**
 * Event Log Composable
 *
 * Provides event logging and filtering functionality for Trinity events.
 * Integrates with the graph store to track and display events in real-time.
 * Supports filtering by event type and highlighting event nodes in the graph.
 */
import { ref, computed, shallowRef, watch, type Ref, type ShallowRef, type ComputedRef } from 'vue'
import type { LGraph, LGraphCanvas, LGraphNode, NodeId } from '@/litegraph'
import { TRINITY_COLORS } from '@/nodes/nodeTheme'
import type { FilterableTrinityType } from '@/composables/useTypeFilter'
import { UI_CONFIG } from '@/config/flowforge.config'

/**
 * Represents a single event log entry
 */
export interface EventLogEntry {
  /** Unique identifier for this log entry */
  id: string
  /** Event type name (e.g., 'CollisionEvent', 'InputEvent') */
  eventType: string
  /** Trinity node type classification */
  trinityType: FilterableTrinityType
  /** Timestamp when the event was logged */
  timestamp: number
  /** Event payload data */
  payload: Record<string, unknown>
  /** Optional node ID that fired this event */
  nodeId?: NodeId
  /** Source file and line if available */
  source?: {
    file: string
    line: number
  }
}

/**
 * Filter configuration for event log
 */
export interface EventLogFilter {
  /** Filter by event type name (partial match) */
  eventTypeQuery: string
  /** Filter by Trinity node types */
  trinityTypes: Set<FilterableTrinityType>
}

/**
 * Options for the useEventLog composable
 */
export interface UseEventLogOptions {
  /** Reference to the LiteGraph graph instance */
  graph?: ShallowRef<LGraph | null>
  /** Reference to the LiteGraph canvas instance */
  canvas?: ShallowRef<LGraphCanvas | null>
  /** Maximum number of events to keep in memory */
  maxEvents?: number
  /** Polling interval in milliseconds when active */
  pollInterval?: number
}

/**
 * Return type for the useEventLog composable
 */
export interface UseEventLogReturn {
  /** All logged events (newest first) */
  events: ShallowRef<EventLogEntry[]>
  /** Filtered events based on current filter settings */
  filteredEvents: ComputedRef<EventLogEntry[]>
  /** Whether the event log is currently active/polling */
  isActive: Ref<boolean>
  /** Whether auto-scroll is enabled */
  autoScroll: Ref<boolean>
  /** Whether logging is paused */
  isPaused: Ref<boolean>
  /** Current filter settings */
  filter: Ref<EventLogFilter>
  /** Currently expanded event ID */
  expandedEventId: Ref<string | null>
  /** Currently highlighted node IDs (flash animation) */
  highlightedNodeIds: Ref<Set<NodeId>>
  /** Count of new events since last view */
  newEventCount: Ref<number>
  /** Add a new event to the log */
  addEvent: (event: Omit<EventLogEntry, 'id' | 'timestamp'>) => void
  /** Clear all events */
  clearEvents: () => void
  /** Toggle pause state */
  togglePause: () => void
  /** Toggle auto-scroll */
  toggleAutoScroll: () => void
  /** Set filter event type query */
  setEventTypeFilter: (query: string) => void
  /** Toggle a Trinity type in the filter */
  toggleTrinityTypeFilter: (type: FilterableTrinityType) => void
  /** Reset all filters */
  resetFilters: () => void
  /** Expand/collapse an event to show full payload */
  toggleEventExpanded: (eventId: string) => void
  /** Highlight the node associated with an event */
  highlightEventNode: (eventId: string) => void
  /** Flash highlight a node briefly */
  flashHighlightNode: (nodeId: NodeId) => void
  /** Start polling for events */
  startPolling: () => void
  /** Stop polling for events */
  stopPolling: () => void
  /** Mark events as viewed (reset new event count) */
  markEventsViewed: () => void
  /** Set the panel as active (visible) */
  setActive: (active: boolean) => void
}

/** Storage key for persisting event log settings */
const STORAGE_KEY = 'flowforge-eventlog-settings'

/** Default maximum events to keep */
const DEFAULT_MAX_EVENTS = 100

/** Default poll interval in ms */
const DEFAULT_POLL_INTERVAL = UI_CONFIG.eventLog.pollInterval

/** Duration of flash highlight animation in ms */
const FLASH_HIGHLIGHT_DURATION = 800

/** Counter for generating unique event IDs */
let eventIdCounter = 0

/**
 * Generate a unique event ID
 */
function generateEventId(): string {
  eventIdCounter++
  return `evt_${Date.now()}_${eventIdCounter}`
}

/**
 * Load settings from localStorage
 */
function loadSettings(): { autoScroll: boolean } {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) {
      const parsed = JSON.parse(stored)
      return {
        autoScroll: parsed.autoScroll ?? true
      }
    }
  } catch (error) {
    console.warn('[useEventLog] Failed to load settings:', error)
  }
  return { autoScroll: true }
}

/**
 * Save settings to localStorage
 */
function saveSettings(settings: { autoScroll: boolean }): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
  } catch (error) {
    console.warn('[useEventLog] Failed to save settings:', error)
  }
}

/**
 * Creates event logging functionality for Trinity events.
 * Provides filtering, highlighting, and real-time updates.
 */
export function useEventLog(options: UseEventLogOptions = {}): UseEventLogReturn {
  const {
    graph,
    canvas,
    maxEvents = DEFAULT_MAX_EVENTS,
    pollInterval = DEFAULT_POLL_INTERVAL
  } = options

  // Load persisted settings
  const savedSettings = loadSettings()

  // State
  const events = shallowRef<EventLogEntry[]>([])
  const isActive = ref(false)
  const autoScroll = ref(savedSettings.autoScroll)
  const isPaused = ref(false)
  const expandedEventId = ref<string | null>(null)
  const highlightedNodeIds = ref<Set<NodeId>>(new Set())
  const newEventCount = ref(0)

  // Filter state
  const filter = ref<EventLogFilter>({
    eventTypeQuery: '',
    trinityTypes: new Set(['component', 'system', 'resource', 'event'])
  })

  // Polling timer
  let pollTimer: ReturnType<typeof setInterval> | null = null

  // Flash highlight timers
  const flashTimers = new Map<NodeId, ReturnType<typeof setTimeout>>()

  // Original node colors for restoration
  const originalNodeColors = new Map<NodeId, { color?: string; boxcolor?: string }>()

  /**
   * Computed filtered events
   */
  const filteredEvents = computed<EventLogEntry[]>(() => {
    const query = filter.value.eventTypeQuery.toLowerCase().trim()
    const types = filter.value.trinityTypes

    return events.value.filter((event) => {
      // Filter by Trinity type
      if (types.size > 0 && !types.has(event.trinityType)) {
        return false
      }

      // Filter by event type query
      if (query && !event.eventType.toLowerCase().includes(query)) {
        return false
      }

      return true
    })
  })

  /**
   * Add a new event to the log
   */
  function addEvent(eventData: Omit<EventLogEntry, 'id' | 'timestamp'>): void {
    if (isPaused.value) return

    const newEvent: EventLogEntry = {
      ...eventData,
      id: generateEventId(),
      timestamp: Date.now()
    }

    // Add to beginning (newest first)
    const updatedEvents = [newEvent, ...events.value]

    // Trim to max size
    if (updatedEvents.length > maxEvents) {
      updatedEvents.splice(maxEvents)
    }

    events.value = updatedEvents

    // Increment new event count if not active
    if (!isActive.value) {
      newEventCount.value++
    }

    // Flash highlight the source node if available
    if (newEvent.nodeId) {
      flashHighlightNode(newEvent.nodeId)
    }
  }

  /**
   * Clear all events
   */
  function clearEvents(): void {
    events.value = []
    expandedEventId.value = null
    newEventCount.value = 0
  }

  /**
   * Toggle pause state
   */
  function togglePause(): void {
    isPaused.value = !isPaused.value
  }

  /**
   * Toggle auto-scroll
   */
  function toggleAutoScroll(): void {
    autoScroll.value = !autoScroll.value
    saveSettings({ autoScroll: autoScroll.value })
  }

  /**
   * Set event type filter query
   */
  function setEventTypeFilter(query: string): void {
    filter.value = {
      ...filter.value,
      eventTypeQuery: query
    }
  }

  /**
   * Toggle a Trinity type in the filter
   */
  function toggleTrinityTypeFilter(type: FilterableTrinityType): void {
    const newTypes = new Set(filter.value.trinityTypes)
    if (newTypes.has(type)) {
      newTypes.delete(type)
    } else {
      newTypes.add(type)
    }
    filter.value = {
      ...filter.value,
      trinityTypes: newTypes
    }
  }

  /**
   * Reset all filters
   */
  function resetFilters(): void {
    filter.value = {
      eventTypeQuery: '',
      trinityTypes: new Set(['component', 'system', 'resource', 'event'])
    }
  }

  /**
   * Toggle event expanded state
   */
  function toggleEventExpanded(eventId: string): void {
    if (expandedEventId.value === eventId) {
      expandedEventId.value = null
    } else {
      expandedEventId.value = eventId
    }
  }

  /**
   * Highlight the node associated with an event and center view on it
   */
  function highlightEventNode(eventId: string): void {
    const event = events.value.find((e) => e.id === eventId)
    if (!event?.nodeId || !graph?.value || !canvas?.value) return

    const node = graph.value.getNodeById(event.nodeId)
    if (!node) return

    // Select the node
    canvas.value.selectNode(node, false)

    // Center view on the node
    centerOnNode(node)

    // Flash highlight
    flashHighlightNode(event.nodeId)
  }

  /**
   * Center the canvas view on a node
   */
  function centerOnNode(node: LGraphNode): void {
    if (!canvas?.value) return

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
   * Flash highlight a node briefly
   */
  function flashHighlightNode(nodeId: NodeId): void {
    if (!graph?.value || !canvas?.value) return

    const node = graph.value.getNodeById(nodeId)
    if (!node) return

    // Clear any existing flash timer for this node
    const existingTimer = flashTimers.get(nodeId)
    if (existingTimer) {
      clearTimeout(existingTimer)
    }

    // Store original colors if not already stored
    if (!originalNodeColors.has(nodeId)) {
      const colors: { color?: string; boxcolor?: string } = {}
      if (node.color !== undefined) colors.color = node.color
      if (node.boxcolor !== undefined) colors.boxcolor = node.boxcolor
      originalNodeColors.set(nodeId, colors)
    }

    // Apply highlight - use event color (orange)
    node.boxcolor = TRINITY_COLORS.event.primary
    highlightedNodeIds.value = new Set([...highlightedNodeIds.value, nodeId])

    // Trigger canvas redraw
    canvas.value.setDirty(true, true)

    // Set timer to restore original colors
    const timer = setTimeout(() => {
      const originalColors = originalNodeColors.get(nodeId)
      if (originalColors && graph?.value) {
        const n = graph.value.getNodeById(nodeId)
        if (n) {
          if (originalColors.color !== undefined) {
            n.color = originalColors.color
          }
          if (originalColors.boxcolor !== undefined) {
            n.boxcolor = originalColors.boxcolor
          }
        }
      }

      originalNodeColors.delete(nodeId)
      flashTimers.delete(nodeId)

      const newHighlighted = new Set(highlightedNodeIds.value)
      newHighlighted.delete(nodeId)
      highlightedNodeIds.value = newHighlighted

      if (canvas?.value) {
        canvas.value.setDirty(true, true)
      }
    }, FLASH_HIGHLIGHT_DURATION)

    flashTimers.set(nodeId, timer)
  }

  /**
   * Start polling for events
   */
  function startPolling(): void {
    if (pollTimer) return

    pollTimer = setInterval(() => {
      // This would integrate with trinityStore or an event bus
      // For now, it's a placeholder for the polling mechanism
      // In a real implementation, this would fetch events from
      // the Trinity ECS runtime or a WebSocket connection
    }, pollInterval)
  }

  /**
   * Stop polling for events
   */
  function stopPolling(): void {
    if (pollTimer) {
      clearInterval(pollTimer)
      pollTimer = null
    }
  }

  /**
   * Mark events as viewed
   */
  function markEventsViewed(): void {
    newEventCount.value = 0
  }

  /**
   * Set the panel as active
   */
  function setActive(active: boolean): void {
    isActive.value = active
    if (active) {
      markEventsViewed()
      startPolling()
    } else {
      stopPolling()
    }
  }

  // Watch for auto-scroll changes and persist
  watch(autoScroll, (value) => {
    saveSettings({ autoScroll: value })
  })

  // Cleanup on unmount would be handled by the component using onUnmounted

  return {
    events,
    filteredEvents,
    isActive,
    autoScroll,
    isPaused,
    filter,
    expandedEventId,
    highlightedNodeIds,
    newEventCount,
    addEvent,
    clearEvents,
    togglePause,
    toggleAutoScroll,
    setEventTypeFilter,
    toggleTrinityTypeFilter,
    resetFilters,
    toggleEventExpanded,
    highlightEventNode,
    flashHighlightNode,
    startPolling,
    stopPolling,
    markEventsViewed,
    setActive
  }
}

export type { FilterableTrinityType }
