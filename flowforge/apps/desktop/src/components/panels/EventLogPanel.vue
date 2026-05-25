<template>
  <div class="event-log-panel">
    <!-- Header -->
    <div class="event-log-header">
      <div class="header-title">
        <span class="title-text">Event Log</span>
        <span v-if="newEventCount > 0" class="new-badge">{{ newEventCount }}</span>
      </div>
      <div class="header-actions">
        <button
          type="button"
          class="action-btn"
          :class="{ active: autoScroll }"
          :title="autoScroll ? 'Disable auto-scroll' : 'Enable auto-scroll'"
          @click="toggleAutoScroll"
        >
          <i class="icon-[lucide--arrow-down-to-line]" />
        </button>
        <button
          type="button"
          class="action-btn"
          :class="{ active: isPaused }"
          :title="isPaused ? 'Resume logging' : 'Pause logging'"
          @click="togglePause"
        >
          <i :class="isPaused ? 'icon-[lucide--play]' : 'icon-[lucide--pause]'" />
        </button>
        <button
          type="button"
          class="action-btn danger"
          title="Clear all events"
          :disabled="filteredEvents.length === 0"
          @click="clearEvents"
        >
          <i class="icon-[lucide--trash-2]" />
        </button>
      </div>
    </div>

    <!-- Filters -->
    <div class="event-log-filters">
      <div class="search-input-wrapper">
        <i class="icon-[lucide--search] search-icon" />
        <input
          v-model="searchQuery"
          type="text"
          class="search-input"
          placeholder="Filter by event type..."
          @input="onSearchInput"
        />
        <button
          v-if="searchQuery"
          type="button"
          class="clear-search-btn"
          @click="clearSearch"
        >
          <i class="icon-[lucide--x]" />
        </button>
      </div>
      <div class="type-filters">
        <button
          v-for="typeInfo in typeInfos"
          :key="typeInfo.type"
          type="button"
          class="type-filter-btn"
          :class="{ active: isTypeActive(typeInfo.type) }"
          :style="getTypeStyle(typeInfo.type)"
          :title="`${isTypeActive(typeInfo.type) ? 'Hide' : 'Show'} ${typeInfo.label} events`"
          @click="toggleTrinityTypeFilter(typeInfo.type)"
        >
          <i :class="typeInfo.icon" />
        </button>
      </div>
    </div>

    <!-- Event List -->
    <div
      ref="eventListRef"
      class="event-list"
      @scroll="onScroll"
    >
      <div v-if="filteredEvents.length === 0" class="empty-state">
        <i class="icon-[lucide--inbox] empty-icon" />
        <span class="empty-text">No events</span>
        <span v-if="searchQuery || !allTypesActive" class="empty-hint">
          Try adjusting your filters
        </span>
      </div>

      <TransitionGroup name="event-item" tag="div">
        <div
          v-for="event in filteredEvents"
          :key="event.id"
          class="event-item"
          :class="{
            expanded: expandedEventId === event.id,
            highlighted: isNodeHighlighted(event.nodeId)
          }"
          :style="getEventStyle(event.trinityType)"
        >
          <div class="event-row" @click="toggleEventExpanded(event.id)">
            <div class="event-type-indicator" :style="getIndicatorStyle(event.trinityType)" />
            <div class="event-content">
              <div class="event-header">
                <span
                  class="event-type-name"
                  @click.stop="onEventTypeClick(event)"
                >
                  {{ event.eventType }}
                </span>
                <span class="event-timestamp">{{ formatTimestamp(event.timestamp) }}</span>
              </div>
              <div class="event-preview">
                {{ getPayloadPreview(event.payload) }}
              </div>
            </div>
            <button
              type="button"
              class="expand-btn"
              :aria-expanded="expandedEventId === event.id"
            >
              <i
                class="icon-[lucide--chevron-down]"
                :class="{ rotated: expandedEventId === event.id }"
              />
            </button>
          </div>

          <Transition name="expand">
            <div v-if="expandedEventId === event.id" class="event-details">
              <div class="detail-section">
                <span class="detail-label">Full Payload</span>
                <pre class="payload-content">{{ formatPayload(event.payload) }}</pre>
              </div>
              <div v-if="event.source" class="detail-section">
                <span class="detail-label">Source</span>
                <span class="source-path">{{ event.source.file }}:{{ event.source.line }}</span>
              </div>
              <div v-if="event.nodeId" class="detail-actions">
                <button
                  type="button"
                  class="detail-btn"
                  @click.stop="highlightEventNode(event.id)"
                >
                  <i class="icon-[lucide--crosshair]" />
                  Locate Node
                </button>
              </div>
            </div>
          </Transition>
        </div>
      </TransitionGroup>
    </div>

    <!-- Status Bar -->
    <div class="event-log-status">
      <span class="status-count">
        {{ filteredEvents.length }} / {{ events.length }} events
      </span>
      <span v-if="isPaused" class="status-paused">
        <i class="icon-[lucide--pause-circle]" />
        Paused
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted, nextTick, shallowRef } from 'vue'
import type { LGraph, LGraphCanvas, NodeId } from '@/litegraph'
import { useEventLog, type FilterableTrinityType, type EventLogEntry } from '@/composables/useEventLog'
import { TRINITY_COLORS } from '@/nodes/nodeTheme'

// Props
const props = withDefaults(defineProps<{
  /** Reference to the LiteGraph graph instance */
  graph?: InstanceType<typeof LGraph> | null
  /** Reference to the LiteGraph canvas instance */
  canvas?: InstanceType<typeof LGraphCanvas> | null
  /** Maximum events to keep in memory */
  maxEvents?: number
}>(), {
  graph: null,
  canvas: null,
  maxEvents: 100
})

// Emit events
const emit = defineEmits<{
  'event-type-click': [event: EventLogEntry]
  'node-highlight': [nodeId: NodeId]
}>()

// Create shallow refs for graph/canvas to pass to composable
const graphRef = shallowRef<LGraph | null>(props.graph)
const canvasRef = shallowRef<LGraphCanvas | null>(props.canvas)

// Watch for prop changes
watch(() => props.graph, (newGraph) => {
  graphRef.value = newGraph
})

watch(() => props.canvas, (newCanvas) => {
  canvasRef.value = newCanvas
})

// Use the event log composable
const {
  events,
  filteredEvents,
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
  toggleEventExpanded,
  highlightEventNode,
  flashHighlightNode,
  stopPolling,
  markEventsViewed,
  setActive
} = useEventLog({
  graph: graphRef,
  canvas: canvasRef,
  maxEvents: props.maxEvents
})

// Local state
const searchQuery = ref('')
const eventListRef = ref<HTMLElement | null>(null)
const userScrolled = ref(false)

// Type information for filters
interface TypeInfo {
  type: FilterableTrinityType
  label: string
  icon: string
}

const typeInfos: TypeInfo[] = [
  { type: 'component', label: 'Components', icon: 'icon-[lucide--box]' },
  { type: 'system', label: 'Systems', icon: 'icon-[lucide--cpu]' },
  { type: 'resource', label: 'Resources', icon: 'icon-[lucide--database]' },
  { type: 'event', label: 'Events', icon: 'icon-[lucide--zap]' }
]

/**
 * Check if a Trinity type is active in the filter
 */
function isTypeActive(type: FilterableTrinityType): boolean {
  return filter.value.trinityTypes.has(type)
}

/**
 * Check if all types are active
 */
const allTypesActive = computed(() => filter.value.trinityTypes.size === 4)

/**
 * Get style for type filter button
 */
function getTypeStyle(type: FilterableTrinityType): Record<string, string> {
  const colors = TRINITY_COLORS[type]
  const isActive = isTypeActive(type)

  return {
    '--btn-color': isActive ? colors.primary : 'var(--flowforge-text-muted)',
    '--btn-bg': isActive ? colors.primary : 'transparent',
    '--btn-border': isActive ? colors.primary : 'var(--flowforge-border)'
  }
}

/**
 * Get style for an event item
 */
function getEventStyle(type: FilterableTrinityType): Record<string, string> {
  const colors = TRINITY_COLORS[type]
  return {
    '--event-color': colors.primary,
    '--event-bg': colors.background,
    '--event-border': colors.border
  }
}

/**
 * Get style for the type indicator
 */
function getIndicatorStyle(type: FilterableTrinityType): Record<string, string> {
  return {
    backgroundColor: TRINITY_COLORS[type].primary
  }
}

/**
 * Check if a node is currently highlighted
 */
function isNodeHighlighted(nodeId?: NodeId): boolean {
  if (!nodeId) return false
  return highlightedNodeIds.value.has(nodeId)
}

/**
 * Format timestamp in HH:MM:SS.mmm format
 */
function formatTimestamp(timestamp: number): string {
  const date = new Date(timestamp)
  const hours = date.getHours().toString().padStart(2, '0')
  const minutes = date.getMinutes().toString().padStart(2, '0')
  const seconds = date.getSeconds().toString().padStart(2, '0')
  const millis = date.getMilliseconds().toString().padStart(3, '0')
  return `${hours}:${minutes}:${seconds}.${millis}`
}

/**
 * Get a preview of the payload (first 60 chars)
 */
function getPayloadPreview(payload: Record<string, unknown>): string {
  try {
    const str = JSON.stringify(payload)
    if (str.length <= 60) return str
    return str.substring(0, 57) + '...'
  } catch {
    return '[Complex Object]'
  }
}

/**
 * Format payload for detailed view
 */
function formatPayload(payload: Record<string, unknown>): string {
  try {
    return JSON.stringify(payload, null, 2)
  } catch {
    return '[Unable to serialize payload]'
  }
}

/**
 * Handle search input
 */
function onSearchInput(): void {
  setEventTypeFilter(searchQuery.value)
}

/**
 * Clear search
 */
function clearSearch(): void {
  searchQuery.value = ''
  setEventTypeFilter('')
}

/**
 * Handle scroll to detect user scrolling
 */
function onScroll(): void {
  if (!eventListRef.value) return

  const { scrollTop, scrollHeight, clientHeight } = eventListRef.value
  const isAtBottom = scrollHeight - scrollTop - clientHeight < 10

  if (!isAtBottom && autoScroll.value) {
    userScrolled.value = true
  } else if (isAtBottom) {
    userScrolled.value = false
  }
}

/**
 * Scroll to top when new events arrive (if auto-scroll is enabled)
 */
function scrollToTop(): void {
  if (!eventListRef.value || !autoScroll.value || userScrolled.value) return
  eventListRef.value.scrollTop = 0
}

/**
 * Handle click on event type to highlight node in graph
 */
function onEventTypeClick(event: EventLogEntry): void {
  emit('event-type-click', event)
  if (event.nodeId) {
    highlightEventNode(event.id)
    emit('node-highlight', event.nodeId)
  }
}

// Watch for new events and auto-scroll
watch(
  () => filteredEvents.value.length,
  () => {
    nextTick(scrollToTop)
  }
)

// Lifecycle
onMounted(() => {
  setActive(true)
})

onUnmounted(() => {
  setActive(false)
  stopPolling()
})

// Expose methods for parent components
defineExpose({
  addEvent,
  clearEvents,
  flashHighlightNode,
  markEventsViewed
})
</script>

<style scoped>
.event-log-panel {
  display: flex;
  flex-direction: column;
  width: 320px;
  height: 100%;
  background-color: var(--flowforge-panel-bg);
  border-left: 1px solid var(--flowforge-border);
  font-family: Inter, system-ui, sans-serif;
}

/* Header */
.event-log-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  background-color: var(--flowforge-panel-secondary);
  border-bottom: 1px solid var(--flowforge-border);
}

.header-title {
  display: flex;
  align-items: center;
  gap: 8px;
}

.title-text {
  font-size: 13px;
  font-weight: 600;
  color: var(--flowforge-text-bright);
}

.new-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 18px;
  height: 18px;
  padding: 0 5px;
  font-size: 10px;
  font-weight: 600;
  color: white;
  background-color: var(--flowforge-node-event);
  border-radius: 9px;
  animation: pulse-badge 1.5s ease-in-out infinite;
}

@keyframes pulse-badge {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.1); }
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 4px;
}

.action-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  padding: 0;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 4px;
  color: var(--flowforge-text-secondary);
  cursor: pointer;
  transition: all 0.15s ease;
}

.action-btn:hover {
  background-color: var(--flowforge-hover-bg);
  color: var(--flowforge-text);
}

.action-btn.active {
  background-color: var(--flowforge-selected-bg);
  color: var(--flowforge-primary);
  border-color: var(--flowforge-primary);
}

.action-btn.danger:hover {
  background-color: rgba(220, 38, 38, 0.15);
  color: var(--flowforge-danger);
}

.action-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.action-btn i {
  font-size: 14px;
}

/* Filters */
.event-log-filters {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--flowforge-border);
}

.search-input-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

.search-icon {
  position: absolute;
  left: 8px;
  font-size: 12px;
  color: var(--flowforge-text-muted);
  pointer-events: none;
}

.search-input {
  width: 100%;
  height: 28px;
  padding: 0 28px;
  font-size: 12px;
  color: var(--flowforge-text);
  background-color: var(--flowforge-input-bg);
  border: 1px solid var(--flowforge-input-border);
  border-radius: 4px;
  outline: none;
  transition: border-color 0.15s ease;
}

.search-input::placeholder {
  color: var(--flowforge-input-placeholder);
}

.search-input:focus {
  border-color: var(--flowforge-input-focus-border);
}

.clear-search-btn {
  position: absolute;
  right: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  padding: 0;
  background: transparent;
  border: none;
  border-radius: 2px;
  color: var(--flowforge-text-muted);
  cursor: pointer;
}

.clear-search-btn:hover {
  color: var(--flowforge-text);
  background-color: var(--flowforge-hover-bg);
}

.clear-search-btn i {
  font-size: 12px;
}

.type-filters {
  display: flex;
  gap: 4px;
}

.type-filter-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 1;
  height: 26px;
  padding: 0;
  background-color: var(--btn-bg);
  border: 1px solid var(--btn-border);
  border-radius: 4px;
  color: var(--btn-color);
  cursor: pointer;
  transition: all 0.15s ease;
}

.type-filter-btn.active {
  color: white;
}

.type-filter-btn:hover {
  opacity: 0.85;
}

.type-filter-btn i {
  font-size: 13px;
}

/* Event List */
.event-list {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
  text-align: center;
}

.empty-icon {
  font-size: 32px;
  color: var(--flowforge-text-muted);
  margin-bottom: 8px;
}

.empty-text {
  font-size: 13px;
  color: var(--flowforge-text-secondary);
}

.empty-hint {
  font-size: 11px;
  color: var(--flowforge-text-muted);
  margin-top: 4px;
}

/* Event Item */
.event-item {
  border-bottom: 1px solid var(--flowforge-border);
  transition: background-color 0.15s ease;
}

.event-item:hover {
  background-color: var(--flowforge-hover-bg);
}

.event-item.highlighted {
  background-color: rgba(249, 115, 22, 0.1);
  animation: flash-highlight 0.8s ease-out;
}

@keyframes flash-highlight {
  0% { background-color: rgba(249, 115, 22, 0.3); }
  100% { background-color: rgba(249, 115, 22, 0.1); }
}

.event-row {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 10px 12px;
  cursor: pointer;
}

.event-type-indicator {
  flex-shrink: 0;
  width: 4px;
  height: 100%;
  min-height: 32px;
  border-radius: 2px;
}

.event-content {
  flex: 1;
  min-width: 0;
}

.event-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 4px;
}

.event-type-name {
  font-size: 12px;
  font-weight: 500;
  color: var(--event-color);
  cursor: pointer;
  transition: opacity 0.15s ease;
}

.event-type-name:hover {
  opacity: 0.8;
  text-decoration: underline;
}

.event-timestamp {
  flex-shrink: 0;
  font-size: 10px;
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  color: var(--flowforge-text-muted);
}

.event-preview {
  font-size: 11px;
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  color: var(--flowforge-text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.expand-btn {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  padding: 0;
  margin-top: 2px;
  background: transparent;
  border: none;
  border-radius: 2px;
  color: var(--flowforge-text-muted);
  cursor: pointer;
  transition: all 0.15s ease;
}

.expand-btn:hover {
  background-color: var(--flowforge-hover-bg);
  color: var(--flowforge-text);
}

.expand-btn i {
  font-size: 14px;
  transition: transform 0.2s ease;
}

.expand-btn i.rotated {
  transform: rotate(180deg);
}

/* Event Details */
.event-details {
  padding: 0 12px 12px 24px;
  overflow: hidden;
}

.detail-section {
  margin-bottom: 10px;
}

.detail-section:last-child {
  margin-bottom: 0;
}

.detail-label {
  display: block;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--flowforge-text-muted);
  margin-bottom: 4px;
}

.payload-content {
  margin: 0;
  padding: 8px;
  font-size: 10px;
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  color: var(--flowforge-text);
  background-color: var(--flowforge-input-bg);
  border: 1px solid var(--flowforge-border);
  border-radius: 4px;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 150px;
}

.source-path {
  font-size: 11px;
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  color: var(--flowforge-text-secondary);
}

.detail-actions {
  display: flex;
  gap: 8px;
  margin-top: 8px;
}

.detail-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 5px 10px;
  font-size: 11px;
  color: var(--flowforge-text);
  background-color: var(--flowforge-btn-bg);
  border: 1px solid var(--flowforge-border);
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.detail-btn:hover {
  background-color: var(--flowforge-btn-hover);
}

.detail-btn i {
  font-size: 12px;
}

/* Status Bar */
.event-log-status {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 12px;
  background-color: var(--flowforge-panel-secondary);
  border-top: 1px solid var(--flowforge-border);
}

.status-count {
  font-size: 10px;
  color: var(--flowforge-text-muted);
}

.status-paused {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 10px;
  color: var(--flowforge-warning);
}

.status-paused i {
  font-size: 12px;
}

/* Transitions */
.event-item-enter-active {
  transition: all 0.3s ease-out;
}

.event-item-leave-active {
  transition: all 0.2s ease-in;
}

.event-item-enter-from {
  opacity: 0;
  transform: translateX(-10px);
}

.event-item-leave-to {
  opacity: 0;
  transform: translateX(10px);
}

.event-item-move {
  transition: transform 0.3s ease;
}

.expand-enter-active,
.expand-leave-active {
  transition: all 0.2s ease;
}

.expand-enter-from,
.expand-leave-to {
  opacity: 0;
  max-height: 0;
}

.expand-enter-to,
.expand-leave-from {
  opacity: 1;
  max-height: 300px;
}

/* Scrollbar */
.event-list::-webkit-scrollbar {
  width: 6px;
}

.event-list::-webkit-scrollbar-track {
  background: var(--flowforge-panel-secondary);
}

.event-list::-webkit-scrollbar-thumb {
  background: var(--flowforge-border-light);
  border-radius: 3px;
}

.event-list::-webkit-scrollbar-thumb:hover {
  background: var(--flowforge-primary);
}

/* Reduced Motion */
@media (prefers-reduced-motion: reduce) {
  .new-badge {
    animation: none;
  }

  .event-item.highlighted {
    animation: none;
  }

  .event-item-enter-active,
  .event-item-leave-active,
  .event-item-move,
  .expand-enter-active,
  .expand-leave-active {
    transition: none;
  }

  .expand-btn i {
    transition: none;
  }
}
</style>
