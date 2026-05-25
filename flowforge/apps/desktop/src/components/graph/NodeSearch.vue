<template>
  <Transition name="search-slide">
    <div
      v-if="isVisible"
      class="node-search"
      @keydown.stop="handleKeyDown"
    >
      <!-- Search Header -->
      <div class="search-header">
        <!-- Search Input -->
        <div class="search-input-wrapper">
          <i class="icon-[lucide--search] search-icon" />
          <input
            ref="inputRef"
            v-model="searchQuery"
            type="text"
            class="search-input"
            placeholder="Search nodes..."
            @input="handleInput"
            @keydown.enter.prevent="handleEnter"
            @keydown.escape.prevent="handleEscape"
            @keydown.arrow-down.prevent="handleArrowDown"
            @keydown.arrow-up.prevent="handleArrowUp"
          />
          <button
            v-if="searchQuery"
            class="clear-button"
            @click="clearInput"
            title="Clear search"
          >
            <i class="icon-[lucide--x]" />
          </button>
        </div>

        <!-- Type Filter Dropdown -->
        <div class="type-filter">
          <select
            v-model="typeFilter"
            class="type-select"
            @change="handleTypeChange"
          >
            <option value="all">All Types</option>
            <option value="component">Components</option>
            <option value="system">Systems</option>
            <option value="resource">Resources</option>
            <option value="event">Events</option>
          </select>
          <i class="icon-[lucide--chevron-down] select-icon" />
        </div>

        <!-- Close Button -->
        <button
          class="close-button"
          @click="handleClose"
          title="Close search (Esc)"
        >
          <i class="icon-[lucide--x]" />
        </button>
      </div>

      <!-- Results List -->
      <div v-if="results.length > 0" class="results-container">
        <div class="results-count">
          {{ results.length }} node{{ results.length !== 1 ? 's' : '' }} found
        </div>
        <ul class="results-list" ref="resultsListRef">
          <li
            v-for="(result, index) in results"
            :key="result.id"
            :class="[
              'result-item',
              { 'selected': index === selectedIndex }
            ]"
            @click="selectResult(result)"
            @mouseenter="selectedIndex = index"
          >
            <div class="result-content">
              <span
                class="result-type-indicator"
                :style="{ backgroundColor: getTypeColor(result.trinityType) }"
              />
              <span class="result-title">{{ result.title }}</span>
              <span class="result-type-label">{{ formatType(result.trinityType) }}</span>
            </div>
            <span class="result-position">
              ({{ Math.round(result.pos[0]) }}, {{ Math.round(result.pos[1]) }})
            </span>
          </li>
        </ul>
      </div>

      <!-- No Results -->
      <div v-else-if="searchQuery || typeFilter !== 'all'" class="no-results">
        <i class="icon-[lucide--search-x] no-results-icon" />
        <span>No nodes found</span>
      </div>

      <!-- Keyboard Hints -->
      <div class="keyboard-hints">
        <span class="hint"><kbd>Enter</kbd> Select</span>
        <span class="hint"><kbd>Esc</kbd> Close</span>
      </div>
    </div>
  </Transition>
</template>

<script setup lang="ts">
import { ref, watch, nextTick, onUnmounted } from 'vue'
import type { NodeTypeFilter, SearchResult } from '@/composables/useNodeSearch'
import { TRINITY_COLORS } from '@/nodes/nodeTheme'
import { UI_CONFIG } from '@/config/flowforge.config'

// =============================================================================
// PROPS & EMITS
// =============================================================================

const props = defineProps<{
  isVisible: boolean
  results: SearchResult[]
  selectedIndex: number
}>()

const emit = defineEmits<{
  (e: 'search', query: string, type: NodeTypeFilter): void
  (e: 'select', result: SearchResult): void
  (e: 'close'): void
  (e: 'update:selectedIndex', index: number): void
  (e: 'selectNext'): void
  (e: 'selectPrevious'): void
  (e: 'confirm'): void
}>()

// =============================================================================
// STATE
// =============================================================================

const searchQuery = ref('')
const typeFilter = ref<NodeTypeFilter>('all')
const inputRef = ref<HTMLInputElement | null>(null)
const resultsListRef = ref<HTMLElement | null>(null)

// Computed selected index (for two-way binding)
const selectedIndex = ref(props.selectedIndex)

// Watch for external selectedIndex changes
watch(() => props.selectedIndex, (newVal) => {
  selectedIndex.value = newVal
  scrollToSelected()
})

// Emit selectedIndex changes
watch(selectedIndex, (newVal) => {
  emit('update:selectedIndex', newVal)
})

// =============================================================================
// TYPE COLORS
// =============================================================================

function getTypeColor(type: NodeTypeFilter): string {
  switch (type) {
    case 'component':
      return TRINITY_COLORS.component.primary
    case 'system':
      return TRINITY_COLORS.system.primary
    case 'resource':
      return TRINITY_COLORS.resource.primary
    case 'event':
      return TRINITY_COLORS.event.primary
    default:
      return TRINITY_COLORS.neutral.textMuted
  }
}

function formatType(type: NodeTypeFilter): string {
  if (type === 'all') return ''
  return type.charAt(0).toUpperCase() + type.slice(1)
}

// =============================================================================
// EVENT HANDLERS
// =============================================================================

function handleInput(): void {
  emit('search', searchQuery.value, typeFilter.value)
}

// Debounced search for better performance
let debounceTimer: ReturnType<typeof setTimeout> | null = null
watch(
  searchQuery,
  (value: string) => {
    if (debounceTimer) clearTimeout(debounceTimer)
    debounceTimer = setTimeout(() => {
      emit('search', value, typeFilter.value)
    }, UI_CONFIG.search.debounceMs)
  }
)

// Cleanup debounce timer on unmount
onUnmounted(() => {
  if (debounceTimer) clearTimeout(debounceTimer)
})

function handleTypeChange(): void {
  emit('search', searchQuery.value, typeFilter.value)
}

function handleEnter(): void {
  if (props.results.length > 0) {
    emit('confirm')
  }
}

function handleEscape(): void {
  if (searchQuery.value) {
    clearInput()
  } else {
    emit('close')
  }
}

function handleArrowDown(): void {
  emit('selectNext')
}

function handleArrowUp(): void {
  emit('selectPrevious')
}

function handleKeyDown(event: KeyboardEvent): void {
  // Prevent default for arrow keys to avoid scrolling
  if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
    event.preventDefault()
  }
}

function clearInput(): void {
  searchQuery.value = ''
  typeFilter.value = 'all'
  emit('search', '', 'all')
}

function handleClose(): void {
  clearInput()
  emit('close')
}

function selectResult(result: SearchResult): void {
  emit('select', result)
}

// =============================================================================
// SCROLL MANAGEMENT
// =============================================================================

function scrollToSelected(): void {
  if (!resultsListRef.value) return

  const items = resultsListRef.value.querySelectorAll('.result-item')
  const selectedItem = items[selectedIndex.value] as HTMLElement | undefined

  if (selectedItem) {
    selectedItem.scrollIntoView({
      block: 'nearest',
      behavior: 'smooth'
    })
  }
}

// =============================================================================
// FOCUS MANAGEMENT
// =============================================================================

watch(() => props.isVisible, (visible) => {
  if (visible) {
    nextTick(() => {
      inputRef.value?.focus()
    })
  } else {
    // Reset state when hidden
    searchQuery.value = ''
    typeFilter.value = 'all'
    selectedIndex.value = 0
  }
})

// =============================================================================
// EXPOSE
// =============================================================================

defineExpose({
  focus: () => inputRef.value?.focus(),
  clear: clearInput
})
</script>

<style scoped>
.node-search {
  position: absolute;
  top: 16px;
  right: 16px;
  z-index: 100;
  width: 320px;
  max-height: 400px;
  display: flex;
  flex-direction: column;
  background-color: var(--flowforge-panel-bg, #252530);
  border: 1px solid var(--flowforge-border, #3a3a4a);
  border-radius: 8px;
  box-shadow: var(--flowforge-shadow-lg, 0 10px 15px rgba(0, 0, 0, 0.5));
  overflow: hidden;
}

/* Search Header */
.search-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px;
  border-bottom: 1px solid var(--flowforge-border, #3a3a4a);
}

.search-input-wrapper {
  position: relative;
  flex: 1;
  display: flex;
  align-items: center;
}

.search-icon {
  position: absolute;
  left: 10px;
  width: 16px;
  height: 16px;
  color: var(--flowforge-text-muted, #666680);
  pointer-events: none;
}

.search-input {
  width: 100%;
  height: 32px;
  padding: 0 32px 0 32px;
  background-color: var(--flowforge-input-bg, #1a1a24);
  border: 1px solid var(--flowforge-input-border, #3a3a4a);
  border-radius: 6px;
  color: var(--flowforge-input-text, #e0e0e0);
  font-size: 13px;
  outline: none;
  transition: border-color 0.2s ease;
}

.search-input::placeholder {
  color: var(--flowforge-input-placeholder, #666680);
}

.search-input:focus {
  border-color: var(--flowforge-primary, #6366f1);
}

.clear-button {
  position: absolute;
  right: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  padding: 0;
  background: transparent;
  border: none;
  border-radius: 4px;
  color: var(--flowforge-text-muted, #666680);
  cursor: pointer;
  transition: all 0.15s ease;
}

.clear-button:hover {
  background-color: var(--flowforge-hover-bg, #3a3a4a);
  color: var(--flowforge-text, #e0e0e0);
}

/* Type Filter */
.type-filter {
  position: relative;
  flex-shrink: 0;
}

.type-select {
  height: 32px;
  padding: 0 28px 0 10px;
  appearance: none;
  background-color: var(--flowforge-input-bg, #1a1a24);
  border: 1px solid var(--flowforge-input-border, #3a3a4a);
  border-radius: 6px;
  color: var(--flowforge-input-text, #e0e0e0);
  font-size: 12px;
  cursor: pointer;
  outline: none;
  transition: border-color 0.2s ease;
}

.type-select:focus {
  border-color: var(--flowforge-primary, #6366f1);
}

.select-icon {
  position: absolute;
  right: 8px;
  top: 50%;
  transform: translateY(-50%);
  width: 14px;
  height: 14px;
  color: var(--flowforge-text-muted, #666680);
  pointer-events: none;
}

/* Close Button */
.close-button {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  padding: 0;
  background: transparent;
  border: none;
  border-radius: 6px;
  color: var(--flowforge-text-muted, #666680);
  cursor: pointer;
  transition: all 0.15s ease;
}

.close-button:hover {
  background-color: var(--flowforge-hover-bg, #3a3a4a);
  color: var(--flowforge-text, #e0e0e0);
}

/* Results */
.results-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.results-count {
  padding: 8px 12px;
  font-size: 11px;
  color: var(--flowforge-text-muted, #666680);
  border-bottom: 1px solid var(--flowforge-border, #3a3a4a);
}

.results-list {
  flex: 1;
  overflow-y: auto;
  margin: 0;
  padding: 4px;
  list-style: none;
}

.result-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.15s ease;
}

.result-item:hover,
.result-item.selected {
  background-color: var(--flowforge-hover-bg, #3a3a4a);
}

.result-item.selected {
  background-color: var(--flowforge-selected-bg, rgba(99, 102, 241, 0.2));
}

.result-content {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
  min-width: 0;
}

.result-type-indicator {
  flex-shrink: 0;
  width: 8px;
  height: 8px;
  border-radius: 2px;
}

.result-title {
  flex: 1;
  font-size: 13px;
  color: var(--flowforge-text, #e0e0e0);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.result-type-label {
  flex-shrink: 0;
  font-size: 10px;
  color: var(--flowforge-text-muted, #666680);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.result-position {
  flex-shrink: 0;
  font-size: 11px;
  font-family: monospace;
  color: var(--flowforge-text-muted, #666680);
}

/* No Results */
.no-results {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 24px;
  color: var(--flowforge-text-muted, #666680);
}

.no-results-icon {
  width: 32px;
  height: 32px;
  opacity: 0.5;
}

/* Keyboard Hints */
.keyboard-hints {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  padding: 8px 12px;
  border-top: 1px solid var(--flowforge-border, #3a3a4a);
  background-color: var(--flowforge-panel-secondary, #1e1e28);
}

.hint {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: var(--flowforge-text-muted, #666680);
}

kbd {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 18px;
  height: 18px;
  padding: 0 4px;
  background-color: var(--flowforge-surface, #2a2a3a);
  border: 1px solid var(--flowforge-border, #3a3a4a);
  border-radius: 3px;
  font-size: 10px;
  font-family: inherit;
  color: var(--flowforge-text-secondary, #a0a0a0);
}

/* Scrollbar */
.results-list::-webkit-scrollbar {
  width: 6px;
}

.results-list::-webkit-scrollbar-track {
  background: transparent;
}

.results-list::-webkit-scrollbar-thumb {
  background-color: var(--flowforge-border-light, #4a4a5a);
  border-radius: 3px;
}

.results-list::-webkit-scrollbar-thumb:hover {
  background-color: var(--flowforge-primary, #6366f1);
}

/* Animation */
.search-slide-enter-active,
.search-slide-leave-active {
  transition: all 0.2s ease;
}

.search-slide-enter-from,
.search-slide-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}
</style>
