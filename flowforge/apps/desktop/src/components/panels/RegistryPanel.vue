<template>
  <div class="registry-panel">
    <!-- Header -->
    <div class="panel-header">
      <div class="header-title">
        <span class="title-text">Registry</span>
        <span
          v-if="!isEmpty"
          class="total-count"
          :title="`${entries.length} total entries`"
        >
          {{ entries.length }}
        </span>
      </div>
      <div class="header-actions">
        <button
          class="action-btn"
          title="Refresh registry"
          :disabled="isLoading"
          @click="handleRefresh"
        >
          <i :class="['icon-refresh', { spinning: isLoading }]">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
              <path d="M3 3v5h5" />
              <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
              <path d="M21 21v-5h-5" />
            </svg>
          </i>
        </button>
        <button
          class="action-btn"
          :title="allExpanded ? 'Collapse all' : 'Expand all'"
          @click="toggleExpandAll"
        >
          <i v-if="allExpanded" class="icon-collapse">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="m7 20 5-5 5 5" />
              <path d="m7 4 5 5 5-5" />
            </svg>
          </i>
          <i v-else class="icon-expand">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="m7 15 5 5 5-5" />
              <path d="m7 9 5-5 5 5" />
            </svg>
          </i>
        </button>
      </div>
    </div>

    <!-- Search -->
    <div class="panel-search">
      <div class="search-input-wrapper">
        <i class="search-icon">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.3-4.3" />
          </svg>
        </i>
        <input
          v-model="searchQuery"
          type="text"
          class="search-input"
          placeholder="Filter entries..."
          @input="handleSearchInput"
        />
        <button
          v-if="searchQuery"
          class="clear-btn"
          title="Clear search"
          @click="clearSearch"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M18 6 6 18" />
            <path d="m6 6 12 12" />
          </svg>
        </button>
      </div>
    </div>

    <!-- Content -->
    <div class="panel-content">
      <!-- Loading State -->
      <div v-if="isLoading && isEmpty" class="panel-state loading-state">
        <div class="spinner" />
        <span>Loading registry...</span>
      </div>

      <!-- Error State -->
      <div v-else-if="error" class="panel-state error-state">
        <i class="state-icon error">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10" />
            <path d="m15 9-6 6" />
            <path d="m9 9 6 6" />
          </svg>
        </i>
        <span class="state-message">{{ error }}</span>
        <button class="retry-btn" @click="handleRefresh">Retry</button>
      </div>

      <!-- Empty State (Not Connected) -->
      <div v-else-if="!isConnected" class="panel-state empty-state">
        <i class="state-icon disconnected">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z" />
            <path d="M8 12h.01" />
            <path d="M16 12h.01" />
            <path d="M9 17s1-1 3-1 3 1 3 1" />
          </svg>
        </i>
        <span class="state-title">No Registry Connection</span>
        <span class="state-subtitle">Open a Trinity project to view registered types</span>
      </div>

      <!-- Empty Search Results -->
      <div v-else-if="filteredEntries.length === 0 && searchQuery" class="panel-state empty-state">
        <i class="state-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.3-4.3" />
          </svg>
        </i>
        <span class="state-message">No results for "{{ searchQuery }}"</span>
      </div>

      <!-- Registry Sections -->
      <template v-else>
        <!-- Components Section -->
        <RegistrySection
          type="component"
          :entries="componentEntries"
          :is-expanded="expandedSections.component"
          :total-count="entryCounts.component"
          :filtered-count="filteredCounts.component"
          :is-filtered="!!searchQuery"
          @toggle="toggleSection('component')"
          @entry-click="handleEntryClick"
        />

        <!-- Systems Section -->
        <RegistrySection
          type="system"
          :entries="systemEntries"
          :is-expanded="expandedSections.system"
          :total-count="entryCounts.system"
          :filtered-count="filteredCounts.system"
          :is-filtered="!!searchQuery"
          @toggle="toggleSection('system')"
          @entry-click="handleEntryClick"
        />

        <!-- Resources Section -->
        <RegistrySection
          type="resource"
          :entries="resourceEntries"
          :is-expanded="expandedSections.resource"
          :total-count="entryCounts.resource"
          :filtered-count="filteredCounts.resource"
          :is-filtered="!!searchQuery"
          @toggle="toggleSection('resource')"
          @entry-click="handleEntryClick"
        />

        <!-- Events Section -->
        <RegistrySection
          type="event"
          :entries="eventEntries"
          :is-expanded="expandedSections.event"
          :total-count="entryCounts.event"
          :filtered-count="filteredCounts.event"
          :is-filtered="!!searchQuery"
          @toggle="toggleSection('event')"
          @entry-click="handleEntryClick"
        />
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, shallowRef } from 'vue'
import type { LGraph, LGraphCanvas } from '@/litegraph'
import { useRegistryPanel, type RegistryEntry } from '@/composables/useRegistryPanel'
import RegistrySection from './RegistrySection.vue'

// =============================================================================
// PROPS & EMITS
// =============================================================================

const props = defineProps<{
  /** Optional LiteGraph graph instance */
  graph?: LGraph | null
  /** Optional LiteGraph canvas instance */
  canvas?: LGraphCanvas | null
}>()

const emit = defineEmits<{
  (e: 'entryClick', entry: RegistryEntry): void
  (e: 'refresh'): void
}>()

// =============================================================================
// COMPOSABLE
// =============================================================================

const graphRef = shallowRef(props.graph ?? null)
const canvasRef = shallowRef(props.canvas ?? null)

const {
  entries,
  isLoading,
  error,
  searchQuery,
  expandedSections,
  isConnected,
  filteredEntries,
  componentEntries,
  systemEntries,
  resourceEntries,
  eventEntries,
  entryCounts,
  filteredCounts,
  isEmpty,
  refresh,
  setSearchQuery,
  toggleSection,
  expandAll,
  collapseAll,
  highlightNode
} = useRegistryPanel({
  graph: graphRef,
  canvas: canvasRef
})

// =============================================================================
// COMPUTED
// =============================================================================

const allExpanded = computed(() =>
  Object.values(expandedSections.value).every(Boolean)
)

// =============================================================================
// METHODS
// =============================================================================

function handleRefresh(): void {
  refresh()
  emit('refresh')
}

function handleSearchInput(event: Event): void {
  const target = event.target as HTMLInputElement
  setSearchQuery(target.value)
}

function clearSearch(): void {
  setSearchQuery('')
}

function toggleExpandAll(): void {
  if (allExpanded.value) {
    collapseAll()
  } else {
    expandAll()
  }
}

function handleEntryClick(entry: RegistryEntry): void {
  highlightNode(entry)
  emit('entryClick', entry)
}
</script>

<style scoped>
.registry-panel {
  display: flex;
  flex-direction: column;
  width: 280px;
  height: 100%;
  background-color: var(--flowforge-panel-bg, #252530);
  color: var(--flowforge-text, #e0e0e0);
  font-family: Inter, system-ui, -apple-system, sans-serif;
  overflow: hidden;
}

/* Header */
.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  border-bottom: 1px solid var(--flowforge-border, #3a3a4a);
  background-color: var(--flowforge-panel-secondary, #1e1e28);
}

.header-title {
  display: flex;
  align-items: center;
  gap: 8px;
}

.title-text {
  font-size: 13px;
  font-weight: 600;
  color: var(--flowforge-text, #e0e0e0);
}

.total-count {
  font-size: 11px;
  color: var(--flowforge-text-muted, #666680);
  background-color: var(--flowforge-surface, #2a2a3a);
  padding: 2px 6px;
  border-radius: 10px;
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
  border: none;
  border-radius: 4px;
  background: transparent;
  color: var(--flowforge-text-secondary, #a0a0a0);
  cursor: pointer;
  transition: all 0.15s ease;
}

.action-btn:hover {
  background-color: var(--flowforge-hover-bg, #3a3a4a);
  color: var(--flowforge-text, #e0e0e0);
}

.action-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.icon-refresh.spinning svg {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* Search */
.panel-search {
  padding: 8px 10px;
  border-bottom: 1px solid var(--flowforge-border, #3a3a4a);
}

.search-input-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

.search-icon {
  position: absolute;
  left: 8px;
  color: var(--flowforge-text-muted, #666680);
  pointer-events: none;
  display: flex;
  align-items: center;
}

.search-input {
  width: 100%;
  padding: 6px 28px 6px 28px;
  border: 1px solid var(--flowforge-input-border, #3a3a4a);
  border-radius: 6px;
  background-color: var(--flowforge-input-bg, #1a1a24);
  color: var(--flowforge-input-text, #e0e0e0);
  font-size: 12px;
  transition: border-color 0.15s ease;
}

.search-input::placeholder {
  color: var(--flowforge-input-placeholder, #666680);
}

.search-input:focus {
  outline: none;
  border-color: var(--flowforge-input-focus-border, #6366f1);
}

.clear-btn {
  position: absolute;
  right: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  padding: 0;
  border: none;
  border-radius: 4px;
  background: transparent;
  color: var(--flowforge-text-muted, #666680);
  cursor: pointer;
}

.clear-btn:hover {
  color: var(--flowforge-text, #e0e0e0);
  background-color: var(--flowforge-hover-bg, #3a3a4a);
}

/* Content */
.panel-content {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
}

/* Scrollbar */
.panel-content::-webkit-scrollbar {
  width: 6px;
}

.panel-content::-webkit-scrollbar-track {
  background: transparent;
}

.panel-content::-webkit-scrollbar-thumb {
  background-color: var(--flowforge-border, #3a3a4a);
  border-radius: 3px;
}

.panel-content::-webkit-scrollbar-thumb:hover {
  background-color: var(--flowforge-border-light, #4a4a5a);
}

/* Panel States */
.panel-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 32px 16px;
  text-align: center;
  gap: 12px;
}

.loading-state {
  color: var(--flowforge-text-muted, #666680);
  font-size: 13px;
}

.spinner {
  width: 24px;
  height: 24px;
  border: 2px solid var(--flowforge-border, #3a3a4a);
  border-top-color: var(--flowforge-primary, #6366f1);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.error-state {
  color: var(--flowforge-danger, #dc2626);
}

.state-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--flowforge-text-muted, #666680);
}

.state-icon.error {
  color: var(--flowforge-danger, #dc2626);
}

.state-icon.disconnected {
  color: var(--flowforge-text-muted, #666680);
  opacity: 0.6;
}

.state-message {
  font-size: 13px;
  color: var(--flowforge-text-secondary, #a0a0a0);
}

.state-title {
  font-size: 14px;
  font-weight: 500;
  color: var(--flowforge-text, #e0e0e0);
}

.state-subtitle {
  font-size: 12px;
  color: var(--flowforge-text-muted, #666680);
}

.retry-btn {
  margin-top: 4px;
  padding: 6px 12px;
  border: 1px solid var(--flowforge-border, #3a3a4a);
  border-radius: 4px;
  background: transparent;
  color: var(--flowforge-text, #e0e0e0);
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.retry-btn:hover {
  background-color: var(--flowforge-hover-bg, #3a3a4a);
}

.empty-state {
  color: var(--flowforge-text-muted, #666680);
}
</style>
