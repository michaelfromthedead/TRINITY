<template>
  <div class="instances-panel">
    <!-- Header -->
    <div class="panel-header">
      <div class="header-title">
        <i class="icon-[lucide--layers] header-icon" />
        <span>Instances</span>
        <span v-if="totalCount > 0" class="total-badge">{{ totalCount }}</span>
      </div>
      <div class="header-actions">
        <button
          class="action-btn"
          :class="{ 'is-loading': isLoading }"
          :disabled="isLoading"
          title="Refresh instances"
          @click="handleRefresh"
        >
          <i class="icon-[lucide--refresh-cw]" :class="{ 'spin': isLoading }" />
        </button>
        <button
          v-if="hasInstances"
          class="action-btn"
          title="Collapse all"
          @click="collapseAll"
        >
          <i class="icon-[lucide--minimize-2]" />
        </button>
        <button
          v-if="hasInstances"
          class="action-btn"
          title="Expand all"
          @click="expandAll"
        >
          <i class="icon-[lucide--maximize-2]" />
        </button>
      </div>
    </div>

    <!-- Search Filter -->
    <div v-if="hasInstances" class="search-container">
      <div class="search-input-wrapper">
        <i class="icon-[lucide--search] search-icon" />
        <input
          v-model="localSearchQuery"
          type="text"
          class="search-input"
          placeholder="Filter by name..."
          @input="handleSearchInput"
        />
        <button
          v-if="localSearchQuery"
          class="clear-button"
          title="Clear filter"
          @click="handleClearSearch"
        >
          <i class="icon-[lucide--x]" />
        </button>
      </div>
    </div>

    <!-- Connection Status -->
    <div v-if="!isConnected" class="status-banner disconnected">
      <i class="icon-[lucide--wifi-off]" />
      <span>Trinity not connected</span>
    </div>

    <!-- Content -->
    <div class="panel-content">
      <!-- Empty State -->
      <div v-if="!hasInstances && !isLoading" class="empty-state">
        <i class="icon-[lucide--box] empty-icon" />
        <span class="empty-text">No active instances</span>
        <button class="refresh-link" @click="handleRefresh">
          Click to refresh
        </button>
      </div>

      <!-- Loading State -->
      <div v-else-if="isLoading && !hasInstances" class="loading-state">
        <i class="icon-[lucide--loader-2] spin loading-icon" />
        <span>Loading instances...</span>
      </div>

      <!-- Instance Tree -->
      <div v-else class="instance-tree">
        <div
          v-for="typeNode in instanceTree"
          :key="typeNode.trinityType"
          class="type-section"
        >
          <!-- Type Header -->
          <button
            class="type-header"
            :class="{ 'expanded': typeNode.isExpanded }"
            @click="toggleTypeExpansion(typeNode.trinityType)"
          >
            <i
              class="expand-icon"
              :class="typeNode.isExpanded ? 'icon-[lucide--chevron-down]' : 'icon-[lucide--chevron-right]'"
            />
            <span
              class="type-indicator"
              :style="{ backgroundColor: getTypeColor(typeNode.trinityType) }"
            />
            <span class="type-name">{{ formatTypeName(typeNode.trinityType) }}</span>
            <span
              class="type-count"
              :style="{ backgroundColor: getTypeColor(typeNode.trinityType) + '30' }"
            >
              {{ typeNode.totalCount }}
            </span>
          </button>

          <!-- Type Content -->
          <Transition name="expand">
            <div
              v-if="typeNode.isExpanded && typeNode.groups.length > 0"
              class="type-content"
            >
              <!-- Component Groups -->
              <div
                v-for="group in typeNode.groups"
                :key="`${group.trinityType}:${group.componentType}`"
                class="component-group"
              >
                <!-- Group Header -->
                <button
                  class="group-header"
                  :class="{ 'expanded': group.isExpanded }"
                  @click="toggleGroupExpansion(group.trinityType, group.componentType)"
                >
                  <i
                    class="expand-icon"
                    :class="group.isExpanded ? 'icon-[lucide--chevron-down]' : 'icon-[lucide--chevron-right]'"
                  />
                  <span
                    class="component-name"
                    @click.stop="handleComponentClick(group)"
                  >
                    {{ group.componentType }}
                  </span>
                  <span class="instance-count">{{ group.instances.length }}</span>
                </button>

                <!-- Instance List -->
                <Transition name="expand">
                  <div v-if="group.isExpanded" class="instance-list">
                    <div
                      v-for="instance in group.instances"
                      :key="instance.id"
                      class="instance-item"
                    >
                      <!-- Instance Header -->
                      <button
                        class="instance-header"
                        :class="{ 'expanded': isInstanceExpanded(instance.id) }"
                        @click="toggleInstanceExpansion(instance.id)"
                      >
                        <i
                          class="expand-icon small"
                          :class="isInstanceExpanded(instance.id) ? 'icon-[lucide--chevron-down]' : 'icon-[lucide--chevron-right]'"
                        />
                        <span class="instance-id">{{ formatInstanceId(instance.id) }}</span>
                        <span class="instance-preview">{{ getDataPreview(instance.data) }}</span>
                        <button
                          v-if="instance.nodeId !== undefined"
                          class="locate-btn"
                          title="Locate in graph"
                          @click.stop="handleLocateInstance(instance)"
                        >
                          <i class="icon-[lucide--locate]" />
                        </button>
                      </button>

                      <!-- Instance Data (JSON Viewer) -->
                      <Transition name="expand">
                        <div
                          v-if="isInstanceExpanded(instance.id)"
                          class="instance-data"
                          :style="getInstanceDataStyle(group.trinityType)"
                        >
                          <div class="instance-data-header">
                            <span class="data-label">Instance Data</span>
                            <button
                              class="copy-btn"
                              title="Copy JSON"
                              @click.stop="handleCopyInstanceData(instance)"
                            >
                              <i class="icon-[lucide--copy]" />
                            </button>
                          </div>
                          <JsonViewer
                            :data="instance.data"
                            :accent-color="getTypeColor(group.trinityType)"
                            compact
                          />
                        </div>
                      </Transition>
                    </div>
                  </div>
                </Transition>
              </div>

              <!-- Empty Type -->
              <div v-if="typeNode.groups.length === 0" class="empty-type">
                No {{ formatTypeName(typeNode.trinityType).toLowerCase() }} instances
              </div>
            </div>
          </Transition>
        </div>
      </div>
    </div>

    <!-- Error Toast -->
    <Transition name="fade">
      <div v-if="error" class="error-toast">
        <i class="icon-[lucide--alert-circle]" />
        <span>{{ error }}</span>
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'
import { useInstances, type TrinityInstance, type InstanceGroup } from '@/composables/useInstances'
import { TRINITY_COLORS } from '@/nodes/nodeTheme'
import type { FilterableTrinityType } from '@/composables/useTypeFilter'
import JsonViewer from './JsonViewer.vue'

// =============================================================================
// COMPOSABLE
// =============================================================================

const {
  instanceTree,
  totalCount,
  hasInstances,
  isConnected,
  isLoading,
  error,
  searchQuery,
  toggleTypeExpansion,
  toggleGroupExpansion,
  toggleInstanceExpansion,
  isInstanceExpanded,
  expandAll,
  collapseAll,
  refreshInstances,
  setSearchQuery,
  clearSearch,
  highlightInstanceNode,
  getDataPreview
} = useInstances()

// =============================================================================
// LOCAL STATE
// =============================================================================

const localSearchQuery = ref('')

// Sync local search query with composable
watch(searchQuery, (val) => {
  localSearchQuery.value = val
})

// =============================================================================
// EMIT EVENTS
// =============================================================================

const emit = defineEmits<{
  (e: 'highlightNode', nodeId: string): void
  (e: 'componentClick', componentType: string, trinityType: FilterableTrinityType): void
}>()

// =============================================================================
// TYPE COLORS
// =============================================================================

function getTypeColor(type: FilterableTrinityType): string {
  return TRINITY_COLORS[type]?.primary || TRINITY_COLORS.component.primary
}

// =============================================================================
// FORMATTING
// =============================================================================

function formatTypeName(type: FilterableTrinityType): string {
  const names: Record<FilterableTrinityType, string> = {
    component: 'Components',
    system: 'Systems',
    resource: 'Resources',
    event: 'Events'
  }
  return names[type] || type
}

function formatInstanceId(id: string): string {
  // Show last 8 characters of ID for brevity
  if (id.length > 12) {
    return '...' + id.slice(-8)
  }
  return id
}

// =============================================================================
// EVENT HANDLERS
// =============================================================================

function handleRefresh(): void {
  refreshInstances()
}

function handleSearchInput(): void {
  setSearchQuery(localSearchQuery.value)
}

function handleClearSearch(): void {
  localSearchQuery.value = ''
  clearSearch()
}

function handleComponentClick(group: InstanceGroup): void {
  emit('componentClick', group.componentType, group.trinityType)
}

function handleLocateInstance(instance: TrinityInstance): void {
  if (instance.nodeId !== undefined) {
    highlightInstanceNode(instance.id)
    emit('highlightNode', instance.nodeId)
  }
}

function handleCopyInstanceData(instance: TrinityInstance): void {
  const jsonStr = JSON.stringify(instance.data, null, 2)
  navigator.clipboard.writeText(jsonStr).catch((err) => {
    console.error('[InstancesPanel] Failed to copy instance data:', err)
  })
}

function getInstanceDataStyle(trinityType: FilterableTrinityType): Record<string, string> {
  const color = getTypeColor(trinityType)
  return {
    borderLeftColor: color,
    '--instance-accent': color
  }
}

// =============================================================================
// LIFECYCLE
// =============================================================================

let refreshInterval: ReturnType<typeof setInterval> | null = null

onMounted(() => {
  // Initial refresh
  refreshInstances()

  // Auto-refresh every 5 seconds when connected
  refreshInterval = setInterval(() => {
    if (isConnected.value && !isLoading.value) {
      // Only auto-refresh if the panel is likely visible
      // In production, this would check visibility
    }
  }, 5000)
})

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
  }
})
</script>

<style scoped>
.instances-panel {
  display: flex;
  flex-direction: column;
  width: 300px;
  height: 100%;
  background-color: var(--flowforge-panel-bg, #252530);
  border-left: 1px solid var(--flowforge-border, #3a3a4a);
  font-family: Inter, system-ui, sans-serif;
  overflow: hidden;
}

/* Header */
.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px;
  border-bottom: 1px solid var(--flowforge-border, #3a3a4a);
  flex-shrink: 0;
}

.header-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  font-weight: 600;
  color: var(--flowforge-text, #e0e0e0);
}

.header-icon {
  width: 16px;
  height: 16px;
  color: var(--flowforge-primary, #6366f1);
}

.total-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 20px;
  height: 18px;
  padding: 0 6px;
  background-color: var(--flowforge-primary, #6366f1);
  border-radius: 9px;
  font-size: 11px;
  font-weight: 600;
  color: white;
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

.action-btn:hover:not(:disabled) {
  background-color: var(--flowforge-hover-bg, #3a3a4a);
  color: var(--flowforge-text, #e0e0e0);
}

.action-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.action-btn i {
  width: 16px;
  height: 16px;
}

/* Search */
.search-container {
  padding: 8px 12px;
  border-bottom: 1px solid var(--flowforge-border, #3a3a4a);
  flex-shrink: 0;
}

.search-input-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

.search-icon {
  position: absolute;
  left: 10px;
  width: 14px;
  height: 14px;
  color: var(--flowforge-text-muted, #666680);
  pointer-events: none;
}

.search-input {
  width: 100%;
  height: 30px;
  padding: 0 28px;
  background-color: var(--flowforge-input-bg, #1a1a24);
  border: 1px solid var(--flowforge-input-border, #3a3a4a);
  border-radius: 6px;
  color: var(--flowforge-input-text, #e0e0e0);
  font-size: 12px;
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
  width: 18px;
  height: 18px;
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

.clear-button i {
  width: 12px;
  height: 12px;
}

/* Status Banner */
.status-banner {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 8px 12px;
  font-size: 11px;
  flex-shrink: 0;
}

.status-banner.disconnected {
  background-color: rgba(239, 68, 68, 0.15);
  color: #f87171;
}

.status-banner i {
  width: 14px;
  height: 14px;
}

/* Content */
.panel-content {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
}

/* Empty State */
.empty-state,
.loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 32px 16px;
  color: var(--flowforge-text-muted, #666680);
}

.empty-icon,
.loading-icon {
  width: 40px;
  height: 40px;
  opacity: 0.5;
}

.empty-text {
  font-size: 13px;
}

.refresh-link {
  background: none;
  border: none;
  color: var(--flowforge-primary, #6366f1);
  font-size: 12px;
  cursor: pointer;
  text-decoration: underline;
  text-underline-offset: 2px;
}

.refresh-link:hover {
  color: var(--flowforge-primary-light, #818cf8);
}

/* Instance Tree */
.instance-tree {
  padding: 4px 0;
}

/* Type Section */
.type-section {
  margin-bottom: 2px;
}

.type-header {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 8px 12px;
  background: transparent;
  border: none;
  color: var(--flowforge-text, #e0e0e0);
  font-size: 12px;
  font-weight: 600;
  text-align: left;
  cursor: pointer;
  transition: background-color 0.15s ease;
}

.type-header:hover {
  background-color: var(--flowforge-hover-bg, #3a3a4a);
}

.expand-icon {
  width: 14px;
  height: 14px;
  color: var(--flowforge-text-muted, #666680);
  flex-shrink: 0;
  transition: transform 0.15s ease;
}

.expand-icon.small {
  width: 12px;
  height: 12px;
}

.type-indicator {
  width: 10px;
  height: 10px;
  border-radius: 3px;
  flex-shrink: 0;
}

.type-name {
  flex: 1;
}

.type-count {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 24px;
  height: 18px;
  padding: 0 6px;
  border-radius: 9px;
  font-size: 11px;
  font-weight: 600;
}

.type-content {
  padding-left: 12px;
}

.empty-type {
  padding: 8px 12px 8px 32px;
  font-size: 11px;
  color: var(--flowforge-text-muted, #666680);
  font-style: italic;
}

/* Component Group */
.component-group {
  margin-bottom: 2px;
}

.group-header {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 6px 12px;
  background: transparent;
  border: none;
  color: var(--flowforge-text-secondary, #a0a0a0);
  font-size: 12px;
  text-align: left;
  cursor: pointer;
  transition: background-color 0.15s ease;
}

.group-header:hover {
  background-color: var(--flowforge-hover-bg, #3a3a4a);
}

.component-name {
  flex: 1;
  color: var(--flowforge-text, #e0e0e0);
  font-weight: 500;
  transition: color 0.15s ease;
}

.component-name:hover {
  color: var(--flowforge-primary, #6366f1);
  text-decoration: underline;
  text-underline-offset: 2px;
}

.instance-count {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 20px;
  height: 16px;
  padding: 0 5px;
  background-color: var(--flowforge-surface, #2a2a3a);
  border-radius: 8px;
  font-size: 10px;
  color: var(--flowforge-text-muted, #666680);
}

/* Instance List */
.instance-list {
  padding-left: 12px;
}

.instance-item {
  margin-bottom: 1px;
}

.instance-header {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 5px 10px;
  background: transparent;
  border: none;
  color: var(--flowforge-text-secondary, #a0a0a0);
  font-size: 11px;
  text-align: left;
  cursor: pointer;
  transition: background-color 0.15s ease;
}

.instance-header:hover {
  background-color: var(--flowforge-hover-bg, #3a3a4a);
}

.instance-id {
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  font-size: 10px;
  color: var(--flowforge-text-muted, #666680);
  flex-shrink: 0;
}

.instance-preview {
  flex: 1;
  font-size: 10px;
  color: var(--flowforge-text-muted, #666680);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
}

.instance-header.expanded .instance-preview {
  display: none;
}

.locate-btn {
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
  opacity: 0;
  transition: all 0.15s ease;
}

.instance-header:hover .locate-btn {
  opacity: 1;
}

.locate-btn:hover {
  background-color: var(--flowforge-primary, #6366f1);
  color: white;
}

.locate-btn i {
  width: 12px;
  height: 12px;
}

/* Instance Data */
.instance-data {
  padding: 8px 10px;
  margin: 0 4px 4px 28px;
  background-color: var(--flowforge-panel-secondary, #1e1e28);
  border-radius: 4px;
  border-left: 2px solid var(--instance-accent, var(--flowforge-primary, #6366f1));
}

.instance-data-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--flowforge-border, #3a3a4a);
}

.data-label {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--instance-accent, var(--flowforge-text-muted, #666680));
}

.copy-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  padding: 0;
  background: transparent;
  border: none;
  border-radius: 4px;
  color: var(--flowforge-text-muted, #666680);
  cursor: pointer;
  transition: all 0.15s ease;
}

.copy-btn:hover {
  background-color: var(--flowforge-hover-bg, #3a3a4a);
  color: var(--instance-accent, var(--flowforge-primary, #6366f1));
}

.copy-btn i {
  width: 14px;
  height: 14px;
}

/* Error Toast */
.error-toast {
  position: absolute;
  bottom: 12px;
  left: 12px;
  right: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  background-color: rgba(239, 68, 68, 0.9);
  border-radius: 6px;
  color: white;
  font-size: 12px;
  box-shadow: var(--flowforge-shadow-lg, 0 10px 15px rgba(0, 0, 0, 0.5));
}

.error-toast i {
  width: 16px;
  height: 16px;
  flex-shrink: 0;
}

/* Animations */
.spin {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

.expand-enter-active,
.expand-leave-active {
  transition: all 0.2s ease;
  overflow: hidden;
}

.expand-enter-from,
.expand-leave-to {
  opacity: 0;
  max-height: 0;
}

.expand-enter-to,
.expand-leave-from {
  opacity: 1;
  max-height: 500px;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

/* Scrollbar */
.panel-content::-webkit-scrollbar {
  width: 6px;
}

.panel-content::-webkit-scrollbar-track {
  background: transparent;
}

.panel-content::-webkit-scrollbar-thumb {
  background-color: var(--flowforge-border-light, #4a4a5a);
  border-radius: 3px;
}

.panel-content::-webkit-scrollbar-thumb:hover {
  background-color: var(--flowforge-primary, #6366f1);
}
</style>
