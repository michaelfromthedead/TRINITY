<template>
  <div class="file-explorer">
    <!-- Breadcrumb Navigation -->
    <div class="explorer-breadcrumbs">
      <div class="breadcrumb-path">
        <template v-for="(segment, index) in pathSegments" :key="segment.path">
          <span
            class="breadcrumb-segment"
            :class="{ 'breadcrumb-current': index === pathSegments.length - 1 }"
            :title="segment.path"
            @click="handleNavigateToSegment(segment.path)"
          >
            {{ segment.name }}
          </span>
          <i
            v-if="index < pathSegments.length - 1"
            class="pi pi-chevron-right breadcrumb-separator"
          />
        </template>
      </div>
      <div class="breadcrumb-actions">
        <button
          class="action-btn"
          title="Navigate up"
          :disabled="!canNavigateUp"
          @click="handleNavigateUp"
        >
          <i class="pi pi-arrow-up" />
        </button>
        <button
          class="action-btn"
          title="Refresh"
          :disabled="fileExplorerStore.isLoading"
          @click="handleRefresh"
        >
          <i class="pi pi-refresh" :class="{ 'pi-spin': fileExplorerStore.isLoading }" />
        </button>
        <button
          class="action-btn"
          title="Collapse All"
          @click="fileExplorerStore.collapseAll()"
        >
          <i class="pi pi-minus" />
        </button>
      </div>
    </div>

    <!-- Loading State -->
    <div v-if="fileExplorerStore.isLoading && !fileExplorerStore.hasContents" class="explorer-loading">
      <i class="pi pi-spinner pi-spin" />
      <span>Loading files...</span>
    </div>

    <!-- Error State -->
    <div v-else-if="fileExplorerStore.error" class="explorer-error">
      <i class="pi pi-exclamation-triangle" />
      <span>{{ fileExplorerStore.error }}</span>
      <button class="retry-btn" @click="handleRefresh">Retry</button>
    </div>

    <!-- No Workspace State -->
    <div v-else-if="!fileExplorerStore.hasWorkspace" class="explorer-empty">
      <i class="pi pi-folder-open" />
      <span>No workspace open</span>
      <p class="hint-text">Open a folder or file to get started</p>
    </div>

    <!-- Empty Directory State (no Python files) -->
    <div v-else-if="filteredContents.length === 0" class="explorer-empty">
      <i class="pi pi-folder" />
      <span>No Python files found</span>
    </div>

    <!-- File Tree -->
    <div v-else class="file-tree">
      <FileTreeItem
        v-for="entry in filteredContents"
        :key="entry.path"
        :entry="entry"
        :depth="0"
        :selected-path="fileExplorerStore.selectedFile"
        :expanded-folders="fileExplorerStore.expandedFolders"
        :filter-python-only="true"
        @select="handleSelect"
        @toggle="handleToggle"
        @open="handleOpen"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useFileExplorerStore } from '@/stores/fileExplorerStore'
import FileTreeItem from './FileTreeItem.vue'
import type { FileEntry } from '@/bridge/files'

// Store
const fileExplorerStore = useFileExplorerStore()

// =============================================================================
// Computed
// =============================================================================

/**
 * Path segments for breadcrumb navigation.
 */
const pathSegments = computed(() => {
  const path = fileExplorerStore.currentPath
  if (!path) return []

  const segments: { name: string; path: string }[] = []
  const parts = path.split('/').filter(Boolean)

  // Handle root
  if (path.startsWith('/')) {
    segments.push({ name: '/', path: '/' })
  }

  // Build path segments
  let accumulatedPath = ''
  for (const part of parts) {
    accumulatedPath = accumulatedPath ? `${accumulatedPath}/${part}` : `/${part}`
    segments.push({ name: part, path: accumulatedPath })
  }

  return segments
})

/**
 * Whether we can navigate up from current location.
 */
const canNavigateUp = computed(() => {
  const currentPath = fileExplorerStore.currentPath
  const workspaceRoot = fileExplorerStore.workspaceRoot

  if (!currentPath) return false
  if (workspaceRoot && currentPath === workspaceRoot) return false
  if (currentPath === '/') return false

  return true
})

/**
 * Filter contents to show only Python files and folders.
 * Sorted: folders first, then files alphabetically.
 */
const filteredContents = computed(() => {
  const contents = fileExplorerStore.contents

  // Filter to Python files and directories
  const filtered = contents.filter((entry: FileEntry) => {
    if (entry.isDir) return true
    return entry.name.endsWith('.py')
  })

  // Sort: directories first, then files, alphabetically
  return [...filtered].sort((a, b) => {
    if (a.isDir && !b.isDir) return -1
    if (!a.isDir && b.isDir) return 1
    return a.name.localeCompare(b.name, undefined, { sensitivity: 'base' })
  })
})

// =============================================================================
// Event Handlers
// =============================================================================

function handleRefresh() {
  fileExplorerStore.refresh()
}

function handleSelect(path: string) {
  fileExplorerStore.selectFile(path)
}

function handleToggle(path: string) {
  fileExplorerStore.toggleFolder(path)
}

function handleOpen(path: string) {
  // Only open Python files
  if (path.endsWith('.py')) {
    fileExplorerStore.openFile(path)
  }
}

function handleNavigateUp() {
  fileExplorerStore.navigateUp()
}

function handleNavigateToSegment(path: string) {
  fileExplorerStore.navigateTo(path)
}

// =============================================================================
// Lifecycle
// =============================================================================

onMounted(() => {
  // Initialize will be called by App.vue
})
</script>

<style scoped>
.file-explorer {
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: var(--panel-bg, #252526);
  color: var(--text-primary, #cccccc);
  overflow: hidden;
}

/* Breadcrumb Navigation */
.explorer-breadcrumbs {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  border-bottom: 1px solid var(--border-color, #3d3d3d);
  min-height: 36px;
  gap: 8px;
  flex-shrink: 0;
}

.breadcrumb-path {
  display: flex;
  align-items: center;
  flex-wrap: nowrap;
  gap: 2px;
  overflow: hidden;
  font-size: 12px;
  flex: 1;
  min-width: 0;
}

.breadcrumb-segment {
  color: var(--text-muted, #888);
  cursor: pointer;
  padding: 2px 4px;
  border-radius: 3px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 80px;
  flex-shrink: 0;
}

.breadcrumb-segment:last-child {
  flex-shrink: 1;
  max-width: none;
}

.breadcrumb-segment:hover {
  background-color: var(--hover-bg, #2a2d2e);
  color: var(--text-primary, #cccccc);
}

.breadcrumb-current {
  color: var(--text-primary, #cccccc);
  font-weight: 500;
  cursor: default;
}

.breadcrumb-current:hover {
  background-color: transparent;
}

.breadcrumb-separator {
  font-size: 8px;
  color: var(--text-muted, #666);
  flex-shrink: 0;
}

.breadcrumb-actions {
  display: flex;
  gap: 4px;
  flex-shrink: 0;
}

.action-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  padding: 0;
  background: transparent;
  border: none;
  border-radius: 4px;
  color: var(--text-muted, #888);
  cursor: pointer;
  transition: all 0.15s ease;
}

.action-btn:hover:not(:disabled) {
  background-color: var(--hover-bg, #2a2d2e);
  color: var(--text-primary, #cccccc);
}

.action-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.action-btn i {
  font-size: 12px;
}

.explorer-loading,
.explorer-error,
.explorer-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 32px 16px;
  gap: 12px;
  color: var(--text-muted, #888);
  text-align: center;
}

.explorer-loading i,
.explorer-error i,
.explorer-empty i {
  font-size: 32px;
}

.explorer-error {
  color: var(--warning-color, #f59e0b);
}

.explorer-error i {
  color: var(--warning-color, #f59e0b);
}

.hint-text {
  font-size: 12px;
  margin: 0;
  opacity: 0.7;
}

.retry-btn {
  margin-top: 8px;
  padding: 6px 12px;
  border: 1px solid var(--border-color, #3d3d3d);
  border-radius: 4px;
  background: transparent;
  color: var(--text-primary, #cccccc);
  cursor: pointer;
  font-size: 12px;
  transition: all 0.15s ease;
}

.retry-btn:hover {
  background-color: var(--hover-bg, #2a2d2e);
  border-color: var(--text-muted, #888);
}

.file-tree {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 4px 0;
}

/* Scrollbar styling */
.file-tree::-webkit-scrollbar {
  width: 8px;
}

.file-tree::-webkit-scrollbar-track {
  background: transparent;
}

.file-tree::-webkit-scrollbar-thumb {
  background-color: var(--scrollbar-thumb, #4a4a4a);
  border-radius: 4px;
}

.file-tree::-webkit-scrollbar-thumb:hover {
  background-color: var(--scrollbar-thumb-hover, #5a5a5a);
}
</style>
