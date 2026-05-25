<script setup lang="ts">
import { ref } from 'vue'
import { useRecentFiles } from '@/composables/useRecentFiles'

const emit = defineEmits<{
  (e: 'open-file', path: string): void
}>()

const { recentFiles, clearRecentFiles } = useRecentFiles()

const isOpen = ref(false)

function toggle() {
  isOpen.value = !isOpen.value
}

function close() {
  isOpen.value = false
}

function handleFileClick(path: string) {
  emit('open-file', path)
  close()
}

function handleClear() {
  clearRecentFiles()
  close()
}

/**
 * Extract just the filename from a full path for display.
 */
function fileName(path: string): string {
  return path.split(/[\\/]/).pop() || path
}
</script>

<template>
  <div class="recent-files-menu" @mouseleave="close">
    <button
      class="header-btn"
      title="Recent Files"
      @click="toggle"
    >
      <i class="pi pi-clock" />
    </button>
    <div v-if="isOpen" class="dropdown">
      <template v-if="recentFiles.length > 0">
        <button
          v-for="path in recentFiles"
          :key="path"
          class="dropdown-item"
          :title="path"
          @click="handleFileClick(path)"
        >
          <i class="pi pi-file" />
          <span class="item-label">{{ fileName(path) }}</span>
        </button>
        <div class="dropdown-divider" />
        <button class="dropdown-item clear-item" @click="handleClear">
          <i class="pi pi-trash" />
          <span class="item-label">Clear Recent</span>
        </button>
      </template>
      <div v-else class="dropdown-empty">
        No recent files
      </div>
    </div>
  </div>
</template>

<style scoped>
.recent-files-menu {
  position: relative;
}

.header-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  padding: 0;
  background: transparent;
  border: none;
  border-radius: 4px;
  color: var(--text-secondary, #cccccc);
  cursor: pointer;
  transition: all 0.15s ease;
}

.header-btn:hover {
  background-color: var(--hover-bg, #3a3a3a);
  color: var(--text-primary, #ffffff);
}

.header-btn i {
  font-size: 14px;
}

.dropdown {
  position: absolute;
  top: 100%;
  right: 0;
  margin-top: 4px;
  min-width: 240px;
  max-width: 360px;
  background-color: var(--panel-bg, #2d2d2d);
  border: 1px solid var(--border-color, #3d3d3d);
  border-radius: 6px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  z-index: 1000;
  overflow: hidden;
}

.dropdown-item {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 8px 12px;
  font-size: 12px;
  color: var(--text-primary, #ffffff);
  background: transparent;
  border: none;
  cursor: pointer;
  text-align: left;
  transition: background-color 0.1s ease;
}

.dropdown-item:hover {
  background-color: var(--hover-bg, #3a3a3a);
}

.dropdown-item i {
  font-size: 12px;
  color: var(--text-secondary, #cccccc);
  flex-shrink: 0;
}

.item-label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.dropdown-divider {
  height: 1px;
  background-color: var(--border-color, #3d3d3d);
  margin: 4px 0;
}

.clear-item {
  color: var(--text-secondary, #cccccc);
}

.dropdown-empty {
  padding: 16px 12px;
  font-size: 12px;
  color: var(--text-muted, #888);
  text-align: center;
}
</style>
