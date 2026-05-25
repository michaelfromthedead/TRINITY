<template>
  <div class="tree-item-container">
    <!-- Item Row -->
    <div
      class="tree-item"
      :class="{
        'is-selected': isSelected,
        'is-directory': entry.isDir
      }"
      :style="{ paddingLeft: `${12 + depth * 16}px` }"
      @click="handleClick"
      @dblclick="handleDoubleClick"
    >
      <!-- Expand/Collapse Arrow (directories only) -->
      <span v-if="entry.isDir" class="expand-arrow" @click.stop="handleToggle">
        <i :class="['pi', isExpanded ? 'pi-chevron-down' : 'pi-chevron-right']" />
      </span>
      <span v-else class="expand-arrow-placeholder" />

      <!-- File/Folder Icon -->
      <i :class="['item-icon', iconClass]" />

      <!-- Name -->
      <span class="item-name" :title="entry.path">{{ entry.name }}</span>
    </div>

    <!-- Children (for expanded directories) -->
    <div v-if="entry.isDir && isExpanded && sortedChildren.length > 0" class="tree-children">
      <FileTreeItem
        v-for="child in sortedChildren"
        :key="child.path"
        :entry="child"
        :depth="depth + 1"
        :selected-path="selectedPath"
        :expanded-folders="expandedFolders"
        :filter-python-only="filterPythonOnly"
        @select="emit('select', $event)"
        @toggle="emit('toggle', $event)"
        @open="emit('open', $event)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, watch, ref } from 'vue'
import { listDirectory, type FileEntry } from '@/bridge/files'

// =============================================================================
// Props & Emits
// =============================================================================

const props = withDefaults(defineProps<{
  entry: FileEntry
  depth: number
  selectedPath: string | null
  expandedFolders: Set<string>
  filterPythonOnly?: boolean
}>(), {
  filterPythonOnly: false
})

const emit = defineEmits<{
  (e: 'select', path: string): void
  (e: 'toggle', path: string): void
  (e: 'open', path: string): void
}>()

// =============================================================================
// State
// =============================================================================

const children = ref<FileEntry[]>([])
const isLoadingChildren = ref(false)

// =============================================================================
// Computed
// =============================================================================

const isSelected = computed(() => props.selectedPath === props.entry.path)

const isExpanded = computed(() => props.expandedFolders.has(props.entry.path))

const sortedChildren = computed(() => {
  // Filter entries based on filterPythonOnly
  let filtered = children.value
  if (props.filterPythonOnly) {
    filtered = children.value.filter(c => c.isDir || c.name.endsWith('.py'))
  }

  // Directories first, then files, alphabetically
  const dirs = filtered.filter(c => c.isDir).sort((a, b) => a.name.localeCompare(b.name))
  const files = filtered.filter(c => !c.isDir).sort((a, b) => a.name.localeCompare(b.name))
  return [...dirs, ...files]
})

const iconClass = computed(() => {
  if (props.entry.isDir) {
    return isExpanded.value ? 'pi pi-folder-open' : 'pi pi-folder'
  }

  // File icon based on extension
  const ext = props.entry.name.split('.').pop()?.toLowerCase()

  switch (ext) {
    case 'py':
      return 'pi pi-code'
    case 'json':
    case 'flowforge':
      return 'pi pi-file'
    case 'md':
    case 'txt':
      return 'pi pi-file-edit'
    case 'ts':
    case 'js':
    case 'tsx':
    case 'jsx':
      return 'pi pi-code'
    case 'vue':
      return 'pi pi-code'
    case 'css':
    case 'scss':
    case 'less':
      return 'pi pi-palette'
    case 'png':
    case 'jpg':
    case 'jpeg':
    case 'gif':
    case 'svg':
      return 'pi pi-image'
    default:
      return 'pi pi-file'
  }
})

// =============================================================================
// Watch for expansion changes
// =============================================================================

watch(
  () => isExpanded.value,
  async (expanded) => {
    if (expanded && props.entry.isDir && children.value.length === 0) {
      await loadChildren()
    }
  },
  { immediate: true }
)

// =============================================================================
// Methods
// =============================================================================

async function loadChildren() {
  if (!props.entry.isDir || isLoadingChildren.value) return

  isLoadingChildren.value = true
  try {
    children.value = await listDirectory(props.entry.path)
  } catch (error) {
    console.error('[FileTreeItem] Failed to load children:', error)
    children.value = []
  } finally {
    isLoadingChildren.value = false
  }
}

function handleClick() {
  emit('select', props.entry.path)
}

function handleDoubleClick() {
  if (props.entry.isDir) {
    emit('toggle', props.entry.path)
  } else {
    emit('open', props.entry.path)
  }
}

function handleToggle() {
  emit('toggle', props.entry.path)
}
</script>

<style scoped>
.tree-item-container {
  user-select: none;
}

.tree-item {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 12px;
  cursor: pointer;
  transition: background-color 0.1s ease;
}

.tree-item:hover {
  background-color: var(--hover-bg, #2a2d2e);
}

.tree-item.is-selected {
  background-color: var(--selected-bg, #094771);
}

.tree-item.is-selected:hover {
  background-color: var(--selected-bg-hover, #0a5a8a);
}

.expand-arrow {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  color: var(--text-muted, #888);
  flex-shrink: 0;
}

.expand-arrow i {
  font-size: 10px;
  transition: transform 0.15s ease;
}

.expand-arrow-placeholder {
  width: 16px;
  flex-shrink: 0;
}

.item-icon {
  font-size: 14px;
  color: var(--text-muted, #888);
  flex-shrink: 0;
  width: 16px;
  text-align: center;
}

.tree-item.is-directory .item-icon {
  color: var(--folder-color, #dcb67a);
}

.item-name {
  flex: 1;
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--text-primary, #cccccc);
}

.tree-children {
  /* Children are indented via their paddingLeft style */
}
</style>
