<template>
  <Teleport to="body">
    <Transition name="context-menu">
      <div
        v-if="isVisible && node"
        ref="menuRef"
        class="node-context-menu"
        :style="menuStyle"
        @contextmenu.prevent
      >
        <div class="menu-header" :style="headerStyle">
          <span class="menu-title">{{ node.title || 'Node' }}</span>
          <span class="menu-type-badge" :style="badgeStyle">{{ nodeType }}</span>
        </div>
        <div class="menu-content">
          <!-- Add Field / Add Method -->
          <button
            class="menu-item"
            :style="getItemStyle()"
            @click="handleAddField"
          >
            <div class="menu-item-icon" :style="iconStyle">
              <i class="icon-[lucide--plus]" />
            </div>
            <div class="menu-item-content">
              <span class="menu-label">{{ isSystemNode ? 'Add Method' : 'Add Field' }}</span>
            </div>
          </button>

          <div class="menu-separator" />

          <!-- Rename -->
          <button
            class="menu-item"
            :style="getItemStyle()"
            @click="handleRename"
          >
            <div class="menu-item-icon" :style="iconStyle">
              <i class="icon-[lucide--edit-2]" />
            </div>
            <div class="menu-item-content">
              <span class="menu-label">Rename</span>
            </div>
            <kbd class="menu-shortcut">F2</kbd>
          </button>

          <!-- Delete -->
          <button
            class="menu-item menu-item-danger"
            @click="handleDelete"
          >
            <div class="menu-item-icon menu-item-icon-danger">
              <i class="icon-[lucide--trash-2]" />
            </div>
            <div class="menu-item-content">
              <span class="menu-label">Delete</span>
            </div>
            <kbd class="menu-shortcut">Del</kbd>
          </button>

          <div class="menu-separator" />

          <!-- View Source -->
          <button
            class="menu-item"
            :style="getItemStyle()"
            :disabled="!hasSourceFile"
            @click="handleViewSource"
          >
            <div class="menu-item-icon" :style="hasSourceFile ? iconStyle : {}">
              <i class="icon-[lucide--code]" />
            </div>
            <div class="menu-item-content">
              <span class="menu-label">View Source</span>
            </div>
          </button>

          <!-- Open in Editor -->
          <button
            class="menu-item"
            :style="getItemStyle()"
            :disabled="!hasSourceFile"
            @click="handleOpenInEditor"
          >
            <div class="menu-item-icon" :style="hasSourceFile ? iconStyle : {}">
              <i class="icon-[lucide--external-link]" />
            </div>
            <div class="menu-item-content">
              <span class="menu-label">Open in Editor</span>
            </div>
          </button>
        </div>

        <div class="menu-footer">
          <span class="footer-hint">
            <kbd>Esc</kbd> to close
          </span>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { computed, ref, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { TRINITY_COLORS } from '@/nodes/nodeTheme'
import { UI_CONFIG } from '@/config/flowforge.config'
import type { TrinityNodeType } from '@/composables/useNodeEditing'
import type { GraphNode } from '@/stores/graphStore'

// =============================================================================
// TYPES
// =============================================================================

export interface NodeContextMenuProps {
  /** Whether the menu is visible */
  isVisible: boolean
  /** Position to display the menu (screen coordinates) */
  position: { x: number; y: number }
  /** The selected node */
  node: GraphNode | null
  /** The type of the node */
  nodeType: TrinityNodeType
}

// =============================================================================
// PROPS & EMITS
// =============================================================================

const props = defineProps<NodeContextMenuProps>()

const emit = defineEmits<{
  (e: 'close'): void
  (e: 'add-field'): void
  (e: 'add-method'): void
  (e: 'rename'): void
  (e: 'delete'): void
  (e: 'view-source'): void
  (e: 'open-in-editor'): void
}>()

// =============================================================================
// REFS
// =============================================================================

const menuRef = ref<HTMLElement | null>(null)

// =============================================================================
// COMPUTED
// =============================================================================

/**
 * Check if the node is a system node
 */
const isSystemNode = computed(() => props.nodeType === 'system')

/**
 * Check if the node has a source file
 */
const hasSourceFile = computed(() => {
  if (!props.node?.properties) return false
  const sourceFile = props.node.properties['sourceFile'] as string | undefined
  return !!sourceFile && sourceFile.trim().length > 0
})

/**
 * Get the color palette for the current node type
 */
const nodeColors = computed(() => TRINITY_COLORS[props.nodeType])

/**
 * Compute menu position style, ensuring it stays within viewport
 */
const menuStyle = computed(() => {
  const { x, y } = props.position
  const menuWidth = UI_CONFIG.contextMenu.width
  const menuHeight = 320 // Estimated height for node context menu
  const edgeMargin = UI_CONFIG.contextMenu.edgeMargin

  // Get viewport dimensions
  const viewportWidth = window.innerWidth
  const viewportHeight = window.innerHeight

  // Calculate position, keeping menu within viewport
  let left = x
  let top = y

  // Adjust horizontal position if menu would overflow right edge
  if (left + menuWidth > viewportWidth - edgeMargin) {
    left = viewportWidth - menuWidth - edgeMargin
  }

  // Adjust vertical position if menu would overflow bottom edge
  if (top + menuHeight > viewportHeight - edgeMargin) {
    top = viewportHeight - menuHeight - edgeMargin
  }

  // Ensure minimum positions
  left = Math.max(edgeMargin, left)
  top = Math.max(edgeMargin, top)

  return {
    left: `${left}px`,
    top: `${top}px`
  }
})

/**
 * Style for the menu header based on node type
 */
const headerStyle = computed(() => {
  const color = nodeColors.value.primary
  return {
    borderBottomColor: `${color}40`
  }
})

/**
 * Style for the type badge
 */
const badgeStyle = computed(() => {
  const color = nodeColors.value.primary
  return {
    backgroundColor: `${color}20`,
    color: color
  }
})

/**
 * Style for menu item icons
 */
const iconStyle = computed(() => {
  const color = nodeColors.value.primary
  return {
    '--icon-color': color,
    '--icon-bg': `${color}20`
  }
})

// =============================================================================
// METHODS
// =============================================================================

/**
 * Get style for menu item hover effect
 */
function getItemStyle(): Record<string, string> {
  const color = nodeColors.value.primary
  return {
    '--item-hover-bg': `${color}15`,
    '--item-active-bg': `${color}25`
  }
}

/**
 * Handle add field action
 */
function handleAddField() {
  if (isSystemNode.value) {
    emit('add-method')
  } else {
    emit('add-field')
  }
  emit('close')
}

/**
 * Handle rename action
 */
function handleRename() {
  emit('rename')
  emit('close')
}

/**
 * Handle delete action
 */
function handleDelete() {
  emit('delete')
  emit('close')
}

/**
 * Handle view source action
 */
function handleViewSource() {
  if (!hasSourceFile.value) return
  emit('view-source')
  emit('close')
}

/**
 * Handle open in editor action
 */
function handleOpenInEditor() {
  if (!hasSourceFile.value) return
  emit('open-in-editor')
  emit('close')
}

/**
 * Close the menu
 */
function handleClose() {
  emit('close')
}

// =============================================================================
// EVENT HANDLERS
// =============================================================================

function handleClickOutside(event: MouseEvent) {
  if (!props.isVisible) return

  const target = event.target as Node
  if (menuRef.value && !menuRef.value.contains(target)) {
    handleClose()
  }
}

function handleKeydown(event: KeyboardEvent) {
  if (!props.isVisible) return

  if (event.key === 'Escape') {
    event.preventDefault()
    handleClose()
  }

  // Keyboard shortcuts
  if (event.key === 'Delete') {
    event.preventDefault()
    handleDelete()
  } else if (event.key === 'F2') {
    event.preventDefault()
    handleRename()
  }
}

// =============================================================================
// LIFECYCLE
// =============================================================================

watch(() => props.isVisible, (visible) => {
  if (visible) {
    // Add event listeners when menu opens
    nextTick(() => {
      document.addEventListener('click', handleClickOutside, true)
      document.addEventListener('keydown', handleKeydown, true)
    })
  } else {
    // Remove event listeners when menu closes
    document.removeEventListener('click', handleClickOutside, true)
    document.removeEventListener('keydown', handleKeydown, true)
  }
})

onMounted(() => {
  if (props.isVisible) {
    document.addEventListener('click', handleClickOutside, true)
    document.addEventListener('keydown', handleKeydown, true)
  }
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside, true)
  document.removeEventListener('keydown', handleKeydown, true)
})
</script>

<style scoped>
.node-context-menu {
  position: fixed;
  z-index: 10000;
  min-width: 220px;
  max-width: 280px;
  background-color: var(--p-surface-0, #ffffff);
  border: 1px solid var(--p-surface-border, #e5e7eb);
  border-radius: 0.625rem;
  box-shadow:
    0 20px 25px -5px rgba(0, 0, 0, 0.1),
    0 8px 10px -6px rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

/* Dark mode support */
:root[data-theme="dark"] .node-context-menu,
.dark .node-context-menu {
  background-color: var(--p-surface-800, #1f2937);
  border-color: var(--p-surface-700, #374151);
}

.menu-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.625rem 0.875rem;
  border-bottom: 1px solid var(--p-surface-border, #e5e7eb);
}

:root[data-theme="dark"] .menu-header,
.dark .menu-header {
  border-color: var(--p-surface-700, #374151);
}

.menu-title {
  font-size: 0.8125rem;
  font-weight: 600;
  color: var(--p-text-color, #1f2937);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 140px;
}

.menu-type-badge {
  font-size: 0.625rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.025em;
  padding: 0.125rem 0.375rem;
  border-radius: 0.25rem;
  flex-shrink: 0;
}

.menu-content {
  padding: 0.5rem;
}

.menu-separator {
  height: 1px;
  margin: 0.375rem 0.5rem;
  background-color: var(--p-surface-border, #e5e7eb);
}

:root[data-theme="dark"] .menu-separator,
.dark .menu-separator {
  background-color: var(--p-surface-700, #374151);
}

.menu-item {
  display: flex;
  align-items: center;
  gap: 0.625rem;
  width: 100%;
  padding: 0.5rem 0.625rem;
  border: none;
  border-radius: 0.375rem;
  background: transparent;
  color: var(--p-text-color, #1f2937);
  font-size: 0.8125rem;
  text-align: left;
  cursor: pointer;
  transition: background-color 0.15s ease;
}

.menu-item:hover:not(:disabled) {
  background-color: var(--item-hover-bg, #f3f4f6);
}

.menu-item:active:not(:disabled) {
  background-color: var(--item-active-bg, #e5e7eb);
}

.menu-item:focus {
  outline: none;
}

.menu-item:focus:not(:disabled) {
  background-color: var(--item-hover-bg, #f3f4f6);
}

.menu-item:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.menu-item-danger:hover:not(:disabled) {
  background-color: rgba(239, 68, 68, 0.1);
}

.menu-item-danger:active:not(:disabled) {
  background-color: rgba(239, 68, 68, 0.2);
}

.menu-item-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 1.625rem;
  height: 1.625rem;
  border-radius: 0.3125rem;
  background-color: var(--icon-bg, #f3f4f6);
  color: var(--icon-color, #6b7280);
  font-size: 0.875rem;
  flex-shrink: 0;
}

.menu-item-icon-danger {
  background-color: rgba(239, 68, 68, 0.15);
  color: #ef4444;
}

.menu-item-content {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-width: 0;
}

.menu-label {
  font-weight: 500;
  color: var(--p-text-color, #1f2937);
}

.menu-item-danger .menu-label {
  color: #ef4444;
}

.menu-shortcut {
  font-size: 0.625rem;
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  color: var(--p-text-muted-color, #9ca3af);
  background-color: var(--p-surface-100, #f3f4f6);
  padding: 0.125rem 0.3125rem;
  border-radius: 0.1875rem;
  flex-shrink: 0;
}

:root[data-theme="dark"] .menu-shortcut,
.dark .menu-shortcut {
  background-color: var(--p-surface-700, #374151);
}

.menu-footer {
  padding: 0.5rem 0.875rem;
  border-top: 1px solid var(--p-surface-border, #e5e7eb);
  background-color: var(--p-surface-50, #f9fafb);
}

:root[data-theme="dark"] .menu-footer,
.dark .menu-footer {
  border-color: var(--p-surface-700, #374151);
  background-color: var(--p-surface-900, #111827);
}

.footer-hint {
  font-size: 0.6875rem;
  color: var(--p-text-muted-color, #9ca3af);
}

.footer-hint kbd {
  display: inline-block;
  padding: 0.0625rem 0.25rem;
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  font-size: 0.625rem;
  background-color: var(--p-surface-200, #e5e7eb);
  border-radius: 0.1875rem;
  margin-right: 0.25rem;
}

:root[data-theme="dark"] .footer-hint kbd,
.dark .footer-hint kbd {
  background-color: var(--p-surface-700, #374151);
}

/* Transition animations */
.context-menu-enter-active,
.context-menu-leave-active {
  transition: opacity 0.15s ease, transform 0.15s ease;
}

.context-menu-enter-from,
.context-menu-leave-to {
  opacity: 0;
  transform: scale(0.95) translateY(-4px);
}

.context-menu-enter-to,
.context-menu-leave-from {
  opacity: 1;
  transform: scale(1) translateY(0);
}
</style>
