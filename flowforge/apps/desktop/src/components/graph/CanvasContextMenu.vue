<template>
  <Teleport to="body">
    <Transition name="context-menu">
      <div
        v-if="isVisible"
        ref="menuRef"
        class="canvas-context-menu"
        :style="menuStyle"
        @contextmenu.prevent
      >
        <div class="menu-header">
          <span class="menu-title">Add Node</span>
        </div>
        <div class="menu-content">
          <!-- Add Component -->
          <button
            class="menu-item"
            :style="getItemStyle('component')"
            @click="handleAddNode('component')"
          >
            <div class="menu-item-icon" :style="getIconStyle('component')">
              <i class="icon-[lucide--box]" />
            </div>
            <div class="menu-item-content">
              <span class="menu-label">Component</span>
              <span class="menu-description">Data container for entities</span>
            </div>
            <span class="menu-decorator">@component</span>
          </button>

          <!-- Add System -->
          <button
            class="menu-item"
            :style="getItemStyle('system')"
            @click="handleAddNode('system')"
          >
            <div class="menu-item-icon" :style="getIconStyle('system')">
              <i class="icon-[lucide--cpu]" />
            </div>
            <div class="menu-item-content">
              <span class="menu-label">System</span>
              <span class="menu-description">Logic that processes entities</span>
            </div>
            <span class="menu-decorator">@system</span>
          </button>

          <!-- Add Resource -->
          <button
            class="menu-item"
            :style="getItemStyle('resource')"
            @click="handleAddNode('resource')"
          >
            <div class="menu-item-icon" :style="getIconStyle('resource')">
              <i class="icon-[lucide--database]" />
            </div>
            <div class="menu-item-content">
              <span class="menu-label">Resource</span>
              <span class="menu-description">Singleton shared data</span>
            </div>
            <span class="menu-decorator">@resource</span>
          </button>

          <!-- Add Event -->
          <button
            class="menu-item"
            :style="getItemStyle('event')"
            @click="handleAddNode('event')"
          >
            <div class="menu-item-icon" :style="getIconStyle('event')">
              <i class="icon-[lucide--zap]" />
            </div>
            <div class="menu-item-content">
              <span class="menu-label">Event</span>
              <span class="menu-description">Trigger with payload data</span>
            </div>
            <span class="menu-decorator">@event</span>
          </button>
        </div>

        <div class="menu-footer">
          <span class="footer-hint">
            <kbd>Esc</kbd> to cancel
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

// =============================================================================
// TYPES
// =============================================================================

export interface CanvasPosition {
  /** Screen X coordinate */
  x: number
  /** Screen Y coordinate */
  y: number
  /** Canvas X coordinate (for node placement) */
  canvasX?: number
  /** Canvas Y coordinate (for node placement) */
  canvasY?: number
}

export interface CanvasContextMenuProps {
  /** Whether the menu is visible */
  isVisible: boolean
  /** Position to display the menu (screen coordinates) */
  position: CanvasPosition
}

// =============================================================================
// PROPS & EMITS
// =============================================================================

const props = defineProps<CanvasContextMenuProps>()

const emit = defineEmits<{
  (e: 'close'): void
  (e: 'add-node', type: TrinityNodeType, position: [number, number]): void
}>()

// =============================================================================
// REFS
// =============================================================================

const menuRef = ref<HTMLElement | null>(null)

// =============================================================================
// COMPUTED
// =============================================================================

/**
 * Compute menu position style, ensuring it stays within viewport
 */
const menuStyle = computed(() => {
  const { x, y } = props.position
  const menuWidth = UI_CONFIG.contextMenu.width
  const menuHeight = UI_CONFIG.contextMenu.height
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

// =============================================================================
// METHODS
// =============================================================================

/**
 * Get style for menu item hover effect
 */
function getItemStyle(type: TrinityNodeType): Record<string, string> {
  const color = TRINITY_COLORS[type].primary
  return {
    '--item-hover-bg': `${color}15`,
    '--item-active-bg': `${color}25`
  }
}

/**
 * Get style for menu item icon
 */
function getIconStyle(type: TrinityNodeType): Record<string, string> {
  const color = TRINITY_COLORS[type].primary
  const bgColor = `${color}20`
  return {
    '--icon-color': color,
    '--icon-bg': bgColor
  }
}

/**
 * Handle adding a new node
 */
function handleAddNode(type: TrinityNodeType) {
  const position: [number, number] = [
    props.position.canvasX ?? props.position.x,
    props.position.canvasY ?? props.position.y
  ]
  emit('add-node', type, position)
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

  // Quick add shortcuts
  if (event.key === 'c' || event.key === 'C') {
    event.preventDefault()
    handleAddNode('component')
  } else if (event.key === 's' || event.key === 'S') {
    event.preventDefault()
    handleAddNode('system')
  } else if (event.key === 'r' || event.key === 'R') {
    event.preventDefault()
    handleAddNode('resource')
  } else if (event.key === 'e' || event.key === 'E') {
    event.preventDefault()
    handleAddNode('event')
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
.canvas-context-menu {
  position: fixed;
  z-index: 10000;
  min-width: 260px;
  max-width: 320px;
  background-color: var(--p-surface-0, #ffffff);
  border: 1px solid var(--p-surface-border, #e5e7eb);
  border-radius: 0.625rem;
  box-shadow:
    0 20px 25px -5px rgba(0, 0, 0, 0.1),
    0 8px 10px -6px rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

/* Dark mode support */
:root[data-theme="dark"] .canvas-context-menu,
.dark .canvas-context-menu {
  background-color: var(--p-surface-800, #1f2937);
  border-color: var(--p-surface-700, #374151);
}

.menu-header {
  padding: 0.625rem 0.875rem;
  border-bottom: 1px solid var(--p-surface-border, #e5e7eb);
}

:root[data-theme="dark"] .menu-header,
.dark .menu-header {
  border-color: var(--p-surface-700, #374151);
}

.menu-title {
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.025em;
  color: var(--p-text-muted-color, #9ca3af);
}

.menu-content {
  padding: 0.5rem;
}

.menu-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  width: 100%;
  padding: 0.625rem 0.75rem;
  border: none;
  border-radius: 0.5rem;
  background: transparent;
  color: var(--p-text-color, #1f2937);
  font-size: 0.875rem;
  text-align: left;
  cursor: pointer;
  transition: background-color 0.15s ease;
}

.menu-item:hover {
  background-color: var(--item-hover-bg, #f3f4f6);
}

.menu-item:active {
  background-color: var(--item-active-bg, #e5e7eb);
}

.menu-item:focus {
  outline: none;
  background-color: var(--item-hover-bg, #f3f4f6);
}

.menu-item-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2rem;
  height: 2rem;
  border-radius: 0.375rem;
  background-color: var(--icon-bg, #f3f4f6);
  color: var(--icon-color, #6b7280);
  font-size: 1rem;
  flex-shrink: 0;
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

.menu-description {
  font-size: 0.6875rem;
  color: var(--p-text-muted-color, #9ca3af);
  margin-top: 0.125rem;
}

.menu-decorator {
  font-size: 0.6875rem;
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  color: var(--p-text-muted-color, #9ca3af);
  background-color: var(--p-surface-100, #f3f4f6);
  padding: 0.125rem 0.375rem;
  border-radius: 0.25rem;
  flex-shrink: 0;
}

:root[data-theme="dark"] .menu-decorator,
.dark .menu-decorator {
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
