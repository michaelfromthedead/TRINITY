<template>
  <div
    class="node-palette-item"
    :class="{ 'is-dragging': isDragging }"
    :style="{ '--node-color': nodeColor }"
    draggable="true"
    @dragstart="handleDragStart"
    @dragend="handleDragEnd"
    @click="$emit('click')"
  >
    <span class="node-color-bar" />
    <div class="node-info">
      <span class="node-name">{{ displayName }}</span>
      <span v-if="description" class="node-description">{{ description }}</span>
    </div>
    <i class="pi pi-plus-circle add-icon" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

// =============================================================================
// PROPS & EMITS
// =============================================================================

interface NodeDef {
  type: string
  name?: string
  displayName?: string
  description?: string
  color?: string
  category?: string
}

const props = defineProps<{
  node: NodeDef | null
}>()

const emit = defineEmits<{
  (e: 'dragstart', event: DragEvent): void
  (e: 'click'): void
}>()

// =============================================================================
// STATE
// =============================================================================

const isDragging = ref(false)

// =============================================================================
// COMPUTED
// =============================================================================

const displayName = computed(() => {
  if (!props.node) return 'Unknown'
  return props.node.displayName || props.node.name || props.node.type.split('/').pop() || props.node.type
})

const description = computed(() => {
  return props.node?.description || ''
})

const nodeColor = computed(() => {
  if (!props.node) return '#888'

  // Use explicit color if provided
  if (props.node.color) return props.node.color

  // Determine color by type
  const type = props.node.type.toLowerCase()

  if (type.includes('component')) return 'var(--trinity-component, #4CAF50)'
  if (type.includes('system')) return 'var(--trinity-system, #2196F3)'
  if (type.includes('resource')) return 'var(--trinity-resource, #FF9800)'
  if (type.includes('event')) return 'var(--trinity-event, #9C27B0)'
  if (type.includes('math')) return '#607D8B'
  if (type.includes('logic')) return '#795548'

  return '#888'
})

// =============================================================================
// METHODS
// =============================================================================

function handleDragStart(event: DragEvent) {
  isDragging.value = true
  emit('dragstart', event)
}

function handleDragEnd() {
  isDragging.value = false
}
</script>

<style scoped>
.node-palette-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px 8px 8px;
  cursor: grab;
  user-select: none;
  transition: background-color 0.15s ease;
}

.node-palette-item:hover {
  background-color: var(--hover-bg, #2a2d2e);
}

.node-palette-item:active {
  cursor: grabbing;
}

.node-palette-item.is-dragging {
  opacity: 0.5;
}

.node-color-bar {
  width: 3px;
  height: 28px;
  border-radius: 2px;
  background-color: var(--node-color, #888);
  flex-shrink: 0;
}

.node-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
  overflow: hidden;
}

.node-name {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary, #cccccc);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.node-description {
  font-size: 11px;
  color: var(--text-muted, #888);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.add-icon {
  font-size: 14px;
  color: var(--text-muted, #666);
  opacity: 0;
  transition: opacity 0.15s ease;
}

.node-palette-item:hover .add-icon {
  opacity: 1;
}
</style>
