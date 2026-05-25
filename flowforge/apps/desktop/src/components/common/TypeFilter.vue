<template>
  <div class="type-filter" role="group" aria-label="Filter nodes by type">
    <button
      v-for="typeInfo in typeInfos"
      :key="typeInfo.type"
      type="button"
      :class="[
        'type-filter-btn',
        isVisible(typeInfo.type) ? 'active' : 'inactive'
      ]"
      :style="getButtonStyle(typeInfo.type)"
      :aria-pressed="isVisible(typeInfo.type)"
      :title="`${isVisible(typeInfo.type) ? 'Hide' : 'Show'} ${typeInfo.label} nodes`"
      @click="toggleType(typeInfo.type)"
    >
      <i :class="typeInfo.icon" />
      <span class="btn-label">{{ typeInfo.label }}</span>
    </button>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useTypeFilter, TRINITY_TYPES, type FilterableTrinityType } from '@/composables/useTypeFilter'
import { TRINITY_COLORS } from '@/nodes/nodeTheme'

const { toggleType, isVisible, visibleTypes } = useTypeFilter()

interface TypeInfo {
  type: FilterableTrinityType
  label: string
  icon: string
  color: string
}

/**
 * Type metadata including labels and icons for each Trinity node type
 */
const typeInfos = computed<TypeInfo[]>(() => [
  {
    type: 'component',
    label: 'Components',
    icon: 'icon-[lucide--box]',
    color: TRINITY_COLORS.component.primary
  },
  {
    type: 'system',
    label: 'Systems',
    icon: 'icon-[lucide--cpu]',
    color: TRINITY_COLORS.system.primary
  },
  {
    type: 'resource',
    label: 'Resources',
    icon: 'icon-[lucide--database]',
    color: TRINITY_COLORS.resource.primary
  },
  {
    type: 'event',
    label: 'Events',
    icon: 'icon-[lucide--zap]',
    color: TRINITY_COLORS.event.primary
  }
])

/**
 * Generate dynamic button styles based on active state and type color
 */
function getButtonStyle(type: FilterableTrinityType): Record<string, string> {
  const color = TRINITY_COLORS[type].primary
  const isActive = visibleTypes[type]

  if (isActive) {
    return {
      '--btn-color': color,
      '--btn-bg': color,
      '--btn-border': color
    }
  }

  return {
    '--btn-color': 'var(--text-muted, #6B7280)',
    '--btn-bg': 'transparent',
    '--btn-border': 'var(--border-color, #3d3d3d)'
  }
}
</script>

<style scoped>
.type-filter {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  background-color: var(--toolbar-bg, rgba(45, 45, 45, 0.9));
  border-radius: 8px;
  backdrop-filter: blur(8px);
  pointer-events: auto;
}

.type-filter-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s ease;
  border: 1px solid var(--btn-border);
  background-color: var(--btn-bg);
  color: var(--btn-color);
}

.type-filter-btn.active {
  background-color: var(--btn-bg);
  color: white;
  border-color: var(--btn-border);
}

.type-filter-btn.inactive {
  background-color: transparent;
  color: var(--btn-color);
  border-color: var(--btn-border);
  opacity: 0.7;
}

.type-filter-btn:hover {
  opacity: 1;
  transform: translateY(-1px);
}

.type-filter-btn:active {
  transform: translateY(0);
}

.type-filter-btn i {
  font-size: 12px;
}

.btn-label {
  font-family: Inter, system-ui, sans-serif;
}

/* Responsive: hide labels on smaller screens */
@media (max-width: 768px) {
  .btn-label {
    display: none;
  }

  .type-filter-btn {
    padding: 6px;
  }

  .type-filter-btn i {
    font-size: 14px;
  }
}
</style>
