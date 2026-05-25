<template>
  <div :class="['registry-section', `section-${type}`]">
    <!-- Section Header -->
    <button
      class="section-header"
      :aria-expanded="isExpanded"
      @click="emit('toggle')"
    >
      <i :class="['section-icon', `icon-${type}`]">
        <component :is="typeIcon" />
      </i>
      <span class="section-title">{{ typeLabel }}</span>
      <span
        class="section-count"
        :class="{ filtered: isFiltered && filteredCount !== totalCount }"
        :title="isFiltered && filteredCount !== totalCount
          ? `${filteredCount} of ${totalCount} shown`
          : `${totalCount} ${typeLabel.toLowerCase()}`"
      >
        {{ isFiltered && filteredCount !== totalCount ? `${filteredCount}/${totalCount}` : totalCount }}
      </span>
      <i class="chevron-icon">
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          :style="{ transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)' }"
        >
          <path d="m9 18 6-6-6-6" />
        </svg>
      </i>
    </button>

    <!-- Section Content -->
    <div v-show="isExpanded" class="section-content">
      <div v-if="entries.length === 0" class="section-empty">
        No {{ typeLabel.toLowerCase() }} {{ isFiltered ? 'match filter' : 'registered' }}
      </div>
      <RegistryEntryItem
        v-for="entry in entries"
        :key="entry.id"
        :entry="entry"
        @click="emit('entryClick', entry)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, h, type FunctionalComponent } from 'vue'
import type { RegistryEntry, FilterableTrinityType } from '@/composables/useRegistryPanel'
import { TRINITY_COLORS } from '@/nodes/nodeTheme'
import RegistryEntryItem from './RegistryEntryItem.vue'

// =============================================================================
// PROPS & EMITS
// =============================================================================

const props = defineProps<{
  type: FilterableTrinityType
  entries: RegistryEntry[]
  isExpanded: boolean
  totalCount: number
  filteredCount: number
  isFiltered: boolean
}>()

const emit = defineEmits<{
  (e: 'toggle'): void
  (e: 'entryClick', entry: RegistryEntry): void
}>()

// =============================================================================
// TYPE CONFIGURATION
// =============================================================================

const typeConfig: Record<FilterableTrinityType, { label: string; iconPath: string }> = {
  component: {
    label: 'Components',
    iconPath: 'M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z'
  },
  system: {
    label: 'Systems',
    iconPath: 'M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2v-4M9 21H5a2 2 0 0 1-2-2v-4m0-6v6m18-6v6'
  },
  resource: {
    label: 'Resources',
    iconPath: 'M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z M12 2v10l6.93 4'
  },
  event: {
    label: 'Events',
    iconPath: 'M13 2 3 14h9l-1 8 10-12h-9l1-8z'
  }
}

// =============================================================================
// COMPUTED
// =============================================================================

const typeLabel = computed(() => typeConfig[props.type].label)

// Create icon components inline with proper SVG paths
const BoxIcon: FunctionalComponent = () =>
  h('svg', { width: 14, height: 14, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, [
    h('path', { d: 'M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z' }),
    h('polyline', { points: '3.29 7 12 12 20.71 7' }),
    h('line', { x1: 12, y1: 22, x2: 12, y2: 12 })
  ])

const CpuIcon: FunctionalComponent = () =>
  h('svg', { width: 14, height: 14, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, [
    h('rect', { x: 4, y: 4, width: 16, height: 16, rx: 2 }),
    h('rect', { x: 9, y: 9, width: 6, height: 6 }),
    h('path', { d: 'M9 1v3' }),
    h('path', { d: 'M15 1v3' }),
    h('path', { d: 'M9 20v3' }),
    h('path', { d: 'M15 20v3' }),
    h('path', { d: 'M20 9h3' }),
    h('path', { d: 'M20 14h3' }),
    h('path', { d: 'M1 9h3' }),
    h('path', { d: 'M1 14h3' })
  ])

const DatabaseIcon: FunctionalComponent = () =>
  h('svg', { width: 14, height: 14, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, [
    h('ellipse', { cx: 12, cy: 5, rx: 9, ry: 3 }),
    h('path', { d: 'M3 5V19A9 3 0 0 0 21 19V5' }),
    h('path', { d: 'M3 12A9 3 0 0 0 21 12' })
  ])

const ZapIcon: FunctionalComponent = () =>
  h('svg', { width: 14, height: 14, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, [
    h('polygon', { points: '13 2 3 14 12 14 11 22 21 10 12 10 13 2' })
  ])

const typeIcons: Record<FilterableTrinityType, FunctionalComponent> = {
  component: BoxIcon,
  system: CpuIcon,
  resource: DatabaseIcon,
  event: ZapIcon
}

const typeIcon = computed(() => typeIcons[props.type])

// Get color from theme
const sectionColor = computed(() => TRINITY_COLORS[props.type].primary)
</script>

<style scoped>
.registry-section {
  border-bottom: 1px solid var(--flowforge-border, #3a3a4a);
}

.registry-section:last-child {
  border-bottom: none;
}

/* Section Header */
.section-header {
  display: flex;
  align-items: center;
  width: 100%;
  padding: 8px 12px;
  border: none;
  background: transparent;
  color: var(--flowforge-text, #e0e0e0);
  cursor: pointer;
  text-align: left;
  transition: background-color 0.15s ease;
  gap: 8px;
}

.section-header:hover {
  background-color: var(--flowforge-hover-bg, #3a3a4a);
}

.section-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

/* Type-specific icon colors */
.section-component .section-icon {
  color: v-bind(sectionColor);
}

.section-system .section-icon {
  color: v-bind(sectionColor);
}

.section-resource .section-icon {
  color: v-bind(sectionColor);
}

.section-event .section-icon {
  color: v-bind(sectionColor);
}

.section-title {
  flex: 1;
  font-size: 12px;
  font-weight: 500;
}

.section-count {
  font-size: 10px;
  color: var(--flowforge-text-muted, #666680);
  background-color: var(--flowforge-surface, #2a2a3a);
  padding: 2px 6px;
  border-radius: 10px;
  min-width: 20px;
  text-align: center;
}

.section-count.filtered {
  background-color: var(--flowforge-primary, #6366f1);
  color: white;
}

.chevron-icon {
  display: flex;
  align-items: center;
  color: var(--flowforge-text-muted, #666680);
  transition: transform 0.15s ease;
}

.chevron-icon svg {
  transition: transform 0.15s ease;
}

/* Section Content */
.section-content {
  padding-bottom: 4px;
}

.section-empty {
  padding: 12px 16px;
  font-size: 11px;
  color: var(--flowforge-text-muted, #666680);
  text-align: center;
  font-style: italic;
}

/* Type-specific section styling */
.section-component .section-header:hover {
  background-color: color-mix(in srgb, var(--flowforge-node-component) 10%, transparent);
}

.section-system .section-header:hover {
  background-color: color-mix(in srgb, var(--flowforge-node-system) 10%, transparent);
}

.section-resource .section-header:hover {
  background-color: color-mix(in srgb, var(--flowforge-node-resource) 10%, transparent);
}

.section-event .section-header:hover {
  background-color: color-mix(in srgb, var(--flowforge-node-event) 10%, transparent);
}
</style>
