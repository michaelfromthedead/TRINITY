<template>
  <button
    :class="['registry-entry', `entry-${entry.type}`]"
    :title="entryTooltip"
    @click="emit('click', entry)"
  >
    <!-- Status Indicator -->
    <span
      :class="['status-dot', statusClass]"
      :title="statusTitle"
    />

    <!-- Entry Info -->
    <div class="entry-info">
      <span class="entry-name">{{ entry.name }}</span>
      <span class="entry-path">{{ truncatedPath }}</span>
    </div>

    <!-- Badges -->
    <div class="entry-badges">
      <!-- AST Badge -->
      <span
        v-if="entry.existsInAST"
        class="badge ast-badge"
        title="Type exists in parsed code"
      >
        AST
      </span>
      <!-- Source Badge (if has source info) -->
      <span
        v-if="entry.sourceFile"
        class="badge source-badge"
        :title="`${entry.sourceFile}${entry.sourceLine ? ':' + entry.sourceLine : ''}`"
      >
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
        </svg>
      </span>
    </div>

    <!-- Highlight Arrow -->
    <i class="highlight-arrow">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="m9 18 6-6-6-6" />
      </svg>
    </i>
  </button>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { RegistryEntry, RegistrationStatus } from '@/composables/useRegistryPanel'
import { TRINITY_COLORS } from '@/nodes/nodeTheme'

// =============================================================================
// PROPS & EMITS
// =============================================================================

const props = defineProps<{
  entry: RegistryEntry
}>()

const emit = defineEmits<{
  (e: 'click', entry: RegistryEntry): void
}>()

// =============================================================================
// COMPUTED
// =============================================================================

const statusConfig: Record<RegistrationStatus, { class: string; title: string }> = {
  registered: { class: 'status-registered', title: 'Registered' },
  pending: { class: 'status-pending', title: 'Pending registration' },
  error: { class: 'status-error', title: 'Registration error' },
  unknown: { class: 'status-unknown', title: 'Unknown status' }
}

const statusClass = computed(() => statusConfig[props.entry.status].class)
const statusTitle = computed(() => statusConfig[props.entry.status].title)

const truncatedPath = computed(() => {
  const path = props.entry.modulePath
  const maxLength = 30
  if (path.length <= maxLength) return path
  return '...' + path.slice(-(maxLength - 3))
})

const entryTooltip = computed(() => {
  let tooltip = `${props.entry.name}\n${props.entry.modulePath}`
  if (props.entry.sourceFile) {
    tooltip += `\n${props.entry.sourceFile}`
    if (props.entry.sourceLine) {
      tooltip += `:${props.entry.sourceLine}`
    }
  }
  return tooltip
})

const entryColor = computed(() => TRINITY_COLORS[props.entry.type].primary)
</script>

<style scoped>
.registry-entry {
  display: flex;
  align-items: center;
  width: 100%;
  padding: 6px 12px 6px 16px;
  border: none;
  background: transparent;
  color: var(--flowforge-text, #e0e0e0);
  cursor: pointer;
  text-align: left;
  transition: background-color 0.15s ease;
  gap: 8px;
}

.registry-entry:hover {
  background-color: var(--flowforge-hover-bg, #3a3a4a);
}

.registry-entry:hover .highlight-arrow {
  opacity: 1;
  transform: translateX(0);
}

/* Status Indicator */
.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}

.status-registered {
  background-color: var(--flowforge-success, #22c55e);
}

.status-pending {
  background-color: var(--flowforge-warning, #eab308);
  animation: pulse 1.5s ease-in-out infinite;
}

.status-error {
  background-color: var(--flowforge-danger, #dc2626);
}

.status-unknown {
  background-color: var(--flowforge-text-muted, #666680);
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* Entry Info */
.entry-info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.entry-name {
  font-size: 12px;
  font-weight: 500;
  color: var(--flowforge-text, #e0e0e0);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.entry-path {
  font-size: 10px;
  color: var(--flowforge-text-muted, #666680);
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Badges */
.entry-badges {
  display: flex;
  align-items: center;
  gap: 4px;
  flex-shrink: 0;
}

.badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 2px 5px;
  border-radius: 4px;
  font-size: 9px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

.ast-badge {
  background-color: color-mix(in srgb, v-bind(entryColor) 20%, transparent);
  color: v-bind(entryColor);
}

.source-badge {
  background-color: var(--flowforge-surface, #2a2a3a);
  color: var(--flowforge-text-muted, #666680);
  padding: 3px 4px;
}

/* Highlight Arrow */
.highlight-arrow {
  display: flex;
  align-items: center;
  color: var(--flowforge-text-muted, #666680);
  opacity: 0;
  transform: translateX(-4px);
  transition: all 0.15s ease;
  flex-shrink: 0;
}

/* Type-specific hover colors */
.entry-component:hover {
  background-color: color-mix(in srgb, var(--flowforge-node-component) 12%, transparent);
}

.entry-component .entry-name {
  color: var(--flowforge-node-component-light, #60a5fa);
}

.entry-system:hover {
  background-color: color-mix(in srgb, var(--flowforge-node-system) 12%, transparent);
}

.entry-system .entry-name {
  color: var(--flowforge-node-system-light, #4ade80);
}

.entry-resource:hover {
  background-color: color-mix(in srgb, var(--flowforge-node-resource) 12%, transparent);
}

.entry-resource .entry-name {
  color: var(--flowforge-node-resource-light, #c084fc);
}

.entry-event:hover {
  background-color: color-mix(in srgb, var(--flowforge-node-event) 12%, transparent);
}

.entry-event .entry-name {
  color: var(--flowforge-node-event-light, #fb923c);
}
</style>
