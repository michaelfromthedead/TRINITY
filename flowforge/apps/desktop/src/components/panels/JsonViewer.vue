<template>
  <div class="json-viewer" :class="{ 'json-viewer--compact': compact }">
    <template v-if="isEmpty">
      <span class="empty-value">Empty object</span>
    </template>
    <template v-else>
      <div
        v-for="entry in entries"
        :key="entry.key"
        class="json-entry"
      >
        <span class="json-key" :style="keyStyle">{{ entry.key }}</span>
        <span class="json-colon">:</span>
        <template v-if="entry.isObject">
          <button
            class="expand-toggle"
            :class="{ expanded: expandedKeys.has(entry.key) }"
            @click="toggleExpand(entry.key)"
          >
            <i
              :class="expandedKeys.has(entry.key) ? 'icon-[lucide--chevron-down]' : 'icon-[lucide--chevron-right]'"
            />
            <span class="object-preview">{{ entry.preview }}</span>
          </button>
          <div
            v-if="expandedKeys.has(entry.key)"
            class="nested-object"
            :style="{ borderColor: accentColor }"
          >
            <JsonViewer
              :data="entry.value as Record<string, unknown>"
              :depth="nextDepth"
              :accent-color="resolvedAccentColor"
              :compact="resolvedCompact"
            />
          </div>
        </template>
        <template v-else-if="entry.isArray">
          <button
            class="expand-toggle"
            :class="{ expanded: expandedKeys.has(entry.key) }"
            @click="toggleExpand(entry.key)"
          >
            <i
              :class="expandedKeys.has(entry.key) ? 'icon-[lucide--chevron-down]' : 'icon-[lucide--chevron-right]'"
            />
            <span class="array-preview">Array[{{ (entry.value as unknown[]).length }}]</span>
          </button>
          <div
            v-if="expandedKeys.has(entry.key)"
            class="nested-array"
            :style="{ borderColor: accentColor }"
          >
            <div
              v-for="(item, i) in (entry.value as unknown[])"
              :key="i"
              class="array-item"
            >
              <span class="array-index">[{{ i }}]</span>
              <template v-if="isObject(item)">
                <JsonViewer
                  :data="item as Record<string, unknown>"
                  :depth="nextDepth"
                  :accent-color="resolvedAccentColor"
                  :compact="resolvedCompact"
                />
              </template>
              <template v-else>
                <span :class="getValueClass(item)">{{ formatValue(item) }}</span>
              </template>
            </div>
          </div>
        </template>
        <template v-else>
          <span :class="getValueClass(entry.value)">{{ formatValue(entry.value) }}</span>
        </template>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { TRINITY_COLORS } from '@/nodes/nodeTheme'

// =============================================================================
// PROPS
// =============================================================================

const props = withDefaults(
  defineProps<{
    /** The data object to display */
    data: Record<string, unknown>
    /** Nesting depth for recursive rendering */
    depth?: number
    /** Accent color for nested borders and key highlighting */
    accentColor?: string
    /** Compact mode for smaller display */
    compact?: boolean
  }>(),
  {
    depth: 0,
    accentColor: TRINITY_COLORS.component.primary,
    compact: false
  }
)

// =============================================================================
// COMPUTED STYLES
// =============================================================================

const keyStyle = computed(() => ({
  color: props.accentColor
}))

// For recursive component calls, ensure we pass non-undefined values
const nextDepth = computed(() => (props.depth ?? 0) + 1)
const resolvedAccentColor = computed(() => props.accentColor ?? TRINITY_COLORS.component.primary)
const resolvedCompact = computed(() => props.compact ?? false)

// =============================================================================
// STATE
// =============================================================================

const expandedKeys = ref(new Set<string>())

// =============================================================================
// COMPUTED
// =============================================================================

interface JsonEntry {
  key: string
  value: unknown
  isObject: boolean
  isArray: boolean
  preview: string
}

const entries = computed<JsonEntry[]>(() => {
  const result: JsonEntry[] = []

  for (const [key, value] of Object.entries(props.data)) {
    const isObj = isObject(value)
    const isArr = Array.isArray(value)

    result.push({
      key,
      value,
      isObject: isObj && !isArr,
      isArray: isArr,
      preview: isObj && !isArr ? getObjectPreview(value as Record<string, unknown>) : ''
    })
  }

  return result
})

const isEmpty = computed(() => Object.keys(props.data).length === 0)

// =============================================================================
// METHODS
// =============================================================================

function toggleExpand(key: string): void {
  if (expandedKeys.value.has(key)) {
    expandedKeys.value.delete(key)
  } else {
    expandedKeys.value.add(key)
  }
}

function isObject(value: unknown): boolean {
  return typeof value === 'object' && value !== null
}

function getObjectPreview(obj: Record<string, unknown>): string {
  const keys = Object.keys(obj)
  if (keys.length === 0) return '{}'
  if (keys.length <= 2) {
    return `{ ${keys.join(', ')} }`
  }
  return `{ ${keys.slice(0, 2).join(', ')}, ... }`
}

function getValueClass(value: unknown): string {
  if (value === null) return 'json-null'
  if (value === undefined) return 'json-undefined'
  if (typeof value === 'string') return 'json-string'
  if (typeof value === 'number') return 'json-number'
  if (typeof value === 'boolean') return 'json-boolean'
  return 'json-value'
}

function formatValue(value: unknown): string {
  if (value === null) return 'null'
  if (value === undefined) return 'undefined'
  if (typeof value === 'string') return `"${value}"`
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
</script>

<style scoped>
.json-viewer {
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  font-size: 11px;
  line-height: 1.6;
}

.json-viewer--compact {
  font-size: 10px;
  line-height: 1.5;
}

.json-entry {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: 4px;
  padding: 2px 0;
}

.json-viewer--compact .json-entry {
  padding: 1px 0;
}

.json-key {
  font-weight: 500;
}

.json-colon {
  color: var(--flowforge-text-muted, #666680);
}

/* Value Types */
.json-string {
  color: var(--flowforge-node-event, #f97316);
  word-break: break-word;
}

.json-number {
  color: var(--flowforge-node-system, #22c55e);
}

.json-boolean {
  color: var(--flowforge-node-resource, #a855f7);
  font-weight: 600;
}

.json-null,
.json-undefined {
  color: var(--flowforge-text-muted, #666680);
  font-style: italic;
}

.json-value {
  color: var(--flowforge-text, #e0e0e0);
}

/* Expand Toggle */
.expand-toggle {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 4px;
  margin: -2px;
  background: none;
  border: none;
  border-radius: 3px;
  color: var(--flowforge-text-secondary, #a0a0a0);
  cursor: pointer;
  font-family: inherit;
  font-size: inherit;
  transition: all 0.15s ease;
}

.expand-toggle:hover {
  color: var(--flowforge-text, #e0e0e0);
  background-color: var(--flowforge-hover-bg, #3a3a4a);
}

.expand-toggle i {
  width: 12px;
  height: 12px;
  flex-shrink: 0;
}

.json-viewer--compact .expand-toggle i {
  width: 10px;
  height: 10px;
}

.object-preview,
.array-preview {
  color: var(--flowforge-text-muted, #666680);
}

/* Nested Content */
.nested-object,
.nested-array {
  width: 100%;
  padding-left: 12px;
  margin-top: 4px;
  border-left: 1px solid;
  border-color: var(--flowforge-border, #3a3a4a);
}

.json-viewer--compact .nested-object,
.json-viewer--compact .nested-array {
  padding-left: 10px;
  margin-top: 2px;
}

.array-item {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: 4px;
  padding: 2px 0;
}

.json-viewer--compact .array-item {
  padding: 1px 0;
}

.array-index {
  color: var(--flowforge-text-muted, #666680);
  font-size: 10px;
  min-width: 20px;
}

.json-viewer--compact .array-index {
  font-size: 9px;
  min-width: 18px;
}

/* Empty State */
.empty-value {
  color: var(--flowforge-text-muted, #666680);
  font-style: italic;
}
</style>
