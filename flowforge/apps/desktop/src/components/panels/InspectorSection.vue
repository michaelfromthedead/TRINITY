<template>
  <div class="inspector-section" :class="{ collapsed: isCollapsed }">
    <button class="section-header" @click="handleToggle">
      <span class="section-icon">
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          :class="{ rotated: !isCollapsed }"
        >
          <polyline points="9,18 15,12 9,6" />
        </svg>
      </span>
      <span class="section-title">{{ title }}</span>
      <span v-if="itemCount !== undefined" class="section-count">{{ itemCount }}</span>
    </button>
    <Transition name="collapse">
      <div v-show="!isCollapsed" class="section-content">
        <slot />
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
// =============================================================================
// PROPS & EMITS
// =============================================================================

defineProps<{
  /** Section title */
  title: string
  /** Whether the section is collapsed */
  isCollapsed: boolean
  /** Number of items in the section (optional) */
  itemCount?: number
}>()

const emit = defineEmits<{
  (e: 'toggle'): void
}>()

// =============================================================================
// METHODS
// =============================================================================

function handleToggle(): void {
  emit('toggle')
}
</script>

<style scoped>
.inspector-section {
  border-bottom: 1px solid var(--flowforge-border, #3a3a4a);
}

.inspector-section:last-child {
  border-bottom: none;
}

.section-header {
  display: flex;
  align-items: center;
  width: 100%;
  padding: 10px 16px;
  background: transparent;
  border: none;
  color: var(--flowforge-text, #e0e0e0);
  font-size: 12px;
  font-weight: 600;
  text-align: left;
  cursor: pointer;
  transition: background-color 0.15s ease;
}

.section-header:hover {
  background-color: var(--flowforge-hover-bg, #3a3a4a);
}

.section-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  margin-right: 8px;
  color: var(--flowforge-text-muted, #666680);
  transition: transform 0.2s ease;
}

.section-icon svg.rotated {
  transform: rotate(90deg);
}

.section-title {
  flex: 1;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.section-count {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 20px;
  height: 18px;
  padding: 0 6px;
  background-color: var(--flowforge-surface, #2a2a3a);
  border-radius: 9px;
  font-size: 10px;
  font-weight: 500;
  color: var(--flowforge-text-secondary, #a0a0a0);
}

.section-content {
  padding: 8px 16px 16px 16px;
}

/* Collapse transition */
.collapse-enter-active,
.collapse-leave-active {
  transition: all 0.2s ease;
  overflow: hidden;
}

.collapse-enter-from,
.collapse-leave-to {
  opacity: 0;
  max-height: 0;
  padding-top: 0;
  padding-bottom: 0;
}

.collapse-enter-to,
.collapse-leave-from {
  opacity: 1;
  max-height: 500px;
}
</style>
