<template>
  <Transition name="slide-up">
    <div
      v-if="hasHighlight"
      class="source-indicator"
      :class="{ copied: showCopiedFeedback }"
      @click="handleClick"
    >
      <!-- File icon -->
      <span class="source-icon">
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14,2 14,8 20,8" />
          <line x1="16" y1="13" x2="8" y2="13" />
          <line x1="16" y1="17" x2="8" y2="17" />
          <polyline points="10,9 9,9 8,9" />
        </svg>
      </span>

      <!-- Source location text -->
      <span class="source-text" :title="fullPath">
        {{ displayText }}
      </span>

      <!-- Line number badge -->
      <span class="line-badge">
        :{{ currentSource?.line }}
      </span>

      <!-- Action buttons -->
      <div class="source-actions">
        <!-- Copy button -->
        <button
          class="action-btn"
          title="Copy path to clipboard"
          @click.stop="handleCopy"
        >
          <svg
            v-if="!showCopiedFeedback"
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
          </svg>
          <svg
            v-else
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <polyline points="20,6 9,17 4,12" />
          </svg>
        </button>

        <!-- Open in editor button -->
        <button
          class="action-btn"
          title="Open in external editor"
          @click.stop="handleOpenInEditor"
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
            <polyline points="15,3 21,3 21,9" />
            <line x1="10" y1="14" x2="21" y2="3" />
          </svg>
        </button>

        <!-- Close button -->
        <button
          class="action-btn close-btn"
          title="Close"
          @click.stop="handleClose"
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>
    </div>
  </Transition>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useSourceNavigation } from '@/composables/useSourceNavigation'
import { UI_CONFIG } from '@/config/flowforge.config'

// =============================================================================
// PROPS & EMITS
// =============================================================================

const emit = defineEmits<{
  (e: 'open-in-editor', file: string, line: number): void
  (e: 'copied', path: string): void
  (e: 'closed'): void
}>()

// =============================================================================
// COMPOSABLE
// =============================================================================

const {
  currentSource,
  hasHighlight,
  clearHighlight,
  copyToClipboard,
  emitExternalNavigation,
} = useSourceNavigation()

// =============================================================================
// STATE
// =============================================================================

const showCopiedFeedback = ref(false)
const MAX_DISPLAY_LENGTH = UI_CONFIG.sourceIndicator.maxDisplayLength

// =============================================================================
// COMPUTED
// =============================================================================

/**
 * Full path for tooltip display.
 */
const fullPath = computed(() => {
  if (!currentSource.value) return ''
  return `${currentSource.value.file}:${currentSource.value.line}`
})

/**
 * Truncated display text for the indicator.
 */
const displayText = computed(() => {
  if (!currentSource.value) return ''

  const file = currentSource.value.file

  // Show just filename if path is too long
  if (file.length > MAX_DISPLAY_LENGTH) {
    const parts = file.split('/')
    const fileName = parts[parts.length - 1]

    // If filename alone is short enough, show ".../" prefix
    if (fileName && fileName.length < MAX_DISPLAY_LENGTH - 4) {
      return `.../${fileName}`
    }

    // Otherwise truncate from the start
    return '...' + file.slice(-(MAX_DISPLAY_LENGTH - 3))
  }

  return file
})

// =============================================================================
// METHODS
// =============================================================================

/**
 * Handle click on the indicator (copy to clipboard).
 */
async function handleClick(): Promise<void> {
  await handleCopy()
}

/**
 * Copy the source path to clipboard.
 */
async function handleCopy(): Promise<void> {
  const success = await copyToClipboard()

  if (success) {
    showCopiedFeedback.value = true
    emit('copied', fullPath.value)

    // Reset feedback after delay
    setTimeout(() => {
      showCopiedFeedback.value = false
    }, UI_CONFIG.sourceIndicator.feedbackDuration)
  }
}

/**
 * Open the source file in an external editor.
 */
function handleOpenInEditor(): void {
  if (!currentSource.value) return

  // Emit external navigation event for IDE integration
  emitExternalNavigation(currentSource.value)

  // Also emit component event for parent handling
  emit('open-in-editor', currentSource.value.file, currentSource.value.line)
}

/**
 * Close the indicator and clear the highlight.
 */
function handleClose(): void {
  clearHighlight()
  emit('closed')
}
</script>

<style scoped>
.source-indicator {
  position: absolute;
  bottom: 16px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  background-color: var(--source-indicator-bg, rgba(30, 30, 46, 0.95));
  border: 1px solid var(--source-indicator-border, rgba(100, 100, 140, 0.3));
  border-radius: 6px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  color: var(--source-indicator-text, #e0e0e0);
  font-family: var(--font-mono, 'SF Mono', 'Monaco', 'Consolas', monospace);
  font-size: 12px;
  cursor: pointer;
  user-select: none;
  transition: all 0.2s ease;
  z-index: 100;
}

.source-indicator:hover {
  background-color: var(--source-indicator-bg-hover, rgba(40, 40, 56, 0.98));
  border-color: var(--source-indicator-border-hover, rgba(120, 120, 160, 0.4));
}

.source-indicator.copied {
  border-color: var(--success-color, #4ade80);
}

.source-icon {
  display: flex;
  align-items: center;
  color: var(--source-indicator-icon, #8b8b9e);
}

.source-text {
  color: var(--source-indicator-file, #b8b8c8);
  max-width: 300px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.line-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 2px 6px;
  background-color: var(--line-badge-bg, rgba(59, 130, 246, 0.2));
  border-radius: 4px;
  color: var(--line-badge-text, #60a5fa);
  font-weight: 500;
  font-size: 11px;
}

.source-actions {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-left: 4px;
  padding-left: 8px;
  border-left: 1px solid var(--source-indicator-divider, rgba(100, 100, 140, 0.2));
}

.action-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  padding: 0;
  background: transparent;
  border: none;
  border-radius: 4px;
  color: var(--source-indicator-icon, #8b8b9e);
  cursor: pointer;
  transition: all 0.15s ease;
}

.action-btn:hover {
  background-color: var(--action-btn-hover-bg, rgba(100, 100, 140, 0.2));
  color: var(--source-indicator-text, #e0e0e0);
}

.action-btn:active {
  background-color: var(--action-btn-active-bg, rgba(100, 100, 140, 0.3));
}

.close-btn:hover {
  background-color: var(--close-btn-hover-bg, rgba(239, 68, 68, 0.2));
  color: var(--close-btn-hover-color, #f87171);
}

/* Transition animations */
.slide-up-enter-active,
.slide-up-leave-active {
  transition: all 0.25s ease;
}

.slide-up-enter-from,
.slide-up-leave-to {
  opacity: 0;
  transform: translateX(-50%) translateY(20px);
}
</style>
