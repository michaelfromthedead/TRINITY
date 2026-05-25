<template>
  <div class="diff-preview-dialog">
    <!-- Header -->
    <div class="diff-preview-dialog__header">
      <div class="diff-preview-dialog__title-section">
        <h2 class="diff-preview-dialog__title">Code Changes Preview</h2>
        <span v-if="filePath" class="diff-preview-dialog__filepath">
          {{ filePath }}
        </span>
      </div>

      <!-- Stats badges -->
      <div v-if="diffResult" class="diff-preview-dialog__stats">
        <span class="diff-preview-dialog__stat diff-preview-dialog__stat--additions">
          +{{ diffResult.stats.additions }}
        </span>
        <span class="diff-preview-dialog__stat diff-preview-dialog__stat--deletions">
          -{{ diffResult.stats.deletions }}
        </span>
        <span class="diff-preview-dialog__stat diff-preview-dialog__stat--changes">
          {{ diffResult.stats.changes }} {{ diffResult.stats.changes === 1 ? 'change' : 'changes' }}
        </span>
      </div>

      <!-- View mode toggle -->
      <div class="diff-preview-dialog__view-toggle">
        <button
          :class="[
            'diff-preview-dialog__view-btn',
            { 'diff-preview-dialog__view-btn--active': viewMode === 'unified' }
          ]"
          @click="setViewMode('unified')"
          title="Unified view"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M1 3h14v2H1V3zm0 4h14v2H1V7zm0 4h14v2H1v-2z"/>
          </svg>
          Unified
        </button>
        <button
          :class="[
            'diff-preview-dialog__view-btn',
            { 'diff-preview-dialog__view-btn--active': viewMode === 'split' }
          ]"
          @click="setViewMode('split')"
          title="Split view"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M1 2h6v12H1V2zm8 0h6v12H9V2z"/>
          </svg>
          Split
        </button>
      </div>
    </div>

    <!-- Loading state -->
    <div v-if="isLoading" class="diff-preview-dialog__loading">
      <div class="diff-preview-dialog__spinner"></div>
      <span>Generating diff...</span>
    </div>

    <!-- Error state -->
    <div v-else-if="error" class="diff-preview-dialog__error">
      <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
      </svg>
      <span>{{ error }}</span>
    </div>

    <!-- No changes state -->
    <div v-else-if="diffResult && !diffResult.hasChanges" class="diff-preview-dialog__no-changes">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
        <path fill-rule="evenodd" d="M12 22C6.477 22 2 17.523 2 12S6.477 2 12 2s10 4.477 10 10-4.477 10-10 10zm-1.177-7.86l-2.765-2.767L7 12.431l3.823 3.827L17 10.088l-1.058-1.058-5.119 5.11z" clip-rule="evenodd"/>
      </svg>
      <span>No changes detected</span>
    </div>

    <!-- Diff content -->
    <div v-else class="diff-preview-dialog__content">
      <!-- Unified view -->
      <div v-if="viewMode === 'unified'" class="diff-preview-dialog__unified">
        <template v-for="(hunk, hunkIndex) in diffResult?.hunks ?? []" :key="hunkIndex">
          <DiffLine
            v-for="(line, lineIndex) in hunk.lines"
            :key="`${hunkIndex}-${lineIndex}`"
            :type="line.type"
            :content="line.content"
            :original-line-number="line.originalLine"
            :modified-line-number="line.modifiedLine"
            :show-original-line-number="true"
            :show-modified-line-number="true"
            :highlight-syntax="true"
          />
        </template>
      </div>

      <!-- Split view -->
      <div v-else class="diff-preview-dialog__split">
        <div class="diff-preview-dialog__split-pane diff-preview-dialog__split-pane--left">
          <div class="diff-preview-dialog__split-header">Original</div>
          <div class="diff-preview-dialog__split-content">
            <DiffLine
              v-for="(line, index) in sideBySideDiff?.left ?? []"
              :key="`left-${index}`"
              :type="line.type"
              :content="line.content"
              :original-line-number="line.lineNumber"
              :show-original-line-number="true"
              :show-modified-line-number="false"
              :highlight-syntax="true"
            />
          </div>
        </div>
        <div class="diff-preview-dialog__split-divider"></div>
        <div class="diff-preview-dialog__split-pane diff-preview-dialog__split-pane--right">
          <div class="diff-preview-dialog__split-header">Modified</div>
          <div class="diff-preview-dialog__split-content">
            <DiffLine
              v-for="(line, index) in sideBySideDiff?.right ?? []"
              :key="`right-${index}`"
              :type="line.type"
              :content="line.content"
              :modified-line-number="line.lineNumber"
              :show-original-line-number="false"
              :show-modified-line-number="true"
              :highlight-syntax="true"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- Footer with actions -->
    <div class="diff-preview-dialog__footer">
      <div class="diff-preview-dialog__footer-info">
        <span v-if="diffResult?.filename">
          {{ diffResult.filename }}
        </span>
      </div>
      <div class="diff-preview-dialog__actions">
        <button
          class="diff-preview-dialog__btn diff-preview-dialog__btn--cancel"
          @click="handleCancel"
          :disabled="isApplying"
        >
          Cancel
        </button>
        <button
          class="diff-preview-dialog__btn diff-preview-dialog__btn--apply"
          @click="handleApply"
          :disabled="!canApply || isApplying"
        >
          <span v-if="isApplying" class="diff-preview-dialog__btn-spinner"></span>
          {{ isApplying ? 'Applying...' : 'Apply Changes' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import DiffLine from '@/components/diff/DiffLine.vue'
import type {
  DiffResult,
  SideBySideDiff,
  DiffViewMode,
} from '@/composables/useDiffPreview'

// =============================================================================
// PROPS
// =============================================================================

interface Props {
  /** Unified diff result */
  diffResult: DiffResult | null
  /** Side-by-side diff data */
  sideBySideDiff: SideBySideDiff | null
  /** Current view mode */
  viewMode: DiffViewMode
  /** File path being modified */
  filePath: string | null
  /** Whether diff is loading */
  isLoading: boolean
  /** Error message */
  error: string | null
  /** Whether applying changes */
  isApplying: boolean
}

const props = defineProps<Props>()

// =============================================================================
// EMITS
// =============================================================================

const emit = defineEmits<{
  (e: 'apply'): void
  (e: 'cancel'): void
  (e: 'set-view-mode', mode: DiffViewMode): void
}>()

// =============================================================================
// COMPUTED
// =============================================================================

const canApply = computed(() => {
  return (
    props.diffResult?.hasChanges &&
    props.filePath &&
    !props.isLoading &&
    !props.error
  )
})

// =============================================================================
// METHODS
// =============================================================================

function handleApply() {
  emit('apply')
}

function handleCancel() {
  emit('cancel')
}

function setViewMode(mode: DiffViewMode) {
  emit('set-view-mode', mode)
}
</script>

<style scoped>
.diff-preview-dialog {
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100%;
  max-height: 80vh;
  min-height: 400px;
  background-color: var(--flowforge-bg-primary, #0d1117);
  color: var(--flowforge-text, #c9d1d9);
  border-radius: 8px;
  overflow: hidden;
}

/* Header */
.diff-preview-dialog__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 16px 20px;
  background-color: var(--flowforge-bg-secondary, #161b22);
  border-bottom: 1px solid var(--flowforge-border, #30363d);
  flex-shrink: 0;
}

.diff-preview-dialog__title-section {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.diff-preview-dialog__title {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: var(--flowforge-text, #c9d1d9);
}

.diff-preview-dialog__filepath {
  font-size: 12px;
  color: var(--flowforge-text-muted, #8b949e);
  font-family: 'JetBrains Mono', monospace;
}

.diff-preview-dialog__stats {
  display: flex;
  gap: 8px;
}

.diff-preview-dialog__stat {
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
  font-family: 'JetBrains Mono', monospace;
}

.diff-preview-dialog__stat--additions {
  background-color: rgba(46, 160, 67, 0.2);
  color: #3fb950;
}

.diff-preview-dialog__stat--deletions {
  background-color: rgba(248, 81, 73, 0.2);
  color: #f85149;
}

.diff-preview-dialog__stat--changes {
  background-color: rgba(56, 139, 253, 0.2);
  color: #58a6ff;
}

.diff-preview-dialog__view-toggle {
  display: flex;
  gap: 4px;
  background-color: var(--flowforge-bg-tertiary, #21262d);
  padding: 4px;
  border-radius: 6px;
}

.diff-preview-dialog__view-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border: none;
  border-radius: 4px;
  background-color: transparent;
  color: var(--flowforge-text-muted, #8b949e);
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.diff-preview-dialog__view-btn:hover {
  background-color: var(--flowforge-bg-hover, #30363d);
  color: var(--flowforge-text, #c9d1d9);
}

.diff-preview-dialog__view-btn--active {
  background-color: var(--flowforge-accent, #238636);
  color: white;
}

/* Loading state */
.diff-preview-dialog__loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  flex: 1;
  color: var(--flowforge-text-muted, #8b949e);
}

.diff-preview-dialog__spinner {
  width: 24px;
  height: 24px;
  border: 2px solid var(--flowforge-border, #30363d);
  border-top-color: var(--flowforge-accent, #58a6ff);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Error state */
.diff-preview-dialog__error {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  flex: 1;
  color: #f85149;
  padding: 20px;
}

/* No changes state */
.diff-preview-dialog__no-changes {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  flex: 1;
  color: #3fb950;
}

/* Content area */
.diff-preview-dialog__content {
  flex: 1;
  overflow: auto;
  background-color: var(--flowforge-bg-primary, #0d1117);
}

/* Unified view */
.diff-preview-dialog__unified {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
}

/* Split view */
.diff-preview-dialog__split {
  display: flex;
  height: 100%;
}

.diff-preview-dialog__split-pane {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.diff-preview-dialog__split-divider {
  width: 1px;
  background-color: var(--flowforge-border, #30363d);
  flex-shrink: 0;
}

.diff-preview-dialog__split-header {
  padding: 8px 12px;
  background-color: var(--flowforge-bg-secondary, #161b22);
  border-bottom: 1px solid var(--flowforge-border, #30363d);
  font-size: 12px;
  font-weight: 500;
  color: var(--flowforge-text-muted, #8b949e);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  flex-shrink: 0;
}

.diff-preview-dialog__split-pane--left .diff-preview-dialog__split-header {
  color: #f85149;
}

.diff-preview-dialog__split-pane--right .diff-preview-dialog__split-header {
  color: #3fb950;
}

.diff-preview-dialog__split-content {
  flex: 1;
  overflow: auto;
}

/* Footer */
.diff-preview-dialog__footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 20px;
  background-color: var(--flowforge-bg-secondary, #161b22);
  border-top: 1px solid var(--flowforge-border, #30363d);
  flex-shrink: 0;
}

.diff-preview-dialog__footer-info {
  font-size: 12px;
  color: var(--flowforge-text-muted, #8b949e);
  font-family: 'JetBrains Mono', monospace;
}

.diff-preview-dialog__actions {
  display: flex;
  gap: 8px;
}

.diff-preview-dialog__btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s ease;
  min-width: 100px;
}

.diff-preview-dialog__btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.diff-preview-dialog__btn--cancel {
  background-color: var(--flowforge-bg-tertiary, #21262d);
  color: var(--flowforge-text, #c9d1d9);
  border: 1px solid var(--flowforge-border, #30363d);
}

.diff-preview-dialog__btn--cancel:hover:not(:disabled) {
  background-color: var(--flowforge-bg-hover, #30363d);
}

.diff-preview-dialog__btn--apply {
  background-color: #238636;
  color: white;
}

.diff-preview-dialog__btn--apply:hover:not(:disabled) {
  background-color: #2ea043;
}

.diff-preview-dialog__btn-spinner {
  width: 14px;
  height: 14px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
</style>
