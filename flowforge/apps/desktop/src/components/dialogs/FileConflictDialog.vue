<template>
  <div class="file-conflict-dialog">
    <!-- Header with warning icon -->
    <div class="file-conflict-dialog__header">
      <div class="file-conflict-dialog__icon">
        <i class="icon-[lucide--alert-triangle]" />
      </div>
      <div class="file-conflict-dialog__title-section">
        <h2 class="file-conflict-dialog__title">File Changed Externally</h2>
        <p class="file-conflict-dialog__subtitle">
          The file has been modified outside of FlowForge
        </p>
      </div>
    </div>

    <!-- File path -->
    <div class="file-conflict-dialog__filepath">
      <i class="icon-[lucide--file-code]" />
      <span>{{ fileName }}</span>
    </div>

    <!-- Description -->
    <div class="file-conflict-dialog__description">
      <p>
        Your local changes may conflict with the external changes.
        How would you like to proceed?
      </p>
    </div>

    <!-- Options -->
    <div class="file-conflict-dialog__options">
      <button
        class="file-conflict-dialog__option"
        @click="handleReload"
        :disabled="isLoading"
      >
        <div class="file-conflict-dialog__option-icon file-conflict-dialog__option-icon--reload">
          <i class="icon-[lucide--refresh-cw]" />
        </div>
        <div class="file-conflict-dialog__option-content">
          <span class="file-conflict-dialog__option-title">Reload from Disk</span>
          <span class="file-conflict-dialog__option-desc">
            Discard local changes and load the external version
          </span>
        </div>
      </button>

      <button
        class="file-conflict-dialog__option"
        @click="handleOverwrite"
        :disabled="isLoading"
      >
        <div class="file-conflict-dialog__option-icon file-conflict-dialog__option-icon--overwrite">
          <i class="icon-[lucide--save]" />
        </div>
        <div class="file-conflict-dialog__option-content">
          <span class="file-conflict-dialog__option-title">Overwrite File</span>
          <span class="file-conflict-dialog__option-desc">
            Keep local changes and overwrite the external version
          </span>
        </div>
      </button>

      <button
        class="file-conflict-dialog__option"
        @click="handleSaveAs"
        :disabled="isLoading"
      >
        <div class="file-conflict-dialog__option-icon file-conflict-dialog__option-icon--saveas">
          <i class="icon-[lucide--file-plus]" />
        </div>
        <div class="file-conflict-dialog__option-content">
          <span class="file-conflict-dialog__option-title">Save As...</span>
          <span class="file-conflict-dialog__option-desc">
            Save local changes to a different file
          </span>
        </div>
      </button>

      <button
        class="file-conflict-dialog__option"
        @click="handleCompare"
        :disabled="isLoading"
      >
        <div class="file-conflict-dialog__option-icon file-conflict-dialog__option-icon--compare">
          <i class="icon-[lucide--git-compare]" />
        </div>
        <div class="file-conflict-dialog__option-content">
          <span class="file-conflict-dialog__option-title">Compare Changes</span>
          <span class="file-conflict-dialog__option-desc">
            View differences between local and external versions
          </span>
        </div>
      </button>
    </div>

    <!-- Footer -->
    <div class="file-conflict-dialog__footer">
      <button
        class="file-conflict-dialog__cancel-btn"
        @click="handleCancel"
        :disabled="isLoading"
      >
        Cancel
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'

// =============================================================================
// PROPS
// =============================================================================

export interface FileConflictDialogProps {
  /** Path to the conflicted file */
  filePath: string
  /** Current in-memory content */
  localContent: string
  /** Callback for reload action (alternative to emit) */
  onReload?: () => void | Promise<void>
  /** Callback for overwrite action (alternative to emit) */
  onOverwrite?: () => void | Promise<void>
  /** Callback for save-as action (alternative to emit) */
  onSaveAs?: () => void | Promise<void>
  /** Callback for compare action (alternative to emit) */
  onCompare?: () => void | Promise<void>
  /** Callback for cancel action (alternative to emit) */
  onCancel?: () => void
}

const props = defineProps<FileConflictDialogProps>()

// =============================================================================
// EMITS
// =============================================================================

const emit = defineEmits<{
  /** User chose to reload from disk */
  (e: 'reload'): void
  /** User chose to overwrite the file */
  (e: 'overwrite'): void
  /** User chose to save to a new file */
  (e: 'save-as'): void
  /** User wants to see the diff */
  (e: 'compare'): void
  /** User cancelled the dialog */
  (e: 'cancel'): void
}>()

// =============================================================================
// STATE
// =============================================================================

const isLoading = ref(false)

// =============================================================================
// COMPUTED
// =============================================================================

/**
 * Extract file name from path for display.
 */
const fileName = computed(() => {
  if (!props.filePath) return 'Unknown file'
  const parts = props.filePath.split(/[/\\]/)
  return parts[parts.length - 1] || props.filePath
})

// =============================================================================
// METHODS
// =============================================================================

async function handleReload() {
  isLoading.value = true
  try {
    if (props.onReload) {
      await props.onReload()
    } else {
      emit('reload')
    }
  } finally {
    isLoading.value = false
  }
}

async function handleOverwrite() {
  isLoading.value = true
  try {
    if (props.onOverwrite) {
      await props.onOverwrite()
    } else {
      emit('overwrite')
    }
  } finally {
    isLoading.value = false
  }
}

async function handleSaveAs() {
  isLoading.value = true
  try {
    if (props.onSaveAs) {
      await props.onSaveAs()
    } else {
      emit('save-as')
    }
  } finally {
    isLoading.value = false
  }
}

async function handleCompare() {
  if (props.onCompare) {
    await props.onCompare()
  } else {
    emit('compare')
  }
}

function handleCancel() {
  if (props.onCancel) {
    props.onCancel()
  } else {
    emit('cancel')
  }
}
</script>

<style scoped>
.file-conflict-dialog {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
  min-width: 420px;
  max-width: 520px;
  padding: 0.5rem;
}

/* Header */
.file-conflict-dialog__header {
  display: flex;
  gap: 1rem;
  align-items: flex-start;
}

.file-conflict-dialog__icon {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2.75rem;
  height: 2.75rem;
  border-radius: 50%;
  background-color: rgba(245, 158, 11, 0.15);
  color: #f59e0b;
}

.file-conflict-dialog__icon i {
  font-size: 1.5rem;
}

.file-conflict-dialog__title-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.file-conflict-dialog__title {
  margin: 0;
  font-size: 1.125rem;
  font-weight: 600;
  color: var(--p-text-color);
}

.file-conflict-dialog__subtitle {
  margin: 0;
  font-size: 0.875rem;
  color: var(--p-text-muted-color);
}

/* File path */
.file-conflict-dialog__filepath {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  background-color: var(--p-surface-100);
  border-radius: 0.5rem;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 0.8125rem;
  color: var(--p-text-muted-color);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-conflict-dialog__filepath i {
  flex-shrink: 0;
  font-size: 1rem;
  color: var(--p-primary-color);
}

/* Description */
.file-conflict-dialog__description {
  padding: 0 0.25rem;
}

.file-conflict-dialog__description p {
  margin: 0;
  font-size: 0.875rem;
  color: var(--p-text-muted-color);
  line-height: 1.5;
}

/* Options */
.file-conflict-dialog__options {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.file-conflict-dialog__option {
  display: flex;
  align-items: center;
  gap: 0.875rem;
  padding: 0.875rem 1rem;
  background-color: var(--p-surface-50);
  border: 1px solid var(--p-surface-200);
  border-radius: 0.5rem;
  cursor: pointer;
  transition: all 0.15s ease;
  text-align: left;
}

.file-conflict-dialog__option:hover:not(:disabled) {
  background-color: var(--p-surface-100);
  border-color: var(--p-surface-300);
}

.file-conflict-dialog__option:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.file-conflict-dialog__option-icon {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2.25rem;
  height: 2.25rem;
  border-radius: 0.375rem;
}

.file-conflict-dialog__option-icon i {
  font-size: 1.125rem;
}

.file-conflict-dialog__option-icon--reload {
  background-color: rgba(59, 130, 246, 0.15);
  color: #3b82f6;
}

.file-conflict-dialog__option-icon--overwrite {
  background-color: rgba(239, 68, 68, 0.15);
  color: #ef4444;
}

.file-conflict-dialog__option-icon--saveas {
  background-color: rgba(34, 197, 94, 0.15);
  color: #22c55e;
}

.file-conflict-dialog__option-icon--compare {
  background-color: rgba(168, 85, 247, 0.15);
  color: #a855f7;
}

.file-conflict-dialog__option-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.125rem;
}

.file-conflict-dialog__option-title {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--p-text-color);
}

.file-conflict-dialog__option-desc {
  font-size: 0.75rem;
  color: var(--p-text-muted-color);
}

/* Footer */
.file-conflict-dialog__footer {
  display: flex;
  justify-content: flex-end;
  padding-top: 0.5rem;
  border-top: 1px solid var(--p-surface-border);
}

.file-conflict-dialog__cancel-btn {
  padding: 0.5rem 1rem;
  background-color: var(--p-surface-200);
  color: var(--p-text-color);
  border: none;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s ease;
}

.file-conflict-dialog__cancel-btn:hover:not(:disabled) {
  background-color: var(--p-surface-300);
}

.file-conflict-dialog__cancel-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
