<template>
  <div class="confirm-dialog" :class="[`confirm-dialog--${type}`]">
    <div class="dialog-content">
      <!-- Icon -->
      <div class="dialog-icon" :class="[`dialog-icon--${type}`]">
        <i :class="iconClass" />
      </div>

      <!-- Message -->
      <div class="dialog-body">
        <h3 v-if="title" class="dialog-title">{{ title }}</h3>
        <p class="dialog-message">{{ message }}</p>
      </div>
    </div>

    <!-- Dialog Footer -->
    <div class="dialog-footer">
      <button
        type="button"
        class="btn btn-secondary"
        @click="handleCancel"
      >
        {{ cancelText }}
      </button>
      <button
        type="button"
        class="btn"
        :class="confirmButtonClass"
        @click="handleConfirm"
      >
        {{ confirmText }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { TRINITY_COLORS } from '@/nodes/nodeTheme'

// Note: TRINITY_COLORS is used in the style section via v-bind

// =============================================================================
// TYPES
// =============================================================================

export type ConfirmDialogType = 'info' | 'warning' | 'danger'

export interface ConfirmDialogProps {
  /** Dialog title (optional) */
  title?: string
  /** Main message to display */
  message: string
  /** Text for confirm button */
  confirmText?: string
  /** Text for cancel button */
  cancelText?: string
  /** Dialog type affects styling */
  type?: ConfirmDialogType
}

// =============================================================================
// PROPS & EMITS
// =============================================================================

const props = withDefaults(defineProps<ConfirmDialogProps>(), {
  title: '',
  confirmText: 'Confirm',
  cancelText: 'Cancel',
  type: 'info'
})

const emit = defineEmits<{
  (e: 'confirm'): void
  (e: 'cancel'): void
}>()

// =============================================================================
// COMPUTED
// =============================================================================

/**
 * Icon class based on dialog type
 */
const iconClass = computed(() => {
  switch (props.type) {
    case 'danger':
      return 'icon-[lucide--alert-triangle]'
    case 'warning':
      return 'icon-[lucide--alert-circle]'
    case 'info':
    default:
      return 'icon-[lucide--info]'
  }
})

/**
 * Confirm button class based on dialog type
 */
const confirmButtonClass = computed(() => {
  switch (props.type) {
    case 'danger':
      return 'btn-danger'
    case 'warning':
      return 'btn-warning'
    case 'info':
    default:
      return 'btn-primary'
  }
})

// =============================================================================
// METHODS
// =============================================================================

function handleConfirm() {
  emit('confirm')
}

function handleCancel() {
  emit('cancel')
}
</script>

<style scoped>
.confirm-dialog {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  min-width: 340px;
  max-width: 480px;
  padding: 0.5rem;
}

.dialog-content {
  display: flex;
  gap: 1rem;
  align-items: flex-start;
}

.dialog-icon {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 50%;
}

.dialog-icon i {
  font-size: 1.25rem;
}

.dialog-icon--info {
  background-color: rgba(59, 130, 246, 0.15);
  color: #3b82f6;
}

.dialog-icon--warning {
  background-color: rgba(245, 158, 11, 0.15);
  color: #f59e0b;
}

.dialog-icon--danger {
  background-color: rgba(239, 68, 68, 0.15);
  color: #ef4444;
}

.dialog-body {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.dialog-title {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--p-text-color);
}

.dialog-message {
  margin: 0;
  font-size: 0.875rem;
  color: var(--p-text-muted-color);
  line-height: 1.5;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 0.75rem;
  padding-top: 0.5rem;
  border-top: 1px solid var(--p-surface-border);
}

.btn {
  padding: 0.5rem 1rem;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s ease;
  border: none;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-secondary {
  background-color: var(--p-surface-200);
  color: var(--p-text-color);
}

.btn-secondary:hover:not(:disabled) {
  background-color: var(--p-surface-300);
}

.btn-primary {
  background-color: v-bind('TRINITY_COLORS.component.primary');
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background-color: v-bind('TRINITY_COLORS.component.primaryDark');
}

.btn-warning {
  background-color: #f59e0b;
  color: white;
}

.btn-warning:hover:not(:disabled) {
  background-color: #d97706;
}

.btn-danger {
  background-color: #ef4444;
  color: white;
}

.btn-danger:hover:not(:disabled) {
  background-color: #dc2626;
}
</style>
