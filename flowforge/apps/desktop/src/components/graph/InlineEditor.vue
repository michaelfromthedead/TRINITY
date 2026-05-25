<template>
  <Teleport to="body">
    <Transition name="fade">
      <div
        v-if="isVisible"
        ref="editorRef"
        class="inline-editor-overlay"
        :style="overlayStyle"
        @click.stop
        @mousedown.stop
        @pointerdown.stop
      >
        <div
          class="inline-editor-container"
          :class="{
            'is-valid': !hasError && localValue.length > 0,
            'is-invalid': hasError,
            'edit-class-name': type === 'class-name',
            'edit-field-name': type === 'field-name',
            'edit-field-type': type === 'field-type',
            'edit-default-value': type === 'default-value'
          }"
        >
          <input
            ref="inputRef"
            v-model="localValue"
            type="text"
            class="inline-editor-input"
            :placeholder="computedPlaceholder"
            :style="inputStyle"
            @keydown.enter.prevent="handleConfirm"
            @keydown.escape.prevent="handleCancel"
            @keydown.tab.prevent="handleConfirm"
            @blur="handleBlur"
            @input="updateInputWidth"
          />
          <div v-if="hasError" class="inline-editor-error">
            {{ errorMessage }}
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch, type CSSProperties } from 'vue'
import { isValidPythonIdentifier, isPythonKeyword } from '@/constants/python'

// =============================================================================
// Types
// =============================================================================

export type InlineEditorType = 'class-name' | 'field-name' | 'field-type' | 'default-value'

export interface InlineEditorProps {
  /** Whether the editor is visible */
  isVisible: boolean
  /** Position on canvas */
  position: { x: number; y: number }
  /** Current value being edited */
  value: string
  /** Type of edit (affects validation and placeholder) */
  type: InlineEditorType
  /** Placeholder text */
  placeholder?: string
  /** Existing names for duplicate checking */
  existingNames?: string[]
}

// =============================================================================
// Props & Emits
// =============================================================================

const props = withDefaults(defineProps<InlineEditorProps>(), {
  placeholder: '',
  existingNames: () => []
})

const emit = defineEmits<{
  /** Emitted when edit is confirmed (Enter key) */
  (e: 'confirm', value: string): void
  /** Emitted when edit is cancelled (Escape key) */
  (e: 'cancel'): void
}>()

// =============================================================================
// Refs
// =============================================================================

const editorRef = ref<HTMLElement | null>(null)
const inputRef = ref<HTMLInputElement | null>(null)
const localValue = ref(props.value)
const isConfirming = ref(false)
const originalValue = ref(props.value)
const inputWidth = ref(100)

// Hidden span for measuring text width
const measureSpan = ref<HTMLSpanElement | null>(null)

// =============================================================================
// Constants
// =============================================================================

const MIN_INPUT_WIDTH = 100
const MAX_INPUT_WIDTH = 300
const INPUT_PADDING = 24 // Padding on both sides

// =============================================================================
// Computed
// =============================================================================

const computedPlaceholder = computed(() => {
  if (props.placeholder) return props.placeholder

  switch (props.type) {
    case 'class-name':
      return 'ClassName'
    case 'field-name':
      return 'field_name'
    case 'field-type':
      return 'str'
    case 'default-value':
      return 'None'
    default:
      return ''
  }
})

const validationResult = computed(() => {
  return validateValue(localValue.value, props.type, props.existingNames)
})

const hasError = computed(() => !validationResult.value.isValid)

const errorMessage = computed(() => validationResult.value.error)

const overlayStyle = computed<CSSProperties>(() => {
  return {
    position: 'fixed',
    left: `${props.position.x}px`,
    top: `${props.position.y}px`,
    zIndex: 10000
  }
})

const inputStyle = computed<CSSProperties>(() => ({
  width: `${inputWidth.value}px`
}))

// =============================================================================
// Validation Functions
// =============================================================================

interface ValidationResult {
  isValid: boolean
  error: string | null
}

function validateValue(
  value: string,
  type: InlineEditorType,
  existingNames: string[]
): ValidationResult {
  if (!value.trim()) {
    return { isValid: false, error: 'Value cannot be empty' }
  }

  const trimmedValue = value.trim()

  switch (type) {
    case 'class-name':
      return validateClassName(trimmedValue, existingNames)
    case 'field-name':
      return validateFieldName(trimmedValue, existingNames)
    case 'field-type':
      return validateFieldType(trimmedValue)
    case 'default-value':
      return validateDefaultValue(trimmedValue)
    default:
      return { isValid: true, error: null }
  }
}

function validateClassName(value: string, existingNames: string[]): ValidationResult {
  // Check if it's a valid Python identifier
  if (!isValidPythonIdentifier(value)) {
    if (isPythonKeyword(value)) {
      return { isValid: false, error: `"${value}" is a Python keyword` }
    }
    return { isValid: false, error: 'Invalid class name format' }
  }

  // Check PascalCase format (starts with uppercase)
  if (!/^[A-Z]/.test(value)) {
    return { isValid: false, error: 'Class name must start with uppercase' }
  }

  // Check for duplicates (case-insensitive for class names)
  const lowerValue = value.toLowerCase()
  const isDuplicate = existingNames.some(
    name => name.toLowerCase() === lowerValue && name !== value
  )
  if (isDuplicate) {
    return { isValid: false, error: `Class "${value}" already exists` }
  }

  return { isValid: true, error: null }
}

function validateFieldName(value: string, existingNames: string[]): ValidationResult {
  // Check if it's a valid Python identifier
  if (!isValidPythonIdentifier(value)) {
    if (isPythonKeyword(value)) {
      return { isValid: false, error: `"${value}" is a Python keyword` }
    }
    return { isValid: false, error: 'Invalid field name format' }
  }

  // Check snake_case format (lowercase with underscores)
  if (!/^[a-z_][a-z0-9_]*$/.test(value)) {
    return { isValid: false, error: 'Use snake_case for field names' }
  }

  // Check for duplicates
  if (existingNames.includes(value)) {
    return { isValid: false, error: `Field "${value}" already exists` }
  }

  return { isValid: true, error: null }
}

function validateFieldType(value: string): ValidationResult {
  // Field types are more permissive - just check it's not empty
  // Valid types include: int, str, float, bool, list, dict, Optional[X], List[X], etc.
  if (!value.trim()) {
    return { isValid: false, error: 'Type cannot be empty' }
  }

  // Basic check for valid type characters
  if (!/^[A-Za-z_][A-Za-z0-9_\[\],\s]*$/.test(value)) {
    return { isValid: false, error: 'Invalid type format' }
  }

  return { isValid: true, error: null }
}

function validateDefaultValue(value: string): ValidationResult {
  // Validate as Python literal
  const trimmed = value.trim()

  if (!trimmed) {
    return { isValid: false, error: 'Default value cannot be empty' }
  }

  // Check for common valid Python literals
  // None, True, False
  if (['None', 'True', 'False'].includes(trimmed)) {
    return { isValid: true, error: null }
  }

  // Numeric literals (int, float, negative numbers)
  if (/^-?\d+(\.\d+)?$/.test(trimmed)) {
    return { isValid: true, error: null }
  }

  // String literals (single or double quoted)
  if (/^["'].*["']$/.test(trimmed)) {
    // Check matching quotes
    const firstChar = trimmed[0]
    const lastChar = trimmed[trimmed.length - 1]
    if (firstChar === lastChar) {
      return { isValid: true, error: null }
    }
    return { isValid: false, error: 'Mismatched string quotes' }
  }

  // List literals
  if (/^\[.*\]$/.test(trimmed)) {
    return { isValid: true, error: null }
  }

  // Dict literals
  if (/^\{.*\}$/.test(trimmed)) {
    return { isValid: true, error: null }
  }

  // Tuple literals
  if (/^\(.*\)$/.test(trimmed)) {
    return { isValid: true, error: null }
  }

  // Variable/constant references (valid identifiers)
  if (isValidPythonIdentifier(trimmed)) {
    return { isValid: true, error: null }
  }

  // Function calls like Vec2(0, 0)
  if (/^[A-Za-z_][A-Za-z0-9_]*\(.*\)$/.test(trimmed)) {
    return { isValid: true, error: null }
  }

  return { isValid: false, error: 'Invalid Python literal' }
}

// =============================================================================
// Methods
// =============================================================================

function updateInputWidth(): void {
  // Create a temporary span to measure text width
  if (!measureSpan.value) {
    measureSpan.value = document.createElement('span')
    measureSpan.value.style.cssText = `
      position: absolute;
      visibility: hidden;
      white-space: pre;
      font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
      font-size: 13px;
    `
    document.body.appendChild(measureSpan.value)
  }

  measureSpan.value.textContent = localValue.value || computedPlaceholder.value
  const measuredWidth = measureSpan.value.offsetWidth + INPUT_PADDING

  inputWidth.value = Math.max(MIN_INPUT_WIDTH, Math.min(MAX_INPUT_WIDTH, measuredWidth))
}

function focusAndSelectInput(): void {
  nextTick(() => {
    if (inputRef.value) {
      inputRef.value.focus()
      inputRef.value.select()
    }
  })
}

function handleConfirm(): void {
  if (hasError.value) {
    // Shake the input to indicate error
    inputRef.value?.classList.add('shake')
    setTimeout(() => {
      inputRef.value?.classList.remove('shake')
    }, 300)
    return
  }

  isConfirming.value = true
  emit('confirm', localValue.value.trim())
  isConfirming.value = false
}

function handleCancel(): void {
  emit('cancel')
}

function handleBlur(): void {
  // Small delay to check if we're in the middle of confirming
  setTimeout(() => {
    if (!isConfirming.value && props.isVisible) {
      // If value unchanged, treat as cancel
      if (localValue.value.trim() === originalValue.value.trim()) {
        handleCancel()
      } else if (!hasError.value) {
        // If value changed and valid, confirm
        handleConfirm()
      }
      // If invalid, don't do anything - keep editor open
    }
  }, 100)
}

function cleanup(): void {
  if (measureSpan.value && measureSpan.value.parentNode) {
    measureSpan.value.parentNode.removeChild(measureSpan.value)
    measureSpan.value = null
  }
}

// =============================================================================
// Watchers
// =============================================================================

watch(
  () => props.value,
  (newValue) => {
    localValue.value = newValue
    originalValue.value = newValue
    nextTick(updateInputWidth)
  }
)

watch(
  () => props.isVisible,
  async (isVisible) => {
    if (isVisible) {
      localValue.value = props.value
      originalValue.value = props.value
      await nextTick()
      updateInputWidth()
      focusAndSelectInput()
    } else {
      cleanup()
    }
  },
  { immediate: true }
)

// =============================================================================
// Lifecycle
// =============================================================================

onMounted(() => {
  if (props.isVisible) {
    updateInputWidth()
    focusAndSelectInput()
  }
})

// =============================================================================
// Expose
// =============================================================================

defineExpose({
  focus: focusAndSelectInput,
  inputRef
})
</script>

<style scoped>
.inline-editor-overlay {
  pointer-events: auto;
}

.inline-editor-container {
  position: relative;
  background: var(--p-surface-0, #1a1a1a);
  border: 2px solid var(--p-surface-400, #525252);
  border-radius: 4px;
  box-shadow:
    0 4px 12px rgba(0, 0, 0, 0.4),
    0 0 0 1px rgba(0, 0, 0, 0.2);
  overflow: visible;
  transition: border-color 0.15s ease;
}

/* Validation states */
.inline-editor-container.is-valid {
  border-color: var(--p-green-500, #22c55e);
}

.inline-editor-container.is-invalid {
  border-color: var(--p-red-500, #ef4444);
}

/* Type-specific styling when valid */
.inline-editor-container.is-valid.edit-class-name {
  border-color: var(--p-primary-500, #3b82f6);
}

.inline-editor-container.is-valid.edit-field-name {
  border-color: var(--p-cyan-500, #06b6d4);
}

.inline-editor-container.is-valid.edit-field-type {
  border-color: var(--p-purple-500, #a855f7);
}

.inline-editor-container.is-valid.edit-default-value {
  border-color: var(--p-green-500, #22c55e);
}

.inline-editor-input {
  min-width: 100px;
  max-width: 300px;
  padding: 6px 12px;
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
  color: var(--p-text-color, #e5e5e5);
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  font-size: 13px;
  line-height: 1.4;
  box-sizing: border-box;
}

.inline-editor-input::placeholder {
  color: var(--p-text-muted-color, #6b7280);
  opacity: 0.6;
}

.inline-editor-error {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  padding: 4px 8px;
  margin-top: 4px;
  background: var(--p-red-600, #dc2626);
  color: white;
  font-size: 11px;
  font-family: system-ui, -apple-system, sans-serif;
  border-radius: 4px;
  white-space: nowrap;
  z-index: 1;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

/* Animations */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.15s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

@keyframes shake {
  0%,
  100% {
    transform: translateX(0);
  }
  20%,
  60% {
    transform: translateX(-4px);
  }
  40%,
  80% {
    transform: translateX(4px);
  }
}

.shake {
  animation: shake 0.3s ease-in-out;
}
</style>
