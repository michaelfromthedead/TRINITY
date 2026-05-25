<template>
  <div class="add-field-dialog">
    <div class="dialog-content">
      <!-- Field Name Input -->
      <div class="field-group">
        <label for="field-name" class="field-label">Field Name</label>
        <InputText
          id="field-name"
          v-model="fieldName"
          placeholder="e.g., velocity, health, position"
          class="field-input"
          :class="{ 'input-error': nameError }"
          autofocus
          @keydown.enter="handleSubmit"
          @keydown.escape="handleCancel"
        />
        <span v-if="nameError" class="error-message">{{ nameError }}</span>
        <span v-else class="help-text">Must be a valid Python identifier</span>
      </div>

      <!-- Field Type Selection -->
      <div class="field-group">
        <label for="field-type" class="field-label">Field Type</label>
        <div class="type-selection">
          <Select
            v-model="selectedType"
            :options="typeOptions"
            option-label="label"
            option-value="value"
            placeholder="Select type"
            class="type-select"
            :class="{ 'select-custom': selectedType === 'custom' }"
          />
          <InputText
            v-if="selectedType === 'custom'"
            v-model="customType"
            placeholder="Custom type name"
            class="custom-type-input"
            @keydown.enter="handleSubmit"
            @keydown.escape="handleCancel"
          />
        </div>
        <span class="help-text">
          {{ selectedType === 'custom' ? 'Enter your custom Python type' : 'Select a common type or choose custom' }}
        </span>
      </div>

      <!-- Default Value Input (Optional) -->
      <div class="field-group">
        <label for="default-value" class="field-label">
          Default Value
          <span class="optional-badge">(optional)</span>
        </label>
        <InputText
          id="default-value"
          v-model="defaultValue"
          :placeholder="defaultPlaceholder"
          class="field-input"
          @keydown.enter="handleSubmit"
          @keydown.escape="handleCancel"
        />
        <span class="help-text">Python literal value (e.g., 0.0, "string", True, None)</span>
      </div>
    </div>

    <!-- Dialog Footer -->
    <div class="dialog-footer">
      <button
        type="button"
        class="btn btn-secondary"
        @click="handleCancel"
      >
        Cancel
      </button>
      <button
        type="button"
        class="btn btn-primary"
        :disabled="!isValid"
        @click="handleSubmit"
      >
        Add Field
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, onMounted } from 'vue'
import InputText from 'primevue/inputtext'
import Select from 'primevue/select'
import { TRINITY_COLORS } from '@/nodes/nodeTheme'
import { PYTHON_KEYWORDS, EXTENDED_FIELD_TYPE_OPTIONS } from '@/constants/python'

// Note: TRINITY_COLORS is used in the style section via v-bind

// =============================================================================
// TYPES
// =============================================================================

export interface AddFieldResult {
  name: string
  type: string
  default?: string
}

export interface AddFieldDialogProps {
  /** Existing field names to check for duplicates */
  existingFields?: string[]
  /** Optional initial values */
  initialName?: string
  initialType?: string
  initialDefault?: string
}

// =============================================================================
// PROPS & EMITS
// =============================================================================

const props = withDefaults(defineProps<AddFieldDialogProps>(), {
  existingFields: () => [],
  initialName: '',
  initialType: 'float',
  initialDefault: ''
})

const emit = defineEmits<{
  (e: 'confirm', result: AddFieldResult): void
  (e: 'cancel'): void
}>()

// =============================================================================
// TYPE OPTIONS
// =============================================================================

const typeOptions = EXTENDED_FIELD_TYPE_OPTIONS

// =============================================================================
// STATE
// =============================================================================

const fieldName = ref(props.initialName)
const selectedType = ref(
  typeOptions.some(t => t.value === props.initialType)
    ? props.initialType
    : 'custom'
)
const customType = ref(
  typeOptions.some(t => t.value === props.initialType)
    ? ''
    : props.initialType
)
const defaultValue = ref(props.initialDefault)

// =============================================================================
// COMPUTED
// =============================================================================

/**
 * Validate the field name
 */
const nameError = computed(() => {
  const name = fieldName.value.trim()

  if (!name) {
    return null // Don't show error for empty field initially
  }

  // Check Python identifier pattern
  if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name)) {
    return 'Must start with letter or underscore, contain only letters, numbers, underscores'
  }

  // Check for reserved keywords
  if (PYTHON_KEYWORDS.has(name)) {
    return `'${name}' is a Python reserved keyword`
  }

  // Check for duplicates
  if (props.existingFields.includes(name)) {
    return `Field '${name}' already exists`
  }

  return null
})

/**
 * Get the effective type value
 */
const effectiveType = computed(() => {
  if (selectedType.value === 'custom') {
    return customType.value.trim() || 'Any'
  }
  return selectedType.value
})

/**
 * Placeholder for default value based on type
 */
const defaultPlaceholder = computed(() => {
  const type = effectiveType.value
  switch (type) {
    case 'float': return 'e.g., 0.0, 1.5'
    case 'int': return 'e.g., 0, 42'
    case 'str': return 'e.g., "hello"'
    case 'bool': return 'e.g., True, False'
    case 'list': return 'e.g., []'
    case 'dict': return 'e.g., {}'
    case 'Vec2': return 'e.g., Vec2(0, 0)'
    case 'Vec3': return 'e.g., Vec3(0, 0, 0)'
    case 'Entity': return 'e.g., None'
    case 'Optional': return 'e.g., None'
    default: return 'Python literal value'
  }
})

/**
 * Check if the form is valid for submission
 */
const isValid = computed(() => {
  const name = fieldName.value.trim()
  if (!name) return false
  if (nameError.value) return false
  if (selectedType.value === 'custom' && !customType.value.trim()) return false
  return true
})

// =============================================================================
// METHODS
// =============================================================================

function handleSubmit() {
  if (!isValid.value) return

  const result: AddFieldResult = {
    name: fieldName.value.trim(),
    type: effectiveType.value
  }

  const defaultVal = defaultValue.value.trim()
  if (defaultVal) {
    result.default = defaultVal
  }

  emit('confirm', result)
}

function handleCancel() {
  emit('cancel')
}

// =============================================================================
// LIFECYCLE
// =============================================================================

onMounted(() => {
  // Focus the name input on mount
  const input = document.getElementById('field-name')
  if (input) {
    input.focus()
  }
})
</script>

<style scoped>
.add-field-dialog {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  min-width: 380px;
  padding: 0.5rem;
}

.dialog-content {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

.field-group {
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
}

.field-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--p-text-color);
}

.optional-badge {
  font-size: 0.75rem;
  font-weight: 400;
  color: var(--p-text-muted-color);
  margin-left: 0.25rem;
}

.field-input {
  width: 100%;
}

.field-input.input-error {
  border-color: #ef4444 !important;
}

.type-selection {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.type-select {
  width: 100%;
}

.custom-type-input {
  width: 100%;
}

.help-text {
  font-size: 0.75rem;
  color: var(--p-text-muted-color);
}

.error-message {
  font-size: 0.75rem;
  color: #ef4444;
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
</style>
