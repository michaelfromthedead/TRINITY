<template>
  <div class="new-node-dialog">
    <div class="dialog-content">
      <!-- Node Type Selection -->
      <div class="field-group">
        <label class="field-label">Node Type</label>
        <div class="type-buttons">
          <button
            v-for="typeInfo in nodeTypes"
            :key="typeInfo.type"
            type="button"
            class="type-button"
            :class="{ active: selectedType === typeInfo.type }"
            :style="getTypeButtonStyle(typeInfo.type)"
            @click="selectType(typeInfo.type)"
          >
            <i :class="typeInfo.icon" />
            <span class="type-label">{{ typeInfo.label }}</span>
          </button>
        </div>
      </div>

      <!-- Class Name Input -->
      <div class="field-group">
        <label for="class-name" class="field-label">Class Name</label>
        <InputText
          id="class-name"
          v-model="className"
          :placeholder="classNamePlaceholder"
          class="field-input"
          :class="{ 'input-error': nameError }"
          autofocus
          @keydown.enter="handleSubmit"
          @keydown.escape="handleCancel"
        />
        <span v-if="nameError" class="error-message">{{ nameError }}</span>
        <span v-else class="help-text">Must be a valid Python class name (PascalCase recommended)</span>
      </div>

      <!-- Decorator Preview -->
      <div class="preview-section">
        <label class="field-label">Preview</label>
        <div class="code-preview">
          <pre><code>{{ decoratorPreview }}</code></pre>
        </div>
      </div>

      <!-- Initial Fields (Optional) -->
      <div class="field-group">
        <label class="field-label">
          Initial Fields
          <span class="optional-badge">(optional)</span>
        </label>
        <div class="fields-list">
          <div
            v-for="(field, index) in initialFields"
            :key="index"
            class="field-item"
          >
            <InputText
              v-model="field.name"
              placeholder="name"
              class="field-name-input"
            />
            <span class="field-separator">:</span>
            <Select
              v-model="field.type"
              :options="fieldTypeOptions"
              option-label="label"
              option-value="value"
              placeholder="type"
              class="field-type-select"
            />
            <button
              type="button"
              class="remove-field-btn"
              @click="removeField(index)"
              title="Remove field"
            >
              <i class="icon-[lucide--x]" />
            </button>
          </div>
          <button
            type="button"
            class="add-field-btn"
            @click="addField"
          >
            <i class="icon-[lucide--plus]" />
            <span>Add field</span>
          </button>
        </div>
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
        :style="getCreateButtonStyle()"
        @click="handleSubmit"
      >
        Create {{ selectedTypeInfo?.label }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, reactive, onMounted, watch } from 'vue'
import InputText from 'primevue/inputtext'
import Select from 'primevue/select'
import { TRINITY_COLORS } from '@/nodes/nodeTheme'
import { PYTHON_KEYWORDS, FIELD_TYPE_OPTIONS } from '@/constants/python'
import type { TrinityNodeType, NodeField } from '@/composables/useNodeEditing'

// =============================================================================
// TYPES
// =============================================================================

export interface NewNodeResult {
  type: TrinityNodeType
  className: string
  fields: NodeField[]
  position?: [number, number]
}

export interface NewNodeDialogProps {
  /** Existing class names to check for duplicates */
  existingClassNames?: string[]
  /** Initial type selection */
  initialType?: TrinityNodeType
  /** Position where the node will be created */
  position?: [number, number]
}

interface TypeInfo {
  type: TrinityNodeType
  label: string
  icon: string
  description: string
  defaultClassName: string
}

interface FieldInput {
  name: string
  type: string
}

// =============================================================================
// PROPS & EMITS
// =============================================================================

const props = withDefaults(defineProps<NewNodeDialogProps>(), {
  existingClassNames: () => [],
  initialType: 'component'
})

const emit = defineEmits<{
  (e: 'confirm', result: NewNodeResult): void
  (e: 'cancel'): void
}>()

// =============================================================================
// NODE TYPES
// =============================================================================

const nodeTypes: TypeInfo[] = [
  {
    type: 'component',
    label: 'Component',
    icon: 'icon-[lucide--box]',
    description: 'Data container attached to entities',
    defaultClassName: 'NewComponent'
  },
  {
    type: 'system',
    label: 'System',
    icon: 'icon-[lucide--cpu]',
    description: 'Logic that processes entities',
    defaultClassName: 'NewSystem'
  },
  {
    type: 'resource',
    label: 'Resource',
    icon: 'icon-[lucide--database]',
    description: 'Singleton shared data',
    defaultClassName: 'NewResource'
  },
  {
    type: 'event',
    label: 'Event',
    icon: 'icon-[lucide--zap]',
    description: 'Trigger with payload data',
    defaultClassName: 'NewEvent'
  }
]

const fieldTypeOptions = FIELD_TYPE_OPTIONS

// =============================================================================
// STATE
// =============================================================================

const selectedType = ref<TrinityNodeType>(props.initialType)
const className = ref('')
const initialFields = reactive<FieldInput[]>([])

// =============================================================================
// COMPUTED
// =============================================================================

/**
 * Selected type info
 */
const selectedTypeInfo = computed(() => {
  return nodeTypes.find(t => t.type === selectedType.value)
})

/**
 * Placeholder for class name input
 */
const classNamePlaceholder = computed(() => {
  return selectedTypeInfo.value?.defaultClassName || 'ClassName'
})

/**
 * Validate the class name
 */
const nameError = computed(() => {
  const name = className.value.trim()

  if (!name) {
    return null // Don't show error for empty field initially
  }

  // Check Python identifier pattern
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
    if (/^[0-9]/.test(name)) {
      return 'Class name cannot start with a number'
    }
    if (/\s/.test(name)) {
      return 'Class name cannot contain spaces'
    }
    return 'Class name can only contain letters, numbers, and underscores'
  }

  // Check for reserved keywords
  if (PYTHON_KEYWORDS.has(name)) {
    return `'${name}' is a Python reserved keyword`
  }

  // Check for duplicates
  if (props.existingClassNames.includes(name)) {
    return `Class '${name}' already exists in this file`
  }

  return null
})

/**
 * Code preview showing the decorator and class
 */
const decoratorPreview = computed(() => {
  const name = className.value.trim() || selectedTypeInfo.value?.defaultClassName || 'ClassName'
  const decorator = `@${selectedType.value}`

  // Build fields string
  let fieldsStr = ''
  const validFields = initialFields.filter(f => f.name.trim() && f.type)
  if (validFields.length > 0) {
    fieldsStr = validFields
      .map(f => `    ${f.name.trim()}: ${f.type}`)
      .join('\n')
  } else {
    fieldsStr = '    pass'
  }

  return `${decorator}\nclass ${name}:\n${fieldsStr}`
})

/**
 * Check if the form is valid for submission
 */
const isValid = computed(() => {
  const name = className.value.trim()
  if (!name) return false
  if (nameError.value) return false
  return true
})

// =============================================================================
// METHODS
// =============================================================================

/**
 * Select a node type
 */
function selectType(type: TrinityNodeType) {
  selectedType.value = type

  // Update class name to match type if empty or still has default name
  const currentName = className.value.trim()
  const isDefaultName = nodeTypes.some(t =>
    currentName === t.defaultClassName ||
    currentName === generateUniqueName(t.defaultClassName)
  )

  if (!currentName || isDefaultName) {
    const typeInfo = nodeTypes.find(t => t.type === type)
    if (typeInfo) {
      className.value = generateUniqueName(typeInfo.defaultClassName)
    }
  }
}

/**
 * Generate a unique name by appending a number if needed
 */
function generateUniqueName(baseName: string): string {
  if (!props.existingClassNames.includes(baseName)) {
    return baseName
  }

  let counter = 1
  let candidate = `${baseName}${counter}`
  while (props.existingClassNames.includes(candidate)) {
    counter++
    candidate = `${baseName}${counter}`
  }
  return candidate
}

/**
 * Add a new empty field
 */
function addField() {
  initialFields.push({ name: '', type: 'float' })
}

/**
 * Remove a field by index
 */
function removeField(index: number) {
  initialFields.splice(index, 1)
}

/**
 * Get button style for a type
 */
function getTypeButtonStyle(type: TrinityNodeType): Record<string, string> {
  const color = TRINITY_COLORS[type].primary
  const isActive = selectedType.value === type

  if (isActive) {
    return {
      '--btn-color': 'white',
      '--btn-bg': color,
      '--btn-border': color
    }
  }

  return {
    '--btn-color': color,
    '--btn-bg': 'transparent',
    '--btn-border': color
  }
}

/**
 * Get create button style based on selected type
 */
function getCreateButtonStyle(): Record<string, string> {
  const color = TRINITY_COLORS[selectedType.value].primary
  const darkColor = TRINITY_COLORS[selectedType.value].primaryDark

  return {
    '--create-btn-bg': color,
    '--create-btn-hover': darkColor
  }
}

/**
 * Handle form submission
 */
function handleSubmit() {
  if (!isValid.value) return

  const validFields: NodeField[] = initialFields
    .filter(f => f.name.trim() && f.type)
    .map(f => ({
      name: f.name.trim(),
      type: f.type
    }))

  const result: NewNodeResult = {
    type: selectedType.value,
    className: className.value.trim(),
    fields: validFields
  }

  if (props.position) {
    result.position = props.position
  }

  emit('confirm', result)
}

/**
 * Handle cancel
 */
function handleCancel() {
  emit('cancel')
}

// =============================================================================
// LIFECYCLE
// =============================================================================

onMounted(() => {
  // Set initial class name based on type
  const typeInfo = nodeTypes.find(t => t.type === props.initialType)
  if (typeInfo) {
    className.value = generateUniqueName(typeInfo.defaultClassName)
  }

  // Focus the class name input
  const input = document.getElementById('class-name')
  if (input) {
    input.focus()
    ;(input as HTMLInputElement).select()
  }
})

// Watch for type changes to update default class name
watch(selectedType, () => {
  const currentName = className.value.trim()
  const defaultNames = nodeTypes.map(t => t.defaultClassName)

  // Only auto-update if current name looks like a default
  if (!currentName || defaultNames.some(d => currentName.startsWith(d.replace(/\d+$/, '')))) {
    const typeInfo = nodeTypes.find(t => t.type === selectedType.value)
    if (typeInfo) {
      className.value = generateUniqueName(typeInfo.defaultClassName)
    }
  }
})
</script>

<style scoped>
.new-node-dialog {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  min-width: 440px;
  max-width: 520px;
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
  gap: 0.5rem;
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

/* Type Buttons */
.type-buttons {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.5rem;
}

.type-button {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.375rem;
  padding: 0.75rem 0.5rem;
  border-radius: 0.5rem;
  border: 2px solid var(--btn-border);
  background-color: var(--btn-bg);
  color: var(--btn-color);
  cursor: pointer;
  transition: all 0.15s ease;
}

.type-button i {
  font-size: 1.25rem;
}

.type-button .type-label {
  font-size: 0.75rem;
  font-weight: 500;
}

.type-button:hover {
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

.type-button.active {
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

/* Class Name Input */
.field-input {
  width: 100%;
}

.field-input.input-error {
  border-color: #ef4444 !important;
}

.help-text {
  font-size: 0.75rem;
  color: var(--p-text-muted-color);
}

.error-message {
  font-size: 0.75rem;
  color: #ef4444;
}

/* Code Preview */
.preview-section {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.code-preview {
  background-color: var(--p-surface-100);
  border: 1px solid var(--p-surface-border);
  border-radius: 0.375rem;
  padding: 0.75rem;
  overflow-x: auto;
}

.code-preview pre {
  margin: 0;
  font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  font-size: 0.8125rem;
  line-height: 1.5;
}

.code-preview code {
  color: var(--p-text-color);
}

/* Initial Fields */
.fields-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.field-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.field-name-input {
  flex: 1;
  min-width: 0;
}

.field-separator {
  color: var(--p-text-muted-color);
  font-weight: 500;
}

.field-type-select {
  width: 120px;
}

.remove-field-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 1.75rem;
  height: 1.75rem;
  border: none;
  border-radius: 0.25rem;
  background-color: transparent;
  color: var(--p-text-muted-color);
  cursor: pointer;
  transition: all 0.15s ease;
}

.remove-field-btn:hover {
  background-color: rgba(239, 68, 68, 0.1);
  color: #ef4444;
}

.add-field-btn {
  display: flex;
  align-items: center;
  gap: 0.375rem;
  padding: 0.5rem;
  border: 1px dashed var(--p-surface-border);
  border-radius: 0.375rem;
  background-color: transparent;
  color: var(--p-text-muted-color);
  font-size: 0.8125rem;
  cursor: pointer;
  transition: all 0.15s ease;
}

.add-field-btn:hover {
  background-color: var(--p-surface-100);
  color: var(--p-text-color);
  border-style: solid;
}

.add-field-btn i {
  font-size: 0.875rem;
}

/* Dialog Footer */
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
  background-color: var(--create-btn-bg, #3b82f6);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background-color: var(--create-btn-hover, #2563eb);
}

/* Responsive */
@media (max-width: 480px) {
  .new-node-dialog {
    min-width: 100%;
  }

  .type-buttons {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
