/**
 * Inline Edit Composable
 *
 * Manages inline editing state for Trinity node elements including:
 * - Node class names (double-click to edit)
 * - Field types (click to change)
 * - Default values (click to edit)
 *
 * Provides validation for Python identifiers and type strings.
 */

import { ref, computed, readonly, type Ref } from 'vue'
import { PYTHON_KEYWORDS, isPythonKeyword } from '@/constants/python'

// =============================================================================
// Types
// =============================================================================

/**
 * Types of inline edits supported.
 */
export type InlineEditType = 'name' | 'type' | 'default'

/**
 * Current edit state.
 */
export interface InlineEditState {
  /** ID of the node being edited */
  nodeId: string | null
  /** Type of edit in progress */
  editType: InlineEditType | null
  /** Field name (for field type/default edits) */
  fieldName: string | null
  /** Original value before editing */
  originalValue: string
  /** Current editing value */
  currentValue: string
  /** Position for the editor overlay */
  position: { x: number; y: number; width: number; height: number } | null
}

/**
 * Result of a validation check.
 */
export interface ValidationResult {
  valid: boolean
  error?: string
}

/**
 * Edit commit event data.
 */
export interface EditCommitEvent {
  nodeId: string
  editType: InlineEditType
  fieldName: string | null
  oldValue: string
  newValue: string
}

/**
 * Options for useInlineEdit composable.
 */
export interface UseInlineEditOptions {
  /** Callback when an edit is committed */
  onCommit?: (event: EditCommitEvent) => void
  /** Callback when an edit is cancelled */
  onCancel?: (nodeId: string, editType: InlineEditType) => void
  /** Custom validator for names */
  nameValidator?: (name: string) => ValidationResult
  /** Custom validator for types */
  typeValidator?: (type: string) => ValidationResult
  /** Custom validator for default values */
  defaultValidator?: (value: string, type: string) => ValidationResult
}

/**
 * Return type of useInlineEdit composable.
 */
export interface UseInlineEditReturn {
  /** Whether an edit is currently in progress */
  isEditing: Readonly<Ref<boolean>>
  /** Current edit state */
  editState: Readonly<Ref<InlineEditState>>
  /** Current validation error (if any) */
  validationError: Readonly<Ref<string | null>>
  /** Start editing a node name, field type, or default value */
  startEditing: (
    nodeId: string,
    editType: InlineEditType,
    options: {
      fieldName?: string
      currentValue: string
      position: { x: number; y: number; width: number; height: number }
    }
  ) => void
  /** Update the current editing value */
  updateValue: (value: string) => void
  /** Commit the current edit */
  commitEdit: (value?: string) => boolean
  /** Cancel the current edit */
  cancelEdit: () => void
  /** Validate a Python identifier */
  validatePythonIdentifier: (name: string) => ValidationResult
  /** Validate a type string */
  validateFieldType: (type: string) => ValidationResult
  /** Validate a default value */
  validateDefaultValue: (value: string, type: string) => ValidationResult
}

// =============================================================================
// Constants
// =============================================================================

// PYTHON_KEYWORDS imported from @/constants/python

/**
 * Built-in Python types.
 */
export const PYTHON_BUILTIN_TYPES = [
  'str', 'int', 'float', 'bool', 'bytes', 'bytearray',
  'list', 'tuple', 'dict', 'set', 'frozenset',
  'None', 'Any', 'object'
] as const

/**
 * Common typing module types.
 */
export const TYPING_TYPES = [
  'Optional', 'Union', 'List', 'Dict', 'Set', 'Tuple',
  'Callable', 'Awaitable', 'Coroutine', 'Generator',
  'Iterable', 'Iterator', 'Sequence', 'Mapping',
  'TypeVar', 'Generic', 'Protocol', 'Literal'
] as const

/**
 * Trinity ECS data types (for field types).
 */
export const TRINITY_DATA_TYPES = [
  'Entity', 'Query', 'Resource', 'Component',
  'Vec2', 'Vec3', 'Vec4', 'Mat3', 'Mat4',
  'Transform', 'Texture', 'Mesh', 'Material',
  'Sprite', 'AudioSource', 'Collider',
  'Color', 'Rect', 'Bounds'
] as const

/**
 * All recognized type names for validation.
 */
export const ALL_RECOGNIZED_TYPES: Set<string> = new Set([
  ...PYTHON_BUILTIN_TYPES,
  ...TYPING_TYPES,
  ...TRINITY_DATA_TYPES
])

// =============================================================================
// Validation Functions
// =============================================================================

/**
 * Validate a Python identifier.
 * Must start with letter or underscore, followed by letters, digits, or underscores.
 * Cannot be a Python keyword.
 */
export function validatePythonIdentifier(name: string): ValidationResult {
  if (!name || name.trim() === '') {
    return { valid: false, error: 'Name cannot be empty' }
  }

  const trimmed = name.trim()

  // Check for Python identifier pattern
  const identifierPattern = /^[a-zA-Z_][a-zA-Z0-9_]*$/
  if (!identifierPattern.test(trimmed)) {
    if (/^[0-9]/.test(trimmed)) {
      return { valid: false, error: 'Name cannot start with a number' }
    }
    if (/\s/.test(trimmed)) {
      return { valid: false, error: 'Name cannot contain spaces' }
    }
    return { valid: false, error: 'Name contains invalid characters' }
  }

  // Check for reserved keywords
  if (PYTHON_KEYWORDS.has(trimmed)) {
    return { valid: false, error: `"${trimmed}" is a Python keyword` }
  }

  // Check length
  if (trimmed.length > 255) {
    return { valid: false, error: 'Name is too long (max 255 characters)' }
  }

  return { valid: true }
}

/**
 * Validate a type string.
 * Accepts Python built-in types, typing module types, Trinity types,
 * and generic syntax like Optional[T], List[T], Dict[K, V].
 */
export function validateFieldType(type: string): ValidationResult {
  if (!type || type.trim() === '') {
    return { valid: false, error: 'Type cannot be empty' }
  }

  const trimmed = type.trim()

  // Simple type (no generics)
  if (!trimmed.includes('[')) {
    // Check if it's a recognized type or a valid identifier (custom class)
    if (ALL_RECOGNIZED_TYPES.has(trimmed)) {
      return { valid: true }
    }
    // Allow custom class names (valid Python identifiers)
    const identifierResult = validatePythonIdentifier(trimmed)
    if (identifierResult.valid) {
      return { valid: true }
    }
    return { valid: false, error: `Unknown type: ${trimmed}` }
  }

  // Generic type (e.g., Optional[str], List[int], Dict[str, Any])
  const genericPattern = /^([a-zA-Z_][a-zA-Z0-9_]*)\[(.+)\]$/
  const match = trimmed.match(genericPattern)

  if (!match) {
    return { valid: false, error: 'Invalid generic type syntax' }
  }

  const baseType = match[1]
  const typeArgs = match[2]

  // Validate base type
  if (baseType && !ALL_RECOGNIZED_TYPES.has(baseType) && !validatePythonIdentifier(baseType).valid) {
    return { valid: false, error: `Invalid base type: ${baseType}` }
  }

  // Validate type arguments (recursive, but simplified)
  if (!typeArgs) {
    return { valid: false, error: 'Missing type arguments' }
  }
  const args = parseTypeArgs(typeArgs)
  if (args === null) {
    return { valid: false, error: 'Invalid type arguments' }
  }

  for (const arg of args) {
    const argResult = validateFieldType(arg)
    if (!argResult.valid) {
      return argResult
    }
  }

  return { valid: true }
}

/**
 * Parse type arguments from a generic type.
 * Handles nested generics like Dict[str, List[int]].
 */
function parseTypeArgs(argsString: string): string[] | null {
  const args: string[] = []
  let current = ''
  let depth = 0

  for (const char of argsString) {
    if (char === '[') {
      depth++
      current += char
    } else if (char === ']') {
      depth--
      current += char
    } else if (char === ',' && depth === 0) {
      const trimmed = current.trim()
      if (trimmed) {
        args.push(trimmed)
      }
      current = ''
    } else {
      current += char
    }
  }

  const trimmed = current.trim()
  if (trimmed) {
    args.push(trimmed)
  }

  // Check for unbalanced brackets
  if (depth !== 0) {
    return null
  }

  return args
}

/**
 * Validate a default value for a given type.
 * This is a basic validation - complex expressions are allowed.
 */
export function validateDefaultValue(value: string, type: string): ValidationResult {
  if (value.trim() === '') {
    // Empty is valid (means no default)
    return { valid: true }
  }

  const trimmed = value.trim()
  const baseType = type.split('[')[0]

  // Type-specific validation
  switch (baseType) {
    case 'int':
      if (!/^-?\d+$/.test(trimmed)) {
        return { valid: false, error: 'Expected integer value' }
      }
      break

    case 'float':
      if (!/^-?\d*\.?\d+([eE][+-]?\d+)?$/.test(trimmed)) {
        return { valid: false, error: 'Expected float value' }
      }
      break

    case 'bool':
      if (trimmed !== 'True' && trimmed !== 'False') {
        return { valid: false, error: 'Expected True or False' }
      }
      break

    case 'str':
      // String literals should be quoted
      if (!(trimmed.startsWith('"') && trimmed.endsWith('"')) &&
          !(trimmed.startsWith("'") && trimmed.endsWith("'"))) {
        return { valid: false, error: 'String values should be quoted' }
      }
      break

    case 'None':
      if (trimmed !== 'None') {
        return { valid: false, error: 'Expected None' }
      }
      break

    case 'list':
    case 'List':
      if (!trimmed.startsWith('[') || !trimmed.endsWith(']')) {
        return { valid: false, error: 'Expected list literal [...]' }
      }
      break

    case 'dict':
    case 'Dict':
      if (!trimmed.startsWith('{') || !trimmed.endsWith('}')) {
        return { valid: false, error: 'Expected dict literal {...}' }
      }
      break

    case 'tuple':
    case 'Tuple':
      if (!trimmed.startsWith('(') || !trimmed.endsWith(')')) {
        return { valid: false, error: 'Expected tuple literal (...)' }
      }
      break

    // For other types (Optional, custom classes, etc.), allow any valid Python expression
    default:
      // Basic syntax check - ensure balanced brackets
      let parenDepth = 0, bracketDepth = 0, braceDepth = 0
      for (const char of trimmed) {
        if (char === '(') parenDepth++
        else if (char === ')') parenDepth--
        else if (char === '[') bracketDepth++
        else if (char === ']') bracketDepth--
        else if (char === '{') braceDepth++
        else if (char === '}') braceDepth--

        if (parenDepth < 0 || bracketDepth < 0 || braceDepth < 0) {
          return { valid: false, error: 'Unbalanced brackets' }
        }
      }
      if (parenDepth !== 0 || bracketDepth !== 0 || braceDepth !== 0) {
        return { valid: false, error: 'Unbalanced brackets' }
      }
  }

  return { valid: true }
}

// =============================================================================
// Composable
// =============================================================================

/**
 * Create an inline edit composable instance.
 */
export function useInlineEdit(options: UseInlineEditOptions = {}): UseInlineEditReturn {
  const {
    onCommit,
    onCancel,
    nameValidator = validatePythonIdentifier,
    typeValidator = validateFieldType,
    defaultValidator = validateDefaultValue
  } = options

  // State
  const editState = ref<InlineEditState>({
    nodeId: null,
    editType: null,
    fieldName: null,
    originalValue: '',
    currentValue: '',
    position: null
  })

  const validationError = ref<string | null>(null)

  // Computed
  const isEditing = computed(() => editState.value.nodeId !== null)

  // Actions
  function startEditing(
    nodeId: string,
    editType: InlineEditType,
    opts: {
      fieldName?: string
      currentValue: string
      position: { x: number; y: number; width: number; height: number }
    }
  ): void {
    // Cancel any existing edit
    if (isEditing.value) {
      cancelEdit()
    }

    editState.value = {
      nodeId,
      editType,
      fieldName: opts.fieldName ?? null,
      originalValue: opts.currentValue,
      currentValue: opts.currentValue,
      position: opts.position
    }
    validationError.value = null
  }

  function updateValue(value: string): void {
    if (!isEditing.value) return

    editState.value.currentValue = value

    // Validate based on edit type
    const state = editState.value
    let result: ValidationResult

    switch (state.editType) {
      case 'name':
        result = nameValidator(value)
        break
      case 'type':
        result = typeValidator(value)
        break
      case 'default':
        result = defaultValidator(value, '') // Type context would need to be passed
        break
      default:
        result = { valid: true }
    }

    validationError.value = result.valid ? null : (result.error ?? 'Invalid value')
  }

  function commitEdit(value?: string): boolean {
    if (!isEditing.value) return false

    const state = editState.value
    const finalValue = value ?? state.currentValue

    // Validate
    let result: ValidationResult
    switch (state.editType) {
      case 'name':
        result = nameValidator(finalValue)
        break
      case 'type':
        result = typeValidator(finalValue)
        break
      case 'default':
        result = defaultValidator(finalValue, '')
        break
      default:
        result = { valid: true }
    }

    if (!result.valid) {
      validationError.value = result.error ?? 'Invalid value'
      return false
    }

    // Notify commit
    if (onCommit && state.nodeId && state.editType) {
      onCommit({
        nodeId: state.nodeId,
        editType: state.editType,
        fieldName: state.fieldName,
        oldValue: state.originalValue,
        newValue: finalValue
      })
    }

    // Reset state
    resetState()
    return true
  }

  function cancelEdit(): void {
    if (!isEditing.value) return

    const state = editState.value
    if (onCancel && state.nodeId && state.editType) {
      onCancel(state.nodeId, state.editType)
    }

    resetState()
  }

  function resetState(): void {
    editState.value = {
      nodeId: null,
      editType: null,
      fieldName: null,
      originalValue: '',
      currentValue: '',
      position: null
    }
    validationError.value = null
  }

  return {
    isEditing: readonly(isEditing),
    editState: readonly(editState),
    validationError: readonly(validationError),
    startEditing,
    updateValue,
    commitEdit,
    cancelEdit,
    validatePythonIdentifier,
    validateFieldType,
    validateDefaultValue
  }
}

export default useInlineEdit
