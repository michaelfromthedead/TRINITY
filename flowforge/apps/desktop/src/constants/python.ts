/**
 * Python language constants for validation and code generation.
 */

/** Python reserved keywords - cannot be used as identifiers */
export const PYTHON_KEYWORDS = new Set([
  'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
  'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
  'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
  'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
  'while', 'with', 'yield'
])

/** Common Python/Trinity field type options for dropdowns */
export const FIELD_TYPE_OPTIONS = [
  { label: 'float', value: 'float' },
  { label: 'int', value: 'int' },
  { label: 'str', value: 'str' },
  { label: 'bool', value: 'bool' },
  { label: 'list', value: 'list' },
  { label: 'dict', value: 'dict' },
  { label: 'Vec2', value: 'Vec2' },
  { label: 'Vec3', value: 'Vec3' },
  { label: 'Entity', value: 'Entity' },
  { label: 'Any', value: 'Any' }
] as const

/** Extended field type options including Optional and Custom */
export const EXTENDED_FIELD_TYPE_OPTIONS = [
  ...FIELD_TYPE_OPTIONS,
  { label: 'Optional[...]', value: 'Optional' },
  { label: 'Custom...', value: 'custom' }
] as const

/** Check if a name is a Python keyword */
export function isPythonKeyword(name: string): boolean {
  return PYTHON_KEYWORDS.has(name)
}

/** Validate a Python identifier name */
export function isValidPythonIdentifier(name: string): boolean {
  if (!name) return false
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) return false
  if (PYTHON_KEYWORDS.has(name)) return false
  return true
}
