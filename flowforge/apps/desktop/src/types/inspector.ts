/**
 * Inspector Panel Types
 *
 * Type definitions for the Trinity Inspector panel that displays
 * detailed information about selected nodes including class hierarchy,
 * decorators, metaclass info, fields, and methods.
 *
 * @module types/inspector
 */

// =============================================================================
// DECORATOR TYPES
// =============================================================================

/**
 * Represents a decorator applied to a Trinity class.
 */
export interface DecoratorInfo {
  /** The decorator name (e.g., "component", "system", "resource", "event") */
  name: string

  /** The decorator category for color coding */
  category: DecoratorCategory

  /** Arguments passed to the decorator */
  args?: DecoratorArg[]

  /** Raw source representation */
  source?: string
}

/**
 * Decorator argument value.
 */
export interface DecoratorArg {
  /** Argument name (for keyword args) or index (for positional) */
  key: string | number

  /** The argument value as a string representation */
  value: string

  /** The inferred or declared type */
  type?: string
}

/**
 * Categories for decorator color coding.
 */
export type DecoratorCategory =
  | 'component'   // Blue - reusable game components
  | 'system'      // Green - core engine systems
  | 'resource'    // Purple - assets, data, resources
  | 'event'       // Orange - triggers, signals, events
  | 'builtin'     // Gray - Python built-in decorators
  | 'custom'      // White - custom/unknown decorators

// =============================================================================
// METACLASS TYPES
// =============================================================================

/**
 * Information about a class's metaclass.
 */
export interface MetaclassInfo {
  /** The metaclass name */
  name: string

  /** The module where the metaclass is defined */
  module?: string

  /** Whether this is a Trinity-specific metaclass */
  isTrinityMeta: boolean

  /** Description of what the metaclass provides */
  description?: string
}

// =============================================================================
// INHERITANCE TYPES
// =============================================================================

/**
 * Represents the inheritance hierarchy of a class.
 */
export interface InheritanceChain {
  /** The base classes in order (direct parents first) */
  bases: BaseClassInfo[]

  /** The full Method Resolution Order (MRO) */
  mro?: string[]

  /** Whether this class uses multiple inheritance */
  isMultipleInheritance: boolean
}

/**
 * Information about a base class.
 */
export interface BaseClassInfo {
  /** The class name */
  name: string

  /** The module where the class is defined */
  module?: string

  /** Source file path (if available) */
  file?: string

  /** Line number in source file */
  line?: number

  /** Whether this is a Trinity base class */
  isTrinityBase: boolean
}

// =============================================================================
// FIELD TYPES
// =============================================================================

/**
 * Information about a class field/attribute.
 */
export interface FieldInfo {
  /** The field name */
  name: string

  /** The field type annotation */
  type: string

  /** Default value (as string representation) */
  default?: string

  /** Whether this field has a default value */
  hasDefault: boolean

  /** Whether this is a class variable vs instance variable */
  isClassVar: boolean

  /** Documentation string for the field */
  doc?: string

  /** Source line number */
  line?: number
}

// =============================================================================
// METHOD TYPES
// =============================================================================

/**
 * Information about a class method.
 */
export interface MethodInfo {
  /** The method name */
  name: string

  /** Full method signature */
  signature: string

  /** Return type annotation */
  returnType?: string

  /** Method parameters */
  params: MethodParam[]

  /** Decorators applied to this method */
  decorators: string[]

  /** Whether this is a static method */
  isStatic: boolean

  /** Whether this is a class method */
  isClassMethod: boolean

  /** Whether this is a property */
  isProperty: boolean

  /** Whether this is an abstract method */
  isAbstract: boolean

  /** Documentation string */
  doc?: string

  /** Source line number */
  line?: number
}

/**
 * Method parameter information.
 */
export interface MethodParam {
  /** Parameter name */
  name: string

  /** Type annotation */
  type?: string

  /** Default value */
  default?: string

  /** Whether this is *args */
  isVarPositional: boolean

  /** Whether this is **kwargs */
  isVarKeyword: boolean
}

// =============================================================================
// MAIN INSPECTOR DATA TYPE
// =============================================================================

/**
 * Complete inspector data for a selected node.
 */
export interface InspectorData {
  /** The node ID being inspected */
  nodeId: string

  /** The class name */
  className: string

  /** The module where the class is defined */
  moduleName: string

  /** Full qualified name (module.ClassName) */
  qualifiedName: string

  /** The Trinity type category */
  trinityType: TrinityType

  /** Source file location */
  source: SourceInfo

  /** Documentation string for the class */
  docstring?: string

  /** Inheritance information */
  inheritance: InheritanceChain

  /** Decorators applied to the class */
  decorators: DecoratorInfo[]

  /** Metaclass information */
  metaclass?: MetaclassInfo

  /** Class fields/attributes */
  fields: FieldInfo[]

  /** Class methods */
  methods: MethodInfo[]

  /** Raw class definition (for "View Source") */
  rawSource?: string
}

/**
 * Source file location information.
 */
export interface SourceInfo {
  /** The source file path */
  file: string

  /** Starting line number */
  line: number

  /** Ending line number (if available) */
  endLine?: number

  /** Column offset */
  column?: number
}

/**
 * Trinity type categories.
 */
export type TrinityType = 'component' | 'system' | 'resource' | 'event' | 'unknown'

// =============================================================================
// SECTION COLLAPSE STATE
// =============================================================================

/**
 * Tracks which inspector sections are collapsed.
 */
export interface InspectorSectionState {
  inheritance: boolean
  decorators: boolean
  metaclass: boolean
  fields: boolean
  methods: boolean
  source: boolean
}

// =============================================================================
// DEFAULT VALUES
// =============================================================================

/**
 * Default collapsed state for inspector sections.
 */
export const DEFAULT_SECTION_STATE: InspectorSectionState = {
  inheritance: false,
  decorators: false,
  metaclass: false,
  fields: false,
  methods: false,
  source: true, // Source is collapsed by default
}

/**
 * Empty inspector data for initialization.
 */
export const EMPTY_INSPECTOR_DATA: InspectorData = {
  nodeId: '',
  className: '',
  moduleName: '',
  qualifiedName: '',
  trinityType: 'unknown',
  source: { file: '', line: 0 },
  inheritance: {
    bases: [],
    isMultipleInheritance: false,
  },
  decorators: [],
  fields: [],
  methods: [],
}
