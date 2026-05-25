/**
 * Types Index
 *
 * Export all type definitions from a single entry point.
 */

export type {
  DecoratorInfo,
  DecoratorArg,
  DecoratorCategory,
  MetaclassInfo,
  InheritanceChain,
  BaseClassInfo,
  FieldInfo,
  MethodInfo,
  MethodParam,
  InspectorData,
  SourceInfo,
  TrinityType,
  InspectorSectionState,
} from './inspector'

export {
  DEFAULT_SECTION_STATE,
  EMPTY_INSPECTOR_DATA,
} from './inspector'

// Code generation types
export type {
  Severity,
  ValidationError,
  ValidationResult,
  ImportInfo,
  GenerationResult,
  GeneratePythonRequest,
  ValidatePythonRequest,
} from './codegen'

export {
  ValidationHelpers,
  GenerationHelpers,
} from './codegen'
