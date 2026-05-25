/**
 * Code Generation Types
 *
 * Re-exports types from the bridge module for backwards compatibility.
 * All types are now defined in src/bridge/codegen.ts.
 */

export type {
  Severity,
  ValidationError,
  ValidationResult,
  ImportInfo,
  GeneratedCode,
  ValidationOptions,
  GenerationOptions,
  DiffLineType,
  DiffLine,
  DiffHunk,
  DiffStats,
  DiffResult,
  DiffOptions,
  ApplyResult,
  GenerationResult,
} from '../bridge/codegen.js';

export {
  ValidationHelpers,
  GenerationHelpers,
  DiffHelpers,
} from '../bridge/codegen.js';

// Legacy type aliases for backwards compatibility
export type GeneratePythonRequest = {
  graph: Record<string, unknown>;
  filename?: string | undefined;
};

export type ValidatePythonRequest = {
  source: string;
  check_semantics?: boolean | undefined;
  filename?: string | undefined;
};
