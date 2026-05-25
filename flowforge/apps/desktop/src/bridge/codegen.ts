/**
 * Code Generation Bridge
 *
 * Bridge functions for Python code generation, validation, diff generation,
 * and file change operations. These functions communicate with the Python
 * backend via Tauri IPC through the sidecar process.
 */

import { invoke } from '@tauri-apps/api/core';
import type { NodeGraph } from './api';
import { UI_CONFIG } from '@/config/flowforge.config';

// ==========================================================================
// Error Types
// ==========================================================================

/**
 * JSON-RPC 2.0 error structure from Python backend.
 */
interface JsonRpcError {
  code: number;
  message: string;
  data?: unknown;
}

/**
 * Possible response shapes from the IPC layer.
 * Supports both wrapped ({ result: T }) and unwrapped (T) responses,
 * as well as error responses ({ error: JsonRpcError }).
 */
type IpcResponse<T> =
  | { result: T; error?: never }
  | { error: JsonRpcError; result?: never }
  | T;

/**
 * Type guard to check if response is an error response.
 */
function isErrorResponse<T>(
  response: IpcResponse<T>
): response is { error: JsonRpcError; result?: never } {
  return (
    typeof response === 'object' &&
    response !== null &&
    'error' in response &&
    typeof (response as { error?: unknown }).error === 'object' &&
    (response as { error: { code?: unknown } }).error?.code !== undefined
  );
}

/**
 * Type guard to check if response is a wrapped result.
 */
function isWrappedResult<T>(
  response: IpcResponse<T>
): response is { result: T; error?: never } {
  return (
    typeof response === 'object' &&
    response !== null &&
    'result' in response &&
    !('error' in response && (response as { error?: unknown }).error !== undefined)
  );
}

/**
 * Standard JSON-RPC error codes.
 */
export const JsonRpcErrorCode = {
  /** Invalid JSON was received by the server. */
  PARSE_ERROR: -32700,
  /** The JSON sent is not a valid Request object. */
  INVALID_REQUEST: -32600,
  /** The method does not exist / is not available. */
  METHOD_NOT_FOUND: -32601,
  /** Invalid method parameter(s). */
  INVALID_PARAMS: -32602,
  /** Internal JSON-RPC error. */
  INTERNAL_ERROR: -32603,
  // Server errors (-32000 to -32099) - application-specific
  /** Generic server error. */
  SERVER_ERROR: -32000,
  /** Code validation failed. */
  VALIDATION_ERROR: -32001,
  /** Code generation failed. */
  GENERATION_ERROR: -32002,
  /** File operation failed. */
  FILE_ERROR: -32003,
  /** Python sidecar not available. */
  SIDECAR_ERROR: -32004,
} as const;

export type JsonRpcErrorCodeType = (typeof JsonRpcErrorCode)[keyof typeof JsonRpcErrorCode];

/**
 * Custom error class for code generation operations.
 *
 * Provides structured error information from the Python backend,
 * including JSON-RPC error codes and optional additional data.
 */
export class CodegenError extends Error {
  /** JSON-RPC error code */
  readonly code: number;

  /** Additional error data from the backend */
  readonly data: unknown | undefined;

  /** The method that was being called when the error occurred */
  readonly method: string | undefined;

  constructor(
    message: string,
    code: number = JsonRpcErrorCode.INTERNAL_ERROR,
    data?: unknown,
    method?: string,
    cause?: Error
  ) {
    super(message, { cause });
    this.name = 'CodegenError';
    this.code = code;
    this.data = data;
    this.method = method;
  }

  /**
   * Create a CodegenError from a JSON-RPC error object.
   */
  static fromJsonRpcError(error: JsonRpcError, method?: string): CodegenError {
    return new CodegenError(error.message, error.code, error.data, method);
  }

  /**
   * Create a CodegenError from an unknown error.
   */
  static fromUnknown(error: unknown, method?: string): CodegenError {
    if (error instanceof CodegenError) {
      return error;
    }
    if (error instanceof Error) {
      return new CodegenError(
        error.message,
        JsonRpcErrorCode.INTERNAL_ERROR,
        undefined,
        method,
        error
      );
    }
    return new CodegenError(
      String(error),
      JsonRpcErrorCode.INTERNAL_ERROR,
      undefined,
      method
    );
  }

  /**
   * Check if this is a validation error.
   */
  isValidationError(): boolean {
    return this.code === JsonRpcErrorCode.VALIDATION_ERROR;
  }

  /**
   * Check if this is a generation error.
   */
  isGenerationError(): boolean {
    return this.code === JsonRpcErrorCode.GENERATION_ERROR;
  }

  /**
   * Check if this is a file-related error.
   */
  isFileError(): boolean {
    return this.code === JsonRpcErrorCode.FILE_ERROR;
  }

  /**
   * Check if the method was not found.
   */
  isMethodNotFound(): boolean {
    return this.code === JsonRpcErrorCode.METHOD_NOT_FOUND;
  }

  /**
   * Check if the Python sidecar is unavailable.
   */
  isSidecarError(): boolean {
    return this.code === JsonRpcErrorCode.SIDECAR_ERROR;
  }

  /**
   * Get a user-friendly error message.
   */
  getUserMessage(): string {
    switch (this.code) {
      case JsonRpcErrorCode.METHOD_NOT_FOUND:
        return `The operation "${this.method ?? 'unknown'}" is not available. The Python backend may not support this feature.`;
      case JsonRpcErrorCode.VALIDATION_ERROR:
        return `Code validation failed: ${this.message}`;
      case JsonRpcErrorCode.GENERATION_ERROR:
        return `Code generation failed: ${this.message}`;
      case JsonRpcErrorCode.FILE_ERROR:
        return `File operation failed: ${this.message}`;
      case JsonRpcErrorCode.INVALID_PARAMS:
        return `Invalid parameters: ${this.message}`;
      case JsonRpcErrorCode.SIDECAR_ERROR:
        return `Python backend is not available: ${this.message}`;
      case JsonRpcErrorCode.PARSE_ERROR:
        return `Failed to parse response: ${this.message}`;
      default:
        return this.message;
    }
  }

  /**
   * Convert to a plain object for serialization.
   */
  toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      message: this.message,
      code: this.code,
      data: this.data,
      method: this.method,
    };
  }
}

// ==========================================================================
// Types
// ==========================================================================

/**
 * Severity level for validation messages.
 */
export type Severity = 'error' | 'warning' | 'info';

/**
 * Represents a single validation error or warning.
 */
export interface ValidationError {
  /** Line number where the error occurred (1-indexed). */
  line: number;
  /** Column number where the error occurred (0-indexed). */
  column: number;
  /** Human-readable error message. */
  message: string;
  /** Severity level (error, warning, info). */
  severity: Severity;
  /** Optional error code for programmatic handling. */
  code?: string | undefined;
  /** Optional end line for multi-line errors. */
  endLine?: number | undefined;
  /** Optional end column. */
  endColumn?: number | undefined;
}

/**
 * Result of validating Python source code.
 */
export interface ValidationResult {
  /** True if the code is valid (no errors, warnings are ok). */
  success: boolean;
  /** List of validation errors (severity=error). */
  errors: ValidationError[];
  /** List of validation warnings (severity=warning or info). */
  warnings: ValidationError[];
  /** Optional hash of the validated source for caching. */
  sourceHash?: string | undefined;
}

/**
 * Information about an import statement.
 */
export interface ImportInfo {
  /** The module being imported (e.g., "trinity.ecs"). */
  module: string;
  /** Names imported from the module (for from imports). */
  names?: string[] | undefined;
  /** Alias if using `import X as Y`. */
  alias?: string | undefined;
  /** True if this is a `from X import Y` statement. */
  isFromImport: boolean;
  /** Line number of the import. */
  line: number;
}

/**
 * Result of generating Python code from a graph (GeneratedCode).
 */
export interface GeneratedCode {
  /** The generated Python source code. */
  source: string;
  /** Validation result for the generated code. */
  validation: ValidationResult;
  /** List of imports used in the generated code. */
  imports: ImportInfo[];
  /** Number of nodes processed. */
  nodeCount: number;
  /** Additional metadata about the generation. */
  metadata?: Record<string, unknown> | undefined;
}

/**
 * Options for code validation.
 */
export interface ValidationOptions {
  /** Whether to perform semantic checks (e.g., undefined names). */
  checkSemantics?: boolean | undefined;
  /** Optional filename for error messages. */
  filename?: string | undefined;
}

/**
 * Options for code generation.
 */
export interface GenerationOptions {
  /** Whether to format code with Black (default: true). */
  formatWithBlack?: boolean | undefined;
  /** Whether to add a header comment (default: true). */
  addHeader?: boolean | undefined;
  /** Optional filename for error messages. */
  filename?: string | undefined;
}

/**
 * Type of a diff line.
 */
export type DiffLineType = 'added' | 'removed' | 'unchanged' | 'context' | 'header' | 'empty';

/**
 * A single line in a diff.
 */
export interface DiffLine {
  type: DiffLineType;
  content: string;
  originalLine: number | null;
  modifiedLine: number | null;
}

/**
 * A contiguous block of changes.
 */
export interface DiffHunk {
  originalStart: number;
  originalCount: number;
  modifiedStart: number;
  modifiedCount: number;
  lines: DiffLine[];
}

/**
 * Statistics about the diff.
 */
export interface DiffStats {
  additions: number;
  deletions: number;
  changes: number;
}

/**
 * Complete diff result.
 */
export interface DiffResult {
  filename: string;
  originalPath: string | null;
  hasChanges: boolean;
  hunks: DiffHunk[];
  stats: DiffStats;
  unifiedDiff: string;
}

/**
 * Side-by-side diff line.
 */
export interface SideBySideLine {
  /** Line number (null for empty/padding lines) */
  lineNumber: number | null;
  /** Line content */
  content: string;
  /** Type of change */
  type: DiffLineType;
}

/**
 * Side-by-side diff result.
 */
export interface SideBySideDiff {
  /** Filename being diffed */
  filename: string;
  /** Left side (original) lines */
  left: SideBySideLine[];
  /** Right side (modified) lines */
  right: SideBySideLine[];
  /** Title for left panel (e.g., "Original") */
  leftTitle: string;
  /** Title for right panel (e.g., "Modified") */
  rightTitle: string;
}

/**
 * Options for diff generation.
 */
export interface DiffOptions {
  /** Filename for display. */
  filename?: string | undefined;
  /** Path to original file. */
  originalPath?: string | undefined;
  /** Number of context lines around changes (default: 3). */
  contextLines?: number | undefined;
}

/**
 * Result of applying changes to a file.
 */
export interface ApplyResult {
  /** Whether the changes were successfully applied. */
  success: boolean;
  /** Path to backup file if created. */
  backupPath: string | null;
  /** Error message if operation failed. */
  error: string | null;
}

// ==========================================================================
// Internal Types for Backend Communication
// ==========================================================================

/**
 * Backend response for validation (snake_case from Python).
 */
interface BackendValidationError {
  line: number;
  column: number;
  message: string;
  severity: Severity;
  code?: string;
  end_line?: number;
  end_column?: number;
}

interface BackendValidationResult {
  success: boolean;
  errors: BackendValidationError[];
  warnings: BackendValidationError[];
  source_hash?: string;
}

/**
 * Backend response for generation (snake_case from Python).
 */
interface BackendImportInfo {
  module: string;
  names?: string[];
  alias?: string;
  is_from_import: boolean;
  line: number;
}

interface BackendGenerationResult {
  source: string;
  validation: BackendValidationResult;
  imports?: BackendImportInfo[];
  node_count: number;
  metadata?: Record<string, unknown>;
}

/**
 * Backend response for diff (snake_case from Python).
 */
interface BackendDiffLine {
  type: DiffLineType;
  content: string;
  original_line: number | null;
  modified_line: number | null;
}

interface BackendDiffHunk {
  original_start: number;
  original_count: number;
  modified_start: number;
  modified_count: number;
  lines: BackendDiffLine[];
}

interface BackendDiffResult {
  filename: string;
  original_path: string | null;
  has_changes: boolean;
  hunks: BackendDiffHunk[];
  stats: DiffStats;
  unified_diff: string;
}

/**
 * Backend response for apply changes (snake_case from Python).
 */
interface BackendApplyResult {
  success: boolean;
  backup_path?: string | null;
  error?: string | null;
}

// ==========================================================================
// Conversion Utilities
// ==========================================================================

/**
 * Convert backend validation error to TypeScript format.
 */
function convertValidationError(error: BackendValidationError): ValidationError {
  return {
    line: error.line,
    column: error.column,
    message: error.message,
    severity: error.severity,
    code: error.code,
    endLine: error.end_line,
    endColumn: error.end_column,
  };
}

/**
 * Convert backend validation result to TypeScript format.
 */
function convertValidationResult(result: BackendValidationResult): ValidationResult {
  return {
    success: result.success,
    errors: result.errors.map(convertValidationError),
    warnings: result.warnings.map(convertValidationError),
    sourceHash: result.source_hash,
  };
}

/**
 * Convert backend import info to TypeScript format.
 */
function convertImportInfo(info: BackendImportInfo): ImportInfo {
  return {
    module: info.module,
    names: info.names,
    alias: info.alias,
    isFromImport: info.is_from_import,
    line: info.line,
  };
}

/**
 * Convert backend generation result to TypeScript format.
 */
function convertGenerationResult(result: BackendGenerationResult): GeneratedCode {
  return {
    source: result.source,
    validation: convertValidationResult(result.validation),
    imports: result.imports?.map(convertImportInfo) ?? [],
    nodeCount: result.node_count,
    metadata: result.metadata,
  };
}

/**
 * Convert backend diff result to TypeScript format.
 */
function convertDiffResult(result: BackendDiffResult): DiffResult {
  return {
    filename: result.filename,
    originalPath: result.original_path,
    hasChanges: result.has_changes,
    hunks: result.hunks.map((hunk) => ({
      originalStart: hunk.original_start,
      originalCount: hunk.original_count,
      modifiedStart: hunk.modified_start,
      modifiedCount: hunk.modified_count,
      lines: hunk.lines.map((line) => ({
        type: line.type,
        content: line.content,
        originalLine: line.original_line,
        modifiedLine: line.modified_line,
      })),
    })),
    stats: result.stats,
    unifiedDiff: result.unified_diff,
  };
}

/**
 * Convert backend apply result to TypeScript format.
 */
function convertApplyResult(result: BackendApplyResult): ApplyResult {
  return {
    success: result.success,
    backupPath: result.backup_path ?? null,
    error: result.error ?? null,
  };
}

// ==========================================================================
// IPC Helper
// ==========================================================================

/**
 * Call a Python backend method via Tauri IPC.
 *
 * Routes the call through the Python sidecar's JSON-RPC interface.
 * Handles JSON-RPC 2.0 error responses and various response formats.
 *
 * @param method - The JSON-RPC method name to call
 * @param params - The parameters to pass to the method
 * @returns Promise resolving to the result
 * @throws {CodegenError} If the backend returns an error or the call fails
 *
 * @example
 * ```typescript
 * try {
 *   const result = await callPythonMethod<ValidationResult>('validate_code', { source });
 * } catch (error) {
 *   if (error instanceof CodegenError) {
 *     console.error('Backend error:', error.getUserMessage());
 *   }
 * }
 * ```
 */
async function callPythonMethod<T>(method: string, params: Record<string, unknown>): Promise<T> {
  let response: IpcResponse<T>;

  try {
    response = await invoke<IpcResponse<T>>('ipc_call', {
      request: {
        id: `${method}-${Date.now()}`,
        method,
        params,
      },
    });
  } catch (invokeError) {
    // Handle Tauri invoke errors (e.g., sidecar not running, IPC failure)
    throw CodegenError.fromUnknown(invokeError, method);
  }

  // Check for JSON-RPC error response: { error: { code, message, data? } }
  if (isErrorResponse(response)) {
    throw CodegenError.fromJsonRpcError(response.error, method);
  }

  // Handle wrapped result: { result: T }
  if (isWrappedResult(response)) {
    return response.result;
  }

  // Handle unwrapped result: T (direct response)
  return response as T;
}

// ==========================================================================
// Code Generation Functions
// ==========================================================================

/**
 * Generate Python code from a node graph.
 *
 * Takes a visual node graph (representing Trinity ECS definitions)
 * and generates the corresponding Python source code.
 *
 * @param graph - The node graph to generate code from.
 * @param options - Optional generation options.
 * @returns Promise resolving to the generated code with validation.
 *
 * @example
 * ```typescript
 * const graph = await api.parsePythonFile('game.py');
 * // ... modify the graph ...
 * const result = await generateCode(graph);
 * if (result.validation.success) {
 *   console.log('Generated code:', result.source);
 * }
 * ```
 */
export async function generateCode(
  graph: NodeGraph,
  options?: GenerationOptions
): Promise<GeneratedCode> {
  const response = await callPythonMethod<BackendGenerationResult>('generate_code', {
    graph: graph as unknown as Record<string, unknown>,
    format_with_black: options?.formatWithBlack ?? true,
    add_header: options?.addHeader ?? true,
  });

  return convertGenerationResult(response);
}

/**
 * Validate Python source code.
 *
 * Parses the source with Python's ast module to check syntax,
 * and optionally performs semantic checks like undefined name detection.
 *
 * @param code - The Python source code to validate.
 * @param options - Optional validation options.
 * @returns Promise resolving to the validation result.
 *
 * @example
 * ```typescript
 * const result = await validateCode('def foo(): return 42');
 * if (!result.success) {
 *   console.error('Validation failed:', result.errors);
 * }
 * ```
 */
export async function validateCode(
  code: string,
  options?: ValidationOptions
): Promise<ValidationResult> {
  const response = await callPythonMethod<BackendValidationResult>('validate_code', {
    source: code,
    check_semantics: options?.checkSemantics ?? false,
  });

  return convertValidationResult(response);
}

/**
 * Generate a diff between original file content and generated code from a graph.
 *
 * Reads the original file, generates code from the graph, and produces
 * a unified diff showing the changes.
 *
 * @param originalPath - Path to the original file.
 * @param graph - The node graph to generate code from.
 * @param options - Optional diff options.
 * @returns Promise resolving to the diff result.
 *
 * @example
 * ```typescript
 * const diff = await generateDiff('game.py', modifiedGraph);
 * if (diff.hasChanges) {
 *   console.log('Changes detected:', diff.stats.additions, 'additions');
 * }
 * ```
 */
export async function generateDiff(
  originalPath: string,
  graph: NodeGraph,
  options?: DiffOptions
): Promise<DiffResult> {
  // First read the original file
  const originalContent = await invoke<{ path: string; content: string }>('read_python_file', {
    path: originalPath,
  });

  // Generate code from the graph
  const generated = await generateCode(graph);

  // Generate the diff
  const response = await callPythonMethod<BackendDiffResult>('generate_diff', {
    original: originalContent.content,
    modified: generated.source,
    filename: options?.filename ?? originalPath,
    original_path: options?.originalPath ?? originalPath,
    context_lines: options?.contextLines ?? UI_CONFIG.diff.contextLines,
  });

  return convertDiffResult(response);
}

/**
 * Generate a diff between two strings.
 *
 * Produces a unified diff showing the changes between original and modified.
 *
 * @param original - The original source code.
 * @param modified - The modified source code.
 * @param options - Optional diff options.
 * @returns Promise resolving to the diff result.
 *
 * @example
 * ```typescript
 * const diff = await generateDiffFromStrings(oldCode, newCode, { filename: 'game.py' });
 * console.log(diff.unifiedDiff);
 * ```
 */
export async function generateDiffFromStrings(
  original: string,
  modified: string,
  options?: DiffOptions
): Promise<DiffResult> {
  const response = await callPythonMethod<BackendDiffResult>('generate_diff', {
    original,
    modified,
    filename: options?.filename ?? '',
    original_path: options?.originalPath,
    context_lines: options?.contextLines ?? UI_CONFIG.diff.contextLines,
  });

  return convertDiffResult(response);
}

/**
 * Apply changes to a file by generating code and writing it.
 *
 * Generates Python code from the graph and writes it to the specified path.
 * Optionally creates a backup of the original file.
 *
 * @param path - The file path to write to.
 * @param graph - The node graph to generate code from.
 * @param backup - Whether to create a backup before writing (default: true).
 * @returns Promise resolving to the result of the operation.
 *
 * @example
 * ```typescript
 * const result = await applyChanges('game.py', modifiedGraph);
 * if (result.success) {
 *   console.log('Changes applied, backup at:', result.backupPath);
 * } else {
 *   console.error('Failed:', result.error);
 * }
 * ```
 */
export async function applyChanges(
  path: string,
  graph: NodeGraph,
  backup: boolean = true
): Promise<ApplyResult> {
  // Generate code from the graph
  const generated = await generateCode(graph);

  // If generation failed, return error
  if (!generated.validation.success) {
    const errorMessages = generated.validation.errors.map((e) => e.message).join('; ');
    return {
      success: false,
      backupPath: null,
      error: `Code generation failed: ${errorMessages}`,
    };
  }

  // Apply the changes
  return applyContent(path, generated.source, backup);
}

/**
 * Apply content directly to a file.
 *
 * Writes the provided content to the specified path.
 * Optionally creates a backup of the original file.
 *
 * @param path - The file path to write to.
 * @param content - The content to write.
 * @param backup - Whether to create a backup before writing (default: true).
 * @returns Promise resolving to the result of the operation.
 *
 * @example
 * ```typescript
 * const result = await applyContent('game.py', newSourceCode);
 * if (result.success) {
 *   console.log('Content written successfully');
 * }
 * ```
 */
export async function applyContent(
  path: string,
  content: string,
  backup: boolean = true
): Promise<ApplyResult> {
  try {
    const response = await callPythonMethod<BackendApplyResult>('apply_changes', {
      file_path: path,
      content,
      create_backup: backup,
    });

    return convertApplyResult(response);
  } catch (err) {
    // Provide user-friendly error message from CodegenError if available
    const errorMessage =
      err instanceof CodegenError
        ? err.getUserMessage()
        : err instanceof Error
          ? err.message
          : 'Unknown error applying changes';
    return {
      success: false,
      backupPath: null,
      error: errorMessage,
    };
  }
}

// ==========================================================================
// File Lock Functions
// ==========================================================================

/**
 * Information about an active file lock.
 */
export interface FileLockInfo {
  /** Process ID holding the lock. */
  pid: number;
  /** Unix timestamp when the lock was acquired. */
  timestamp: number;
  /** Hostname of the machine that acquired the lock. */
  hostname: string;
}

/**
 * Result of checking a file lock.
 */
export interface FileLockStatus {
  /** Whether the file is currently locked. */
  locked: boolean;
  /** Lock details if locked, null otherwise. */
  lockInfo: FileLockInfo | null;
  /** True if the lock exists but the owning process is dead. */
  stale: boolean;
}

/**
 * Backend response for file lock check (snake_case from Python).
 */
interface BackendFileLockStatus {
  locked: boolean;
  lock_info: { pid: number; timestamp: number; hostname: string } | null;
  stale: boolean;
}

/**
 * Check whether a file is currently locked.
 *
 * @param path - The file path to check.
 * @returns Promise resolving to the lock status.
 */
export async function checkFileLock(path: string): Promise<FileLockStatus> {
  const response = await callPythonMethod<BackendFileLockStatus>('check_file_lock', {
    file_path: path,
  });

  return {
    locked: response.locked,
    lockInfo: response.lock_info
      ? {
          pid: response.lock_info.pid,
          timestamp: response.lock_info.timestamp,
          hostname: response.lock_info.hostname,
        }
      : null,
    stale: response.stale,
  };
}

/**
 * Release a file lock.
 *
 * @param path - The file path to unlock.
 * @param force - If true, force-release even if owned by another process (default: false).
 * @returns Promise resolving to success/error result.
 */
export async function releaseFileLock(
  path: string,
  force: boolean = false
): Promise<{ success: boolean; error: string | null }> {
  const response = await callPythonMethod<{ success: boolean; error?: string | null }>(
    'release_file_lock',
    { file_path: path, force }
  );

  return {
    success: response.success,
    error: response.error ?? null,
  };
}

// ==========================================================================
// Legacy API Compatibility
// ==========================================================================

/**
 * Result type for legacy API compatibility.
 */
export interface GenerationResult {
  source: string;
  validation: ValidationResult;
  imports: ImportInfo[];
  node_count: number;
  metadata?: Record<string, unknown> | undefined;
}

/**
 * Generate Python code from a node graph.
 *
 * @param graph - The node graph to generate code from.
 * @param filename - Optional filename for error messages.
 * @returns Promise resolving to the generation result with source and validation.
 *
 * @example
 * ```typescript
 * const graph = await api.parsePythonFile('game.py');
 * // ... modify the graph ...
 * const result = await generatePython(graph);
 * if (result.validation.success) {
 *   await api.savePythonFile('game_modified.py', result.source);
 * }
 * ```
 */
export async function generatePython(
  graph: NodeGraph,
  filename?: string
): Promise<GenerationResult> {
  const result = await generateCode(graph, { filename });
  return {
    source: result.source,
    validation: result.validation,
    imports: result.imports,
    node_count: result.nodeCount,
    metadata: result.metadata,
  };
}

/**
 * Validate Python source code.
 *
 * Parses the source with Python's ast module to check syntax,
 * and optionally performs semantic checks like undefined name detection.
 *
 * @param source - The Python source code to validate.
 * @param options - Optional validation options.
 * @returns Promise resolving to the validation result.
 *
 * @example
 * ```typescript
 * const result = await validatePython('def foo(): return 42');
 * if (!result.success) {
 *   console.error('Validation failed:', result.errors);
 * }
 * ```
 */
export async function validatePython(
  source: string,
  options?: {
    checkSemantics?: boolean;
    filename?: string;
  }
): Promise<ValidationResult> {
  return validateCode(source, options);
}

/**
 * Quick validation check for Python source.
 *
 * A simplified version that just returns a boolean indicating
 * whether the source is valid Python syntax.
 *
 * @param source - The Python source code to validate.
 * @returns Promise resolving to true if valid, false otherwise.
 */
export async function isValidPython(source: string): Promise<boolean> {
  const result = await validateCode(source);
  return result.success;
}

/**
 * Generate and save Python code to a file.
 *
 * Convenience function that generates code and saves it in one step.
 * Only saves if generation is successful.
 *
 * @param graph - The node graph to generate code from.
 * @param path - The file path to save to.
 * @returns Promise resolving to the generation result.
 * @throws Error if saving fails.
 */
export async function generateAndSave(
  graph: NodeGraph,
  path: string
): Promise<GeneratedCode> {
  const result = await generateCode(graph, { filename: path });

  if (result.validation.success) {
    await invoke<void>('write_python_file', {
      path,
      content: result.source,
    });
  }

  return result;
}

/**
 * Load a Python file, validate it, and return the result.
 *
 * Convenience function for validating an existing file.
 *
 * @param path - The file path to load and validate.
 * @param options - Optional validation options.
 * @returns Promise resolving to validation result and source.
 */
export async function loadAndValidate(
  path: string,
  options?: {
    checkSemantics?: boolean;
  }
): Promise<{ source: string; validation: ValidationResult }> {
  // Read the file
  const { content: source } = await invoke<{ path: string; content: string }>('read_python_file', {
    path,
  });

  // Validate it
  const validation = await validateCode(source, {
    filename: path,
    checkSemantics: options?.checkSemantics,
  });

  return { source, validation };
}

/**
 * Batch validate multiple Python sources.
 *
 * Validates multiple sources in parallel for efficiency.
 *
 * @param sources - Array of sources to validate.
 * @returns Promise resolving to array of validation results.
 */
export async function validateBatch(
  sources: Array<{ source: string; filename?: string }>
): Promise<ValidationResult[]> {
  const promises = sources.map((item) =>
    validateCode(item.source, { filename: item.filename })
  );
  return Promise.all(promises);
}

// ==========================================================================
// Helper Utilities
// ==========================================================================

/**
 * Helper functions for working with validation results.
 */
export const ValidationHelpers = {
  /**
   * Check if a validation result has any errors.
   */
  hasErrors(result: ValidationResult): boolean {
    return result.errors.length > 0;
  },

  /**
   * Check if a validation result has any warnings.
   */
  hasWarnings(result: ValidationResult): boolean {
    return result.warnings.length > 0;
  },

  /**
   * Get all issues (errors and warnings) combined.
   */
  getAllIssues(result: ValidationResult): ValidationError[] {
    return [...result.errors, ...result.warnings];
  },

  /**
   * Get a human-readable summary of the validation result.
   */
  getSummary(result: ValidationResult): string {
    if (result.success && result.warnings.length === 0) {
      return 'Valid';
    }
    const parts: string[] = [];
    if (result.errors.length > 0) {
      parts.push(`${result.errors.length} error(s)`);
    }
    if (result.warnings.length > 0) {
      parts.push(`${result.warnings.length} warning(s)`);
    }
    return parts.join(', ');
  },

  /**
   * Format a validation error for display.
   */
  formatError(error: ValidationError): string {
    const location = error.endLine
      ? `${error.line}:${error.column}-${error.endLine}:${error.endColumn}`
      : `${error.line}:${error.column}`;
    const code = error.code ? `[${error.code}] ` : '';
    return `${location}: ${code}${error.message}`;
  },

  /**
   * Create an empty successful validation result.
   */
  createSuccess(): ValidationResult {
    return {
      success: true,
      errors: [],
      warnings: [],
    };
  },

  /**
   * Create a failed validation result with the given errors.
   */
  createFailure(errors: ValidationError[]): ValidationResult {
    return {
      success: false,
      errors,
      warnings: [],
    };
  },
};

/**
 * Helper functions for working with generation results.
 */
export const GenerationHelpers = {
  /**
   * Check if a generation result was successful.
   */
  isSuccess(result: GeneratedCode): boolean {
    return result.validation.success;
  },

  /**
   * Create an empty generation result.
   */
  createEmpty(): GeneratedCode {
    return {
      source: '',
      validation: ValidationHelpers.createSuccess(),
      imports: [],
      nodeCount: 0,
    };
  },

  /**
   * Create a generation result with an error.
   */
  createError(message: string, line = 1, column = 0): GeneratedCode {
    return {
      source: '',
      validation: ValidationHelpers.createFailure([
        {
          line,
          column,
          message,
          severity: 'error',
          code: 'E001',
        },
      ]),
      imports: [],
      nodeCount: 0,
    };
  },
};

/**
 * Helper functions for working with diff results.
 */
export const DiffHelpers = {
  /**
   * Check if a diff has any changes.
   */
  hasChanges(result: DiffResult): boolean {
    return result.hasChanges;
  },

  /**
   * Get total number of changes (additions + deletions).
   */
  getTotalChanges(result: DiffResult): number {
    return result.stats.additions + result.stats.deletions;
  },

  /**
   * Format diff statistics for display.
   */
  formatStats(result: DiffResult): string {
    const { additions, deletions } = result.stats;
    const parts: string[] = [];
    if (additions > 0) {
      parts.push(`+${additions}`);
    }
    if (deletions > 0) {
      parts.push(`-${deletions}`);
    }
    return parts.length > 0 ? parts.join(', ') : 'No changes';
  },

  /**
   * Create an empty diff result (no changes).
   */
  createEmpty(filename: string = ''): DiffResult {
    return {
      filename,
      originalPath: null,
      hasChanges: false,
      hunks: [],
      stats: { additions: 0, deletions: 0, changes: 0 },
      unifiedDiff: '',
    };
  },
};
