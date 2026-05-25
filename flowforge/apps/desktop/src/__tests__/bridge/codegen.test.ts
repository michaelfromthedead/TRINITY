/**
 * Code Generation Bridge Tests
 *
 * Tests for the codegen bridge module that handles Python code generation,
 * validation, diff generation, and file change operations.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  generateCode,
  validateCode,
  generateDiff,
  generateDiffFromStrings,
  applyChanges,
  applyContent,
  generatePython,
  validatePython,
  isValidPython,
  generateAndSave,
  loadAndValidate,
  validateBatch,
  CodegenError,
  JsonRpcErrorCode,
  ValidationHelpers,
  GenerationHelpers,
  DiffHelpers,
} from '@/bridge/codegen';
import type { NodeGraph } from '@/bridge/api';
import {
  getMockInvoke,
  createMockNodeGraph,
  createMockIpcResponse,
  createMockIpcError,
  createMockGenerationResult,
  createMockValidationResult,
  createMockDiffResult,
  createMockApplyResult,
} from '../mocks/tauri';

describe('CodegenError', () => {
  describe('constructor', () => {
    it('should create error with default code', () => {
      const error = new CodegenError('Test error');

      expect(error.message).toBe('Test error');
      expect(error.code).toBe(JsonRpcErrorCode.INTERNAL_ERROR);
      expect(error.name).toBe('CodegenError');
      expect(error.data).toBeUndefined();
      expect(error.method).toBeUndefined();
    });

    it('should create error with all parameters', () => {
      const cause = new Error('Original error');
      const error = new CodegenError(
        'Test error',
        JsonRpcErrorCode.VALIDATION_ERROR,
        { details: 'extra info' },
        'test_method',
        cause
      );

      expect(error.message).toBe('Test error');
      expect(error.code).toBe(JsonRpcErrorCode.VALIDATION_ERROR);
      expect(error.data).toEqual({ details: 'extra info' });
      expect(error.method).toBe('test_method');
      expect(error.cause).toBe(cause);
    });
  });

  describe('fromJsonRpcError', () => {
    it('should create from JSON-RPC error object', () => {
      const rpcError = {
        code: JsonRpcErrorCode.INVALID_PARAMS,
        message: 'Invalid parameters',
        data: { param: 'value' },
      };

      const error = CodegenError.fromJsonRpcError(rpcError, 'test_method');

      expect(error.code).toBe(JsonRpcErrorCode.INVALID_PARAMS);
      expect(error.message).toBe('Invalid parameters');
      expect(error.data).toEqual({ param: 'value' });
      expect(error.method).toBe('test_method');
    });
  });

  describe('fromUnknown', () => {
    it('should return existing CodegenError unchanged', () => {
      const original = new CodegenError('Original', JsonRpcErrorCode.FILE_ERROR);
      const result = CodegenError.fromUnknown(original, 'method');

      expect(result).toBe(original);
    });

    it('should wrap standard Error', () => {
      const standardError = new Error('Standard error');
      const result = CodegenError.fromUnknown(standardError, 'method');

      expect(result.message).toBe('Standard error');
      expect(result.code).toBe(JsonRpcErrorCode.INTERNAL_ERROR);
      expect(result.method).toBe('method');
      expect(result.cause).toBe(standardError);
    });

    it('should convert string to error', () => {
      const result = CodegenError.fromUnknown('String error', 'method');

      expect(result.message).toBe('String error');
      expect(result.code).toBe(JsonRpcErrorCode.INTERNAL_ERROR);
    });
  });

  describe('error type checks', () => {
    it('should identify validation error', () => {
      const error = new CodegenError('error', JsonRpcErrorCode.VALIDATION_ERROR);
      expect(error.isValidationError()).toBe(true);
      expect(error.isGenerationError()).toBe(false);
    });

    it('should identify generation error', () => {
      const error = new CodegenError('error', JsonRpcErrorCode.GENERATION_ERROR);
      expect(error.isGenerationError()).toBe(true);
      expect(error.isValidationError()).toBe(false);
    });

    it('should identify file error', () => {
      const error = new CodegenError('error', JsonRpcErrorCode.FILE_ERROR);
      expect(error.isFileError()).toBe(true);
    });

    it('should identify method not found', () => {
      const error = new CodegenError('error', JsonRpcErrorCode.METHOD_NOT_FOUND);
      expect(error.isMethodNotFound()).toBe(true);
    });

    it('should identify sidecar error', () => {
      const error = new CodegenError('error', JsonRpcErrorCode.SIDECAR_ERROR);
      expect(error.isSidecarError()).toBe(true);
    });
  });

  describe('getUserMessage', () => {
    it('should return user-friendly message for METHOD_NOT_FOUND', () => {
      const error = new CodegenError('not found', JsonRpcErrorCode.METHOD_NOT_FOUND, undefined, 'my_method');
      expect(error.getUserMessage()).toContain('my_method');
      expect(error.getUserMessage()).toContain('not available');
    });

    it('should return user-friendly message for VALIDATION_ERROR', () => {
      const error = new CodegenError('syntax error', JsonRpcErrorCode.VALIDATION_ERROR);
      expect(error.getUserMessage()).toContain('validation failed');
    });

    it('should return user-friendly message for GENERATION_ERROR', () => {
      const error = new CodegenError('gen failed', JsonRpcErrorCode.GENERATION_ERROR);
      expect(error.getUserMessage()).toContain('generation failed');
    });

    it('should return user-friendly message for FILE_ERROR', () => {
      const error = new CodegenError('file issue', JsonRpcErrorCode.FILE_ERROR);
      expect(error.getUserMessage()).toContain('File operation failed');
    });

    it('should return user-friendly message for SIDECAR_ERROR', () => {
      const error = new CodegenError('sidecar down', JsonRpcErrorCode.SIDECAR_ERROR);
      expect(error.getUserMessage()).toContain('not available');
    });

    it('should return original message for unknown codes', () => {
      const error = new CodegenError('unknown error', -99999);
      expect(error.getUserMessage()).toBe('unknown error');
    });
  });

  describe('toJSON', () => {
    it('should serialize to plain object', () => {
      const error = new CodegenError(
        'Test error',
        JsonRpcErrorCode.VALIDATION_ERROR,
        { line: 5 },
        'validate'
      );

      const json = error.toJSON();

      expect(json).toEqual({
        name: 'CodegenError',
        message: 'Test error',
        code: JsonRpcErrorCode.VALIDATION_ERROR,
        data: { line: 5 },
        method: 'validate',
      });
    });
  });
});

describe('generateCode', () => {
  let mockInvoke: ReturnType<typeof vi.fn>;
  let mockGraph: NodeGraph;

  beforeEach(() => {
    mockInvoke = getMockInvoke();
    mockGraph = createMockNodeGraph();
  });

  it('should generate code from a graph successfully', async () => {
    const mockResult = createMockGenerationResult({
      source: '# Generated\nclass MyComponent:\n    pass\n',
      node_count: 2,
    });

    mockInvoke.mockResolvedValueOnce(createMockIpcResponse(mockResult));

    const result = await generateCode(mockGraph);

    expect(result.source).toBe('# Generated\nclass MyComponent:\n    pass\n');
    expect(result.validation.success).toBe(true);
    expect(result.nodeCount).toBe(2);
    expect(result.imports).toHaveLength(1);
    expect(result.imports[0]?.module).toBe('trinity.ecs');
  });

  it('should handle unwrapped response format', async () => {
    const mockResult = createMockGenerationResult();
    mockInvoke.mockResolvedValueOnce(mockResult);

    const result = await generateCode(mockGraph);

    expect(result.validation.success).toBe(true);
  });

  it('should pass generation options correctly', async () => {
    const mockResult = createMockGenerationResult();
    mockInvoke.mockResolvedValueOnce(createMockIpcResponse(mockResult));

    await generateCode(mockGraph, {
      formatWithBlack: false,
      addHeader: false,
      filename: 'custom.py',
    });

    expect(mockInvoke).toHaveBeenCalledWith(
      'ipc_call',
      expect.objectContaining({
        request: expect.objectContaining({
          method: 'generate_code',
          params: expect.objectContaining({
            format_with_black: false,
            add_header: false,
          }),
        }),
      })
    );
  });

  it('should throw CodegenError on IPC error response', async () => {
    mockInvoke.mockResolvedValueOnce(
      createMockIpcError(JsonRpcErrorCode.GENERATION_ERROR, 'Generation failed')
    );

    await expect(generateCode(mockGraph)).rejects.toThrow(CodegenError);
  });

  it('should throw CodegenError on invoke failure', async () => {
    mockInvoke.mockRejectedValueOnce(new Error('IPC connection failed'));

    await expect(generateCode(mockGraph)).rejects.toThrow(CodegenError);
  });

  it('should convert snake_case response to camelCase', async () => {
    const mockResult = {
      source: 'code',
      validation: {
        success: true,
        errors: [],
        warnings: [
          {
            line: 1,
            column: 0,
            message: 'warning',
            severity: 'warning',
            end_line: 1,
            end_column: 5,
          },
        ],
        source_hash: 'abc123',
      },
      imports: [
        {
          module: 'test',
          names: ['A'],
          is_from_import: true,
          line: 1,
        },
      ],
      node_count: 3,
    };

    mockInvoke.mockResolvedValueOnce(createMockIpcResponse(mockResult));

    const result = await generateCode(mockGraph);

    expect(result.validation.sourceHash).toBe('abc123');
    expect(result.validation.warnings[0]?.endLine).toBe(1);
    expect(result.validation.warnings[0]?.endColumn).toBe(5);
    expect(result.imports[0]?.isFromImport).toBe(true);
    expect(result.nodeCount).toBe(3);
  });
});

describe('validateCode', () => {
  let mockInvoke: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockInvoke = getMockInvoke();
  });

  it('should validate valid Python code', async () => {
    mockInvoke.mockResolvedValueOnce(
      createMockIpcResponse(createMockValidationResult(true))
    );

    const result = await validateCode('def foo(): return 42');

    expect(result.success).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it('should return errors for invalid code', async () => {
    const errors = [
      { line: 1, column: 0, message: 'Syntax error', severity: 'error' as const },
    ];
    mockInvoke.mockResolvedValueOnce(
      createMockIpcResponse(createMockValidationResult(false, errors))
    );

    const result = await validateCode('def foo(');

    expect(result.success).toBe(false);
    expect(result.errors).toHaveLength(1);
    expect(result.errors[0]?.message).toBe('Syntax error');
  });

  it('should pass validation options', async () => {
    mockInvoke.mockResolvedValueOnce(
      createMockIpcResponse(createMockValidationResult(true))
    );

    await validateCode('x = 1', { checkSemantics: true, filename: 'test.py' });

    expect(mockInvoke).toHaveBeenCalledWith(
      'ipc_call',
      expect.objectContaining({
        request: expect.objectContaining({
          params: expect.objectContaining({
            check_semantics: true,
          }),
        }),
      })
    );
  });
});

describe('generateDiff', () => {
  let mockInvoke: ReturnType<typeof vi.fn>;
  let mockGraph: NodeGraph;

  beforeEach(() => {
    mockInvoke = getMockInvoke();
    mockGraph = createMockNodeGraph();
  });

  it('should generate diff between file and graph', async () => {
    // Mock read_python_file
    mockInvoke.mockResolvedValueOnce({
      path: '/path/to/test.py',
      content: '# Original content\n',
    });

    // Mock generate_code
    mockInvoke.mockResolvedValueOnce(
      createMockIpcResponse(createMockGenerationResult())
    );

    // Mock generate_diff
    mockInvoke.mockResolvedValueOnce(
      createMockIpcResponse(createMockDiffResult({ hasChanges: true }))
    );

    const result = await generateDiff('/path/to/test.py', mockGraph);

    expect(result.hasChanges).toBe(true);
    expect(result.stats.additions).toBeGreaterThan(0);
  });

  it('should return empty diff when no changes', async () => {
    mockInvoke.mockResolvedValueOnce({
      path: '/path/to/test.py',
      content: '# Same content\n',
    });

    mockInvoke.mockResolvedValueOnce(
      createMockIpcResponse(createMockGenerationResult({ source: '# Same content\n' }))
    );

    mockInvoke.mockResolvedValueOnce(
      createMockIpcResponse(createMockDiffResult({ hasChanges: false }))
    );

    const result = await generateDiff('/path/to/test.py', mockGraph);

    expect(result.hasChanges).toBe(false);
    expect(result.hunks).toHaveLength(0);
  });

  it('should pass diff options', async () => {
    mockInvoke.mockResolvedValueOnce({
      path: '/path/to/test.py',
      content: 'content',
    });

    mockInvoke.mockResolvedValueOnce(
      createMockIpcResponse(createMockGenerationResult())
    );

    mockInvoke.mockResolvedValueOnce(
      createMockIpcResponse(createMockDiffResult())
    );

    await generateDiff('/path/to/test.py', mockGraph, {
      filename: 'custom.py',
      contextLines: 5,
    });

    expect(mockInvoke).toHaveBeenLastCalledWith(
      'ipc_call',
      expect.objectContaining({
        request: expect.objectContaining({
          params: expect.objectContaining({
            context_lines: 5,
          }),
        }),
      })
    );
  });

  it('should convert snake_case diff result to camelCase', async () => {
    mockInvoke.mockResolvedValueOnce({
      path: '/path/to/test.py',
      content: 'content',
    });

    mockInvoke.mockResolvedValueOnce(
      createMockIpcResponse(createMockGenerationResult())
    );

    mockInvoke.mockResolvedValueOnce(
      createMockIpcResponse(createMockDiffResult())
    );

    const result = await generateDiff('/path/to/test.py', mockGraph);

    expect(result.originalPath).toBeDefined();
    expect(result.hasChanges).toBeDefined();
    expect(result.unifiedDiff).toBeDefined();
    expect(result.hunks[0]?.originalStart).toBeDefined();
    expect(result.hunks[0]?.modifiedCount).toBeDefined();
  });
});

describe('generateDiffFromStrings', () => {
  let mockInvoke: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockInvoke = getMockInvoke();
  });

  it('should generate diff between two strings', async () => {
    mockInvoke.mockResolvedValueOnce(
      createMockIpcResponse(createMockDiffResult({ additions: 5, deletions: 2 }))
    );

    const result = await generateDiffFromStrings('old code', 'new code');

    expect(result.stats.additions).toBe(5);
    expect(result.stats.deletions).toBe(2);
  });
});

describe('applyChanges', () => {
  let mockInvoke: ReturnType<typeof vi.fn>;
  let mockGraph: NodeGraph;

  beforeEach(() => {
    mockInvoke = getMockInvoke();
    mockGraph = createMockNodeGraph();
  });

  it('should apply changes successfully', async () => {
    // Mock generate_code
    mockInvoke.mockResolvedValueOnce(
      createMockIpcResponse(createMockGenerationResult())
    );

    // Mock apply_changes
    mockInvoke.mockResolvedValueOnce(
      createMockIpcResponse(createMockApplyResult(true))
    );

    const result = await applyChanges('/path/to/test.py', mockGraph);

    expect(result.success).toBe(true);
    expect(result.backupPath).toBe('/path/to/test.py.bak');
  });

  it('should return failure when generation fails', async () => {
    mockInvoke.mockResolvedValueOnce(
      createMockIpcResponse(
        createMockGenerationResult({
          success: false,
          errors: [{ line: 1, column: 0, message: 'Error', severity: 'error' }],
        })
      )
    );

    const result = await applyChanges('/path/to/test.py', mockGraph);

    expect(result.success).toBe(false);
    expect(result.error).toContain('Code generation failed');
  });

  it('should skip backup when specified', async () => {
    mockInvoke.mockResolvedValueOnce(
      createMockIpcResponse(createMockGenerationResult())
    );

    mockInvoke.mockResolvedValueOnce(
      createMockIpcResponse(createMockApplyResult(true))
    );

    await applyChanges('/path/to/test.py', mockGraph, false);

    expect(mockInvoke).toHaveBeenLastCalledWith(
      'ipc_call',
      expect.objectContaining({
        request: expect.objectContaining({
          params: expect.objectContaining({
            create_backup: false,
          }),
        }),
      })
    );
  });
});

describe('applyContent', () => {
  let mockInvoke: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockInvoke = getMockInvoke();
  });

  it('should apply content directly', async () => {
    mockInvoke.mockResolvedValueOnce(
      createMockIpcResponse(createMockApplyResult(true))
    );

    const result = await applyContent('/path/to/test.py', '# New content\n');

    expect(result.success).toBe(true);
  });

  it('should handle apply failure gracefully', async () => {
    mockInvoke.mockResolvedValueOnce(
      createMockIpcResponse(createMockApplyResult(false))
    );

    const result = await applyContent('/path/to/test.py', 'content');

    expect(result.success).toBe(false);
    expect(result.error).toBe('Mock apply error');
  });

  it('should catch exceptions and return error result', async () => {
    mockInvoke.mockRejectedValueOnce(new Error('Write permission denied'));

    const result = await applyContent('/path/to/test.py', 'content');

    expect(result.success).toBe(false);
    expect(result.error).toContain('permission denied');
  });
});

describe('Legacy API Compatibility', () => {
  let mockInvoke: ReturnType<typeof vi.fn>;
  let mockGraph: NodeGraph;

  beforeEach(() => {
    mockInvoke = getMockInvoke();
    mockGraph = createMockNodeGraph();
  });

  describe('generatePython', () => {
    it('should return result with node_count (snake_case)', async () => {
      mockInvoke.mockResolvedValueOnce(
        createMockIpcResponse(createMockGenerationResult({ node_count: 5 }))
      );

      const result = await generatePython(mockGraph);

      expect(result.node_count).toBe(5);
    });
  });

  describe('validatePython', () => {
    it('should be an alias for validateCode', async () => {
      mockInvoke.mockResolvedValueOnce(
        createMockIpcResponse(createMockValidationResult(true))
      );

      const result = await validatePython('def foo(): pass');

      expect(result.success).toBe(true);
    });
  });

  describe('isValidPython', () => {
    it('should return true for valid code', async () => {
      mockInvoke.mockResolvedValueOnce(
        createMockIpcResponse(createMockValidationResult(true))
      );

      const isValid = await isValidPython('def foo(): pass');

      expect(isValid).toBe(true);
    });

    it('should return false for invalid code', async () => {
      mockInvoke.mockResolvedValueOnce(
        createMockIpcResponse(createMockValidationResult(false))
      );

      const isValid = await isValidPython('def foo(');

      expect(isValid).toBe(false);
    });
  });

  describe('generateAndSave', () => {
    it('should generate and save when valid', async () => {
      mockInvoke.mockResolvedValueOnce(
        createMockIpcResponse(createMockGenerationResult())
      );

      mockInvoke.mockResolvedValueOnce(undefined); // write_python_file returns void

      const result = await generateAndSave(mockGraph, '/path/to/output.py');

      expect(result.validation.success).toBe(true);
      expect(mockInvoke).toHaveBeenCalledWith('write_python_file', {
        path: '/path/to/output.py',
        content: expect.any(String),
      });
    });

    it('should not save when validation fails', async () => {
      mockInvoke.mockResolvedValueOnce(
        createMockIpcResponse(
          createMockGenerationResult({
            success: false,
            errors: [{ line: 1, column: 0, message: 'Error', severity: 'error' }],
          })
        )
      );

      const result = await generateAndSave(mockGraph, '/path/to/output.py');

      expect(result.validation.success).toBe(false);
      // write_python_file should not be called
      expect(mockInvoke).toHaveBeenCalledTimes(1);
    });
  });

  describe('loadAndValidate', () => {
    it('should load and validate a file', async () => {
      mockInvoke.mockResolvedValueOnce({
        path: '/path/to/test.py',
        content: 'def foo(): pass\n',
      });

      mockInvoke.mockResolvedValueOnce(
        createMockIpcResponse(createMockValidationResult(true))
      );

      const result = await loadAndValidate('/path/to/test.py');

      expect(result.source).toBe('def foo(): pass\n');
      expect(result.validation.success).toBe(true);
    });
  });

  describe('validateBatch', () => {
    it('should validate multiple sources in parallel', async () => {
      mockInvoke
        .mockResolvedValueOnce(createMockIpcResponse(createMockValidationResult(true)))
        .mockResolvedValueOnce(createMockIpcResponse(createMockValidationResult(true)))
        .mockResolvedValueOnce(createMockIpcResponse(createMockValidationResult(false)));

      const results = await validateBatch([
        { source: 'code1' },
        { source: 'code2', filename: 'file2.py' },
        { source: 'invalid' },
      ]);

      expect(results).toHaveLength(3);
      expect(results[0]?.success).toBe(true);
      expect(results[1]?.success).toBe(true);
      expect(results[2]?.success).toBe(false);
    });
  });
});

describe('ValidationHelpers', () => {
  it('should check for errors', () => {
    expect(ValidationHelpers.hasErrors({ success: false, errors: [{ line: 1, column: 0, message: 'err', severity: 'error' }], warnings: [] })).toBe(true);
    expect(ValidationHelpers.hasErrors({ success: true, errors: [], warnings: [] })).toBe(false);
  });

  it('should check for warnings', () => {
    expect(ValidationHelpers.hasWarnings({ success: true, errors: [], warnings: [{ line: 1, column: 0, message: 'warn', severity: 'warning' }] })).toBe(true);
    expect(ValidationHelpers.hasWarnings({ success: true, errors: [], warnings: [] })).toBe(false);
  });

  it('should get all issues combined', () => {
    const result = {
      success: false,
      errors: [{ line: 1, column: 0, message: 'err', severity: 'error' as const }],
      warnings: [{ line: 2, column: 0, message: 'warn', severity: 'warning' as const }],
    };

    const issues = ValidationHelpers.getAllIssues(result);
    expect(issues).toHaveLength(2);
  });

  it('should get summary', () => {
    expect(ValidationHelpers.getSummary({ success: true, errors: [], warnings: [] })).toBe('Valid');
    expect(ValidationHelpers.getSummary({ success: false, errors: [{ line: 1, column: 0, message: 'e', severity: 'error' }], warnings: [] })).toContain('1 error');
    expect(ValidationHelpers.getSummary({ success: true, errors: [], warnings: [{ line: 1, column: 0, message: 'w', severity: 'warning' }] })).toContain('1 warning');
  });

  it('should format error', () => {
    const error = { line: 5, column: 10, message: 'Test error', severity: 'error' as const, code: 'E001' };
    const formatted = ValidationHelpers.formatError(error);

    expect(formatted).toContain('5:10');
    expect(formatted).toContain('E001');
    expect(formatted).toContain('Test error');
  });

  it('should format multiline error', () => {
    const error = {
      line: 5,
      column: 10,
      message: 'Test error',
      severity: 'error' as const,
      endLine: 7,
      endColumn: 5,
    };
    const formatted = ValidationHelpers.formatError(error);

    expect(formatted).toContain('5:10-7:5');
  });

  it('should create success result', () => {
    const result = ValidationHelpers.createSuccess();
    expect(result.success).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it('should create failure result', () => {
    const errors = [{ line: 1, column: 0, message: 'Error', severity: 'error' as const }];
    const result = ValidationHelpers.createFailure(errors);

    expect(result.success).toBe(false);
    expect(result.errors).toEqual(errors);
  });
});

describe('GenerationHelpers', () => {
  it('should check success', () => {
    expect(GenerationHelpers.isSuccess({ source: '', validation: { success: true, errors: [], warnings: [] }, imports: [], nodeCount: 0 })).toBe(true);
    expect(GenerationHelpers.isSuccess({ source: '', validation: { success: false, errors: [], warnings: [] }, imports: [], nodeCount: 0 })).toBe(false);
  });

  it('should create empty result', () => {
    const result = GenerationHelpers.createEmpty();

    expect(result.source).toBe('');
    expect(result.validation.success).toBe(true);
    expect(result.nodeCount).toBe(0);
  });

  it('should create error result', () => {
    const result = GenerationHelpers.createError('Generation failed', 5, 10);

    expect(result.validation.success).toBe(false);
    expect(result.validation.errors[0]?.line).toBe(5);
    expect(result.validation.errors[0]?.column).toBe(10);
    expect(result.validation.errors[0]?.message).toBe('Generation failed');
  });
});

describe('DiffHelpers', () => {
  it('should check for changes', () => {
    expect(DiffHelpers.hasChanges({ filename: '', originalPath: null, hasChanges: true, hunks: [], stats: { additions: 1, deletions: 0, changes: 1 }, unifiedDiff: '' })).toBe(true);
    expect(DiffHelpers.hasChanges({ filename: '', originalPath: null, hasChanges: false, hunks: [], stats: { additions: 0, deletions: 0, changes: 0 }, unifiedDiff: '' })).toBe(false);
  });

  it('should get total changes', () => {
    const result = {
      filename: '',
      originalPath: null,
      hasChanges: true,
      hunks: [],
      stats: { additions: 5, deletions: 3, changes: 8 },
      unifiedDiff: '',
    };

    expect(DiffHelpers.getTotalChanges(result)).toBe(8);
  });

  it('should format stats', () => {
    expect(DiffHelpers.formatStats({ filename: '', originalPath: null, hasChanges: true, hunks: [], stats: { additions: 5, deletions: 3, changes: 8 }, unifiedDiff: '' })).toBe('+5, -3');
    expect(DiffHelpers.formatStats({ filename: '', originalPath: null, hasChanges: false, hunks: [], stats: { additions: 0, deletions: 0, changes: 0 }, unifiedDiff: '' })).toBe('No changes');
  });

  it('should create empty diff', () => {
    const result = DiffHelpers.createEmpty('test.py');

    expect(result.filename).toBe('test.py');
    expect(result.hasChanges).toBe(false);
    expect(result.hunks).toHaveLength(0);
  });
});
