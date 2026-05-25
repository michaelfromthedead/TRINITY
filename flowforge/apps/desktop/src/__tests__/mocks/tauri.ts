/**
 * Tauri API Mock Utilities
 *
 * Provides mock implementations and helpers for testing Tauri IPC calls.
 */

import { vi } from 'vitest';
import type { NodeGraph, GraphNode, GraphEdge } from '@/services';
import { TEST_CONFIG } from '@/config/flowforge.config';

// =============================================================================
// GLOBAL TYPE DECLARATIONS
// =============================================================================

declare global {
  // eslint-disable-next-line no-var
  var mockTauriInvoke: ReturnType<typeof vi.fn> | undefined;
}

// =============================================================================
// MOCK INVOKE HANDLER
// =============================================================================

type InvokeHandler = (cmd: string, args?: Record<string, unknown>) => unknown;

const invokeHandlers: Map<string, InvokeHandler> = new Map();

/**
 * Get the global mock invoke function.
 * @throws {Error} If mockTauriInvoke has not been initialized
 */
export function getMockInvoke(): ReturnType<typeof vi.fn> {
  if (!globalThis.mockTauriInvoke) {
    throw new Error('mockTauriInvoke not initialized. Make sure test setup runs first.');
  }
  return globalThis.mockTauriInvoke;
}

/**
 * Register a handler for a specific Tauri command.
 */
export function registerInvokeHandler(cmd: string, handler: InvokeHandler): void {
  invokeHandlers.set(cmd, handler);
}

/**
 * Clear all registered invoke handlers.
 */
export function clearInvokeHandlers(): void {
  invokeHandlers.clear();
}

/**
 * Setup mock invoke to route to registered handlers.
 */
export function setupMockInvokeRouting(): void {
  const mockInvoke = getMockInvoke();
  mockInvoke.mockImplementation((cmd: string, args?: Record<string, unknown>) => {
    const handler = invokeHandlers.get(cmd);
    if (handler) {
      return Promise.resolve(handler(cmd, args));
    }
    return Promise.reject(new Error(`No handler registered for command: ${cmd}`));
  });
}

// =============================================================================
// MOCK DATA FACTORIES
// =============================================================================

/**
 * Create a mock GraphNode for testing.
 */
export function createMockGraphNode(overrides: Partial<GraphNode> = {}): GraphNode {
  return {
    id: `node_${Math.random().toString(36).substring(7)}`,
    type: 'component',
    name: 'TestComponent',
    position: [100, 100],
    data: {},
    source: { file: 'test.py', line: 1 },
    ...overrides,
  };
}

/**
 * Create a mock GraphEdge for testing.
 */
export function createMockGraphEdge(overrides: Partial<GraphEdge> = {}): GraphEdge {
  return {
    id: `edge_${Math.random().toString(36).substring(7)}`,
    source: 'node_1',
    target: 'node_2',
    type: 'reference',
    ...overrides,
  };
}

/**
 * Create a mock NodeGraph for testing.
 */
export function createMockNodeGraph(
  nodeCount: number = 2,
  edgeCount: number = 1
): NodeGraph {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];

  for (let i = 0; i < nodeCount; i++) {
    nodes.push(
      createMockGraphNode({
        id: `node_${i + 1}`,
        name: `TestNode${i + 1}`,
        position: [100 + i * TEST_CONFIG.mocks.nodePositionOffset, 100],
      })
    );
  }

  for (let i = 0; i < edgeCount && i < nodeCount - 1; i++) {
    edges.push(
      createMockGraphEdge({
        id: `edge_${i + 1}`,
        source: `node_${i + 1}`,
        target: `node_${i + 2}`,
      })
    );
  }

  return { nodes, edges };
}

// =============================================================================
// MOCK IPC RESPONSE FACTORIES
// =============================================================================

/**
 * Create a mock IPC success response.
 */
export function createMockIpcResponse<T>(result: T): { result: T } {
  return { result };
}

/**
 * Create a mock IPC error response.
 */
export function createMockIpcError(
  code: number,
  message: string,
  data?: unknown
): { error: { code: number; message: string; data?: unknown } } {
  return {
    error: {
      code,
      message,
      data,
    },
  };
}

/**
 * Create a mock validation result.
 */
export function createMockValidationResult(
  success: boolean = true,
  errors: Array<{
    line: number;
    column: number;
    message: string;
    severity: 'error' | 'warning' | 'info';
  }> = []
): {
  success: boolean;
  errors: typeof errors;
  warnings: typeof errors;
  source_hash?: string;
} {
  return {
    success,
    errors: errors.filter((e) => e.severity === 'error'),
    warnings: errors.filter((e) => e.severity !== 'error'),
    source_hash: 'mock_hash_' + Date.now(),
  };
}

/**
 * Create a mock generation result from Python backend.
 */
export function createMockGenerationResult(overrides: {
  source?: string;
  success?: boolean;
  node_count?: number;
  errors?: Array<{
    line: number;
    column: number;
    message: string;
    severity: 'error' | 'warning' | 'info';
  }>;
} = {}): {
  source: string;
  validation: ReturnType<typeof createMockValidationResult>;
  imports: Array<{
    module: string;
    names?: string[];
    is_from_import: boolean;
    line: number;
  }>;
  node_count: number;
  metadata?: Record<string, unknown>;
} {
  return {
    source: overrides.source ?? '# Generated Python code\nclass TestComponent:\n    pass\n',
    validation: createMockValidationResult(overrides.success ?? true, overrides.errors ?? []),
    imports: [
      { module: 'trinity.ecs', names: ['Component'], is_from_import: true, line: 1 },
    ],
    node_count: overrides.node_count ?? 1,
    metadata: { generator: 'test' },
  };
}

/**
 * Create a mock diff result from Python backend.
 */
export function createMockDiffResult(overrides: {
  hasChanges?: boolean;
  additions?: number;
  deletions?: number;
} = {}): {
  filename: string;
  original_path: string | null;
  has_changes: boolean;
  hunks: Array<{
    original_start: number;
    original_count: number;
    modified_start: number;
    modified_count: number;
    lines: Array<{
      type: string;
      content: string;
      original_line: number | null;
      modified_line: number | null;
    }>;
  }>;
  stats: { additions: number; deletions: number; changes: number };
  unified_diff: string;
} {
  const hasChanges = overrides.hasChanges ?? true;
  const additions = overrides.additions ?? 3;
  const deletions = overrides.deletions ?? 1;

  return {
    filename: 'test.py',
    original_path: '/path/to/test.py',
    has_changes: hasChanges,
    hunks: hasChanges
      ? [
          {
            original_start: 1,
            original_count: 5,
            modified_start: 1,
            modified_count: 7,
            lines: [
              { type: 'unchanged', content: '# Comment', original_line: 1, modified_line: 1 },
              { type: 'removed', content: 'old_line', original_line: 2, modified_line: null },
              { type: 'added', content: 'new_line_1', original_line: null, modified_line: 2 },
              { type: 'added', content: 'new_line_2', original_line: null, modified_line: 3 },
              { type: 'added', content: 'new_line_3', original_line: null, modified_line: 4 },
            ],
          },
        ]
      : [],
    stats: {
      additions,
      deletions,
      changes: additions + deletions,
    },
    unified_diff: hasChanges
      ? `--- a/test.py\n+++ b/test.py\n@@ -1,5 +1,7 @@\n # Comment\n-old_line\n+new_line_1\n+new_line_2\n+new_line_3`
      : '',
  };
}

/**
 * Create a mock apply result from Python backend.
 */
export function createMockApplyResult(success: boolean = true): {
  success: boolean;
  backup_path: string | null;
  error: string | null;
} {
  return {
    success,
    backup_path: success ? '/path/to/test.py.bak' : null,
    error: success ? null : 'Mock apply error',
  };
}

// =============================================================================
// MOCK FILE INFO FACTORY
// =============================================================================

/**
 * Create a mock file info response.
 */
export function createMockFileInfo(overrides: {
  path?: string;
  exists?: boolean;
  size?: number;
  modified?: number;
  is_readonly?: boolean;
} = {}): {
  path: string;
  exists: boolean;
  size?: number;
  modified?: number;
  is_readonly: boolean;
} {
  return {
    path: overrides.path ?? '/path/to/test.py',
    exists: overrides.exists ?? true,
    size: overrides.size ?? 1024,
    modified: overrides.modified ?? Date.now(),
    is_readonly: overrides.is_readonly ?? false,
  };
}
