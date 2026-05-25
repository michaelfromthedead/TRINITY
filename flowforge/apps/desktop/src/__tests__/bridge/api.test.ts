/**
 * Tauri API Bridge Tests
 *
 * Tests for the TauriAPI class that provides the main interface
 * for backend communication via Tauri IPC.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TauriAPI, api, generateDiff, generateSideBySideDiff } from '@/bridge/api';
import type { NodeGraph, TrinityNodeTypes, AppInfo } from '@/bridge/api';
import {
  getMockInvoke,
  createMockNodeGraph,
  createMockDiffResult,
} from '../mocks/tauri';

describe('TauriAPI', () => {
  let mockInvoke: ReturnType<typeof vi.fn>;
  let tauriApi: TauriAPI;

  beforeEach(() => {
    mockInvoke = getMockInvoke();
    tauriApi = new TauriAPI();
  });

  describe('getObjectInfo', () => {
    it('should return node definitions', async () => {
      const mockNodeDefs = {
        TestNode: {
          name: 'TestNode',
          display_name: 'Test Node',
          category: 'test',
          inputs: {},
          outputs: {},
        },
      };

      mockInvoke.mockResolvedValueOnce(mockNodeDefs);

      const result = await tauriApi.getObjectInfo();

      expect(result).toEqual(mockNodeDefs);
      expect(mockInvoke).toHaveBeenCalledWith('get_object_info');
    });
  });

  describe('getNodeDefinition', () => {
    it('should return specific node definition', async () => {
      const mockDef = {
        name: 'SpecificNode',
        display_name: 'Specific Node',
        category: 'specific',
        inputs: {},
        outputs: {},
      };

      mockInvoke.mockResolvedValueOnce(mockDef);

      const result = await tauriApi.getNodeDefinition('SpecificNode');

      expect(result).toEqual(mockDef);
      expect(mockInvoke).toHaveBeenCalledWith('get_node_definition', {
        nodeType: 'SpecificNode',
      });
    });

    it('should return null for non-existent node type', async () => {
      mockInvoke.mockResolvedValueOnce(null);

      const result = await tauriApi.getNodeDefinition('NonExistent');

      expect(result).toBeNull();
    });
  });

  describe('searchNodes', () => {
    it('should search nodes by query', async () => {
      const mockResults = {
        Node1: { name: 'Node1', display_name: 'Node 1', category: 'cat1', inputs: {}, outputs: {} },
        Node2: { name: 'Node2', display_name: 'Node 2', category: 'cat2', inputs: {}, outputs: {} },
      };

      mockInvoke.mockResolvedValueOnce(mockResults);

      const result = await tauriApi.searchNodes('Node');

      expect(Object.keys(result)).toHaveLength(2);
      expect(mockInvoke).toHaveBeenCalledWith('search_nodes', {
        request: { query: 'Node', limit: undefined },
      });
    });

    it('should respect limit parameter', async () => {
      mockInvoke.mockResolvedValueOnce({});

      await tauriApi.searchNodes('test', 5);

      expect(mockInvoke).toHaveBeenCalledWith('search_nodes', {
        request: { query: 'test', limit: 5 },
      });
    });
  });

  describe('executeWorkflow', () => {
    it('should execute workflow and return execution ID', async () => {
      const mockResponse = { executionId: 'exec_123' };
      mockInvoke.mockResolvedValueOnce(mockResponse);

      const workflow = { nodes: [], links: [], version: 1 };
      const result = await tauriApi.executeWorkflow(workflow as never);

      expect(result.executionId).toBe('exec_123');
      expect(mockInvoke).toHaveBeenCalledWith('execute_workflow', {
        request: { workflow, config: undefined },
      });
    });

    it('should pass execution config', async () => {
      mockInvoke.mockResolvedValueOnce({ executionId: 'exec_456' });

      const workflow = { nodes: [], links: [], version: 1 };
      const config = { timeout: 5000 };

      await tauriApi.executeWorkflow(workflow as never, config);

      expect(mockInvoke).toHaveBeenCalledWith('execute_workflow', {
        request: { workflow, config },
      });
    });
  });

  describe('getQueueStatus', () => {
    it('should return queue status', async () => {
      const mockStatus = {
        running: ['exec_1'],
        pending: ['exec_2', 'exec_3'],
        completed: [],
      };

      mockInvoke.mockResolvedValueOnce(mockStatus);

      const result = await tauriApi.getQueueStatus();

      expect(result).toEqual(mockStatus);
      expect(mockInvoke).toHaveBeenCalledWith('get_queue_status');
    });
  });

  describe('cancelExecution', () => {
    it('should cancel execution and return true on success', async () => {
      mockInvoke.mockResolvedValueOnce(true);

      const result = await tauriApi.cancelExecution('exec_123');

      expect(result).toBe(true);
      expect(mockInvoke).toHaveBeenCalledWith('cancel_execution', {
        executionId: 'exec_123',
      });
    });

    it('should return false when execution not found', async () => {
      mockInvoke.mockResolvedValueOnce(false);

      const result = await tauriApi.cancelExecution('non_existent');

      expect(result).toBe(false);
    });
  });

  describe('getAppInfo', () => {
    it('should return application info', async () => {
      const mockInfo: AppInfo = {
        name: 'FlowForge',
        version: '1.0.0',
        tauriVersion: '2.0.0',
        platform: 'linux',
        arch: 'x86_64',
      };

      mockInvoke.mockResolvedValueOnce(mockInfo);

      const result = await tauriApi.getAppInfo();

      expect(result).toEqual(mockInfo);
      expect(mockInvoke).toHaveBeenCalledWith('get_app_info');
    });
  });

  describe('ping', () => {
    it('should return pong with timestamp', async () => {
      const now = Date.now();
      mockInvoke.mockResolvedValueOnce({ pong: true, timestamp: now });

      const result = await tauriApi.ping();

      expect(result.pong).toBe(true);
      expect(result.timestamp).toBe(now);
      expect(mockInvoke).toHaveBeenCalledWith('ping');
    });
  });

  describe('parsePythonFile', () => {
    it('should parse Python file and return node graph', async () => {
      const mockGraph = createMockNodeGraph(3, 2);
      mockInvoke.mockResolvedValueOnce(mockGraph);

      const result = await tauriApi.parsePythonFile('/path/to/game.py');

      expect(result.nodes).toHaveLength(3);
      expect(result.edges).toHaveLength(2);
      expect(mockInvoke).toHaveBeenCalledWith('parse_python_file', {
        path: '/path/to/game.py',
      });
    });

    it('should handle empty Python file', async () => {
      mockInvoke.mockResolvedValueOnce({ nodes: [], edges: [] });

      const result = await tauriApi.parsePythonFile('/path/to/empty.py');

      expect(result.nodes).toHaveLength(0);
      expect(result.edges).toHaveLength(0);
    });
  });

  describe('openPythonFile', () => {
    it('should open file dialog and return selected path', async () => {
      mockInvoke.mockResolvedValueOnce('/path/to/selected.py');

      const result = await tauriApi.openPythonFile();

      expect(result).toBe('/path/to/selected.py');
      expect(mockInvoke).toHaveBeenCalledWith('open_file_dialog', {
        request: {
          filters: [
            { name: 'Python Files', extensions: ['py'] },
            { name: 'All Files', extensions: ['*'] },
          ],
          title: 'Open Python File',
        },
      });
    });

    it('should return null when dialog is cancelled', async () => {
      mockInvoke.mockResolvedValueOnce(null);

      const result = await tauriApi.openPythonFile();

      expect(result).toBeNull();
    });
  });

  describe('savePythonFile', () => {
    it('should save Python file', async () => {
      mockInvoke.mockResolvedValueOnce(undefined);

      await tauriApi.savePythonFile('/path/to/output.py', '# Generated code\n');

      expect(mockInvoke).toHaveBeenCalledWith('write_python_file', {
        path: '/path/to/output.py',
        content: '# Generated code\n',
      });
    });
  });

  describe('getTrinityNodeTypes', () => {
    it('should return Trinity-specific node type definitions', async () => {
      const mockTypes: TrinityNodeTypes = {
        component: {
          name: 'Component',
          description: 'ECS Component',
          category: 'trinity',
          inputs: [],
          outputs: [],
          properties: [],
        },
        system: {
          name: 'System',
          description: 'ECS System',
          category: 'trinity',
          inputs: [],
          outputs: [],
          properties: [],
        },
        resource: {
          name: 'Resource',
          description: 'ECS Resource',
          category: 'trinity',
          inputs: [],
          outputs: [],
          properties: [],
        },
        event: {
          name: 'Event',
          description: 'ECS Event',
          category: 'trinity',
          inputs: [],
          outputs: [],
          properties: [],
        },
      };

      mockInvoke.mockResolvedValueOnce(mockTypes);

      const result = await tauriApi.getTrinityNodeTypes();

      expect(result.component.name).toBe('Component');
      expect(result.system.name).toBe('System');
      expect(result.resource.name).toBe('Resource');
      expect(result.event.name).toBe('Event');
      expect(mockInvoke).toHaveBeenCalledWith('get_trinity_node_types');
    });
  });
});

describe('generateDiff function', () => {
  let mockInvoke: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockInvoke = getMockInvoke();
  });

  it('should generate unified diff between two strings', async () => {
    const mockResult = createMockDiffResult({ hasChanges: true });
    mockInvoke.mockResolvedValueOnce(mockResult);

    const result = await generateDiff('original code', 'modified code');

    expect(result.has_changes).toBe(true);
    expect(mockInvoke).toHaveBeenCalledWith('ipc_call', {
      request: expect.objectContaining({
        method: 'generate_diff',
        params: expect.objectContaining({
          original: 'original code',
          modified: 'modified code',
          side_by_side: false,
        }),
      }),
    });
  });

  it('should pass options correctly', async () => {
    mockInvoke.mockResolvedValueOnce(createMockDiffResult());

    await generateDiff('original', 'modified', {
      filename: 'test.py',
      originalPath: '/path/to/original.py',
      contextLines: 5,
    });

    expect(mockInvoke).toHaveBeenCalledWith('ipc_call', {
      request: expect.objectContaining({
        params: expect.objectContaining({
          filename: 'test.py',
          original_path: '/path/to/original.py',
          context_lines: 5,
        }),
      }),
    });
  });

  it('should use default context lines from config', async () => {
    mockInvoke.mockResolvedValueOnce(createMockDiffResult());

    await generateDiff('a', 'b');

    expect(mockInvoke).toHaveBeenCalledWith('ipc_call', {
      request: expect.objectContaining({
        params: expect.objectContaining({
          context_lines: 3, // From UI_CONFIG.diff.contextLines
        }),
      }),
    });
  });
});

describe('generateSideBySideDiff function', () => {
  let mockInvoke: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockInvoke = getMockInvoke();
  });

  it('should generate side-by-side diff', async () => {
    const mockSideBySide = {
      filename: 'test.py',
      left: [
        { lineNumber: 1, content: 'old line', type: 'removed' },
      ],
      right: [
        { lineNumber: 1, content: 'new line', type: 'added' },
      ],
      leftTitle: 'Original',
      rightTitle: 'Modified',
    };

    mockInvoke.mockResolvedValueOnce(mockSideBySide);

    const result = await generateSideBySideDiff('old code', 'new code', 'test.py');

    expect(result.left).toHaveLength(1);
    expect(result.right).toHaveLength(1);
    expect(mockInvoke).toHaveBeenCalledWith('ipc_call', {
      request: expect.objectContaining({
        method: 'generate_diff',
        params: expect.objectContaining({
          side_by_side: true,
          filename: 'test.py',
        }),
      }),
    });
  });
});

describe('Default API instance', () => {
  let mockInvoke: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockInvoke = getMockInvoke();
  });

  it('should expose api singleton', () => {
    expect(api).toBeDefined();
    expect(typeof api.getObjectInfo).toBe('function');
    expect(typeof api.parsePythonFile).toBe('function');
  });

  it('should work with api singleton', async () => {
    mockInvoke.mockResolvedValueOnce({ pong: true, timestamp: Date.now() });

    const result = await api.ping();

    expect(result.pong).toBe(true);
  });
});

describe('NodeGraph type structure', () => {
  it('should have correct node structure', () => {
    const graph = createMockNodeGraph();

    expect(graph.nodes[0]).toMatchObject({
      id: expect.any(String),
      type: expect.stringMatching(/^(component|system|resource|event)$/),
      name: expect.any(String),
      position: expect.arrayContaining([expect.any(Number), expect.any(Number)]),
      data: expect.any(Object),
      source: expect.objectContaining({
        file: expect.any(String),
        line: expect.any(Number),
      }),
    });
  });

  it('should have correct edge structure', () => {
    const graph = createMockNodeGraph(3, 2);

    expect(graph.edges[0]).toMatchObject({
      id: expect.any(String),
      source: expect.any(String),
      target: expect.any(String),
      type: expect.stringMatching(/^(reference|inheritance|query)$/),
    });
  });
});

describe('Error handling', () => {
  let mockInvoke: ReturnType<typeof vi.fn>;
  let tauriApi: TauriAPI;

  beforeEach(() => {
    mockInvoke = getMockInvoke();
    tauriApi = new TauriAPI();
  });

  it('should propagate invoke errors', async () => {
    mockInvoke.mockRejectedValueOnce(new Error('Connection failed'));

    await expect(tauriApi.getObjectInfo()).rejects.toThrow('Connection failed');
  });

  it('should propagate parse errors', async () => {
    mockInvoke.mockRejectedValueOnce(new Error('Parse error: invalid syntax'));

    await expect(tauriApi.parsePythonFile('/invalid.py')).rejects.toThrow('Parse error');
  });

  it('should handle timeout errors', async () => {
    mockInvoke.mockRejectedValueOnce(new Error('Request timeout'));

    await expect(tauriApi.executeWorkflow({} as never)).rejects.toThrow('timeout');
  });
});
