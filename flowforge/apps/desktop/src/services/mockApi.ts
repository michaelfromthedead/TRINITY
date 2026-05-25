/**
 * Mock API Implementation
 *
 * Implements FlowForgeAPI with mock data for browser development.
 * This allows the UI to be developed and tested without Tauri.
 */

import type { NodeDefinitionMap, NodeDefinition } from '@flowforge/core';
import type {
  FlowForgeAPI,
  NodeGraph,
  OpenPythonFileResult,
  TrinityNodeTypes,
  ExecutionResponse,
  ExecutionStatus,
  AppInfo,
  GraphNode,
  GraphEdge,
} from './api';

// =============================================================================
// MOCK DATA
// =============================================================================

/**
 * Mock Trinity node type definitions.
 */
const MOCK_TRINITY_NODE_TYPES: TrinityNodeTypes = {
  component: {
    name: 'Component',
    description: 'ECS Component - data container attached to entities',
    category: 'Trinity/ECS',
    inputs: [
      { name: 'entity', type: 'Entity', required: true, description: 'Target entity' },
    ],
    outputs: [
      { name: 'component', type: 'Component', required: true, description: 'Component instance' },
    ],
    properties: [
      { name: 'name', type: 'string', description: 'Component name' },
      { name: 'fields', type: 'object', default: {}, description: 'Component fields' },
    ],
  },
  system: {
    name: 'System',
    description: 'ECS System - processes entities with specific components',
    category: 'Trinity/ECS',
    inputs: [
      { name: 'query', type: 'Query', required: true, description: 'Entity query' },
    ],
    outputs: [
      { name: 'entities', type: 'Entity[]', required: true, description: 'Matched entities' },
    ],
    properties: [
      { name: 'name', type: 'string', description: 'System name' },
      { name: 'priority', type: 'number', default: 0, description: 'Execution priority' },
    ],
  },
  resource: {
    name: 'Resource',
    description: 'ECS Resource - global singleton data',
    category: 'Trinity/ECS',
    inputs: [],
    outputs: [
      { name: 'resource', type: 'Resource', required: true, description: 'Resource instance' },
    ],
    properties: [
      { name: 'name', type: 'string', description: 'Resource name' },
      { name: 'data', type: 'object', default: {}, description: 'Resource data' },
    ],
  },
  event: {
    name: 'Event',
    description: 'ECS Event - message passed between systems',
    category: 'Trinity/ECS',
    inputs: [
      { name: 'trigger', type: 'Trigger', required: true, description: 'Event trigger' },
    ],
    outputs: [
      { name: 'event', type: 'Event', required: true, description: 'Event instance' },
    ],
    properties: [
      { name: 'name', type: 'string', description: 'Event name' },
      { name: 'payload', type: 'object', default: {}, description: 'Event payload schema' },
    ],
  },
};

/**
 * Mock node definitions for testing.
 */
const MOCK_NODE_DEFINITIONS: NodeDefinitionMap = {
  'Trinity/Component': {
    type: 'Trinity/Component',
    displayName: 'Component',
    category: 'Trinity/ECS',
    description: 'Define an ECS Component',
    inputs: {
      entity: {
        name: 'entity',
        type: 'Entity',
        required: true,
        description: 'Target entity',
      },
    },
    outputs: {
      component: {
        name: 'component',
        type: 'Component',
        description: 'Component instance',
      },
    },
    widgets: {
      name: {
        name: 'name',
        type: 'text',
        label: 'Component Name',
        defaultValue: 'MyComponent',
      },
    },
    color: '#4CAF50',
  },
  'Trinity/System': {
    type: 'Trinity/System',
    displayName: 'System',
    category: 'Trinity/ECS',
    description: 'Define an ECS System',
    inputs: {
      query: {
        name: 'query',
        type: 'Query',
        required: true,
        description: 'Entity query',
      },
    },
    outputs: {
      entities: {
        name: 'entities',
        type: 'Entity[]',
        description: 'Matched entities',
      },
    },
    widgets: {
      name: {
        name: 'name',
        type: 'text',
        label: 'System Name',
        defaultValue: 'MySystem',
      },
      priority: {
        name: 'priority',
        type: 'integer',
        label: 'Priority',
        defaultValue: 0,
        min: -100,
        max: 100,
      },
    },
    color: '#2196F3',
  },
  'Trinity/Resource': {
    type: 'Trinity/Resource',
    displayName: 'Resource',
    category: 'Trinity/ECS',
    description: 'Define a global Resource',
    inputs: {},
    outputs: {
      resource: {
        name: 'resource',
        type: 'Resource',
        description: 'Resource instance',
      },
    },
    widgets: {
      name: {
        name: 'name',
        type: 'text',
        label: 'Resource Name',
        defaultValue: 'MyResource',
      },
    },
    color: '#FF9800',
  },
  'Trinity/Event': {
    type: 'Trinity/Event',
    displayName: 'Event',
    category: 'Trinity/ECS',
    description: 'Define an ECS Event',
    inputs: {
      trigger: {
        name: 'trigger',
        type: 'Trigger',
        required: true,
        description: 'Event trigger',
      },
    },
    outputs: {
      event: {
        name: 'event',
        type: 'Event',
        description: 'Event instance',
      },
    },
    widgets: {
      name: {
        name: 'name',
        type: 'text',
        label: 'Event Name',
        defaultValue: 'MyEvent',
      },
    },
    color: '#9C27B0',
  },
  'Math/Add': {
    type: 'Math/Add',
    displayName: 'Add',
    category: 'Math',
    description: 'Add two numbers',
    inputs: {
      a: { name: 'a', type: 'number', required: true },
      b: { name: 'b', type: 'number', required: true },
    },
    outputs: {
      result: { name: 'result', type: 'number' },
    },
    color: '#607D8B',
  },
  'Logic/Branch': {
    type: 'Logic/Branch',
    displayName: 'Branch',
    category: 'Logic',
    description: 'Conditional branching',
    inputs: {
      condition: { name: 'condition', type: 'boolean', required: true },
      true_value: { name: 'true_value', type: '*' },
      false_value: { name: 'false_value', type: '*' },
    },
    outputs: {
      result: { name: 'result', type: '*' },
    },
    color: '#795548',
  },
};

/**
 * Mock node graph for testing.
 */
const MOCK_GRAPH: NodeGraph = {
  nodes: [
    {
      id: 'node-1',
      type: 'component',
      name: 'Position',
      position: [100, 100],
      data: {
        fields: { x: 'float', y: 'float', z: 'float' },
      },
      source: { file: 'components.py', line: 10 },
    },
    {
      id: 'node-2',
      type: 'component',
      name: 'Velocity',
      position: [100, 250],
      data: {
        fields: { x: 'float', y: 'float', z: 'float' },
      },
      source: { file: 'components.py', line: 20 },
    },
    {
      id: 'node-3',
      type: 'system',
      name: 'MovementSystem',
      position: [350, 175],
      data: {
        query: ['Position', 'Velocity'],
        priority: 100,
      },
      source: { file: 'systems.py', line: 5 },
    },
    {
      id: 'node-4',
      type: 'resource',
      name: 'Time',
      position: [100, 400],
      data: {
        fields: { delta: 'float', elapsed: 'float' },
      },
      source: { file: 'resources.py', line: 3 },
    },
  ],
  edges: [
    {
      id: 'edge-1',
      source: 'node-1',
      target: 'node-3',
      type: 'query',
    },
    {
      id: 'edge-2',
      source: 'node-2',
      target: 'node-3',
      type: 'query',
    },
    {
      id: 'edge-3',
      source: 'node-4',
      target: 'node-3',
      type: 'reference',
    },
  ],
};

// =============================================================================
// MOCK API IMPLEMENTATION
// =============================================================================

/**
 * MockAPI - Development implementation with mock data.
 */
export class MockAPI implements FlowForgeAPI {
  private simulatedDelay = 100; // ms
  private executionCounter = 0;
  private executions: Map<string, ExecutionStatus> = new Map();

  /**
   * Create a mock API with optional delay simulation.
   */
  constructor(options?: { simulatedDelay?: number }) {
    if (options?.simulatedDelay !== undefined) {
      this.simulatedDelay = options.simulatedDelay;
    }
  }

  /**
   * Simulate async delay.
   */
  private async delay(): Promise<void> {
    if (this.simulatedDelay > 0) {
      await new Promise((resolve) => setTimeout(resolve, this.simulatedDelay));
    }
  }

  // ===========================================================================
  // Node Definitions
  // ===========================================================================

  async getObjectInfo(): Promise<NodeDefinitionMap> {
    await this.delay();
    return { ...MOCK_NODE_DEFINITIONS };
  }

  async getNodeDefinition(type: string): Promise<NodeDefinition | null> {
    await this.delay();
    return MOCK_NODE_DEFINITIONS[type] ?? null;
  }

  async searchNodes(query: string, limit = 10): Promise<NodeDefinitionMap> {
    await this.delay();
    const lowerQuery = query.toLowerCase();
    const results: Record<string, NodeDefinition> = {};
    let count = 0;

    for (const [type, def] of Object.entries(MOCK_NODE_DEFINITIONS)) {
      if (count >= limit) break;

      const matches =
        type.toLowerCase().includes(lowerQuery) ||
        def.displayName?.toLowerCase().includes(lowerQuery) ||
        def.description?.toLowerCase().includes(lowerQuery);

      if (matches) {
        results[type] = def;
        count++;
      }
    }

    return results as NodeDefinitionMap;
  }

  async getTrinityNodeTypes(): Promise<TrinityNodeTypes> {
    await this.delay();
    return { ...MOCK_TRINITY_NODE_TYPES };
  }

  // ===========================================================================
  // Python File Operations
  // ===========================================================================

  async parsePythonFile(path: string): Promise<NodeGraph> {
    await this.delay();
    console.log(`[MockAPI] Parsing Python file: ${path}`);

    // Return mock graph with updated source file
    return {
      nodes: MOCK_GRAPH.nodes.map((node) => ({
        ...node,
        source: { ...node.source, file: path },
      })),
      edges: [...MOCK_GRAPH.edges],
    };
  }

  async openPythonFile(): Promise<OpenPythonFileResult | null> {
    await this.delay();

    // Simulate file dialog with a mock path
    // In browser, we could use the File System Access API if available
    const mockPath = '/mock/path/to/game.py';

    console.log(`[MockAPI] Opening Python file dialog - returning: ${mockPath}`);

    const graph = await this.parsePythonFile(mockPath);

    return { path: mockPath, graph };
  }

  async savePythonFile(path: string, graph: NodeGraph): Promise<void> {
    await this.delay();
    console.log(`[MockAPI] Saving graph to Python file: ${path}`);
    console.log(`[MockAPI] Graph has ${graph.nodes.length} nodes and ${graph.edges.length} edges`);
  }

  async savePythonFileAs(graph: NodeGraph): Promise<string | null> {
    await this.delay();

    // Simulate save dialog with a mock path
    const mockPath = '/mock/path/to/saved_game.py';

    console.log(`[MockAPI] Save As dialog - returning: ${mockPath}`);
    await this.savePythonFile(mockPath, graph);

    return mockPath;
  }

  // ===========================================================================
  // Graph Operations (Future)
  // ===========================================================================

  async executeGraph(graph: NodeGraph): Promise<ExecutionResponse> {
    await this.delay();
    const executionId = `exec-${++this.executionCounter}`;

    console.log(`[MockAPI] Executing graph with ${graph.nodes.length} nodes`);

    // Create execution status
    const status: ExecutionStatus = {
      executionId,
      status: 'running',
      progress: 0,
    };
    this.executions.set(executionId, status);

    // Simulate execution progress
    this.simulateExecution(executionId);

    return { executionId };
  }

  private async simulateExecution(executionId: string): Promise<void> {
    const status = this.executions.get(executionId);
    if (!status) return;

    // Simulate progress over 2 seconds
    for (let i = 1; i <= 10; i++) {
      await new Promise((resolve) => setTimeout(resolve, 200));

      const currentStatus = this.executions.get(executionId);
      if (!currentStatus || currentStatus.status === 'cancelled') return;

      currentStatus.progress = i * 10;

      if (i === 10) {
        currentStatus.status = 'completed';
      }
    }
  }

  async getExecutionStatus(executionId: string): Promise<ExecutionStatus> {
    await this.delay();
    const status = this.executions.get(executionId);

    if (!status) {
      return {
        executionId,
        status: 'error',
        error: 'Execution not found',
      };
    }

    return { ...status };
  }

  async cancelExecution(executionId: string): Promise<boolean> {
    await this.delay();
    const status = this.executions.get(executionId);

    if (!status) {
      return false;
    }

    if (status.status === 'running' || status.status === 'pending') {
      status.status = 'cancelled';
      return true;
    }

    return false;
  }

  // ===========================================================================
  // System
  // ===========================================================================

  async getAppInfo(): Promise<AppInfo> {
    await this.delay();
    return {
      name: 'FlowForge (Mock)',
      version: '0.1.0-dev',
      tauriVersion: 'N/A (Mock)',
      platform: 'browser',
      arch: 'wasm',
    };
  }

  async ping(): Promise<{ pong: boolean; timestamp: number }> {
    await this.delay();
    return {
      pong: true,
      timestamp: Date.now(),
    };
  }
}

/**
 * Default Mock API instance.
 */
export const mockApi = new MockAPI();

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Create a mock graph node.
 */
export function createMockNode(
  id: string,
  type: GraphNode['type'],
  name: string,
  position: [number, number] = [0, 0]
): GraphNode {
  return {
    id,
    type,
    name,
    position,
    data: {},
    source: { file: 'mock.py', line: 1 },
  };
}

/**
 * Create a mock graph edge.
 */
export function createMockEdge(
  id: string,
  source: string,
  target: string,
  type: GraphEdge['type'] = 'reference'
): GraphEdge {
  return { id, source, target, type };
}

/**
 * Create an empty mock graph.
 */
export function createEmptyMockGraph(): NodeGraph {
  return { nodes: [], edges: [] };
}
