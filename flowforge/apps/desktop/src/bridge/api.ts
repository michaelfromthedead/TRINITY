/**
 * Tauri API Bridge
 *
 * Replaces ComfyUI's fetch-based API with Tauri IPC commands.
 */

import { invoke } from '@tauri-apps/api/core';
import type {
  NodeDefinitionMap,
  WorkflowSchema,
  ExecutionConfig,
  ExecutionQueueStatus,
} from '@flowforge/core';
import { UI_CONFIG } from '@/config/flowforge.config';

/**
 * Execution response from the backend.
 */
export interface ExecutionResponse {
  executionId: string;
}

// ==========================================================================
// Trinity/Python Graph Types
// ==========================================================================

/**
 * A node in the visual graph representation of Python code.
 */
export interface GraphNode {
  id: string;
  type: 'component' | 'system' | 'resource' | 'event';
  name: string;
  position: [number, number];
  data: Record<string, unknown>;
  source: { file: string; line: number };
}

/**
 * An edge connecting nodes in the graph.
 */
export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: 'reference' | 'inheritance' | 'query';
}

/**
 * Complete node graph representing parsed Python code.
 */
export interface NodeGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

/**
 * Definition of a node type's properties and behavior.
 */
export interface NodeTypeDefinition {
  name: string;
  description: string;
  category: string;
  inputs: NodePortDefinition[];
  outputs: NodePortDefinition[];
  properties: NodePropertyDefinition[];
}

/**
 * Definition of a node input/output port.
 */
export interface NodePortDefinition {
  name: string;
  type: string;
  required: boolean;
  description?: string;
}

/**
 * Definition of a node property.
 */
export interface NodePropertyDefinition {
  name: string;
  type: string;
  default?: unknown;
  description?: string;
}

/**
 * Trinity-specific node type definitions for ECS patterns.
 */
export interface TrinityNodeTypes {
  component: NodeTypeDefinition;
  system: NodeTypeDefinition;
  resource: NodeTypeDefinition;
  event: NodeTypeDefinition;
}

/**
 * Application info from the backend.
 */
export interface AppInfo {
  name: string;
  version: string;
  tauriVersion: string;
  platform: string;
  arch: string;
}

/**
 * TauriAPI class - main interface for backend communication.
 */
export class TauriAPI {
  // ==========================================================================
  // Node Definitions
  // ==========================================================================

  /**
   * Get all node definitions (equivalent to /object_info).
   */
  async getObjectInfo(): Promise<NodeDefinitionMap> {
    return await invoke<NodeDefinitionMap>('get_object_info');
  }

  /**
   * Get a specific node definition by type.
   */
  async getNodeDefinition(type: string): Promise<NodeDefinitionMap[string] | null> {
    return await invoke<NodeDefinitionMap[string] | null>('get_node_definition', {
      nodeType: type,
    });
  }

  /**
   * Search node definitions by query.
   */
  async searchNodes(query: string, limit?: number): Promise<NodeDefinitionMap> {
    return await invoke<NodeDefinitionMap>('search_nodes', {
      request: { query, limit },
    });
  }

  // ==========================================================================
  // Workflow Execution
  // ==========================================================================

  /**
   * Execute a workflow.
   */
  async executeWorkflow(
    workflow: WorkflowSchema,
    config?: Partial<ExecutionConfig>
  ): Promise<ExecutionResponse> {
    return await invoke<ExecutionResponse>('execute_workflow', {
      request: { workflow, config },
    });
  }

  /**
   * Get execution queue status.
   */
  async getQueueStatus(): Promise<ExecutionQueueStatus> {
    return await invoke<ExecutionQueueStatus>('get_queue_status');
  }

  /**
   * Cancel a running execution.
   */
  async cancelExecution(executionId: string): Promise<boolean> {
    return await invoke<boolean>('cancel_execution', { executionId });
  }

  // ==========================================================================
  // System
  // ==========================================================================

  /**
   * Get application info.
   */
  async getAppInfo(): Promise<AppInfo> {
    return await invoke<AppInfo>('get_app_info');
  }

  /**
   * Ping the backend (health check).
   */
  async ping(): Promise<{ pong: boolean; timestamp: number }> {
    return await invoke<{ pong: boolean; timestamp: number }>('ping');
  }

  // ==========================================================================
  // Python/Trinity Integration
  // ==========================================================================

  /**
   * Parse a Python file and return its node graph representation.
   * Analyzes the file for Trinity ECS patterns (components, systems, resources, events).
   */
  async parsePythonFile(path: string): Promise<NodeGraph> {
    return await invoke<NodeGraph>('parse_python_file', { path });
  }

  /**
   * Open a file dialog filtered for Python files.
   * Returns the selected file path or null if cancelled.
   */
  async openPythonFile(): Promise<string | null> {
    return await invoke<string | null>('open_file_dialog', {
      request: {
        filters: [
          { name: 'Python Files', extensions: ['py'] },
          { name: 'All Files', extensions: ['*'] },
        ],
        title: 'Open Python File',
      },
    });
  }

  /**
   * Save generated Python code to a file.
   */
  async savePythonFile(path: string, content: string): Promise<void> {
    await invoke<void>('write_python_file', { path, content });
  }

  /**
   * Get Trinity-specific node type definitions.
   * Returns definitions for component, system, resource, and event nodes.
   */
  async getTrinityNodeTypes(): Promise<TrinityNodeTypes> {
    return await invoke<TrinityNodeTypes>('get_trinity_node_types');
  }
}

// ==========================================================================
// Diff Generation Types
// ==========================================================================

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
  lineNumber: number | null;
  content: string;
  type: DiffLineType;
}

/**
 * Side-by-side diff result.
 */
export interface SideBySideDiff {
  filename: string;
  left: SideBySideLine[];
  right: SideBySideLine[];
  leftTitle: string;
  rightTitle: string;
}

/**
 * Options for diff generation.
 */
export interface GenerateDiffOptions {
  filename?: string;
  originalPath?: string;
  contextLines?: number;
  sideBySide?: boolean;
}

/**
 * Generate a unified diff between original and modified source.
 */
export async function generateDiff(
  original: string,
  modified: string,
  options: GenerateDiffOptions = {}
): Promise<DiffResult> {
  return await invoke<DiffResult>('ipc_call', {
    request: {
      id: `diff-${Date.now()}`,
      method: 'generate_diff',
      params: {
        original,
        modified,
        filename: options.filename ?? '',
        original_path: options.originalPath,
        context_lines: options.contextLines ?? UI_CONFIG.diff.contextLines,
        side_by_side: false,
      },
    },
  });
}

/**
 * Generate a side-by-side diff between original and modified source.
 */
export async function generateSideBySideDiff(
  original: string,
  modified: string,
  filename: string = ''
): Promise<SideBySideDiff> {
  return await invoke<SideBySideDiff>('ipc_call', {
    request: {
      id: `diff-sbs-${Date.now()}`,
      method: 'generate_diff',
      params: {
        original,
        modified,
        filename,
        side_by_side: true,
      },
    },
  });
}

/**
 * Default API instance.
 */
export const api = new TauriAPI();
