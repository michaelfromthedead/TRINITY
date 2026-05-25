/**
 * Tauri API Implementation
 *
 * Implements FlowForgeAPI using Tauri IPC commands.
 * This is the production implementation for the desktop app.
 */

import { invoke } from '@tauri-apps/api/core';
import type { NodeDefinitionMap, NodeDefinition } from '@flowforge/core';
import type {
  FlowForgeAPI,
  NodeGraph,
  OpenPythonFileResult,
  TrinityNodeTypes,
  ExecutionResponse,
  ExecutionStatus,
  AppInfo,
} from './api';

/**
 * File filter for dialogs.
 */
interface FileFilter {
  name: string;
  extensions: string[];
}

/**
 * Python file filters.
 */
const PYTHON_FILTERS: FileFilter[] = [
  { name: 'Python Files', extensions: ['py'] },
  { name: 'All Files', extensions: ['*'] },
];

/**
 * TauriAPI - Production implementation using Tauri IPC.
 */
export class TauriAPI implements FlowForgeAPI {
  // ===========================================================================
  // Node Definitions
  // ===========================================================================

  async getObjectInfo(): Promise<NodeDefinitionMap> {
    return await invoke<NodeDefinitionMap>('get_object_info');
  }

  async getNodeDefinition(type: string): Promise<NodeDefinition | null> {
    return await invoke<NodeDefinition | null>('get_node_definition', {
      nodeType: type,
    });
  }

  async searchNodes(query: string, limit?: number): Promise<NodeDefinitionMap> {
    return await invoke<NodeDefinitionMap>('search_nodes', {
      request: { query, limit },
    });
  }

  async getTrinityNodeTypes(): Promise<TrinityNodeTypes> {
    return await invoke<TrinityNodeTypes>('get_trinity_node_types');
  }

  // ===========================================================================
  // Python File Operations
  // ===========================================================================

  async parsePythonFile(path: string): Promise<NodeGraph> {
    return await invoke<NodeGraph>('parse_python_file', { path });
  }

  async openPythonFile(): Promise<OpenPythonFileResult | null> {
    // First, open file dialog
    const path = await invoke<string | null>('open_file_dialog', {
      request: {
        filters: PYTHON_FILTERS,
        title: 'Open Python File',
      },
    });

    if (path === null) {
      return null;
    }

    // Then, parse the file
    const graph = await this.parsePythonFile(path);

    return { path, graph };
  }

  async savePythonFile(path: string, graph: NodeGraph): Promise<void> {
    // Convert graph to Python code (backend handles this)
    await invoke<void>('save_python_graph', { path, graph });
  }

  async savePythonFileAs(graph: NodeGraph): Promise<string | null> {
    // Open save dialog
    const path = await invoke<string | null>('save_file_dialog', {
      request: {
        filters: PYTHON_FILTERS,
        title: 'Save Python File',
      },
    });

    if (path === null) {
      return null;
    }

    // Save the graph
    await this.savePythonFile(path, graph);

    return path;
  }

  // ===========================================================================
  // Graph Operations (Future)
  // ===========================================================================

  async executeGraph(graph: NodeGraph): Promise<ExecutionResponse> {
    return await invoke<ExecutionResponse>('execute_graph', { graph });
  }

  async getExecutionStatus(executionId: string): Promise<ExecutionStatus> {
    return await invoke<ExecutionStatus>('get_execution_status', { executionId });
  }

  async cancelExecution(executionId: string): Promise<boolean> {
    return await invoke<boolean>('cancel_execution', { executionId });
  }

  // ===========================================================================
  // System
  // ===========================================================================

  async getAppInfo(): Promise<AppInfo> {
    return await invoke<AppInfo>('get_app_info');
  }

  async ping(): Promise<{ pong: boolean; timestamp: number }> {
    return await invoke<{ pong: boolean; timestamp: number }>('ping');
  }
}

/**
 * Default Tauri API instance.
 */
export const tauriApi = new TauriAPI();
