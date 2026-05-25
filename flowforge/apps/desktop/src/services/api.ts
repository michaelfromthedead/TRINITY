/**
 * FlowForge API Interface
 *
 * Abstract API interface that replaces ComfyUI API calls.
 * Implementations: TauriAPI (desktop) and MockAPI (browser development).
 */

import type { NodeDefinitionMap, NodeDefinition } from '@flowforge/core';

// =============================================================================
// GRAPH TYPES
// =============================================================================

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

// =============================================================================
// NODE TYPE DEFINITIONS
// =============================================================================

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
 * Trinity-specific node type definitions for ECS patterns.
 */
export interface TrinityNodeTypes {
  component: NodeTypeDefinition;
  system: NodeTypeDefinition;
  resource: NodeTypeDefinition;
  event: NodeTypeDefinition;
}

// =============================================================================
// FILE OPERATIONS RESULT
// =============================================================================

/**
 * Result of opening a Python file.
 */
export interface OpenPythonFileResult {
  path: string;
  graph: NodeGraph;
}

// =============================================================================
// EXECUTION TYPES
// =============================================================================

/**
 * Execution response from the backend.
 */
export interface ExecutionResponse {
  executionId: string;
}

/**
 * Execution status.
 */
export interface ExecutionStatus {
  executionId: string;
  status: 'pending' | 'running' | 'completed' | 'error' | 'cancelled';
  progress?: number;
  error?: string;
}

// =============================================================================
// APPLICATION INFO
// =============================================================================

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

// =============================================================================
// FLOWFORGE API INTERFACE
// =============================================================================

/**
 * FlowForge API interface.
 *
 * This is the main abstraction layer that replaces ComfyUI's fetch-based API.
 * Implementations can use Tauri IPC (desktop) or mock data (browser dev).
 */
export interface FlowForgeAPI {
  // ===========================================================================
  // Node Definitions
  // ===========================================================================

  /**
   * Get all node definitions (equivalent to ComfyUI's /object_info endpoint).
   * Returns a map of node type names to their definitions.
   */
  getObjectInfo(): Promise<NodeDefinitionMap>;

  /**
   * Get a specific node definition by type.
   */
  getNodeDefinition?(type: string): Promise<NodeDefinition | null>;

  /**
   * Search node definitions by query.
   */
  searchNodes?(query: string, limit?: number): Promise<NodeDefinitionMap>;

  /**
   * Get Trinity-specific node type definitions.
   */
  getTrinityNodeTypes?(): Promise<TrinityNodeTypes>;

  // ===========================================================================
  // Python File Operations
  // ===========================================================================

  /**
   * Parse a Python file and return its node graph representation.
   * Analyzes the file for Trinity ECS patterns (components, systems, resources, events).
   */
  parsePythonFile(path: string): Promise<NodeGraph>;

  /**
   * Open a Python file via native dialog.
   * Returns the file path and parsed graph, or null if cancelled.
   */
  openPythonFile(): Promise<OpenPythonFileResult | null>;

  /**
   * Save a node graph as Python code to a file.
   */
  savePythonFile(path: string, graph: NodeGraph): Promise<void>;

  /**
   * Save a Python file with a native save dialog.
   * Returns the saved file path, or null if cancelled.
   */
  savePythonFileAs?(graph: NodeGraph): Promise<string | null>;

  // ===========================================================================
  // Graph Operations (Future)
  // ===========================================================================

  /**
   * Execute a node graph.
   * This is for future use when we add execution capabilities.
   */
  executeGraph?(graph: NodeGraph): Promise<ExecutionResponse>;

  /**
   * Get execution status.
   */
  getExecutionStatus?(executionId: string): Promise<ExecutionStatus>;

  /**
   * Cancel a running execution.
   */
  cancelExecution?(executionId: string): Promise<boolean>;

  // ===========================================================================
  // System
  // ===========================================================================

  /**
   * Get application info.
   */
  getAppInfo?(): Promise<AppInfo>;

  /**
   * Health check ping.
   */
  ping?(): Promise<{ pong: boolean; timestamp: number }>;
}
