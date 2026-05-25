/**
 * IPC Command Definitions
 *
 * Defines all commands available for Tauri <-> Bun communication.
 * These are the RPC-style methods that can be invoked.
 */

import type { ExecutionId } from '../types/primitives.js';
import type { NodeDefinitionMap } from '../types/node.js';
import type { WorkflowSchema } from '../types/workflow.js';
import type { ExecutionConfig, ExecutionResult, ExecutionQueueStatus } from '../types/execution.js';
import type { DiscoveredPlugin, LoadedPlugin } from '../types/plugin.js';

// =============================================================================
// COMMAND REGISTRY TYPE
// =============================================================================

/**
 * Type-safe command definition.
 */
export interface CommandDefinition<TParams = void, TResult = void> {
  readonly params: TParams;
  readonly result: TResult;
}

// =============================================================================
// NODE COMMANDS
// =============================================================================

/**
 * Get all available node definitions.
 */
export interface GetNodeDefinitionsCommand extends CommandDefinition<void, NodeDefinitionMap> {}

/**
 * Get a specific node definition by type.
 */
export interface GetNodeDefinitionCommand extends CommandDefinition<
  { type: string },
  NodeDefinitionMap[string] | null
> {}

/**
 * Search node definitions by query.
 */
export interface SearchNodesCommand extends CommandDefinition<
  { query: string; limit?: number },
  NodeDefinitionMap
> {}

// =============================================================================
// WORKFLOW COMMANDS
// =============================================================================

/**
 * Execute a workflow.
 */
export interface ExecuteWorkflowCommand extends CommandDefinition<
  {
    workflow: WorkflowSchema;
    config?: ExecutionConfig;
  },
  { executionId: ExecutionId }
> {}

/**
 * Cancel a running execution.
 */
export interface CancelExecutionCommand extends CommandDefinition<
  { executionId: ExecutionId },
  { cancelled: boolean }
> {}

/**
 * Pause a running execution.
 */
export interface PauseExecutionCommand extends CommandDefinition<
  { executionId: ExecutionId },
  { paused: boolean }
> {}

/**
 * Resume a paused execution.
 */
export interface ResumeExecutionCommand extends CommandDefinition<
  { executionId: ExecutionId },
  { resumed: boolean }
> {}

/**
 * Get execution result.
 */
export interface GetExecutionResultCommand extends CommandDefinition<
  { executionId: ExecutionId },
  ExecutionResult | null
> {}

/**
 * Get execution queue status.
 */
export interface GetQueueStatusCommand extends CommandDefinition<void, ExecutionQueueStatus> {}

/**
 * Clear execution queue.
 */
export interface ClearQueueCommand extends CommandDefinition<
  { keepRunning?: boolean },
  { cleared: number }
> {}

// =============================================================================
// PLUGIN COMMANDS
// =============================================================================

/**
 * Discover available plugins.
 */
export interface DiscoverPluginsCommand extends CommandDefinition<void, readonly DiscoveredPlugin[]> {}

/**
 * Get loaded plugins.
 */
export interface GetLoadedPluginsCommand extends CommandDefinition<void, readonly LoadedPlugin[]> {}

/**
 * Load a plugin by name.
 */
export interface LoadPluginCommand extends CommandDefinition<
  { name: string },
  LoadedPlugin
> {}

/**
 * Unload a plugin.
 */
export interface UnloadPluginCommand extends CommandDefinition<
  { name: string },
  { unloaded: boolean }
> {}

/**
 * Enable a plugin.
 */
export interface EnablePluginCommand extends CommandDefinition<
  { name: string },
  { enabled: boolean }
> {}

/**
 * Disable a plugin.
 */
export interface DisablePluginCommand extends CommandDefinition<
  { name: string },
  { disabled: boolean }
> {}

// =============================================================================
// SYSTEM COMMANDS
// =============================================================================

/**
 * Get engine status.
 */
export interface GetEngineStatusCommand extends CommandDefinition<
  void,
  {
    version: string;
    uptime: number;
    memory: {
      used: number;
      total: number;
    };
    activeExecutions: number;
    loadedPlugins: number;
  }
> {}

/**
 * Ping the engine (health check).
 */
export interface PingCommand extends CommandDefinition<void, { pong: true; timestamp: number }> {}

/**
 * Shutdown the engine.
 */
export interface ShutdownCommand extends CommandDefinition<
  { graceful?: boolean; timeout?: number },
  { shutdownInitiated: boolean }
> {}

// =============================================================================
// COMMAND MAP
// =============================================================================

/**
 * Map of all available commands.
 * Used for type-safe command invocation.
 */
export interface CommandMap {
  // Node commands
  'nodes.getDefinitions': GetNodeDefinitionsCommand;
  'nodes.getDefinition': GetNodeDefinitionCommand;
  'nodes.search': SearchNodesCommand;

  // Workflow commands
  'workflow.execute': ExecuteWorkflowCommand;
  'execution.cancel': CancelExecutionCommand;
  'execution.pause': PauseExecutionCommand;
  'execution.resume': ResumeExecutionCommand;
  'execution.getResult': GetExecutionResultCommand;
  'queue.getStatus': GetQueueStatusCommand;
  'queue.clear': ClearQueueCommand;

  // Plugin commands
  'plugins.discover': DiscoverPluginsCommand;
  'plugins.getLoaded': GetLoadedPluginsCommand;
  'plugins.load': LoadPluginCommand;
  'plugins.unload': UnloadPluginCommand;
  'plugins.enable': EnablePluginCommand;
  'plugins.disable': DisablePluginCommand;

  // System commands
  'engine.status': GetEngineStatusCommand;
  'engine.ping': PingCommand;
  'engine.shutdown': ShutdownCommand;
}

/**
 * All command names.
 */
export type CommandName = keyof CommandMap;

/**
 * Extract params type for a command.
 */
export type CommandParams<T extends CommandName> = CommandMap[T]['params'];

/**
 * Extract result type for a command.
 */
export type CommandResult<T extends CommandName> = CommandMap[T]['result'];
