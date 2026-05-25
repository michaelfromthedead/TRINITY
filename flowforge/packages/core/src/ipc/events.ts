/**
 * IPC Event Definitions
 *
 * Defines all events that can be emitted during IPC communication.
 * Events are used for real-time updates during execution.
 */

import type { ExecutionId } from '../types/primitives.js';
import type {
  ExecutionEvent,
  ExecutionStartEvent,
  ExecutionProgressEvent,
  ExecutionCompleteEvent,
  ExecutionErrorEvent,
  ExecutionCancelledEvent,
  NodeStartEvent,
  NodeProgressEvent,
  NodeCompleteEvent,
  NodeErrorEvent,
  NodeSkippedEvent,
} from '../types/execution.js';

// =============================================================================
// EVENT NAMES
// =============================================================================

/**
 * All event names.
 */
export const EventNames = {
  // Execution lifecycle
  EXECUTION_START: 'execution:start',
  EXECUTION_PROGRESS: 'execution:progress',
  EXECUTION_COMPLETE: 'execution:complete',
  EXECUTION_ERROR: 'execution:error',
  EXECUTION_CANCELLED: 'execution:cancelled',

  // Node lifecycle
  NODE_START: 'node:start',
  NODE_PROGRESS: 'node:progress',
  NODE_COMPLETE: 'node:complete',
  NODE_ERROR: 'node:error',
  NODE_SKIPPED: 'node:skipped',

  // Engine status
  ENGINE_READY: 'engine:ready',
  ENGINE_SHUTDOWN: 'engine:shutdown',

  // Plugin events
  PLUGIN_LOADED: 'plugin:loaded',
  PLUGIN_UNLOADED: 'plugin:unloaded',
  PLUGIN_ERROR: 'plugin:error',

  // Queue events
  QUEUE_UPDATED: 'queue:updated',

  // Log events
  LOG: 'log',
} as const;

export type EventName = (typeof EventNames)[keyof typeof EventNames];

// =============================================================================
// ENGINE EVENTS
// =============================================================================

/**
 * Engine ready event.
 */
export interface EngineReadyEvent {
  readonly type: 'engine:ready';
  readonly version: string;
  readonly timestamp: number;
}

/**
 * Engine shutdown event.
 */
export interface EngineShutdownEvent {
  readonly type: 'engine:shutdown';
  readonly reason?: string;
  readonly timestamp: number;
}

// =============================================================================
// PLUGIN EVENTS
// =============================================================================

/**
 * Plugin loaded event.
 */
export interface PluginLoadedEvent {
  readonly type: 'plugin:loaded';
  readonly pluginName: string;
  readonly version: string;
  readonly nodeCount: number;
  readonly timestamp: number;
}

/**
 * Plugin unloaded event.
 */
export interface PluginUnloadedEvent {
  readonly type: 'plugin:unloaded';
  readonly pluginName: string;
  readonly timestamp: number;
}

/**
 * Plugin error event.
 */
export interface PluginErrorEvent {
  readonly type: 'plugin:error';
  readonly pluginName: string;
  readonly error: {
    readonly code: string;
    readonly message: string;
  };
  readonly timestamp: number;
}

// =============================================================================
// QUEUE EVENTS
// =============================================================================

/**
 * Queue updated event.
 */
export interface QueueUpdatedEvent {
  readonly type: 'queue:updated';
  readonly pendingCount: number;
  readonly runningId?: ExecutionId;
  readonly timestamp: number;
}

// =============================================================================
// LOG EVENTS
// =============================================================================

/**
 * Log event for debugging.
 */
export interface LogEvent {
  readonly type: 'log';
  readonly level: 'debug' | 'info' | 'warn' | 'error';
  readonly message: string;
  readonly data?: unknown;
  readonly source?: string;
  readonly timestamp: number;
}

// =============================================================================
// EVENT MAP
// =============================================================================

/**
 * Map of event names to event payloads.
 */
export interface EventMap {
  // Execution events
  'execution:start': ExecutionStartEvent;
  'execution:progress': ExecutionProgressEvent;
  'execution:complete': ExecutionCompleteEvent;
  'execution:error': ExecutionErrorEvent;
  'execution:cancelled': ExecutionCancelledEvent;

  // Node events
  'node:start': NodeStartEvent;
  'node:progress': NodeProgressEvent;
  'node:complete': NodeCompleteEvent;
  'node:error': NodeErrorEvent;
  'node:skipped': NodeSkippedEvent;

  // Engine events
  'engine:ready': EngineReadyEvent;
  'engine:shutdown': EngineShutdownEvent;

  // Plugin events
  'plugin:loaded': PluginLoadedEvent;
  'plugin:unloaded': PluginUnloadedEvent;
  'plugin:error': PluginErrorEvent;

  // Queue events
  'queue:updated': QueueUpdatedEvent;

  // Log events
  'log': LogEvent;
}

/**
 * Extract payload type for an event.
 */
export type EventPayload<T extends EventName> = T extends keyof EventMap ? EventMap[T] : never;

// =============================================================================
// EVENT HELPERS
// =============================================================================

/**
 * Type guard for execution events.
 */
export function isExecutionEvent(
  event: { type: string }
): event is ExecutionEvent {
  return event.type.startsWith('execution:') || event.type.startsWith('node:');
}

/**
 * Type guard for specific event type.
 */
export function isEventType<T extends EventName>(
  event: { type: string },
  eventType: T
): event is EventMap[T] {
  return event.type === eventType;
}
