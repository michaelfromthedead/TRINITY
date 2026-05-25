/**
 * IPC Event Definitions
 *
 * Defines all events that can be emitted during IPC communication.
 * Events are used for real-time updates during execution.
 */
import type { ExecutionId } from '../types/primitives.js';
import type { ExecutionEvent, ExecutionStartEvent, ExecutionProgressEvent, ExecutionCompleteEvent, ExecutionErrorEvent, ExecutionCancelledEvent, NodeStartEvent, NodeProgressEvent, NodeCompleteEvent, NodeErrorEvent, NodeSkippedEvent } from '../types/execution.js';
/**
 * All event names.
 */
export declare const EventNames: {
    readonly EXECUTION_START: "execution:start";
    readonly EXECUTION_PROGRESS: "execution:progress";
    readonly EXECUTION_COMPLETE: "execution:complete";
    readonly EXECUTION_ERROR: "execution:error";
    readonly EXECUTION_CANCELLED: "execution:cancelled";
    readonly NODE_START: "node:start";
    readonly NODE_PROGRESS: "node:progress";
    readonly NODE_COMPLETE: "node:complete";
    readonly NODE_ERROR: "node:error";
    readonly NODE_SKIPPED: "node:skipped";
    readonly ENGINE_READY: "engine:ready";
    readonly ENGINE_SHUTDOWN: "engine:shutdown";
    readonly PLUGIN_LOADED: "plugin:loaded";
    readonly PLUGIN_UNLOADED: "plugin:unloaded";
    readonly PLUGIN_ERROR: "plugin:error";
    readonly QUEUE_UPDATED: "queue:updated";
    readonly LOG: "log";
};
export type EventName = (typeof EventNames)[keyof typeof EventNames];
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
/**
 * Queue updated event.
 */
export interface QueueUpdatedEvent {
    readonly type: 'queue:updated';
    readonly pendingCount: number;
    readonly runningId?: ExecutionId;
    readonly timestamp: number;
}
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
/**
 * Map of event names to event payloads.
 */
export interface EventMap {
    'execution:start': ExecutionStartEvent;
    'execution:progress': ExecutionProgressEvent;
    'execution:complete': ExecutionCompleteEvent;
    'execution:error': ExecutionErrorEvent;
    'execution:cancelled': ExecutionCancelledEvent;
    'node:start': NodeStartEvent;
    'node:progress': NodeProgressEvent;
    'node:complete': NodeCompleteEvent;
    'node:error': NodeErrorEvent;
    'node:skipped': NodeSkippedEvent;
    'engine:ready': EngineReadyEvent;
    'engine:shutdown': EngineShutdownEvent;
    'plugin:loaded': PluginLoadedEvent;
    'plugin:unloaded': PluginUnloadedEvent;
    'plugin:error': PluginErrorEvent;
    'queue:updated': QueueUpdatedEvent;
    'log': LogEvent;
}
/**
 * Extract payload type for an event.
 */
export type EventPayload<T extends EventName> = T extends keyof EventMap ? EventMap[T] : never;
/**
 * Type guard for execution events.
 */
export declare function isExecutionEvent(event: {
    type: string;
}): event is ExecutionEvent;
/**
 * Type guard for specific event type.
 */
export declare function isEventType<T extends EventName>(event: {
    type: string;
}, eventType: T): event is EventMap[T];
//# sourceMappingURL=events.d.ts.map