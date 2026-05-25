/**
 * IPC Event Definitions
 *
 * Defines all events that can be emitted during IPC communication.
 * Events are used for real-time updates during execution.
 */
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
};
// =============================================================================
// EVENT HELPERS
// =============================================================================
/**
 * Type guard for execution events.
 */
export function isExecutionEvent(event) {
    return event.type.startsWith('execution:') || event.type.startsWith('node:');
}
/**
 * Type guard for specific event type.
 */
export function isEventType(event, eventType) {
    return event.type === eventType;
}
//# sourceMappingURL=events.js.map