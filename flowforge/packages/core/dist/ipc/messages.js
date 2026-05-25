/**
 * IPC Message Types
 *
 * Core message structures for Tauri <-> Bun communication.
 * Uses a JSON-based protocol over stdio.
 */
// =============================================================================
// MESSAGE IDS
// =============================================================================
/**
 * Generate a unique message ID.
 * Uses a combination of timestamp and random string.
 */
export function generateMessageId() {
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substring(2, 10);
    return `${timestamp}-${random}`;
}
/**
 * Create a request message.
 */
export function createRequest(method, params, timeout) {
    const message = {
        id: generateMessageId(),
        type: 'request',
        timestamp: Date.now(),
        method,
    };
    if (params !== undefined) {
        message.params = params;
    }
    if (timeout !== undefined) {
        message.timeout = timeout;
    }
    return message;
}
/**
 * Check if response is an error.
 */
export function isErrorResponse(response) {
    return 'error' in response;
}
/**
 * Create a success response.
 */
export function createSuccessResponse(requestId, result) {
    return {
        id: generateMessageId(),
        type: 'response',
        timestamp: Date.now(),
        requestId,
        result,
    };
}
/**
 * Create an error response.
 */
export function createErrorResponse(requestId, error) {
    return {
        id: generateMessageId(),
        type: 'response',
        timestamp: Date.now(),
        requestId,
        error,
    };
}
/**
 * Create an event message.
 */
export function createEvent(event, payload, correlationId) {
    const message = {
        id: generateMessageId(),
        type: 'event',
        timestamp: Date.now(),
        event,
    };
    if (payload !== undefined) {
        message.payload = payload;
    }
    if (correlationId !== undefined) {
        message.correlationId = correlationId;
    }
    return message;
}
/**
 * Parse a JSON string into an IPC message.
 */
export function parseIPCMessage(json) {
    const message = JSON.parse(json);
    // Validate required fields
    if (typeof message.id !== 'string') {
        throw new Error('Invalid IPC message: missing id');
    }
    if (typeof message.type !== 'string') {
        throw new Error('Invalid IPC message: missing type');
    }
    if (typeof message.timestamp !== 'number') {
        throw new Error('Invalid IPC message: missing timestamp');
    }
    return message;
}
/**
 * Serialize an IPC message to JSON string with newline.
 */
export function serializeIPCMessage(message) {
    return JSON.stringify(message) + '\n';
}
// =============================================================================
// ERROR CODES
// =============================================================================
/**
 * Standard IPC error codes.
 */
export const IPCErrorCodes = {
    // General errors
    UNKNOWN_ERROR: 'UNKNOWN_ERROR',
    INTERNAL_ERROR: 'INTERNAL_ERROR',
    TIMEOUT: 'TIMEOUT',
    // Request errors
    METHOD_NOT_FOUND: 'METHOD_NOT_FOUND',
    INVALID_PARAMS: 'INVALID_PARAMS',
    INVALID_REQUEST: 'INVALID_REQUEST',
    // Execution errors
    EXECUTION_FAILED: 'EXECUTION_FAILED',
    NODE_NOT_FOUND: 'NODE_NOT_FOUND',
    TYPE_MISMATCH: 'TYPE_MISMATCH',
    VALIDATION_FAILED: 'VALIDATION_FAILED',
    CANCELLED: 'CANCELLED',
    // Plugin errors
    PLUGIN_NOT_FOUND: 'PLUGIN_NOT_FOUND',
    PLUGIN_LOAD_FAILED: 'PLUGIN_LOAD_FAILED',
    PLUGIN_ERROR: 'PLUGIN_ERROR',
    // Permission errors
    PERMISSION_DENIED: 'PERMISSION_DENIED',
    CAPABILITY_REQUIRED: 'CAPABILITY_REQUIRED',
};
/**
 * Create a standard IPC error.
 */
export function createIPCError(code, message, data) {
    return {
        code,
        message,
        data,
    };
}
//# sourceMappingURL=messages.js.map