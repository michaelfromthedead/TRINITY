/**
 * IPC Message Types
 *
 * Core message structures for Tauri <-> Bun communication.
 * Uses a JSON-based protocol over stdio.
 */
/**
 * Generate a unique message ID.
 * Uses a combination of timestamp and random string.
 */
export declare function generateMessageId(): string;
/**
 * Message type discriminator.
 */
export type IPCMessageType = 'request' | 'response' | 'event' | 'error';
/**
 * Base message interface.
 */
interface BaseIPCMessage {
    /** Unique message ID for correlation */
    readonly id: string;
    /** Message type */
    readonly type: IPCMessageType;
    /** Timestamp when message was created */
    readonly timestamp: number;
}
/**
 * Request message from Tauri to Bun.
 */
export interface IPCRequest<TParams = unknown> extends BaseIPCMessage {
    readonly type: 'request';
    /** Method to invoke */
    readonly method: string;
    /** Method parameters */
    readonly params?: TParams;
    /** Request timeout in milliseconds */
    readonly timeout?: number;
}
/**
 * Create a request message.
 */
export declare function createRequest<TParams>(method: string, params?: TParams, timeout?: number): IPCRequest<TParams>;
/**
 * Successful response message from Bun to Tauri.
 */
export interface IPCSuccessResponse<TResult = unknown> extends BaseIPCMessage {
    readonly type: 'response';
    /** Original request ID */
    readonly requestId: string;
    /** Response result */
    readonly result: TResult;
}
/**
 * Error response message from Bun to Tauri.
 */
export interface IPCErrorResponse extends BaseIPCMessage {
    readonly type: 'response';
    /** Original request ID */
    readonly requestId: string;
    /** Error details */
    readonly error: IPCError;
}
/**
 * IPC error structure.
 */
export interface IPCError {
    /** Error code */
    readonly code: string;
    /** Human-readable message */
    readonly message: string;
    /** Additional error data */
    readonly data?: unknown;
    /** Stack trace (development only) */
    readonly stack?: string;
}
/**
 * Union of response types.
 */
export type IPCResponse<TResult = unknown> = IPCSuccessResponse<TResult> | IPCErrorResponse;
/**
 * Check if response is an error.
 */
export declare function isErrorResponse(response: IPCResponse): response is IPCErrorResponse;
/**
 * Create a success response.
 */
export declare function createSuccessResponse<TResult>(requestId: string, result: TResult): IPCSuccessResponse<TResult>;
/**
 * Create an error response.
 */
export declare function createErrorResponse(requestId: string, error: IPCError): IPCErrorResponse;
/**
 * Event message from Bun to Tauri (or vice versa).
 * Events are fire-and-forget, no response expected.
 */
export interface IPCEvent<TPayload = unknown> extends BaseIPCMessage {
    readonly type: 'event';
    /** Event name */
    readonly event: string;
    /** Event payload */
    readonly payload?: TPayload;
    /** Optional correlation ID (e.g., execution ID) */
    readonly correlationId?: string;
}
/**
 * Create an event message.
 */
export declare function createEvent<TPayload>(event: string, payload?: TPayload, correlationId?: string): IPCEvent<TPayload>;
/**
 * Union of all IPC message types.
 */
export type IPCMessage = IPCRequest | IPCResponse | IPCEvent;
/**
 * Parse a JSON string into an IPC message.
 */
export declare function parseIPCMessage(json: string): IPCMessage;
/**
 * Serialize an IPC message to JSON string with newline.
 */
export declare function serializeIPCMessage(message: IPCMessage): string;
/**
 * Standard IPC error codes.
 */
export declare const IPCErrorCodes: {
    readonly UNKNOWN_ERROR: "UNKNOWN_ERROR";
    readonly INTERNAL_ERROR: "INTERNAL_ERROR";
    readonly TIMEOUT: "TIMEOUT";
    readonly METHOD_NOT_FOUND: "METHOD_NOT_FOUND";
    readonly INVALID_PARAMS: "INVALID_PARAMS";
    readonly INVALID_REQUEST: "INVALID_REQUEST";
    readonly EXECUTION_FAILED: "EXECUTION_FAILED";
    readonly NODE_NOT_FOUND: "NODE_NOT_FOUND";
    readonly TYPE_MISMATCH: "TYPE_MISMATCH";
    readonly VALIDATION_FAILED: "VALIDATION_FAILED";
    readonly CANCELLED: "CANCELLED";
    readonly PLUGIN_NOT_FOUND: "PLUGIN_NOT_FOUND";
    readonly PLUGIN_LOAD_FAILED: "PLUGIN_LOAD_FAILED";
    readonly PLUGIN_ERROR: "PLUGIN_ERROR";
    readonly PERMISSION_DENIED: "PERMISSION_DENIED";
    readonly CAPABILITY_REQUIRED: "CAPABILITY_REQUIRED";
};
export type IPCErrorCode = (typeof IPCErrorCodes)[keyof typeof IPCErrorCodes];
/**
 * Create a standard IPC error.
 */
export declare function createIPCError(code: IPCErrorCode, message: string, data?: unknown): IPCError;
export {};
//# sourceMappingURL=messages.d.ts.map