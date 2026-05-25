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
export function generateMessageId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 10);
  return `${timestamp}-${random}`;
}

// =============================================================================
// BASE MESSAGE TYPES
// =============================================================================

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

// =============================================================================
// REQUEST MESSAGE
// =============================================================================

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
export function createRequest<TParams>(
  method: string,
  params?: TParams,
  timeout?: number
): IPCRequest<TParams> {
  const message: IPCRequest<TParams> = {
    id: generateMessageId(),
    type: 'request',
    timestamp: Date.now(),
    method,
  };
  if (params !== undefined) {
    (message as { params: TParams }).params = params;
  }
  if (timeout !== undefined) {
    (message as { timeout: number }).timeout = timeout;
  }
  return message;
}

// =============================================================================
// RESPONSE MESSAGE
// =============================================================================

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
export function isErrorResponse(response: IPCResponse): response is IPCErrorResponse {
  return 'error' in response;
}

/**
 * Create a success response.
 */
export function createSuccessResponse<TResult>(
  requestId: string,
  result: TResult
): IPCSuccessResponse<TResult> {
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
export function createErrorResponse(
  requestId: string,
  error: IPCError
): IPCErrorResponse {
  return {
    id: generateMessageId(),
    type: 'response',
    timestamp: Date.now(),
    requestId,
    error,
  };
}

// =============================================================================
// EVENT MESSAGE
// =============================================================================

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
export function createEvent<TPayload>(
  event: string,
  payload?: TPayload,
  correlationId?: string
): IPCEvent<TPayload> {
  const message: IPCEvent<TPayload> = {
    id: generateMessageId(),
    type: 'event',
    timestamp: Date.now(),
    event,
  };
  if (payload !== undefined) {
    (message as { payload: TPayload }).payload = payload;
  }
  if (correlationId !== undefined) {
    (message as { correlationId: string }).correlationId = correlationId;
  }
  return message;
}

// =============================================================================
// UNION TYPE
// =============================================================================

/**
 * Union of all IPC message types.
 */
export type IPCMessage =
  | IPCRequest
  | IPCResponse
  | IPCEvent;

/**
 * Parse a JSON string into an IPC message.
 */
export function parseIPCMessage(json: string): IPCMessage {
  const message = JSON.parse(json) as IPCMessage;

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
export function serializeIPCMessage(message: IPCMessage): string {
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
} as const;

export type IPCErrorCode = (typeof IPCErrorCodes)[keyof typeof IPCErrorCodes];

/**
 * Create a standard IPC error.
 */
export function createIPCError(
  code: IPCErrorCode,
  message: string,
  data?: unknown
): IPCError {
  return {
    code,
    message,
    data,
  };
}
