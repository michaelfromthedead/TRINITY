/**
 * FlowForge Bridge Layer - Mock WebSocket Implementation
 *
 * This module provides a mock WebSocket implementation that simulates
 * the ComfyUI WebSocket protocol for development and testing.
 */

import type {
  ExecutionEvent,
  StatusEvent,
  ProgressEvent,
  ExecutingEvent,
  ExecutedEvent,
  ExecutionStartEvent,
  ExecutionSuccessEvent,
  ExecutionErrorEvent,
  PromptId,
  NodeId,
  NodeExecutionOutput,
  QueueStatusInfo,
} from './types';

// ============================================================================
// Types
// ============================================================================

/** WebSocket ready states */
export enum MockWebSocketReadyState {
  CONNECTING = 0,
  OPEN = 1,
  CLOSING = 2,
  CLOSED = 3,
}

/** WebSocket event types */
export type MockWebSocketEventType = 'open' | 'close' | 'message' | 'error';

/** WebSocket event handler */
export type MockWebSocketEventHandler = (event: unknown) => void;

/** Binary message types from ComfyUI protocol */
export enum BinaryMessageType {
  PREVIEW_IMAGE = 1,
  PROGRESS_TEXT = 3,
  PREVIEW_IMAGE_WITH_METADATA = 4,
}

// ============================================================================
// Mock WebSocket Event Classes
// ============================================================================

/** Mock open event */
export class MockOpenEvent {
  readonly type = 'open';
}

/** Mock close event */
export class MockCloseEvent {
  readonly type = 'close';
  readonly code: number;
  readonly reason: string;
  readonly wasClean: boolean;

  constructor(code = 1000, reason = '', wasClean = true) {
    this.code = code;
    this.reason = reason;
    this.wasClean = wasClean;
  }
}

/** Mock message event */
export class MockMessageEvent {
  readonly type = 'message';
  readonly data: string | ArrayBuffer;

  constructor(data: string | ArrayBuffer) {
    this.data = data;
  }
}

/** Mock error event */
export class MockErrorEvent {
  readonly type = 'error';
  readonly message: string;
  readonly error: Error;

  constructor(message: string, error?: Error) {
    this.message = message;
    this.error = error || new Error(message);
  }
}

// ============================================================================
// Mock WebSocket Implementation
// ============================================================================

/**
 * Mock WebSocket that simulates the ComfyUI WebSocket protocol.
 *
 * This implementation can be used for:
 * - Unit testing without a real server
 * - Development previews without backend
 * - Simulating various server states and events
 */
export class MockWebSocket {
  // WebSocket standard properties
  readonly url: string;
  readonly protocol: string;
  binaryType: BinaryType = 'arraybuffer';

  private _readyState: MockWebSocketReadyState = MockWebSocketReadyState.CONNECTING;

  // Event handlers
  onopen: ((event: MockOpenEvent) => void) | null = null;
  onclose: ((event: MockCloseEvent) => void) | null = null;
  onmessage: ((event: MockMessageEvent) => void) | null = null;
  onerror: ((event: MockErrorEvent) => void) | null = null;

  // Event listener storage
  private eventListeners: Map<MockWebSocketEventType, Set<MockWebSocketEventHandler>> = new Map();

  // Mock state
  private clientId: string;
  private isConnected = false;
  private autoConnect: boolean;
  private connectDelay: number;
  private messageQueue: Array<string | ArrayBuffer> = [];

  constructor(
    url: string,
    _protocols?: string | string[],
    options: {
      autoConnect?: boolean;
      connectDelay?: number;
      clientId?: string;
    } = {}
  ) {
    this.url = url;
    this.protocol = '';
    this.autoConnect = options.autoConnect ?? true;
    this.connectDelay = options.connectDelay ?? 50;
    this.clientId = options.clientId || `mock-${Date.now()}`;

    // Auto-connect if enabled
    if (this.autoConnect) {
      setTimeout(() => this.simulateConnect(), this.connectDelay);
    }
  }

  // ==========================================================================
  // WebSocket Standard Interface
  // ==========================================================================

  get readyState(): MockWebSocketReadyState {
    return this._readyState;
  }

  send(data: string | ArrayBuffer | Blob): void {
    if (this._readyState !== MockWebSocketReadyState.OPEN) {
      throw new Error('WebSocket is not open');
    }

    // Parse incoming messages (for bidirectional communication)
    if (typeof data === 'string') {
      try {
        const message = JSON.parse(data);
        this.handleIncomingMessage(message);
      } catch {
        // Ignore non-JSON messages
      }
    }
  }

  close(code = 1000, reason = ''): void {
    if (this._readyState === MockWebSocketReadyState.CLOSED) return;

    this._readyState = MockWebSocketReadyState.CLOSING;

    setTimeout(() => {
      this._readyState = MockWebSocketReadyState.CLOSED;
      this.isConnected = false;

      const event = new MockCloseEvent(code, reason, true);
      this.dispatchEvent('close', event);
      this.onclose?.(event);
    }, 10);
  }

  addEventListener(type: MockWebSocketEventType, handler: MockWebSocketEventHandler): void {
    if (!this.eventListeners.has(type)) {
      this.eventListeners.set(type, new Set());
    }
    this.eventListeners.get(type)!.add(handler);
  }

  removeEventListener(type: MockWebSocketEventType, handler: MockWebSocketEventHandler): void {
    this.eventListeners.get(type)?.delete(handler);
  }

  // ==========================================================================
  // Mock Control Methods
  // ==========================================================================

  /**
   * Simulate successful connection
   */
  simulateConnect(): void {
    if (this._readyState !== MockWebSocketReadyState.CONNECTING) return;

    this._readyState = MockWebSocketReadyState.OPEN;
    this.isConnected = true;

    const event = new MockOpenEvent();
    this.dispatchEvent('open', event);
    this.onopen?.(event);

    // Send initial status message with client ID
    this.simulateMessage({
      type: 'status',
      data: {
        status: { exec_info: { queue_remaining: 0 } },
        sid: this.clientId,
      },
    } as StatusEvent);

    // Flush any queued messages
    for (const msg of this.messageQueue) {
      this.simulateRawMessage(msg);
    }
    this.messageQueue = [];
  }

  /**
   * Simulate connection error
   */
  simulateError(message = 'Connection failed'): void {
    const event = new MockErrorEvent(message);
    this.dispatchEvent('error', event);
    this.onerror?.(event);

    // Close the connection after error
    this.close(1006, message);
  }

  /**
   * Simulate receiving a message from the server
   */
  simulateMessage(event: ExecutionEvent): void {
    if (this._readyState !== MockWebSocketReadyState.OPEN) {
      // Queue message for later if not connected
      this.messageQueue.push(JSON.stringify(event));
      return;
    }

    const messageEvent = new MockMessageEvent(JSON.stringify(event));
    this.dispatchEvent('message', messageEvent);
    this.onmessage?.(messageEvent);
  }

  /**
   * Simulate receiving a raw message (string or binary)
   */
  simulateRawMessage(data: string | ArrayBuffer): void {
    if (this._readyState !== MockWebSocketReadyState.OPEN) {
      this.messageQueue.push(data);
      return;
    }

    const messageEvent = new MockMessageEvent(data);
    this.dispatchEvent('message', messageEvent);
    this.onmessage?.(messageEvent);
  }

  /**
   * Simulate receiving a binary preview image
   */
  simulateBinaryPreview(imageData: Uint8Array, imageType: 'jpeg' | 'png' = 'jpeg'): void {
    const typeCode = imageType === 'png' ? 2 : 1;

    // Create binary message: [eventType: u32, imageType: u32, imageData: bytes]
    const buffer = new ArrayBuffer(8 + imageData.length);
    const view = new DataView(buffer);
    view.setUint32(0, BinaryMessageType.PREVIEW_IMAGE);
    view.setUint32(4, typeCode);
    new Uint8Array(buffer, 8).set(imageData);

    this.simulateRawMessage(buffer);
  }

  /**
   * Simulate receiving a binary preview with metadata
   */
  simulateBinaryPreviewWithMetadata(
    imageData: Uint8Array,
    metadata: {
      node_id: string;
      display_node_id: string;
      parent_node_id: string;
      real_node_id: string;
      prompt_id: string;
      image_type?: string;
    }
  ): void {
    const metadataJson = JSON.stringify({
      ...metadata,
      image_type: metadata.image_type || 'image/jpeg',
    });
    const metadataBytes = new TextEncoder().encode(metadataJson);

    // Create binary message: [eventType: u32, metadataLength: u32, metadata: bytes, imageData: bytes]
    const buffer = new ArrayBuffer(8 + metadataBytes.length + imageData.length);
    const view = new DataView(buffer);
    view.setUint32(0, BinaryMessageType.PREVIEW_IMAGE_WITH_METADATA);
    view.setUint32(4, metadataBytes.length);
    new Uint8Array(buffer, 8, metadataBytes.length).set(metadataBytes);
    new Uint8Array(buffer, 8 + metadataBytes.length).set(imageData);

    this.simulateRawMessage(buffer);
  }

  /**
   * Simulate receiving progress text
   */
  simulateProgressText(nodeId: string, text: string): void {
    const nodeIdBytes = new TextEncoder().encode(nodeId);
    const textBytes = new TextEncoder().encode(text);

    // Create binary message: [eventType: u32, nodeIdLength: u32, nodeId: bytes, text: bytes]
    const buffer = new ArrayBuffer(8 + nodeIdBytes.length + textBytes.length);
    const view = new DataView(buffer);
    view.setUint32(0, BinaryMessageType.PROGRESS_TEXT);
    view.setUint32(4, nodeIdBytes.length);
    new Uint8Array(buffer, 8, nodeIdBytes.length).set(nodeIdBytes);
    new Uint8Array(buffer, 8 + nodeIdBytes.length).set(textBytes);

    this.simulateRawMessage(buffer);
  }

  /**
   * Simulate a complete workflow execution sequence
   */
  async simulateWorkflowExecution(
    promptId: PromptId,
    nodeIds: NodeId[],
    options: {
      stepDelay?: number;
      progressSteps?: number;
      generateOutputs?: boolean;
      failOnNode?: NodeId;
    } = {}
  ): Promise<void> {
    const {
      stepDelay = 100,
      progressSteps = 10,
      generateOutputs = true,
      failOnNode,
    } = options;

    // Execution start
    this.simulateMessage({
      type: 'execution_start',
      data: {
        prompt_id: promptId,
        timestamp: Date.now(),
      },
    } as ExecutionStartEvent);

    // Process each node
    for (let i = 0; i < nodeIds.length; i++) {
      const nodeId = nodeIds[i];

      // Check if this node should fail
      if (failOnNode === nodeId) {
        this.simulateMessage({
          type: 'execution_error',
          data: {
            prompt_id: promptId,
            timestamp: Date.now(),
            node_id: nodeId,
            node_type: 'MockNode',
            executed: nodeIds.slice(0, i),
            exception_message: 'Simulated failure',
            exception_type: 'MockError',
            traceback: ['Mock traceback line 1', 'Mock traceback line 2'],
            current_inputs: {},
            current_outputs: {},
          },
        } as ExecutionErrorEvent);
        return;
      }

      // Executing event
      this.simulateMessage({
        type: 'executing',
        data: {
          node: nodeId,
          display_node: nodeId,
          prompt_id: promptId,
        },
      } as ExecutingEvent);

      // Progress events
      for (let step = 0; step <= progressSteps; step++) {
        await this.delay(stepDelay / progressSteps);

        this.simulateMessage({
          type: 'progress',
          data: {
            value: step,
            max: progressSteps,
            prompt_id: promptId,
            node: nodeId,
          },
        } as ProgressEvent);
      }

      // Executed event
      const output: NodeExecutionOutput = generateOutputs
        ? { result: `output_${nodeId}` }
        : {};

      this.simulateMessage({
        type: 'executed',
        data: {
          node: nodeId,
          display_node: nodeId,
          prompt_id: promptId,
          output,
        },
      } as ExecutedEvent);

      // Update queue status
      const remaining = (nodeIds.length - i - 1) / nodeIds.length;
      this.simulateMessage({
        type: 'status',
        data: {
          status: { exec_info: { queue_remaining: remaining } },
        },
      } as StatusEvent);
    }

    // Execution complete (executing null)
    this.simulateMessage({
      type: 'executing',
      data: {
        node: null as unknown as NodeId,
        display_node: null as unknown as NodeId,
        prompt_id: promptId,
      },
    } as ExecutingEvent);

    // Success event
    this.simulateMessage({
      type: 'execution_success',
      data: {
        prompt_id: promptId,
        timestamp: Date.now(),
      },
    } as ExecutionSuccessEvent);

    // Final status update
    this.simulateMessage({
      type: 'status',
      data: {
        status: { exec_info: { queue_remaining: 0 } },
      },
    } as StatusEvent);
  }

  /**
   * Get the client ID
   */
  getClientId(): string {
    return this.clientId;
  }

  /**
   * Update queue status
   */
  updateQueueStatus(status: QueueStatusInfo): void {
    this.simulateMessage({
      type: 'status',
      data: { status },
    } as StatusEvent);
  }

  // ==========================================================================
  // Private Helpers
  // ==========================================================================

  private dispatchEvent(type: MockWebSocketEventType, event: unknown): void {
    const listeners = this.eventListeners.get(type);
    if (listeners) {
      for (const handler of listeners) {
        try {
          handler(event);
        } catch (error) {
          console.error(`[MockWebSocket] Error in ${type} handler:`, error);
        }
      }
    }
  }

  private handleIncomingMessage(message: Record<string, unknown>): void {
    // Handle feature flags message from client
    if (message.type === 'feature_flags') {
      console.log('[MockWebSocket] Received client feature flags:', message.data);
      // Could emit server feature flags in response
      this.simulateMessage({
        type: 'feature_flags',
        data: {
          mock_server: true,
          version: '1.0.0',
        },
      } as ExecutionEvent);
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a mock WebSocket that can be used as a drop-in replacement
 * for the native WebSocket in tests and development.
 */
export function createMockWebSocket(
  url: string,
  options?: {
    autoConnect?: boolean;
    connectDelay?: number;
    clientId?: string;
  }
): MockWebSocket {
  return new MockWebSocket(url, undefined, options);
}

/**
 * Install the mock WebSocket globally (for testing environments)
 */
export function installMockWebSocket(): void {
  (globalThis as unknown as { WebSocket: typeof MockWebSocket }).WebSocket = MockWebSocket;
}

/**
 * Restore the original WebSocket (after testing)
 */
let originalWebSocket: typeof WebSocket | undefined;

export function saveMockWebSocket(): void {
  originalWebSocket = globalThis.WebSocket;
}

export function restoreWebSocket(): void {
  if (originalWebSocket) {
    globalThis.WebSocket = originalWebSocket;
    originalWebSocket = undefined;
  }
}
