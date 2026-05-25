/**
 * FlowForge Bridge Layer - Main Export
 *
 * This module provides environment-aware exports for the bridge layer.
 * It automatically selects the appropriate API implementation based on
 * the runtime environment (Tauri vs Browser/Node).
 */

// ============================================================================
// Type Exports
// ============================================================================

export type {
  // Basic types
  NodeId,
  PromptId,
  ClientId,
  DataType,
  SlotIndex,
  ResultItemType,

  // Input types
  BaseInputOptions,
  IntInputOptions,
  FloatInputOptions,
  BooleanInputOptions,
  StringInputOptions,
  ComboInputOptions,
  RemoteWidgetConfig,
  MultiSelectOption,
  InputSpec,
  IntInputSpec,
  FloatInputSpec,
  BooleanInputSpec,
  StringInputSpec,
  ComboInputSpec,
  ComboInputSpecV2,
  CustomInputSpec,
  ComfyInputsSpec,

  // Output types
  OutputTypeSpec,
  ComfyOutputTypesSpec,

  // Node definition types
  WidgetDependency,
  PriceBadgeDepends,
  PriceBadge,
  ComfyNodeDef,
  NodeDefinitions,

  // Workflow types
  Vector2,
  NodeFlags,
  NodeInput,
  NodeOutput,
  NodeProperties,
  WorkflowNode,
  WorkflowLink,
  WorkflowLinkObject,
  WorkflowGroup,
  DisplayState,
  WorkflowConfig,
  Reroute,
  LinkExtension,
  WorkflowExtra,
  GraphState,
  ModelFile,
  ComfyWorkflowJSON04,
  ComfyWorkflowJSON1,
  ComfyWorkflowJSON,

  // API workflow types
  NodeInputValue,
  ApiNodeData,
  ComfyApiWorkflow,

  // Execution types
  ResultItem,
  NodeExecutionOutput,
  TaskOutput,
  ExecutionErrorDetails,
  NodeError,
  PromptResponse,
  ExecutionResult,

  // Event types
  QueueStatusInfo,
  StatusEvent,
  ProgressEvent,
  NodeProgressState,
  ProgressStateEvent,
  ExecutingEvent,
  ExecutedEvent,
  ExecutionStartEvent,
  ExecutionSuccessEvent,
  ExecutionCachedEvent,
  ExecutionInterruptedEvent,
  ExecutionErrorEvent,
  ProgressTextEvent,
  BinaryPreviewEvent,
  BinaryPreviewWithMetadataEvent,
  LogEntry,
  TerminalSize,
  LogsEvent,
  NotificationEvent,
  FeatureFlagsEvent,
  AssetDownloadEvent,
  GraphChangedEvent,
  PromptQueuedEvent,
  GraphClearedEvent,
  ReconnectingEvent,
  ReconnectedEvent,
  ExecutionEvent,

  // Queue types
  JobStatus,
  PreviewOutput,
  JobExecutionError,
  QueueItem,
  QueueItemDetail,
  QueueStatus,
  PaginationInfo,
  JobsListResponse,

  // System types
  DeviceStats,
  SystemStats,
  UserConfig,
  UserDataFullInfo,

  // Asset types
  ModelFolderInfo,
  ModelFileInfo,
  AssetUploadResult,
  AssetInfo,

  // Options types
  PreviewMethod,
  NodeExecutionId,
  QueuePromptOptions,

  // Extension types
  ExtensionsResponse,
  EmbeddingsResponse,
  WorkflowTemplateInfo,
  WorkflowTemplates,

  // IPC types
  IPCMessage,

  // Trinity bridge protocol types
  TypeKind,
  FieldDescriptor,
  TypeRegisterRequest,
  TypeRegisterResponse,
  TypeRegistryEntry,
  TypeListRequest,
  TypeListResponse,
  TypeGetRequest,
  TypeRemoveRequest,
  TypeCountResponse,
  FieldKey,
  FieldReadResult,
  FieldInit,
  ComponentBlock,
  ComponentReadRequest,
  ComponentWriteRequest,
  ComponentDeleteRequest,
  ComponentBatchReadRequest,
  ComponentBatchReadResponse,
  ComponentBatchWriteRequest,
  ComponentBatchWriteResponse,
  WorldCreateRequest,
  WorldCreateResponse,
  WorldSpawnRequest,
  WorldSpawnResponse,
  WorldDespawnRequest,
  WorldQueryRequest,
  WorldQueryResponse,
  WorldResetRequest,
  WorldStatsResponse,
  TrinityConnectRequest,
  TrinityConnectResponse,
  TrinityStatusResponse,
  TrinityInspectRequest,
  TrinityInspectResponse,
  InspectorGetRequest,
  SourceLocation,
  DecoratorEntry,
  HierarchyEntry,
  EventEntry,
  EventsRecentResponse,
  ChecksumResponse,
  JsonRpcRequest,
  JsonRpcResponse,
  JsonRpcNotification,
} from './types';

// Type guards
export {
  isComboInputSpecV1,
  isComboInputSpecV2,
  isComboInputSpec,
  getInputSpecType,
  getComboSpecComboOptions,

  // Trinity bridge protocol
  channelForMethod,
  TYPE_CHANNEL_PREFIX,
  DATA_CHANNEL_PREFIX,
  COMMAND_CHANNEL_PREFIX,
  SYSTEM_CHANNEL_PREFIX,
  TYPE_CHANNEL_ENDPOINTS,
  DATA_CHANNEL_ENDPOINTS,
  COMMAND_CHANNEL_ENDPOINTS,
  SYSTEM_CHANNEL_ENDPOINTS,
  TOTAL_ENDPOINTS,
  METHOD_TABLE,
} from './types';

// ============================================================================
// API Interface Exports
// ============================================================================

export type {
  IAPI,
  EventListener,
  UnsubscribeFn,
  APIFactory,
  APIEventMap,
} from './api';

// ============================================================================
// Implementation Exports
// ============================================================================

export { MockAPI } from './mock-api';
export { TauriAPI, createTauriAPI, isTauriEnvironment } from './tauri-api';
export {
  MockWebSocket,
  MockWebSocketReadyState,
  MockOpenEvent,
  MockCloseEvent,
  MockMessageEvent,
  MockErrorEvent,
  BinaryMessageType,
  createMockWebSocket,
  installMockWebSocket,
  saveMockWebSocket,
  restoreWebSocket,
} from './mock-socket';

// ============================================================================
// Environment Detection
// ============================================================================

/**
 * Check if running in a Tauri environment
 */
export function isTauri(): boolean {
  return typeof window !== 'undefined' && '__TAURI__' in window;
}

/**
 * Check if running in a browser environment
 */
export function isBrowser(): boolean {
  return typeof window !== 'undefined' && typeof document !== 'undefined';
}

/**
 * Check if running in a Node.js environment
 */
export function isNode(): boolean {
  return typeof process !== 'undefined' && process.versions?.node !== undefined;
}

/**
 * Check if running in development mode
 */
export function isDevelopment(): boolean {
  if (typeof process !== 'undefined') {
    return process.env.NODE_ENV === 'development';
  }
  return false;
}

// ============================================================================
// API Factory
// ============================================================================

import type { IAPI } from './api';
import { MockAPI } from './mock-api';
import { TauriAPI } from './tauri-api';

/** API instance cache */
let apiInstance: IAPI | null = null;

/** Configuration for API creation */
export interface APIConfig {
  /** Force using mock API even in Tauri environment */
  forceMock?: boolean;
  /** Options for mock API */
  mockOptions?: {
    executionDelay?: number;
  };
}

/**
 * Create an API instance based on the runtime environment.
 *
 * In Tauri environments, creates a TauriAPI instance.
 * In browser/Node environments, creates a MockAPI instance.
 *
 * @param config - Optional configuration
 * @returns The appropriate API implementation
 */
export function createAPI(config?: APIConfig): IAPI {
  if (config?.forceMock || !isTauri()) {
    return new MockAPI(config?.mockOptions);
  }
  return new TauriAPI();
}

/**
 * Get or create a singleton API instance.
 *
 * This is useful for applications that want a single shared API instance.
 *
 * @param config - Optional configuration (only used on first call)
 * @returns The singleton API instance
 */
export function getAPI(config?: APIConfig): IAPI {
  if (!apiInstance) {
    apiInstance = createAPI(config);
  }
  return apiInstance;
}

/**
 * Reset the singleton API instance.
 *
 * This is useful for testing or when switching environments.
 */
export async function resetAPI(): Promise<void> {
  if (apiInstance) {
    await apiInstance.disconnect();
    apiInstance = null;
  }
}

/**
 * Set a custom API instance.
 *
 * This is useful for testing or for injecting custom implementations.
 *
 * @param api - The API instance to use
 */
export function setAPI(api: IAPI): void {
  apiInstance = api;
}

// ============================================================================
// Convenience Re-exports for Common Use Cases
// ============================================================================

/**
 * Initialize and get the API instance.
 *
 * This is a convenience function that creates the API and initializes it.
 *
 * @param config - Optional configuration
 * @returns Initialized API instance
 */
export async function initAPI(config?: APIConfig): Promise<IAPI> {
  const api = getAPI(config);
  if (!api.isConnected()) {
    await api.init();
  }
  return api;
}

// ============================================================================
// Default Export
// ============================================================================

/**
 * Default export provides the most common use case:
 * - Use getAPI() for singleton access
 * - Use createAPI() for new instances
 * - Use initAPI() for initialized singletons
 */
export default {
  create: createAPI,
  get: getAPI,
  init: initAPI,
  reset: resetAPI,
  set: setAPI,
  isTauri,
  isBrowser,
  isNode,
  isDevelopment,
};
