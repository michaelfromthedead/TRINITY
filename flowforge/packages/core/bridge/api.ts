/**
 * FlowForge Bridge Layer - API Interface
 *
 * This module defines the abstract API interface that both the mock and Tauri
 * implementations must conform to. It mirrors the ComfyUI API contract to ensure
 * compatibility during migration.
 */

import type {
  NodeDefinitions,
  QueueStatus,
  QueueItem,
  QueueItemDetail,
  ComfyWorkflowJSON,
  ComfyApiWorkflow,
  PromptResponse,
  ExecutionResult,
  ExecutionEvent,
  SystemStats,
  UserConfig,
  UserDataFullInfo,
  ModelFolderInfo,
  ModelFileInfo,
  AssetUploadResult,
  AssetInfo,
  ExtensionsResponse,
  EmbeddingsResponse,
  WorkflowTemplates,
  QueuePromptOptions,
  ClientId,
  PromptId,
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

// ============================================================================
// Event Subscription Types
// ============================================================================

/** Event listener callback */
export type EventListener<T extends ExecutionEvent = ExecutionEvent> = (event: T) => void;

/** Unsubscribe function returned by event subscriptions */
export type UnsubscribeFn = () => void;

// ============================================================================
// API Interface
// ============================================================================

/**
 * Abstract API interface for FlowForge.
 *
 * This interface defines all methods that the frontend uses to communicate
 * with the execution backend. Implementations include:
 * - MockAPI: For development and testing without a backend
 * - TauriAPI: For production use with Tauri IPC
 *
 * The interface mirrors the ComfyUI API contract to ensure compatibility.
 */
export interface IAPI {
  // ==========================================================================
  // Connection Management
  // ==========================================================================

  /**
   * Initialize the API connection.
   * Sets up WebSocket/event listeners for realtime updates.
   */
  init(): Promise<void>;

  /**
   * Disconnect and clean up resources.
   */
  disconnect(): Promise<void>;

  /**
   * Get the current client ID.
   */
  getClientId(): ClientId | undefined;

  /**
   * Check if the API is connected and ready.
   */
  isConnected(): boolean;

  // ==========================================================================
  // Node Definitions
  // ==========================================================================

  /**
   * Get all available node definitions.
   * Equivalent to ComfyUI's GET /object_info
   */
  getObjectInfo(): Promise<NodeDefinitions>;

  /**
   * Get extensions list.
   */
  getExtensions(): Promise<ExtensionsResponse>;

  /**
   * Get embeddings list.
   */
  getEmbeddings(): Promise<EmbeddingsResponse>;

  // ==========================================================================
  // Workflow Execution
  // ==========================================================================

  /**
   * Queue a workflow for execution.
   *
   * @param number - Queue position (-1 for front, 0 for back)
   * @param data - Workflow data containing output (API format) and workflow (JSON format)
   * @param options - Optional execution options
   * @returns Promise resolving to prompt response with prompt_id
   */
  queuePrompt(
    number: number,
    data: {
      output: ComfyApiWorkflow;
      workflow: ComfyWorkflowJSON;
    },
    options?: QueuePromptOptions
  ): Promise<PromptResponse>;

  /**
   * Execute a workflow and wait for completion.
   * This is a convenience method that queues and waits for results.
   *
   * @param workflow - The workflow to execute (API format)
   * @returns Promise resolving to execution result
   */
  executeWorkflow(workflow: ComfyApiWorkflow): Promise<ExecutionResult>;

  /**
   * Interrupt the currently running execution.
   *
   * @param runningPromptId - Optional prompt ID to interrupt
   */
  interrupt(runningPromptId?: string | null): Promise<void>;

  // ==========================================================================
  // Queue Management
  // ==========================================================================

  /**
   * Get the current queue status.
   */
  getQueue(): Promise<QueueStatus>;

  /**
   * Get execution history.
   *
   * @param maxItems - Maximum number of items to return
   * @param options - Optional offset for pagination
   */
  getHistory(maxItems?: number, options?: { offset?: number }): Promise<QueueItem[]>;

  /**
   * Get detailed information about a specific job.
   *
   * @param jobId - The job/prompt ID
   */
  getJobDetail(jobId: string): Promise<QueueItemDetail | undefined>;

  /**
   * Delete an item from queue or history.
   *
   * @param type - 'queue' or 'history'
   * @param id - The item ID to delete
   */
  deleteItem(type: 'queue' | 'history', id: string): Promise<void>;

  /**
   * Clear all items from queue or history.
   *
   * @param type - 'queue' or 'history'
   */
  clearItems(type: 'queue' | 'history'): Promise<void>;

  // ==========================================================================
  // Event Subscription
  // ==========================================================================

  /**
   * Subscribe to execution events.
   *
   * @param callback - Function called when events are received
   * @returns Unsubscribe function
   */
  onExecutionEvent(callback: EventListener): UnsubscribeFn;

  /**
   * Subscribe to a specific event type.
   *
   * @param eventType - The event type to subscribe to
   * @param callback - Function called when the event is received
   * @returns Unsubscribe function
   */
  on<T extends ExecutionEvent['type']>(
    eventType: T,
    callback: (event: Extract<ExecutionEvent, { type: T }>) => void
  ): UnsubscribeFn;

  /**
   * Emit an event (for frontend-generated events).
   *
   * @param event - The event to emit
   */
  emit(event: ExecutionEvent): void;

  // ==========================================================================
  // System Information
  // ==========================================================================

  /**
   * Get system statistics.
   */
  getSystemStats(): Promise<SystemStats>;

  /**
   * Free memory by unloading models.
   *
   * @param options - Options for memory cleanup
   */
  freeMemory(options: { freeExecutionCache: boolean }): Promise<void>;

  // ==========================================================================
  // User Data
  // ==========================================================================

  /**
   * Get user configuration.
   */
  getUserConfig(): Promise<UserConfig>;

  /**
   * Create a new user.
   *
   * @param username - The username to create
   */
  createUser(username: string): Promise<void>;

  /**
   * Get all settings.
   */
  getSettings(): Promise<Record<string, unknown>>;

  /**
   * Get a specific setting.
   *
   * @param id - The setting ID
   */
  getSetting(id: string): Promise<unknown>;

  /**
   * Store multiple settings.
   *
   * @param settings - Dictionary of settings to store
   */
  storeSettings(settings: Record<string, unknown>): Promise<void>;

  /**
   * Store a single setting.
   *
   * @param id - The setting ID
   * @param value - The value to store
   */
  storeSetting(id: string, value: unknown): Promise<void>;

  /**
   * Get user data file content.
   *
   * @param file - The file name
   * @param options - Request options
   */
  getUserData(file: string, options?: RequestInit): Promise<Response>;

  /**
   * Store user data file.
   *
   * @param file - The file name
   * @param data - The data to store
   * @param options - Storage options
   */
  storeUserData(
    file: string,
    data: unknown,
    options?: {
      overwrite?: boolean;
      stringify?: boolean;
      throwOnError?: boolean;
      full_info?: boolean;
    }
  ): Promise<Response>;

  /**
   * Delete user data file.
   *
   * @param file - The file name to delete
   */
  deleteUserData(file: string): Promise<void>;

  /**
   * Move user data file.
   *
   * @param source - Source file path
   * @param dest - Destination file path
   * @param options - Move options
   */
  moveUserData(
    source: string,
    dest: string,
    options?: { overwrite?: boolean }
  ): Promise<void>;

  /**
   * List user data files with full info.
   *
   * @param dir - Directory to list
   */
  listUserDataFullInfo(dir: string): Promise<UserDataFullInfo[]>;

  // ==========================================================================
  // Model Management
  // ==========================================================================

  /**
   * Get list of model folders.
   */
  getModelFolders(): Promise<ModelFolderInfo[]>;

  /**
   * Get list of models in a folder.
   *
   * @param folder - The folder name (e.g., 'checkpoints', 'loras')
   */
  getModels(folder: string): Promise<ModelFileInfo[]>;

  /**
   * Get model metadata.
   *
   * @param folder - The folder containing the model
   * @param model - The model filename
   */
  viewMetadata(folder: string, model: string): Promise<unknown>;

  // ==========================================================================
  // Asset Management
  // ==========================================================================

  /**
   * Import an asset (image, video, etc.) into the workspace.
   *
   * @param path - Path to the file to import
   * @returns Asset upload result with ID and local path
   */
  importAsset(path: string): Promise<AssetUploadResult>;

  /**
   * Get the URL for viewing an asset.
   *
   * @param asset - Asset info with filename, subfolder, and type
   * @returns URL string for the asset
   */
  getAssetUrl(asset: AssetInfo): string;

  /**
   * Upload an asset from a File or Blob.
   *
   * @param file - The file to upload
   * @param options - Upload options
   */
  uploadAsset(
    file: File | Blob,
    options?: {
      filename?: string;
      subfolder?: string;
      type?: 'input' | 'output' | 'temp';
    }
  ): Promise<AssetUploadResult>;

  // ==========================================================================
  // Workflow Templates
  // ==========================================================================

  /**
   * Get workflow templates from custom nodes.
   */
  getWorkflowTemplates(): Promise<WorkflowTemplates>;

  /**
   * Get core workflow templates.
   *
   * @param locale - Optional locale code
   */
  getCoreWorkflowTemplates(locale?: string): Promise<unknown[]>;

  // ==========================================================================
  // File Operations
  // ==========================================================================

  /**
   * Open a file dialog and load a workflow.
   *
   * @returns Promise resolving to the loaded workflow and path, or undefined if cancelled
   */
  openWorkflow(): Promise<{ workflow: ComfyWorkflowJSON; path: string } | undefined>;

  /**
   * Save a workflow to a file.
   *
   * @param workflow - The workflow to save
   * @param path - Optional path (shows save dialog if not provided)
   * @returns Promise resolving to the saved path
   */
  saveWorkflow(workflow: ComfyWorkflowJSON, path?: string): Promise<string | undefined>;

  // ==========================================================================
  // Folder Paths
  // ==========================================================================

  /**
   * Get folder paths configuration.
   */
  getFolderPaths(): Promise<Record<string, string[]>>;

  // ==========================================================================
  // Feature Flags
  // ==========================================================================

  /**
   * Get client feature flags.
   */
  getClientFeatureFlags(): Record<string, unknown>;

  /**
   * Get server feature flags.
   */
  getServerFeatures(): Record<string, unknown>;

  /**
   * Check if server supports a feature.
   *
   * @param featureName - The feature name (supports dot notation)
   */
  serverSupportsFeature(featureName: string): boolean;

  /**
   * Get a server feature value.
   *
   * @param featureName - The feature name (supports dot notation)
   * @param defaultValue - Default value if not found
   */
  getServerFeature<T = unknown>(featureName: string, defaultValue?: T): T;

  // ==========================================================================
  // Logs
  // ==========================================================================

  /**
   * Get logs as text.
   */
  getLogs(): Promise<string>;

  /**
   * Get raw logs with metadata.
   */
  getRawLogs(): Promise<{ size: { cols: number; row: number }; entries: Array<{ t: string; m: string }> }>;

  /**
   * Subscribe/unsubscribe to log streaming.
   *
   * @param enabled - Whether to enable log streaming
   */
  subscribeLogs(enabled: boolean): Promise<void>;

  // ==========================================================================
  // Internationalization
  // ==========================================================================

  /**
   * Get custom nodes i18n data.
   */
  getCustomNodesI18n(): Promise<Record<string, unknown>>;

  // ==========================================================================
  // Trinity Bridge Protocol (T-TL-1.1)
  // ==========================================================================

  /**
   * Make a generic JSON-RPC 2.0 request to the Trinity bridge.
   * Routes to the appropriate channel based on the method prefix.
   */
  trinityRequest<T = unknown>(method: string, params?: unknown): Promise<T>;

  /**
   * Register a new type (component, system, resource, or event) in the
   * Trinity type registry.
   */
  trinityRegisterType(request: TypeRegisterRequest): Promise<TypeRegisterResponse>;

  /**
   * List all registered types, optionally filtered by kind.
   */
  trinityListTypes(request?: TypeListRequest): Promise<TypeListResponse>;

  /**
   * Get details for a specific type by its type_id.
   */
  trinityGetType(request: TypeGetRequest): Promise<TypeRegistryEntry>;

  /**
   * Read a single field value from a component instance by its FieldKey.
   */
  trinityReadField(request: ComponentReadRequest): Promise<unknown>;
}

// ============================================================================
// API Factory Type
// ============================================================================

/** Factory function type for creating API instances */
export type APIFactory = () => IAPI;

// ============================================================================
// API Events Interface (for typed event handling)
// ============================================================================

/** Typed event map for the API */
export interface APIEventMap {
  status: Extract<ExecutionEvent, { type: 'status' }>;
  progress: Extract<ExecutionEvent, { type: 'progress' }>;
  progress_state: Extract<ExecutionEvent, { type: 'progress_state' }>;
  executing: Extract<ExecutionEvent, { type: 'executing' }>;
  executed: Extract<ExecutionEvent, { type: 'executed' }>;
  execution_start: Extract<ExecutionEvent, { type: 'execution_start' }>;
  execution_success: Extract<ExecutionEvent, { type: 'execution_success' }>;
  execution_cached: Extract<ExecutionEvent, { type: 'execution_cached' }>;
  execution_interrupted: Extract<ExecutionEvent, { type: 'execution_interrupted' }>;
  execution_error: Extract<ExecutionEvent, { type: 'execution_error' }>;
  progress_text: Extract<ExecutionEvent, { type: 'progress_text' }>;
  b_preview: Extract<ExecutionEvent, { type: 'b_preview' }>;
  b_preview_with_metadata: Extract<ExecutionEvent, { type: 'b_preview_with_metadata' }>;
  logs: Extract<ExecutionEvent, { type: 'logs' }>;
  notification: Extract<ExecutionEvent, { type: 'notification' }>;
  feature_flags: Extract<ExecutionEvent, { type: 'feature_flags' }>;
  asset_download: Extract<ExecutionEvent, { type: 'asset_download' }>;
  graphChanged: Extract<ExecutionEvent, { type: 'graphChanged' }>;
  promptQueued: Extract<ExecutionEvent, { type: 'promptQueued' }>;
  graphCleared: Extract<ExecutionEvent, { type: 'graphCleared' }>;
  reconnecting: Extract<ExecutionEvent, { type: 'reconnecting' }>;
  reconnected: Extract<ExecutionEvent, { type: 'reconnected' }>;
}
