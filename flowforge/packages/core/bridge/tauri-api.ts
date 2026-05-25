/**
 * FlowForge Bridge Layer - Tauri IPC Implementation
 *
 * This module provides the production implementation of the IAPI interface
 * using Tauri's IPC mechanism. It communicates with the Rust backend which
 * in turn manages the Bun sidecar process for workflow execution.
 *
 * NOTE: This is a skeleton implementation. The actual Tauri commands must
 * be implemented in the Rust backend (src-tauri/).
 */

import type { IAPI, EventListener, UnsubscribeFn } from './api';
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
  // Trinity bridge protocol types
  TypeRegisterRequest,
  TypeRegisterResponse,
  TypeRegistryEntry,
  TypeListRequest,
  TypeListResponse,
  TypeGetRequest,
  ComponentReadRequest,
} from './types';

// ============================================================================
// Tauri API Types (will be provided by @tauri-apps/api)
// ============================================================================

/**
 * Type definitions for Tauri API functions.
 * These are stubs - the actual types come from @tauri-apps/api
 */
interface TauriCore {
  invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T>;
  convertFileSrc(path: string): string;
}

interface TauriEvent {
  listen<T>(
    event: string,
    handler: (event: { payload: T }) => void
  ): Promise<() => void>;
  emit(event: string, payload?: unknown): Promise<void>;
}

interface TauriDialog {
  open(options?: {
    multiple?: boolean;
    directory?: boolean;
    filters?: Array<{ name: string; extensions: string[] }>;
    defaultPath?: string;
  }): Promise<string | string[] | null>;
  save(options?: {
    filters?: Array<{ name: string; extensions: string[] }>;
    defaultPath?: string;
  }): Promise<string | null>;
}

// Lazy-loaded Tauri modules
let tauriCore: TauriCore | null = null;
let tauriEvent: TauriEvent | null = null;
let tauriDialog: TauriDialog | null = null;

/**
 * Initialize Tauri modules lazily
 */
async function initTauri(): Promise<void> {
  if (tauriCore && tauriEvent && tauriDialog) return;

  try {
    // Dynamic import to avoid issues in non-Tauri environments
    const core = await import('@tauri-apps/api/core');
    const event = await import('@tauri-apps/api/event');
    const dialog = await import('@tauri-apps/plugin-dialog');

    tauriCore = core as unknown as TauriCore;
    tauriEvent = event as unknown as TauriEvent;
    tauriDialog = dialog as unknown as TauriDialog;
  } catch (error) {
    throw new Error(
      'Failed to initialize Tauri API. Make sure you are running in a Tauri environment.'
    );
  }
}

/**
 * Get Tauri core module (invoke, convertFileSrc)
 */
function getCore(): TauriCore {
  if (!tauriCore) {
    throw new Error('Tauri not initialized. Call init() first.');
  }
  return tauriCore;
}

/**
 * Get Tauri event module (listen, emit)
 */
function getEvent(): TauriEvent {
  if (!tauriEvent) {
    throw new Error('Tauri not initialized. Call init() first.');
  }
  return tauriEvent;
}

/**
 * Get Tauri dialog module (open, save)
 */
function getDialog(): TauriDialog {
  if (!tauriDialog) {
    throw new Error('Tauri not initialized. Call init() first.');
  }
  return tauriDialog;
}

// ============================================================================
// Tauri API Implementation
// ============================================================================

/**
 * Production API implementation using Tauri IPC.
 *
 * This class communicates with the Rust backend via invoke() for commands
 * and listen() for events. The Rust backend manages the Bun sidecar process.
 */
export class TauriAPI implements IAPI {
  private clientId: ClientId | undefined;
  private connected = false;
  private eventListeners: Map<string, Set<EventListener>> = new Map();
  private globalListeners: Set<EventListener> = new Set();
  private unsubscribeFns: Array<() => void> = [];
  private serverFeatureFlags: Record<string, unknown> = {};
  private clientFeatureFlags: Record<string, unknown> = {};

  // ==========================================================================
  // Connection Management
  // ==========================================================================

  async init(): Promise<void> {
    await initTauri();

    // Initialize the backend sidecar
    const initResult = await getCore().invoke<{ client_id: string }>('init_engine');
    this.clientId = initResult.client_id;
    this.connected = true;

    // Subscribe to execution events from the backend
    const unsubscribe = await getEvent().listen<ExecutionEvent>(
      'execution_event',
      (event) => {
        this.handleEvent(event.payload);
      }
    );
    this.unsubscribeFns.push(unsubscribe);

    // Request feature flags from server
    try {
      this.serverFeatureFlags = await getCore().invoke('get_feature_flags');
    } catch {
      // Feature flags are optional
    }

    console.log('[TauriAPI] Initialized with client ID:', this.clientId);
  }

  async disconnect(): Promise<void> {
    // Unsubscribe from all events
    for (const unsubscribe of this.unsubscribeFns) {
      unsubscribe();
    }
    this.unsubscribeFns = [];
    this.eventListeners.clear();
    this.globalListeners.clear();

    // Shutdown the engine
    try {
      await getCore().invoke('shutdown_engine');
    } catch (error) {
      console.warn('[TauriAPI] Error during shutdown:', error);
    }

    this.connected = false;
    this.clientId = undefined;
    console.log('[TauriAPI] Disconnected');
  }

  getClientId(): ClientId | undefined {
    return this.clientId;
  }

  isConnected(): boolean {
    return this.connected;
  }

  // ==========================================================================
  // Node Definitions
  // ==========================================================================

  async getObjectInfo(): Promise<NodeDefinitions> {
    return getCore().invoke('get_object_info');
  }

  async getExtensions(): Promise<ExtensionsResponse> {
    return getCore().invoke('get_extensions');
  }

  async getEmbeddings(): Promise<EmbeddingsResponse> {
    return getCore().invoke('get_embeddings');
  }

  // ==========================================================================
  // Workflow Execution
  // ==========================================================================

  async queuePrompt(
    number: number,
    data: {
      output: ComfyApiWorkflow;
      workflow: ComfyWorkflowJSON;
    },
    options?: QueuePromptOptions
  ): Promise<PromptResponse> {
    return getCore().invoke('queue_prompt', {
      number,
      prompt: data.output,
      workflow: data.workflow,
      options: options || {},
    });
  }

  async executeWorkflow(workflow: ComfyApiWorkflow): Promise<ExecutionResult> {
    return getCore().invoke('execute_workflow', { workflow });
  }

  async interrupt(runningPromptId?: string | null): Promise<void> {
    await getCore().invoke('interrupt', { prompt_id: runningPromptId });
  }

  // ==========================================================================
  // Queue Management
  // ==========================================================================

  async getQueue(): Promise<QueueStatus> {
    return getCore().invoke('get_queue');
  }

  async getHistory(maxItems = 200, options?: { offset?: number }): Promise<QueueItem[]> {
    return getCore().invoke('get_history', {
      max_items: maxItems,
      offset: options?.offset,
    });
  }

  async getJobDetail(jobId: string): Promise<QueueItemDetail | undefined> {
    return getCore().invoke('get_job_detail', { job_id: jobId });
  }

  async deleteItem(type: 'queue' | 'history', id: string): Promise<void> {
    await getCore().invoke('delete_item', { type, id });
  }

  async clearItems(type: 'queue' | 'history'): Promise<void> {
    await getCore().invoke('clear_items', { type });
  }

  // ==========================================================================
  // Event Subscription
  // ==========================================================================

  onExecutionEvent(callback: EventListener): UnsubscribeFn {
    this.globalListeners.add(callback);
    return () => {
      this.globalListeners.delete(callback);
    };
  }

  on<T extends ExecutionEvent['type']>(
    eventType: T,
    callback: (event: Extract<ExecutionEvent, { type: T }>) => void
  ): UnsubscribeFn {
    if (!this.eventListeners.has(eventType)) {
      this.eventListeners.set(eventType, new Set());
    }
    const listeners = this.eventListeners.get(eventType)!;
    listeners.add(callback as EventListener);

    return () => {
      listeners.delete(callback as EventListener);
    };
  }

  emit(event: ExecutionEvent): void {
    // Emit to backend
    getEvent().emit('frontend_event', event).catch((error) => {
      console.error('[TauriAPI] Error emitting event:', error);
    });

    // Also handle locally
    this.handleEvent(event);
  }

  private handleEvent(event: ExecutionEvent): void {
    // Handle status events for client ID and feature flags
    if (event.type === 'status' && event.data.sid) {
      this.clientId = event.data.sid;
    }
    if (event.type === 'feature_flags') {
      this.serverFeatureFlags = event.data;
    }

    // Notify type-specific listeners
    const listeners = this.eventListeners.get(event.type);
    if (listeners) {
      for (const listener of listeners) {
        try {
          listener(event);
        } catch (error) {
          console.error('[TauriAPI] Error in event listener:', error);
        }
      }
    }

    // Notify global listeners
    for (const listener of this.globalListeners) {
      try {
        listener(event);
      } catch (error) {
        console.error('[TauriAPI] Error in global listener:', error);
      }
    }
  }

  // ==========================================================================
  // System Information
  // ==========================================================================

  async getSystemStats(): Promise<SystemStats> {
    return getCore().invoke('get_system_stats');
  }

  async freeMemory(options: { freeExecutionCache: boolean }): Promise<void> {
    await getCore().invoke('free_memory', options);
  }

  // ==========================================================================
  // User Data
  // ==========================================================================

  async getUserConfig(): Promise<UserConfig> {
    return getCore().invoke('get_user_config');
  }

  async createUser(username: string): Promise<void> {
    await getCore().invoke('create_user', { username });
  }

  async getSettings(): Promise<Record<string, unknown>> {
    return getCore().invoke('get_settings');
  }

  async getSetting(id: string): Promise<unknown> {
    return getCore().invoke('get_setting', { id });
  }

  async storeSettings(settings: Record<string, unknown>): Promise<void> {
    await getCore().invoke('store_settings', { settings });
  }

  async storeSetting(id: string, value: unknown): Promise<void> {
    await getCore().invoke('store_setting', { id, value });
  }

  async getUserData(file: string, _options?: RequestInit): Promise<Response> {
    const data = await getCore().invoke<unknown>('get_user_data', { file });
    return new Response(data ? JSON.stringify(data) : null, {
      status: data ? 200 : 404,
    });
  }

  async storeUserData(
    file: string,
    data: unknown,
    options?: {
      overwrite?: boolean;
      stringify?: boolean;
      throwOnError?: boolean;
      full_info?: boolean;
    }
  ): Promise<Response> {
    await getCore().invoke('store_user_data', {
      file,
      data: options?.stringify !== false ? JSON.stringify(data) : data,
      overwrite: options?.overwrite ?? true,
      full_info: options?.full_info ?? false,
    });
    return new Response(null, { status: 200 });
  }

  async deleteUserData(file: string): Promise<void> {
    await getCore().invoke('delete_user_data', { file });
  }

  async moveUserData(
    source: string,
    dest: string,
    options?: { overwrite?: boolean }
  ): Promise<void> {
    await getCore().invoke('move_user_data', {
      source,
      dest,
      overwrite: options?.overwrite ?? false,
    });
  }

  async listUserDataFullInfo(dir: string): Promise<UserDataFullInfo[]> {
    return getCore().invoke('list_user_data', { dir });
  }

  // ==========================================================================
  // Model Management
  // ==========================================================================

  async getModelFolders(): Promise<ModelFolderInfo[]> {
    return getCore().invoke('get_model_folders');
  }

  async getModels(folder: string): Promise<ModelFileInfo[]> {
    return getCore().invoke('get_models', { folder });
  }

  async viewMetadata(folder: string, model: string): Promise<unknown> {
    return getCore().invoke('view_metadata', { folder, model });
  }

  // ==========================================================================
  // Asset Management
  // ==========================================================================

  async importAsset(path: string): Promise<AssetUploadResult> {
    return getCore().invoke('import_asset', { path });
  }

  getAssetUrl(asset: AssetInfo): string {
    // Use Tauri's asset protocol
    const subfolder = asset.subfolder ? `${asset.subfolder}/` : '';
    const fullPath = `${asset.type}/${subfolder}${asset.path}`;

    try {
      return getCore().convertFileSrc(fullPath);
    } catch {
      // Fallback if Tauri not available
      return `tauri://localhost/assets/${fullPath}`;
    }
  }

  async uploadAsset(
    file: File | Blob,
    options?: {
      filename?: string;
      subfolder?: string;
      type?: 'input' | 'output' | 'temp';
    }
  ): Promise<AssetUploadResult> {
    // Convert File/Blob to base64 for IPC
    const arrayBuffer = await file.arrayBuffer();
    const base64 = btoa(
      new Uint8Array(arrayBuffer).reduce(
        (data, byte) => data + String.fromCharCode(byte),
        ''
      )
    );

    return getCore().invoke('upload_asset', {
      data: base64,
      filename: options?.filename || (file instanceof File ? file.name : 'blob'),
      subfolder: options?.subfolder,
      type: options?.type || 'input',
    });
  }

  // ==========================================================================
  // Workflow Templates
  // ==========================================================================

  async getWorkflowTemplates(): Promise<WorkflowTemplates> {
    return getCore().invoke('get_workflow_templates');
  }

  async getCoreWorkflowTemplates(locale?: string): Promise<unknown[]> {
    return getCore().invoke('get_core_workflow_templates', { locale });
  }

  // ==========================================================================
  // File Operations
  // ==========================================================================

  async openWorkflow(): Promise<{ workflow: ComfyWorkflowJSON; path: string } | undefined> {
    const path = await getDialog().open({
      filters: [{ name: 'Workflow', extensions: ['json', 'workflow'] }],
    });

    if (!path || Array.isArray(path)) return undefined;

    const workflow = await getCore().invoke<ComfyWorkflowJSON>('load_workflow', {
      path,
    });

    return { workflow, path };
  }

  async saveWorkflow(
    workflow: ComfyWorkflowJSON,
    path?: string
  ): Promise<string | undefined> {
    let savePath = path;

    if (!savePath) {
      savePath = await getDialog().save({
        filters: [{ name: 'Workflow', extensions: ['json'] }],
      }) || undefined;
    }

    if (!savePath) return undefined;

    await getCore().invoke('save_workflow', {
      workflow,
      path: savePath,
    });

    return savePath;
  }

  // ==========================================================================
  // Folder Paths
  // ==========================================================================

  async getFolderPaths(): Promise<Record<string, string[]>> {
    return getCore().invoke('get_folder_paths');
  }

  // ==========================================================================
  // Feature Flags
  // ==========================================================================

  getClientFeatureFlags(): Record<string, unknown> {
    return { ...this.clientFeatureFlags };
  }

  getServerFeatures(): Record<string, unknown> {
    return { ...this.serverFeatureFlags };
  }

  serverSupportsFeature(featureName: string): boolean {
    const parts = featureName.split('.');
    let current: unknown = this.serverFeatureFlags;
    for (const part of parts) {
      if (typeof current !== 'object' || current === null) {
        return false;
      }
      current = (current as Record<string, unknown>)[part];
    }
    return current === true;
  }

  getServerFeature<T = unknown>(featureName: string, defaultValue?: T): T {
    const parts = featureName.split('.');
    let current: unknown = this.serverFeatureFlags;
    for (const part of parts) {
      if (typeof current !== 'object' || current === null) {
        return defaultValue as T;
      }
      current = (current as Record<string, unknown>)[part];
    }
    return (current ?? defaultValue) as T;
  }

  /**
   * Set client feature flags (called during init)
   */
  setClientFeatureFlags(flags: Record<string, unknown>): void {
    this.clientFeatureFlags = flags;
  }

  // ==========================================================================
  // Logs
  // ==========================================================================

  async getLogs(): Promise<string> {
    return getCore().invoke('get_logs');
  }

  async getRawLogs(): Promise<{
    size: { cols: number; row: number };
    entries: Array<{ t: string; m: string }>;
  }> {
    return getCore().invoke('get_raw_logs');
  }

  async subscribeLogs(enabled: boolean): Promise<void> {
    await getCore().invoke('subscribe_logs', { enabled, client_id: this.clientId });
  }

  // ==========================================================================
  // Internationalization
  // ==========================================================================

  async getCustomNodesI18n(): Promise<Record<string, unknown>> {
    return getCore().invoke('get_custom_nodes_i18n');
  }

  // ==========================================================================
  // Trinity Bridge Protocol (T-TL-1.1)
  // ==========================================================================

  async trinityRequest<T = unknown>(method: string, params?: unknown): Promise<T> {
    return getCore().invoke('ipc_call', { method, params });
  }

  async trinityRegisterType(request: TypeRegisterRequest): Promise<TypeRegisterResponse> {
    return getCore().invoke('ipc_call', {
      method: 'type.register',
      params: request,
    });
  }

  async trinityListTypes(request?: TypeListRequest): Promise<TypeListResponse> {
    return getCore().invoke('ipc_call', {
      method: 'type.list',
      params: request || {},
    });
  }

  async trinityGetType(request: TypeGetRequest): Promise<TypeRegistryEntry> {
    return getCore().invoke('ipc_call', {
      method: 'type.get',
      params: request,
    });
  }

  async trinityReadField(request: ComponentReadRequest): Promise<unknown> {
    return getCore().invoke('ipc_call', {
      method: 'data.read',
      params: request,
    });
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a new TauriAPI instance.
 */
export function createTauriAPI(): TauriAPI {
  return new TauriAPI();
}

/**
 * Check if running in a Tauri environment.
 */
export function isTauriEnvironment(): boolean {
  return typeof window !== 'undefined' && '__TAURI__' in window;
}
