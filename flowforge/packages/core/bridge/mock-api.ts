/**
 * FlowForge Bridge Layer - Mock API Implementation
 *
 * This module provides a mock implementation of the IAPI interface for
 * development and testing purposes. It simulates the backend behavior
 * without requiring an actual Tauri/Bun backend.
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
  TaskOutput,
  NodeId,
  // Trinity bridge protocol types
  TypeKind,
  FieldDescriptor,
  TypeRegisterRequest,
  TypeRegisterResponse,
  TypeRegistryEntry,
  TypeListRequest,
  TypeListResponse,
  TypeGetRequest,
  FieldKey,
  ComponentReadRequest,
} from './types';

// ============================================================================
// Constants
// ============================================================================

/** Default configuration values */
const MOCK_DEFAULTS = {
  /** Default execution delay in milliseconds */
  EXECUTION_DELAY: 100,
  /** Maximum random latency added to simulate network */
  MAX_LATENCY_JITTER: 50,
  /** Minimum latency in milliseconds */
  MIN_LATENCY: 10,
  /** Simulated RAM total in bytes (16GB) */
  RAM_TOTAL: 16 * 1024 * 1024 * 1024,
  /** Simulated RAM free in bytes (8GB) */
  RAM_FREE: 8 * 1024 * 1024 * 1024,
  /** Simulated VRAM total in bytes (8GB) */
  VRAM_TOTAL: 8 * 1024 * 1024 * 1024,
  /** Simulated VRAM free in bytes (6GB) */
  VRAM_FREE: 6 * 1024 * 1024 * 1024,
  /** Number of progress steps to emit during node execution */
  PROGRESS_STEPS: 10,
} as const;

// ============================================================================
// Mock Data
// ============================================================================

/** Generate a unique ID */
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
}

/** Mock node definitions */
const MOCK_NODE_DEFINITIONS: NodeDefinitions = {
  'Math/Add': {
    name: 'Math/Add',
    display_name: 'Add',
    description: 'Adds two numbers together',
    category: 'Math',
    input: {
      required: {
        a: ['FLOAT', { default: 0, min: -Infinity, max: Infinity }],
        b: ['FLOAT', { default: 0, min: -Infinity, max: Infinity }],
      },
    },
    output: ['FLOAT'],
    output_name: ['result'],
    output_is_list: [false],
    output_node: false,
    python_module: 'flowforge.nodes.math',
  },
  'Math/Subtract': {
    name: 'Math/Subtract',
    display_name: 'Subtract',
    description: 'Subtracts one number from another',
    category: 'Math',
    input: {
      required: {
        a: ['FLOAT', { default: 0 }],
        b: ['FLOAT', { default: 0 }],
      },
    },
    output: ['FLOAT'],
    output_name: ['result'],
    output_is_list: [false],
    output_node: false,
    python_module: 'flowforge.nodes.math',
  },
  'Math/Multiply': {
    name: 'Math/Multiply',
    display_name: 'Multiply',
    description: 'Multiplies two numbers',
    category: 'Math',
    input: {
      required: {
        a: ['FLOAT', { default: 1 }],
        b: ['FLOAT', { default: 1 }],
      },
    },
    output: ['FLOAT'],
    output_name: ['result'],
    output_is_list: [false],
    output_node: false,
    python_module: 'flowforge.nodes.math',
  },
  'Math/Divide': {
    name: 'Math/Divide',
    display_name: 'Divide',
    description: 'Divides one number by another',
    category: 'Math',
    input: {
      required: {
        a: ['FLOAT', { default: 1 }],
        b: ['FLOAT', { default: 1, min: 0.0001 }],
      },
    },
    output: ['FLOAT'],
    output_name: ['result'],
    output_is_list: [false],
    output_node: false,
    python_module: 'flowforge.nodes.math',
  },
  'Logic/If': {
    name: 'Logic/If',
    display_name: 'If Condition',
    description: 'Conditionally routes data based on a boolean',
    category: 'Logic',
    input: {
      required: {
        condition: ['BOOLEAN', { default: false }],
        if_true: ['*'],
        if_false: ['*'],
      },
    },
    output: ['*'],
    output_name: ['result'],
    output_is_list: [false],
    output_node: false,
    python_module: 'flowforge.nodes.logic',
  },
  'Logic/Compare': {
    name: 'Logic/Compare',
    display_name: 'Compare',
    description: 'Compares two values',
    category: 'Logic',
    input: {
      required: {
        a: ['*'],
        b: ['*'],
        operator: [['==', '!=', '<', '>', '<=', '>='], { default: '==' }],
      },
    },
    output: ['BOOLEAN'],
    output_name: ['result'],
    output_is_list: [false],
    output_node: false,
    python_module: 'flowforge.nodes.logic',
  },
  'String/Concat': {
    name: 'String/Concat',
    display_name: 'Concatenate',
    description: 'Concatenates two strings',
    category: 'String',
    input: {
      required: {
        a: ['STRING', { default: '' }],
        b: ['STRING', { default: '' }],
      },
      optional: {
        separator: ['STRING', { default: '' }],
      },
    },
    output: ['STRING'],
    output_name: ['result'],
    output_is_list: [false],
    output_node: false,
    python_module: 'flowforge.nodes.string',
  },
  'String/Format': {
    name: 'String/Format',
    display_name: 'Format String',
    description: 'Formats a string with placeholders',
    category: 'String',
    input: {
      required: {
        template: ['STRING', { default: 'Hello, {name}!', multiline: true }],
      },
      optional: {
        values: ['DICT'],
      },
    },
    output: ['STRING'],
    output_name: ['result'],
    output_is_list: [false],
    output_node: false,
    python_module: 'flowforge.nodes.string',
  },
  'Input/Number': {
    name: 'Input/Number',
    display_name: 'Number Input',
    description: 'A number input widget',
    category: 'Input',
    input: {
      required: {
        value: ['FLOAT', { default: 0 }],
      },
    },
    output: ['FLOAT'],
    output_name: ['value'],
    output_is_list: [false],
    output_node: false,
    python_module: 'flowforge.nodes.input',
  },
  'Input/Text': {
    name: 'Input/Text',
    display_name: 'Text Input',
    description: 'A text input widget',
    category: 'Input',
    input: {
      required: {
        value: ['STRING', { default: '', multiline: true }],
      },
    },
    output: ['STRING'],
    output_name: ['value'],
    output_is_list: [false],
    output_node: false,
    python_module: 'flowforge.nodes.input',
  },
  'Output/Display': {
    name: 'Output/Display',
    display_name: 'Display',
    description: 'Displays a value',
    category: 'Output',
    input: {
      required: {
        value: ['*'],
      },
    },
    output: [],
    output_name: [],
    output_is_list: [],
    output_node: true,
    python_module: 'flowforge.nodes.output',
  },
  'Output/Log': {
    name: 'Output/Log',
    display_name: 'Log',
    description: 'Logs a value to the console',
    category: 'Output',
    input: {
      required: {
        value: ['*'],
        label: ['STRING', { default: 'Value' }],
      },
    },
    output: [],
    output_name: [],
    output_is_list: [],
    output_node: true,
    python_module: 'flowforge.nodes.output',
  },
};

/** Mock system stats */
const MOCK_SYSTEM_STATS: SystemStats = {
  system: {
    os: 'FlowForge/Mock',
    python_version: 'N/A (TypeScript)',
    embedded_python: false,
    comfyui_version: 'FlowForge 1.0.0',
    pytorch_version: 'N/A',
    argv: [],
    ram_total: MOCK_DEFAULTS.RAM_TOTAL,
    ram_free: MOCK_DEFAULTS.RAM_FREE,
  },
  devices: [
    {
      name: 'Mock GPU',
      type: 'cuda',
      index: 0,
      vram_total: MOCK_DEFAULTS.VRAM_TOTAL,
      vram_free: MOCK_DEFAULTS.VRAM_FREE,
      torch_vram_total: MOCK_DEFAULTS.VRAM_TOTAL,
      torch_vram_free: MOCK_DEFAULTS.VRAM_FREE,
    },
  ],
};

// ============================================================================
// Mock API Implementation
// ============================================================================

/**
 * Mock API implementation for development and testing.
 */
export class MockAPI implements IAPI {
  private clientId: ClientId;
  private connected = false;
  private eventListeners: Map<string, Set<EventListener>> = new Map();
  private globalListeners: Set<EventListener> = new Set();
  private queue: QueueItem[] = [];
  private history: QueueItem[] = [];
  private settings: Record<string, unknown> = {};
  private userData: Map<string, unknown> = new Map();
  private serverFeatureFlags: Record<string, unknown> = {};
  private executionDelay: number;
  private trinityTypeRegistry: Map<string, TypeRegistryEntry> = new Map();
  private trinityComponentStore: Map<string, Map<string, unknown>> = new Map();

  constructor(options: { executionDelay?: number } = {}) {
    this.clientId = generateId();
    this.executionDelay = options.executionDelay ?? MOCK_DEFAULTS.EXECUTION_DELAY;
    this.trinityTypeRegistry = new Map();
    this.trinityComponentStore = new Map();
  }

  // ==========================================================================
  // Connection Management
  // ==========================================================================

  async init(): Promise<void> {
    this.connected = true;

    // Emit initial status
    this.emitInternal({
      type: 'status',
      data: {
        status: { exec_info: { queue_remaining: 0 } },
        sid: this.clientId,
      },
    });

    console.log('[MockAPI] Initialized with client ID:', this.clientId);
  }

  async disconnect(): Promise<void> {
    this.connected = false;
    this.eventListeners.clear();
    this.globalListeners.clear();
    console.log('[MockAPI] Disconnected');
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
    await this.simulateLatency();
    return { ...MOCK_NODE_DEFINITIONS };
  }

  async getExtensions(): Promise<ExtensionsResponse> {
    await this.simulateLatency();
    return ['flowforge-core'];
  }

  async getEmbeddings(): Promise<EmbeddingsResponse> {
    await this.simulateLatency();
    return [];
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
    _options?: QueuePromptOptions
  ): Promise<PromptResponse> {
    await this.simulateLatency();

    const promptId = generateId();
    const now = Date.now();

    const queueItem: QueueItem = {
      id: promptId,
      status: 'pending',
      create_time: now,
      priority: number === -1 ? 1000 : number || 0,
    };

    if (number === -1) {
      this.queue.unshift(queueItem);
    } else {
      this.queue.push(queueItem);
    }

    // Emit prompt queued event
    this.emitInternal({
      type: 'promptQueued',
      data: { number: this.queue.length, batchCount: 1 },
    });

    // Emit status update
    this.emitInternal({
      type: 'status',
      data: {
        status: { exec_info: { queue_remaining: this.queue.length } },
      },
    });

    // Start execution in background
    this.simulateExecution(promptId, data.output);

    return {
      prompt_id: promptId,
      exec_info: { queue_remaining: this.queue.length },
    };
  }

  async executeWorkflow(workflow: ComfyApiWorkflow): Promise<ExecutionResult> {
    const response = await this.queuePrompt(0, {
      output: workflow,
      workflow: {} as ComfyWorkflowJSON,
    });

    // Wait for execution to complete
    return new Promise((resolve) => {
      let unsubscribeSuccess: UnsubscribeFn | null = null;
      let unsubscribeError: UnsubscribeFn | null = null;

      const cleanup = () => {
        unsubscribeSuccess?.();
        unsubscribeError?.();
      };

      unsubscribeSuccess = this.on('execution_success', (event) => {
        if (event.data.prompt_id === response.prompt_id) {
          cleanup();
          resolve({
            success: true,
            prompt_id: response.prompt_id!,
            outputs: {},
          });
        }
      });

      unsubscribeError = this.on('execution_error', (event) => {
        if (event.data.prompt_id === response.prompt_id) {
          cleanup();
          resolve({
            success: false,
            prompt_id: response.prompt_id!,
            error: event.data.exception_message,
          });
        }
      });
    });
  }

  async interrupt(runningPromptId?: string | null): Promise<void> {
    await this.simulateLatency();

    const runningIndex = this.queue.findIndex((item) => item.status === 'in_progress');
    if (runningIndex !== -1) {
      const item = this.queue[runningIndex];
      if (!runningPromptId || item.id === runningPromptId) {
        item.status = 'cancelled';
        this.queue.splice(runningIndex, 1);
        this.history.unshift(item);

        this.emitInternal({
          type: 'execution_interrupted',
          data: {
            prompt_id: item.id,
            timestamp: Date.now(),
            node_id: '',
            node_type: '',
            executed: [],
          },
        });
      }
    }
  }

  // ==========================================================================
  // Queue Management
  // ==========================================================================

  async getQueue(): Promise<QueueStatus> {
    await this.simulateLatency();

    return {
      Running: this.queue.filter((item) => item.status === 'in_progress'),
      Pending: this.queue.filter((item) => item.status === 'pending'),
    };
  }

  async getHistory(maxItems = 200, _options?: { offset?: number }): Promise<QueueItem[]> {
    await this.simulateLatency();
    return this.history.slice(0, maxItems);
  }

  async getJobDetail(jobId: string): Promise<QueueItemDetail | undefined> {
    await this.simulateLatency();

    const item = [...this.queue, ...this.history].find((i) => i.id === jobId);
    if (item) {
      return {
        ...item,
        outputs: {},
      };
    }
    return undefined;
  }

  async deleteItem(type: 'queue' | 'history', id: string): Promise<void> {
    await this.simulateLatency();

    if (type === 'queue') {
      const index = this.queue.findIndex((item) => item.id === id);
      if (index !== -1) {
        this.queue.splice(index, 1);
      }
    } else {
      const index = this.history.findIndex((item) => item.id === id);
      if (index !== -1) {
        this.history.splice(index, 1);
      }
    }
  }

  async clearItems(type: 'queue' | 'history'): Promise<void> {
    await this.simulateLatency();

    if (type === 'queue') {
      this.queue = this.queue.filter((item) => item.status === 'in_progress');
    } else {
      this.history = [];
    }
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
    this.emitInternal(event);
  }

  private emitInternal(event: ExecutionEvent): void {
    // Notify type-specific listeners
    const listeners = this.eventListeners.get(event.type);
    if (listeners) {
      for (const listener of listeners) {
        try {
          listener(event);
        } catch (error) {
          console.error('[MockAPI] Error in event listener:', error);
        }
      }
    }

    // Notify global listeners
    for (const listener of this.globalListeners) {
      try {
        listener(event);
      } catch (error) {
        console.error('[MockAPI] Error in global listener:', error);
      }
    }
  }

  // ==========================================================================
  // System Information
  // ==========================================================================

  async getSystemStats(): Promise<SystemStats> {
    await this.simulateLatency();
    return { ...MOCK_SYSTEM_STATS };
  }

  async freeMemory(_options: { freeExecutionCache: boolean }): Promise<void> {
    await this.simulateLatency();
    console.log('[MockAPI] Memory freed (simulated)');
  }

  // ==========================================================================
  // User Data
  // ==========================================================================

  async getUserConfig(): Promise<UserConfig> {
    await this.simulateLatency();
    return { storage: 'server' };
  }

  async createUser(_username: string): Promise<void> {
    await this.simulateLatency();
  }

  async getSettings(): Promise<Record<string, unknown>> {
    await this.simulateLatency();
    return { ...this.settings };
  }

  async getSetting(id: string): Promise<unknown> {
    await this.simulateLatency();
    return this.settings[id];
  }

  async storeSettings(settings: Record<string, unknown>): Promise<void> {
    await this.simulateLatency();
    this.settings = { ...this.settings, ...settings };
  }

  async storeSetting(id: string, value: unknown): Promise<void> {
    await this.simulateLatency();
    this.settings[id] = value;
  }

  async getUserData(_file: string, _options?: RequestInit): Promise<Response> {
    await this.simulateLatency();
    const data = this.userData.get(_file);
    return new Response(data ? JSON.stringify(data) : null, {
      status: data ? 200 : 404,
    });
  }

  async storeUserData(
    file: string,
    data: unknown,
    _options?: {
      overwrite?: boolean;
      stringify?: boolean;
      throwOnError?: boolean;
      full_info?: boolean;
    }
  ): Promise<Response> {
    await this.simulateLatency();
    this.userData.set(file, data);
    return new Response(null, { status: 200 });
  }

  async deleteUserData(file: string): Promise<void> {
    await this.simulateLatency();
    this.userData.delete(file);
  }

  async moveUserData(
    source: string,
    dest: string,
    _options?: { overwrite?: boolean }
  ): Promise<void> {
    await this.simulateLatency();
    const data = this.userData.get(source);
    if (data) {
      this.userData.set(dest, data);
      this.userData.delete(source);
    }
  }

  async listUserDataFullInfo(_dir: string): Promise<UserDataFullInfo[]> {
    await this.simulateLatency();
    const files: UserDataFullInfo[] = [];
    for (const [path] of this.userData) {
      if (path.startsWith(_dir)) {
        files.push({
          path,
          size: 0,
          modified: Date.now(),
        });
      }
    }
    return files;
  }

  // ==========================================================================
  // Model Management
  // ==========================================================================

  async getModelFolders(): Promise<ModelFolderInfo[]> {
    await this.simulateLatency();
    return [
      { name: 'checkpoints' },
      { name: 'loras' },
      { name: 'embeddings' },
      { name: 'controlnet' },
    ];
  }

  async getModels(folder: string): Promise<ModelFileInfo[]> {
    await this.simulateLatency();
    // Return mock models based on folder
    const mockModels: Record<string, ModelFileInfo[]> = {
      checkpoints: [
        { name: 'mock_model_v1.safetensors' },
        { name: 'mock_model_v2.safetensors' },
      ],
      loras: [
        { name: 'mock_lora_1.safetensors' },
      ],
      embeddings: [],
      controlnet: [],
    };
    return mockModels[folder] || [];
  }

  async viewMetadata(_folder: string, _model: string): Promise<unknown> {
    await this.simulateLatency();
    return null;
  }

  // ==========================================================================
  // Asset Management
  // ==========================================================================

  async importAsset(path: string): Promise<AssetUploadResult> {
    await this.simulateLatency();
    const id = generateId();
    const name = path.split('/').pop() || 'unknown';
    return {
      id,
      localPath: path,
      name,
    };
  }

  getAssetUrl(asset: AssetInfo): string {
    // In mock mode, just return a placeholder URL
    const subfolder = asset.subfolder ? `${asset.subfolder}/` : '';
    return `mock://assets/${asset.type}/${subfolder}${asset.path}`;
  }

  async uploadAsset(
    file: File | Blob,
    options?: {
      filename?: string;
      subfolder?: string;
      type?: 'input' | 'output' | 'temp';
    }
  ): Promise<AssetUploadResult> {
    await this.simulateLatency();
    const id = generateId();
    const name = options?.filename || (file instanceof File ? file.name : 'blob');
    return {
      id,
      localPath: `mock://uploads/${id}/${name}`,
      name,
      subfolder: options?.subfolder,
    };
  }

  // ==========================================================================
  // Workflow Templates
  // ==========================================================================

  async getWorkflowTemplates(): Promise<WorkflowTemplates> {
    await this.simulateLatency();
    return {};
  }

  async getCoreWorkflowTemplates(_locale?: string): Promise<unknown[]> {
    await this.simulateLatency();
    return [];
  }

  // ==========================================================================
  // File Operations
  // ==========================================================================

  async openWorkflow(): Promise<{ workflow: ComfyWorkflowJSON; path: string } | undefined> {
    // In mock mode, we can't actually open a file dialog
    console.log('[MockAPI] openWorkflow called - returning undefined (mock mode)');
    return undefined;
  }

  async saveWorkflow(
    _workflow: ComfyWorkflowJSON,
    _path?: string
  ): Promise<string | undefined> {
    // In mock mode, we can't actually save to disk
    console.log('[MockAPI] saveWorkflow called - returning mock path');
    return 'mock://workflow.json';
  }

  // ==========================================================================
  // Folder Paths
  // ==========================================================================

  async getFolderPaths(): Promise<Record<string, string[]>> {
    await this.simulateLatency();
    return {
      checkpoints: ['mock://models/checkpoints'],
      loras: ['mock://models/loras'],
      embeddings: ['mock://models/embeddings'],
      controlnet: ['mock://models/controlnet'],
      input: ['mock://input'],
      output: ['mock://output'],
      temp: ['mock://temp'],
    };
  }

  // ==========================================================================
  // Feature Flags
  // ==========================================================================

  getClientFeatureFlags(): Record<string, unknown> {
    return {
      flowforge_mock: true,
    };
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

  // ==========================================================================
  // Logs
  // ==========================================================================

  async getLogs(): Promise<string> {
    await this.simulateLatency();
    return '[MockAPI] This is a mock log output\n[MockAPI] No actual logs in mock mode';
  }

  async getRawLogs(): Promise<{ size: { cols: number; row: number }; entries: Array<{ t: string; m: string }> }> {
    await this.simulateLatency();
    return {
      size: { cols: 120, row: 24 },
      entries: [
        { t: new Date().toISOString(), m: '[MockAPI] Mock log entry' },
      ],
    };
  }

  async subscribeLogs(_enabled: boolean): Promise<void> {
    await this.simulateLatency();
  }

  // ==========================================================================
  // Internationalization
  // ==========================================================================

  async getCustomNodesI18n(): Promise<Record<string, unknown>> {
    await this.simulateLatency();
    return {};
  }

  // ==========================================================================
  // Private Helpers
  // ==========================================================================

  private async simulateLatency(): Promise<void> {
    if (this.executionDelay > 0) {
      const jitter = Math.random() * MOCK_DEFAULTS.MAX_LATENCY_JITTER + MOCK_DEFAULTS.MIN_LATENCY;
      await new Promise((resolve) => setTimeout(resolve, jitter));
    }
  }

  private async simulateExecution(promptId: string, workflow: ComfyApiWorkflow): Promise<void> {
    // Find the queue item and mark it as running
    const queueItem = this.queue.find((item) => item.id === promptId);
    if (!queueItem) return;

    queueItem.status = 'in_progress';
    queueItem.execution_start_time = Date.now();

    // Emit execution start
    this.emitInternal({
      type: 'execution_start',
      data: {
        prompt_id: promptId,
        timestamp: Date.now(),
      },
    });

    // Get nodes in execution order (simplified - just process in order)
    const nodeIds = Object.keys(workflow);
    const totalNodes = nodeIds.length;
    const outputs: TaskOutput = {};

    try {
      for (let i = 0; i < nodeIds.length; i++) {
        const nodeId = nodeIds[i] as NodeId;
        const nodeData = workflow[nodeId as string];

        // Emit executing event
        this.emitInternal({
          type: 'executing',
          data: {
            node: nodeId,
            display_node: nodeId,
            prompt_id: promptId,
          },
        });

        // Simulate execution with progress
        const steps = MOCK_DEFAULTS.PROGRESS_STEPS;
        for (let step = 0; step <= steps; step++) {
          await new Promise((resolve) => setTimeout(resolve, this.executionDelay / steps));

          this.emitInternal({
            type: 'progress',
            data: {
              value: step,
              max: steps,
              prompt_id: promptId,
              node: nodeId,
            },
          });
        }

        // Generate mock output based on node type
        const output = this.generateMockOutput(nodeData.class_type);
        outputs[nodeId] = output;

        // Emit executed event
        this.emitInternal({
          type: 'executed',
          data: {
            node: nodeId,
            display_node: nodeId,
            prompt_id: promptId,
            output,
          },
        });

        // Update status with remaining nodes
        this.emitInternal({
          type: 'status',
          data: {
            status: {
              exec_info: {
                queue_remaining: this.queue.length - 1 + (totalNodes - i - 1) / totalNodes,
              },
            },
          },
        });
      }

      // Emit completion event with null node to signal end of execution.
      // Note: ComfyUI API uses null for node ID to indicate execution complete.
      // The ExecutingEvent type requires NodeId, but null is the expected protocol.
      this.emitInternal({
        type: 'executing',
        data: {
          node: null as unknown as NodeId,
          display_node: null as unknown as NodeId,
          prompt_id: promptId,
        },
      });

      // Mark as completed
      queueItem.status = 'completed';
      queueItem.execution_end_time = Date.now();
      queueItem.outputs_count = Object.keys(outputs).length;

      // Move to history
      const index = this.queue.findIndex((item) => item.id === promptId);
      if (index !== -1) {
        this.queue.splice(index, 1);
        this.history.unshift(queueItem);
      }

      // Emit success event
      this.emitInternal({
        type: 'execution_success',
        data: {
          prompt_id: promptId,
          timestamp: Date.now(),
        },
      });

      // Update status
      this.emitInternal({
        type: 'status',
        data: {
          status: { exec_info: { queue_remaining: this.queue.length } },
        },
      });
    } catch (error) {
      // Mark as failed
      queueItem.status = 'failed';
      queueItem.execution_end_time = Date.now();
      queueItem.execution_error = {
        prompt_id: promptId,
        timestamp: Date.now(),
        node_id: '',
        node_type: '',
        exception_message: error instanceof Error ? error.message : 'Unknown error',
        exception_type: 'Error',
        traceback: [],
        current_inputs: null,
        current_outputs: null,
      };

      // Move to history
      const index = this.queue.findIndex((item) => item.id === promptId);
      if (index !== -1) {
        this.queue.splice(index, 1);
        this.history.unshift(queueItem);
      }

      // Emit error event
      this.emitInternal({
        type: 'execution_error',
        data: {
          prompt_id: promptId,
          timestamp: Date.now(),
          node_id: '',
          node_type: '',
          executed: [],
          exception_message: error instanceof Error ? error.message : 'Unknown error',
          exception_type: 'Error',
          traceback: [],
          current_inputs: null,
          current_outputs: null,
        },
      });
    }
  }

  private generateMockOutput(classType: string): Record<string, unknown> {
    // Generate appropriate mock output based on node type
    if (classType.startsWith('Math/') || classType === 'Input/Number') {
      return { result: Math.random() * 100 };
    }
    if (classType.startsWith('String/') || classType === 'Input/Text') {
      return { result: 'Mock output string' };
    }
    if (classType.startsWith('Logic/')) {
      return { result: Math.random() > 0.5 };
    }
    if (classType === 'Output/Display' || classType === 'Output/Log') {
      return {}; // Output nodes don't produce outputs
    }
    return { result: 'mock_value' };
  }

  // ==========================================================================
  // Test Utilities
  // ==========================================================================

  /**
   * Set custom node definitions (for testing)
   */
  setNodeDefinitions(definitions: NodeDefinitions): void {
    Object.assign(MOCK_NODE_DEFINITIONS, definitions);
  }

  /**
   * Set server feature flags (for testing)
   */
  setServerFeatureFlags(flags: Record<string, unknown>): void {
    this.serverFeatureFlags = flags;
  }

  /**
   * Set execution delay (for testing)
   */
  setExecutionDelay(delay: number): void {
    this.executionDelay = delay;
  }

  /**
   * Clear all queues and history (for testing)
   */
  clear(): void {
    this.queue = [];
    this.history = [];
    this.userData.clear();
    this.settings = {};
    this.serverFeatureFlags = {};
    this.trinityTypeRegistry.clear();
    this.trinityComponentStore.clear();
  }
  // ==========================================================================
  // Trinity Bridge Protocol (T-TL-1.1)
  // ==========================================================================

  async trinityRequest<T = unknown>(method: string, params?: unknown): Promise<T> {
    await this.simulateLatency();
    console.log('[MockAPI] trinityRequest:', method, JSON.stringify(params));
    return { handled: true, method } as unknown as T;
  }

  async trinityRegisterType(request: TypeRegisterRequest): Promise<TypeRegisterResponse> {
    await this.simulateLatency();
    const typeId = generateId();
    const entry: TypeRegistryEntry = {
      type_id: typeId,
      type_name: request.type_name,
      type_kind: request.type_kind,
      fields: request.fields || [],
    };
    this.trinityTypeRegistry.set(typeId, entry);
    this.trinityTypeRegistry.set(request.type_name, entry);
    return { type_id: typeId };
  }

  async trinityListTypes(request?: TypeListRequest): Promise<TypeListResponse> {
    await this.simulateLatency();
    let types = Array.from(this.trinityTypeRegistry.values());
    // Deduplicate by type_id (registry stores both by id and name)
    const seen = new Set<string>();
    types = types.filter((t) => {
      if (seen.has(t.type_id)) return false;
      seen.add(t.type_id);
      return true;
    });
    if (request?.type_kind) {
      types = types.filter((t) => t.type_kind === request.type_kind);
    }
    return { types, total: types.length };
  }

  async trinityGetType(request: TypeGetRequest): Promise<TypeRegistryEntry> {
    await this.simulateLatency();
    const entry = this.trinityTypeRegistry.get(request.type_id);
    if (!entry) {
      throw new Error(`Type not found: ${request.type_id}`);
    }
    return entry;
  }

  async trinityReadField(request: ComponentReadRequest): Promise<unknown> {
    await this.simulateLatency();
    const { field } = request;
    const entityKey = `${field.entity_id}:${field.component_id}`;
    const store = this.trinityComponentStore.get(entityKey);
    if (!store) return null;
    const value = store.get(String(field.offset));
    return value !== undefined ? value : null;
  }
}
