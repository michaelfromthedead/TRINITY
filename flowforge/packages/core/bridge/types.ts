/**
 * FlowForge Bridge Layer - Type Definitions
 *
 * This module contains all TypeScript interfaces that define the contract
 * between the frontend and the execution backend. These types mirror the
 * ComfyUI API format to ensure compatibility during migration.
 */

// ============================================================================
// Basic Types
// ============================================================================

/** Node identifier - can be number or string (for nested/group nodes) */
export type NodeId = number | string;

/** Prompt/execution identifier */
export type PromptId = string;

/** Client session identifier */
export type ClientId = string;

/** Data type for node connections */
export type DataType = string | string[] | number;

/** Slot index for node inputs/outputs */
export type SlotIndex = number;

// ============================================================================
// Input Definition Types
// ============================================================================

/** Base options for all input types */
export interface BaseInputOptions {
  default?: unknown;
  defaultInput?: boolean;
  display_name?: string;
  forceInput?: boolean;
  tooltip?: string;
  socketless?: boolean;
  hidden?: boolean;
  advanced?: boolean;
  widgetType?: string;
  /** Backend-only properties */
  rawLink?: boolean;
  lazy?: boolean;
}

/** Options for INT inputs */
export interface IntInputOptions extends BaseInputOptions {
  min?: number;
  max?: number;
  step?: number;
  default?: number | number[];
  display?: 'slider' | 'number' | 'knob';
  control_after_generate?: boolean;
}

/** Options for FLOAT inputs */
export interface FloatInputOptions extends BaseInputOptions {
  min?: number;
  max?: number;
  step?: number;
  default?: number | number[];
  display?: 'slider' | 'number' | 'knob';
  round?: number | false;
}

/** Options for BOOLEAN inputs */
export interface BooleanInputOptions extends BaseInputOptions {
  label_on?: string;
  label_off?: string;
  default?: boolean;
}

/** Options for STRING inputs */
export interface StringInputOptions extends BaseInputOptions {
  default?: string;
  multiline?: boolean;
  dynamicPrompts?: boolean;
  defaultVal?: string;
  placeholder?: string;
}

/** Remote widget configuration */
export interface RemoteWidgetConfig {
  route: string;
  refresh?: number;
  response_key?: string;
  query_params?: Record<string, string>;
  refresh_button?: boolean;
  control_after_refresh?: 'first' | 'last';
  timeout?: number;
  max_retries?: number;
}

/** Multi-select options */
export interface MultiSelectOption {
  placeholder?: string;
  chip?: boolean;
}

/** Result item type */
export type ResultItemType = 'input' | 'output' | 'temp';

/** Options for COMBO (dropdown) inputs */
export interface ComboInputOptions extends BaseInputOptions {
  control_after_generate?: boolean;
  image_upload?: boolean;
  image_folder?: ResultItemType;
  allow_batch?: boolean;
  video_upload?: boolean;
  audio_upload?: boolean;
  animated_image_upload?: boolean;
  options?: (string | number)[];
  remote?: RemoteWidgetConfig;
  multi_select?: MultiSelectOption;
}

/** Input specification tuple types */
export type IntInputSpec = ['INT', IntInputOptions?];
export type FloatInputSpec = ['FLOAT', FloatInputOptions?];
export type BooleanInputSpec = ['BOOLEAN', BooleanInputOptions?];
export type StringInputSpec = ['STRING', StringInputOptions?];
export type ComboInputSpec = [(string | number)[], ComboInputOptions?];
export type ComboInputSpecV2 = ['COMBO', ComboInputOptions?];
export type CustomInputSpec = [string, BaseInputOptions?];

export type InputSpec =
  | IntInputSpec
  | FloatInputSpec
  | BooleanInputSpec
  | StringInputSpec
  | ComboInputSpec
  | ComboInputSpecV2
  | CustomInputSpec;

/** Node inputs specification */
export interface ComfyInputsSpec {
  required?: Record<string, InputSpec>;
  optional?: Record<string, InputSpec>;
  hidden?: Record<string, unknown>;
}

// ============================================================================
// Output Definition Types
// ============================================================================

/** Output type specification - can be data type or combo options */
export type OutputTypeSpec = string | (string | number)[];

/** Complete output types array */
export type ComfyOutputTypesSpec = OutputTypeSpec[];

// ============================================================================
// Node Definition Types
// ============================================================================

/** Widget dependency for pricing calculations */
export interface WidgetDependency {
  name: string;
  type: string;
}

/** Price badge dependencies */
export interface PriceBadgeDepends {
  widgets?: WidgetDependency[];
  inputs?: string[];
  input_groups?: string[];
}

/** Price badge definition for API nodes */
export interface PriceBadge {
  engine?: 'jsonata';
  depends_on?: PriceBadgeDepends;
  expr: string;
}

/** Complete node definition as returned by /object_info */
export interface ComfyNodeDef {
  input?: ComfyInputsSpec;
  output?: ComfyOutputTypesSpec;
  output_is_list?: boolean[];
  output_name?: string[];
  output_tooltips?: string[];
  output_matchtypes?: (string | undefined)[];
  name: string;
  display_name: string;
  description: string;
  help?: string;
  category: string;
  output_node: boolean;
  python_module: string;
  deprecated?: boolean;
  experimental?: boolean;
  api_node?: boolean;
  input_order?: Record<string, string[]>;
  search_aliases?: string[];
  price_badge?: PriceBadge;
}

/** Dictionary of all node definitions */
export type NodeDefinitions = Record<string, ComfyNodeDef>;

// ============================================================================
// Workflow Types
// ============================================================================

/** 2D vector position/size */
export type Vector2 = [number, number];

/** Node flags */
export interface NodeFlags {
  collapsed?: boolean;
  pinned?: boolean;
  allow_interaction?: boolean;
  horizontal?: boolean;
  skip_repeated_outputs?: boolean;
}

/** Node input slot */
export interface NodeInput {
  name: string;
  type: DataType;
  link?: number | null;
  slot_index?: SlotIndex;
}

/** Node output slot */
export interface NodeOutput {
  name: string;
  type: DataType;
  links?: number[] | null;
  slot_index?: SlotIndex;
}

/** Node properties */
export interface NodeProperties {
  'Node name for S&R'?: string;
  cnr_id?: string;
  aux_id?: string;
  ver?: string;
  [key: string]: unknown;
}

/** Complete workflow node */
export interface WorkflowNode {
  id: NodeId;
  type: string;
  pos: Vector2;
  size: Vector2;
  flags: NodeFlags;
  order: number;
  mode: number;
  inputs?: NodeInput[];
  outputs?: NodeOutput[];
  properties: NodeProperties;
  widgets_values?: unknown[] | Record<string, unknown>;
  color?: string;
  bgcolor?: string;
}

/** Link tuple format (schema version 0.4) */
export type WorkflowLink = [
  number,     // Link id
  NodeId,     // Source node id
  SlotIndex,  // Source output slot
  NodeId,     // Target node id
  SlotIndex,  // Target input slot
  DataType    // Data type
];

/** Link object format (schema version 1) */
export interface WorkflowLinkObject {
  id: number;
  origin_id: NodeId;
  origin_slot: SlotIndex;
  target_id: NodeId;
  target_slot: SlotIndex;
  type: DataType;
  parentId?: number;
}

/** Workflow group */
export interface WorkflowGroup {
  id?: number;
  title: string;
  bounding: [number, number, number, number];
  color?: string;
  font_size?: number;
  locked?: boolean;
}

/** Display state (zoom/pan) */
export interface DisplayState {
  scale: number;
  offset: Vector2;
}

/** Workflow configuration */
export interface WorkflowConfig {
  links_ontop?: boolean;
  align_to_grid?: boolean;
}

/** Reroute node */
export interface Reroute {
  id: number;
  parentId?: number;
  pos: Vector2;
  linkIds?: number[] | null;
  floating?: {
    slotType: 'input' | 'output';
  };
}

/** Link extension for reroute tracking */
export interface LinkExtension {
  id: number;
  parentId: number;
}

/** Extra workflow metadata */
export interface WorkflowExtra {
  ds?: DisplayState;
  frontendVersion?: string;
  linkExtensions?: LinkExtension[];
  reroutes?: Reroute[];
  workflowRendererVersion?: 'LG' | 'Vue';
}

/** Graph state tracking */
export interface GraphState {
  lastGroupId: number;
  lastNodeId: number;
  lastLinkId: number;
  lastRerouteId: number;
}

/** Model file reference */
export interface ModelFile {
  name: string;
  url: string;
  hash?: string;
  hash_type?: string;
  directory: string;
}

/** Complete workflow JSON (version 0.4) */
export interface ComfyWorkflowJSON04 {
  id?: string;
  revision?: number;
  last_node_id: NodeId;
  last_link_id: number;
  nodes: WorkflowNode[];
  links: WorkflowLink[];
  floatingLinks?: WorkflowLinkObject[];
  groups?: WorkflowGroup[];
  config?: WorkflowConfig | null;
  extra?: WorkflowExtra | null;
  version: number;
  models?: ModelFile[];
}

/** Complete workflow JSON (version 1) */
export interface ComfyWorkflowJSON1 {
  id?: string;
  revision?: number;
  version: 1;
  config?: WorkflowConfig | null;
  state: GraphState;
  groups?: WorkflowGroup[];
  nodes: WorkflowNode[];
  links?: WorkflowLinkObject[];
  floatingLinks?: WorkflowLinkObject[];
  reroutes?: Reroute[];
  extra?: WorkflowExtra | null;
  models?: ModelFile[];
}

/** Union of all workflow versions */
export type ComfyWorkflowJSON = ComfyWorkflowJSON04 | ComfyWorkflowJSON1;

// ============================================================================
// API Workflow Format (for execution)
// ============================================================================

/** Node input value - widget value or link reference */
export type NodeInputValue = unknown | [NodeId, SlotIndex];

/** Node data for API workflow */
export interface ApiNodeData {
  inputs: Record<string, NodeInputValue>;
  class_type: string;
  _meta: {
    title: string;
  };
}

/** API format workflow for execution */
export type ComfyApiWorkflow = Record<string, ApiNodeData>;

// ============================================================================
// Execution Types
// ============================================================================

/** Result item (image, video, etc.) */
export interface ResultItem {
  filename?: string;
  subfolder?: string;
  type?: ResultItemType;
}

/** Node execution output */
export interface NodeExecutionOutput {
  audio?: ResultItem[];
  images?: ResultItem[];
  video?: ResultItem[];
  animated?: boolean[];
  text?: string | string[];
  [key: string]: unknown;
}

/** Task output - mapping of node IDs to their outputs */
export type TaskOutput = Record<NodeId, NodeExecutionOutput>;

/** Error details */
export interface ExecutionErrorDetails {
  type: string;
  message: string;
  details: string;
  extra_info?: {
    input_name?: string;
    [key: string]: unknown;
  };
}

/** Node-specific error */
export interface NodeError {
  errors: ExecutionErrorDetails[];
  class_type: string;
  dependent_outputs: unknown[];
}

/** Prompt/execution response */
export interface PromptResponse {
  node_errors?: Record<NodeId, NodeError>;
  prompt_id?: string;
  exec_info?: {
    queue_remaining?: number;
  };
  error?: string | ExecutionErrorDetails;
}

/** Complete execution result */
export interface ExecutionResult {
  success: boolean;
  prompt_id: string;
  outputs?: TaskOutput;
  error?: ExecutionErrorDetails | string;
  node_errors?: Record<NodeId, NodeError>;
}

// ============================================================================
// Execution Event Types (WebSocket messages)
// ============================================================================

/** Queue status info */
export interface QueueStatusInfo {
  exec_info: {
    queue_remaining: number;
  };
}

/** Status event */
export interface StatusEvent {
  type: 'status';
  data: {
    status?: QueueStatusInfo | null;
    sid?: string | null;
  };
}

/** Progress event */
export interface ProgressEvent {
  type: 'progress';
  data: {
    value: number;
    max: number;
    prompt_id: PromptId;
    node: NodeId;
  };
}

/** Node progress state */
export interface NodeProgressState {
  value: number;
  max: number;
  state: 'pending' | 'running' | 'finished' | 'error';
  node_id: NodeId;
  prompt_id: PromptId;
  display_node_id?: NodeId;
  parent_node_id?: NodeId;
  real_node_id?: NodeId;
}

/** Progress state event */
export interface ProgressStateEvent {
  type: 'progress_state';
  data: {
    prompt_id: PromptId;
    nodes: Record<NodeId, NodeProgressState>;
  };
}

/** Executing event - indicates which node is currently running */
export interface ExecutingEvent {
  type: 'executing';
  data: {
    node: NodeId;
    display_node: NodeId;
    prompt_id: PromptId;
  };
}

/** Executed event - node completed with output */
export interface ExecutedEvent {
  type: 'executed';
  data: {
    node: NodeId;
    display_node: NodeId;
    prompt_id: PromptId;
    output: NodeExecutionOutput;
    merge?: boolean;
  };
}

/** Execution start event */
export interface ExecutionStartEvent {
  type: 'execution_start';
  data: {
    prompt_id: PromptId;
    timestamp: number;
  };
}

/** Execution success event */
export interface ExecutionSuccessEvent {
  type: 'execution_success';
  data: {
    prompt_id: PromptId;
    timestamp: number;
  };
}

/** Execution cached event */
export interface ExecutionCachedEvent {
  type: 'execution_cached';
  data: {
    prompt_id: PromptId;
    timestamp: number;
    nodes: NodeId[];
  };
}

/** Execution interrupted event */
export interface ExecutionInterruptedEvent {
  type: 'execution_interrupted';
  data: {
    prompt_id: PromptId;
    timestamp: number;
    node_id: NodeId;
    node_type: string;
    executed: NodeId[];
  };
}

/** Execution error event */
export interface ExecutionErrorEvent {
  type: 'execution_error';
  data: {
    prompt_id: PromptId;
    timestamp: number;
    node_id: NodeId;
    node_type: string;
    executed: NodeId[];
    exception_message: string;
    exception_type: string;
    traceback: string[];
    current_inputs: unknown;
    current_outputs: unknown;
  };
}

/** Progress text event */
export interface ProgressTextEvent {
  type: 'progress_text';
  data: {
    nodeId: NodeId;
    text: string;
  };
}

/** Binary preview event */
export interface BinaryPreviewEvent {
  type: 'b_preview';
  data: Blob;
}

/** Binary preview with metadata event */
export interface BinaryPreviewWithMetadataEvent {
  type: 'b_preview_with_metadata';
  data: {
    blob: Blob;
    nodeId: string;
    parentNodeId: string;
    displayNodeId: string;
    realNodeId: string;
    promptId: string;
  };
}

/** Log entry */
export interface LogEntry {
  t: string;  // timestamp
  m: string;  // message
}

/** Terminal size */
export interface TerminalSize {
  cols: number;
  row: number;
}

/** Logs event */
export interface LogsEvent {
  type: 'logs';
  data: {
    size?: TerminalSize;
    entries: LogEntry[];
  };
}

/** Notification event */
export interface NotificationEvent {
  type: 'notification';
  data: {
    value: string;
    id?: string;
  };
}

/** Feature flags event */
export interface FeatureFlagsEvent {
  type: 'feature_flags';
  data: Record<string, unknown>;
}

/** Asset download event */
export interface AssetDownloadEvent {
  type: 'asset_download';
  data: {
    task_id: string;
    asset_name: string;
    bytes_total: number;
    bytes_downloaded: number;
    progress: number;
    status: 'created' | 'running' | 'completed' | 'failed';
    asset_id?: string;
    error?: string;
  };
}

/** Frontend-generated events */
export interface GraphChangedEvent {
  type: 'graphChanged';
  data: ComfyWorkflowJSON;
}

export interface PromptQueuedEvent {
  type: 'promptQueued';
  data: {
    number: number;
    batchCount: number;
  };
}

export interface GraphClearedEvent {
  type: 'graphCleared';
  data: never;
}

export interface ReconnectingEvent {
  type: 'reconnecting';
  data: never;
}

export interface ReconnectedEvent {
  type: 'reconnected';
  data: never;
}

/** Union of all execution events */
export type ExecutionEvent =
  | StatusEvent
  | ProgressEvent
  | ProgressStateEvent
  | ExecutingEvent
  | ExecutedEvent
  | ExecutionStartEvent
  | ExecutionSuccessEvent
  | ExecutionCachedEvent
  | ExecutionInterruptedEvent
  | ExecutionErrorEvent
  | ProgressTextEvent
  | BinaryPreviewEvent
  | BinaryPreviewWithMetadataEvent
  | LogsEvent
  | NotificationEvent
  | FeatureFlagsEvent
  | AssetDownloadEvent
  | GraphChangedEvent
  | PromptQueuedEvent
  | GraphClearedEvent
  | ReconnectingEvent
  | ReconnectedEvent;

// ============================================================================
// Queue Types
// ============================================================================

/** Job status */
export type JobStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'cancelled';

/** Preview output for queue items */
export interface PreviewOutput {
  filename: string;
  subfolder: string;
  type: ResultItemType;
  nodeId: string;
  mediaType: string;
}

/** Execution error from Jobs API */
export interface JobExecutionError {
  prompt_id?: string;
  timestamp?: number;
  node_id: string;
  node_type: string;
  executed?: string[];
  exception_message: string;
  exception_type: string;
  traceback: string[];
  current_inputs: unknown;
  current_outputs: unknown;
}

/** Queue item (job) */
export interface QueueItem {
  id: string;
  status: JobStatus;
  create_time: number;
  execution_start_time?: number | null;
  execution_end_time?: number | null;
  preview_output?: PreviewOutput | null;
  outputs_count?: number | null;
  execution_error?: JobExecutionError | null;
  workflow_id?: string | null;
  priority: number;
}

/** Queue item with full details */
export interface QueueItemDetail extends QueueItem {
  workflow?: unknown;
  outputs?: TaskOutput;
  update_time?: number;
  execution_status?: unknown;
  execution_meta?: unknown;
}

/** Queue status - running and pending items */
export interface QueueStatus {
  Running: QueueItem[];
  Pending: QueueItem[];
}

/** Pagination info */
export interface PaginationInfo {
  offset: number;
  limit: number;
  total: number;
  has_more: boolean;
}

/** Jobs list response */
export interface JobsListResponse {
  jobs: QueueItem[];
  pagination: PaginationInfo;
}

// ============================================================================
// System Types
// ============================================================================

/** Device stats */
export interface DeviceStats {
  name: string;
  type: string;
  index: number;
  vram_total: number;
  vram_free: number;
  torch_vram_total: number;
  torch_vram_free: number;
}

/** System stats */
export interface SystemStats {
  system: {
    os: string;
    python_version: string;
    embedded_python: boolean;
    comfyui_version: string;
    pytorch_version: string;
    required_frontend_version?: string;
    argv: string[];
    ram_total: number;
    ram_free: number;
    cloud_version?: string;
    comfyui_frontend_version?: string;
    workflow_templates_version?: string;
  };
  devices: DeviceStats[];
}

/** User configuration */
export interface UserConfig {
  storage: 'server';
  migrated?: boolean;
  users?: Record<string, string>;
}

/** User data file info */
export interface UserDataFullInfo {
  path: string;
  size: number;
  modified: number;
}

// ============================================================================
// Asset Types
// ============================================================================

/** Model folder info */
export interface ModelFolderInfo {
  name: string;
  [key: string]: unknown;
}

/** Model file info */
export interface ModelFileInfo {
  name: string;
  path?: string;
  size?: number;
  [key: string]: unknown;
}

/** Asset upload result */
export interface AssetUploadResult {
  id: string;
  localPath: string;
  name: string;
  subfolder?: string;
}

/** Asset info */
export interface AssetInfo {
  id: string;
  path: string;
  type: ResultItemType;
  subfolder?: string;
}

// ============================================================================
// Prompt Options
// ============================================================================

/** Preview method */
export type PreviewMethod = 'default' | 'none' | 'auto' | 'latent2rgb' | 'taesd';

/** Node execution ID for partial execution */
export type NodeExecutionId = string;

/** Options for queuePrompt */
export interface QueuePromptOptions {
  partialExecutionTargets?: NodeExecutionId[];
  previewMethod?: PreviewMethod;
}

// ============================================================================
// Extension Types
// ============================================================================

/** Extensions list response */
export type ExtensionsResponse = string[];

/** Embeddings list response */
export type EmbeddingsResponse = string[];

/** Workflow templates */
export interface WorkflowTemplateInfo {
  name: string;
  path: string;
  [key: string]: unknown;
}

/** Workflow templates by custom node */
export type WorkflowTemplates = Record<string, string[]>;

// ============================================================================
// IPC Message Types (for Tauri/Bun communication)
// ============================================================================

/** IPC message for Tauri-Bun communication */
export interface IPCMessage {
  id: string;
  type: 'request' | 'response' | 'event';
  method?: string;
  params?: unknown;
  result?: unknown;
  error?: {
    code: number;
    message: string;
  };
  event?: string;
  payload?: unknown;
}

// ============================================================================
// Type Guards
// ============================================================================

/** Check if input spec is a combo (v1 format - array of options) */
export function isComboInputSpecV1(inputSpec: InputSpec): inputSpec is ComboInputSpec {
  return Array.isArray(inputSpec[0]);
}

/** Check if input spec is a combo (v2 format - 'COMBO' literal) */
export function isComboInputSpecV2(inputSpec: InputSpec): inputSpec is ComboInputSpecV2 {
  return inputSpec[0] === 'COMBO';
}

/** Check if input spec is any combo type */
export function isComboInputSpec(inputSpec: InputSpec): inputSpec is ComboInputSpec | ComboInputSpecV2 {
  return isComboInputSpecV1(inputSpec) || isComboInputSpecV2(inputSpec);
}

/** Get the type string from an input spec */
export function getInputSpecType(inputSpec: InputSpec): string {
  return isComboInputSpec(inputSpec) ? 'COMBO' : (inputSpec[0] as string);
}

/** Get combo options from a combo input spec */
export function getComboSpecComboOptions(inputSpec: ComboInputSpec | ComboInputSpecV2): (string | number)[] {
  return (isComboInputSpecV2(inputSpec) ? inputSpec[1]?.options : inputSpec[0]) ?? [];
}


// ============================================================================
// Trinity Bridge Protocol Types (T-TL-1.1)
// ============================================================================

// ---------------------------------------------------------------------------
// Channel Constants
// ---------------------------------------------------------------------------

export const TYPE_CHANNEL_PREFIX = 'type';
export const DATA_CHANNEL_PREFIX = 'data';
export const COMMAND_CHANNEL_PREFIX = 'command';
export const SYSTEM_CHANNEL_PREFIX = 'system';

export const TYPE_CHANNEL_ENDPOINTS = 5;
export const DATA_CHANNEL_ENDPOINTS = 5;
export const COMMAND_CHANNEL_ENDPOINTS = 6;
export const SYSTEM_CHANNEL_ENDPOINTS = 6;
export const TOTAL_ENDPOINTS = 22;

export const METHOD_TABLE: readonly string[] = [
  'type.register', 'type.list', 'type.get', 'type.remove', 'type.count',
  'data.read', 'data.write', 'data.delete', 'data.batch_read', 'data.batch_write',
  'command.create', 'command.spawn', 'command.despawn', 'command.query', 'command.reset', 'command.stats',
  'system.connect', 'system.status', 'system.inspect', 'system.inspector_get', 'system.events_recent', 'system.checksum',
];

export function channelForMethod(method: string): string | null {
  if (method.startsWith('type.')) return 'type';
  if (method.startsWith('data.')) return 'data';
  if (method.startsWith('command.')) return 'command';
  if (method.startsWith('system.')) return 'system';
  return null;
}

// ---------------------------------------------------------------------------
// Type Channel
// ---------------------------------------------------------------------------

export type TypeKind = string;

export interface FieldDescriptor {
  name: string;
  type_kind: string;
  offset?: number;
}

export interface TypeRegisterRequest {
  type_name: string;
  type_kind: TypeKind;
  fields: FieldDescriptor[];
}

export interface TypeRegisterResponse {
  type_id: string;
}

export interface TypeRegistryEntry {
  type_id: string;
  type_name: string;
  type_kind: TypeKind;
  fields: FieldDescriptor[];
}

export interface TypeListRequest {
  type_kind?: TypeKind;
}

export interface TypeListResponse {
  types: TypeRegistryEntry[];
  total: number;
}

export interface TypeGetRequest {
  type_id: string;
}

export interface TypeRemoveRequest {
  type_id: string;
}

export interface TypeCountResponse {
  count: number;
}

// ---------------------------------------------------------------------------
// Data Channel
// ---------------------------------------------------------------------------

export interface FieldKey {
  entity_id: string;
  component_id: string;
  offset: number;
}

export interface ComponentReadRequest {
  field: FieldKey;
}

export interface ComponentWriteRequest {
  field: FieldKey;
  value: unknown;
}

export interface ComponentDeleteRequest {
  field: FieldKey;
}

export interface FieldReadResult {
  value: unknown;
  exists: boolean;
}

export interface FieldInit {
  offset: number;
  value: unknown;
}

export interface ComponentBlock {
  component_id: string;
  fields: FieldInit[];
}

export interface ComponentBatchReadRequest {
  fields: FieldKey[];
}

export interface ComponentBatchReadResponse {
  results: FieldReadResult[];
}

export interface ComponentBatchWriteRequest {
  fields: { field: FieldKey; value: unknown }[];
}

export interface ComponentBatchWriteResponse {
  written: number;
}

// ---------------------------------------------------------------------------
// Command Channel
// ---------------------------------------------------------------------------

export interface WorldCreateRequest {
  label?: string;
}

export interface WorldCreateResponse {
  world_id: string;
}

export interface WorldSpawnRequest {
  world_id: string;
  archetype?: string;
  components?: Record<string, Record<string, unknown>>;
}

export interface WorldSpawnResponse {
  entity_id: string;
}

export interface WorldDespawnRequest {
  world_id: string;
  entity_id: string;
}

export interface WorldQueryRequest {
  world_id: string;
  component_filter?: string[];
}

export interface WorldQueryResponse {
  entities: string[];
  count: number;
}

export interface WorldResetRequest {
  world_id: string;
}

export interface WorldStatsResponse {
  entity_count: number;
  component_count: number;
}

// ---------------------------------------------------------------------------
// System Channel
// ---------------------------------------------------------------------------

export interface TrinityConnectRequest {
  session_id?: string;
}

export interface TrinityConnectResponse {
  session_id: string;
  version: string;
  transport: string;
}

export interface TrinityStatusResponse {
  connected: boolean;
  session_id?: string;
  uptime_ms: number;
}

export interface TrinityInspectRequest {
  depth?: number;
}

export interface SourceLocation {
  file: string;
  line: number;
  column: number;
}

export interface DecoratorEntry {
  name: string;
  path?: string;
  source?: SourceLocation;
}

export interface HierarchyEntry {
  name: string;
  type_name: string;
  children?: HierarchyEntry[];
}

export interface TrinityInspectResponse {
  version: string;
  hierarchy?: HierarchyEntry[];
  decorators?: DecoratorEntry[];
}

export interface InspectorGetRequest {
  path: string;
  depth?: number;
}

export interface EventEntry {
  id: string;
  type: string;
  timestamp: number;
  data?: Record<string, unknown>;
}

export interface EventsRecentResponse {
  events: EventEntry[];
  count: number;
}

export interface ChecksumResponse {
  checksum: string;
  algorithm: string;
}

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 Envelope
// ---------------------------------------------------------------------------

export interface JsonRpcRequest {
  jsonrpc: '2.0';
  id: string | number;
  method: string;
  params?: unknown;
}

export interface JsonRpcResponse {
  jsonrpc: '2.0';
  id: string | number | null;
  result?: unknown;
  error?: {
    code: number;
    message: string;
    data?: unknown;
  };
}

export interface JsonRpcNotification {
  jsonrpc: '2.0';
  method: string;
  params?: unknown;
}
